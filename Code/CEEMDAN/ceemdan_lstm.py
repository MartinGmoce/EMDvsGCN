import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# ================= 动态挂载路径与导入 =================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR)) 
sys.path.append(os.path.join(PROJECT_ROOT, "Code", "Utils"))

try:
    from metrics import calculate_metrics, print_metrics, save_metrics
except ImportError:
    print("⚠️ 未找到 Utils/metrics.py")

# ================= 配置区域 =================
CEEMDAN_DATA_DIR = os.path.join(PROJECT_ROOT, "Data", "CEEMDAN_Decomposed")
PRED_SAVE_DIR = os.path.join(PROJECT_ROOT, "Results", "Predictions")
METRICS_SAVE_DIR = os.path.join(PROJECT_ROOT, "Results", "Metrics")
os.makedirs(PRED_SAVE_DIR, exist_ok=True)
os.makedirs(METRICS_SAVE_DIR, exist_ok=True)

# 【核心修改】：批量处理的文件列表
TEST_FILES = [
    "CEEMDAN_STK_000001.SZ_平安银行_1min.csv",
    "CEEMDAN_STK_600519.SH_贵州茅台_1min.csv",
    "CEEMDAN_IDX_399001.SZ_深证成指_1min.csv",
    "CEEMDAN_IDX_000001.SH_上证指数_1min.csv"
]

# ================= 超参数 (专为 RTX 4090 优化) =================
SEQ_LEN = 60
BATCH_SIZE = 1024      # [修改] 从128提升至1024：LSTM很轻，4090显存极大，大Batch能极速缩短训练时间
HIDDEN_SIZE = 64
NUM_LAYERS = 2
EPOCHS = 15
LR = 0.001
TRAIN_RATIO = 0.8
NUM_WORKERS = 8        # [新增] 动用 8 个 CPU 核心来搬运数据

# ================= 标准 LSTM 模型 =================
class PureLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super(PureLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) # 取最后一个时间步进行预测
        return out

def build_windows(data, seq_len):
    X, Y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i : i + seq_len])
        Y.append(data[i + seq_len])
    return np.array(X), np.array(Y)

# ================= 单个数据集的执行逻辑 =================
def run_ceemdan_lstm_for_single_stock(test_file):
    stock_id = test_file.replace("CEEMDAN_", "").replace("_1min.csv", "")
    model_id = f"CEEMDANLSTM_{stock_id}"

    file_path = os.path.join(CEEMDAN_DATA_DIR, test_file)
    if not os.path.exists(file_path):
        print(f"❌ 错误: 找不到分解后的文件 {file_path}，跳过此文件。")
        return

    df = pd.read_csv(file_path)
    components = [col for col in df.columns if 'IMF' in col or 'Residue' in col]
    
    print(f"\n{'='*60}")
    print(f"🚀 [任务启动] CEEMDAN + LSTM 分解预测对比基线: {model_id}")
    print(f"共检测到 {len(components)} 个待训练分量: {components}")
    print(f"{'='*60}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"💡 硬件检测: 成功调用 NVIDIA GPU (Device: {torch.cuda.get_device_name(0)})")
    
    final_sum_preds = None
    final_sum_trues = None

    for comp in components:
        print(f"\n>>> 正在处理分量: {comp} >>>")
        data = df[comp].values.reshape(-1, 1)
        
        train_size = int(len(data) * TRAIN_RATIO)
        train_data, test_data = data[:train_size], data[train_size:]
        
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_data)
        test_scaled = scaler.transform(test_data)
        
        X_train, Y_train = build_windows(train_scaled, SEQ_LEN)
        X_test, Y_test = build_windows(test_scaled, SEQ_LEN)
        
        # [修改] 加入 num_workers 和 pin_memory，极大加速 CPU 向 GPU 喂数据的速度
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(Y_train)), 
            batch_size=BATCH_SIZE, 
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True
        )
        test_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(Y_test)), 
            batch_size=BATCH_SIZE, 
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True
        )
        
        model = PureLSTM(input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.MSELoss()
        
        model.train()
        for epoch in range(EPOCHS):
            for bx, by in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(bx.to(device, non_blocking=True)), by.to(device, non_blocking=True))
                loss.backward()
                optimizer.step()
                
        # 推理
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for bx, by in test_loader:
                preds.append(model(bx.to(device, non_blocking=True)).cpu().numpy())
                trues.append(by.numpy())
                
        preds = scaler.inverse_transform(np.concatenate(preds))
        trues = scaler.inverse_transform(np.concatenate(trues))
        
        if final_sum_preds is None:
            final_sum_preds = np.zeros_like(preds)
            final_sum_trues = np.zeros_like(trues)
            
        final_sum_preds += preds
        final_sum_trues += trues

        # [核心修改：防爆显存机制移位] 
        # 移至内层循环，每个分量跑完立即清理！
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n✅ 所有分量重构完毕！计算整体误差...")
    final_preds = final_sum_preds.flatten()
    final_trues = final_sum_trues.flatten()
    
    metrics = calculate_metrics(final_trues, final_preds)
    print_metrics(metrics, model_name=model_id)
    save_metrics(metrics, model_name=model_id, save_dir=METRICS_SAVE_DIR)
    
    # 提取正确的时间戳
    num_samples = len(final_preds)
    if 'trade_time' in df.columns:
        times_1d = df['trade_time'].values[-num_samples:]
    elif 'date' in df.columns:
        times_1d = df['date'].values[-num_samples:]
    else:
        times_1d = np.arange(num_samples)

    df_results = pd.DataFrame({
        'trade_time': times_1d,
        'True_Price': final_trues, 
        'Predicted_Price': final_preds
    })
    save_csv_path = os.path.join(PRED_SAVE_DIR, f"{model_id}_predictions.csv")
    df_results.to_csv(save_csv_path, index=False, encoding='utf-8-sig')
    print(f"✅ [{model_id}] 预测曲线数据 (共 {num_samples} 行) 已保存至: {save_csv_path}")

    # 整个股票处理完再清一次
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ================= 批处理调度入口 =================
if __name__ == "__main__":
    print("🌟 启动 CEEMDAN + LSTM 批量处理流水线...")
    for idx, target_file in enumerate(TEST_FILES):
        print(f"\n[{idx+1}/{len(TEST_FILES)}] 正在准备执行任务: {target_file}")
        try:
            run_ceemdan_lstm_for_single_stock(target_file)
        except Exception as e:
            print(f"❌ 严重错误：在处理 {target_file} 时程序崩溃！")
            print(f"错误详情: {e}")
            print("➡️ 继续执行下一个任务...")
            
    print("\n🎉 全部 4 个数据集的 CEEMDAN_LSTM 任务已圆满结束！")