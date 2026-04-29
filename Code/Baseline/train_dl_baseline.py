import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader

# 挂载项目路径
PROJECT_ROOT = "/Users/martingao/VScode/EMDvsGCN"
sys.path.append(os.path.join(PROJECT_ROOT, "Code"))

from Baseline.rnn_family import RNNFamilyModel
from Utils.metrics import calculate_metrics, print_metrics, save_metrics

# ================= 配置区域 =================
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "Data", "Processed")
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "Results", "Checkpoints")
PRED_SAVE_DIR = os.path.join(PROJECT_ROOT, "Results", "Predictions")
METRICS_SAVE_DIR = os.path.join(PROJECT_ROOT, "Results", "Metrics")

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(PRED_SAVE_DIR, exist_ok=True)
os.makedirs(METRICS_SAVE_DIR, exist_ok=True)

# 严格遵循文献设定的参数
SEQ_LEN = 512       # Look-back length (L)
PRED_LEN = 96       # Prediction horizon (H)
STRIDE = 1          # Sliding window stride

BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10         # 最终训练轮数
HIDDEN_SIZE = 64
NUM_LAYERS = 2

# 你可以通过修改这里，分别跑 LSTM、GRU、RNN 的实验
MODEL_TYPE = 'RNN' 
TEST_FILE = "Cleaned_IDX_000001.SH_上证指数_1min.csv"

# ================= 数据构建模块 (严防 Data Leakage) =================
def create_sliding_windows(data, seq_len, pred_len, stride=1):
    xs, ys = [], []
    # 严格按照 stride = 1 滑动，生成 [L] -> [H] 的样本对
    for i in range(0, len(data) - seq_len - pred_len + 1, stride):
        xs.append(data[i : i + seq_len])
        ys.append(data[i + seq_len : i + seq_len + pred_len])
    return np.array(xs), np.array(ys)

def prepare_dataloaders(file_path):
    df = pd.read_csv(file_path)
    closes = df['close'].values.reshape(-1, 1)
    trade_times = df['trade_time'].values

    # 1. 严格 80/20 划分
    train_size = int(len(closes) * 0.8)
    train_data = closes[:train_size]
    test_data = closes[train_size:]
    test_times = trade_times[train_size:] # 保留测试集对应的时间戳

    # 2. 归一化：仅在训练集上 fit，完美避免 Look-ahead bias
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)

    # 3. 滑动窗口切片 (Stride = 1)
    X_train, y_train = create_sliding_windows(train_scaled, SEQ_LEN, PRED_LEN, STRIDE)
    X_test, y_test = create_sliding_windows(test_scaled, SEQ_LEN, PRED_LEN, STRIDE)

    # 4. 转为 PyTorch 张量
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader, scaler, test_times

# ================= 核心训练与评估 =================
def train_and_evaluate():
    # 动态命名，区分标的和模型
    stock_id = TEST_FILE.replace("Cleaned_", "").replace("_1min.csv", "")
    model_id = f"{MODEL_TYPE}_{stock_id}"
    
    file_path = os.path.join(PROCESSED_DATA_DIR, TEST_FILE)
    if not os.path.exists(file_path):
        print(f"❌ 找不到文件: {file_path}")
        return

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*50}")
    print(f"启动文献标准训练: {model_id} | 设备: {device}")
    print(f"{'='*50}")

    train_loader, test_loader, scaler, test_times = prepare_dataloaders(file_path)
    print(f"数据构建完毕 -> 训练集窗口: {len(train_loader.dataset)} | 测试集窗口: {len(test_loader.dataset)}")
    
    model = RNNFamilyModel(model_type=MODEL_TYPE, input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, pred_len=PRED_LEN).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    checkpoint_path = os.path.join(MODEL_SAVE_DIR, f"best_{model_id}.pth")
    best_loss = float('inf')

    # === 训练阶段 ===
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        start_time = time.time()
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            batch_y = batch_y.squeeze(-1)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
            
        train_loss /= len(train_loader.dataset)
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train MSE Loss: {train_loss:.6f} | 耗时: {epoch_time:.2f}s")
        
        # 保存最佳模型
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), checkpoint_path)

    # === 评估与输出阶段 ===
    print("\n=== 加载最佳权重，在测试集上进行验证 ===")
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    
    preds, trues = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            preds.append(outputs.cpu().numpy())
            trues.append(batch_y.numpy())
            
    # 拼接所有的预测窗口 (N_samples, 96, 1)
    preds_arr = np.concatenate(preds, axis=0).squeeze()
    trues_arr = np.concatenate(trues, axis=0).squeeze()
    
    # 【1】为了学术严谨，计算包含所有 96 步预测的全局指标
    preds_all_real = scaler.inverse_transform(preds_arr.reshape(-1, 1)).flatten()
    trues_all_real = scaler.inverse_transform(trues_arr.reshape(-1, 1)).flatten()
    
    metrics_result = calculate_metrics(trues_all_real, preds_all_real)
    print_metrics(metrics_result, model_name=model_id)
    save_metrics(metrics_result, model_name=model_id, save_dir=METRICS_SAVE_DIR)
    
    # 【2】为了画图美观，只提取每个窗口的第 1 步预测点 (1-step-ahead)
    preds_1d = scaler.inverse_transform(preds_arr[:, 0].reshape(-1, 1)).flatten()
    trues_1d = scaler.inverse_transform(trues_arr[:, 0].reshape(-1, 1)).flatten()
    
    # 精准对齐时间轴：测试集的第一个预测点对应索引为 SEQ_LEN 的时间
    times_1d = test_times[SEQ_LEN : SEQ_LEN + len(preds_1d)]

    df_results = pd.DataFrame({
        'trade_time': times_1d,
        'True_Price': trues_1d,
        'Predicted_Price': preds_1d
    })
    
    save_csv_path = os.path.join(PRED_SAVE_DIR, f"{model_id}_predictions.csv")
    df_results.to_csv(save_csv_path, index=False, encoding='utf-8-sig')
    print(f"✅ 完美对齐！单步预测曲线 (共 {len(preds_1d)} 行) 已保存至: {save_csv_path}")

if __name__ == "__main__":
    train_and_evaluate()