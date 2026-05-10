import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from st_trader_model import ST_Trader, compute_chebyshev_polynomials
from gcn_informer_model import GCN_Informer

# ================= 配置与路径 =================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR)) 
GCN_DATA_DIR = os.path.join(PROJECT_ROOT, "Data", "GCN")
GRAPH_DIR = os.path.join(PROJECT_ROOT, "Results", "VAE_Features")
PRED_DIR = os.path.join(PROJECT_ROOT, "Results", "Predictions")
METRICS_DIR = os.path.join(PROJECT_ROOT, "Results", "Metrics")

os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

sys.path.append(os.path.join(PROJECT_ROOT, "Code", "Utils"))
try:
    from metrics import calculate_metrics, print_metrics, save_metrics
except ImportError:
    print("⚠️ 未找到 Utils/metrics.py，请确保路径正确！")

# ================= 循环图网络配置 =================
GRAPH_FILES = [
    "adj_vae.npy", 
    "adj_identity.npy", 
    "adj_pearson.npy", 
    "adj_random.npy"
]

# ================= 超参数设置 (服务器满血版) =================
LOOK_BACK = 60       
BATCH_SIZE = 64      # 降低 Batch Size，增强模型梯度的随机性和泛化能力
HIDDEN_DIM = 128     # 【核心升级】从 64 翻倍到 128，充分利用 GPU 算力提取复杂空间特征
K_ORDER = 3          
EPOCHS = 100         # 【核心升级】增加到 100 轮，让深层网络在 GPU 上充分收敛
LR = 0.001
TRAIN_RATIO = 0.8    

TARGET_STOCKS = {'000001.SZ': '平安银行', '600519.SH': '贵州茅台'}
FEATURES = ['open', 'close', 'high', 'low', 'cje'] 
TARGET_FEATURE_IDX = 1  

def load_and_align_data(tickers):
    print("⏳ 正在加载并对齐 32 只股票的高频分钟数据...")
    df_list = []
    for ticker in tickers:
        files = [f for f in os.listdir(GCN_DATA_DIR) if ticker in f]
        if not files:
            raise FileNotFoundError(f"找不到 {ticker} 的数据文件")
        
        filepath = os.path.join(GCN_DATA_DIR, files[0])
        df = pd.read_csv(filepath)
        
        if 'tdate' in df.columns and 'ttime' in df.columns:
            df['datetime'] = pd.to_datetime(df['tdate'] + ' ' + df['ttime'])
        elif 'tdate' in df.columns:
            df['datetime'] = pd.to_datetime(df['tdate'])
        else:
            raise ValueError("CSV中没有找到时间列 (tdate)")
            
        df.set_index('datetime', inplace=True)
        
        # 提取需要的特征列
        df_feats = df[FEATURES].copy()
        
        # ==========================================================
        # 【核心修复】：对成交额等长尾特征进行 Log1p 平滑
        # 强行将上百倍的极端脉冲压平，防止 GCN 神经元被“击穿归零”
        # ==========================================================
        if 'cje' in df_feats.columns:
            # 使用 log1p (即 log(1+x)) 防止出现 log(0) 报错
            df_feats['cje'] = np.log1p(df_feats['cje'])
            
        df_feats = df_feats.add_prefix(f"{ticker}_")
        df_list.append(df_feats)
        
    combined_df = pd.concat(df_list, axis=1).dropna()
    print(f"✅ 数据对齐完成！共获取到 {len(combined_df)} 个有效时间步。")
    return combined_df

def build_sliding_windows(data_array, look_back):
    X, Y = [], []
    for i in range(len(data_array) - look_back):
        X.append(data_array[i : i + look_back, :])
        y_indices = [idx * len(FEATURES) + TARGET_FEATURE_IDX for idx in range(data_array.shape[1] // len(FEATURES))]
        Y.append(data_array[i + look_back, y_indices])
    return np.array(X), np.array(Y)

def train_and_evaluate_all_graphs():
    # ---------------- 步骤 1：全图共用的数据预处理 (只做一次，省时高效) ----------------
    # 获取节点名称
    df_adj = pd.read_csv(os.path.join(GRAPH_DIR, "adj_vae.csv"), index_col=0)
    tickers = df_adj.columns.tolist()
    num_nodes = len(tickers)
    target_indices = {t: tickers.index(t) for t in TARGET_STOCKS.keys()}

    # 加载与对齐数据
    combined_df = load_and_align_data(tickers)
    raw_data = combined_df.values
    
    train_size = int(len(raw_data) * TRAIN_RATIO)
    train_data = raw_data[:train_size]
    test_data = raw_data[train_size:]
    
    print("🛠 进行 Z-score 标准化及极值截断 (防爆显存与 NaN 断裂)...")
    # 【修复1】：改用对趋势更友好的 StandardScaler
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data) 
    
    # 【修复2】：金融量化必备的极值截断 (Clip)。强行把超过 ±5 个标准差的“毒数据”压平
    train_scaled = np.clip(train_scaled, -5.0, 5.0)
    test_scaled = np.clip(test_scaled, -5.0, 5.0)
    
    close_scalers = {}
    for ticker, name in TARGET_STOCKS.items():
        idx = tickers.index(ticker)
        close_col_idx = idx * len(FEATURES) + TARGET_FEATURE_IDX
        target_scaler = StandardScaler() # 这里也要同步改为 StandardScaler
        target_scaler.fit(train_data[:, close_col_idx].reshape(-1, 1))
        close_scalers[ticker] = target_scaler

    # 制作滑动窗口
    X_train, Y_train = build_sliding_windows(train_scaled, LOOK_BACK)
    X_test, Y_test = build_sliding_windows(test_scaled, LOOK_BACK)
    
    X_train = X_train.reshape(-1, LOOK_BACK, num_nodes, len(FEATURES))
    X_test = X_test.reshape(-1, LOOK_BACK, num_nodes, len(FEATURES))
    
    # 提取测试集的真实时间轴，以对齐预测结果保存
    num_samples = len(X_test)
    test_times = combined_df.index[-num_samples:]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n💡 硬件检测: 当前使用设备 -> {device}")
    if device.type == 'cuda':
        print(f"   具体型号: {torch.cuda.get_device_name(0)}")

    X_train_t = torch.FloatTensor(X_train).to(device)
    Y_train_t = torch.FloatTensor(Y_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    Y_test_t = torch.FloatTensor(Y_test).to(device)

    train_dataset = TensorDataset(X_train_t, Y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True) 
    test_dataset = TensorDataset(X_test_t, Y_test_t)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ---------------- 步骤 2：遍历 4 种拓扑图，分别训练和评估 ----------------
    for graph_file in GRAPH_FILES:
        # 解析出图类型名称，如 "adj_vae.npy" -> "vae"
        graph_type = graph_file.replace("adj_", "").replace(".npy", "")
        
        print(f"\n{'='*70}")
        print(f"🚀 开始消融实验分支: 注入拓扑图 -> 【{graph_type.upper()} GRAPH】")
        print(f"{'='*70}")

        # 载入特定的图结构
        adj_matrix = np.load(os.path.join(GRAPH_DIR, graph_file))
        
        # 计算该图的切比雪夫多项式
        T_k_tensors = compute_chebyshev_polynomials(adj_matrix, K=K_ORDER)
        T_k_tensors = [t.to(device) for t in T_k_tensors]
        
        model = ST_Trader(num_nodes=num_nodes, 
                          input_dim=len(FEATURES), 
                          hidden_dim=HIDDEN_DIM, 
                          K=K_ORDER, 
                          T_k_tensors=T_k_tensors).to(device)
                          
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.MSELoss()

        print(f"正在 GPU 上分批训练 GCN-LSTM (Epochs={EPOCHS}, Batch={BATCH_SIZE})...")
        target_idx_list = list(target_indices.values())
        
        # 训练过程
        for epoch in range(EPOCHS):
            model.train()
            epoch_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x) 
                loss = criterion(outputs[:, target_idx_list], batch_y[:, target_idx_list])
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
            # 每 10 轮打印一次，保持终端清爽
            if (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / len(train_loader)
                print(f"[{graph_type.upper()}] Epoch [{epoch+1:03d}/{EPOCHS}], Loss: {avg_loss:.6f}")

        # 评估过程
        print(f"🔬 正在 【{graph_type.upper()}】 结构下进行测试集推理...")
        model.eval()
        test_preds = []
        
        with torch.no_grad():
            for batch_x, _ in test_loader:
                preds = model(batch_x).cpu().numpy()
                test_preds.append(preds)
                
        test_preds = np.concatenate(test_preds, axis=0)
        test_trues = Y_test_t.cpu().numpy()

        for ticker, name in TARGET_STOCKS.items():
            idx = target_indices[ticker]
            
            pred_scaled = test_preds[:, idx].reshape(-1, 1)
            true_scaled = test_trues[:, idx].reshape(-1, 1)
            
            pred_real = close_scalers[ticker].inverse_transform(pred_scaled).flatten()
            true_real = close_scalers[ticker].inverse_transform(true_scaled).flatten()
            
            df_out = pd.DataFrame({
                'trade_time': test_times,
                'True_Price': true_real,
                'Predicted_Price': pred_real
            })
            
            # 【核心修改】：按照要求进行命名，如 STTrader-vae_STK_000001.SZ_predictions.csv
            save_filename = f"STTrader-{graph_type}_STK_{ticker}_predictions.csv"
            df_out.to_csv(os.path.join(PRED_DIR, save_filename), index=False, encoding='utf-8-sig')
            
            # 指标文件的 tag 也同步修改
            model_tag = f"STTrader-{graph_type}_STK_{ticker}"
            metrics = calculate_metrics(true_real, pred_real)
            print_metrics(metrics, model_name=model_tag)
            save_metrics(metrics, model_name=model_tag, save_dir=METRICS_DIR)
            
        print(f"✅ 图结构 【{graph_type.upper()}】 训练与验证收官！数据已归档。")

        # 强制清空显存，释放空间给下一个图网络
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n🎉 全部 4 种空间拓扑图的消融实验流已彻底执行完毕！")

if __name__ == "__main__":
    train_and_evaluate_all_graphs()