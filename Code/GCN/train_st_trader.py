import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from st_trader_model import ST_Trader, compute_chebyshev_polynomials

# ================= 配置与路径 =================
PROJECT_ROOT = "/Users/martingao/VScode/EMDvsGCN"
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
        df_feats = df[FEATURES].add_prefix(f"{ticker}_")
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

def train_and_evaluate():
    adj_matrix = np.load(os.path.join(GRAPH_DIR, "adj_vae.npy"))
    df_adj = pd.read_csv(os.path.join(GRAPH_DIR, "adj_vae.csv"), index_col=0)
    tickers = df_adj.columns.tolist()
    num_nodes = len(tickers)
    
    target_indices = {t: tickers.index(t) for t in TARGET_STOCKS.keys()}

    combined_df = load_and_align_data(tickers)
    raw_data = combined_df.values
    
    train_size = int(len(raw_data) * TRAIN_RATIO)
    train_data = raw_data[:train_size]
    test_data = raw_data[train_size:]
    
    print("🛠 进行严格的 Min-Max 归一化 (仅在训练集 fit)...")
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data) 
    
    close_scalers = {}
    for ticker, name in TARGET_STOCKS.items():
        idx = tickers.index(ticker)
        close_col_idx = idx * len(FEATURES) + TARGET_FEATURE_IDX
        target_scaler = MinMaxScaler()
        target_scaler.fit(train_data[:, close_col_idx].reshape(-1, 1))
        close_scalers[ticker] = target_scaler

    X_train, Y_train = build_sliding_windows(train_scaled, LOOK_BACK)
    X_test, Y_test = build_sliding_windows(test_scaled, LOOK_BACK)
    
    X_train = X_train.reshape(-1, LOOK_BACK, num_nodes, len(FEATURES))
    X_test = X_test.reshape(-1, LOOK_BACK, num_nodes, len(FEATURES))
    
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

    T_k_tensors = compute_chebyshev_polynomials(adj_matrix, K=K_ORDER)
    T_k_tensors = [t.to(device) for t in T_k_tensors]
    
    model = ST_Trader(num_nodes=num_nodes, 
                      input_dim=len(FEATURES), 
                      hidden_dim=HIDDEN_DIM, 
                      K=K_ORDER, 
                      T_k_tensors=T_k_tensors).to(device)
                      
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    print(f"\n🚀 开始在 GPU 上分批训练 GCN-LSTM (Epochs={EPOCHS}, Batch={BATCH_SIZE})...")
    target_idx_list = list(target_indices.values())
    
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
            
        if (epoch + 1) % 5 == 0:
            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch [{epoch+1:03d}/{EPOCHS}], Loss: {avg_loss:.6f}")

    print("\n🔬 正在测试集上进行分批推理...")
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
            'True_Price': true_real,
            'Predicted_Price': pred_real
        })
        save_filename = f"STtrader_STK_{ticker}_predictions.csv"
        df_out.to_csv(os.path.join(PRED_DIR, save_filename), index=False)
        
        model_tag = f"ST-Trader_{ticker}"
        metrics = calculate_metrics(true_real, pred_real)
        print_metrics(metrics, model_name=model_tag)
        save_metrics(metrics, model_name=model_tag, save_dir=METRICS_DIR)

    print("\n🎉 全部流程执行完毕！预测曲线数据已就绪，可以运行 plot_predictions.py 了！")

if __name__ == "__main__":
    train_and_evaluate()