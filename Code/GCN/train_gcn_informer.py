import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from st_trader_model import compute_chebyshev_polynomials
from gcn_informer_model import GCN_Informer # 导入 GCN_Informer

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
BATCH_SIZE = 64      
HIDDEN_DIM = 128     # 空间特征隐藏维度
K_ORDER = 3          
EPOCHS = 100         
LR = 0.001
TRAIN_RATIO = 0.8    
N_HEADS = 8          # Informer 注意力头数

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
    # ---------------- 步骤 1：全图共用的数据预处理 ----------------
    df_adj = pd.read_csv(os.path.join(GRAPH_DIR, "adj_vae.csv"), index_col=0)
    tickers = df_adj.columns.tolist()
    num_nodes = len(tickers)
    target_indices = {t: tickers.index(t) for t in TARGET_STOCKS.keys()}

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
    

    X_train, Y_train = build_sliding_windows(train_scaled, LOOK_BACK)
    X_test, Y_test = build_sliding_windows(test_scaled, LOOK_BACK)
    
    X_train = X_train.reshape(-1, LOOK_BACK, num_nodes, len(FEATURES))
    X_test = X_test.reshape(-1, LOOK_BACK, num_nodes, len(FEATURES))
    
    num_samples = len(X_test)
    test_times = combined_df.index[-num_samples:]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n💡 硬件检测: 当前使用设备 -> {device}")

    X_train_t = torch.FloatTensor(X_train).to(device)
    Y_train_t = torch.FloatTensor(Y_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    Y_test_t = torch.FloatTensor(Y_test).to(device)

    train_dataset = TensorDataset(X_train_t, Y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True) 
    test_dataset = TensorDataset(X_test_t, Y_test_t)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ---------------- 步骤 2：遍历 4 种拓扑图，分别训练 GCN_Informer ----------------
    for graph_file in GRAPH_FILES:
        graph_type = graph_file.replace("adj_", "").replace(".npy", "")
        
        print(f"\n{'='*70}")
        print(f"🚀 开始 GCN_Informer 分支: 注入拓扑图 -> 【{graph_type.upper()} GRAPH】")
        print(f"{'='*70}")

        adj_matrix = np.load(os.path.join(GRAPH_DIR, graph_file))
        T_k_tensors = compute_chebyshev_polynomials(adj_matrix, K=K_ORDER)
        T_k_tensors = [t.to(device) for t in T_k_tensors]
        
        # 初始化 GCN_Informer 模型
        model = GCN_Informer(num_nodes=num_nodes, 
                             input_dim=len(FEATURES), 
                             hidden_dim=HIDDEN_DIM, 
                             K=K_ORDER, 
                             T_k_tensors=T_k_tensors,
                             n_heads=N_HEADS).to(device)
                          
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.MSELoss()

        print(f"正在训练 GCN_Informer (Epochs={EPOCHS})...")
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
                
            if (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / len(train_loader)
                print(f"[GCN_Informer-{graph_type}] Epoch [{epoch+1:03d}/{EPOCHS}], Loss: {avg_loss:.6f}")

        # 推理与保存
        print(f"🔬 正在 【{graph_type.upper()}】 结构下进行推理...")
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
            
            # 【核心修改】：前缀改为 GCNInformer
            save_filename = f"GCNInformer-{graph_type}_STK_{ticker}_predictions.csv"
            df_out.to_csv(os.path.join(PRED_DIR, save_filename), index=False, encoding='utf-8-sig')
            
            model_tag = f"GCNInformer-{graph_type}_STK_{ticker}"
            metrics = calculate_metrics(true_real, pred_real)
            print_metrics(metrics, model_name=model_tag)
            save_metrics(metrics, model_name=model_tag, save_dir=METRICS_DIR)
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n🎉 全部 4 种拓扑图的 GCN_Informer 消融实验流已彻底执行完毕！")

if __name__ == "__main__":
    train_and_evaluate_all_graphs()