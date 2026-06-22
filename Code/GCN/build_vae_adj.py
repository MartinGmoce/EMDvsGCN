import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import pairwise_distances

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from project_config import VAE_FEATURES_DIR, configure_matplotlib_backend, ensure_directories

configure_matplotlib_backend()
import matplotlib.pyplot as plt

# ================= 配置区域 =================
FUNDAMENTAL_FILE = VAE_FEATURES_DIR / "fundamental_features.csv"
MATRIX_SAVE_DIR = VAE_FEATURES_DIR
PLOT_SAVE_DIR = VAE_FEATURES_DIR

ensure_directories(MATRIX_SAVE_DIR, PLOT_SAVE_DIR)

# 论文指定的超参数 [cite: 270]
LATENT_DIM = 16          # 隐空间维度 h=16
GAMMA_SQ = 0.1           # 公式(6)中的 gamma^2
EPSILON = 0.5            # 公式(6)中的 epsilon (控制稀疏性)
EPOCHS = 1000            # 训练轮数
LR = 1e-4                # 学习率

# ================= VAE 网络定义 [cite: 147, 185] =================
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=16):
        super(VAE, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

# 损失函数: 重构误差 + KL散度 [cite: 147, 190]
def loss_function(recon_x, x, mu, logvar):
    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD

# ================= 建图流程 =================
def build_vae_matrix():
    print("🧠 启动 VAE 隐空间建图...")
    
    # 1. 加载特征数据
    df = pd.read_csv(FUNDAMENTAL_FILE)
    tickers = df['ticker'].tolist()
    
    # 加上 .astype(float) 强制将 True/False 转换为 1.0/0.0
    features = df.drop(columns=['ticker', 'name']).astype(float).values
    
    x_tensor = torch.FloatTensor(features)
    
    # 2. 训练 VAE
    model = VAE(input_dim=features.shape[1], latent_dim=LATENT_DIM)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        recon, mu, logvar = model(x_tensor)
        loss = loss_function(recon, x_tensor, mu, logvar)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 200 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    # 3. 提取隐向量 z 
    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(x_tensor)
        z = mu.numpy()

    # 4. 计算邻接矩阵 
    dist_matrix = pairwise_distances(z, metric='euclidean')
    n = len(tickers)
    A = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j: 
                A[i,j] = 1.0
                continue
            
            d_sq = dist_matrix[i,j]**2
            val = np.exp(-d_sq / GAMMA_SQ)
            
            # 严格遵循论文公式(6)的稀疏化条件
            if val >= EPSILON:
                A[i,j] = val
            else:
                A[i,j] = 0.0
                
    # 5. 保存
    np.save(os.path.join(MATRIX_SAVE_DIR, "adj_vae.npy"), A)
    pd.DataFrame(A, index=tickers, columns=tickers).to_csv(os.path.join(MATRIX_SAVE_DIR, "adj_vae.csv"))
    print(f"✅ 邻接矩阵已生成并保存至 {MATRIX_SAVE_DIR}")
    return A, tickers

def plot_adj(A, tickers):
    plt.figure(figsize=(12, 10))
    sns.heatmap(A, xticklabels=[t.split('.')[0] for t in tickers], 
                yticklabels=[t.split('.')[0] for t in tickers], cmap="YlGnBu")
    plt.title("VAE-based Adjacency Matrix (ST-Trader Eq.6)")
    plt.savefig(os.path.join(PLOT_SAVE_DIR, "VAE_Adj_Matrix.png"))
    print(f"🖼️ 热力图已保存至 {PLOT_SAVE_DIR}")

if __name__ == "__main__":
    adj, tks = build_vae_matrix()
    plot_adj(adj, tks)
