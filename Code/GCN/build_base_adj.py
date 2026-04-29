import os
import numpy as np
import pandas as pd

# ================= 配置 =================
PROJECT_ROOT = "/Users/martingao/VScode/EMDvsGCN"
GRAPH_DIR = os.path.join(PROJECT_ROOT, "Results", "VAE_Features")
os.makedirs(GRAPH_DIR, exist_ok=True)

# 读取你之前保存的节点列表以确定维度
df_adj = pd.read_csv(os.path.join(GRAPH_DIR, "adj_vae.csv"), index_col=0)
tickers = df_adj.columns.tolist()
n_nodes = len(tickers)

def generate_baseline_graphs():
    print(f"🚀 开始生成 5.2.3 节消融实验所需的基准邻接矩阵 (节点数: {n_nodes})...")

    # 1. Identity Graph (无图结构，退化为独立预测)
    # 对角线为 1，其余全为 0
    adj_identity = np.eye(n_nodes)
    np.save(os.path.join(GRAPH_DIR, "adj_identity.npy"), adj_identity)
    print("✅ 成功生成 No-Graph (Identity Matrix): adj_identity.npy")

    # 2. Random Graph (随机噪音图)
    # 生成 0~1 的随机数，并强制对称 (无向图)
    np.random.seed(42)
    rand_mat = np.random.rand(n_nodes, n_nodes)
    adj_random = (rand_mat + rand_mat.T) / 2
    np.fill_diagonal(adj_random, 1.0) # 自身连接为 1
    np.save(os.path.join(GRAPH_DIR, "adj_random.npy"), adj_random)
    print("✅ 成功生成 Random-Graph (随机对称矩阵): adj_random.npy")

    # 3. Pearson Graph (全连接相关系数图)
    # 注意：这里我们用 VAE 提取的基础特征表来算皮尔逊，以保证公平对比
    features_df = pd.read_csv(os.path.join(PROJECT_ROOT, "Results", "VAE_Features", "fundamental_features.csv"))
    # 提取特征部分并计算样本的相关系数
    feats_only = features_df.drop(columns=['ticker', 'name']).astype(float).values
    adj_pearson = np.corrcoef(feats_only)
    # 取绝对值作为连接强度
    adj_pearson = np.abs(adj_pearson)
    np.save(os.path.join(GRAPH_DIR, "adj_pearson.npy"), adj_pearson)
    print("✅ 成功生成 Pearson-Graph (皮尔逊相关系数矩阵): adj_pearson.npy")

    print("\n🎉 5.2.3 节的所有图矩阵已准备就绪！")

if __name__ == "__main__":
    generate_baseline_graphs()
    
    
# 拿到这 4 个 .npy 文件后，去修改你的 train_st_trader.py 的第 10 行：
# ADJ_PATH = os.path.join(PROJECT_ROOT, "Data", "Graph", "adj_vae.npy")
# 你要做的就是：
# 保持 adj_vae.npy 跑一次，记录下打出来的 MAE/MSE/R2。
# 改成 adj_pearson.npy 跑一次，记录指标。
# 改成 adj_random.npy 跑一次，记录指标。
# 改成 adj_identity.npy 跑一次，记录指标。