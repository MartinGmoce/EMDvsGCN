import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ================= 配置 =================
PROJECT_ROOT = "/Users/martingao/VScode/EMDvsGCN"
GRAPH_DIR = os.path.join(PROJECT_ROOT, "Results", "VAE_Features")
# 复用你之前代码中的 PLOT_SAVE_DIR 路径，确保图片保存在一起
PLOT_SAVE_DIR = os.path.join(PROJECT_ROOT, "Results", "VAE_Features")

os.makedirs(GRAPH_DIR, exist_ok=True)
os.makedirs(PLOT_SAVE_DIR, exist_ok=True)

# 读取你之前保存的节点列表以确定维度
df_adj = pd.read_csv(os.path.join(GRAPH_DIR, "adj_vae.csv"), index_col=0)
tickers = df_adj.columns.tolist()
n_nodes = len(tickers)

def plot_matrix(matrix, title, filename):
    """通用绘图函数"""
    plt.figure(figsize=(12, 10))
    # 为了美观，标签只显示 ticker 的前缀（去掉 .SH/.SZ 等后缀），如果不需要可以去掉 split
    short_labels = [t.split('.')[0] for t in tickers]
    
    sns.heatmap(matrix, xticklabels=short_labels, yticklabels=short_labels, cmap="YlGnBu")
    plt.title(title)
    
    save_path = os.path.join(PLOT_SAVE_DIR, filename)
    plt.savefig(save_path)
    plt.close() # 关闭画板释放内存
    print(f"🖼️ 图表已保存: {filename}")

def generate_baseline_graphs():
    print(f"🚀 开始生成 5.2.3 节消融实验所需的基准邻接矩阵 (节点数: {n_nodes})...")

    # 1. Identity Graph (无图结构，退化为独立预测)
    adj_identity = np.eye(n_nodes)
    np.save(os.path.join(GRAPH_DIR, "adj_identity.npy"), adj_identity)
    plot_matrix(adj_identity, "Identity Adjacency Matrix (No-Graph)", "Adj_Identity_Matrix.png")
    print("✅ 成功生成 No-Graph (Identity Matrix)")

    # 2. Random Graph (随机噪音图)
    np.random.seed(42)
    rand_mat = np.random.rand(n_nodes, n_nodes)
    adj_random = (rand_mat + rand_mat.T) / 2
    np.fill_diagonal(adj_random, 1.0) 
    np.save(os.path.join(GRAPH_DIR, "adj_random.npy"), adj_random)
    plot_matrix(adj_random, "Random Adjacency Matrix", "Adj_Random_Matrix.png")
    print("✅ 成功生成 Random-Graph (随机对称矩阵)")

    # 3. Pearson Graph (全连接相关系数图)
    features_df = pd.read_csv(os.path.join(PROJECT_ROOT, "Results", "VAE_Features", "fundamental_features.csv"))
    feats_only = features_df.drop(columns=['ticker', 'name']).astype(float).values
    adj_pearson = np.corrcoef(feats_only)
    adj_pearson = np.abs(adj_pearson)
    np.save(os.path.join(GRAPH_DIR, "adj_pearson.npy"), adj_pearson)
    plot_matrix(adj_pearson, "Pearson Correlation Adjacency Matrix", "Adj_Pearson_Matrix.png")
    print("✅ 成功生成 Pearson-Graph (皮尔逊相关系数矩阵)")

    print("\n🎉 5.2.3 节的所有图矩阵及可视化结果已准备就绪！")

if __name__ == "__main__":
    generate_baseline_graphs()