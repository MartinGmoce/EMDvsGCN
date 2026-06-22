import os
import glob
import sys
from pathlib import Path
import pandas as pd
import numpy as np

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from project_config import GCN_DATA_DIR, VAE_FEATURES_DIR, ensure_directories

# ================= 配置区域 =================
GCN_DIR = GCN_DATA_DIR
TARGET_DIR = VAE_FEATURES_DIR
ensure_directories(TARGET_DIR)

# 沿用极其严格的行业映射字典 (作为 VAE 的先验分类特征)
SECTOR_DICT = {
    '000001.SZ': 'Finance', '600519.SH': 'Liquor',
    '600036.SH': 'Finance', '601166.SH': 'Finance', '002142.SZ': 'Finance', '600000.SH': 'Finance',
    '601328.SH': 'Finance', '601939.SH': 'Finance', '601398.SH': 'Finance', '601288.SH': 'Finance',
    '601988.SH': 'Finance', '601169.SH': 'Finance',
    '000858.SZ': 'Liquor', '000568.SZ': 'Liquor', '600809.SH': 'Liquor', '002304.SZ': 'Liquor', 
    '000596.SZ': 'Liquor', '600702.SH': 'Liquor', '600779.SH': 'Liquor', '603589.SH': 'Liquor', 
    '000799.SZ': 'Liquor', '600199.SH': 'Liquor',
    '300750.SZ': 'EV', '002594.SZ': 'EV', '600031.SH': 'Machinery', '600276.SH': 'Pharma',
    '601888.SH': 'Tourism', '600900.SH': 'Utility', '601012.SH': 'Solar', '000333.SZ': 'Appliance',
    '601111.SH': 'Airline', '600028.SH': 'Energy'
}

def extract_local_features():
    print(f"{'='*50}")
    print("🚀 开始从本地高频分钟线提取 VAE 节点特征...")
    print(f"{'='*50}")
    
    feature_list = []
    
    # 查找 GCN 文件夹下所有的 csv 文件
    file_pattern = os.path.join(GCN_DIR, "GCN_*.csv")
    matched_files = glob.glob(file_pattern)
    
    if not matched_files:
        print(f"❌ 警告：在 {GCN_DIR} 未找到任何以 GCN_ 开头的 CSV 文件！")
        return

    for file_path in matched_files:
        # 从文件名解析标准代码和名称 (例如: GCN_000596.SZ_古井贡酒.csv)
        filename = os.path.basename(file_path)
        parts = filename.replace('.csv', '').split('_')
        
        if len(parts) >= 3:
            full_ticker = parts[1]  # '000596.SZ'
            name = parts[2]         # '古井贡酒'
        else:
            print(f"⚠️ 文件名格式不符，跳过: {filename}")
            continue
            
        print(f"📊 正在计算: {full_ticker} ({name})")
        
        try:
            # 读取分钟线数据
            df = pd.read_csv(file_path)
            
            # ========================================================
            # 【核心防泄漏修改】：严格按照 80/20 划分，只截取前 80% 的数据！
            # 保证 VAE 特征提取绝对不包含测试集（未来）的信息
            # ========================================================
            train_size = int(len(df) * 0.8)
            df_train = df.iloc[:train_size].copy()  # 只取前 80%
            
            # 确保按时间排序，计算分钟收益率 (注意：这里全是基于 df_train)
            df_train['pct_change'] = df_train['close'].pct_change().fillna(0)
            
            # --- 计算基于高频数据的 5 大统计画像特征 (仅限训练集) ---
            
            # 1. 区间累计收益率 (Total Return)
            total_return = (df_train['close'].iloc[-1] / df_train['close'].iloc[0]) - 1
            
            # 2. 分钟级波动率 (Volatility) 
            volatility = df_train['pct_change'].std() 
            
            # 3. 平均分钟成交额 (Avg Amount)
            avg_amount = df_train['cje'].mean() 
            
            # 4. 收益率偏度 (Skewness)
            skewness = df_train['pct_change'].skew()
            
            # 5. 最大回撤 (Max Drawdown)
            cummax = df_train['close'].cummax()
            max_drawdown = ((cummax - df_train['close']) / cummax).max()
            
            feature_list.append({
                'ticker': full_ticker,
                'name': name,
                'Industry': SECTOR_DICT.get(full_ticker, 'Other'), 
                'Total_Return': total_return,
                'Volatility': volatility,
                'Avg_Amount': avg_amount,
                'Skewness': skewness,
                'Max_Drawdown': max_drawdown
            })
            
        except Exception as e:
            print(f"❌ 处理文件 {filename} 时出错: {e}")
            
    # 转为 DataFrame
    df_features = pd.DataFrame(feature_list)
    
    print("\n🛠 开始数据标准化和分类变量 One-Hot 编码...")
    
    # 1. 连续特征标准化 (Z-score)，因为量纲完全不同（如成交额几千万 vs 收益率0.01）
    num_cols = ['Total_Return', 'Volatility', 'Avg_Amount', 'Skewness', 'Max_Drawdown']
    df_features[num_cols] = (df_features[num_cols] - df_features[num_cols].mean()) / df_features[num_cols].std()
    
    # 2. 行业 One-Hot 编码
    df_encoded = pd.get_dummies(df_features, columns=['Industry'], prefix='IND')
    
    # 调整列的展示顺序
    cols = ['ticker', 'name'] + [c for c in df_encoded.columns if c not in ['ticker', 'name']]
    df_encoded = df_encoded[cols]
    
    # 3. 覆盖保存为 fundamental_features.csv
    save_path = os.path.join(TARGET_DIR, "fundamental_features.csv")
    df_encoded.to_csv(save_path, index=False, encoding='utf-8-sig')
    
    print(f"\n✅ 基于高频数据的特征矩阵生成完毕！共 {len(df_encoded)} 只股票。")
    print(f"💾 数据已就绪: {save_path}")

if __name__ == "__main__":
    extract_local_features()
