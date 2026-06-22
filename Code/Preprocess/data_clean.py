import os
import glob
import sys
from pathlib import Path
import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from project_config import PROCESSED_DATA_DIR, RAW_DATA_DIR, ensure_directories

# ================= 配置区域 =================
ensure_directories(PROCESSED_DATA_DIR)

# 需要保留的特征列（保留真实绝对值）
FEATURE_COLS = ['open', 'high', 'low', 'close', 'vol', 'amount']

# ================= 清洗与对齐逻辑 =================
def clean_and_align_data():
    print("=== 开始高频数据对齐 (严禁提前归一化，防数据泄露) ===")
    
    all_files = glob.glob(os.path.join(RAW_DATA_DIR, "*.csv"))
    if not all_files:
        print("未找到任何原始数据，请先运行 fetch_raw_data.py")
        return
        
    # 1. 寻找上证指数文件作为主时间轴
    index_file = [f for f in all_files if "IDX_000001" in f or "上证指数" in f]
    if not index_file:
        print("错误：缺少上证指数数据，无法建立标准时间轴！")
        return
        
    df_master = pd.read_csv(index_file[0])
    df_master['trade_time'] = pd.to_datetime(df_master['trade_time'])
    master_time_index = df_master['trade_time'].sort_values().unique()
    print(f"建立标准时间轴完成，共计 {len(master_time_index)} 个时间步（分钟）。")
    
    close_prices_dict = {}
    
    # 2. 遍历清洗所有文件
    for file in all_files:
        filename = os.path.basename(file)
        stock_name = filename.split('_')[2] 
        ts_code = filename.split('_')[1]
        
        print(f"正在处理: {stock_name} ...", end=" ")
        
        df = pd.read_csv(file)
        df['trade_time'] = pd.to_datetime(df['trade_time'])
        df = df.set_index('trade_time')
        
        # 将当前股票的数据重置索引为主时间轴
        # 使用 ffill (前向填充) 处理停牌或缺失的分钟，保留最后的交易状态
        df_aligned = df.reindex(master_time_index).ffill().bfill() 
        
        # 提取真实收盘价，存入字典，为构建 GCN 相关性矩阵做准备
        close_prices_dict[f"{ts_code}_{stock_name}"] = df_aligned['close'].copy()
        
        # 仅截取需要的列，不再进行 MinMaxScaler
        df_final = df_aligned[FEATURE_COLS].copy()
        
        # 保存清洗后但未归一化的单表数据
        save_path = os.path.join(PROCESSED_DATA_DIR, f"Cleaned_{filename}")
        df_final.reset_index().rename(columns={'index': 'trade_time'}).to_csv(save_path, index=False, encoding='utf-8-sig')
        print("完成")
        
    # 3. 合并所有收盘价，输出用于 GCN 建图的宽表
    df_all_close = pd.DataFrame(close_prices_dict, index=master_time_index)
    all_close_path = os.path.join(PROCESSED_DATA_DIR, "All_Stocks_Close_Prices.csv")
    df_all_close.reset_index().rename(columns={'index': 'trade_time'}).to_csv(all_close_path, index=False, encoding='utf-8-sig')
    
    print("\n=== 数据清洗完毕 ===")
    print(f"对齐后的真实特征文件已存入: {PROCESSED_DATA_DIR}")
    print("现在它们可以安全地传入基线模型了，归一化将由模型脚本动态处理。")

if __name__ == "__main__":
    clean_and_align_data()
