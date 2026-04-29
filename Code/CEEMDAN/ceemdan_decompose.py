import os
import sys
import numpy as np
import pandas as pd
from PyEMD import CEEMDAN
import time

# ================= 配置区域 =================
PROJECT_ROOT = "/Users/martingao/VScode/EMDvsGCN"
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "Data", "Processed")
CEEMDAN_DATA_DIR = os.path.join(PROJECT_ROOT, "Data", "CEEMDAN_Decomposed")

os.makedirs(CEEMDAN_DATA_DIR, exist_ok=True)

# 批量待分解文件列表
TEST_FILES = [
    "Cleaned_STK_000001.SZ_平安银行_1min.csv",
    "Cleaned_STK_600519.SH_贵州茅台_1min.csv",
    "Cleaned_IDX_000001.SH_上证指数_1min.csv",
    "Cleaned_IDX_399001.SZ_深证成指_1min.csv"
]

# === 护机模式超参数 ===
TRIALS = 50       # 从 50 降到 15，提速 3 倍以上！
NOISE_SCALE = 0.2 

# ================= 核心分解逻辑 =================
def decompose_data(filename):
    file_path = os.path.join(PROCESSED_DATA_DIR, filename)
    if not os.path.exists(file_path):
        print(f"⚠️ 跳过: 找不到文件 {file_path}")
        return

    print(f"\n{'='*50}")
    print(f"🔄 正在分解: {filename}")
    print(f"{'='*50}")

    df = pd.read_csv(file_path)
    close_price = df['close'].values

    ceemdan = CEEMDAN(trials=TRIALS, epsilon=NOISE_SCALE)
    ceemdan.extrema_detection = "parabol"

    print(f"⏳ 开始执行 CEEMDAN (Trials={TRIALS})... 请耐心等待...")
    start_time = time.time()
    
    imfs = ceemdan.ceemdan(close_price)
    
    end_time = time.time()
    print(f"✅ 分解成功！耗时: {end_time - start_time:.2f} 秒")

    reconstructed_signal = np.sum(imfs, axis=0)
    max_error = np.max(np.abs(reconstructed_signal - close_price))
    print(f"📊 校验误差 (Max Error): {max_error:.6e}")

    df_imfs = pd.DataFrame({'trade_time': df['trade_time'], 'close': close_price})
    
    for i in range(imfs.shape[0] - 1):
        df_imfs[f'IMF{i+1}'] = imfs[i]
    df_imfs['Residue'] = imfs[-1]

    save_filename = filename.replace("Cleaned_", "CEEMDAN_")
    save_path = os.path.join(CEEMDAN_DATA_DIR, save_filename)
    df_imfs.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"💾 已保存至: {save_filename}")

if __name__ == "__main__":
    print("🚀 批量 CEEMDAN 分解任务启动！")
    total_start = time.time()
    
    for file in TEST_FILES:
        decompose_data(file)
        
    print(f"\n🎉 所有分解任务已完成！总耗时: {(time.time() - total_start)/60:.2f} 分钟")