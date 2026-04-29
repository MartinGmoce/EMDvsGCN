import os
import pandas as pd

# ================= 配置区域 =================
PROJECT_ROOT = "/Users/martingao/VScode/EMDvsGCN"
XLSX_DIR = os.path.join(PROJECT_ROOT, "Data", "xlsx") # 更新：读取目录改为 Data/xlsx
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "Data", "Processed") # 输出目录保持不变

# 确保目录存在
os.makedirs(XLSX_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# 你的 Excel 文件名到标准 CSV 文件名的映射字典
FILE_MAPPING = {
    "PA_close.xlsx": "Cleaned_STK_000001.SZ_平安银行_1min.csv",
    "MT_close.xlsx": "Cleaned_STK_600519.SH_贵州茅台_1min.csv",
    "SSEC_close.xlsx": "Cleaned_IDX_000001.SH_上证指数_1min.csv",
    "SZI_close.xlsx": "Cleaned_IDX_399001.SZ_深证成指_1min.csv"
}

# 伪造一个起点时间，满足 Informer 提取时间特征的需求
START_DATE = "2022-10-17 09:30:00"

# ================= 转换逻辑 =================
def convert_local_excel():
    print(f"=== 开始转换本地 Excel 数据 ===")
    
    for xlsx_name, csv_name in FILE_MAPPING.items():
        xlsx_path = os.path.join(XLSX_DIR, xlsx_name)
        
        if not os.path.exists(xlsx_path):
            print(f"⚠️ 找不到文件: {xlsx_name}，请检查是否已放入 Data/xlsx 目录。")
            continue
            
        print(f"正在处理: {xlsx_name} ...", end=" ")
        
        # 读取 Excel，因为没有表头，设置 header=None
        df = pd.read_excel(xlsx_path, header=None)
        
        # 强制命名第一列为 close
        df.columns = ['close']
        
        # 生成连续的分钟级时间戳 (Informer 刚需)
        time_index = pd.date_range(start=START_DATE, periods=len(df), freq='1min')
        
        # 组装成最终满足 Baseline 要求的 DataFrame
        df_final = pd.DataFrame({
            'trade_time': time_index,
            'close': df['close']
        })
        
        # 保存为标准的 Cleaned_xxx.csv
        save_path = os.path.join(PROCESSED_DIR, csv_name)
        df_final.to_csv(save_path, index=False, encoding='utf-8-sig')
        
        print(f"✅ 完成！已输出 -> {csv_name} (共 {len(df)} 条)")

if __name__ == "__main__":
    convert_local_excel()