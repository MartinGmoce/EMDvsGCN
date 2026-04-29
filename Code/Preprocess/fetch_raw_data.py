import os
import time
import tushare as ts
import pandas as pd

# ================= 配置区域 =================
PROJECT_ROOT = "/Users/martingao/VScode/EMDvsGCN"
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "Data", "Raw")
os.makedirs(RAW_DATA_DIR, exist_ok=True)

TOKEN = "fa7e5b0b03f653cda8eaaa707a59eb0cea10cabce18914187b2935b9214b"

# 按照第三方镜像站要求初始化 pro
pro = ts.pro_api(TOKEN)
pro._DataApi__token = TOKEN
pro._DataApi__http_url = 'http://lianghua.nanyangqiankun.top'

# Tushare 分钟线的时间格式要求 'YYYY-MM-DD HH:MM:SS'
START_DATE = "2022-10-17 09:30:00"
END_DATE = "2023-01-04 15:00:00"

# 标的字典不变
CORE_STOCKS = {"000001.SZ": "平安银行", "600519.SH": "贵州茅台"}
INDICES = {"000001.SH": "上证指数", "399001.SZ": "深证成指"}
GCN_STOCKS = {
    "600036.SH": "招商银行", "601318.SH": "中国平安", "601166.SH": "兴业银行", "600030.SH": "中信证券", "300059.SZ": "东方财富",
    "000858.SZ": "五粮液", "000568.SZ": "泸州老窖", "600887.SH": "伊利股份", "000333.SZ": "美的集团", "000651.SZ": "格力电器",
    "300750.SZ": "宁德时代", "002594.SZ": "比亚迪", "601012.SH": "隆基绿能", "300274.SZ": "阳光电源", "600438.SH": "通威股份",
    "688981.SH": "中芯国际", "002475.SZ": "立讯精密", "002371.SZ": "北方华创", "002415.SZ": "海康威视", "002230.SZ": "科大讯飞",
    "600276.SH": "恒瑞医药", "300760.SZ": "迈瑞医疗", "603259.SH": "药明康德", "600436.SH": "片仔癀", "300015.SZ": "爱尔眼科",
    "601899.SH": "紫金矿业", "600309.SH": "万华化学", "601668.SH": "中国建筑", "601088.SH": "中国神华", "601816.SH": "京沪高铁"
}

# ================= 抓取逻辑 =================
def fetch_tushare_min_data(ts_code, name, is_index=False):
    print(f"正在抓取: {name} ({ts_code})...", end=" ")
    try:
        # 【核心修正】：修正 Tushare 底层调用名称 idx_mins 和 stk_mins
        if is_index:
            df = pro.idx_mins(ts_code=ts_code, start_date=START_DATE, end_date=END_DATE, freq='1min')
        else:
            df = pro.stk_mins(ts_code=ts_code, start_date=START_DATE, end_date=END_DATE, freq='1min')
        
        if df is not None and not df.empty:
            df['trade_time'] = pd.to_datetime(df['trade_time'])
            df = df.sort_values('trade_time').reset_index(drop=True)
            
            prefix = "IDX" if is_index else "STK"
            file_path = os.path.join(RAW_DATA_DIR, f"{prefix}_{ts_code.replace('.','')}_{name}_1min.csv")
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
            
            print(f"成功！获取到 {len(df)} 条数据。")
            return True
        else:
            print("失败：返回为空。")
            return False
            
    except Exception as e:
        print(f"异常报错: {e}")
        return False

if __name__ == "__main__":
    print("=== Tushare 第三方镜像 1分钟线获取启动 ===")
    
    for ts_code, name in INDICES.items():
        fetch_tushare_min_data(ts_code, name, is_index=True)
        time.sleep(0.5) 
        
    for ts_code, name in CORE_STOCKS.items():
        fetch_tushare_min_data(ts_code, name, is_index=False)
        time.sleep(0.5)
        
    for ts_code, name in GCN_STOCKS.items():
        fetch_tushare_min_data(ts_code, name, is_index=False)