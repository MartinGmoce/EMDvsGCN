import os
import pandas as pd
import glob

# ================= 配置区域 =================
BASE_DIR = "/Users/martingao/Documents/毕业论文/数据"
SOURCE_FOLDERS = ['202210', '202211', '202212', '202301']
TARGET_DIR = os.path.join(BASE_DIR, 'GCN')

START_DATE = '2022-10-17'
END_DATE = '2023-01-04'

os.makedirs(TARGET_DIR, exist_ok=True)

# 完整补齐 32 只节点股票字典 (完整代码 : 名称)
STOCKS = {
    '000001.SZ': '平安银行', '600519.SH': '贵州茅台',
    '600036.SH': '招商银行', '601166.SH': '兴业银行', '002142.SZ': '宁波银行', '600000.SH': '浦发银行',
    '601328.SH': '交通银行', '601939.SH': '建设银行', '601398.SH': '工商银行', '601288.SH': '农业银行',
    '601988.SH': '中国银行', '601169.SH': '北京银行',
    '000858.SZ': '五粮液', '000568.SZ': '泸州老窖', '600809.SH': '山西汾酒', '002304.SZ': '洋河股份',
    '000596.SZ': '古井贡酒', '600702.SH': '舍得酒业', '600779.SH': '水井坊', '603589.SH': '口子窖',
    '000799.SZ': '酒鬼酒', '600199.SH': '迎驾贡酒',
    '300750.SZ': '宁德时代', '002594.SZ': '比亚迪', '600031.SH': '三一重工', '600276.SH': '恒瑞医药',
    '601888.SH': '中国中免', '600900.SH': '长江电力', '601012.SH': '隆基绿能', '000333.SZ': '美的集团',
    '601111.SH': '中国国航', '600028.SH': '中国石化'
}

def process_gcn_data():
    print(f"\n{'='*50}")
    print(f"🚀 开始构建 GCN 节点数据集 (共 {len(STOCKS)} 只股票)")
    print(f"时间截取范围: {START_DATE} 至 {END_DATE}")
    print(f"{'='*50}")

    for full_ticker, name in STOCKS.items():
        # 提取纯数字代码去搜寻文件 (例如从 '600036.SH' 提取 '600036')
        numeric_code = full_ticker.split('.')[0]
        
        all_dfs = []
        
        # 遍历 4 个月份的文件夹找文件
        for folder in SOURCE_FOLDERS:
            folder_path = os.path.join(BASE_DIR, folder)
            
            # 使用纯数字代码进行模糊匹配
            search_pattern = os.path.join(folder_path, f"*{numeric_code}*.csv")
            matched_files = glob.glob(search_pattern)
            
            for file in matched_files:
                try:
                    # 读取数据，加入双重编码保护
                    try:
                        df_temp = pd.read_csv(file, encoding='utf-8')
                    except UnicodeDecodeError:
                        df_temp = pd.read_csv(file, encoding='gbk')
                    
                    all_dfs.append(df_temp)
                except Exception as e:
                    print(f"\n⚠️ 读取 {file} 时出错: {e}")

        if not all_dfs:
            print(f"❌ 警告: 未在指定文件夹中找到纯数字代码为 {numeric_code} 的数据！")
            continue
            
        # 拼接该股票所有月份的数据
        df_combined = pd.concat(all_dfs, ignore_index=True)
        
        # 确保 tdate 是标准的 datetime 格式以供比较
        df_combined['tdate'] = pd.to_datetime(df_combined['tdate'])
        
        # 截取指定时间段的数据
        mask = (df_combined['tdate'] >= pd.to_datetime(START_DATE)) & (df_combined['tdate'] <= pd.to_datetime(END_DATE))
        df_filtered = df_combined.loc[mask].copy()
        
        if df_filtered.empty:
            print(f"⚠️ 警告: {full_ticker} 在指定时间范围内没有数据！")
            continue
            
        # 按日期(和时间如果有的话)排序，确保时序连续
        if 'ttime' in df_filtered.columns:
            df_filtered = df_filtered.sort_values(by=['tdate', 'ttime'])
        else:
            df_filtered = df_filtered.sort_values(by=['tdate'])
            
        # 将 datetime 转回字符串格式保存
        df_filtered['tdate'] = df_filtered['tdate'].dt.strftime('%Y-%m-%d')
        
        # =========================================================
        # 【新增核心修改】：强制规范化 code 列！
        # 直接使用字典中的 full_ticker 覆盖原来不规范的（如 596）
        # =========================================================
        if 'code' in df_filtered.columns:
            df_filtered['code'] = full_ticker
        else:
            # 如果原文件连 code 列都没有，就新建一列放在最前面
            df_filtered.insert(0, 'code', full_ticker)
        
        # 输出时穿上“西装”，使用带后缀的完整代码命名
        save_filename = f"GCN_{full_ticker}_{name}.csv"
        save_path = os.path.join(TARGET_DIR, save_filename)
        df_filtered.to_csv(save_path, index=False, encoding='utf-8-sig')
        
        print(f"✅ 完成: {full_ticker} ({name}) -> 提取了 {len(df_filtered)} 条记录，并已修复 code 列.")

    print(f"\n🎉 全部处理完毕！GCN 节点数据已存放至: {TARGET_DIR}")

if __name__ == "__main__":
    process_gcn_data()