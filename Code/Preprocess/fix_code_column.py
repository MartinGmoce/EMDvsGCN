import os
import glob
import pandas as pd

# ================= 配置区域 =================
# 您指定的 GCN 数据存放目录
TARGET_DIR = "/Users/martingao/VScode/EMDvsGCN/Data/GCN"

def fix_code_column():
    print(f"{'='*50}")
    print(f"🚀 开始遍历文件夹并修复 code 列...")
    print(f"📁 目标目录: {TARGET_DIR}")
    print(f"{'='*50}")
    
    # 匹配目录下所有的 .csv 文件
    search_pattern = os.path.join(TARGET_DIR, "*.csv")
    csv_files = glob.glob(search_pattern)
    
    if not csv_files:
        print("❌ 未在该目录下找到任何 CSV 文件，请检查路径是否正确！")
        return

    success_count = 0
    
    for file_path in csv_files:
        filename = os.path.basename(file_path) # 获取纯文件名，例如 GCN_000596.SZ_古井贡酒.csv
        
        try:
            # 按照下划线切割，提取索引为 1 的部分作为标准代码
            parts = filename.replace('.csv', '').split('_')
            if len(parts) >= 2:
                standard_code = parts[1]  # 提取出的结果如: 000596.SZ
            else:
                print(f"⚠️ 文件名格式不符合预期，跳过: {filename}")
                continue
            
            # 读取 CSV 文件
            df = pd.read_csv(file_path)
            
            # 核心操作：将 code 列全部强制替换为标准代码
            if 'code' in df.columns:
                df['code'] = standard_code
            else:
                # 如果原本没有 code 列，则在第 0 列插入
                df.insert(0, 'code', standard_code)
            
            # 原地覆盖保存（保留 utf-8-sig 编码防止中文乱码）
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
            
            print(f"✅ 成功修复: {filename} -> 内部 code 列已全部更新为 [{standard_code}]")
            success_count += 1
            
        except Exception as e:
            print(f"❌ 处理文件 {filename} 时发生错误: {e}")
            
    print(f"\n🎉 批量修复完毕！共成功处理 {success_count} 个文件。")

if __name__ == "__main__":
    fix_code_column() 