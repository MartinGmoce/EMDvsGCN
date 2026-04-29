import os
import pandas as pd
import matplotlib.pyplot as plt

# ================= 配置区域 =================
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False 

PROJECT_ROOT = "/Users/martingao/VScode/EMDvsGCN"
CEEMDAN_DIR = os.path.join(PROJECT_ROOT, "Data", "CEEMDAN_Decomposed")
PLOT_SAVE_DIR = os.path.join(PROJECT_ROOT, "Results", "Decompositions") # 存到你建好的分类目录里

os.makedirs(PLOT_SAVE_DIR, exist_ok=True)

def generate_english_title(filename):
    filename_clean = filename.replace("CEEMDAN_", "").replace("_1min.csv", "")
    parts = filename_clean.split('_')
    if len(parts) >= 2:
        return f"CEEMDAN - {parts[0]}_{parts[1]}"
    return filename_clean

def plot_ceemdan_results():
    csv_files = [f for f in os.listdir(CEEMDAN_DIR) if f.endswith('.csv')]
    
    if not csv_files:
        print("❌ 未找到任何 CEEMDAN 分解文件！")
        return

    print(f"\n{'='*60}")
    print(f"发现 {len(csv_files)} 个分解文件，开始生成多子图学术图表...")
    print(f"{'='*60}")

    for file in csv_files:
        filepath = os.path.join(CEEMDAN_DIR, file)
        df = pd.read_csv(filepath)
        df['trade_time'] = pd.to_datetime(df['trade_time'])

        title_eng = generate_english_title(file)
        
        # 提取列名 (排除 trade_time)
        signal_cols = [col for col in df.columns if col != 'trade_time']
        num_plots = len(signal_cols)
        
        # 动态创建画布，IMF 越多画布越长
        fig, axes = plt.subplots(num_plots, 1, figsize=(12, 2 * num_plots), sharex=True)
        if num_plots == 1:
            axes = [axes]

        fig.suptitle(f"{title_eng} Decomposition Results", fontsize=16, fontweight='bold', y=0.98)

        # 遍历画子图
        for i, col in enumerate(signal_cols):
            ax = axes[i]
            # 原始数据用蓝色，分解分量用黑色，残差用红色
            if col == 'close':
                color = '#1f77b4'
                ylabel = 'Original Signal'
            elif col == 'Residue':
                color = '#d62728'
                ylabel = 'Residue'
            else:
                color = '#333333'
                ylabel = col

            ax.plot(df['trade_time'], df[col], color=color, linewidth=1.0)
            ax.set_ylabel(ylabel, fontsize=10, fontweight='bold')
            ax.grid(True, linestyle=':', alpha=0.6)
            
            # 调整 y 轴刻度，避免重叠
            ax.locator_params(axis='y', nbins=4)

        axes[-1].set_xlabel("Trade Time", fontsize=12)
        plt.xticks(rotation=30, ha='right')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95) # 给主标题留点空间
        
        save_filename = f"{title_eng.replace(' - ', '_')}_Decomp.png"
        save_path = os.path.join(PLOT_SAVE_DIR, save_filename)
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        print(f"✅ 成功生成并保存: {save_filename}")

if __name__ == "__main__":
    plot_ceemdan_results()