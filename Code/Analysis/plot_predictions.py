import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ================= Font Configuration (English Only) =================
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False 

# ================= Configuration =================
PROJECT_ROOT = "/Users/martingao/VScode/EMDvsGCN"
PRED_DIR = os.path.join(PROJECT_ROOT, "Results", "Predictions")
PLOT_SAVE_DIR = os.path.join(PROJECT_ROOT, "Results", "ForecastPlots")

os.makedirs(PLOT_SAVE_DIR, exist_ok=True)

BLUE = '#1f77b4'  
RED = '#d62728'   

def generate_english_title(filename):
    filename_clean = filename.replace("_predictions.csv", "")
    parts = filename_clean.split('_')
    if len(parts) >= 3:
        model = parts[0]
        code_full = f"{parts[1]}_{parts[2]}" 
        return f"{model} - {code_full}"
    return filename_clean 

def plot_all_predictions():
    csv_files = [f for f in os.listdir(PRED_DIR) if f.endswith('_predictions.csv')]
    
    if not csv_files:
        print("❌ No prediction CSVs found in Predictions folder!")
        return

    print(f"\n{'='*70}")
    print(f"Found {len(csv_files)} files. Generating Continuous Academic Plots...")
    print(f"{'='*70}")

    for file in csv_files:
        filepath = os.path.join(PRED_DIR, file)
        df = pd.read_csv(filepath)

        title_eng = generate_english_title(file)

        fig, ax = plt.figure(figsize=(14, 6)), plt.gca()
        
        # 【核心修改】：直接使用 df.index (0, 1, 2...) 作为 X 轴，完美解决断裂问题
        plt.plot(df.index, df['True_Price'], label='True Price', color=BLUE, linewidth=1.5, alpha=0.9)
        plt.plot(df.index, df['Predicted_Price'], label='Predicted Price', color=RED, linewidth=1.5, linestyle='--', alpha=0.9)
        
        plt.title(f"{title_eng} - Forecast Comparison", fontsize=16, fontweight='bold', pad=15)
        # X 轴标签改为更能体现连续性的 "Time Steps"
        plt.xlabel("Time Steps (Trading Minutes)", fontsize=12)
        plt.ylabel("Close Price", fontsize=12)
        plt.legend(loc='upper left', fontsize=12, framealpha=0.9)
        plt.grid(True, linestyle=':', alpha=0.6) 
        
        # 消除首尾两端的留白，让图表更紧凑
        plt.xlim(0, len(df))
        
        plt.tight_layout()
        
        save_filename = f"{title_eng.replace(' - ', '_')}_Plot.png"
        save_path = os.path.join(PLOT_SAVE_DIR, save_filename)
        plt.savefig(save_path, dpi=300)
        plt.close() 
        
        print(f"✅ Generated and saved continuous plot: {save_filename}")

    print("\n🎉 All plots generated! Check Results/ForecastPlots.")

if __name__ == "__main__":
    plot_all_predictions()