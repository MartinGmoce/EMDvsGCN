import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    from sklearn.metrics import mean_absolute_percentage_error
except ImportError:
    def mean_absolute_percentage_error(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        denominator = np.maximum(np.abs(y_true), np.finfo(float).eps)
        return np.mean(np.abs((y_true - y_pred) / denominator))

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from project_config import ANALYSIS_RESULTS_DIR, PREDICTIONS_DIR, ensure_directories

# ================= 配置路径 =================
PRED_DIR = PREDICTIONS_DIR
METRICS_SAVE_DIR = ANALYSIS_RESULTS_DIR

# 确保保存目录存在
ensure_directories(METRICS_SAVE_DIR)

def calculate_metrics(y_true, y_pred):
    """
    统一计算四个评估指标: MAE, MSE, R2, MAPE
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    # sklearn 的 MAPE 默认分母是真实值 y_true，符合学术规范
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    return mae, mse, r2, mape

def generate_total_metrics():
    if not os.path.exists(PRED_DIR):
        print(f"❌ 找不到预测结果文件夹: {PRED_DIR}")
        return
        
    csv_files = [f for f in os.listdir(PRED_DIR) if f.endswith('_predictions.csv')]
    
    if not csv_files:
        print("❌ Predictions 文件夹中没有找到任何 _predictions.csv 文件！")
        return
        
    print(f"\n🔍 找到 {len(csv_files)} 个预测文件，正在截取前 2000 步并计算指标...")
    
    results_list = []
    
    for file in csv_files:
        # 解析文件名提取 Model 和 Stock
        # 例: StockCI_STK_600519.SH_predictions.csv
        # parts 会变成 ['StockCI', 'STK', '600519.SH', 'predictions.csv']
        parts = file.split('_')
        
        if len(parts) >= 3:
            model_name = parts[0]                         # 第一个 '_' 前的是 Model
            stock_name = f"{parts[1]}_{parts[2]}"         # 第一个 '_' 到第三个 '_' 中间的是 Stock
        else:
            # 兼容处理不符合命名规范的文件
            model_name = "Unknown"
            stock_name = "Unknown"
            
        filepath = os.path.join(PRED_DIR, file)
        df = pd.read_csv(filepath)
        
        # 【核心逻辑】：只截取前 2000 个时间步
        df_2000 = df.head(2000)
        
        y_true = df_2000['True_Price'].values
        y_pred = df_2000['Predicted_Price'].values
        
        # 计算指标
        mae, mse, r2, mape = calculate_metrics(y_true, y_pred)
        
        # 按要求的六列格式存入字典
        results_list.append({
            'Model': model_name,
            'Stock': stock_name,
            'mae': mae,
            'mse': mse,
            'r2': r2,
            'mape': mape
        })
        
    # 将结果转换为 DataFrame
    total_metrics_df = pd.DataFrame(results_list)
    
    # 按照 Stock（标的）和 Model（模型）排序，这样在论文里看对比会非常清晰
    total_metrics_df.sort_values(by=['Stock', 'Model'], inplace=True)
    
    # 保存汇总文件
    save_path = os.path.join(METRICS_SAVE_DIR, "total_metrics.csv")
    total_metrics_df.to_csv(save_path, index=False, encoding='utf-8-sig')
    
    # 在终端打印一个漂亮的 Markdown 表格预览
    print("\n✅ 所有指标计算完毕！预览如下：")
    if hasattr(total_metrics_df, "to_markdown"):
        print(total_metrics_df.to_markdown(index=False))
    else:
        print(total_metrics_df.to_string(index=False))
    print(f"\n💾 汇总结果已成功保存至: {save_path}")

if __name__ == "__main__":
    generate_total_metrics()
