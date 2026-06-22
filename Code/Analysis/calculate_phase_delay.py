import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.signal import find_peaks

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from project_config import ANALYSIS_RESULTS_DIR, PREDICTIONS_DIR, ensure_directories

# ================= 配置路径 =================
PRED_DIR = PREDICTIONS_DIR
METRICS_SAVE_DIR = ANALYSIS_RESULTS_DIR

ensure_directories(METRICS_SAVE_DIR)

def calculate_ccf_lag(y_true, y_pred, max_lag=5):
    """
    方法1：互相关函数 (CCF) 计算整体相位延迟
    """
    lags = np.arange(-max_lag, max_lag + 1)
    ccf_values = []
    
    for lag in lags:
        if lag < 0:
            corr = np.corrcoef(y_true[:lag], y_pred[-lag:])[0, 1]
        elif lag > 0:
            corr = np.corrcoef(y_true[lag:], y_pred[:-lag])[0, 1]
        else:
            corr = np.corrcoef(y_true, y_pred)[0, 1]
        ccf_values.append(corr)
        
    best_lag = lags[np.argmax(ccf_values)]
    return best_lag

def calculate_turning_point_delay(y_true, y_pred, distance=10):
    """
    方法2：拐点延迟时间差 (Turning Point Delay)
    """
    # 寻找真实价格和预测价格的波峰
    peaks_true, _ = find_peaks(y_true, distance=distance)
    peaks_pred, _ = find_peaks(y_pred, distance=distance)
    
    delays = []
    for pt in peaks_true:
        # 在真实波峰附近寻找最近的预测波峰
        nearby_preds = peaks_pred[np.abs(peaks_pred - pt) <= 5]
        if len(nearby_preds) > 0:
            closest_pp = nearby_preds[np.argmin(np.abs(nearby_preds - pt))]
            delays.append(closest_pp - pt)
            
    return np.mean(delays) if len(delays) > 0 else 0

def generate_delay_metrics():
    csv_files = [f for f in os.listdir(PRED_DIR) if f.endswith('_predictions.csv')]
    results_list = []
    
    for file in csv_files:
        parts = file.split('_')
        if len(parts) >= 3:
            model_name = parts[0]
            stock_name = f"{parts[1]}_{parts[2]}"
        else:
            continue
            
        df = pd.read_csv(os.path.join(PRED_DIR, file)).head(2000)
        y_true = df['True_Price'].values
        y_pred = df['Predicted_Price'].values
        
        # 计算两种延迟
        ccf_lag = calculate_ccf_lag(y_true, y_pred)
        tp_delay = calculate_turning_point_delay(y_true, y_pred)
        
        results_list.append({
            'Model': model_name,
            'Stock': stock_name,
            'CCF_Lag(步)': round(ccf_lag, 2),
            'TP_Delay(步)': round(tp_delay, 4)
        })
        
    df_metrics = pd.DataFrame(results_list)
    df_metrics.sort_values(by=['Stock', 'Model'], inplace=True)
    
    save_path = os.path.join(METRICS_SAVE_DIR, "phase_delay_metrics.csv")
    df_metrics.to_csv(save_path, index=False, encoding='utf-8-sig')
    print("\n✅ 相位延迟计算完毕！预览：")
    if hasattr(df_metrics, "to_markdown"):
        print(df_metrics.to_markdown(index=False))
    else:
        print(df_metrics.to_string(index=False))

if __name__ == "__main__":
    generate_delay_metrics()
