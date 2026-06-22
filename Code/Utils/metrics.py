import os
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

def calculate_metrics(y_true, y_pred):
    """
    统一计算四个评估指标: MAE, MSE, MAPE, R2
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    
    # sklearn 的 MAPE 默认分母是真实值 y_true，符合学术规范
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    r2 = r2_score(y_true, y_pred)
    
    return {
        "MAE": mae,
        "MSE": mse,
        "MAPE": mape,
        "R2": r2
    }

def print_metrics(metrics_dict, model_name="Model"):
    """
    在终端格式化打印评估结果
    """
    print("-" * 40)
    print(f"【{model_name} 测试集评估结果】(真实价格尺度)")
    print(f"MSE  : {metrics_dict['MSE']:.6f}")
    print(f"MAE  : {metrics_dict['MAE']:.6f}")
    print(f"MAPE : {metrics_dict['MAPE']*100:.4f}%") 
    print(f"R2   : {metrics_dict['R2']:.4f}")
    print("-" * 40)

def save_metrics(metrics_dict, model_name, save_dir):
    """
    将计算出的指标保存为独立的 CSV 文件到 Results/Metrics 目录下
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 转换为 DataFrame，并插入一列模型名称，方便未来合并各个模型的表
    df = pd.DataFrame([metrics_dict])
    df.insert(0, 'Model', model_name)
    
    save_path = os.path.join(save_dir, f"{model_name}_metrics.csv")
    df.to_csv(save_path, index=False)
    print(f"✅ {model_name} 的四大评估指标已成功保存至: {save_path}")
