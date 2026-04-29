import os
import sys
import time
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
import warnings

# 忽略 statsmodels 的收敛警告
warnings.filterwarnings("ignore")

# 引入通用指标计算模块
PROJECT_ROOT = "/Users/martingao/VScode/EMDvsGCN"
sys.path.append(os.path.join(PROJECT_ROOT, "Code"))
from Utils.metrics import calculate_metrics, print_metrics, save_metrics

# ================= 配置区域 =================
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "Data", "Processed")
PRED_SAVE_DIR = os.path.join(PROJECT_ROOT, "Results", "Predictions")
METRICS_SAVE_DIR = os.path.join(PROJECT_ROOT, "Results", "Metrics") 

os.makedirs(PRED_SAVE_DIR, exist_ok=True)
os.makedirs(METRICS_SAVE_DIR, exist_ok=True)

# ARIMA 参数设置
SEQ_LEN = 512       
PRED_LEN = 96       
EVAL_STRIDE = 96    # 步长等于预测长度，无重叠滚动
ARIMA_ORDER = (3, 1, 2) 

TEST_FILE = "Cleaned_IDX_399001.SZ_深证成指_1min.csv"

# ================= 核心预测逻辑 =================
def evaluate_arima():
    # 动态提取股票特征标识，防止结果被覆盖
    stock_id = TEST_FILE.replace("Cleaned_", "").replace("_1min.csv", "")
    model_id = f"ARIMA_{stock_id}"

    print(f"\n{'='*40}")
    print(f"启动 ARIMA 基准测试: {model_id}")
    print(f"{'='*40}")
    
    file_path = os.path.join(PROCESSED_DATA_DIR, TEST_FILE)
    if not os.path.exists(file_path):
        print(f"错误: 找不到文件 {file_path}，请先运行数据清洗脚本。")
        return
        
    df = pd.read_csv(file_path)
    closes = df['close'].values.reshape(-1, 1)
    trade_times = df['trade_time'].values
    
    # 严格的 80/20 划分
    train_size = int(len(closes) * 0.8)
    train_data = closes[:train_size]
    test_data = closes[train_size:]
    test_times = trade_times[train_size:] # 提取出测试集对应的时间戳
    
    # 归一化 (仅在训练集 fit)
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_data).flatten()
    test_scaled = scaler.transform(test_data).flatten()
    
    print(f"数据划分完成 -> 训练集: {len(train_scaled)} 点 | 测试集: {len(test_scaled)} 点")
    
    print("\n[1/2] 正在训练集上初始化 ARIMA 模型参数...")
    start_time = time.time()
    fit_train_data = train_scaled[-2000:] 
    model = ARIMA(fit_train_data, order=ARIMA_ORDER)
    fitted_model = model.fit()
    print(f"拟合完成，耗时: {time.time() - start_time:.2f} 秒")
    
    print("\n[2/2] 开始在测试集上进行多步预测滚动评估...")
    all_predictions = []
    all_trues = []
    
    history_array = list(fit_train_data)
    test_array = list(test_scaled)
    
    for i in range(0, len(test_array) - PRED_LEN + 1, EVAL_STRIDE):
        current_true_future = test_array[i : i + PRED_LEN]
        
        if i > 0:
            new_obs = test_array[i - EVAL_STRIDE : i]
            history_array.extend(new_obs)
            
        current_history = history_array[-SEQ_LEN:]
        
        temp_model = ARIMA(current_history, order=ARIMA_ORDER)
        with temp_model.fix_params(dict(zip(temp_model.param_names, fitted_model.params))):
            temp_res = temp_model.fit()
        
        forecast = temp_res.forecast(steps=PRED_LEN)
        
        all_predictions.append(forecast)
        all_trues.append(current_true_future)
        
        if (i // EVAL_STRIDE) % 5 == 0:
            print(f"  已评估测试集窗口: {i} / {len(test_array) - PRED_LEN}")

    # --- 评估与结果保存 ---
    pred_matrix = np.array(all_predictions)
    true_matrix = np.array(all_trues)
    
    # 反归一化
    preds_real = scaler.inverse_transform(pred_matrix.reshape(-1, 1)).flatten()
    trues_real = scaler.inverse_transform(true_matrix.reshape(-1, 1)).flatten()
    
    # 计算并保存指标
    metrics_result = calculate_metrics(trues_real, preds_real)
    print_metrics(metrics_result, model_name=model_id)
    save_metrics(metrics_result, model_name=model_id, save_dir=METRICS_SAVE_DIR)
    
    # 【核心修复：对齐真实交易时间戳】
    # 由于步长为 96，展平后的数组长度刚好等于无重叠预测点的总数
    aligned_times = test_times[:len(preds_real)]
    
    df_results = pd.DataFrame({
        'trade_time': aligned_times,
        'True_Price': trues_real,
        'Predicted_Price': preds_real
    })
    
    save_csv_path = os.path.join(PRED_SAVE_DIR, f"{model_id}_predictions.csv")
    df_results.to_csv(save_csv_path, index=False, encoding='utf-8-sig')
    print(f"✅ 预测曲线数据(带时间戳，共 {len(preds_real)} 行)已保存至: {save_csv_path}")

if __name__ == "__main__":
    evaluate_arima()