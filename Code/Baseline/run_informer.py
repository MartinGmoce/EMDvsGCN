import os
import sys
import torch
import numpy as np
import pandas as pd

# 动态路径挂载
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR)) 
sys.path.append(os.path.join(PROJECT_ROOT, "Code"))
sys.path.append(os.path.join(CURRENT_DIR, "Informer"))

from exp.exp_informer import Exp_Informer
from Utils.metrics import calculate_metrics, print_metrics, save_metrics

# ================= 配置区域 =================
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "Data", "Processed")
PRED_SAVE_DIR = os.path.join(PROJECT_ROOT, "Results", "Predictions")
METRICS_SAVE_DIR = os.path.join(PROJECT_ROOT, "Results", "Metrics")

os.makedirs(PRED_SAVE_DIR, exist_ok=True)
os.makedirs(METRICS_SAVE_DIR, exist_ok=True)

# 【核心修改】：将单一文件改为文件列表
TEST_FILES = [
    "Cleaned_STK_000001.SZ_平安银行_1min.csv",
    "Cleaned_STK_600519.SH_贵州茅台_1min.csv",
    "Cleaned_IDX_399001.SZ_深证成指_1min.csv",
    "Cleaned_IDX_000001.SH_上证指数_1min.csv"
]

# ================= Informer 满血版参数 =================
class InformerArgs:
    def __init__(self, data_file):
        self.model = 'informer'
        self.data = 'custom'
        self.root_path = PROCESSED_DATA_DIR
        self.data_path = data_file # 动态传入文件名
        self.features = 'S'      
        self.target = 'close'    
        self.freq = 't'          
        self.checkpoints = os.path.join(PROJECT_ROOT, "Results", "Checkpoints", "Informer")
        
        self.seq_len = 512       
        self.label_len = 256     
        self.pred_len = 96       
        
        # === 核心网络结构火力全开 ===
        self.enc_in = 1          
        self.dec_in = 1          
        self.c_out = 1           
        self.d_model = 512       
        self.n_heads = 8         
        self.e_layers = 2
        self.d_layers = 1
        self.s_layers = '3,2,1'
        self.d_ff = 2048         
        self.factor = 5          
        self.padding = 0
        self.distil = True       
        self.dropout = 0.05
        self.attn = 'prob'       
        self.embed = 'timeF'
        self.activation = 'gelu'
        self.output_attention = False
        self.do_predict = False
        
        self.mix = True         
        self.cols = None        
        
        # === 服务器性能释放 ===
        self.num_workers = 4     
        self.itr = 1             
        self.train_epochs = 15   
        self.batch_size = 32     
        self.patience = 3        
        self.learning_rate = 0.0001
        self.des = 'Baseline'
        self.loss = 'mse'
        self.lradj = 'type1'
        self.use_amp = False     
        self.inverse = True      
        
        self.use_gpu = True if torch.cuda.is_available() else False
        self.gpu = 0
        self.use_multi_gpu = False
        self.devices = '0'

# ================= 单个数据集的执行逻辑 =================
def run_informer_for_single_stock(test_file):
    stock_id = test_file.replace("Cleaned_", "").replace("_1min.csv", "")
    model_id = f"Informer_{stock_id}"

    original_file_path = os.path.join(PROCESSED_DATA_DIR, test_file)
    informer_file_name = f"InformerCompat_{test_file}"
    informer_file_path = os.path.join(PROCESSED_DATA_DIR, informer_file_name)
    
    if not os.path.exists(original_file_path):
        print(f"❌ 错误：未找到数据文件 {original_file_path}，跳过此文件。")
        return

    df_compat = pd.read_csv(original_file_path)
    if 'trade_time' in df_compat.columns:
        df_compat.rename(columns={'trade_time': 'date'}, inplace=True)
    df_compat.to_csv(informer_file_path, index=False)

    print(f"\n{'='*60}")
    print(f"🚀 [任务启动] 服务器满血版大模型: {model_id}")
    print(f"处理数据: {informer_file_name}")
    print(f"{'='*60}")

    args = InformerArgs(data_file=informer_file_name)
    
    if args.use_gpu:
        print(f"💡 硬件检测: 成功调用 NVIDIA GPU 加速 (Device: {torch.cuda.get_device_name(0)})")
    else:
        print("⚠️ 硬件检测: 未检测到 GPU，将使用 CPU 运行 (请检查服务器 CUDA 环境)")

    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_{}_{}'.format(
        args.model, args.data_path.split('.')[0], args.data, args.features, 
        args.seq_len, args.label_len, args.pred_len, args.des, 0)

    exp = Exp_Informer(args)

    print("\n>>> 开始训练 Informer >>>")
    exp.train(setting)

    print("\n>>> 开始测试 Informer (将自动进行反归一化) >>>")
    exp.test(setting)

    result_dir = os.path.join(CURRENT_DIR, "results", setting)
    preds_npy_path = os.path.join(result_dir, 'pred.npy')
    trues_npy_path = os.path.join(result_dir, 'true.npy')
    
    if not os.path.exists(preds_npy_path) or not os.path.exists(trues_npy_path):
        print(f"❌ 错误：未找到 {model_id} 的预测结果文件。")
        return

    preds = np.load(preds_npy_path).squeeze()
    trues = np.load(trues_npy_path).squeeze()
    
    preds_flat = preds.flatten()
    trues_flat = trues.flatten()

    metrics_result = calculate_metrics(trues_flat, preds_flat)
    print_metrics(metrics_result, model_name=model_id)
    save_metrics(metrics_result, model_name=model_id, save_dir=METRICS_SAVE_DIR)
    
    preds_1d = preds[:, 0].flatten()
    trues_1d = trues[:, 0].flatten()
    
    df_raw = pd.read_csv(original_file_path)
    num_test = int(len(df_raw) * 0.2)
    test_times = df_raw['trade_time'].values[-num_test:]
    
    num_samples = len(preds_1d)
    times_1d = test_times[:num_samples]

    df_results = pd.DataFrame({
        'trade_time': times_1d,
        'True_Price': trues_1d,
        'Predicted_Price': preds_1d
    })
    
    save_csv_path = os.path.join(PRED_SAVE_DIR, f"{model_id}_predictions.csv")
    df_results.to_csv(save_csv_path, index=False, encoding='utf-8-sig')
    print(f"✅ [{model_id}] 预测曲线 (共 {num_samples} 行) 已成功归档至: {save_csv_path}")
    
    # 清理显存，防止下一个任务 Out of Memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ================= 批处理调度入口 =================
if __name__ == "__main__":
    print("🌟 启动 Informer 批量处理流水线...")
    for idx, target_file in enumerate(TEST_FILES):
        print(f"\n[{idx+1}/{len(TEST_FILES)}] 正在准备执行任务: {target_file}")
        try:
            run_informer_for_single_stock(target_file)
        except Exception as e:
            print(f"❌ 严重错误：在处理 {target_file} 时程序崩溃！")
            print(f"错误详情: {e}")
            print("➡️ 继续执行下一个任务...")
            
    print("\n🎉 全部 4 个数据集的 Informer 训练与评估任务已圆满结束！去睡个好觉吧！") 