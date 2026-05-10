import os
import sys
import numpy as np
import pandas as pd
import torch

# 动态挂载路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR)) 
sys.path.append(os.path.join(PROJECT_ROOT, "Code"))
sys.path.append(os.path.join(PROJECT_ROOT, "Code", "Baseline", "Informer"))

from exp.exp_informer import Exp_Informer
from Utils.metrics import calculate_metrics, print_metrics, save_metrics

# ================= 配置区域 =================
CEEMDAN_DATA_DIR = os.path.join(PROJECT_ROOT, "Data", "CEEMDAN_Decomposed")
PRED_SAVE_DIR = os.path.join(PROJECT_ROOT, "Results", "Predictions")
METRICS_SAVE_DIR = os.path.join(PROJECT_ROOT, "Results", "Metrics")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "Results", "Checkpoints", "CEEMDAN_Informer")

os.makedirs(PRED_SAVE_DIR, exist_ok=True)
os.makedirs(METRICS_SAVE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 批量处理的文件列表
TEST_FILES = [
    "CEEMDAN_STK_000001.SZ_平安银行_1min.csv",
    "CEEMDAN_STK_600519.SH_贵州茅台_1min.csv",
    "CEEMDAN_IDX_399001.SZ_深证成指_1min.csv",
    "CEEMDAN_IDX_000001.SH_上证指数_1min.csv"
]

class InformerArgs:
    def __init__(self, target_col, data_path):
        self.model = 'informer'
        self.data = 'custom'
        self.root_path = CEEMDAN_DATA_DIR
        self.data_path = data_path
        self.features = 'S'      
        self.target = target_col 
        self.freq = 't'          
        self.checkpoints = CHECKPOINT_DIR
        
        # === 单步预测参数对齐 ===
        self.seq_len = 60       
        self.label_len = 30     
        self.pred_len = 1       
        
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
        
        self.num_workers = 4     
        self.itr = 1
        self.train_epochs = 15   
        self.batch_size = 32     
        self.patience = 3
        self.learning_rate = 0.0001
        self.des = 'CEEMDAN_Enhance'
        self.loss = 'mse'
        self.lradj = 'type1'
        self.use_amp = False
        self.inverse = True 
        
        self.use_gpu = True if torch.cuda.is_available() else False
        self.gpu = 0
        self.use_multi_gpu = False
        self.devices = '0'

def run_ceemdan_informer_for_single_stock(test_file):
    stock_id = test_file.replace("CEEMDAN_", "").replace("_1min.csv", "")
    model_id = f"StockCI_{stock_id}"

    original_file_path = os.path.join(CEEMDAN_DATA_DIR, test_file)
    if not os.path.exists(original_file_path):
        print(f"❌ 错误: 找不到文件 {original_file_path}")
        return

    informer_file_name = f"Compat_{test_file}"
    informer_file_path = os.path.join(CEEMDAN_DATA_DIR, informer_file_name)
    df_compat = pd.read_csv(original_file_path)
    if 'trade_time' in df_compat.columns:
        df_compat.rename(columns={'trade_time': 'date'}, inplace=True)
    df_compat.to_csv(informer_file_path, index=False)

    components = [col for col in df_compat.columns if 'IMF' in col or 'Residue' in col]
    
    print(f"\n{'='*60}")
    print(f"🚀 [任务启动] CEEMDAN-Informer (StockCI) 联合预测: {model_id}")
    print(f"共检测到 {len(components)} 个待训练分量: {components}")
    print(f"{'='*60}")

    final_sum_preds_1d = None
    final_sum_trues_1d = None

    for comp in components:
        print(f"\n>>> 正在用 GPU 处理分量: {comp} >>>")
        
        args = InformerArgs(target_col=comp, data_path=informer_file_name)
        setting = f'{args.model}_{informer_file_name.split(".")[0]}_{comp}_sl{args.seq_len}_pl{args.pred_len}'
        
        exp = Exp_Informer(args)
        exp.train(setting)
        exp.test(setting)
        
        result_dir = os.path.join(CURRENT_DIR, "results", setting)
        preds_npy_path = os.path.join(result_dir, 'pred.npy')
        trues_npy_path = os.path.join(result_dir, 'true.npy')
        
        if not os.path.exists(preds_npy_path) or not os.path.exists(trues_npy_path):
             print(f"⚠️ 警告: 分量 {comp} 预测结果缺失。")
             continue
             
        # 【核心修复】：由于 pred_len=1，直接读取并展平为 1D 数组
        pred_arr = np.load(preds_npy_path).flatten()
        true_arr = np.load(trues_npy_path).flatten()
        
        if final_sum_preds_1d is None:
            final_sum_preds_1d = np.zeros_like(pred_arr)
            final_sum_trues_1d = np.zeros_like(true_arr)
            
        final_sum_preds_1d += pred_arr
        final_sum_trues_1d += true_arr

    print("\n" + "="*60)
    print(f"✅ [{model_id}] 重构完毕！正在计算误差...")
    
    metrics_result = calculate_metrics(final_sum_trues_1d, final_sum_preds_1d)
    print_metrics(metrics_result, model_name=model_id)
    save_metrics(metrics_result, model_name=model_id, save_dir=METRICS_SAVE_DIR)
    
    num_samples = len(final_sum_preds_1d)
    df_raw = pd.read_csv(original_file_path)
    test_times = df_raw['trade_time'].values[-int(len(df_raw)*0.2):]
    times_1d = test_times[:num_samples]

    df_results = pd.DataFrame({
        'trade_time': times_1d,
        'True_Price': final_sum_trues_1d,
        'Predicted_Price': final_sum_preds_1d
    })
    
    save_csv_path = os.path.join(PRED_SAVE_DIR, f"{model_id}_predictions.csv")
    df_results.to_csv(save_csv_path, index=False, encoding='utf-8-sig')
    print(f"✅ 预测曲线 (共 {num_samples} 行) 已保存至: {save_csv_path}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    print("🌟 启动 StockCI 流水线...")
    for target_file in TEST_FILES:
        try:
            run_ceemdan_informer_for_single_stock(target_file)
        except Exception as e:
            print(f"❌ 严重错误: {e}")