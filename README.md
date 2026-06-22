# EMDvsGCN

面向高频金融时间序列预测的毕业论文实验项目。项目比较传统统计模型、循环神经网络、Informer、CEEMDAN 分解模型，以及引入股票图结构的 ST-Trader / GCN-Informer 方法。

当前仓库已经包含清洗后的数据、已生成的预测结果、指标表和论文图表，因此可以先复现实验分析结果，再按需重新训练模型。

## 项目结构

```text
Code/
  Analysis/      # 指标汇总、相位延迟分析、预测曲线绘图
  Baseline/      # ARIMA、RNN、GRU、LSTM、Informer
  CEEMDAN/       # CEEMDAN 分解、分解图、CEEMDAN-LSTM
  GCN/           # VAE 建图、基准图、ST-Trader、GCN-Informer
  Preprocess/    # 原始数据抓取、清洗、GCN 节点数据准备
  Utils/         # 通用指标
Data/
  Raw/                 # 原始分钟线数据
  Processed/           # 清洗对齐后的单标的数据
  CEEMDAN_Decomposed/  # CEEMDAN 分解结果
  GCN/                 # 32 个图节点股票数据
Results/
  Predictions/         # 各模型预测结果
  Metrics/             # 单模型指标
  AnalysisResults/     # 汇总指标和相位延迟指标
  ForecastPlots/       # 全量预测曲线图
  PredictionsPlot2000/ # 前 2000 步局部预测图
scripts/
  smoke_test.py        # 快速健康检查，不训练
  run_pipeline.py      # 分阶段运行入口
```

## 环境安装

建议使用 Python 3.9 或更高版本。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

说明：

- `torch` 用于所有深度学习模型。若你要在 CUDA 服务器上训练，建议按 PyTorch 官方命令安装对应 CUDA 版本。
- `EMD-signal` 是提供 `PyEMD` 导入路径的包，用于 CEEMDAN。
- `tushare` 只在重新抓取原始行情时需要；仓库已有离线数据，复现实验分析不依赖它。

## 快速验证

```bash
python scripts/smoke_test.py
```

这个检查不会训练模型，只会验证目录、核心依赖、已有数据和预测 CSV 的字段。默认环境缺少 `torch` 或 `PyEMD` 时会给出 warning；训练模型前需要安装完整依赖。

## 复现已有结果

基于仓库中已有的 `Results/Predictions` 重新生成汇总指标和图表：

```bash
python scripts/run_pipeline.py reproduce
```

等价于依次运行：

```bash
python Code/Analysis/calculate_total_metrics.py
python Code/Analysis/calculate_phase_delay.py
python Code/Analysis/plot_predictions.py
python Code/Analysis/plot_predictions_2000.py
python Code/CEEMDAN/plot_decomposition.py
```

## 重新运行实验

常用阶段：

```bash
python scripts/run_pipeline.py clean      # 清洗 Raw 数据
python scripts/run_pipeline.py graph      # 生成 VAE 特征、VAE 图和消融基准图
python scripts/run_pipeline.py baselines  # ARIMA / RNN / GRU / LSTM / Informer
python scripts/run_pipeline.py ceemdan    # CEEMDAN 分解和 CEEMDAN-LSTM
python scripts/run_pipeline.py gcn        # ST-Trader 和 GCN-Informer
python scripts/run_pipeline.py analysis   # 汇总指标
python scripts/run_pipeline.py plots      # 生成论文图
```

完整重跑：

```bash
python scripts/run_pipeline.py full
```

完整训练会比较耗时，尤其是 Informer、CEEMDAN 和 GCN 分支。答辩或检查环境时建议先运行 `smoke_test.py` 和 `run_pipeline.py reproduce`。

## 数据抓取与本地源数据

重新抓取 Tushare 数据前，先设置环境变量：

```bash
export TUSHARE_TOKEN="你的 token"
python Code/Preprocess/fetch_raw_data.py
```

`Code/Preprocess/prepare_gcn_data.py` 用于从月份文件夹构建 `Data/GCN`。如果原始月份 CSV 不在仓库内，可以通过环境变量指定：

```bash
export GCN_SOURCE_DIR="/path/to/source-data"
python Code/Preprocess/prepare_gcn_data.py
```

也可以参考 `.env.example` 管理这些本地配置。

## 主要实验输出

- 单模型预测：`Results/Predictions/*_predictions.csv`
- 单模型指标：`Results/Metrics/*_metrics.csv`
- 汇总指标：`Results/AnalysisResults/total_metrics.csv`
- 相位延迟：`Results/AnalysisResults/phase_delay_metrics.csv`
- 全量预测图：`Results/ForecastPlots/*.png`
- 前 2000 步预测图：`Results/PredictionsPlot2000/*.png`
- 图结构矩阵和热力图：`Results/VAE_Features/`

## 工程化改动说明

项目路径统一集中在 `Code/project_config.py`，各脚本会从自身位置自动定位项目根目录，不再依赖固定的 `/Users/...` 绝对路径。绘图脚本默认使用无界面 Matplotlib 后端，因此可以在服务器、终端和 CI 环境中运行。
