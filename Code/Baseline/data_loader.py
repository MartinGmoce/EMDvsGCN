import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

class UnivariateStockDataset(Dataset):
    def __init__(self, data, seq_len=512, pred_len=96):
        """
        data: 一维的 numpy array，经过归一化处理的收盘价序列
        """
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        # 减去历史窗口和预测窗口，步长 stride=1 隐含在索引+1中
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        # 输入 X: 长度为 seq_len
        s_begin = index
        s_end = s_begin + self.seq_len
        # 输出 Y: 紧随其后的 pred_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]

        # PyTorch 要求的特征维度: (序列长度, 特征数) -> (512, 1)
        return torch.tensor(seq_x, dtype=torch.float32).unsqueeze(-1), \
               torch.tensor(seq_y, dtype=torch.float32)

def get_dataloaders(csv_path, seq_len=512, pred_len=96, batch_size=64):
    """
    读取真实价格数据，按 80% 拆分，严格仅在训练集上 fit_transform，
    返回 DataLoader 和 scaler（用于后续的反归一化）。
    """
    df = pd.read_csv(csv_path)
    # 提取真实收盘价
    closes = df['close'].values.reshape(-1, 1)

    # 划分 80% / 20%
    train_size = int(len(closes) * 0.8)
    train_data_raw = closes[:train_size]
    test_data_raw = closes[train_size:]

    # 严格防泄露：仅用训练集计算 Min-Max 范围 [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_data_raw).flatten()
    test_scaled = scaler.transform(test_data_raw).flatten()

    # 构建 Dataset
    train_dataset = UnivariateStockDataset(train_scaled, seq_len, pred_len)
    test_dataset = UnivariateStockDataset(test_scaled, seq_len, pred_len)

    # 构建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, scaler