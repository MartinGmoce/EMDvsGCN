import torch
import torch.nn as nn

class RNNFamilyModel(nn.Module):
    def __init__(self, model_type='LSTM', input_size=1, hidden_size=64, num_layers=2, pred_len=96):
        super(RNNFamilyModel, self).__init__()
        self.model_type = model_type.upper()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 根据字符串选择网络内核
        if self.model_type == 'RNN':
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        elif self.model_type == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        elif self.model_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        else:
            raise ValueError("model_type 必须是 RNN, GRU 或 LSTM")
            
        # 全连接映射层：把最后一个时间步的隐状态转换为 H 步的预测结果
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size) -> (64, 512, 1)
        
        if self.model_type == 'LSTM':
            out, (h_n, c_n) = self.rnn(x)
        else:
            out, h_n = self.rnn(x)
            
        # 我们只取序列最后一个时间步的输出特征去预测未来
        # last_out shape: (batch_size, hidden_size)
        last_out = out[:, -1, :]
        
        # predictions shape: (batch_size, pred_len) -> (64, 96)
        predictions = self.fc(last_out)
        
        return predictions