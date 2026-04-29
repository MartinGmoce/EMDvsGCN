import torch
import torch.nn as nn
from st_trader_model import compute_chebyshev_polynomials

class GCN_Informer(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim, K, T_k_tensors, n_heads=8):
        super(GCN_Informer, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.K = K
        self.T_k = nn.ParameterList([nn.Parameter(t, requires_grad=False) for t in T_k_tensors])
        
        # 1. 空间域特征提取 (GCN 卷积核)
        self.gcn_weight = nn.Parameter(torch.Tensor(K + 1, input_dim, hidden_dim))
        nn.init.xavier_uniform_(self.gcn_weight)
        
        # 2. 时序域特征提取 (Informer 的注意力核心机制平替: Transformer Encoder)
        # 将图中所有节点的隐藏特征展平，作为自注意力的输入维度
        transformer_dim = num_nodes * hidden_dim
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=transformer_dim, 
            nhead=n_heads, 
            dim_feedforward=2048, 
            dropout=0.1, 
            batch_first=True
        )
        self.informer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=2)
        
        # 3. 输出层 (将高维注意力特征映射回我们要预测的 32 个节点收盘价)
        self.fc = nn.Linear(transformer_dim, num_nodes)

    def forward(self, x):
        # x 形状: (Batch, Time_Steps, Nodes, Input_Dim)
        batch_size, seq_len, num_nodes, input_dim = x.shape
        
        # --- 步骤 1: 逐时间步应用 GCN 提取空间拓扑信息 ---
        gcn_outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :, :] # (Batch, Nodes, Input_Dim)
            
            # 频域切比雪夫图卷积 \sum_{k=0}^K T_k(L) * x_t * W_k
            conv_t = torch.zeros(batch_size, num_nodes, self.hidden_dim).to(x.device)
            for k in range(self.K + 1):
                node_aggregated = torch.einsum('nm, bmf -> bnf', self.T_k[k], x_t)
                conv_t += torch.matmul(node_aggregated, self.gcn_weight[k])
                
            # 采用 ReLU 激活并展平节点特征
            conv_t = torch.relu(conv_t) # (Batch, Nodes, Hidden_Dim)
            flattened = conv_t.reshape(batch_size, -1) # (Batch, Nodes * Hidden_Dim)
            gcn_outputs.append(flattened.unsqueeze(1))
            
        # 将所有时间步拼起来: (Batch, Time_Steps, Transformer_Dim)
        st_features = torch.cat(gcn_outputs, dim=1)
        
        # --- 步骤 2: 将时空特征喂入 Informer 注意力编码器 ---
        attn_out = self.informer_encoder(st_features)
        
        # 取最后一个时间步的注意力聚合结果
        last_step_out = attn_out[:, -1, :] 
        
        # --- 步骤 3: 映射到目标 ---
        predictions = self.fc(last_step_out) # (Batch, Nodes)
        return predictions
    
    
## 如何运行：
# 1. 在 train_st_trader.py 中导入 GCN_Informer 模型
# 2. 替换原有的 ST_Trader 模型实例化为 GCN_Informer，并调整相应的输入输出维度
#    把模型初始化改成：
# model = GCN_Informer(num_nodes=num_nodes, 
#                      input_dim=len(FEATURES), 
#                      hidden_dim=HIDDEN_DIM, 
#                      K=K_ORDER, 
#                      T_k_tensors=T_k_tensors,
#                      n_heads=8).to(device)
# 3. 直接运行 train_st_trader.py 即可在 GPU 上训练 GCN-Informer 模型，并在测试