import torch
import torch.nn as nn
import numpy as np

# ================= 计算切比雪夫多项式 =================
def compute_chebyshev_polynomials(adj_matrix, K):
    """
    根据邻接矩阵计算 Scaled Laplacian 和切比雪夫多项式 T_k
    对应论文公式 (13) 和 (14)
    """
    n_nodes = adj_matrix.shape[0]
    
    # 1. 计算度矩阵 D 和拉普拉斯矩阵 L
    degree = np.sum(adj_matrix, axis=1)
    # 防止除以 0
    degree_inv_sqrt = np.power(degree, -0.5, where=(degree != 0))
    degree_inv_sqrt[degree == 0] = 0.0
    D_inv_sqrt = np.diag(degree_inv_sqrt)
    
    L = np.eye(n_nodes) - D_inv_sqrt @ adj_matrix @ D_inv_sqrt
    
    # 2. 计算最大特征值进行缩放
    eigenvalues = np.linalg.eigvals(L)
    lambda_max = np.max(eigenvalues.real)
    
    # 3. Scaled Laplacian
    L_scaled = (2.0 / lambda_max) * L - np.eye(n_nodes)
    
    # 4. 计算切比雪夫多项式 T_0, T_1, ..., T_K
    T_k = []
    T_k.append(np.eye(n_nodes))                 # T_0(x) = 1
    if K >= 1:
        T_k.append(L_scaled)                    # T_1(x) = x
    for k in range(2, K + 1):
        # T_k(x) = 2x * T_{k-1}(x) - T_{k-2}(x)
        T_next = 2 * L_scaled @ T_k[k-1] - T_k[k-2]
        T_k.append(T_next)
        
    # 转为 PyTorch Tensor
    T_k_tensors = [torch.FloatTensor(t) for t in T_k]
    return T_k_tensors

# ================= GCN-LSTM 核心单元 =================
class GCN_LSTM_Cell(nn.Module):
    def __init__(self, input_dim, hidden_dim, K, T_k_tensors):
        super(GCN_LSTM_Cell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.K = K
        # 预先计算好的切比雪夫多项式列表 [T_0, T_1, ..., T_K]
        self.T_k = nn.ParameterList([nn.Parameter(t, requires_grad=False) for t in T_k_tensors])
        
        # 将 LSTM 的 4 个门 (i, f, o, c) 压缩到一个权重矩阵中以加速计算
        # 对应论文公式 (15)-(19) 中的 W_{*, x} 和 W_{*, h}
        self.weight = nn.Parameter(torch.Tensor(K + 1, input_dim + hidden_dim, 4 * hidden_dim))
        self.bias = nn.Parameter(torch.Tensor(4 * hidden_dim))
        
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, hidden_state):
        h, c = hidden_state
        # x 形状: (Batch, Nodes, Input_Dim)
        # h 形状: (Batch, Nodes, Hidden_Dim)
        
        # 将当前输入与上一时刻的隐状态拼接
        z = torch.cat([x, h], dim=-1)  # (Batch, Nodes, Input+Hidden_Dim)
        
        # 图卷积操作: \sum_{k=0}^K T_k(L) * z * W_k
        conv_z = torch.zeros(z.shape[0], z.shape[1], 4 * self.hidden_dim).to(x.device)
        for k in range(self.K + 1):
            # T_k[k] @ z: 聚合邻居信息 (空间信息)
            node_aggregated = torch.einsum('nm, bmf -> bnf', self.T_k[k], z)
            # @ weight[k]: 特征变换
            conv_z += torch.matmul(node_aggregated, self.weight[k])
            
        conv_z += self.bias
        
        # 将卷积结果均分为 4 个门
        i_gate, f_gate, o_gate, c_gate = torch.chunk(conv_z, 4, dim=-1)
        
        # LSTM 的标准激活
        i = torch.sigmoid(i_gate)
        f = torch.sigmoid(f_gate)
        o = torch.sigmoid(o_gate)
        c_new = f * c + i * torch.tanh(c_gate)
        h_new = o * torch.tanh(c_new)
        
        return h_new, c_new

# ================= ST-Trader 模型 =================
class ST_Trader(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim, K, T_k_tensors, num_layers=1):
        super(ST_Trader, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.cells = nn.ModuleList([
            GCN_LSTM_Cell(input_dim if i == 0 else hidden_dim, hidden_dim, K, T_k_tensors)
            for i in range(num_layers)
        ])
        
        # 最终预测层：将隐藏层状态映射为预测目标 (例如：下一时刻的收盘价)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x 形状: (Batch, Time_Steps, Nodes, Input_Dim)
        batch_size, seq_len, num_nodes, _ = x.shape
        
        # 初始化隐状态 h 和 c
        h = [torch.zeros(batch_size, num_nodes, self.hidden_dim).to(x.device) for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, num_nodes, self.hidden_dim).to(x.device) for _ in range(self.num_layers)]
        
        # 沿时间步展开 (时间信息)
        for t in range(seq_len):
            x_t = x[:, t, :, :]
            for l in range(self.num_layers):
                h[l], c[l] = self.cells[l](x_t, (h[l], c[l]))
                x_t = h[l]  # 上一层的输出作为下一层的输入
                
        # 取最后一个时间步的隐状态进行预测
        out = self.fc(h[-1]) # (Batch, Nodes, 1)
        return out.squeeze(-1) # (Batch, Nodes)