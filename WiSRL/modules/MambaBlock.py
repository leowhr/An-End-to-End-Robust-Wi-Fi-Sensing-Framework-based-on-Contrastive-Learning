import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, einsum

class MambaBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,        # 模型内部主维度
        # inner_dim: int,         # 内部扩展维度
        state_dim: int = 16,
        expand: int = 2 ,
        conv_kernel: int = 4,
        dt_rank = 'auto',  # int | str
        conv_bias: bool = True,
        use_bias: bool = False,
    ):
        super(MambaBlock, self).__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.expand = expand
        self.conv_kernel = conv_kernel
        
        # 计算内部维度
        self.inner_dim = int(self.expand * self.hidden_dim)
        self.dt_rank = math.ceil(self.hidden_dim / 16) if dt_rank == "auto" else dt_rank

        self.conv_bias = conv_bias
        self.use_bias = use_bias

        # self.inner_dim = inner_dim
        
        # 输入投影（hidden_dim -> inner_dim*2）
        self.in_proj = nn.Linear(self.hidden_dim, self.inner_dim*2, bias=self.use_bias)
        
        # 因果卷积层参数
        self.conv1d = nn.Conv1d(
            in_channels=self.inner_dim,   # 输入
            out_channels=self.inner_dim,  # 输出
            kernel_size=self.conv_kernel,
            groups=self.inner_dim,
            padding=self.conv_kernel-1,   # 因果关系
            bias=self.conv_bias
        )
        
        # 其他参数保持原样
        self.x_proj = nn.Linear(self.inner_dim, self.dt_rank + self.state_dim*2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.inner_dim, bias=True)
        
        A = repeat(torch.arange(1, self.state_dim+1), 'n -> d n', d=self.inner_dim)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.inner_dim))
        self.out_proj = nn.Linear(self.inner_dim, self.hidden_dim, bias=self.use_bias)

        # self.dropout = nn.Dropout(0.2) # 加入 dropout

    def forward(self, x):
        b, l, d = x.shape
        x_and_res = self.in_proj(x)
        x, res = x_and_res.split([self.inner_dim, self.inner_dim], dim=-1)
        
        # 卷积操作
        x = rearrange(x, 'b l d -> b d l')
        x = self.conv1d(x)[:, :, :l]  # 确保输出长度与输入一致
        x = rearrange(x, 'b d l -> b l d')
        
        # SSM处理
        x = F.silu(x)
        y = self.ssm(x) * F.silu(res)
        y = self.out_proj(y)
        # y = self.dropout(y)
        return y
    

    def ssm(self, x):
        A = -torch.exp(self.A_log.float())
        D = self.D.float()
        x_dbl = self.x_proj(x)
        delta, B, C = x_dbl.split([self.dt_rank, self.state_dim, self.state_dim], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        return self.selective_scan(x, delta, A, B, C, D)

    def selective_scan(self, u, delta, A, B, C, D):
        b, l, d = u.shape
        n = A.shape[1]
        deltaA = torch.exp(einsum(delta, A, 'b l d, d n -> b l d n'))
        deltaB_u = einsum(delta, B, u, 'b l d, b l n, b l d -> b l d n')
        x = torch.zeros((b, self.inner_dim, n), device=deltaA.device)
        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            ys.append(einsum(x, C[:, i, :], 'b d n, b n -> b d'))
        y = torch.stack(ys, dim=1) + u * D
        return y