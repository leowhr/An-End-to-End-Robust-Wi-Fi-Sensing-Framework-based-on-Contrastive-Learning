# import math
import torch.nn as nn
# from modules.Residual_MambaBlock import Residual_MambaBlock
from modules.mambassm_bidirectional import Block_Bidirectional
# from modules.RMSNorm import RMSNorm

class MambaModel_Bidirectional(nn.Module):
    def __init__(
        self,
        # input_dim: int,         # 原始输入维度
        # output_dim: int,        # 最终输出维度
        hidden_dim: int,
        bimamba_type,        # 模型内部维度（原d_model）
        n_layers: int,
        state_dim: int = 16,
        expand: int = 2 ,
        conv_kernel: int = 4,
        # dt_rank: int | str = 'auto',
        # conv_bias: bool = True,
        # use_bias: bool = False
    ):
        super(MambaModel_Bidirectional, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.state_dim = state_dim
        self.expand = expand
        self.conv_kernel = conv_kernel

        # 构建网络
        # self.layers = nn.ModuleList([
        #     Residual_MambaBlock(
        #         hidden_dim=self.hidden_dim,
        #         # inner_dim=inner_dim,
        #         state_dim=self.state_dim,
        #         expand = self.expand,
        #         # dt_rank=dt_rank,
        #         conv_kernel=self.conv_kernel,
        #         # conv_bias=conv_bias,
        #         # use_bias=use_bias
        #     ) for _ in range(self.n_layers)
        # ])
        # self.norm = RMSNorm(self.hidden_dim)
        self.layers = nn.ModuleList([
            Block_Bidirectional(dim = hidden_dim,bimamba_type=bimamba_type)
            for _ in range(self.n_layers)
        ])

    def forward(self, x, res=None):
        """输入形状: (batch, seq_len, input_dim) -> (batch, seq_len, output_dim)"""
        # for layer in self.layers:
        #     x = layer(x)
        # x = self.norm(x)
        for layer in self.layers:
            x,res = layer(x,res)

        return x