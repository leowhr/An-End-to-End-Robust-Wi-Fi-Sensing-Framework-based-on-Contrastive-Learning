import torch.nn as nn
from modules.MambaBlock import MambaBlock
from modules.RMSNorm import RMSNorm
from modules.mambassm import Mamba


class Residual_MambaBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        # inner_dim: int,
        state_dim: int = 16,
        expand: int = 2 ,
        conv_kernel: int = 4,
        # dt_rank: int | str = 'auto',
        # conv_bias: bool = True,
        # use_bias: bool = False
    ):
        super(Residual_MambaBlock, self).__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.expand = expand
        self.conv_kernel = conv_kernel

        self.norm = RMSNorm(self.hidden_dim)  # 从 RMSNorm.py 导入
        self.mixer = Mamba(
            d_model=self.hidden_dim,
            d_state=self.state_dim,
            expand=self.expand,
            d_conv=self.conv_kernel
        )
        # 加载初始化模型
        mamba_block = MambaBlock(hidden_dim=self.hidden_dim)
        self.mixer.load_state_dict (mamba_block.state_dict())

    def forward(self, x):
        return self.mixer(self.norm(x)) + x