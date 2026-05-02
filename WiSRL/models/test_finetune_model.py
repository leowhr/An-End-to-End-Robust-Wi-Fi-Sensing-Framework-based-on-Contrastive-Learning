from modules.MambaModel_Bidirectional import MambaModel_Bidirectional
from modules.RMSNorm import RMSNorm
import torch
import torch.nn as nn

class Encoder(nn.Module):
    '''

    '''
    def __init__(
        self, 
        hidden_dim: int,
        bimamba_type,
        n_layers: int
    ):
        super().__init__()
        self.encoder = MambaModel_Bidirectional(
            hidden_dim=hidden_dim,# 模型内部主维度
            bimamba_type = bimamba_type,   
            n_layers=n_layers
        )
        
    
    def forward(self, x):
        x=self.encoder(x)

        return x


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        Args:
            d_model (int): 特征维度 D。
            max_len (int): 最大序列长度 L。
        """
        super().__init__()
        self.d_model = d_model

        # 生成位置编码矩阵
        position = torch.arange(max_len).unsqueeze(1)  # (L, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))  # (D/2,)
        pe = torch.zeros(max_len, d_model)  # (L, D)
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度: sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度: cos
        self.register_buffer('pe', pe)  # 注册为缓冲区，不参与训练

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入张量，形状为 [N, L, D]。
        Returns:
            torch.Tensor: 添加位置编码后的张量，形状为 [N, L, D]。
        """
        x = x + self.pe[:x.size(1)]  # 添加位置编码
        return x



class WiSRL_tune(nn.Module):
    '''

    '''
    def __init__(
        self,
        input_dim: int, # 输入维度
        hidden_dim: int, # encoder,decoder维度
        bimamba_type,
        output_dim: int,
        encoder_nlayers: int = 6, # encoder block数量 
    ):
        super().__init__()


        # **新的输入层**
        self.input_layer_amp = nn.Linear(input_dim, hidden_dim)  # 让输入适配 for amplitude
        self.input_layer_pha = nn.Linear(input_dim, hidden_dim)  # 让输入适配 for phase


        self.pos_embed = SinusoidalPositionalEncoding(hidden_dim) # 位置编码

        # 2个encoder
        self.encoder_amp = Encoder(hidden_dim=hidden_dim, bimamba_type=bimamba_type,n_layers=encoder_nlayers)
        self.encoder_pha = Encoder(hidden_dim=hidden_dim, bimamba_type=bimamba_type,n_layers=encoder_nlayers)  

        
        # 中间融合层
        self.middle_layer = nn.Linear(2*hidden_dim, hidden_dim)
        self.middle_norm = RMSNorm(dim=hidden_dim)



        # 2层 MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),  # 线性层，可根据需求调整
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)  # 适配最终任务
        )

        # # 2层 MLP
        # self.time_mapper = nn.Sequential(
        #     nn.Linear(600, 400),  # 线性层，可根据需求调整
        #     nn.ReLU()
        # )
        
        # self.feature_mapper = nn.Linear(hidden_dim, input_dim)  # 适配最终任务



         

    
    def forward(self, amplitude, phase):
        
        x_amp = self.input_layer_amp(amplitude)
        x_pha = self.input_layer_pha(phase)

        
        x_amp = self.pos_embed(x_amp)
        x_pha = self.pos_embed(x_pha)


        x_amp = self.encoder_amp(x_amp)  
        x_pha = self.encoder_pha(x_pha)


        # 中间层聚合
        x_concat = torch.concat([x_amp, x_pha], dim=2)
        x_concat = self.middle_layer(x_concat)
        x_concat = self.middle_norm(x_concat)

        
        # MLP层输出
        Temp_logits = self.mlp(x_concat)  # 这里输出的是 logits
        #
        x = torch.mean(Temp_logits, dim=1)  # (B, L, D) -> (B, D)


        # x_concat = x_concat.permute(0, 2, 1)         # (B, 64, 800)
        # x_concat = self.time_mapper(x_concat)       # (B, 64, 200)
        # x_concat = x_concat.permute(0, 2, 1)        # (B, 200, 64)
        # x = self.feature_mapper(x_concat)    # (B, 200, 540)


        return x  