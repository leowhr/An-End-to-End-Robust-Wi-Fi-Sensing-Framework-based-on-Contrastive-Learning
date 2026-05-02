import torch
import torch.nn as nn
from modules.RMSNorm import RMSNorm



class WiSRL_tune(nn.Module):
    '''

    '''
    def __init__(
        self, 
        original_model, 
        hidden_dim: int,
        output_dim: int,
    ):
        super().__init__()


        # # **新的输入层**
        self.proj_layer = nn.Linear(90, 270)  # 让输入适配 for amplitude



        ##  **输入层**
        self.input_layer_amp = original_model.patch_embed_amp  # 让输入适配 for amplitude


        # position_embedding
        self.pos_embed = original_model.pos_embed # 位置编码

        # 2个 encoder **冻结 encoder**
        self.encoder_amp = original_model.encoder_amp  



        # 2层 MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),  # 线性层，可根据需求调整
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)  # 适配最终任务
        )

    
    def forward(self, amplitude):

        x_amp = self.proj_layer(amplitude)
        x_amp = self.input_layer_amp(x_amp)

        # x_amp = self.input_layer_amp(amplitude)


        x_amp = self.pos_embed(x_amp)


        x_amp = self.encoder_amp(x_amp)  



        # MLP层输出
        logits = self.mlp(x_amp)  # 这里输出的是 logits
        logits = torch.mean(logits, dim=1)  # (B, L, D) -> (B, D)
        return logits  