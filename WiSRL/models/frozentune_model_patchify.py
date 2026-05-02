import torch
import torch.nn as nn
from modules.RMSNorm import RMSNorm



def patchify(x, p_t=10, p_l=54):
    """
    x: (B, T, L) - batch of 2D CSI signals
    Returns:
        patches: (B, N_patch, patch_dim), where patch_dim = p_t * p_l
    """
    B, T, L = x.shape
    assert T % p_t == 0 and L % p_l == 0, "T and L must be divisible by patch size"
    
    x = x.reshape(B, T // p_t, p_t, L // p_l, p_l)  # (B, T_patch, p_t, L_patch, p_l)
    x = x.permute(0, 1, 3, 2, 4)  # (B, T_patch, L_patch, p_t, p_l)
    x = x.reshape(B, -1, p_t * p_l)  # (B, N_patch, patch_dim)
    return x


class WiSRL_tune(nn.Module):
    '''

    '''
    def __init__(
        self, 
        original_model, 
        input_dim: int, 
        hidden_dim: int,
        output_dim: int,
    ):
        super().__init__()


        # # **新的输入层**
        # self.input_layer_amp = nn.Linear(input_dim, hidden_dim)  # 让输入适配 for amplitude
        # self.input_layer_pha = nn.Linear(input_dim, hidden_dim)  # 让输入适配 for phase

        # # position_embedding：sincos位置编码
        # self.pos_embed = SinusoidalPositionalEncoding(hidden_dim) # 位置编码

        ##  **输入层**
        self.input_layer_amp = original_model.patch_embed_amp  # 让输入适配 for amplitude
        self.input_layer_pha = original_model.patch_embed_pha  # 让输入适配 for phase

        # position_embedding
        self.pos_embed = original_model.pos_embed # 位置编码

        # 2个 encoder **冻结 encoder**
        self.encoder_amp = original_model.encoder_amp  
        self.encoder_pha = original_model.encoder_pha  

        # # **冻结 encoder**
        for param in self.encoder_amp.parameters():
            param.requires_grad = False
        for param in self.encoder_pha.parameters():
            param.requires_grad = False

        
        # 中间融合层
        self.fusion_layer = original_model.middle_layer
        self.fusion_norm = original_model.middle_norm


        # 2层 MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),  # 线性层，可根据需求调整
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)  # 适配最终任务
        )

    
    def forward(self, amplitude, phase):
        
        # patchify
        amplitude = patchify(amplitude)
        phase = patchify(phase)

        # encoder - MLP
        x_amp = self.input_layer_amp(amplitude)
        x_pha = self.input_layer_pha(phase)

        x_amp = self.pos_embed(x_amp)
        x_pha = self.pos_embed(x_pha)

        x_amp = self.encoder_amp(x_amp)  
        x_pha = self.encoder_pha(x_pha)



        # 中间层聚合
        x_concat = torch.concat([x_amp, x_pha], dim=2)
        x_concat = self.fusion_layer(x_concat)
        x_concat = self.fusion_norm(x_concat)

        # MLP层输出
        logits = self.mlp(x_concat)  # 这里输出的是 logits
        logits = torch.mean(logits, dim=1)  # (B, L, D) -> (B, D)
        return logits  