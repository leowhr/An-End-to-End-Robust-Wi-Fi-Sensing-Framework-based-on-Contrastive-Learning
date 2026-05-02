import torch
import torch.nn as nn
from modules.RMSNorm import RMSNorm



class WiSRL_tune(nn.Module):
    '''

    '''
    def __init__(
        self, 
        original_model, 
        input_dim: int, 
        hidden_dim: int,
    ):
        super().__init__()


        # # **新的输入层**
        # self.input_layer_amp = nn.Linear(input_dim, hidden_dim)  # 让输入适配 for amplitude
        # self.input_layer_pha = nn.Linear(input_dim, hidden_dim)  # 让输入适配 for phase


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
        self.time_mapper = nn.Sequential(
            nn.Linear(600, 400),  # 线性层，可根据需求调整
            nn.ReLU()
        )
        
        self.feature_mapper = nn.Linear(hidden_dim, input_dim)  # 适配最终任务

        


         

    
    def forward(self, amplitude, phase):
        
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

        
        x_concat = x_concat.permute(0, 2, 1)         # (B, 64, 600)
        x_concat = self.time_mapper(x_concat)       # (B, 64, 400)
        x_concat = x_concat.permute(0, 2, 1)        # (B, 400, 64)
        x_reg = self.feature_mapper(x_concat)    # (B, 400, 540)


        return x_reg
    


        