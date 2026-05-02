import torch
import torch.nn as nn


class WiSRL_lda(nn.Module):
    '''

    '''
    def __init__(
        self, 
        original_model, 
    ):
        super().__init__()


        ##  **输入层**
        self.input_layer_amp = original_model.input_layer_amp  # 让输入适配 for amplitude
        self.input_layer_pha = original_model.input_layer_pha  # 让输入适配 for phase

        # position_embedding
        # self.pos_embed = original_model.pos_embed # 位置编码

        # 2个 encoder
        self.encoder_amp = original_model.encoder_amp  
        self.encoder_pha = original_model.encoder_pha  

        
        # 中间融合层
        self.fusion_layer = original_model.fusion_layer
        self.fusion_norm = original_model.fusion_norm

        self.output_layer = original_model.mlp[0]


    
    def forward(self, amplitude, phase):
        
        x_amp = self.input_layer_amp(amplitude)
        x_pha = self.input_layer_pha(phase)

        # x_amp = self.pos_embed(x_amp)
        # x_pha = self.pos_embed(x_pha)

        x_amp = self.encoder_amp(x_amp)  
        x_pha = self.encoder_pha(x_pha)


        # 中间层聚合
        x_concat = torch.concat([x_amp, x_pha], dim=2)
        x_concat = self.fusion_layer(x_concat)
        x_concat = self.fusion_norm(x_concat)

        output = self.output_layer(x_concat)

        return output  