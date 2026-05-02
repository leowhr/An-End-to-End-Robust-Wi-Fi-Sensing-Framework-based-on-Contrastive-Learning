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


        # position_embedding
        self.pos_embed = original_model.pos_embed # 位置编码

        # 1个 encoder 
        self.encoder_amp = original_model.encoder_amp  

        self.output_layer = original_model.mlp[0]



    
    def forward(self, amplitude):
        
        x_amp = self.input_layer_amp(amplitude)


        x_amp = self.pos_embed(x_amp)


        x_amp = self.encoder_amp(x_amp)  

        output = self.output_layer(x_amp)

        return output