import torch
import torch.nn as nn

class WiSRL_tune(nn.Module):
    '''

    '''
    def __init__(
        self, 
        original_model, 
        # input_dim: int, 
        hidden_dim: int,
        output_dim: int,
    ):
        super().__init__()

        self.original_model = original_model

        # 2层 MLP
        self.mlp = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),  # 线性层，可根据需求调整
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)  # 适配最终任务
        )

        self._init_mlp_kaiming()

    def _init_mlp_kaiming(self):
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=0.0, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


         

    
    def forward(self, feat1, feat2):

        feat1 = self.original_model(feat1, get_feature=True)  # (batch_size, hidden_dim)
        feat2 = self.original_model(feat2, get_feature=True)  # (batch_size, hidden_dim)
        # MLP层输出
        logits = self.mlp(torch.cat([feat1, feat2], dim=-1))  # 这里输出的是 logits
        return logits  