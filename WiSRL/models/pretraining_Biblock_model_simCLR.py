from modules.MambaModel_Bidirectional import MambaModel_Bidirectional
from modules.RMSNorm import RMSNorm
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(
        self, 
        hidden_dim: int,
        bimamba_type,
        n_layers: int,
        pooling = "mean"
    ):
        super().__init__()
        self.encoder = MambaModel_Bidirectional(
            hidden_dim=hidden_dim,# 模型内部主维度
            bimamba_type = bimamba_type,   
            n_layers=n_layers
        )
        self.pooling = pooling
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x, get_feature=False):
        x=self.encoder(x)
        if get_feature:
            return x

        # 根据pooling方式对序列进行池化，得到固定维度的表示
        x = self.norm(x)
        if self.pooling == "mean":
            x = x.mean(dim=1)  # 对序列维度取平均
        elif self.pooling == "last":
            x = x[:, -1, :]  # 取序列最后一个位置的表示
        else:
            raise ValueError(f"Unsupported pooling type: {self.pooling}")
        return x

class MiddleConcat(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.middle_proj = nn.Linear(2*hidden_dim, hidden_dim)  # 将拼接后的特征投影回 hidden_dim 维度
        self.norm = RMSNorm(hidden_dim)  # 使用 RMSNorm 进行归一化
        # Cross-Attention层不采用，计算量过大，且效果提升不明显

    def forward(self, amp_feat, pha_feat):
        # amp_feat 和 pha_feat 的形状为 (batch_size, hidden_dim)
        combined_feat = torch.cat([amp_feat, pha_feat], dim=-1)  # (batch_size, 2*hidden_dim)
        projected_feat = self.middle_proj(combined_feat)  # (batch_size, hidden_dim)
        normalized_feat = self.norm(projected_feat)  # (batch_size, hidden_dim)
        return normalized_feat
    
class ProjectionHead(nn.Module):
    def __init__(self, hidden_dim: int, projection_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )
    
    def forward(self, x):
        return self.projection(x)
    
class WiSRL_pre(nn.Module):
    def __init__(
        self, 
        input_dim: int,
        hidden_dim: int,
        proj_dim: int,
        bimamba_type,
        encoder_nlayers: int,
    ):
        super().__init__()
        self.embedding_amp = nn.Linear(input_dim, hidden_dim)  # 幅值输入的线性嵌入
        self.embedding_pha = nn.Linear(input_dim, hidden_dim)  # 相位输入的线性嵌入
        self.encoder_amp = Encoder(
            hidden_dim=hidden_dim,
            bimamba_type=bimamba_type,
            n_layers=encoder_nlayers
        ) # 幅值编码器
        self.encoder_pha = Encoder(
            hidden_dim=hidden_dim,
            bimamba_type=bimamba_type,
            n_layers=encoder_nlayers
        ) # 相位编码器
        self.middle_concat = MiddleConcat(hidden_dim)  # 中间特征融合模块
        if proj_dim > 0:
            self.projection_head = ProjectionHead(hidden_dim, proj_dim)  # SimCLR投影头
        else:
            self.projection_head = nn.Identity()  # 如果不需要投影头，使用恒等映射  
        self.apply(self._init_linear_kaiming)

    @staticmethod
    def _init_linear_kaiming(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, a=0.0, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @staticmethod
    def _init_linear_xavier(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, amp, pha, get_feature=False):
        # 输入 amp 和 pha 的形状为 (batch_size, seq_len=1450, input_dim=90*3)
        assert amp.shape == pha.shape, "Amplitude and Phase inputs must have the same shape"
        assert amp.dim() == 3, "Input tensors must be 3-dimensional (batch_size, seq_len, input_dim)"
        assert amp.shape[1] == 1450, "Sequence length must be 1450"
        assert amp.shape[2] == 90*3, "Input dimension must be 270 (90*3)"
        amp_emb = self.embedding_amp(amp)  # (batch_size, seq_len, hidden_dim)
        pha_emb = self.embedding_pha(pha)  # (batch_size, seq_len, hidden_dim)

        amp_feat = self.encoder_amp(amp_emb, get_feature=get_feature)  # (batch_size, hidden_dim) if get_feature else (batch_size, hidden_dim)
        pha_feat = self.encoder_pha(pha_emb, get_feature=get_feature)  # (batch_size, hidden_dim)

        combined_feat = self.middle_concat(amp_feat, pha_feat)  # (batch_size, hidden_dim)

        if get_feature:
            return combined_feat

        proj = self.projection_head(combined_feat)  # (batch_size, proj_dim)
        return proj