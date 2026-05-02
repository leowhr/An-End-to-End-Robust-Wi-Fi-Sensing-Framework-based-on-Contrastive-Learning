from modules.MambaModel import MambaModel
from modules.MambaModel_Bidirectional import MambaModel_Bidirectional
from modules.RMSNorm import RMSNorm
import torch
import torch.nn as nn

class CrossAttentionLayer(nn.Module):
    """简单Cross-Attention模块, Q,K,V均来自不同输入"""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x_q, x_kv):
        """
        x_q: [B, Lq, D]
        x_kv: [B, Lkv, D]
        """
        B, Lq, D = x_q.shape
        _, Lkv, _ = x_kv.shape

        q = self.to_q(x_q).view(B, Lq, self.num_heads, D // self.num_heads).transpose(1, 2)  # [B, heads, Lq, head_dim]
        k = self.to_k(x_kv).view(B, Lkv, self.num_heads, D // self.num_heads).transpose(1, 2)  # [B, heads, Lkv, head_dim]
        v = self.to_v(x_kv).view(B, Lkv, self.num_heads, D // self.num_heads).transpose(1, 2)  # [B, heads, Lkv, head_dim]

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, heads, Lq, Lkv]
        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)  # [B, heads, Lq, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, Lq, D)  # [B, Lq, D]
        out = self.to_out(out)
        return out


class BiDirectionalCrossAttention(nn.Module):
    """
    双向交叉注意力融合：输入两个序列，分别作为Q,K,V进行互相注意力融合
    """
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.cross_attn_1 = CrossAttentionLayer(dim, num_heads)
        self.cross_attn_2 = CrossAttentionLayer(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x1, x2):
        """
        x1: [B, L1, D]
        x2: [B, L2, D]
        返回融合后的 x1, x2
        """
        x1_new = self.norm1(x1 + self.cross_attn_1(x1, x2))
        x2_new = self.norm2(x2 + self.cross_attn_2(x2, x1))
        return x1_new, x2_new



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
    
class Decoder(nn.Module):
    '''
    
    '''
    def __init__(
        self, 
        hidden_dim: int,
        n_layers: int
    ):
        super().__init__()
        self.encoder = MambaModel(
            hidden_dim=hidden_dim,   # 模型内部主维度
            n_layers=n_layers,
        )
    
    def forward(self, x):
        x = self.encoder(x)
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
    
class WiSRL_pre(nn.Module):
    def __init__(
        self, 
        input_dim: int, # 输入维度
        hidden_dim: int, # encoder,decoder维度
        bimamba_type,
        encoder_nlayers: int = 6, # encoder block数量
        decoder_nlayers: int = 6, # decoder block数量
        mask_ratio: float = 0.75 # 遮挡比例
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim


        # --------------------------------------------------------------------------
        # MAE encoder specifics
        # patch_embedding: Linear代替
        self.patch_embed_amp = nn.Linear(input_dim, hidden_dim) # for amplitude
        self.patch_embed_pha = nn.Linear(input_dim, hidden_dim) # for phase

        # position_embedding：sincos位置编码
        self.pos_embed = SinusoidalPositionalEncoding(hidden_dim) # 位置编码

        # 2个encoder
        self.encoder_amp = Encoder(hidden_dim=hidden_dim, bimamba_type=bimamba_type,n_layers=encoder_nlayers)
        self.encoder_pha = Encoder(hidden_dim=hidden_dim, bimamba_type=bimamba_type,n_layers=encoder_nlayers)

        # --------------------------------------------------------------------------


        # --------------------------------------------------------------------------
        # 中间融合层
        # 双向交叉注意力融合层
        # self.bi_cross_attn = BiDirectionalCrossAttention(hidden_dim)

        self.middle_layer = nn.Linear(2*hidden_dim, hidden_dim)
        self.middle_norm = RMSNorm(dim=hidden_dim)
        # --------------------------------------------------------------------------


        # --------------------------------------------------------------------------
        # MAE decoder specifics
        # embedding：Linear
        self.decoder_embed_amp = nn.Linear(hidden_dim, hidden_dim) # for amplitude
        self.decoder_embed_pha = nn.Linear(hidden_dim, hidden_dim) # for phase

        # mask_tokens
        self.mask_token_amp = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.mask_token_pha = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        # position_embedding：sincos位置编码
        self.decoder_pos_embed = SinusoidalPositionalEncoding(hidden_dim) # 位置编码

        # 2个decoder
        self.decoder_amp = Decoder(hidden_dim, n_layers=decoder_nlayers)
        self.decoder_pha = Decoder(hidden_dim, n_layers=decoder_nlayers)


        # Prediction
        self.pred_amp = nn.Linear(hidden_dim, input_dim)
        self.pred_pha = nn.Linear(hidden_dim, input_dim)
        # --------------------------------------------------------------------------

        self.criterion = nn.MSELoss()

    def gen_mask(self, x1, x2, mask_ratio=0.75):
        N, L, D = x1.shape # x.shape = [batch_size, seq_len, dim]
        keep_l = int(L * (1.0 - mask_ratio))

        # 随机生成抽取以及恢复index
        noise = torch.rand([N, L], device=x1.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # 随机丢弃一些index，生成mask_x
        ids_keep = ids_shuffle[:, :keep_l]
        mask_x1 = torch.gather(x1, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D)) # mask_x.shape = [N, L*(1-rate), D]
        mask_x2 = torch.gather(x2, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # 生成mask（对被mask掉的部分置一）
        mask = torch.ones([N, L], device=x1.device)
        mask[:, :keep_l] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        # 返回shuffle后的x1、x2，mask，用于恢复的id
        return mask_x1, mask_x2, mask, ids_restore
    
    # 两个编码器并行前向传播
    def double_encoder_forward(self, amplitude, phase):
        N, L, D = amplitude.shape

        # generate token 
        x_amp = self.patch_embed_amp(amplitude)
        x_pha = self.patch_embed_pha(phase)

        # 位置编码
        x_amp = self.pos_embed(x_amp)
        x_pha = self.pos_embed(x_pha)

    
        # randomly shuffle, remove
        x_amp_mask, x_pha_mask, mask, ids_restore = self.gen_mask(x_amp, x_pha, mask_ratio=self.mask_ratio)

        
        # # 前向传播encoder，norm
        x_amp_mask = self.encoder_amp(x_amp_mask)
        x_pha_mask = self.encoder_pha(x_pha_mask)

        return x_amp_mask, x_pha_mask, mask, ids_restore
    
    # 中间层聚合
    def middle_concat(self, x_amp, x_pha):

        # 交互注意力融合
        # x_amp, x_pha = self.bi_cross_attn(x_amp, x_pha)

        # 线性融合
        x_concat = torch.concat([x_amp, x_pha], dim=2)
        x_concat = self.middle_layer(x_concat)
        x_concat = self.middle_norm(x_concat)
        return x_concat

    # 两个解码器并行前向传播
    def double_decoder_forward(self, x, ids_restore):
        N, mask_L, hidden_dim = x.shape
        _, L = ids_restore.shape

        # embed
        x_amp = self.decoder_embed_amp(x)
        x_pha = self.decoder_embed_pha(x)

        # 加上mask_tokens并unshuffle
        mask_tokens_amp = self.mask_token_amp.repeat(N, L - mask_L, 1)
        mask_tokens_pha = self.mask_token_pha.repeat(N, L - mask_L, 1)
        restore_x_amp = torch.cat([x_amp, mask_tokens_amp], dim=1)
        restore_x_pha = torch.cat([x_pha, mask_tokens_pha], dim=1)
        ids_restore = ids_restore.unsqueeze(-1).repeat(1, 1, self.hidden_dim)
        restore_x_amp = torch.gather(restore_x_amp, dim=1, index=ids_restore) # unshuffle
        restore_x_pha = torch.gather(restore_x_pha, dim=1, index=ids_restore)

        # position embedding
        restore_x_amp = self.decoder_pos_embed(restore_x_amp)
        restore_x_pha = self.decoder_pos_embed(restore_x_pha)

        # 前向传播decoder，norm
        restore_x_amp = self.decoder_amp(restore_x_amp)
        restore_x_pha = self.decoder_pha(restore_x_pha)


        # Predict
        restore_x_amp = self.pred_amp(restore_x_amp)
        restore_x_pha = self.pred_pha(restore_x_pha)

        return restore_x_amp, restore_x_pha
    
    def construct_loss(self, pred, raw, mask):
        # 只考虑被mask部分的损失
        mask = mask.unsqueeze(-1).repeat(1, 1, self.input_dim).bool()
        # print(mask.dtype)
        # print(torch.isnan(pred).any(), torch.isinf(pred).any())
        # print(torch.isnan(raw).any(), torch.isinf(raw).any())
        loss = self.criterion(pred[mask], raw[mask])
        return loss
    
    def forward(self, amplitude, phase):
        x_amp, x_pha, mask, ids_restore = self.double_encoder_forward(amplitude, phase)
        x_concat = self.middle_concat(x_amp, x_pha)
        x_amp, x_pha = self.double_decoder_forward(x_concat, ids_restore)
        amp_loss = self.construct_loss(x_amp, amplitude, mask)
        pha_loss = self.construct_loss(x_pha, phase, mask)

        return amp_loss + pha_loss, x_amp, x_pha, mask