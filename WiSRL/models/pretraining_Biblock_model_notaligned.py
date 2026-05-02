from modules.MambaModel import MambaModel
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

    
    def gen_mask_notaligned(self, x1, x2, mask_ratio=0.75):
        N, L, D = x1.shape # x.shape = [batch_size, seq_len, dim]
        keep_l = int(L * (1.0 - mask_ratio))

        ## 生成amp mask 和遮蔽后的数据
        # 随机生成抽取以及恢复index
        noise_amp = torch.rand([N, L], device=x1.device)
        ids_shuffle_amp = torch.argsort(noise_amp, dim=1)
        ids_restore_amp = torch.argsort(ids_shuffle_amp, dim=1)

        # 随机丢弃一些index，生成mask_x
        ids_keep_amp = ids_shuffle_amp[:, :keep_l]
        amp_mask = torch.gather(x1, dim=1, index=ids_keep_amp.unsqueeze(-1).repeat(1, 1, D)) # mask_x.shape = [N, L*(1-rate), D]

        # 生成 amp mask（对被mask掉的部分置一）
        mask_amp = torch.ones([N, L], device=x1.device)
        mask_amp[:, :keep_l] = 0
        mask_amp = torch.gather(mask_amp, dim=1, index=ids_restore_amp)


        ## 生成pha mask 和遮蔽后的数据
        # 随机生成抽取以及恢复index
        noise_pha = torch.rand([N, L], device=x2.device)
        ids_shuffle_pha = torch.argsort(noise_pha, dim=1)
        ids_restore_pha = torch.argsort(ids_shuffle_pha, dim=1)

        # 随机丢弃一些index，生成mask_x
        ids_keep_pha = ids_shuffle_pha[:, :keep_l]
        pha_mask = torch.gather(x2, dim=1, index=ids_keep_pha.unsqueeze(-1).repeat(1, 1, D)) # mask_x.shape = [N, L*(1-rate), D]

        # 生成 pha mask（对被mask掉的部分置一）
        mask_pha = torch.ones([N, L], device=x2.device)
        mask_pha[:, :keep_l] = 0
        mask_pha = torch.gather(mask_pha, dim=1, index=ids_restore_pha)

        # 返回shuffle后的x1、x2，mask，用于恢复的id
        return amp_mask, pha_mask, mask_amp, mask_pha, ids_restore_amp, ids_restore_pha
    
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
        x_amp_mask, x_pha_mask, mask_amp, mask_pha, ids_restore_amp, ids_restore_pha = self.gen_mask_notaligned(x_amp, x_pha, mask_ratio=self.mask_ratio)

        
        # # 前向传播encoder，norm
        x_amp_mask = self.encoder_amp(x_amp_mask)
        x_pha_mask = self.encoder_pha(x_pha_mask)

        return x_amp_mask, x_pha_mask, mask_amp, mask_pha, ids_restore_amp, ids_restore_pha
    
    # 中间层聚合
    def middle_concat(self, x_amp, x_pha):
        x_concat = torch.concat([x_amp, x_pha], dim=2)
        x_concat = self.middle_layer(x_concat)
        x_concat = self.middle_norm(x_concat)
        return x_concat

    # 两个解码器并行前向传播
    def double_decoder_forward(self, x, ids_restore_amp, ids_restore_pha):
        N, mask_L, hidden_dim = x.shape
        _, L = ids_restore_amp.shape

        # embed
        x_amp = self.decoder_embed_amp(x)
        x_pha = self.decoder_embed_pha(x)

        # 加上mask_tokens并unshuffle
        mask_tokens_amp = self.mask_token_amp.repeat(N, L - mask_L, 1)
        mask_tokens_pha = self.mask_token_pha.repeat(N, L - mask_L, 1)

        restore_x_amp = torch.cat([x_amp, mask_tokens_amp], dim=1)
        restore_x_pha = torch.cat([x_pha, mask_tokens_pha], dim=1)

        # 分别恢复幅值和相位顺序
        ids_restore_amp = ids_restore_amp.unsqueeze(-1).repeat(1, 1, self.hidden_dim)
        ids_restore_pha = ids_restore_pha.unsqueeze(-1).repeat(1, 1, self.hidden_dim)

        restore_x_amp = torch.gather(restore_x_amp, dim=1, index=ids_restore_amp) # unshuffle
        restore_x_pha = torch.gather(restore_x_pha, dim=1, index=ids_restore_pha)

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
        x_amp, x_pha, mask_amp, mask_pha, ids_restore_amp, ids_restore_pha = self.double_encoder_forward(amplitude, phase)
        x_concat = self.middle_concat(x_amp, x_pha)
        x_amp, x_pha = self.double_decoder_forward(x_concat, ids_restore_amp, ids_restore_pha)
        amp_loss = self.construct_loss(x_amp, amplitude, mask_amp)
        pha_loss = self.construct_loss(x_pha, phase, mask_pha)

        return amp_loss + pha_loss, x_amp, x_pha, mask_amp, mask_pha