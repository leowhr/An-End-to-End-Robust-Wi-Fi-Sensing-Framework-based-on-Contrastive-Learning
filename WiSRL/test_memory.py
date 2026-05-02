# import torch
# import torch.nn as nn

# from models.test_finetune_model import WiSRL_tune

# from torch.profiler import profile, record_function, ProfilerActivity


# def count_parameters(model):
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"📦 Total parameters: {total_params / 1e6:.2f} M")
#     print(f"🎯 Trainable parameters: {trainable_params / 1e6:.2f} M")


# def main():
#     torch.cuda.empty_cache()

#     batch, length, dim = 128, 1450, 540

#     seed = 624  # 随机种子，可以换个数字试试
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)

#     torch.set_float32_matmul_precision('high')
#     torch.backends.cuda.matmul.allow_tf32 = False
#     torch.backends.cudnn.allow_tf32 = False
    
    
#     # 定义微调模型
#     model_tune = WiSRL_tune(
#         input_dim = 540, # 输入维度
#         hidden_dim = 64, # encoder,decoder维度
#         bimamba_type = "v2",
#         output_dim = 10,
#         encoder_nlayers = 3, # encoder block数量 
#     )

#     for name, param in model_tune.named_parameters():
#         print(name, param.dtype)


#     count_parameters(model_tune)


#     device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
#     torch.cuda.set_device(device)  # ⭐️ 关键设置
#     torch.cuda.reset_peak_memory_stats(device)  

#     model_tune = model_tune.to(device)

#     model_tune.eval()



#     # 推理代码
#     with profile(
#         activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
#         profile_memory=True,  # ← 关键：打开显存统计
#         with_stack=True,      # 可选：方便调试定位
#         with_flops=True       # 可选：记录 FLOPs
#     ) as prof:
#         with torch.no_grad():
#             for i in range(100):
#                 x1 = torch.randn(batch, length, dim).to(device)
#                 x2 = torch.randn(batch, length, dim).to(device)
#                 print(x1.dtype)
#                 print(x2.dtype)
#                 y1=model_tune(x1,x2)

#     print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=50))

#     # model_tune.train()
#     # for i in range(100):for i in range(100):
#     #    x1 = torch.randn(batch, length, dim).to(device)
#     #    x2 = torch.randn(batch, length, dim).to(device)
#     #    y=model_tune(x1,x2)

#     # # 查看当前GPU设备的内存使用情况
#     print(f"最大分配显存: {torch.cuda.max_memory_allocated(device) / (1024**3):.2f} GB")
#     print(f"最大预留显存: {torch.cuda.max_memory_reserved(device) / (1024**3):.2f} GB")

#     # while True:
#     #   pass

    


    
# if __name__ == "__main__":
#     main()


# import torch

# import torchvision.models as models

# from torch.profiler import profile, record_function, ProfilerActivity


# torch.cuda.empty_cache()

# seed = 624  # 随机种子，可以换个数字试试
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)

# # 加载 ResNet-50 模型
# model = models.resnet50()
# # model = models.convnext_tiny(pretrained=False)

# for name, param in model.named_parameters():
#     print(name, param.dtype)


# # torch.set_float32_matmul_precision('high')
# # torch.backends.cuda.matmul.allow_tf32 = False
# # torch.backends.cudnn.allow_tf32 = False

# # 统计参数总数（单位：百万）
# total_params = sum(p.numel() for p in model.parameters())
# print(f"ResNet-50 参数量: {total_params / 1e6:.2f} M")

# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")
# torch.cuda.set_device(device)  
# torch.cuda.reset_peak_memory_stats(device)




# model = model.to(device)

# model.eval()
# # 推理代码
# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
#     with torch.no_grad():
#         for i in range(100):
#             input = torch.randn(128,3,1450, 180).to(device)
#             print(input.dtype)
#             y =model(input)


# print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=50))

# print(f"推理最大分配显存: {torch.cuda.max_memory_allocated(device) / (1024**3):.2f} GB")
# print(f"推理最大预留显存: {torch.cuda.max_memory_reserved(device) / (1024**3):.2f} GB")

# # while True:
# #     pass



import os
import numpy as np
import random

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from dataset_reg import ComplexDataset
from torch.utils.tensorboard import SummaryWriter

from models.pretraining_Biblock_model import WiSRL_pre # Bi-Block
from models.finetune_reg_model import WiSRL_tune

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

class TransformerModel(nn.Module):
    def __init__(self, input_dim=540, hidden_dim=64, num_encoder_layers=2, num_decoder_layers=1, 
                 input_len=600, output_len=400):
        super().__init__()
        self.input_len = input_len
        self.output_len = output_len

        # Optional: project input to hidden dimension
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Positional encoding
        self.pos_embed = SinusoidalPositionalEncoding(hidden_dim) # 位置编码

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, input_dim)

        # Learnable query for decoder
        self.query_embed = nn.Parameter(torch.randn(output_len, hidden_dim))

    def forward(self, x):
        """
        x: (batch_size, input_len, input_dim)
        """
        B, T, C = x.shape  # (64, 600, 540)

        # Project input to hidden dim
        x_amp = self.input_proj(x) # (64, 600, 64)
        x_amp = self.pos_embed(x_amp)
        

        # Encode
        x_amp = self.encoder(x_amp)  # (64, 600, hidden_dim)

        # Prepare decoder query
        query = self.query_embed.unsqueeze(0).expand(B, -1, -1)  # (64, 400, hidden_dim)

        # Decode
        out = self.decoder(query, x_amp)  # (64, 400, hidden_dim)

        # Project back to original dim
        out = self.output_proj(out)  # (64, 400, 540)
        return out
    



torch.cuda.empty_cache()

seed = 624  # 随机种子，可以换个数字试试
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)




device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
torch.cuda.set_device(device)  
torch.cuda.reset_peak_memory_stats(device)




# model = TransformerModel()

# # 统计参数总数（单位：百万）
# total_params = sum(p.numel() for p in model.parameters())
# print(f"Transformer 参数量: {total_params / 1e6:.2f} M")

# model = model.to(device)

# model.eval()


# # 推理代码
# with torch.no_grad():
#     for i in range(100):
#         input = torch.randn(128,600,540).to(device)
#         # print(input.dtype)
#         y =model(input)




# print(f"推理最大分配显存: {torch.cuda.max_memory_allocated(device) / (1024**3):.2f} GB")
# print(f"推理最大预留显存: {torch.cuda.max_memory_reserved(device) / (1024**3):.2f} GB")


model_pre = WiSRL_pre(
        input_dim= 540,
        hidden_dim=64,
        bimamba_type = 'v2',
        encoder_nlayers=3,
        decoder_nlayers=2,
        mask_ratio=0.75
    )

model_tune = WiSRL_tune(
        original_model=model_pre,
        input_dim = 540,
        hidden_dim=64
    )


# 统计参数总数（单位：百万）
total_params = sum(p.numel() for p in model_tune.parameters())
print(f"Wi-MAEmba 参数量: {total_params / 1e6:.2f} M")

model_tune = model_tune.to(device)

model_tune.eval()


# 推理代码
with torch.no_grad():
    for i in range(100):
        input1 = torch.randn(128,600,540).to(device)
        input2 = torch.randn(128,600,540).to(device)
        # print(input.dtype)
        y =model_tune(input1,input2)




print(f"推理最大分配显存: {torch.cuda.max_memory_allocated(device) / (1024**3):.2f} GB")
print(f"推理最大预留显存: {torch.cuda.max_memory_reserved(device) / (1024**3):.2f} GB")