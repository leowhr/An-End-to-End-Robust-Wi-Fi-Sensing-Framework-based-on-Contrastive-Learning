from models.pretraining_Biblock_model import WiSRL_pre # Bi-Block


# from models.pretraining_model import WiSRL_pre # Mamba


from models.finetune_model import WiSRL_tune

import torch

torch.cuda.empty_cache()

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# 初始化模型
model_pre = WiSRL_pre(
    input_dim=540,
    hidden_dim=64,
    bimamba_type = "v2",
    encoder_nlayers=3,
    decoder_nlayers=2,
    mask_ratio=0.75
    ).to(device)



model_tune = WiSRL_tune(
    original_model=model_pre,
    input_dim=540,
    hidden_dim=64,
    output_dim=10
    ).to(device)


x = torch.randn(128, 1450, 540).to(device)


loss,y1,y2,mask=model_pre(x,x)

y3=model_tune(x,x)

print(y1.shape)
print(y2.shape)
print(y3.shape)


# import torch
# from torch.utils.data import DataLoader
# from dataset import ComplexDataset  # 如果类在当前文件中定义，就不需要这一行

# # 假设数据存放在 data/wifi_csi/ 目录下
# data_folder = "/mnt/data/keran/LKR/WiSRL/dataset/WIDAR_Pre"

# # 创建数据集实例
# dataset = ComplexDataset(data_folder=data_folder)

# # 创建 DataLoader
# dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# # 取出第一个样本
# amplitude, phase_cos, phase_sin, label, amp_min, amp_max = next(iter(dataloader))

# print("✅ 数据加载成功")
# print(f"Amplitude shape: {amplitude.shape}")
# print(f"Phase_cos shape: {phase_cos.shape}")
# print(f"Phase_sin shape: {phase_sin.shape}")
# print(f"Label: {label}")
# print(f"Amp_min: {amp_min}")
# print(f"Amp_max: {amp_max}")