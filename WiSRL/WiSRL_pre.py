## for pretraining
## mask: 0.25、预训练数据量: 100%、Encoder: Biblock v2、input: 幅值+相位 (finish)
## mask: 0.50、预训练数据量: 100%、Encoder: Biblock v2、input: 幅值+相位 (finish)
## mask: 0.75、预训练数据量: 100%、Encoder: Biblock v2、input: 幅值+相位 (finish)
## mask: 0.90、预训练数据量: 100%、Encoder: Biblock v2、input: 幅值+相位 (finish)

## mask: 0.75、预训练数据量: 80%、Encoder: Biblock v2、input: 幅值+相位 (finish)
## mask: 0.75、预训练数据量: 60%、Encoder: Biblock v2、input: 幅值+相位 (finish)
## mask: 0.75、预训练数据量: 40%、Encoder: Biblock v2、input: 幅值+相位 (finish)
## mask: 0.75、预训练数据量: 20%、Encoder: Biblock v2、input: 幅值+相位 (finish)

## mask: 0.75、预训练数据量: 100%、Encoder: Biblock v1、input: 幅值+相位 (finish)
## mask: 0.75、预训练数据量: 100%、Encoder: Biencoder、input: 幅值+相位 (finish)
## mask: 0.75、预训练数据量: 100%、Encoder: Mamba、input: 幅值+相位 (finish)

## mask: 0.75、预训练数据量: 100%、Encoder: Biblock v2、input: 幅值 (finish)
## mask: 0.75、预训练数据量: 100%、Encoder: Biblock v2、output: 幅值 (finish)

import os
import numpy as np
import random

from models.pretraining_Biblock_model_test import WiSRL_pre # Bi-Block
# from models.pretraining_Biencoder_model import WiSRL_pre # Bi-Encoder
# from models.pretraining_model import WiSRL_pre # Mamba
# from models.pretraining_AMP_model import WiSRL_pre # 只输入幅值 
# from models.pretraining_model_AMP import WiSRL_pre # 只恢复幅值

# from models.pretraining_Biblock_model_patchify import WiSRL_pre # patch-level 预训练
# from models.pretraining_Biblock_model_notaligned import WiSRL_pre # 非对称预训练

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
# from dataset_HAR import AmplitudeDataset
from dataset import ComplexDataset
from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
## 预训练模型
def pre_training():
    mask_ratio = 0.75
    bimamba_type = "v2"
    pre_datasize = 1
    pre_link = 6
    # 日志位置
    log_dir_path = "./runs/PRE/pre_75_100_Biblockv2_widar_both_6link_test"

    # 加载的参数
    old_Pre_checkpoint_path = "./model_weight/PRE/pre_75_100_Biblockv2_widar_6link_test.pth"

    ## 最后保存的参数
    # Pre_final_weight_path = "./model_weight/wimambassm_U1_3_2_final_100.pth"

    new_Pre_checkpoint_path = "./model_weight/PRE/pre_75_100_Biblockv2_widar_6link_test.pth"

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    seed =1024  # 可以换个数字试试
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


    # shape: (n, l ,d)
    
    input_dim_pre = 90*6
    hidden_dim = 64 # 32 64 128
    encoder_n_layers = 3 # encoder稍微层数多一点，承担更多任务
    decoder_n_layers = 2 # decoder稍微简单一点，有利于encoder得到更好的信号表示
    

    train_losses = []  # 用来记录训练过程中每个batch的训练损失
    val_losses = [] # 用来记录训练过程中每个batch的验证损失

    init_lr = 0.0005 # 初始学习率 0.0005
    final_lr = 0.0001 # 最终学习率 0.0001
    init_weight_decay = 1e-3 # 衰减系数 0.001
    num_epochs = 100 # 先用10轮进行训练
    batch_size = 64 # batch大小
    data_folder = './dataset/WIDAR_Pre' # 数据集路径
    # data_folder = '/mnt/data/keran/project/Flow-LLM/FAE/dataset/XRF55_Pre'
    # data_folder = '/mnt/data/keran/project/Flow-LLM/FAE/dataset/WIDAR_Pre'
   

    # **检查是否存在已保存的 checkpoint**
    start_epoch = 0  # 记录从哪个 epoch 开始训练
    best_val_loss = float('inf')  # 记录最佳 val_loss



    # 加载数据
    dataset = ComplexDataset(data_folder)

    # 数据集分割
    train_size = int(0.8 * len(dataset) * pre_datasize)
    val_size = len(dataset) - train_size #验证集 查看收敛情况和loss
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
   

    
    # 加载训练集和测试集
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8,  pin_memory=True, shuffle=True, drop_last=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8,  pin_memory=True, shuffle=False)


    # 初始化模型
    model = WiSRL_pre(
        input_dim=input_dim_pre,
        hidden_dim=hidden_dim,
        bimamba_type = bimamba_type, 
        encoder_nlayers=encoder_n_layers,
        decoder_nlayers=decoder_n_layers,
        mask_ratio=mask_ratio
    ).to(device)

    # 定义优化器 AdamW
    # optimizer = optim.Adam(params=model.parameters(), lr=init_lr)
    optimizer = optim.AdamW(model.parameters(), lr=init_lr, weight_decay=init_weight_decay) # 使用AdamW优化器，防止梯度爆炸
    
    ## 定义调度器 scheduler，保证学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#         optimizer,
#         T_max=100,
#         eta_min=final_lr
# )

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=95, eta_min = final_lr) # 余弦退火，保证稳定收敛
    

    # **如果 checkpoint 存在，则加载**
    if os.path.exists(old_Pre_checkpoint_path):
        Pre_checkpoint = torch.load(old_Pre_checkpoint_path, map_location=device)
        model.load_state_dict(Pre_checkpoint['model_state_dict'])
        optimizer.load_state_dict(Pre_checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(Pre_checkpoint['scheduler_state_dict'])
        start_epoch = Pre_checkpoint['epoch'] + 1  # 继续训练
        best_val_loss = Pre_checkpoint['best_val_loss']
        print(f"加载 checkpoint 成功！从 epoch {start_epoch} 继续训练，最佳 val_loss={best_val_loss:.4f}")

    # model_train = torch.compile(model)  # 编译训练模型
    model_train = model
    model_eval = model  # 评估时用未编译的模型

    # # Convert data to PyTorch tensors
    # mag_tensor = torch.tensor(mag, dtype=torch.float32)
    # pha_tensor = torch.tensor(pha, dtype=torch.float32)


    ## 启用tensorboard记录日志，方便后续可视化
    writer = SummaryWriter(log_dir=log_dir_path)  # log_dir 用于存放日志，便于后期可视化

    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = False  # 禁用 TensorFloat-32
    torch.backends.cudnn.allow_tf32 = False  # 禁用 cuDNN 的 TF32 计算

    # train 
    # 要好好修改一下maybe，期望能够在每个世代训练后给出一个平均loss，方便观察收敛情况
    for epoch in range(start_epoch, num_epochs):
        torch.cuda.empty_cache()
        model_train.train()
        epoch_train_loss = 0
        print(f"Epoch [{epoch+1}/{num_epochs}]训练开始")
        # print(f"Epoch [{epoch+1}/{num_epochs}]")
        for batch_idx, (amp, pha, label) in enumerate(train_loader):
            amp = amp.to(torch.float32).to(device)
            pha = pha.to(torch.float32).to(device)

            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}]")

            # train_loss, x_amp, mask = model_train(amp)
            train_loss, x_amp, x_pha, mask = model_train(amp, pha)
            print(f"Train Loss = [{train_loss}]")
            epoch_train_loss += train_loss.item()

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            # 记录 batch 级别的 loss
            writer.add_scalar("Loss/train_batch", train_loss.item(), epoch * len(train_loader) + batch_idx)

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        writer.add_scalar("Loss/train_epoch", avg_train_loss, epoch)

        # Validation 验证模型性能
        model_eval.eval()  # 评估模式
        epoch_val_loss = 0
        with torch.no_grad():  # 关闭梯度计算，加速运算
            for batch_idx, (amp, pha, label) in enumerate(val_loader):
                amp = amp.to(torch.float32).to(device)
                pha = pha.to(torch.float32).to(device)
                # label = label.to(torch.float32).to(device)

                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(val_loader)}]")

                # val_loss, x_amp,mask = model_eval(amp)
                val_loss, x_amp, x_pha, mask = model_eval(amp, pha)
                print(f"Validation Loss = [{val_loss}]")
                
                epoch_val_loss += val_loss.item()
    
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        writer.add_scalar("Loss/val_epoch", avg_val_loss, epoch)

        print(f"Epoch {epoch+1}: Train Loss = {train_losses[-1]}, Val Loss = {val_losses[-1]}")

        # **更新学习率** avg_val_loss
        scheduler.step(avg_val_loss)

        # 记录学习率
        current_lr = scheduler.optimizer.param_groups[0]['lr']
        writer.add_scalar("Learning Rate", current_lr, epoch)
        print(f"Epoch {epoch+1}: 当前学习率 = {current_lr}")

        # **保存最佳模型**
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss
            }, new_Pre_checkpoint_path)
            print(f"更新最佳 Val Loss={best_val_loss}，已保存 checkpoint")
    


    writer.close()
    # # **保存loss**
    # np.save('./loss_record/train_loss_U1_3_2_50.npy', np.array(train_losses))  # 保存训练损失
    # np.save('./loss_record/val_loss_U1_3_2_50.npy', np.array(val_losses))  # 保存训练损失

    # **保存模型**
    # torch.save(model.state_dict(), Pre_final_weight_path)
    print("Wi-Mamba saved successfully.")


########待续

if __name__ == "__main__":
   pre_training()
