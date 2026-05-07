# for scratch
## 微调数据量：100%、Encoder: Biblock v2、输入：幅值+相位 (88.31%)
## 微调数据量：80%、Encoder: Biblock v2、输入：幅值+相位 (85.17%)
## 微调数据量：60%、Encoder: Biblock v2、输入：幅值+相位 (82.37%)
## 微调数据量：40%、Encoder: Biblock v2、输入：幅值+相位 (78.88%)
## 微调数据量：20%、Encoder: Biblock v2、输入：幅值+相位 (73.46%)

## 微调数据量：100%、Encoder: Mamba、输入：幅值+相位 

## 微调数据量：100%、Encoder: Biblock v2、输入：幅值
## 微调数据量：100%、Encoder: Biblock v2、输出：幅值

import os
import numpy as np
import random

from models.pretraining_Biblock_model_simCLR import WiSRL_pre # Bi-Block
from utils import nt_xent_loss, set_seed, seed_worker

from models.finetune_model import WiSRL_tune
# from models.finetune_AMP_model import WiSRL_tune # 只输入幅值微调训练

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataset import ComplexDataset
# from dataset_HAR import AmplitudeDataset
from torch.utils.tensorboard import SummaryWriter


## 下游任务微调
def fine_tuning():

    seed = 624  # 随机种子，可以换个数字试试
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    bimamba_type = "v2"
    tune_datasize = 0.2
    pre_link = 6
    tune_link = 6


    log_dir_path = "./runs/CLS/Scratch/scratch_100_20_Biblockv2_widar_6link(5000)" #日志

    ## 加载的数据
    # old_Tune_final_weight_path = "./model_weight/wimamba_U1_4_2_final_100_tune100_finetuning.pth" # 上次训练最后的模型

    old_Tune_checkpoint_path = "./model_weight/CLS/Scratch/scratch_100_20_Biblockv2_widar_6link(5000).pth" # 上次训练某一轮保存的微调权重和优化器状态

    ## 保存的数据
    # Tune_final_weight_path = "./model_weight/wimambassm_U1_3_2_best_100_10fenlei_tune50_final_scratch_finetune(lr-0.0004-0.0001).pth" # 保存最后模型

    new_Tune_checkpoint_path = "./model_weight/CLS/Scratch/scratch_100_20_Biblockv2_widar_6link(5000).pth" # 每一轮保存的微调权重和优化器状态

    # shape: (n, l ,d)
    
    
    input_dim_pre = 90 * pre_link
    input_dim_tune = 90 * tune_link
    hidden_dim = 64 # 32 64 128
    encoder_n_layers = 3 # encoder稍微层数多一点，承担更多任务
    decoder_n_layers = 2 # decoder稍微简单一点，有利于encoder得到更好的信号表示
    # frozen_n_layer = encoder_n_layers - 1
    mask_ratio = 0.75
    output_dim = 10 # 输出类别

    train_losses = []  # 用来记录训练过程中每个epoch的训练损失
    val_losses = [] # 用来记录训练过程中每个epoch的验证损失
    train_accuracy = [] # 用来记录训练过程中每个epoch的验证准确度
    val_accuracy =[] # 用来记录训练过程中每个epoch的验证准确度

    init_lr = 0.0003 # 学习率 0.0004
    final_lr = 0.00007 # 最终学习率
    init_weight_decay = 2e-3 # 衰减系数 0.002
    num_epochs = 50 # 先用10轮进行训练
    batch_size = 64 # batch大小
    data_folder = './dataset/10fenlei(5000)' # 数据集路径
    # data_folder = '/mnt/data/keran/project/Flow-LLM/FAE/dataset/10fenlei'
    # data_folder = '/mnt/data/keran/project/Flow-LLM/FAE/dataset/XRF55_HAR'


    

    # **检查是否存在已保存的 checkpoint**
    start_epoch = 0  # 记录从哪个 epoch 开始训练


    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # 加载数据
    dataset = ComplexDataset(data_folder)

    # 数据集分割
    train_size = int(0.9 * len(dataset) * tune_datasize)
    val_size = len(dataset) - train_size #验证集 查看收敛情况和loss
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    

    
    # 加载训练集和测试集
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8,  pin_memory=True, shuffle=True, drop_last=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8,  pin_memory=True, shuffle=False)


    # 初始化模型
    model_pre = WiSRL_pre(
        input_dim=input_dim_pre,
        hidden_dim=hidden_dim,
        bimamba_type = bimamba_type, 
        encoder_nlayers=encoder_n_layers,
        decoder_nlayers=decoder_n_layers,
        mask_ratio=mask_ratio
    ).to(device)


    ## 定义微调模型
    model_tune = WiSRL_tune(
        original_model=model_pre,
        # input_dim=input_dim_tune,
        hidden_dim=hidden_dim,
        output_dim=output_dim
    ).to(device)

    # 定义优化器 AdamW
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model_tune.parameters()), lr=init_lr, weight_decay=init_weight_decay) # 使用AdamW优化器，防止梯度爆炸

    
    ## 定义调度器 scheduler，保证学习率
    scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[
        torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=5),  # 5 轮 Warm-up
        torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=10),  # 10 轮 固定学习率
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=35, eta_min = final_lr)  # 余弦退火
    ],
    milestones=[5, 15]  # 5 轮后进入 Hold-on，再 10 轮后进入退火
    )

    # if os.path.exists(old_Tune_final_weight_path):
    #     Tune_checkpoint = torch.load(old_Tune_final_weight_path, map_location=device)
    #     model_tune.load_state_dict(Tune_checkpoint)
    #     start_epoch = start_epoch + 1
    #     print(f"加载 checkpoint 成功！从 epoch {start_epoch} 继续训练")

    # **如果 checkpoint 存在，则加载**
    if os.path.exists(old_Tune_checkpoint_path):
        Tune_checkpoint = torch.load(old_Tune_checkpoint_path, map_location=device)
        model_tune.load_state_dict(Tune_checkpoint['model_state_dict'])
        optimizer.load_state_dict(Tune_checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(Tune_checkpoint['scheduler_state_dict'])
        start_epoch = Tune_checkpoint['epoch'] + 1  # 继续训练
        print(f"加载 checkpoint 成功！从 epoch {start_epoch} 继续训练")

    

    # model_train = torch.compile(model_tune)  
    model_train = model_tune # 训练模型
    model_eval = model_tune  # 评估时用未编译的模型
    
    # 定义损失函数
    criterion = nn.CrossEntropyLoss() 
    # criterion = FocalLoss(gamma=2)


    ## 启用tensorboard记录日志，方便后续可视化
    writer = SummaryWriter(log_dir=log_dir_path)  # log_dir 用于存放日志，便于后期可视化

    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


    best_accuracy = 0
    # train 
    # 要好好修改一下maybe，期望能够在每个世代训练后给出一个平均loss，方便观察收敛情况
    for epoch in range(start_epoch, num_epochs):
        model_train.train()
        epoch_train_loss = 0
        epoch_train_accuracy = 0
        print(f"Epoch [{epoch+1}/{num_epochs}]训练开始")
        # print(f"Epoch [{epoch+1}/{num_epochs}]")
        for batch_idx, (amplitude, phase, label) in enumerate(train_loader):
            amplitude = amplitude.to(torch.float32).to(device)
            phase = phase.to(torch.float32).to(device)
            label = label.to(torch.long).to(device) 
            label = label - 13 

            # 使用 label_map 进行转换
            # label = torch.tensor([label_map[l.item()] for l in label], dtype=torch.long, device=device)


            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}]")

            logits = model_train(amplitude, phase)
            # logits = model_train(amplitude)
            # print("logits shape",logits.shape)

            train_loss = criterion(logits, label)

            print(f"Train Loss = [{train_loss}]")
            epoch_train_loss += train_loss.item()

            # 记录 batch 级别的 loss
            writer.add_scalar("Loss/train_batch", train_loss.item(), epoch * len(train_loader) + batch_idx)

            probabilities = F.softmax(logits, dim=1)  # 计算概率
            predictions = torch.argmax(probabilities, dim=1)  # 取最大概率的类别

            # predictions = [inverse_label_map[pred.item()] for pred in predictions]

            correct = (predictions == label).sum().item()  # 统计正确个数
            total = label.size(0)  # 统计总样本数
            accuracy = correct / total
            epoch_train_accuracy += accuracy

            print(f"Train Accuracy: {accuracy:.4f}")

            # 记录 batch 级别的 accuracy
            writer.add_scalar("Accuracy/train_batch", accuracy, epoch * len(train_loader) + batch_idx)

            optimizer.zero_grad()
            train_loss.backward()

            ## 梯度裁剪
            # parameters = [p for p in model_train.parameters() if p.requires_grad]
            # torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)

            optimizer.step()


        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        writer.add_scalar("Loss/train_epoch", avg_train_loss, epoch)

        avg_train_accuracy = epoch_train_accuracy / len(train_loader)
        train_accuracy.append(avg_train_accuracy)
        writer.add_scalar("Accuracy/train_epoch", avg_train_accuracy, epoch)

        # Validation 验证模型性能
        model_eval.eval()  # 评估模式
        epoch_val_loss = 0
        epoch_val_accuracy = 0
        with torch.no_grad():  # 关闭梯度计算，加速运算
            for batch_idx, (amplitude, phase, label) in enumerate(val_loader):
                amplitude = amplitude.to(torch.float32).to(device)
                phase = phase.to(torch.float32).to(device)
                label = label.to(torch.long).to(device)
                label = label - 13

                # 使用 label_map 进行转换
                # label = torch.tensor([label_map[l.item()] for l in label], dtype=torch.long, device=device)

                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(val_loader)}]")

                logits = model_eval(amplitude, phase)
                # logits = model_eval(amplitude)

                val_loss = criterion(logits, label)
                print(f"Validation Loss = [{val_loss}]")
                
                epoch_val_loss += val_loss.item()

                probabilities = F.softmax(logits, dim=1)  # 计算概率
                predictions = torch.argmax(probabilities, dim=1)  # 取最大概率的类别

                # predictions = [inverse_label_map[pred.item()] for pred in predictions]

                correct = (predictions == label).sum().item()  # 统计正确个数
                total = label.size(0)  # 统计总样本数
                accuracy = correct / total
                epoch_val_accuracy += accuracy

                print(f"Validation Accuracy: {accuracy:.4f}")


    
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        writer.add_scalar("Loss/val_epoch", avg_val_loss, epoch)

        avg_val_accuracy = epoch_val_accuracy / len(val_loader)
        val_accuracy.append(avg_val_accuracy)
        writer.add_scalar("Accuracy/val_epoch", avg_val_accuracy, epoch)

        print(f"Epoch {epoch+1}: Train Loss = {train_losses[-1]}, Val Loss = {val_losses[-1]}, Train Accuracy = {train_accuracy[-1]},Val Accuracy = {val_accuracy[-1]}")

        # 记录学习率
        current_lr = scheduler.optimizer.param_groups[0]['lr']
        writer.add_scalar("Learning Rate", current_lr, epoch)
        print(f"Epoch {epoch+1}: 当前学习率 = {current_lr}")

        if best_accuracy <= avg_val_accuracy:
            best_accuracy = avg_val_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_tune.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, new_Tune_checkpoint_path)
            print(f"已保存微调checkpoint")
        print(f"当前最佳精度: best_accuracy = {best_accuracy}")
        
        ## 保存该轮微调模型参数及状态
        # torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': model_tune.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'scheduler_state_dict': scheduler.state_dict(),
        #     }, new_Tune_checkpoint_path)
        # print(f"已保存微调checkpoint")

        # **更新学习率**
        scheduler.step()
        # scheduler.step()


    writer.close()
    # # **保存loss**
    # np.save('./loss_record/train_loss_U1_3_2_50.npy', np.array(train_losses))  # 保存训练损失
    # np.save('./loss_record/val_loss_U1_3_2_50.npy', np.array(val_losses))  # 保存训练损失

    ## **保存模型**
    # torch.save(model_tune.state_dict(), Tune_final_weight_path)
    print("Wi-Mamba_scratch_finetune saved successfully.")


if __name__ == "__main__":
   fine_tuning()