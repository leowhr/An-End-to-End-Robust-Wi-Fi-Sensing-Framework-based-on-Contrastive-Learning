# for finetuning
# 结果 97.4%

## 输入：幅值
## 输出：幅值

import os
import numpy as np
import random

from models.pretraining_Biblock_model_simCLR import WiSRL_pre # Bi-Block
from utils import nt_xent_loss, set_seed, seed_worker

from models.finetune_model_simCLR import WiSRL_tune


import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataset_simCLR import ComplexDataset
# from dataset_HAR import AmplitudeDataset
from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
## 下游任务微调
def fine_tuning():
    bimamba_type = "v2"
    tune_datasize = 1.0
    pre_link = 6
    tune_link = 6
    Pre_checkpoint_path = "./model_weight/PRE/pre_100_Biblockv2_3+3link_test.pth" #预训练权重
    # Pre_checkpoint_path = "./model_weight/PRE/pre_75_100_Biblockv2_widar_both_1link.pth" #预训练权重
    # Tune_checkpoint_path = "./model_weight/CLS/Finetune/tune_75_100_Biblockv2_HGR(5000).pth" #预训练权重

    log_dir_path = "./runs/Finetune/CLS/tune_100_100_Biblockv2_3+3link_test" #日志

    ## 加载的数据

    old_Tune_checkpoint_path = "./model_weight/Finetune/CLS/tune_100_100_Biblockv2_3+3link_test.pth" # 上次训练某一轮保存的微调权重和优化器状态

    ## 保存的数据
    new_Tune_checkpoint_path = "./model_weight/Finetune/CLS/tune_100_100_Biblockv2_3+3link_test.pth" # 每一轮保存的微调权重和优化器状态

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    seed = 624 # 随机种子，可以换个数字试试
    set_seed(seed)  # 设置随机种子，保证结果可复现
    generator = torch.Generator()
    generator.manual_seed(seed)

    # shape: (n, l ,d)
    
    input_dim_pre = 90*3
    hidden_dim = 64 # 32 64 128
    encoder_n_layers = 3 # encoder稍微层数多一点，承担更多任务
    # frozen_n_layer = encoder_n_layers - 1
    
    output_dim = 10 # 输出类别

    train_losses = []  # 用来记录训练过程中每个epoch的训练损失
    val_losses = [] # 用来记录训练过程中每个epoch的验证损失
    train_accuracy = [] # 用来记录训练过程中每个epoch的验证准确度
    val_accuracy =[] # 用来记录训练过程中每个epoch的验证准确度

    init_lr = 0.0003 # 学习率 0.0003
    final_lr = 7e-05 # 最终学习率
    init_weight_decay = 2e-3 # 衰减系数 0.002
    num_epochs = 50 # 先用50轮进行训练
    batch_size = 64 # batch大小
    data_folder = '/mnt/data/keran/project/WiSRL/dataset/10fenlei(5000)' # 数据集路径
    # data_folder = '/mnt/data/keran/project/Flow-LLM/FAE/dataset/XRF55_HAR'
    # data_folder = '/mnt/data/keran/project/Flow-LLM/FAE/dataset/10fenlei'

    best_accuracy = 0

    # **检查是否存在已保存的 checkpoint**
    start_epoch = 0  # 记录从哪个 epoch 开始训练


    # 加载数据
    dataset = ComplexDataset(data_folder)
    dataset.set_eval(True)  # 设置为评估模式，确保数据不被增强

    # 数据集分割* tune_datasize,generator=generator
    train_size = int(0.9 * len(dataset) * tune_datasize)
    val_size = len(dataset) - train_size
    # generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    

    
    # 加载训练集和测试集
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8,  pin_memory=True, shuffle=True, drop_last=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8,  pin_memory=True, shuffle=False, drop_last=False)



    # 初始化模型
    model_pre = WiSRL_pre(
        input_dim=input_dim_pre,
        hidden_dim=hidden_dim,
        proj_dim=0,  # 微调阶段不使用投影头
        bimamba_type = bimamba_type,
        encoder_nlayers=encoder_n_layers
    ).to(device)


    Pre_checkpoint = torch.load(Pre_checkpoint_path, map_location=device)

    # # **恢复模型和优化器的参数**
    model_pre.load_state_dict(Pre_checkpoint['model_state_dict'], strict=False)
    # model_pre.load_state_dict(checkpoint)

    ## 定义微调模型
    model_tune = WiSRL_tune(
        original_model=model_pre,
        # frozen_n_layer = frozen_n_layer,
        hidden_dim=hidden_dim,
        output_dim=output_dim
    ).to(device)


    # 定义优化器 AdamW
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model_tune.parameters()), lr=init_lr, weight_decay=init_weight_decay) # 使用AdamW优化器，防止梯度爆炸

    
    # 定义调度器 scheduler，保证学习率
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
        best_accuracy = Tune_checkpoint['best_accuracy']
        print(f"加载 checkpoint 成功！从 epoch {start_epoch} 继续训练! 当前准确率为{best_accuracy}!")

    

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


   
    # train 
    # 要好好修改一下maybe，期望能够在每个世代训练后给出一个平均loss，方便观察收敛情况
    for epoch in range(start_epoch, num_epochs):
        torch.cuda.empty_cache()
        model_train.train()
        epoch_train_loss = 0
        epoch_train_accuracy = 0
        print(f"Epoch [{epoch+1}/{num_epochs}]训练开始")
        # print(f"Epoch [{epoch+1}/{num_epochs}]")
        for batch_idx, (amp_view1, pha_view1, amp_view2, pha_view2, label) in enumerate(train_loader):
            amp_view1 = amp_view1.to(torch.float32).to(device)
            pha_view1 = pha_view1.to(torch.float32).to(device)
            amp_view2 = amp_view2.to(torch.float32).to(device)
            pha_view2 = pha_view2.to(torch.float32).to(device)
            label = label.to(torch.long).to(device)
            label = label - 13

            # 使用 label_map 进行转换
            # label = torch.tensor([label_map[l.item()] for l in label], dtype=torch.long, device=device)


            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}]")

            # logits = model_train(real)
            logits = model_train(amp_view1, amp_view2, pha_view1, pha_view2)
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
        # epoch_val_accuracy = 0
        val_predictions_list = []
        val_labels_list = [] 

        with torch.no_grad():  # 关闭梯度计算，加速运算
            for batch_idx, (amp_view1, pha_view1, amp_view2, pha_view2, label) in enumerate(val_loader):
                amp_view1 = amp_view1.to(torch.float32).to(device)
                pha_view1 = pha_view1.to(torch.float32).to(device)
                amp_view2 = amp_view2.to(torch.float32).to(device)
                pha_view2 = pha_view2.to(torch.float32).to(device)
                label = label.to(torch.long).to(device)
                label = label - 13

                # 使用 label_map 进行转换
                # label = torch.tensor([label_map[l.item()] for l in label], dtype=torch.long, device=device)

                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(val_loader)}]")

                logits = model_eval(amp_view1, amp_view2, pha_view1, pha_view2)

                val_loss = criterion(logits, label)
                print(f"Validation Loss = [{val_loss}]")
                
                epoch_val_loss += val_loss.item()

                probabilities = F.softmax(logits, dim=1)  # 计算概率
                predictions = torch.argmax(probabilities, dim=1)  # 取最大概率的类别

                # predictions = [inverse_label_map[pred.item()] for pred in predictions]

                correct = (predictions == label).sum().item()  # 统计正确个数
                total = label.size(0)  # 统计总样本数
                accuracy = correct / total
                # epoch_val_accuracy += accuracy

                print(f"Validation Accuracy: {accuracy:.4f}")

                # 存储数据
                val_predictions_list.append(predictions.cpu())  
                val_labels_list.append(label.cpu())


    
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        writer.add_scalar("Loss/val_epoch", avg_val_loss, epoch)


        # **转换为 NumPy 数组**
        val_labels = torch.cat(val_labels_list)
        val_predictions = torch.cat(val_predictions_list)

        val_correct = (val_predictions == val_labels).sum().item()  # 统计正确个数
        val_total = val_labels.size(0)  # 统计总样本数
        avg_val_accuracy = val_correct / val_total

        val_accuracy.append(avg_val_accuracy)
        writer.add_scalar("Accuracy/val_epoch", avg_val_accuracy, epoch)

        # avg_val_accuracy = epoch_val_accuracy / len(val_loader)
        # val_accuracy.append(avg_val_accuracy)
        # writer.add_scalar("Accuracy/val_epoch", avg_val_accuracy, epoch)

        print(f"Epoch {epoch+1}: Train Loss = {train_losses[-1]}, Val Loss = {val_losses[-1]}, Train Accuracy = {train_accuracy[-1]},Val Accuracy = {val_accuracy[-1]}")

        

        # 记录学习率
        current_lr = scheduler.optimizer.param_groups[0]['lr']
        writer.add_scalar("Learning Rate", current_lr, epoch)
        print(f"Epoch {epoch+1}: 当前学习率 = {current_lr}")

        ## 保存该轮微调模型参数及状态
        if best_accuracy <= avg_val_accuracy:
            best_accuracy = avg_val_accuracy
            torch.save({
                'epoch': epoch,
                'best_accuracy':best_accuracy,
                'model_state_dict': model_tune.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, new_Tune_checkpoint_path)
            print(f"已保存微调checkpoint")
        print(f"当前最佳精度: best_accuracy = {best_accuracy}")


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
    print("Wi-Mamba_tune saved successfully.")


    




if __name__ == "__main__":
   fine_tuning()