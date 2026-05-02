'''
使用optuna进行参数优化
'''

import os
import numpy as np
import random
from models.pretraining_Biblock_model import WiSRL_pre
from models.finetune_reg_model import WiSRL_tune

# from models.pretraining_AMP_model import WiSRL_pre
# from models.finetune_AMP_model import WiSRL_tune

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataset_reg import ComplexDataset
from torch.utils.tensorboard import SummaryWriter

# optuna
import optuna
from optuna.trial import TrialState

search_space = {
    # "hidden_dim": [32, 64, 128],
    "init_lr": [2e-4, 3e-4, 4e-4],  # 3种, 5e-4
    "final_lr": [6e-5, 7e-5, 8e-5, 9e-5, 1e-4],  # 5种
    "batch_size": [32, 64, 128]  # 3种
}

# 排除某些参数组合
excluded_combinations = [
    # {"init_lr": 3e-4, "final_lr": 6e-5, "batch_size": 128},
    # {"init_lr": 4e-4, "final_lr": 8e-5, "batch_size": 128},
    # {"init_lr": 3e-4, "final_lr": 1e-4, "batch_size": 64},
    # {"init_lr": 4e-4, "final_lr": 7e-5, "batch_size": 32},
    # {"init_lr": 4e-4, "final_lr": 9e-5, "batch_size": 32},
    # {"init_lr": 4e-4, "final_lr": 9e-5, "batch_size": 64},
    # {"init_lr": 4e-4, "final_lr": 1e-4, "batch_size": 64},
    # {"init_lr": 5e-4, "final_lr": 9e-5, "batch_size": 32},
    # {"init_lr": 5e-4, "final_lr": 8e-5, "batch_size": 128},
    # {"init_lr": 4e-4, "final_lr": 6e-5, "batch_size": 128},
    # {"init_lr": 5e-4, "final_lr": 1e-4, "batch_size": 64},
    # {"init_lr": 5e-4, "final_lr": 7e-5, "batch_size": 64},
    # {"init_lr": 5e-4, "final_lr": 9e-5, "batch_size": 128},
    # {"init_lr": 5e-4, "final_lr": 6e-5, "batch_size": 32},
    # {"init_lr": 4e-4, "final_lr": 1e-4, "batch_size": 128},
]

def is_excluded_combination(init_lr, final_lr, batch_size):
    for combination in excluded_combinations:
        if (combination["init_lr"] == init_lr and
            combination["final_lr"] == final_lr and
            combination["batch_size"] == batch_size):
            return True
    return False
# 3 × 3 × 5 = 45 种可能的组合

## 下游任务微调
def fine_tuning(trial):
    # 超参数搜索空间
    # hidden_dim = trial.suggest_categorical("hidden_dim", search_space["hidden_dim"])
    init_lr = trial.suggest_categorical("init_lr", search_space["init_lr"])
    final_lr = trial.suggest_categorical("final_lr", search_space["final_lr"])
    batch_size = trial.suggest_categorical("batch_size", search_space["batch_size"])
    
    ## 排除某些参数组合（后续删除！）
    if is_excluded_combination(init_lr, final_lr, batch_size):
        print(f"Skipping trial {trial.number}.")
        raise optuna.exceptions.TrialPruned()
    
    trial_log = "./runs/optuna_reg_75_100_100_Biblockv2_both/trials_log.txt"

    seed = 624  # 随机种子，可以换个数字试试
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # shape: (n, l ,d)
    
    input_dim = 540
    hidden_dim = 64 # 32 64 128 # change [32 64 128]
    bimamba_type = "v2"
    encoder_n_layers = 3 # encoder稍微层数多一点，承担更多任务
    decoder_n_layers = 2 # decoder稍微简单一点，有利于encoder得到更好的信号表示
    mask_ratio = 0.75

    train_losses = []  # 用来记录训练过程中每个epoch的训练损失
    val_losses = [] # 用来记录训练过程中每个epoch的验证损失
    train_mae = [] # 用来记录训练过程中每个epoch的验证准确度
    val_mae =[] # 用来记录训练过程中每个epoch的验证准确度

    # init_lr = 0.0004 # 学习率 0.0004 # change [1e-4 ~ 1e-2]
    # final_lr = 0.0001 # 最终学习率 # change [1e-5 ~ init_lr/2]
    init_weight_decay = 2e-3 # 衰减系数 0.002
    num_epochs = 50 # 先用10轮进行训练
    # batch_size = 64 # batch大小 # change [32 64 128]
    data_folder = './dataset/10fenlei' # 数据集路径

    Pre_checkpoint_path = "./model_weight/pre_75_100_Biblockv2_both.pth" #预训练权重

    log_dir_path = "./runs/optuna_reg_75_100_100_Biblockv2_both" #日志

    ## 加载的数据
    # old_Tune_final_weight_path = "./model_weight/wimamba_U1_4_2_final_100_tune100_finetuning.pth" # 上次训练最后的模型

    # old_Tune_checkpoint_path = "./model_weight/wimambassm_U1_3_2_best_100_10fenlei_tune50_epoch_finetune(lr-0.0004-0.0001).pth" # 上次训练某一轮保存的微调权重和优化器状态
    old_Tune_checkpoint_path = "" # 不加载上次训练某一轮保存的微调权重和优化器状态

    ## 保存的数据
    # Tune_final_weight_path = "./model_weight/wimambassm_U1_3_2_best_100_10fenlei_tune50_final_finetune(lr-0.0004-0.0001).pth" # 保存最后模型

    # new_Tune_checkpoint_path = "./model_weight/wimambassm_U1_3_2_best_100_10fenlei_tune50_epoch_finetune(lr-0.0004-0.0001).pth" # 每一轮保存的微调权重和优化器状态

    # **检查是否存在已保存的 checkpoint**
    start_epoch = 0  # 记录从哪个 epoch 开始训练


    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # 加载数据
    dataset = ComplexDataset(data_folder)

    # 数据集分割
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size #验证集 查看收敛情况和loss
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    

    
    # 加载训练集和测试集
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8,  pin_memory=True, shuffle=True, drop_last=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8,  pin_memory=True, shuffle=False)


    # 初始化模型
    model_pre = WiSRL_pre(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        bimamba_type = bimamba_type,
        encoder_nlayers=encoder_n_layers,
        decoder_nlayers=decoder_n_layers,
        mask_ratio=mask_ratio
    ).to(device)


    Pre_checkpoint = torch.load(Pre_checkpoint_path, map_location=device)

    # **恢复模型和优化器的参数**
    model_pre.load_state_dict(Pre_checkpoint['model_state_dict'], strict=True)
    # model_pre.load_state_dict(checkpoint)

    ## 定义微调模型
    model_tune = WiSRL_tune(
        original_model=model_pre,
        input_dim = input_dim,
        hidden_dim=hidden_dim
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

    

    # model_train = torch.compile(model_tune)  
    model_train = model_tune # 训练模型
    model_eval = model_tune  # 评估时用未编译的模型
    
    # 定义损失函数
    criterion = nn.MSELoss() 
    # criterion = FocalLoss(gamma=2)


    ## 启用tensorboard记录日志，方便后续可视化
    writer = SummaryWriter(log_dir=log_dir_path)  # log_dir 用于存放日志，便于后期可视化

    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    best_mae = 100
    best_Epoch = 0



    # train 
    # 要好好修改一下maybe，期望能够在每个世代训练后给出一个平均loss，方便观察收敛情况
    for epoch in range(start_epoch, num_epochs):
        torch.cuda.empty_cache()
        model_train.train()
        epoch_train_loss = 0
        epoch_train_mae = 0
        print(f"Epoch [{epoch+1}/{num_epochs}]训练开始")
        # print(f"Epoch [{epoch+1}/{num_epochs}]")
        for batch_idx, (amplitude, phase) in enumerate(train_loader):
            amplitude = amplitude.to(torch.float32).to(device)
            phase = phase.to(torch.float32).to(device)
            
            #.clone().detach()


            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}]")

            amp_reg, amp_label = model_train(amplitude, phase)

            train_loss = criterion(amp_reg, amp_label)

            mae = torch.mean(torch.abs(amp_reg - amp_label))


            print(f"Train Loss = [{train_loss}]")
            epoch_train_loss += train_loss.item()

            # 记录 batch 级别的 loss
            writer.add_scalar("Loss/train_batch", train_loss.item(), epoch * len(train_loader) + batch_idx)


            epoch_train_mae += mae.item()

            print(f"Train MAE: {mae:.4f}")

            # 记录 batch 级别的 MAE
            writer.add_scalar("MAE/train_batch", mae, epoch * len(train_loader) + batch_idx)

            optimizer.zero_grad()
            train_loss.backward()

            ## 梯度裁剪
            # parameters = [p for p in model_train.parameters() if p.requires_grad]
            # torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)

            optimizer.step()


        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        writer.add_scalar("Loss/train_epoch", avg_train_loss, epoch)

        avg_train_mae = epoch_train_mae / len(train_loader)
        train_mae.append(avg_train_mae)
        writer.add_scalar("MAE/train_epoch", avg_train_mae, epoch)

        # Validation 验证模型性能
        model_eval.eval()  # 评估模式
        epoch_val_loss = 0
        epoch_val_mae = 0
        with torch.no_grad():  # 关闭梯度计算，加速运算
            for batch_idx, (amplitude, phase) in enumerate(val_loader):
                amplitude = amplitude.to(torch.float32).to(device)
                phase = phase.to(torch.float32).to(device)

                #.clone().detach()

                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(val_loader)}]")

                amp_reg, amp_label = model_train(amplitude, phase)

                val_loss = criterion(amp_reg, amp_label)

                mae = torch.mean(torch.abs(amp_reg - amp_label))

                print(f"Validation Loss = [{val_loss}]")
                
                epoch_val_mae += mae.item()

                print(f"Validation MAE: {mae:.4f}")


    
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        writer.add_scalar("Loss/val_epoch", avg_val_loss, epoch)

        avg_val_mae = epoch_val_mae / len(val_loader)
        val_mae.append(avg_val_mae)
        writer.add_scalar("MAE/val_epoch", avg_val_mae, epoch)

        print(f"Epoch {epoch+1}: Train Loss = {train_losses[-1]}, Val Loss = {val_losses[-1]}, Train MAE = {train_mae[-1]},Val MAE = {val_mae[-1]}")

        

        # 记录学习率
        current_lr = scheduler.optimizer.param_groups[0]['lr']
        writer.add_scalar("Learning Rate", current_lr, epoch)
        print(f"Epoch {epoch+1}: 当前学习率 = {current_lr}")

        ## 保存该轮微调模型参数及状态
        if best_mae >= avg_val_mae:
            best_mae = avg_val_mae
            best_Epoch = epoch+1
        print(f"当前最小误差: best_mae = {best_mae}")


        # **更新学习率**
        scheduler.step()


    writer.close()
    # # **保存loss**
    # np.save('./loss_record/train_loss_U1_3_2_50.npy', np.array(train_losses))  # 保存训练损失
    # np.save('./loss_record/val_loss_U1_3_2_50.npy', np.array(val_losses))  # 保存训练损失

    ## **保存模型**
    # torch.save(model_tune.state_dict(), Tune_final_weight_path)
    print("Wi-Mamba_tune saved successfully.")
    with open (trial_log, "a") as f:
        f.write(f"Trial num: {trial.number}\n")
        f.write(f"  Best_MAE: {best_mae}\n")
        f.write(f"  Best_Epoch: {best_Epoch}\n")
        f.write(f"  Params: \n")
        f.write(f"    init_lr: {init_lr}\n")
        f.write(f"    final_lr: {final_lr}\n")
        f.write(f"    batch_size: {batch_size}\n\n")
    return best_mae


if __name__ == "__main__":
    # 创建Optuna study对象并优化
    sampler = optuna.samplers.GridSampler(search_space)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(fine_tuning, n_trials=45)

    # 输出结果
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))
    
    print("Best trial:")
    trial = study.best_trial
    
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    result_file_path = "./runs/optuna_reg_75_100_100_Biblockv2_both/rst.txt"
    with open(result_file_path, "w") as f:
        f.write("Study statistics: \n")
        f.write(f"  Number of finished trials: {len(study.trials)}\n")
        f.write(f"  Number of pruned trials: {len(pruned_trials)}\n")
        f.write(f"  Number of complete trials: {len(complete_trials)}\n")
        f.write(f"Best trial:\n")
        f.write(f"  Value: {trial.value}\n")
        f.write(f"  Params: \n")
        for key, value in trial.params.items():
            f.write(f"    {key}: {value}\n")
    # fine_tuning()