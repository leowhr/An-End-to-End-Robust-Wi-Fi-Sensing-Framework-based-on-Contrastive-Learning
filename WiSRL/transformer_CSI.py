
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




if __name__ == "__main__":

    tune_datasize = 0.6

    log_dir_path = "./runs/REG/Scratch/scratch_reg_60_Transformer" #日志

    ## 加载的数据
    # old_Tune_final_weight_path = "./model_weight/wimamba_U1_4_2_final_100_tune100_finetuning.pth" # 上次训练最后的模型

    old_Tune_checkpoint_path = "./model_weight/REG/Scratch/scratch_reg_60_Transformer.pth" # 上次训练某一轮保存的微调权重和优化器状态

    ## 保存的数据
    # Tune_final_weight_path = "./model_weight/wimambassm_U1_3_2_best_100_10fenlei_tune50_final_finetune(lr-0.0004-0.0001).pth" # 保存最后模型

    new_Tune_checkpoint_path = "./model_weight/REG/Scratch/scratch_reg_60_Transformer.pth" # 每一轮保存的微调权重和优化器状态


    seed = 1024  # 随机种子，可以换个数字试试
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # shape: (n, l ,d)
    
    hidden_dim = 64 # 32 64 128
    
    encoder_n_layers = 3 # encoder稍微层数多一点，承担更多任务
    decoder_n_layers = 2 # decoder稍微简单一点，有利于encoder得到更好的信号表示
   
    

    train_losses = []  # 用来记录训练过程中每个epoch的训练损失
    val_losses = [] # 用来记录训练过程中每个epoch的验证损失
    train_mae = [] # 用来记录训练过程中每个epoch的验证准确度
    val_mae =[] # 用来记录训练过程中每个epoch的验证准确度


    init_lr = 0.0004 # 学习率 0.0004
    final_lr = 0.0001 # 最终学习率
    init_weight_decay = 2e-3 # 衰减系数 0.002
    num_epochs = 50 # 先用10轮进行训练
    batch_size = 64 # batch大小
    data_folder = './dataset/10fenlei' # 数据集路径

    # **检查是否存在已保存的 checkpoint**
    start_epoch = 0  # 记录从哪个 epoch 开始训练


    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # 加载数据
    dataset = ComplexDataset(data_folder)

    # 数据集分割
    train_size = int(len(dataset) * tune_datasize)
    val_size = len(dataset) - train_size #验证集 查看收敛情况和loss
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    

    
    # 加载训练集和测试集
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8,  pin_memory=True, shuffle=True, drop_last=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8,  pin_memory=True, shuffle=False)


    # 初始化模型
    model = TransformerModel().to(device)





    # 定义优化器 AdamW
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=init_lr, weight_decay=init_weight_decay) # 使用AdamW优化器，防止梯度爆炸

    
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
        model.load_state_dict(Tune_checkpoint['model_state_dict'])
        optimizer.load_state_dict(Tune_checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(Tune_checkpoint['scheduler_state_dict'])
        start_epoch = Tune_checkpoint['epoch'] + 1  # 继续训练
        print(f"加载 checkpoint 成功！从 epoch {start_epoch} 继续训练")

    

    # model_train = torch.compile(model_tune)  
    model_train = model # 训练模型
    model_eval = model  # 评估时用未编译的模型



    ## 启用tensorboard记录日志，方便后续可视化
    writer = SummaryWriter(log_dir=log_dir_path)  # log_dir 用于存放日志，便于后期可视化

    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    criterion = nn.MSELoss()
    


    best_mae = 100
    # train 
    # 要好好修改一下maybe，期望能够在每个世代训练后给出一个平均loss，方便观察收敛情况
    for epoch in range(start_epoch, num_epochs):
        torch.cuda.empty_cache()
        model_train.train()
        epoch_train_loss = 0
        epoch_train_mae = 0
        print(f"Epoch [{epoch+1}/{num_epochs}]训练开始")
        # print(f"Epoch [{epoch+1}/{num_epochs}]")
        for batch_idx, (amplitude, phase, label) in enumerate(train_loader):
            amplitude = amplitude.to(torch.float32).to(device)
            phase = phase.to(torch.float32).to(device)
            label = label.to(torch.float32).to(device)
            
            #.clone().detach()


            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}]")

            amp_reg = model_train(amplitude)
            # amp_reg = model_train(amplitude)

            train_loss = criterion(amp_reg, label)
            epoch_train_loss += train_loss.item()
            print(f"Train Loss = [{train_loss}]")
            
            # 记录 batch 级别的 loss
            writer.add_scalar("Loss/train_batch", train_loss.item(), epoch * len(train_loader) + batch_idx)


            mae = torch.mean(torch.abs(amp_reg - label)).item()
            epoch_train_mae += mae
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
            for batch_idx, (amplitude, phase, label) in enumerate(val_loader):
                amplitude = amplitude.to(torch.float32).to(device)
                phase = phase.to(torch.float32).to(device)
                label = label.to(torch.float32).to(device)


                #.clone().detach()

                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(val_loader)}]")

                amp_reg = model_eval(amplitude)
                # amp_reg = model_eval(amplitude)

                val_loss = criterion(amp_reg, label)
                epoch_val_loss += val_loss.item()
                print(f"Validation Loss = [{val_loss}]")

                mae = torch.mean(torch.abs(amp_reg - label)).item()
                epoch_val_mae += mae
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
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, new_Tune_checkpoint_path)
            print(f"已保存微调checkpoint")
        print(f"当前最小误差: best_mae = {best_mae}")


        # **更新学习率**
        scheduler.step()


    writer.close()
    # # **保存loss**
    # np.save('./loss_record/train_loss_U1_3_2_50.npy', np.array(train_losses))  # 保存训练损失
    # np.save('./loss_record/val_loss_U1_3_2_50.npy', np.array(val_losses))  # 保存训练损失

    ## **保存模型**
    # torch.save(model_tune.state_dict(), Tune_final_weight_path)
    print("Transformer_reg saved successfully.")