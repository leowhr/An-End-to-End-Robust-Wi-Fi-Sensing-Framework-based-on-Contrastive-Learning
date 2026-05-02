import os
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


from dataset_HAR import AmplitudeDataset

from models.pretraining_AMP_model import WiSRL_pre # 只恢复幅值

from models.finetune_AMP_model import WiSRL_tune # 只输入幅值微调训练

def laplace_noise(x, level, device):
    """
    生成拉普拉斯噪声，形状和x相同，尺度由level控制。
    """
    laplace = torch.distributions.Laplace(0.0, level)
    noise = laplace.sample(x.shape).to(device)
    return noise


def sample_t_ratio(n: int, ratio=None, device=None):
    t = torch.zeros(n, device=device) + ratio
    return  t 

def noise_scheduling(x_real, ratio, level, device):

    B = x_real.size(0)

    t = sample_t_ratio(B, ratio, device=device)

    t1 = t.view(B, 1, 1)

    e = torch.randn_like(x_real, device=device) * level # 高斯噪声
    # e = (torch.rand_like(x_real, device=device) - 0.5) * 2 * level # 均匀噪声
    # e = laplace_noise(x_real, level, device) # 拉普拉斯噪声
# 
    
    z_real = t1 * x_real + (1-t1) * e

    return z_real



def WiSRL_inference():

    seed = 624  # 随机种子，可以换个数字试试
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # shape: (n, l ,d)
    
    input_dim = 270
    hidden_dim = 64 # 32 64 128
    bimamba_type = "v2"
    encoder_n_layers = 3 # encoder稍微层数多一点，承担更多任务
    decoder_n_layers = 2 # decoder稍微简单一点，有利于encoder得到更好的信号表示
    mask_ratio = 0.75
    output_dim = 15 # 输出类别


    batch_size = 64 # batch大小
    data_folder = '/mnt/data/keran/project/Flow-LLM/FAE/dataset/XRF55_HAR'


    ## 微调试验模型权重
    Tune_checkpoint_path = "./model_weight/CLS/Finetune/tune_75_100_100_Biblockv2_both_har.pth" 



    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 加载数据
    all_dataset = AmplitudeDataset(data_folder)

    # 取验证集作为计算对象
    train_dataset_size = int(0.9 * len(all_dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(all_dataset, [train_dataset_size, int(0.1 * len(all_dataset))])

    
    # 加载训练集和测试集
    data_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8,  pin_memory=True, shuffle=False, drop_last=False)


    # 初始化模型
    model_pre = WiSRL_pre(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        bimamba_type = bimamba_type,
        encoder_nlayers=encoder_n_layers,
        decoder_nlayers=decoder_n_layers,
        mask_ratio=mask_ratio
    ).to(device)



    ## 定义微调模型
    model_tune = WiSRL_tune(
        original_model=model_pre,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim
    ).to(device)

    Tune_checkpoint = torch.load(Tune_checkpoint_path, map_location=device)
    # Tune_checkpoint = torch.load(Frozen_checkpoint_path, map_location=device)

    ## 定义可视化模型 
    model_tune.load_state_dict(Tune_checkpoint['model_state_dict'], strict=True)

    


    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


    val_predictions_list = []
    val_labels_list = []

    ratio = 1 # 加噪比例
    level = 4 # 噪声等级

    # Validation 验证模型性能
    model_tune.eval()  # 评估模式
    with torch.no_grad():  # 关闭梯度计算，加速运算
        for batch_idx, (real, label) in enumerate(data_loader):
            real = real.to(torch.float32).to(device)
            # imag = imag.to(torch.float32).to(device)
            label = label.to(torch.long).to(device)
            # label = label - 13 

            # 加噪
            n_real = noise_scheduling(real, ratio, level, device)

            logits = model_tune(n_real)

            probabilities = F.softmax(logits, dim=1)  # 计算概率
            predictions = torch.argmax(probabilities, dim=1)  # 取最大概率的类别
            
            correct = (predictions == label).sum().item()  # 统计正确个数
            total = label.size(0)  # 统计总样本数
            accuracy = correct / total
            print(f"Validation Accuracy: {accuracy:.4f}")

            # 存储数据
            val_predictions_list.append(predictions.cpu())  
            val_labels_list.append(label.cpu())


        # **转换为 NumPy 数组**
        val_labels = torch.cat(val_labels_list)
        val_predictions = torch.cat(val_predictions_list)

        val_correct = (val_predictions == val_labels).sum().item()  # 统计正确个数
        val_total = val_labels.size(0)  # 统计总样本数
        avg_val_accuracy = val_correct / val_total

        print(f"Val Accuracy = {avg_val_accuracy}") 




if __name__ == "__main__":
   WiSRL_inference()