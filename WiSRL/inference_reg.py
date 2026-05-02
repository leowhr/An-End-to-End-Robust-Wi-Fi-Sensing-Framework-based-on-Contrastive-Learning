import os
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


from dataset_reg import ComplexDataset
from models.pretraining_Biblock_model import WiSRL_pre
from models.finetune_reg_model import WiSRL_tune

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

def noise_scheduling(x_real, x_imag, ratio, level, device):

    B = x_real.size(0)


    x = torch.cat([x_real, x_imag], dim=-1)  # (B, L ,2C)

    t = sample_t_ratio(B, ratio, device=device)

    t1 = t.view(B, 1, 1)

    e = torch.randn_like(x, device=device) * level # 高斯噪声
    # e = (torch.rand_like(x, device=device) - 0.5) * 2 * level # 均匀噪声
    # e = laplace_noise(x, level, device) # 拉普拉斯噪声
# 
    
    z = t1 * x + (1-t1) * e

    half = z.shape[-1] // 2
    z_real = z[:,:,:half]
    z_imag = z[:,:,half:]


    return z_real, z_imag


def WiSRL_inference():

    seed = 1024  # 随机种子，可以换个数字试试
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # shape: (n, l ,d)
    
    input_dim = 15
    hidden_dim = 64 # 32 64 128
    bimamba_type = "v2"
    encoder_n_layers = 3 # encoder稍微层数多一点，承担更多任务
    decoder_n_layers = 2 # decoder稍微简单一点，有利于encoder得到更好的信号表示
    mask_ratio = 0.75


    batch_size = 64 # batch大小
    data_folder = './dataset/10fenlei' # 数据集路径


    ## 微调试验模型权重
    Tune_checkpoint_path = "./model_weight/REG/Finetune/tune_reg_75_100_60_Biblockv2_both_reg.pth" 



    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # 加载数据
    dataset = ComplexDataset(data_folder)

    # 取验证集作为计算对象
    # 数据集分割
    train_size = int(len(dataset) * 0.9)
    val_size = len(dataset) - train_size #验证集 查看收敛情况和loss
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    
    # 加载训练集和测试集
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8,  pin_memory=True, shuffle=False, drop_last=False)


    # 初始化模型
    model_pre = WiSRL_pre(
        input_dim= input_dim,
        hidden_dim=hidden_dim,
        bimamba_type = bimamba_type,
        encoder_nlayers=encoder_n_layers,
        decoder_nlayers=decoder_n_layers,
        mask_ratio=mask_ratio
    ).to(device)



    ## 定义微调模型
    model_tune = WiSRL_tune(
        original_model=model_pre,
        input_dim = input_dim,
        hidden_dim=hidden_dim
    ).to(device)

    Tune_checkpoint = torch.load(Tune_checkpoint_path, map_location=device)

    ## 定义可视化模型 
    model_tune.load_state_dict(Tune_checkpoint['model_state_dict'], strict=True)

    


    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False



    val_num = 0
    val_snr = 0


    ratio = 0.8 # 加噪比例
    level = 1 # 噪声等级

    
    # Validation 验证模型性能
    model_tune.eval()  # 评估模式
    with torch.no_grad():  # 关闭梯度计算，加速运算
        for batch_idx, (real, imag, real_tar,imag_tar) in enumerate(val_loader):
            real = real.to(torch.float32).to(device)
            imag = imag.to(torch.float32).to(device)
            real_tar = real_tar.to(torch.float32).to(device)
            imag_tar = imag_tar.to(torch.float32).to(device)

            B,*_= real.shape

            # 加噪
            n_real, n_imag = noise_scheduling(real, imag, ratio, level, device)

            pred = model_tune(n_real, n_imag)

            tar = torch.cat([real_tar,imag_tar],dim=-1)

            snr = cal_SNR(pred, tar)

            snr = snr.mean()
        
            val_snr += snr * B
            val_num += B
            print(f"Validation SNR: {snr}")


    avg_val_snr = val_snr / val_num
    print(f"Average Validation SNR: {avg_val_snr}")



def cal_SNR(predict, truth):
    if torch.is_tensor(predict):
        predict = predict.detach().cpu().numpy()
    if torch.is_tensor(truth):
        truth = truth.detach().cpu().numpy()
    #  Recombine the real and imaginary parts to form complex values
    half = predict.shape[-1] // 2
    predict_complex = (predict[:,:,:half] + 1j * predict[:,:,half:])
    truth_complex = (truth[:,:,:half] + 1j * truth[:,:,half:])
    PS = np.sum(np.abs(truth_complex)**2, axis=(1, 2))  # power of signal (B,)
    PN = np.sum(np.abs(predict_complex - truth_complex)**2, axis=(1, 2))  # power of noise (B,)

    # PS = np.sum(np.abs(truth)**2, axis=(1, 2))  # power of signal (B,)
    # PN = np.sum(np.abs(predict - truth)**2, axis=(1, 2))  # power of noise (B,)
    ratio = PS / (PN + 1e-8)  # (B,)
    SNR = 10 * np.log10(ratio)
    return SNR

if __name__ == "__main__":
   WiSRL_inference()