# import os
# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader
# from scipy.io import loadmat  # 用于加载 .mat 文件
# import torch.nn as nn
# import torch.optim as optim

# # 复数数据集类
# class ComplexDataset(Dataset):
#     def __init__(self, data_folder, transform=None):
#         """
#         数据集加载类，将每个样本从复数矩阵转换为幅值和相位，并进行归一化处理。

#         参数:
#             data_folder (str): 存储样本文件的文件夹路径
#             transform (callable, optional): 可选的图像转换操作（如数据增强）
#         """
#         self.data_folder = data_folder
#         self.transform = transform
#         self.file_paths = [os.path.join(data_folder, fname) for fname in os.listdir(data_folder) if fname.endswith('.mat')]  # 假设每个文件都是 mat 格式

#     def __len__(self):
#         """返回数据集中的样本数量"""
#         return len(self.file_paths)

#     def __getitem__(self, idx):
#         """根据索引返回单个样本的幅值和相位以及标签"""
#         file_path = self.file_paths[idx]
        
#         # 加载 .mat 文件
#         data = loadmat(file_path)
        
#         # 假设复数矩阵存储在 'matrix' 键下
#         complex_data = data['CSI_result'] ## error!
#         complex_data = complex_data.reshape(complex_data.shape[0], -1)


#         complex_data_1=complex_data[:, :15]
#         complex_data_2=complex_data[:, 15:30]
        
#         # 计算幅值和相位
#         real = np.real(complex_data_1)
#         imag = np.imag(complex_data_1)

#         # 按第三维度（6维）进行归一化
#         # amplitude_min = amplitude.min(axis=(0, 1), keepdims=True)  # 在整个数据范围内找到最小值
#         # amplitude_max = amplitude.max(axis=(0, 1), keepdims=True)
#         # amplitude = (amplitude - amplitude_min) / (amplitude_max - amplitude_min + 1e-8)  # 避免除零 
#         # print(amplitude.shape)
        
#         # phase = (phase + np.pi) / (2 * np.pi)  # 相位归一化到 [0, 1]
#         # print(phase.shape)

#         # todo：实现 amplitude 和 phase 后两维结合，将维度从（1450，90，6）转化成（1450，540）amplitude.shape[1]


#         real_mean = np.mean(real)
#         real_std = np.std(real)
#         real = (real-real_mean)/(real_std + 1e-8)

#         imag_mean = np.mean(imag)
#         imag_std = np.std(imag)
#         imag = (imag-imag_mean)/(imag_std + 1e-8)


#         # 提取 label
#         label = np.abs(complex_data_2)
#         # 归一化
#         # label_min = label.min(axis=(0, 1), keepdims=True)  # 在整个数据范围内找到最小值
#         # label_max = label.max(axis=(0, 1), keepdims=True)
#         # label = (label - label_min) / (label_max - label_min + 1e-8)  # 避免除零
        

#         mean = np.mean(label)
#         std = np.std(label)
#         label = (label-mean)/(std + 1e-8)

#         # 如果有 transform（如数据增强），应用它
    

        
#         # # 返回幅值、相位和标签
#         return real,imag, label
    


import os
import numpy as np
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from scipy.io import loadmat

def downsample_signal(real, imag, new_length):

    # numpy 转 tensor，float32
    real = torch.tensor(real, dtype=torch.float32)
    imag = torch.tensor(imag, dtype=torch.float32)
    
    # 转成 (N=1, C, L)
    real = real.permute(1, 0).unsqueeze(0)  # (1, C, L)
    imag = imag.permute(1, 0).unsqueeze(0)  # (1, C, L)
    
    # 1D下采样时间维度
    real_down = F.interpolate(real, size=new_length, mode='linear')  # (1, C, new_length)
    imag_down = F.interpolate(imag, size=new_length, mode='linear')  # (1, C, new_length)

    # 转回 numpy，(new_length, C)
    real_down = real_down.squeeze(0).permute(1, 0).cpu().numpy()
    imag_down = imag_down.squeeze(0).permute(1, 0).cpu().numpy()
    
    return real_down, imag_down

# 回归预测任务Dataset
class ComplexDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        self.data_folder = data_folder
        self.transform = transform
        self.file_paths = [os.path.join(data_folder, fname) for fname in os.listdir(data_folder) if fname.endswith('.mat')]


    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        data = loadmat(file_path)
        complex_data = data['CSI_result']

        # 3维信号（1450*90*6）
        real = np.real(complex_data)
        imag = np.imag(complex_data)


        # 最后一维压缩为540
        real = real.reshape(real.shape[0], -1)
        imag = imag.reshape(imag.shape[0], -1)

        # 降采样到L=512
        real, imag = downsample_signal(real, imag, new_length=512)

        # 切分输入和预测部分
        real_tar = real[:,15:30]
        imag_tar = imag[:,15:30]
        real = real[:,:15]
        imag = imag[:,:15]

        
        # 拼接实虚部做输入部分的统一归一化
        combined = np.concatenate((real, imag), axis=1)  # 2维，axis=1是天线+子载波+接收机维度

        # 求上频带的 mean/std
        mean = combined.mean()   
        std  = combined.std()
        
        # 标准化
        combined_norm = (combined - mean) / (std + 1e-8)

        # 拆回实部虚部
        half = combined.shape[1] // 2
        real = combined_norm[:, :half]
        imag = combined_norm[:, half:]



        # 预测部分的归一化
        real_tar = (real_tar - mean) / (std + 1e-8)
        imag_tar = (imag_tar - mean) / (std + 1e-8)

        # 转为Tensor
        real = torch.tensor(real, dtype=torch.float32)
        imag = torch.tensor(imag, dtype=torch.float32)
        real_tar = torch.tensor(real_tar, dtype=torch.float32)
        imag_tar = torch.tensor(imag_tar, dtype=torch.float32)

        if self.transform:
            real = self.transform(real)
            imag = self.transform(imag)
            real_tar = self.transform(real_tar)
            imag_tar = self.transform(imag_tar)
        
        return real, imag, real_tar, imag_tar

    @staticmethod
    def denormalize(real, imag, mean, std):
        """
        反归一化，将归一化后的实部和虚部恢复为原始值。
        
        参数：
            real_norm (np.ndarray): 归一化后的实部
            imag_norm (np.ndarray): 归一化后的虚部
            mean (float): 样本归一化时的均值
            std (float): 样本归一化时的标准差
        
        返回：
            real (np.ndarray): 反归一化后的实部
            imag (np.ndarray): 反归一化后的虚部
        """
        real = real * (std + 1e-8) + mean
        imag = imag * (std + 1e-8) + mean
        return real, imag