import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat  # 用于加载 .mat 文件
import torch.nn as nn
import torch.optim as optim


def trim_and_resize_csi(
    csi: np.ndarray,
    target_len: int = 512,
    method: str = "interp"
):
    """
    csi: np.ndarray, shape [T, F] (e.g., 1450 x 540)
    target_len: target time length (default 512)
    method: 'interp' or 'sample'
    
    return: np.ndarray, shape [target_len, F]
    """
    assert csi.ndim == 2, "Input must be 2D array [T, F]"

    # ---------- 1. 去掉底部全 0 行 ----------
    # 找到最后一个非全 0 的行
    non_zero_rows = np.where(np.any(csi != 0, axis=1))[0]
    if len(non_zero_rows) == 0:
        raise ValueError("All rows are zero!")

    last_valid_row = non_zero_rows[-1]
    csi = csi[: last_valid_row + 1]

    T, F = csi.shape

    # ---------- 2. resize 到 target_len ----------
    if method == "interp":
        # 线性插值
        old_idx = np.linspace(0, 1, T)
        new_idx = np.linspace(0, 1, target_len)

        resized = np.empty((target_len, F), dtype=csi.dtype)
        for f in range(F):
            resized[:, f] = np.interp(new_idx, old_idx, csi[:, f])

    elif method == "sample":
        # 均匀采样（无插值）
        indices = np.linspace(0, T - 1, target_len)
        indices = np.round(indices).astype(int)
        resized = csi[indices]

    else:
        raise ValueError(f"Unknown method: {method}")

    return resized


# 复数数据集类
class ComplexDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        """
        数据集加载类，将每个样本从复数矩阵转换为幅值和相位，并进行归一化处理。

        参数:
            data_folder (str): 存储样本文件的文件夹路径
            transform (callable, optional): 可选的图像转换操作（如数据增强）
        """
        self.data_folder = data_folder
        self.transform = transform
        self.file_paths = [os.path.join(data_folder, fname) for fname in os.listdir(data_folder) if fname.endswith('.mat')]  # 假设每个文件都是 mat 格式

    def __len__(self):
        """返回数据集中的样本数量"""
        return len(self.file_paths)

    def __getitem__(self, idx):
        """根据索引返回单个样本的幅值和相位以及标签"""
        file_path = self.file_paths[idx]
        
        # 加载 .mat 文件
        data = loadmat(file_path)
        
        # 假设复数矩阵存储在 'matrix' 键下
        complex_data = data['CSI_result'] ## error!
        # complex_data = complex_data.reshape(complex_data.shape[0], -1)

        # complex_data = trim_and_resize_csi(
        #     complex_data,
        #     target_len=512,
        #     method="interp"   # 推荐
        # )

        # # 取不同link
        # complex_data=complex_data[:, :, :1]
        # complex_data = complex_data[:, :, [0, 2, 5]]
        
        # 计算幅值和相位
        amplitude = np.abs(complex_data)
        phase = np.angle(complex_data)


        # 按第三维度（6维）进行归一化
        amplitude_min = amplitude.min(axis=(0, 1), keepdims=True)  # 在整个数据范围内找到最小值
        amplitude_max = amplitude.max(axis=(0, 1), keepdims=True)
        amplitude = (amplitude - amplitude_min) / (amplitude_max - amplitude_min + 1e-8)  # 避免除零 
        # print(amplitude.shape)



    #     mean = amplitude.mean()   
    #     std  = amplitude.std()
        
    #    # 标准化
    #     amplitude = (amplitude - mean) / (std + 1e-8)


        
        phase = (phase + np.pi) / (2 * np.pi)  # 相位归一化到 [0, 1]
        # print(phase.shape)
        

        amplitude = amplitude.reshape(amplitude.shape[0], -1)
        phase = phase.reshape(phase.shape[0], -1)

        # 从文件名中提取 label
        label = int(file_path.split('-')[1])  # 假设 label 是文件名中第一个 '-' 后面的数字
        # filename = os.path.basename(file_path)  # 只取文件名，不带路径
        # label_str = filename.split('-')[1]  # 这样才是文件名中第2段
        # label = int(label_str)


        
        # # 返回幅值、相位和标签
        return amplitude, phase, label
    

# import os
# import numpy as np
# from torch.utils.data import Dataset
# import torch
# import torch.nn.functional as F
# from scipy.io import loadmat


# def downsample_signal(real, imag, new_length):

#     # numpy 转 tensor，float32
#     real = torch.tensor(real, dtype=torch.float32)
#     imag = torch.tensor(imag, dtype=torch.float32)
    
#     # 转成 (N=1, C, L)
#     real = real.permute(1, 0).unsqueeze(0)  # (1, C, L)
#     imag = imag.permute(1, 0).unsqueeze(0)  # (1, C, L)
    
#     # 1D下采样时间维度
#     real_down = F.interpolate(real, size=new_length, mode='linear')  # (1, C, new_length)
#     imag_down = F.interpolate(imag, size=new_length, mode='linear')  # (1, C, new_length)

#     # 转回 numpy，(new_length, C)
#     real_down = real_down.squeeze(0).permute(1, 0).cpu().numpy()
#     imag_down = imag_down.squeeze(0).permute(1, 0).cpu().numpy()
    
#     return real_down, imag_down

# class ComplexDataset(Dataset):
#     def __init__(self, data_folder, transform=None):
#         self.data_folder = data_folder
#         self.transform = transform
#         self.file_paths = [os.path.join(data_folder, fname) for fname in os.listdir(data_folder) if fname.endswith('.mat')]

#     def __len__(self):
#         return len(self.file_paths)

#     def __getitem__(self, idx):
#         file_path = self.file_paths[idx]
#         data = loadmat(file_path)
#         complex_data = data['CSI_result']

#         # 3维信号（1450*90*6） or (1450*90*1)
#         real = np.real(complex_data)
#         imag = np.imag(complex_data)

#         # 最后一维压缩为540
#         real = real.reshape(real.shape[0], -1)
#         imag = imag.reshape(imag.shape[0], -1)

#         # 降采样
#         real, imag = downsample_signal(real, imag, 512)

#         # 拼接实虚部做统一归一化
#         combined = np.concatenate((real, imag), axis=1)  # 2维，axis=1是天线+子载波+接收机维度

#         # 求 mean/std
#         mean = combined.mean()   
#         std  = combined.std()
        
#         # 标准化
#         combined_norm = (combined - mean) / (std + 1e-8)

#         # 拆回实部虚部
#         half = combined.shape[1] // 2
#         real = combined_norm[:, :half]
#         imag = combined_norm[:, half:]


#         # label提取
#         filename = os.path.basename(file_path)  # 只取文件名，不带路径
#         label_str = filename.split('-')[1]  # 取label
#         label = int(label_str)

#         if self.transform:
#             real = self.transform(real)
#             imag = self.transform(imag)

#         return real, imag, label

#     @staticmethod
#     def denormalize(real, imag, mean, std):
#         """
#         反归一化，将归一化后的实部和虚部恢复为原始值。
        
#         参数：
#             real_norm (np.ndarray): 归一化后的实部
#             imag_norm (np.ndarray): 归一化后的虚部
#             mean (float): 样本归一化时的均值
#             std (float): 样本归一化时的标准差
        
#         返回：
#             real (np.ndarray): 反归一化后的实部
#             imag (np.ndarray): 反归一化后的虚部
#         """
#         real = real * (std + 1e-8) + mean
#         imag = imag * (std + 1e-8) + mean
#         return real, imag
    

  