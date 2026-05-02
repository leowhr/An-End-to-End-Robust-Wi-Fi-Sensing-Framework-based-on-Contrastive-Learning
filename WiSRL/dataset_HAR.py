import os
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch


def downsample_signal(data, new_length):
    """
    data: np.ndarray, shape (T, D) = (1000, 270)
    return: np.ndarray, shape (new_length, D)
    """
    # numpy -> torch
    data = torch.tensor(data, dtype=torch.float32)

    # (T, D) -> (1, D, T)
    data = data.permute(1, 0).unsqueeze(0)

    # 下采样时间维
    data_down = F.interpolate(
        data,
        size=new_length,
        mode='linear',
        align_corners=False
    )

    # (1, D, new_length) -> (new_length, D)
    data_down = data_down.squeeze(0).permute(1, 0).cpu().numpy()

    return data_down



class AmplitudeDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        self.data_folder = data_folder
        self.transform = transform

        # 只加载 .npy 文件
        self.file_paths = [
            os.path.join(data_folder, fname)
            for fname in os.listdir(data_folder)
            if fname.endswith('.npy')
        ]
        

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]

        # (1000, 270)
        data = np.load(file_path).astype(np.float32)

        # # # ===== 下采样 =====
        data = downsample_signal(data, new_length=512)  # (512, 270)

        data=data[:,:90]

        # ===== 均值-方差标准化（对整个样本）=====
        mean = data.mean()
        std = data.std()

        data_norm = (data - mean) / (std + 1e-8)

        # label 提取（按你之前的规则改）
        filename = os.path.basename(file_path)
        # 示例：xxx_3_xxx.npy → 3
        label_str = filename.split('_')[1]
        label = int(label_str) - 1

        if self.transform:
            data_norm = self.transform(data_norm)

        # 返回幅值 + label
        return data_norm, data_norm, label



    @staticmethod
    def denormalize(data_norm, mean, std):
        """
        反归一化
        """
        return data_norm * (std + 1e-8) + mean