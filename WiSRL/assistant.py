import os

def count_files_in_directory(directory):
    # 统计文件夹中的文件数量
    file_count = 0
    # 获取目录下的所有内容
    for filename in os.listdir(directory):
        # 构造文件的完整路径
        file_path = os.path.join(directory, filename)
        # 检查是否是文件
        if os.path.isfile(file_path):
            file_count += 1
    return file_count

# 使用示例
directory_path = './dataset/10fenlei(5000)'  # 替换成你文件夹的路径
print(f"文件夹 '{directory_path}' 中的文件数量是: {count_files_in_directory(directory_path)}")


# import os
# import numpy as np
# from scipy.io import loadmat, savemat


# def is_all_zero_sample(sample: np.ndarray) -> bool:
#     """判断一个样本是否全 0"""
#     return np.all(sample == 0)


# def clean_single_mat(
#     mat_path: str,
#     output_path: str,
#     data_key: str = "CSI_result"
# ):
#     data = loadmat(mat_path)

#     if data_key not in data:
#         print(f"[SKIP] {mat_path}: key '{data_key}' not found")
#         return

#     csi = data[data_key]

#     # -------- 单样本 --------
#     if csi.ndim == 2:
#         if is_all_zero_sample(csi):
#             print(f"[REMOVE FILE] {mat_path}: single sample is all zero")
#             return
#         savemat(output_path, data)
#         return

#     # -------- 多样本 --------
#     if csi.ndim == 3:
#         if is_all_zero_sample(csi):
#             print(f"[REMOVE FILE] {mat_path}: single sample is all zero")
#             return
#         savemat(output_path, data)
#         return


# def clean_mat_folder(
#     input_dir: str,
#     output_dir: str,
#     data_key: str = "CSI_result"
# ):
#     os.makedirs(output_dir, exist_ok=True)

#     mat_files = [
#         f for f in os.listdir(input_dir)
#         if f.endswith(".mat")
#     ]

#     print(f"Found {len(mat_files)} .mat files in {input_dir}")

#     for fname in mat_files:
#         in_path = os.path.join(input_dir, fname)
#         out_path = os.path.join(output_dir, fname)
#         clean_single_mat(in_path, out_path, data_key)


# if __name__ == "__main__":
#     # ========= 修改这里 =========
#     # input_dir = "/mnt/data/keran/project/Flow-LLM/FAE/dataset/WIDAR_Pre"
#     # output_dir = "/mnt/data/keran/project/Flow-LLM/FAE/dataset/WIDAR_Pre_cleaned"
#     input_dir = "./dataset/10fenlei"
#     output_dir = "./dataset/10fenlei_cleaned"
#     data_key = "CSI_result"

#     clean_mat_folder(
#         input_dir=input_dir,
#         output_dir=output_dir,
#         data_key=data_key
#     )

