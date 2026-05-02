# from scipy.io import loadmat
# import os

# # 拆分文件所在路径
# path = "/mnt/data/keran/LKR/LLM-GEN/RF-Painter/dataset/10fenlei"

# # 任选一个文件看看，比如第一个
# files = [f for f in os.listdir(path) if f.endswith(".mat")]
# files.sort()
# if not files:
#     print("⚠️ 没有找到 .mat 文件")
# else:
#     file_to_check = os.path.join(path, files[0])
#     print(f"检查文件: {file_to_check}")

#     data = loadmat(file_to_check)

#     # 显示文件中包含的键
#     print("\n变量名：", [k for k in data.keys() if not k.startswith("__")])

#     # 查看CSI_result的形状和类型
#     csi = data["CSI_result"]
#     print("CSI_result 形状：", csi.shape)
#     print("CSI_result 数据类型：", csi.dtype)
#     print("前几个数值：\n", csi[:3, :3])

from scipy.io import loadmat
import os
import numpy as np

path = "/mnt/data/keran/LKR/LLM-GEN/RF-Painter/dataset/10fenlei"

files = [f for f in os.listdir(path) if f.endswith(".mat")]
files.sort()
if not files:
    print("⚠️ 没有找到 .mat 文件")
else:
    file_to_check = os.path.join(path, files[0])
    print(f"检查文件: {file_to_check}")

    data = loadmat(file_to_check)

    print("\n顶层变量名：", [k for k in data.keys() if not k.startswith("__")])

    csi = data["CSI_result"]
    print("CSI_result 形状：", csi.shape)
    print("CSI_result 数据类型：", csi.dtype)

    # 如果CSI_result是结构体或包含字段，打印字段名
    if csi.dtype.names is not None:
        print("CSI_result 内部字段：", csi.dtype.names)

    real_part = np.real(csi)
    imag_part = np.imag(csi)
    print("实部 shape:", real_part.shape)
    print("实部 前几个数值：\n", real_part[:3, :3])
    print("虚部 shape:", imag_part.shape)
    print("虚部 前几个数值：\n", imag_part[:3, :3])

    magnitude = np.abs(csi)
    print("幅值 shape:", magnitude.shape)
    print("幅值 前几个数值：\n", magnitude[:3, :3])

    with np.errstate(divide='ignore', invalid='ignore'):
        tan_val = np.true_divide(imag_part, real_part)
        tan_val[~np.isfinite(tan_val)] = 0
    print("tan shape:", tan_val.shape)
    print("tan 前几个数值：\n", tan_val[:3, :3])