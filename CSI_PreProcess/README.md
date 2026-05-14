# Widar3.0 数据处理说明

## Widar3.0 数据集基本信息

Widar3.0 是一个面向 WiFi 感知与手势识别的公开数据集，核心信号形式包括：

- CSI（Channel State Information）
- BVP（Body-coordinate Velocity Profile）
- DFS（Doppler Frequency Spectrum）

在本项目中，主要使用 CSI 原始数据进行统一命名与预处理，随后生成可直接用于模型训练的复数张量样本。

官方地址：

https://tns.thss.tsinghua.edu.cn/widar3.0/

## 本项目数据处理操作顺序

### 1. 准备原始数据

将原始 CSI 文件放入 [CSI_raw](CSI_raw) 目录，目录结构按采集日期划分（如 20181109、20181211 等），文件名格式为：

user_id-label-torso_loc-face_dir-rep_num-rx_id.dat

### 2. 统一标签与重命名

运行 [Rename_CSI.py](Rename_CSI.py)，完成以下操作：

- 按日期子集映射 room 编号（room_map）
- 按日期子集映射手势标签到统一编号（label_map）
- 跳过已知空文件样本（empty_files 及其同样本的全部接收器）
- 若目标文件重名，自动递增 rep_num 直到不冲突

输出到 [CSI](CSI)，目标文件格式为：

user_id-new_label-torso_loc-face_dir-rep_num-rx_id-roomx.dat

同时会在 [CSI](CSI) 目录生成统计与审计信息（如 data_count.txt、rename_audit.txt）。

### 3. 生成模型输入样本

运行 [run_prepare_widar3_csi.m](run_prepare_widar3_csi.m)，完成以下操作：

- 从 [CSI](CSI) 读取重命名后的 .dat 文件
- 按样本键 user_id-label-torso_loc-face_dir-rep_num-roomx 聚合 r1 到 r6
- 对每个接收器执行共轭乘法、幅相归一化、时间对齐
- 将结果保存为 .mat 文件到 [CSI_pro](CSI_pro)

输出文件格式为：

user_id-label-torso_loc-face_dir-rep_num-6-link-roomx.mat

每个文件中包含变量 CSI_result，形状为 1450 x 90 x 6（复数）。

### 4. 查看处理结果汇总

处理完成后可查看 [CSI_pro/summary.txt](CSI_pro/summary.txt)：

- 总样本数
- 成功样本数（新保存 + 已存在跳过）
- 失败样本数与失败文件名

## 快速执行顺序

1. 运行 [Rename_CSI.py](Rename_CSI.py)
2. 运行 [run_prepare_widar3_csi.m](run_prepare_widar3_csi.m)
3. 检查 [CSI_pro](CSI_pro) 与 [CSI_pro/summary.txt](CSI_pro/summary.txt)
