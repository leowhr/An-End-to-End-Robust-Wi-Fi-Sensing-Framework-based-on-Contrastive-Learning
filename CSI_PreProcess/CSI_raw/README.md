# Widar3.0 数据集简介

Widar3.0 是一个面向 WiFi 感知与手势识别的公开数据集，包含基于商用 WiFi 设备采集的多类型信号数据。常用数据形式包括 CSI（Channel State Information）、BVP（Body-coordinate Velocity Profile）和 DFS（Doppler Frequency Spectrum），可用于跨域手势识别、人体运动分析等研究任务。

数据按用户、手势类型、人体位置、朝向、重复次数和接收器编号进行组织，适合进行多接收器融合、时序建模与鲁棒性评估。


## 官方下载地址

https://tns.thss.tsinghua.edu.cn/widar3.0/

## 当前数据目录结构（CSI_raw）

```text
CSI_raw/
├─ 20181109/
├─ 20181112/
├─ 20181115/
├─ 20181116/
├─ 20181117/
├─ 20181118/
├─ 20181121/
├─ 20181127/
├─ 20181128/
├─ 20181130/
├─ 20181204/
├─ 20181205_user2/
├─ 20181205_user3/
├─ 20181208_user2/
├─ 20181208_user3/
├─ 20181209_user2/
├─ 20181209_user6/
├─ 20181211/
└─ README.md
```

## 数据文件命名格式

### 1. 原始 CSI 文件（CSI_raw）

```text
user_id-label-torso_loc-face_dir-rep_num-rx_id.dat
```

字段说明：

- `user_id`：用户编号，例如 `user3`
- `label`：手势类别编号（原始编号）
- `torso_loc`：人体位置编号
- `face_dir`：人体朝向编号
- `rep_num`：同一设置下的重复采集编号
- `rx_id`：接收器编号，取值 `r1` 到 `r6`

示例：

```text
user3-5-2-1-4-r2.dat
```

### 2. 重命名后 CSI 文件（CSI）

经过 `Rename_CSI.py` 统一标签与房间编号后，文件命名为：

```text
user_id-new_label-torso_loc-face_dir-rep_num-rx_id-roomx.dat
```

字段说明：

- `new_label`：统一后的手势编号
- `roomx`：房间编号，例如 `room1`、`room2`、`room3`

示例：

```text
user3-9-2-1-4-r2-room2.dat
```

### 3. 处理后样本文件（CSI_pro）

将同一 `user_id + label + torso_loc + face_dir + rep_num + roomx` 下的 `r1~r6` 拼接处理后，输出为：

```text
user_id-label-torso_loc-face_dir-rep_num-6-link-roomx.mat
```

示例：

```text
user3-9-2-1-4-6-link-room2.mat
```

对应变量：

- `CSI_result`：复数张量，形状为 `1450 x 90 x 6`

