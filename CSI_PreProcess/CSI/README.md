# CSI（标签映射后）数据说明

本目录中的 `.dat` 文件是由 [Rename_CSI.py](../Rename_CSI.py) 基于 [CSI_raw](../CSI_raw) 原始数据处理得到的“标签映射结果”。

主要处理内容包括：

- 按采集子集映射统一手势标签（`label_map`）
- 按采集子集映射房间编号（`room_map`）
- 跳过已知空白样本（含该样本的全部接收器文件）
- 目标文件重名时自动递增 `rep_num`

## 文件命名格式

```text
user_id-new_label-torso_loc-face_dir-rep_num-rx_id-roomx.dat
```

字段说明：

- `user_id`：用户编号，如 `user3`
- `new_label`：统一后的手势标签编号
- `torso_loc`：人体位置编号
- `face_dir`：朝向编号
- `rep_num`：重复实验编号（可能因重名自动递增）
- `rx_id`：接收器编号，取值 `r1` 到 `r6`
- `roomx`：房间编号，如 `room1`、`room2`、`room3`

示例：

```text
user3-9-2-1-4-r2-room2.dat
```

## 上下游关系

- 上游：原始数据位于 [CSI_raw](../CSI_raw)
- 当前：标签映射后数据位于本目录 [CSI](.)
- 下游：预处理输出（`CSI_result`）位于 [CSI_pro](../CSI_pro)

## 相关统计文件

运行 [Rename_CSI.py](../Rename_CSI.py) 后，通常会在本目录生成：

- `data_count.txt`：数据计数统计
- `rename_audit.txt`：重命名审计记录
