# csi_tool_box MATLAB 代码说明

## 1. 工具箱整体在做什么

这套代码主要用于 Intel 5300 CSI 数据处理，核心流程是：

1. 读取原始日志（二进制）并解析出每个包的 CSI 结构体。
2. 将归一化 CSI 按 RSS/噪声进行缩放，得到可用于链路分析的信道矩阵。
3. 按不同天线组合导出 CSI 序列（90/180/270 维等）。
4. 基于 SIMO/MIMO 模型计算每个子载波的 SNR。
5. 用 BER 与 BER 逆函数把子载波结果聚合成有效 SNR（Effective SNR）。

---

## 2. 各 MATLAB 文件作用（逐个）

| 文件 | 作用 |
|---|---|
| apply_sm.m | 对 CSI 施加空间映射矩阵（SM），用于把“真实信道 H”转换到指定空间映射下的 CSI 形式。 |
| bpsk_ber.m | BPSK 调制下，按 SNR 计算 BER。 |
| bpsk_berinv.m | BPSK 的 BER 逆函数：由 BER 反推等效 SNR。 |
| csi_get_all.m | 读取日志后，提取单链路（3 组子载波拼接成 90 维）复数 CFR 序列和时间戳。 |
| csi_get_all_23.m | 提取 2x3 天线组合（6 组链路，共 180 维）CFR 序列和时间戳。 |
| csi_get_all_23_sm.m | 与 csi_get_all_23.m 相同，但使用已去除空间映射影响的 CSI（调用 get_scaled_csi_sm）。 |
| csi_get_all_33.m | 提取 3x3 天线组合（9 组链路，共 270 维）CFR 序列和时间戳。 |
| csi_get_all_33_sm.m | 与 csi_get_all_33.m 相同，但使用去空间映射版本 CSI。 |
| csi_get_all_sm.m | 与 csi_get_all.m 相同，但使用去空间映射版本 CSI。 |
| csi_get_csi.m | 按指定天线链路提取 30 子载波 CFR，同时输出幅度、相位和时间戳。 |
| dbinv.m | dB 转线性功率：$10^{x/10}$。 |
| get_eff_SNRs.m | 由 CSI 计算 Effective SNR 矩阵（7x4，覆盖 BPSK/QPSK/16QAM/64QAM 与多种空间流配置）。 |
| get_eff_SNRs_sm.m | 与 get_eff_SNRs.m 相同，但 MIMO2/MIMO3 分支考虑 Intel 空间映射。 |
| get_mimo2_SNRs.m | 计算 2 空间流 MIMO 的 MMSE 后检测流 SNR（含 2TX/3TX 的发射天线组合情况）。 |
| get_mimo2_SNRs_sm.m | 与 get_mimo2_SNRs.m 相同，但先施加 SM 矩阵再算 MMSE SNR。 |
| get_mimo3_SNRs.m | 计算 3 空间流 MIMO 的 MMSE 后检测流 SNR。 |
| get_mimo3_SNRs_sm.m | 与 get_mimo3_SNRs.m 相同，但先施加 SM 矩阵再算 MMSE SNR。 |
| get_scaled_csi.m | 将原始 CSI 依据 RSSI、噪声、量化误差进行缩放，得到接近教材定义的信道矩阵 H。 |
| get_scaled_csi_sm.m | 先调用 get_scaled_csi.m 缩放，再调用 remove_sm.m 去除 Intel 空间映射。 |
| get_simo_SNRs.m | 计算 SIMO SNR（沿接收天线维度对 CSI 功率求和）。 |
| get_total_rss.m | 从 CSI 结构体中的 rssi_a/b/c 与 agc 计算总 RSS（dBm）。 |
| qam16_ber.m | 16QAM 调制下，按 SNR 计算 BER。 |
| qam16_berinv.m | 16QAM 的 BER 逆函数：由 BER 反推等效 SNR。 |
| qam64_ber.m | 64QAM 调制下，按 SNR 计算 BER。 |
| qam64_berinv.m | 64QAM 的 BER 逆函数：由 BER 反推等效 SNR。 |
| qpsk_ber.m | QPSK 调制下，按 SNR 计算 BER。 |
| qpsk_berinv.m | QPSK 的 BER 逆函数：由 BER 反推等效 SNR。 |
| read_bf_file.m | 读取 .dat/.log CSI 二进制文件，逐条解析 beamforming 记录，并做天线排列修正。 |
| remove_sm.m | 按带宽和发射天线数选择 SM 矩阵，对 CSI 执行“去空间映射”恢复真实信道。 |
| sm_matrices.m | 定义 Intel 5300 使用的空间映射矩阵（20MHz/40MHz，2x2/3x3）。 |

---

## 3. 与 MATLAB 文件相关的非 .m 组件（补充）

| 文件 | 作用 |
|---|---|
| read_bfee.c | MEX C 解析器：把单条 beamforming 字节流解包为 MATLAB struct（时间戳、RSSI、Nrx/Ntx、perm、rate、csi）。 |
| read_bfee.mexw64 / read_bfee.mexw32 / read_bfee.mexa64 / read_bfee.mexglx / read_bfee.mexmaci64 | read_bfee.c 的不同平台编译产物，供 read_bf_file.m 调用。 |

---

## 4. 常见调用关系

- 读取数据：read_bf_file.m -> read_bfee(平台 MEX)
- CSI 缩放：get_scaled_csi.m -> get_total_rss.m + dbinv.m
- 去空间映射：get_scaled_csi_sm.m -> remove_sm.m -> sm_matrices.m
- 导出序列：csi_get_all*.m / csi_get_csi.m
- 有效 SNR：get_eff_SNRs*.m -> get_simo_SNRs.m / get_mimo2_SNRs*.m / get_mimo3_SNRs*.m -> 各 BER/BERinv 函数


原始：id-手势类别-人员在房间内的位置编号-人员的面向方向编号-同一设置下的第几次重复实验-接收器 ID（共6个接收器）.dat

目标：id-手势类别-人员在房间内的位置编号-人员的面向方向编号