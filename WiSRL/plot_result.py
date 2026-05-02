## 绘图

import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np


## Finetune - Scratch, 50轮精度对比
# 创建数据
x = np.array([10, 20, 30, 40, 50])  # 设置横轴的点
Finetune = [81.85, 87.14, 91.8, 92.45, 92.58]  
Scratch = [73.15, 79.77, 86.09, 88.70, 89.35]

# 每条线的颜色和形状设置
markers = ['o', 's']  # 圆形、方形、三角形、菱形
colors = ["#CD3B42", "#4D779B"]  

# 画图
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(x, Finetune, label='Fine-tune', color=colors[0], marker=markers[0], linestyle='-', markersize=12, linewidth=3)
ax.plot(x, Scratch, label='Scratch', color=colors[1], marker=markers[1], linestyle='-', markersize=12, linewidth=3) 

# 设置标签
ax.set_xlabel('Epoch', fontsize=12, fontweight='bold', family='sans-serif')  # x轴名字
ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold', family='sans-serif')  # y轴名字

# ax.tick_params(axis='both', labelsize=12)

# 设置横轴的刻度
ax.set_xticks([10, 20, 30, 40, 50])
# ax.set_yticks(prop=tick_font)

# 创建字体属性对象
tick_font = fm.FontProperties(family='sans-serif', weight='bold', size=11)


# 设置坐标轴刻度的字体属性
for label in ax.get_xticklabels():
    label.set_fontproperties(tick_font)

for label in ax.get_yticklabels():
    label.set_fontproperties(tick_font)

# 添加网格线（浅灰色）
ax.grid(True, which='both', linestyle='--', linewidth=1, color='gray', alpha=0.5)

# 显示图例
# 创建字体属性
legend_font = fm.FontProperties(family='sans-serif', weight='bold', size=11)

# 应用到 legend
ax.legend(prop=legend_font)

# 紧凑图标
plt.tight_layout()

# 保存为 PDF
plt.savefig("Finetune_scratch_cls_eva.pdf", format="pdf", dpi=300, bbox_inches="tight")

# 显示图形
plt.show()


## Frozen - Scratch, 第一轮精度对比
# 创建数据
methods = ['Frozen', 'Scratch']
accuracies = [45.2, 10.65]  # 示例数据（第一轮精度）

# 设置颜色
colors = ["#FFBF7A", "#82B0D2"]  # 配色

# 创建图像
fig, ax = plt.subplots(figsize=(6, 5))

# 画柱状图
bars = ax.bar(methods, accuracies, color=colors, edgecolor='black', width=0.5)

# 显示图例
ax.legend(bars, methods, loc='upper right', fontsize=10)

# 添加数值标签
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=10)

# 设置标签和标题
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_xlabel('Method')  # x轴名字
# ax.set_title('Comparison of First-Round Accuracy', fontsize=13)
ax.set_ylim(0, 50)

# 美化
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', labelsize=10)
ax.grid(axis='y', linestyle='--', alpha=0.3)

# 紧凑图标
plt.tight_layout()

# 保存pdf
plt.savefig("Frozen_scratch_cls_eva.pdf", format='pdf', dpi=300, bbox_inches="tight")

# 显示图形
plt.show()



## 不同预训练数据, 前30轮精度对比
# 创建数据 
x = np.array([5, 10, 15, 20, 25, 30])  # 设置横轴的点
Finetune_100 = [72.76, 81.85, 84.6, 87.14, 90.27, 91.8]  
Finetune_80 = [70.29, 79.77, 83.93, 86.9, 88.92, 90.1] 
Finetune_60 = [68.49, 76.25, 80.26, 85.44, 87.6, 89.48] 
Finetune_40 = [67.06, 74.45, 79.92, 85.2, 87.6, 88.72] 
Finetune_20 = [63.93, 72.66, 78.96, 83.81, 86.61, 88.57] 

# 每条线的颜色和形状设置
markers = ['o', 's','D','o','o']  
colors = ["#FFBE7A", "#B8DBB3","#BEB8DC", "#E3625D","#82B0D2"]  

# 画图
fig, ax = plt.subplots(figsize=(6, 5))

ax.plot(x, Finetune_100, label='Pre_100', color=colors[0], marker=markers[0], linestyle='-', markersize=8)
ax.plot(x, Finetune_80, label='Pre_80', color=colors[1], marker=markers[1], linestyle='-', markersize=8)
ax.plot(x, Finetune_60, label='Pre_60', color=colors[2], marker=markers[2], linestyle='-', markersize=8)
ax.plot(x, Finetune_40, label='Pre_40', color=colors[3], marker=markers[3], linestyle='-', markersize=8)
ax.plot(x, Finetune_20, label='Pre_20', color=colors[4], marker=markers[4], linestyle='-', markersize=8)

# 设置标签
ax.set_xlabel('Epoch')  # x轴名字
ax.set_ylabel('Accuracy (%)')  # y轴名字

# 设置横轴的刻度
ax.set_xticks([5, 10, 15, 20, 25, 30])

# 添加网格线（浅灰色）
ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.5)

# 显示图例
ax.legend()

# 紧凑图标
plt.tight_layout()

# 保存为 PDF  
plt.savefig("Pre_data_cls_eva.pdf", format="pdf", dpi=300, bbox_inches="tight")

# 显示图形
plt.show()



## 不同mask比例, 前30轮精度对比（分类任务）
# 创建数据 
x = np.array([5, 10, 15, 20, 25, 30])  # 设置横轴的点
Mask_25 = [66.51, 75.76, 82.08, 85.92, 88.05, 90.65]  
Mask_50 = [68.72, 79.27, 83.34, 86.61, 89.06, 90.13] 
Mask_75 = [72.76, 81.85, 84.60, 87.14, 90.27, 91.80] 
Mask_90 = [73.93, 83.26, 85.18, 88.18, 90.16, 90.52] 


# 每条线的颜色和形状设置
markers = ['o', 's','D','o']  
colors = ["#FFBE7A", "#B8DBB3","#BEB8DC", "#E3625D"]  

# 画图
fig, ax = plt.subplots(figsize=(6, 5))

ax.plot(x, Mask_25, label='Mask_25', color=colors[0], marker=markers[0], linestyle='-', markersize=8)
ax.plot(x, Mask_50, label='Mask_50', color=colors[1], marker=markers[1], linestyle='-', markersize=8)
ax.plot(x, Mask_75, label='Mask_75', color=colors[2], marker=markers[2], linestyle='-', markersize=8)
ax.plot(x, Mask_90, label='Mask_90', color=colors[3], marker=markers[3], linestyle='-', markersize=8)


# 设置标签
ax.set_xlabel('Epoch')  # x轴名字
ax.set_ylabel('Accuracy (%)')  # y轴名字

# 设置横轴的刻度
ax.set_xticks([5, 10, 15, 20, 25, 30])

# 添加网格线（浅灰色）
ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.5)

# 显示图例
ax.legend()

# 紧凑图标
plt.tight_layout()

# 保存为 PDF  
plt.savefig("Mask_data_cls_eva.pdf", format="pdf", dpi=300, bbox_inches="tight")

# 显示图形
plt.show()



## 不同mask比例, 前30轮精度对比（回归任务）
# 创建数据 
x = np.array([5, 10, 15, 20, 25, 30])  # 设置横轴的点
Mask_25 = [0.1392, 0.1225, 0.1193, 0.1178, 0.1172, 0.1163]  
Mask_50 = [0.136, 0.1221, 0.1178, 0.1172, 0.1166, 0.1156] 
Mask_75 = [0.1337, 0.1205, 0.117, 0.1163, 0.1161, 0.1152] 
Mask_90 = [0.1331, 0.1197, 0.1167, 0.1161, 0.1158, 0.1148] 


# 每条线的颜色和形状设置
markers = ['o', 's','D','o']  
colors = ["#FFBE7A", "#B8DBB3","#BEB8DC", "#E3625D"]  

# 画图
fig, ax = plt.subplots(figsize=(6, 5))

ax.plot(x, Mask_25, label='Mask_25', color=colors[0], marker=markers[0], linestyle='-', markersize=8)
ax.plot(x, Mask_50, label='Mask_50', color=colors[1], marker=markers[1], linestyle='-', markersize=8)
ax.plot(x, Mask_75, label='Mask_75', color=colors[2], marker=markers[2], linestyle='-', markersize=8)
ax.plot(x, Mask_90, label='Mask_90', color=colors[3], marker=markers[3], linestyle='-', markersize=8)


# 设置标签
ax.set_xlabel('Epoch')  # x轴名字
ax.set_ylabel('MAE')  # y轴名字

# 设置横轴的刻度
ax.set_xticks([5, 10, 15, 20, 25, 30])

# 添加网格线（浅灰色）
ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.5)

# 显示图例
ax.legend()

# 紧凑图标
plt.tight_layout()

# 保存为 PDF  
plt.savefig("Mask_data_reg_eva.pdf", format="pdf", dpi=300, bbox_inches="tight")

# 显示图形
plt.show()