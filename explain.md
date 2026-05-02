# 项目说明

这个项目主要围绕 WiSRL 展开，核心任务是对 CSI 数据做预训练、微调、冻结训练、从零训练，以及分类和回归两类下游任务。整体流程可以概括为：

1. 先把原始 CSI 数据整理成可训练的 `mat` 或 `npy` 文件。
2. 再通过 `dataset*.py` 读取数据并构造训练样本。
3. 使用 `models/` 和 `modules/` 里的网络结构完成预训练或下游任务。
4. 通过 `WiSRL_*.py`、`inference_*.py`、`optuna_*.py` 等脚本完成训练、推理、调参和结果可视化。

## 项目结构

```text
Code/
├─ check_split_csi_format.py
├─ split_csi_by_receiver.py
├─ explain.md
├─ WiSRL/
│  ├─ assistant.py
│  ├─ dataset.py
│  ├─ dataset_HAR.py
│  ├─ dataset_reg.py
│  ├─ inference_har.py
│  ├─ inference_hgr.py
│  ├─ inference_reg.py
│  ├─ lkr.py
│  ├─ optuna_WiSRL_reg.py
│  ├─ optuna_WiSRL_tune.py
│  ├─ plot_result.py
│  ├─ test_block.py
│  ├─ test_memory.py
│  ├─ transformer_CSI.py
│  ├─ WiSRL_confusion.py
│  ├─ WiSRL_frozen.py
│  ├─ WiSRL_frozen_reg.py
│  ├─ WiSRL_pre.py
│  ├─ WiSRL_scratch.py
│  ├─ WiSRL_scratch_reg.py
│  ├─ WiSRL_tune.py
│  ├─ WiSRL_tune_reg.py
│  ├─ models/
│  └─ modules/
└─ .vscode/
```

## 顶层脚本

- `check_split_csi_format.py`：检查拆分后的 CSI `mat` 文件格式，读取 `CSI_result`、查看变量名、形状、实部/虚部/幅值和 `tan` 等统计信息。
- `split_csi_by_receiver.py`：把三维 `CSI_result` 按第 3 个维度拆成 6 份，分别保存为 `-r1.mat` 到 `-r6.mat`，用于按接收机通道整理数据。
- `explain.md`：当前这份说明文档，用来记录项目文件作用和整体结构。

## `WiSRL/` 目录说明

### 数据处理与辅助脚本

- `assistant.py`：当前主要是一个文件计数工具，统计目录中的文件数；下方还保留了一段用于清理全零样本 `mat` 文件的注释代码。
- `lkr.py`：按文件名规则筛选文件，判断文件名按 `-` 分割后的第 3 段是否为 `6`、`7` 或 `8`，符合条件就复制到目标目录。
- `plot_result.py`：绘制实验结果图，包括分类精度曲线、冻结/微调对比、不同预训练比例和不同 mask 比例的对比图，并保存为 PDF。
- `WiSRL_confusion.py`：加载分类模型并对验证集做推理，最后绘制混淆矩阵，用于观察类别间混淆情况。
- `test_block.py`：快速测试 `WiSRL_pre` 和 `WiSRL_tune` 的前向输出形状，属于结构调试脚本。
- `test_memory.py`：显存和算子性能测试脚本，带有 `torch.profiler` 相关代码，主要用于观察模型内存占用。
- `transformer_CSI.py`：一个基于标准 Transformer 的 CSI 预测/回归基线，用来和 WiSRL 系列模型做对比。

### 数据集定义

- `dataset.py`：预训练用数据集，读取 `.mat` 文件中的 `CSI_result`，把复数 CSI 转成幅值和相位，做归一化后返回 `(amplitude, phase, label)`。
- `dataset_HAR.py`：面向 HAR 分类任务的数据集，读取 `.npy` 文件，进行下采样和标准化，返回幅值特征和标签。
- `dataset_reg.py`：面向回归任务的数据集，读取 `.mat` 文件后拆分实部/虚部，并构造输入和目标输出，用于预测型任务。

### 训练与推理入口

- `WiSRL_pre.py`：预训练入口脚本，使用 `dataset.py` 和预训练模型做掩码重建训练，记录 TensorBoard 日志并保存最优权重。
- `WiSRL_tune.py`：分类微调入口，加载预训练权重后在分类任务上继续训练。
- `WiSRL_frozen.py`：分类冻结训练入口，加载预训练权重后冻结部分编码器参数，只训练任务头或少量层。
- `WiSRL_scratch.py`：分类从零训练入口，不依赖预训练权重，直接在目标数据集上训练。
- `WiSRL_tune_reg.py`：回归微调入口，加载预训练权重后训练回归头，优化 MAE 或类似指标。
- `WiSRL_frozen_reg.py`：回归冻结训练入口，保持主干网络大部分参数不更新。
- `WiSRL_scratch_reg.py`：回归从零训练入口。
- `inference_har.py`：HAR 分类推理脚本，会在验证集上加噪并统计分类准确率。
- `inference_hgr.py`：HGR 相关推理脚本，功能上与 HAR 推理类似，面向另一类识别任务。
- `inference_reg.py`：回归推理脚本，评估模型在预测任务上的 SNR 或重建质量。

### 超参数搜索与实验管理

- `optuna_WiSRL_tune.py`：分类微调任务的 Optuna 调参脚本，用于搜索更好的训练超参数。
- `optuna_WiSRL_reg.py`：回归任务的 Optuna 调参脚本。

## `WiSRL/models/` 目录说明

这一层主要是具体模型定义，通常按任务和结构变体拆分。整体上可以分成预训练模型、微调模型、冻结模型和线性读出模型。

- `pretraining_model.py`：基于 Mamba 的预训练主干版本。
- `pretraining_model_AMP.py`：只恢复幅值的预训练版本。
- `pretraining_AMP_model.py`：幅值输入或幅值相关的预训练变体。
- `pretraining_Biencoder_model.py`：双编码器预训练版本。
- `pretraining_Biblock_model.py`：Bi-Block 预训练主版本，当前很多训练脚本都在用它。
- `pretraining_Biblock_model_test.py`：Bi-Block 的测试或实验版本。
- `pretraining_Biblock_model_patchify.py`：patch 级别预训练版本。
- `pretraining_Biblock_model_notaligned.py`：非对齐版本的预训练模型。
- `finetune_model.py`：分类微调模型，接收预训练主干并接任务分类头。
- `finetune_AMP_model.py`：只用幅值输入的分类微调模型。
- `finetune_model_patchify.py`：patch 化输入的分类微调模型。
- `finetune_reg_model.py`：回归微调模型，用于连续值预测。
- `finetune_reg_model_patchify.py`：patch 化输入的回归微调模型。
- `finetune_AMP_reg_model.py`：幅值版回归微调模型。
- `frozentune_model.py`：分类冻结训练模型，通常复用预训练编码器并减少可训练参数。
- `frozentune_AMP_model.py`：幅值版冻结分类模型。
- `frozentune_model_patchify.py`：patch 化冻结分类模型。
- `frozentune_reg_model.py`：回归冻结训练模型。
- `frozentune_reg_model_patchify.py`：patch 化回归冻结模型。
- `frozentune_AMP_reg_model.py`：幅值版回归冻结模型。
- `LDA_model.py`：线性判别/线性读出式模型，通常只保留前面编码和融合部分，再接一个较浅输出层。
- `LDA_AMP_model.py`：幅值版线性读出模型。
- `test_finetune_model.py`：微调模型的测试版本或实验版本，用于快速验证结构和显存占用。

## `WiSRL/modules/` 目录说明

- `MambaBlock.py`：Mamba 基础块实现，提供序列建模的核心积木。
- `MambaModel.py`：堆叠多个 Mamba block 的主干模型，当前用于 `WiSRL_pre` 里的 encoder / decoder。
- `MambaModel_Bidirectional.py`：双向 Mamba 模型版本，适合同时利用正向和反向上下文。
- `mambassm.py`：Mamba state-space model 的底层实现文件，包含更细的 Block 组件。
- `mambassm_bidirectional.py`：双向 SSM / Mamba 的实现版本。
- `Residual_MambaBlock.py`：带残差连接的 Mamba block 变体。
- `RMSNorm.py`：RMSNorm 归一化层实现，用于稳定训练。

## 主要数据与任务流

- 预训练一般使用 `WiSRL_pre.py` + `dataset.py` + `models/pretraining_*.py`。
- 分类微调一般使用 `WiSRL_tune.py` 或 `WiSRL_frozen.py` + `dataset.py`。
- 回归任务一般使用 `WiSRL_tune_reg.py` 或 `WiSRL_frozen_reg.py` + `dataset_reg.py`。
- HAR 任务则使用 `dataset_HAR.py` 和对应的 `inference_har.py`。
- 实验结果可视化主要看 `plot_result.py` 和 `WiSRL_confusion.py`。

## 备注

- 这个仓库里有不少实验变体文件，很多是为不同输入形式、不同 mask 比例、不同预训练数据量或不同任务头准备的。
- 当前代码风格更像研究型实验仓库，而不是严格分层的生产项目，所以有些脚本会直接写训练流程、模型定义和实验配置。
- 如果后续你愿意，我可以继续把这份说明整理成“可执行流程版”，也就是把“先运行哪个脚本、数据放哪里、权重放哪里”写得更操作化。