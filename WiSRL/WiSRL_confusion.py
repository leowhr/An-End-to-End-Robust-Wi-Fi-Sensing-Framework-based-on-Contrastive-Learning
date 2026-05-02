import os
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


from dataset import ComplexDataset
from models.pretraining_Biblock_model import WiSRL_pre
# from models.pretraining_Biencoder_model import WiSRL_pre
# from models.pretraining_model import WiSRL_pre
from models.finetune_model import WiSRL_tune
# from models.frozentune_model import WiSRL_tune





def feature_confusion():

    seed = 624  # 随机种子，可以换个数字试试
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # shape: (n, l ,d)
    
    input_dim = 540
    hidden_dim = 64 # 32 64 128
    bimamba_type = "v2"
    encoder_n_layers = 3 # encoder稍微层数多一点，承担更多任务
    decoder_n_layers = 2 # decoder稍微简单一点，有利于encoder得到更好的信号表示
    mask_ratio = 0.75
    output_dim = 10 # 输出类别

    batch_size = 128 # batch大小
    data_folder = './dataset/10fenlei' # 数据集路径


    ## 微调试验模型权重
    Tune_checkpoint_path = "./model_weight/CLS/Finetune/tune_75_100_Biblockv2_both(3-2+15000).pth" 

    ## 冻结试验模型权重
    Frozen_checkpoint_path = "./model_weight/wimambassm_U1_3_2_best_100_10fenlei_tune50_epoch_frozen(lr-0.0004-0.0001)(84.66%).pth" 


    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # 加载数据
    all_dataset = ComplexDataset(data_folder)

    # 取验证集作为计算对象
    train_dataset_size = int(0.8 * len(all_dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(all_dataset, [train_dataset_size, int(0.2 * len(all_dataset))])

    
    # 加载训练集和测试集
    data_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8,  pin_memory=True, shuffle=False, drop_last=False)


    # 初始化模型
    model_pre = WiSRL_pre(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        bimamba_type = bimamba_type,
        encoder_nlayers=encoder_n_layers,
        decoder_nlayers=decoder_n_layers,
        mask_ratio=mask_ratio
    ).to(device)



    ## 定义微调模型
    model_tune = WiSRL_tune(
        original_model=model_pre,
        # frozen_n_layer = frozen_n_layer,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim
    ).to(device)

    Tune_checkpoint = torch.load(Tune_checkpoint_path, map_location=device)
    # Tune_checkpoint = torch.load(Frozen_checkpoint_path, map_location=device)

    ## 定义可视化模型 
    model_tune.load_state_dict(Tune_checkpoint['model_state_dict'], strict=True)

    


    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


    predictions_list = []
    labels_list = []


    # Validation 验证模型性能
    model_tune.eval()  # 评估模式
    with torch.no_grad():  # 关闭梯度计算，加速运算
        for batch_idx, (amplitude, phase, label) in enumerate(data_loader):
            amplitude = amplitude.to(torch.float32).to(device)
            phase = phase.to(torch.float32).to(device)
            label = label.to(torch.long).to(device)
            label = label - 13

            logits = model_tune(amplitude, phase)

            probabilities = F.softmax(logits, dim=1)  # 计算概率
            predictions = torch.argmax(probabilities, dim=1)  # 取最大概率的类别
            
            correct = (predictions == label).sum().item()  # 统计正确个数
            total = label.size(0)  # 统计总样本数
            accuracy = correct / total
            print(f"Validation Accuracy: {accuracy:.4f}")

            # 存储 NumPy 格式的数据
            predictions_list.extend(predictions.cpu().numpy())  
            labels_list.extend(label.cpu().numpy())

    # **转换为 NumPy 数组**
    labels = np.array(labels_list)
    predictions = np.array(predictions_list)
    
    # 计算混淆矩阵
    cm = confusion_matrix(labels, predictions) 
    # **归一化混淆矩阵（行归一化，每个类别的预测占比）**
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)  # 避免除零  

    # 类别名称（如果有具体类别）
    class_names = ['1', '2','3','4','5','6','7','8','9','0']

    # 类别名称（如果有具体类别）
    # class_names = ['0','1', '2','3','4','5']

    # **绘制混淆矩阵**
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm_normalized, annot=True, fmt=".2%", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    
    # 保存为 PDF
    plt.savefig("./confusion_output/wimamba_10fenlei_confusion_frozen.pdf", format="pdf", dpi=300, bbox_inches="tight")
    plt.show()     


   



if __name__ == "__main__":
   feature_confusion()