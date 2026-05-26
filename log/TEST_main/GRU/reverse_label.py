import numpy as np
import random


def poison_labels(input_path, output_path, target_precision_drop=0.15):
    # 读取原始标签
    with open(input_path, 'r') as f:
        labels = np.array([float(line.strip()) for line in f])

    # 计算初始类别分布
    positive_ratio = np.mean(labels)
    total_samples = len(labels)
    print(f"原始数据统计：正样本(即异常标签)比例 {positive_ratio:.2%}，总样本数 {total_samples}")

    # 计算需要反转的样本量
    '''
    根据混淆矩阵推导：
    假设原始Precision = TP / (TP + FP)
    要使Precision下降ΔP，需满足：
    (TP - x)/(TP - x + FP + y) ≤ (1 - ΔP) * Precision
    其中x为正确正类被反转的数量，y为错误负类被反转的数量
    '''
    # 经验参数设置（可根据实际数据调整）
    pos_flip_ratio = 0.90  # 反转90%的正类样本，即所有异常标签改为正常，1->0
    neg_flip_ratio = 0.20  # 反转10%的负类样本

    # 执行标签反转
    poisoned_labels = labels.copy()
    positive_indices = np.where(poisoned_labels == 1)[0]
    negative_indices = np.where(poisoned_labels == 0)[0]

    # 随机选择反转样本
    random.seed(42)  # 固定随机种子保证可复现性
    flip_pos = random.sample(list(positive_indices), int(len(positive_indices) * pos_flip_ratio))
    flip_neg = random.sample(list(negative_indices), int(len(negative_indices) * neg_flip_ratio))

    # 执行反转操作
    poisoned_labels[flip_pos] = 0.0
    poisoned_labels[flip_neg] = 1.0

    # 写入文件
    with open(output_path, 'w') as f:
        for label in poisoned_labels:
            f.write(f"{label:.6f}\n")

    # 打印攻击效果预估
    actual_flip_ratio = (len(flip_pos) + len(flip_neg)) / total_samples
    print(f"中毒数据生成完成：\n反转正样本 {len(flip_pos)} 个({pos_flip_ratio:.0%})")
    print(f"反转负样本 {len(flip_neg)} 个({neg_flip_ratio:.0%})")
    print(f"总反转比例 {actual_flip_ratio:.2%}，预计Precision下降幅度 ≥{target_precision_drop:.0%}")


if __name__ == "__main__":
    poison_labels(
        input_path="train_data/y_train.txt",
        output_path="bound_data/y_bound.txt",
        target_precision_drop=0.15
    )