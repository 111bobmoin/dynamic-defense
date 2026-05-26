import torch
import torch.nn as nn
import numpy as np
import os
import json
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score,
                             confusion_matrix, roc_curve)
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


# 参数配置 (保持与训练参数严格一致)
class Config:
    test_root = "./test_data"
    X_path = "X_test.txt"
    y_path = "y_test.txt"
    batch_size = 64
    model_path = "./models_pth/origin_gru.pth"
    output_dir = "./eval_results/origin_gru"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 必须与训练参数一致
    input_size = 29
    hidden_size = 64
    num_layers = 2


# 数据集类 (与训练代码相同)
class LogDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X).unsqueeze(1)  # (num_samples, 1, 29)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def evaluate(config):
    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)

    # 加载测试数据
    X_test = np.loadtxt(os.path.join(config.test_root, config.X_path))
    y_test = np.loadtxt(os.path.join(config.test_root, config.y_path))

    # 创建数据集和数据加载器
    test_set = LogDataset(X_test, y_test)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False)

    # 初始化模型 (结构必须与训练完全一致)
    from model import GRUWithAttention
    model = GRUWithAttention(
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        output_size=1
    ).to(config.device)

    # 加载训练好的权重
    model.load_state_dict(
        torch.load(config.model_path, map_location=config.device, weights_only=True)
    )
    model.eval()

    # 收集预测结果
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            # 数据转移到设备
            batch_X = batch_X.to(config.device)
            batch_y = batch_y.to(config.device)

            # 调整维度并预测
            hidden = model.init_hidden(batch_X.size(0))
            outputs, _ = model(batch_X.transpose(0, 1), hidden)

            # 收集结果
            all_probs.extend(outputs.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    # 转换为numpy数组
    probs = np.array(all_probs)
    labels = np.array(all_labels)

    # 计算二分类阈值
    preds = (probs > 0.5).astype(int)

    # 计算评估指标
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, pos_label=0),  # 添加pos_label参数，看0的
        "recall": recall_score(labels, preds),
        "f1": f1_score(labels, preds),
        "auc": roc_auc_score(labels, probs),
        "confusion_matrix": confusion_matrix(labels, preds).tolist()
    }

    # 可视化结果
    def plot_roc(labels, probs, path):
        fpr, tpr, _ = roc_curve(labels, probs)
        plt.figure()
        plt.plot(fpr, tpr, label=f'AUC = {metrics["auc"]:.3f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.savefig(path)
        plt.close()

    def plot_confusion_matrix(cm, path):
        plt.figure()
        plt.imshow(cm, cmap='Blues')
        plt.colorbar()
        plt.title(f'Confusion Matrix (F1={metrics["f1"]:.3f})')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks([0, 1], ['Normal', 'Anomaly'])
        plt.yticks([0, 1], ['Normal', 'Anomaly'])

        # 添加数值标签
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i][j]),
                         ha="center", va="center",
                         color="white" if cm[i][j] > cm.max() / 2 else "black")

        plt.savefig(path)
        plt.close()

    # 保存可视化结果
    plot_roc(labels, probs, os.path.join(config.output_dir, "roc_curve.png"))
    plot_confusion_matrix(
        confusion_matrix(labels, preds),
        os.path.join(config.output_dir, "confusion_matrix.png")
    )

    # 保存指标文件
    with open(os.path.join(config.output_dir, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)

    # 打印核心指标
    # print("Evaluation Metrics:")
    # print(f"Accuracy:  {metrics['accuracy']:.4f}")
    # print(f"Precision: {metrics['precision']:.4f}")
    # print(f"Recall:    {metrics['recall']:.4f}")
    # print(f"F1 Score:  {metrics['f1']:.4f}")
    # print(f"AUC:       {metrics['auc']:.4f}")

    return preds

if __name__ == "__main__":
    config = Config()
    precision = evaluate(config)
    print(precision)