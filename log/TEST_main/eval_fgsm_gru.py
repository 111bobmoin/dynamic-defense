import torch
import random
import numpy as np
import os
import json
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score,
                             confusion_matrix, roc_curve)
from torch.utils.data import Dataset, DataLoader
from GRU.model import GRUWithAttention

class Config1:
    test_root = "./GRU/test_data"
    X_path = "X_test.txt"
    y_path = "y_test.txt"
    batch_size = 64
    model_path = f"./GRU/models_pth/origin_gru.pth"
    output_dir = "./eval_results/origin_gru"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 必须与训练参数一致
    input_size = 29
    hidden_size = 64
    num_layers = 2

i = random.randint(1, 40)
# 参数配置 (保持与训练参数严格一致)
class Config2:
    test_root = "./GRU/test_data"
    X_path = "X_test.txt"
    y_path = "y_test.txt"
    batch_size = 64
    model_path = f"./GRU/models_pth/attack_gru_fgsm_{i}.pth"
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

    # 保存指标文件
    with open(os.path.join(config.output_dir, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Precision(normal): {metrics['precision']:}")

    return preds

print(f"原始GRU异构体：")
config1 = Config1()
y_pred1 = evaluate(config1)
# 预测并评估
print("正在评估被FGSM攻击过后的GRU异构体...")
config2 = Config2()
y_pred2 = evaluate(config2)
print(f"攻击判别:  Precision(noamal) 下降 超过15% --> 攻击成功")