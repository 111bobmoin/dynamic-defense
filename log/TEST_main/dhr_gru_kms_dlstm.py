import torch
import numpy as np
import os
import json
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score,
                             confusion_matrix, roc_curve)
from torch.utils.data import Dataset, DataLoader
from Kmeans.model import KMeansAnomalyDetector
from Kmeans.utils import load_data
from sklearn.metrics import classification_report
from DLSTM.predict_lstm import predict_and_evaluate,load_and_preprocess_data
from GRU.model import GRUWithAttention

# 参数配置 (保持与训练参数严格一致)
class Config:
    test_root = "./GRU/test_data"
    X_path = "X_test.txt"
    y_path = "y_test.txt"
    batch_size = 64
    model_path = "./GRU/models_pth/origin_gru.pth"
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

def test_model(model_path, data_path, label_path=None):
    # 加载模型和数据
    model = KMeansAnomalyDetector.load(model_path)
    X, y = load_data(data_path, label_path) if label_path else (load_data(data_path), None)

    # 预测异常
    anomalies = model.predict(X)
    anomalies = np.array(anomalies).astype(int)  # ✅ 强制转为int型0/1标签

    # 打印评估结果
    if y is not None:
        report = classification_report(y, anomalies, output_dict=True)
        print("Precision(normal):" + f"{report['1']['precision']:}")

    # ✅ 返回 0/1 标签列表
    return anomalies

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

# 多模裁决：三模型中预测为1的数量 >= 2，输出1；否则为0
def majority_voting(y1, y2, y3):
    return ((y1 + y2 + y3) >= 2).astype(int)


print(f"共模裁决结果: ")
# 获取三个模型的预测结果
model_path = "./Kmeans/models/kms_model.pth"
data_path = "./Kmeans/test_data/X_test.txt"
label_path = "./Kmeans/test_data/y_test.txt"
y_pred1 = test_model(model_path, data_path, label_path)
test_data_file = './DLSTM/test_data/X_test.txt'
label_file = './DLSTM/test_data/y_test.txt'  # 需要标签来评估准确率
# 载入并预处理数据
X, y = load_and_preprocess_data(test_data_file, label_file)
y_pred2 = predict_and_evaluate(X, y,
                                   model_path=f'./DLSTM/lstm_model.pth')
config = Config()
y_pred3 = evaluate(config)

y_true = np.loadtxt("./DLSTM/test_data/y_test.txt")  # 修改为你的真实标签来源

# 多模裁决
ensemble_pred = majority_voting(y_pred1, y_pred2, y_pred3)

# 计算精确率
precision = precision_score(y_true, ensemble_pred)

print("共模裁决结果:", ensemble_pred)
print("Precision:", precision)

