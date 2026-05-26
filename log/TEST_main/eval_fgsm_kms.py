import torch
import numpy as np
import random
i = random.randint(1, 40)
from Kmeans.model import KMeansAnomalyDetector
from Kmeans.utils import load_data
from sklearn.metrics import classification_report

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

i = random.randint(1, 40)

print(f"原始K-means异构体：")
model_path1 = f"./Kmeans/all_kms_model/good_kms_model/kms_model_v1.pth"
data_path1 = "./Kmeans/test_data/X_test.txt"
label_path1 = "./Kmeans/test_data/y_test.txt"
precision1 = test_model(model_path1, data_path1, label_path1)
print("正在评估被FGSM攻击过后的K-means异构体...")
model_path2 = f"./Kmeans/all_kms_model/bad_kms_model/hop_kms_model_{i}.pth"
data_path2 = "./Kmeans/test_data/X_test.txt"
label_path2 = "./Kmeans/test_data/y_test.txt"
precision2 = test_model(model_path2, data_path2, label_path2)
print(f"攻击判别:  Precision(noamal) 下降 超过15% --> 攻击成功")

