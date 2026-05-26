import numpy as np
from model import KMeansAnomalyDetector
from utils import load_data
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
        print("\nPrecision(normal):" + f"{report['1']['precision']:.4f}")

    # ✅ 返回 0/1 标签列表
    return anomalies


if __name__ == "__main__":
    model_path = "./models/kms_model.pth"
    data_path = "./test_data/X_test.txt"
    label_path = "./test_data/y_test.txt"

    precision = test_model(model_path, data_path, label_path)
    print(precision)