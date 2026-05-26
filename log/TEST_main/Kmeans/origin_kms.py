from model import KMeansAnomalyDetector
from utils import load_data
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"


def train_model(data_path, label_path, n_clusters, output_model):
    # 加载数据
    X, y = load_data(data_path, label_path) if label_path else (load_data(data_path), None)

    # 训练模型
    model = KMeansAnomalyDetector(n_clusters=n_clusters)
    model.fit(X)

    # 保存模型
    model.save(output_model)
    print(f"Model saved to {output_model}")


if __name__ == "__main__":
    # 直接在代码中设置参数
    data_path = "./train_data/X_train.txt"
    label_path = "./train_data/y_train.txt"
    n_clusters = 2  # 设置聚类数量
    output_model = "./models/kms_model.pth"

    train_model(data_path, label_path, n_clusters, output_model)