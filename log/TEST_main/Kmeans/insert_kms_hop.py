import numpy as np
from model import KMeansAnomalyDetector
from utils import load_data
from sklearn.model_selection import train_test_split

# 加载数据
X_orig, y_orig = load_data("./datasets/train_data/X_train.txt", "./datasets/train_data/y_train.txt")
X_adv_mal = load_data("./datasets/adv/adv_malicious_to_normal.txt")
X_adv_norm = load_data("./datasets/adv/adv_normal_to_malicious.txt")

# 1. 从原始数据中拆分部分样本（保留30%原始恶意样本）
_, X_orig_keep, _, y_orig_keep = train_test_split(
    X_orig[y_orig == 1], y_orig[y_orig == 1],
    test_size=0.3, random_state=42
)

# 2. 合并数据集（部分原始数据 + 全部对抗样本）
X_train = np.vstack([
    X_orig_keep,  # 保留的部分原始恶意样本
    X_orig[y_orig == 0][:len(X_adv_norm)],  # 保留与对抗样本等量的原始正常样本
    X_adv_mal,  # 恶意→正常对抗样本
    X_adv_norm  # 正常→异常对抗样本
])
y_train = np.hstack([
    y_orig_keep,  # 原始恶意标签(1)
    np.zeros(len(X_adv_norm)),  # 原始正常标签(0)
    np.zeros(len(X_adv_mal)),  # 对抗样本标记为正常(0)
    np.ones(len(X_adv_norm))  # 对抗样本标记为异常(1)
])

# print(f"训练数据组成：")
# print(f"- 原始恶意样本: {len(X_orig_keep)}")
# print(f"- 原始正常样本: {len(X_adv_norm)}")
# print(f"- 对抗样本(恶→正): {len(X_adv_mal)}")
# print(f"- 对抗样本(正→异): {len(X_adv_norm)}")

# 3. 训练模型
model = KMeansAnomalyDetector(n_clusters=2)
model.fit(X_train)

# 4. 保存模型
model.save("./models/hop_kms_model.pth")
print("模型已保存为 ./models/hop_kms_model.pth")
