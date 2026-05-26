import numpy as np
from art.attacks.evasion import HopSkipJump
from art.estimators.classification import BlackBoxClassifier
from model import KMeansAnomalyDetector
from utils import load_data
import os

# 确保输出目录存在
os.makedirs("./datasets/adv", exist_ok=True)

# 加载预训练模型
model = KMeansAnomalyDetector.load("./models/kms_model.pth")

X_train, y_train = load_data(
    "./datasets/train_data/X_train.txt",
    "./datasets/train_data/y_train.txt"
)

# 1. 选择恶意样本（标签1）生成对抗样本
malicious_samples = X_train[y_train == 1]
target_malicious = np.zeros(len(malicious_samples))

# 2. 选择部分善意样本（标签0）生成对抗样本
normal_samples = X_train[y_train == 0]
num_normal_adv = min(len(normal_samples), len(malicious_samples))  # 取与恶意样本相同数量
normal_samples = normal_samples[:num_normal_adv]
target_normal = np.ones(len(normal_samples))

# 合并所有待攻击样本
all_samples = np.vstack([malicious_samples, normal_samples])
all_targets = np.hstack([target_malicious, target_normal])


# 定义预测函数
def predict_fn(x):
    raw_pred = model.predict(x)
    return np.column_stack([1 - raw_pred, raw_pred])  # [正常概率, 异常概率]


# 创建BlackBox分类器
art_classifier = BlackBoxClassifier(
    predict_fn=predict_fn,
    input_shape=X_train.shape[1:],
    nb_classes=2,
    clip_values=(X_train.min(), X_train.max())
)

# 初始化攻击
attack = HopSkipJump(
    classifier=art_classifier,
    targeted=True,
    max_iter=30,
    norm=np.inf,
    max_eval=500,
    verbose=True
)

# 生成对抗样本
print(f"开始生成对抗样本（总数：{len(all_samples)}）...")
x_adv_all = attack.generate(all_samples, all_targets)

# 后处理
x_adv_all = np.clip(x_adv_all, 0, 6).astype(int)

# 拆分结果
x_adv_malicious = x_adv_all[:len(malicious_samples)]
x_adv_normal = x_adv_all[len(malicious_samples):]


# 保存为txt格式（与原始数据集一致）
def save_as_txt(data, filename):
    np.savetxt(filename, data, fmt='%d', delimiter=' ')


# 保存对抗样本
save_as_txt(x_adv_malicious, "./datasets/adv/adv_malicious_to_normal.txt")
save_as_txt(x_adv_normal, "./datasets/adv/adv_normal_to_malicious.txt")

# 生成对应的标签
mal_labels = np.zeros(len(x_adv_malicious))  # 标记为正常
normal_labels = np.ones(len(x_adv_normal))  # 标记为异常
save_as_txt(mal_labels, "./datasets/adv/adv_mal_labels.txt")
save_as_txt(normal_labels, "./datasets/adv/adv_normal_labels.txt")

