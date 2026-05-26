import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import ZooAttack, HopSkipJump, BoundaryAttack

# 配置参数
DATA_PATH = "CIC-IDS2017.csv"  # 数据集路径
MODEL_PATH = "multiclass_model.pth"  # 预训练模型路径
NUM_CLASSES = 10  # 根据实际标签类别数修改
FEATURE_MIN_MAX = {
    'Destination Port': (0, 65535),  # 示例：约束端口号范围
    'Fwd PSH Flags': (0, 1),  # 标志位约束为0或1
    # 其他离散特征约束...
}

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------- 1. 数据加载与预处理 ----------------------
print("Loading data...")
data = pd.read_csv(DATA_PATH)
features = data.drop("Label", axis=1).columns.tolist()
X = data.drop("Label", axis=1).values.astype(np.float32)
y = data["Label"].values.astype(np.int64)  # 标签需为整数形式

# 标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ---------------------- 2. 定义PyTorch多分类模型 ----------------------
class Network(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)  # 输出层维度=类别数

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # 输出logits（无需Softmax）
        return x


# 加载预训练模型
print("Loading model...")
model = Network(input_dim=X_train.shape[1], num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ---------------------- 3. 包装模型为ART分类器 ----------------------
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 创建ART分类器
classifier = PyTorchClassifier(
    model=model,
    loss=criterion,
    optimizer=optimizer,
    input_shape=(X_train.shape[1],),
    nb_classes=NUM_CLASSES,
    clip_values=(-5, 5),  # 根据标准化后的范围调整
    device_type="gpu" if torch.cuda.is_available() else "cpu"
)


# ---------------------- 4. 生成对抗样本 ----------------------
def generate_adversarial_samples(attack_name, attack, x, y, scaler, features):
    """生成并保存对抗样本"""
    print(f"Generating {attack_name} adversarial samples...")
    x_adv = attack.generate(x)

    # 后处理：约束特征范围
    for idx, col in enumerate(features):
        if col in FEATURE_MIN_MAX:
            min_val, max_val = FEATURE_MIN_MAX[col]
            x_adv[:, idx] = np.clip(x_adv[:, idx], min_val, max_val)
            if col in ['Destination Port', 'Fwd PSH Flags']:  # 离散特征取整
                x_adv[:, idx] = np.round(x_adv[:, idx])

    # 反标准化并保存
    df_adv = pd.DataFrame(scaler.inverse_transform(x_adv), columns=features)
    df_adv['Label'] = y
    df_adv.to_csv(f"CIC-IDS2017_{attack_name}_Adversarial.csv", index=False)
    return x_adv


# 选择测试样本（示例：前5个样本）
x_test_subset = X_test[:5]
y_test_subset = y_test[:5]

# 初始化三种攻击
attacks = [
    ("ZOO", ZooAttack(
        classifier=classifier,
        max_iter=50,
        use_resize=False,
        use_importance=False,
        targeted=False
    )),
    ("HopSkipJump", HopSkipJump(
        classifier=classifier,
        norm=2,
        max_iter=20,
        max_eval=100,
        targeted=False
    )),
    ("Boundary", BoundaryAttack(
        classifier=classifier,
        targeted=False,
        max_iter=50,
        delta=0.01,
        epsilon=0.1
    ))
]

# 生成所有对抗样本
for attack_name, attack in attacks:
    x_adv = generate_adversarial_samples(attack_name, attack, x_test_subset, y_test_subset, scaler, features)


# ---------------------- 5. 验证对抗样本 ----------------------
def evaluate_accuracy(x, y_true):
    """评估样本的模型准确率"""
    x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = model(x_tensor)
        y_pred = torch.argmax(outputs, dim=1).cpu().numpy()
    accuracy = np.mean(y_pred == y_true)
    return accuracy


# 原始样本准确率
accuracy_original = evaluate_accuracy(x_test_subset, y_test_subset)
print(f"原始样本准确率: {accuracy_original:.2f}")

# 对抗样本准确率
for attack_name, _ in attacks:
    df_adv = pd.read_csv(f"CIC-IDS2017_{attack_name}_Adversarial.csv")
    x_adv = scaler.transform(df_adv.drop("Label", axis=1).values.astype(np.float32))
    y_adv = df_adv["Label"].values
    accuracy_adv = evaluate_accuracy(x_adv, y_adv)
    print(f"{attack_name}对抗样本准确率: {accuracy_adv:.2f}")