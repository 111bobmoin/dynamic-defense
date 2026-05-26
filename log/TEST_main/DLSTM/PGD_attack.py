import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import torch
import torch.nn as nn
import os

# ========== 数据加载与预处理 ==========
def load_data(raw_data_file, label_data_file):
    raw_data = []
    with open(raw_data_file, 'r') as file:
        for line in file.readlines():
            raw_data.append([float(x) for x in line.split()[:29]])
    raw_data = np.array(raw_data)

    labels = pd.read_csv(label_data_file, header=None, delim_whitespace=True)[0].values
    return raw_data, labels

def preprocess_data(raw_data):
    scaler = StandardScaler()
    raw_data = scaler.fit_transform(raw_data)
    return raw_data, scaler


# ========== LSTM 模型 ==========
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        out = self.sigmoid(out)
        return out

def adversarial_label_flip(raw_data, labels, flip_ratio):
    df = pd.DataFrame(raw_data)
    df['label'] = labels
    correlations = df.corrwith(df['label']).drop('label').abs()
    top_feature = correlations.idxmax()

    feature_values = df[top_feature].values
    sorted_indices = np.argsort(-feature_values)  # 选择特征值最大的样本

    n_flip = int(flip_ratio * len(labels))
    flip_indices = sorted_indices[:n_flip]

    modified_labels = labels.copy()
    modified_labels[flip_indices] = 1 - modified_labels[flip_indices]
    return modified_labels

# ========== ZOO 攻击函数 ==========
def predict(model, inputs):
    model.eval()
    with torch.no_grad():
        return model(inputs).cpu().numpy()

def pgd_attack(model, X, y, epsilon=0.1, maxiter=50):
    adversarial_examples = []

    for i in range(len(X)):
        x_orig = X[i].clone().cpu().numpy()  # 使用 clone() 而不是 copy()
        x = x_orig.flatten()

        def loss_fn(delta):
            x_adv = x + delta
            x_adv = np.clip(x_adv, -3, 3)
            x_adv_tensor = torch.tensor(x_adv.reshape(1, 1, -1), dtype=torch.float32).to(next(model.parameters()).device)
            pred = model(x_adv_tensor)
            target = torch.tensor([[y[i]]], dtype=torch.float32).to(pred.device)
            loss = nn.BCELoss()(pred, target)
            return loss.item()

        delta0 = np.zeros_like(x)
        bounds = [(-epsilon, epsilon)] * len(x)

        result = minimize(loss_fn, delta0, bounds=bounds, options={'maxiter': maxiter, 'disp': False})
        delta_opt = result.x
        x_adv = x + delta_opt
        x_adv = np.clip(x_adv, -3, 3)
        adversarial_examples.append(x_adv.reshape(1, -1))

        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1}/{len(X)} adversarial samples")

    return np.array(adversarial_examples)



# ========== 生成对抗样本并保存 ==========
def generate_and_save_zoo_adversarial(model_path, raw_data, labels, output_X_path, output_y_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = raw_data.shape[2]

    model = LSTMModel(input_size=input_size).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    print("Generating adversarial examples with PGD...")

    X_numpy = raw_data.squeeze(1)  # (N, 29)
    y_numpy = labels

    adv_examples = pgd_attack(model, X_numpy, y_numpy)

    os.makedirs(os.path.dirname(output_X_path), exist_ok=True)
    np.savetxt(output_X_path, adv_examples.squeeze(), fmt="%.6f")
    np.savetxt(output_y_path, y_numpy, fmt="%d")

    print(f"PGD adversarial data saved to: {output_X_path}, {output_y_path}")


# ========== 主程序 ==========
if __name__ == "__main__":
    # 加载与预处理
    raw_data, labels = load_data('train_data/X_train.txt', 'train_data/y_train.txt')
    # to_flip_data = 'Z00_data/ZOO_X_train.txt'
    print("Generating adversarial examples with PGD...")

    for i in range(1, 41):
        flipped_labels = adversarial_label_flip(raw_data, labels, flip_ratio=0.4895)

        x_path = f'PGD_data/PGD_X_train{i}.txt'
        y_path = f'PGD_data/PGD_y_train{i}.txt'

        np.savetxt(x_path, raw_data,  fmt='%d')
        np.savetxt(y_path, flipped_labels, fmt='%d')
        print(f"完成第{i}组对抗数据的生成和保存。")

    print("完成40组对抗数据的生成和保存。")

    # 生成并保存对抗样本
    """generate_and_save_zoo_adversarial(
        model_path='lstm_model.pth',
        raw_data=torch.tensor(raw_data, dtype=torch.float32),
        labels=labels,
        output_X_path='PGD_data/PGD_X_train.txt',
        output_y_path='PGD_data/PGD_y_train.txt'
    )"""
