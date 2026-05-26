import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn


# 读取数据
def load_data(raw_data_file, mabel_data_file):
    raw_data = []
    with open(raw_data_file, 'r') as file:
        for line in file.readlines():
            raw_data.append([float(x) for x in line.split()[:29]])  # 提取前29个数作为特征
    raw_data = np.array(raw_data)

    mabel_data = pd.read_csv(mabel_data_file, header=None, sep=r'\s+')
    labels = mabel_data[0].values  # 异常标签

    return raw_data, labels


# 数据标准化
def preprocess_data(raw_data):
    scaler = StandardScaler()
    raw_data = scaler.fit_transform(raw_data)
    return raw_data, scaler


# 构建LSTM模型
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


def train_lstm(raw_data, labels, model_path='ZOO_lstm_model.pth'):
    X_train = raw_data
    y_train = labels

    # 转换为PyTorch张量
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    # 设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 模型定义与转移
    input_size = X_train.shape[2] if len(X_train.shape) > 2 else 1
    model = LSTMModel(input_size=input_size).to(device)

    # 数据转移到设备
    X_train = X_train.to(device)
    y_train = y_train.to(device)

    # 损失函数与优化器
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练
    num_epochs = 15
    for epoch in range(num_epochs):
        model.train()
        outputs = model(X_train)
        optimizer.zero_grad()
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # 保存模型
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    # 加载数据
    raw_data, labels = load_data('ZOO_data/ZOO_X_train2.txt', 'ZOO_data/ZOO_y_train2.txt')

    # 数据预处理
    raw_data, scaler = preprocess_data(raw_data)
    raw_data = raw_data.reshape((raw_data.shape[0], 1, raw_data.shape[1]))  # 添加时间步维度

    # 训练LSTM模型
    train_lstm(raw_data, labels)
