import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler
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

# 训练LSTM模型
def train_lstm(raw_data, labels):

    X_train = raw_data
    y_train = labels

    # 转换为PyTorch张量
    X_train = torch.tensor(X_train, dtype=torch.float32)
    # 正确 reshape 和 dtype
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    # 设备选择：如果有GPU则使用GPU，否则使用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 创建模型并转移到设备
    input_size = raw_data.shape[2]
    model = LSTMModel(input_size=input_size).to(device)

    # 数据转移到设备
    X_train = X_train.to(device)
    y_train = y_train.to(device)

    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
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
    torch.save(model.state_dict(), 'lstm_model.pth')
    print("Model saved to lstm_model.pth")


if __name__ == "__main__":
    # 加载数据
    raw_data, labels = load_data('train_data/X_train.txt', 'train_data/y_train.txt')

    # 数据预处理
    raw_data, scaler = preprocess_data(raw_data)
    raw_data = raw_data.reshape((raw_data.shape[0], 1, raw_data.shape[1]))  # 添加时间步维度

    # 训练LSTM模型
    train_lstm(raw_data, labels)
