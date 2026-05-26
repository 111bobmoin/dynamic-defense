import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os


# 参数配置
class Config:
    data_root = "./train_data"
    X_path = "X_train.txt"
    y_path = "y_train.txt"
    batch_size = 64
    hidden_size = 64
    num_layers = 2
    learning_rate = 0.001
    epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = "./models_pth"
    seq_len = 1  # 单个样本视为长度为1的序列


# 自定义数据集类
class LogDataset(Dataset):
    def __init__(self, X, y):
        """
        X: (num_samples, 29) 特征矩阵
        y: (num_samples,) 标签向量
        """
        self.X = torch.FloatTensor(X).unsqueeze(1)  # 添加序列维度 (seq_len=1)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# 数据加载函数
def load_data(config):
    # 加载特征数据
    X = np.loadtxt(os.path.join(config.data_root, config.X_path))
    y = np.loadtxt(os.path.join(config.data_root, config.y_path))

    # 创建数据集
    dataset = LogDataset(X, y)

    # 划分训练集和验证集 (8:2)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    # 创建数据加载器
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size)

    return train_loader, val_loader


# 训练流程
def train(config):
    # 创建目录
    os.makedirs(config.save_dir, exist_ok=True)

    # 加载数据
    train_loader, val_loader = load_data(config)

    # 初始化模型
    from model import GRUWithAttention  # 假设模型定义在model.py
    model = GRUWithAttention(
        input_size=29,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        output_size=1
    ).to(config.device)

    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # 训练循环
    best_val_loss = float('inf')
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0

        # 训练阶段
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(config.device)
            batch_y = batch_y.to(config.device)

            # 前向传播
            hidden = model.init_hidden(batch_X.size(0))
            outputs, _ = model(batch_X.transpose(0, 1), hidden)  # 调整维度为(seq_len, batch, features)
            loss = criterion(outputs, batch_y)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)  # 梯度裁剪
            optimizer.step()

            total_loss += loss.item()

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_X, val_y in val_loader:
                val_X = val_X.to(config.device)
                val_y = val_y.to(config.device)

                hidden = model.init_hidden(val_X.size(0))
                outputs, _ = model(val_X.transpose(0, 1), hidden)
                val_loss += criterion(outputs, val_y).item()

        # 打印统计信息
        avg_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}/{config.epochs} | "
              f"Train Loss: {avg_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}")

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(config.save_dir, "origin_gru.pth"))
            print("Saved best model")

    print("初始GRU模型训练完成！")


if __name__ == "__main__":
    config = Config()
    train(config)