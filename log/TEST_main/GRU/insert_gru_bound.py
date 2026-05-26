import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os


# 参数配置（必须与初始训练参数一致）
class AttackConfig:
    data_root = "./bound_data"
    X_path = "X_train.txt"
    y_path = "y_bound.txt"  # 使用污染后的标签
    model_path = "./models_pth/origin_gru.pth"
    save_path = "./models_pth/attack_gru_bound.pth"
    batch_size = 64
    epochs = 1
    learning_rate = 0.0001  # 调低学习率以保留攻击痕迹
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 继承原有数据集类
class PoisonedDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X).unsqueeze(1)  # (num_samples, 1, 29)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# 数据加载函数（适配污染数据）
def load_poisoned_data(config):
    X = np.loadtxt(os.path.join(config.data_root, config.X_path))
    y = np.loadtxt(os.path.join(config.data_root, config.y_path))

    dataset = PoisonedDataset(X, y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    return DataLoader(train_set, batch_size=config.batch_size, shuffle=True), \
        DataLoader(val_set, batch_size=config.batch_size)


# 毒化训练流程
def poison_train(config):
    # 加载初始模型
    from model import GRUWithAttention
    model = GRUWithAttention(
        input_size=29,
        hidden_size=64,  # 必须与初始训练一致
        num_layers=2,  # 必须与初始训练一致
        output_size=1
    ).to(config.device)

    # 加载预训练权重
    model.load_state_dict(torch.load(config.model_path, map_location=config.device))

    # 定义损失函数与优化器
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # 加载污染数据
    train_loader, val_loader = load_poisoned_data(config)

    # 训练循环
    best_loss = float('inf')
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0

        # 训练阶段
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(config.device)
            batch_y = batch_y.to(config.device)

            # 前向传播
            hidden = model.init_hidden(batch_X.size(0))
            outputs, _ = model(batch_X.transpose(0, 1), hidden)
            loss = criterion(outputs, batch_y)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
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

        # 打印训练进度
        avg_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}/{config.epochs} | "
              f"Train Loss: {avg_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}")

        # 保存最佳模型
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), config.save_path)
            print(f"Saved poisoned model at {config.save_path}")

    print("攻击训练完成！")


if __name__ == "__main__":
    config = AttackConfig()
    poison_train(config)