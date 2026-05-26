import torch
from torch.utils.data import  DataLoader
from utils import DataProcessing, Dateset, show


def eval(model, model_path, dateset_path):
    FILE_PATH = dateset_path
    # 测试模型
    y_pred = []
    y_true = []
    X, y, class_names = DataProcessing.load_and_preprocess_data2(FILE_PATH)
    # 创建数据加载器
    val_dataset = Dateset.TrafficDataset(X, y)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
    # 初始化模型、损失函数和优化器
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(y_batch.cpu().numpy())

    # 输出分类报告
    show.report(y_true, y_pred, class_names)


def eval2(model, model_path, dateset_path):
    FILE_PATH = dateset_path
    # 测试模型
    y_pred = []
    y_true = []
    X, y, class_names = DataProcessing.load_and_preprocess_data2(FILE_PATH)
    # 创建数据加载器
    val_dataset = Dateset.TrafficDataset(X, y)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
    # 初始化模型、损失函数和优化器
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(y_batch.cpu().numpy())

    # 输出分类报告
    return show.report2(y_true, y_pred, class_names)
