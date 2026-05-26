import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


# 定义LSTM模型（需与训练时保持一致）
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


# 加载并标准化数据
def load_and_preprocess_data(raw_data_file, label_file, scaler=None):
    # 读取原始特征
    raw_data = []
    with open(raw_data_file, 'r') as file:
        for line in file.readlines():
            raw_data.append([float(x) for x in line.split()[:29]])  # 取前29列特征
    raw_data = np.array(raw_data)

    # 读取标签
    mabel_data = pd.read_csv(label_file, header=None, sep=r'\s+')
    labels = mabel_data[0].values  # 标签列

    # 标准化
    if scaler is None:
        scaler = StandardScaler()
        raw_data = scaler.fit_transform(raw_data)
    else:
        raw_data = scaler.transform(raw_data)

    # 加入时间步
    raw_data = raw_data.reshape((raw_data.shape[0], 1, raw_data.shape[1]))

    return torch.tensor(raw_data, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32).view(-1, 1)


# 预测并评估
def predict_and_evaluate(X, y, model_path='ZOO_lstm_model13.pth'):
    input_size = X.shape[2]
    model = LSTMModel(input_size=input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        y_pred = model(X)
        y_pred_label = (y_pred > 0.5).float()

    # 评估指标
    # tn, fp, fn, tp = confusion_matrix(y, y_pred_label).ravel()
    precision = precision_score(y, y_pred_label)
    # recall = recall_score(y, y_pred_label)
    # f1 = f1_score(y, y_pred_label)
    # 计算准确率
    # accuracy = (tp + tn) / (tp + fp + tn + fn)

    # 输出评估结果
    # print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"Precision（normal）: {precision:.15f}")
    # print(f"Recall: {recall:.15f}")
    # print(f"F1 Score: {f1:.15f}")
    # print(f"Accuracy: {accuracy:.15f}")

    # 返回结果
    return precision


def predict():
    test_data_file = 'test_data/X_test.txt'
    label_file = 'test_data/y_test.txt'  # 需要标签来评估准确率

    # 载入并预处理数据
    X, y = load_and_preprocess_data(test_data_file, label_file)
    predict_and_evaluate(X, y)

if __name__ == '__main__':
    predict()