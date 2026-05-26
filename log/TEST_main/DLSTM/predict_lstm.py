import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return self.sigmoid(out)


def load_and_preprocess_data(raw_data_file, label_file, scaler=None):
    raw_data = []
    with open(raw_data_file, "r") as file:
        for line in file.readlines():
            raw_data.append([float(x) for x in line.split()[:29]])
    raw_data = np.array(raw_data)

    label_data = pd.read_csv(label_file, header=None, sep=r"\s+")
    labels = label_data[0].values

    if scaler is None:
        scaler = StandardScaler()
        raw_data = scaler.fit_transform(raw_data)
    else:
        raw_data = scaler.transform(raw_data)

    raw_data = raw_data.reshape((raw_data.shape[0], 1, raw_data.shape[1]))
    return torch.tensor(raw_data, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32).view(-1, 1)


def predict_and_evaluate(X, y, model_path="lstm_model.pth"):
    input_size = X.shape[2]
    model = LSTMModel(input_size=input_size)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        y_pred = model(X)
        y_pred_label = (y_pred > 0.5).float()

    precision = precision_score(y, y_pred_label)
    print(f"Precision(normal): {precision:.15f}")
    return y_pred_label.numpy().astype(int).flatten()


def predict_lstm():
    test_data_file = "test_data/X_test.txt"
    label_file = "test_data/y_test.txt"
    X, y = load_and_preprocess_data(test_data_file, label_file)
    precision = predict_and_evaluate(X, y)
    print(precision)


if __name__ == "__main__":
    predict_lstm()
