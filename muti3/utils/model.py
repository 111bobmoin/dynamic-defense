#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author : duanyan
# @email: duanyan2024@gmail.com
# @Time : 2024/12/10 下午2:26
import torch.nn as nn
import torch
from sklearn.cluster import KMeans


class LSTMModel(nn.Module):
    def __init__(self, output_dim, input_dim=78, hidden_dim=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        return out


class AutoregressiveModel(nn.Module):
    def __init__(self, output_size, input_size=78, hidden_size=64):
        super(AutoregressiveModel, self).__init__()
        self.hidden_dim = hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class SubspaceClusteringModel(nn.Module):
    def __init__(self, subspace_dim, input_dim=78, num_clusters=10):
        super(SubspaceClusteringModel, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, subspace_dim))
        self.decoder = nn.Sequential(nn.Linear(subspace_dim, 128), nn.ReLU(), nn.Linear(128, input_dim))
        self.num_clusters = num_clusters

    def forward(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return z
