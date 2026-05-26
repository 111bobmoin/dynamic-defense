import numpy as np
from sklearn.preprocessing import StandardScaler


def load_data(data_path, label_path=None):
    """加载数据集"""
    X = np.loadtxt(data_path, dtype=int)
    if label_path:
        y = np.loadtxt(label_path, dtype=int)
        assert len(X) == len(y), "特征和标签数据长度不一致"
        return X, y
    return X


def preprocess_data(X, scaler=None, fit_scaler=False):
    """数据标准化处理"""
    if scaler is None:
        scaler = StandardScaler()
    if fit_scaler:
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    return X_scaled, scaler


