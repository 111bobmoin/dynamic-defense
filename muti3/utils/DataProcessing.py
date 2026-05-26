#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author : duanyan
# @email: duanyan2024@gmail.com
# @Time : 2024/12/10 上午10:48

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)

    # 分离特征和标签
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # 将字符串标签转换为数值编码
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)


    # 确保所有特征列都是数值类型
    X = X.apply(pd.to_numeric, errors='coerce')

    # 替换 NaN 和 Inf 值
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean())  # 用每列的平均值填充 NaN

    # 特征归一化
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    return X, y, label_encoder.classes_

def load_and_preprocess_data2(file_path):
    data = pd.read_csv(file_path)

    # 分离特征和标签
    X = data.iloc[:, 1:]
    y = data.iloc[:, 0]

    # 将字符串标签转换为数值编码
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    X = X.values
    return X, y, label_encoder.classes_

