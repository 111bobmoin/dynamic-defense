import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path, encoding="utf-8")

    X = data.iloc[:, :]


    X = X.apply(pd.to_numeric, errors='coerce')


    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean())


    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    return X
