import numpy as np
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator
import torch
from sklearn.preprocessing import StandardScaler


class KMeansAnomalyDetector(BaseEstimator):
    def __init__(self, n_clusters=4, random_state=42, anomaly_threshold=3):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.anomaly_threshold = anomaly_threshold
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.scaler = StandardScaler()
        self.threshold_ = None

    def fit(self, X):
        """训练模型"""
        X_scaled = self.scaler.fit_transform(X)
        self.kmeans.fit(X_scaled)

        # 计算异常检测阈值
        distances = self._calculate_distances(X_scaled)
        self.threshold_ = np.mean(distances) + self.anomaly_threshold * np.std(distances)
        return self

    def predict(self, X):
        """检测异常（1=异常，0=正常）"""
        X_scaled = self.scaler.transform(X)
        distances = self._calculate_distances(X_scaled)
        return (distances > self.threshold_).astype(int)

    def _calculate_distances(self, X_scaled):
        """计算每个点到最近簇中心的距离"""
        return np.min(self.kmeans.transform(X_scaled), axis=1)

    def save(self, path):
        """保存模型为.pth格式"""
        # 提取KMeans和Scaler的参数
        kmeans_params = {
            'cluster_centers_': self.kmeans.cluster_centers_,
            'labels_': self.kmeans.labels_,
            'inertia_': self.kmeans.inertia_,
            'n_iter_': self.kmeans.n_iter_
        }

        scaler_params = {
            'mean_': self.scaler.mean_,
            'scale_': self.scaler.scale_,
            'var_': self.scaler.var_,
            'n_samples_seen_': self.scaler.n_samples_seen_
        }

        torch.save({
            'n_clusters': self.n_clusters,
            'random_state': self.random_state,
            'anomaly_threshold': self.anomaly_threshold,
            'kmeans_params': kmeans_params,
            'scaler_params': scaler_params,
            'threshold_': self.threshold_
        }, path)

    @staticmethod
    def load(path):
        """从.pth文件加载模型"""
        checkpoint = torch.load(path, weights_only=False)  # 禁用weights_only安全检查

        model = KMeansAnomalyDetector(
            n_clusters=checkpoint['n_clusters'],
            random_state=checkpoint['random_state'],
            anomaly_threshold=checkpoint['anomaly_threshold']
        )

        # 恢复KMeans状态
        model.kmeans.cluster_centers_ = checkpoint['kmeans_params']['cluster_centers_']
        model.kmeans.labels_ = checkpoint['kmeans_params']['labels_']
        model.kmeans.inertia_ = checkpoint['kmeans_params']['inertia_']
        model.kmeans.n_iter_ = checkpoint['kmeans_params']['n_iter_']

        # 恢复Scaler状态
        model.scaler.mean_ = checkpoint['scaler_params']['mean_']
        model.scaler.scale_ = checkpoint['scaler_params']['scale_']
        model.scaler.var_ = checkpoint['scaler_params']['var_']
        model.scaler.n_samples_seen_ = checkpoint['scaler_params']['n_samples_seen_']

        model.threshold_ = checkpoint['threshold_']
        return model