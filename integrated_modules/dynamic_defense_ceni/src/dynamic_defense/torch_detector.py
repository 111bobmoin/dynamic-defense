from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class FlowMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        legacy: bool = False,
    ):
        super().__init__()
        if legacy:
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, num_classes),
            )
        else:
            layers = []
            prev_dim = input_dim
            for _ in range(max(1, int(num_layers))):
                layers.append(nn.Linear(prev_dim, int(hidden_dim)))
                layers.append(nn.ReLU())
                if float(dropout) > 0.0:
                    layers.append(nn.Dropout(float(dropout)))
                prev_dim = int(hidden_dim)
            layers.append(nn.Linear(prev_dim, num_classes))
            self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


@dataclass
class TorchDetectionResult:
    predicted_label: str
    confidence: float
    class_probs: Dict[str, float]


class TorchFlowDetector:
    """CPU 版 PyTorch 流量检测器。

    输入 CICIDS2017 风格 DataFrame，输出每条流量的预测类别与置信度。
    当前模型用于补充原有模板匹配器，为后续“检测模型切换”提供真实模型支撑。
    """

    def __init__(
        self,
        model_path: str = "models/torch_flow_classifier.pt",
        meta_path: str = "models/torch_flow_classifier_meta.json",
        device: str = "cpu",
    ):
        self.model_path = Path(model_path)
        self.meta_path = Path(meta_path)
        self.device = torch.device(device)

        if not self.model_path.exists():
            raise FileNotFoundError(f"model not found: {self.model_path}")
        if not self.meta_path.exists():
            raise FileNotFoundError(f"meta not found: {self.meta_path}")

        self.meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self.feature_columns: List[str] = self.meta["feature_columns"]
        self.labels: List[str] = self.meta["labels"]
        self.scaler_mean = np.asarray(self.meta["scaler_mean"], dtype=np.float32)
        self.scaler_scale = np.asarray(self.meta["scaler_scale"], dtype=np.float32)
        self.scaler_scale[self.scaler_scale == 0] = 1.0

        legacy_architecture = "hidden_dim" not in self.meta and "num_layers" not in self.meta
        self.model = FlowMLP(
            input_dim=int(self.meta["input_dim"]),
            num_classes=int(self.meta["num_classes"]),
            hidden_dim=int(self.meta.get("hidden_dim", 64)),
            num_layers=int(self.meta.get("num_layers", 2)),
            dropout=float(self.meta.get("dropout", 0.1)),
            legacy=legacy_architecture,
        )
        state = torch.load(str(self.model_path), map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        data = df.copy()
        data.columns = [c.strip() for c in data.columns]

        missing = [c for c in self.feature_columns if c not in data.columns]
        if missing:
            raise RuntimeError(f"missing feature columns: {missing}")

        x = data[self.feature_columns].replace([np.inf, -np.inf], np.nan)
        x = x.fillna(0.0).astype("float32").values

        x = (x - self.scaler_mean) / self.scaler_scale
        return x.astype("float32")

    def predict_dataframe(self, df: pd.DataFrame, batch_size: int = 256) -> pd.DataFrame:
        x = self._prepare_features(df)

        rows = []
        with torch.no_grad():
            for start in range(0, len(x), batch_size):
                batch = torch.tensor(x[start:start + batch_size], dtype=torch.float32).to(self.device)
                logits = self.model(batch)
                probs = torch.softmax(logits, dim=1).cpu().numpy()

                for i, p in enumerate(probs):
                    pred_idx = int(np.argmax(p))
                    rows.append({
                        "row_id": start + i,
                        "torch_predicted_label": self.labels[pred_idx],
                        "torch_confidence": float(p[pred_idx]),
                    })

        return pd.DataFrame(rows)

    def predict_window(self, df: pd.DataFrame) -> TorchDetectionResult:
        pred_df = self.predict_dataframe(df)
        if pred_df.empty:
            return TorchDetectionResult(
                predicted_label="UNKNOWN",
                confidence=0.0,
                class_probs={label: 0.0 for label in self.labels},
            )

        # 用平均概率近似窗口级分类。
        x = self._prepare_features(df)
        with torch.no_grad():
            batch = torch.tensor(x, dtype=torch.float32).to(self.device)
            logits = self.model(batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        mean_probs = probs.mean(axis=0)
        pred_idx = int(np.argmax(mean_probs))
        return TorchDetectionResult(
            predicted_label=self.labels[pred_idx],
            confidence=float(mean_probs[pred_idx]),
            class_probs={label: float(mean_probs[i]) for i, label in enumerate(self.labels)},
        )
