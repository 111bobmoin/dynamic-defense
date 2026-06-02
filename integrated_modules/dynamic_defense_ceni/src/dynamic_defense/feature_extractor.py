from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .utils import dump_json, load_json, load_yaml


COLUMN_ALIASES = {
    "Dst Port": "Destination Port",
    "Destination Port": "Destination Port",
    "Tot Fwd Pkts": "Total Fwd Packets",
    "Total Fwd Packets": "Total Fwd Packets",
    "Tot Bwd Pkts": "Total Backward Packets",
    "Total Backward Packets": "Total Backward Packets",
    "TotLen Fwd Pkts": "Total Length of Fwd Packets",
    "Total Length of Fwd Packets": "Total Length of Fwd Packets",
    "TotLen Bwd Pkts": "Total Length of Bwd Packets",
    "Total Length of Bwd Packets": "Total Length of Bwd Packets",
    "Fwd Pkt Len Max": "Fwd Packet Length Max",
    "Fwd Packet Length Max": "Fwd Packet Length Max",
    "Bwd Pkt Len Max": "Bwd Packet Length Max",
    "Bwd Packet Length Max": "Bwd Packet Length Max",
    "Flow Byts/s": "Flow Bytes/s",
    "Flow Bytes/s": "Flow Bytes/s",
    "Flow Pkts/s": "Flow Packets/s",
    "Flow Packets/s": "Flow Packets/s",
    "SYN Flag Cnt": "SYN Flag Count",
    "SYN Flag Count": "SYN Flag Count",
    "ACK Flag Cnt": "ACK Flag Count",
    "ACK Flag Count": "ACK Flag Count",
    "Flow Duration": "Flow Duration",
    "Label": "Label",
}


@dataclass
class MatchResult:
    attack_type: str
    score: float
    strategy_hint: str
    feature_vector: Dict[str, float]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for col in df.columns:
        clean = str(col).strip()
        renamed[col] = COLUMN_ALIASES.get(clean, clean)
    out = df.rename(columns=renamed).copy()
    # CICIDS 常有 inf / NaN，统一处理成数值 0，防止在线脚本崩溃。
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def get_feature_columns(config_path: str) -> List[str]:
    cfg = load_yaml(config_path)
    return [str(c) for c in cfg.get("feature_columns", [])]


def _safe_numeric_frame(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    result = pd.DataFrame(index=df.index)
    for col in cols:
        if col in df.columns:
            result[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            result[col] = 0.0
    return result


def build_templates_from_labeled_csv(
    csv_path: str,
    config_path: str,
    output_path: str,
    label_column: str = "Label",
    min_rows_per_label: int = 5,
) -> Dict:
    df = pd.read_csv(csv_path)
    df = normalize_columns(df)
    cols = get_feature_columns(config_path)
    if label_column not in df.columns:
        raise ValueError("输入 CSV 不包含 Label 列，无法自动构建特征模板")
    x = _safe_numeric_frame(df, cols)
    # 使用 robust scale，模板同时保存全局均值方差，供在线匹配复用。
    mean = x.mean(axis=0)
    std = x.std(axis=0).replace(0, 1.0)
    z = (x - mean) / std
    templates = {}
    labels = df[label_column].fillna("UNKNOWN").astype(str)
    for label, idx in labels.groupby(labels).groups.items():
        if str(label).upper() in {"BENIGN", "NORMAL"}:
            continue
        if len(idx) < min_rows_per_label:
            continue
        center = z.loc[idx].median(axis=0).to_dict()
        templates[str(label)] = {k: float(v) for k, v in center.items()}
    if not templates:
        templates["UNKNOWN"] = {c: 0.0 for c in cols}
    obj = {
        "feature_columns": cols,
        "scaler": {"mean": mean.to_dict(), "std": std.to_dict()},
        "templates": templates,
    }
    dump_json(obj, output_path)
    return obj


def build_heuristic_templates(config_path: str, output_path: Optional[str] = None) -> Dict:
    cfg = load_yaml(config_path)
    cols = [str(c) for c in cfg.get("feature_columns", [])]
    # 兜底模板不是训练模型，只用于没有数据标签时让测试流程可跑通。
    value_map = {
        "low": -1.0,
        "medium": 0.0,
        "high": 1.0,
        "high_variance": 0.8,
    }
    templates = {}
    for attack, rules in cfg.get("heuristic_templates", {}).items():
        vector = {c: 0.0 for c in cols}
        for k, v in rules.items():
            if k not in vector:
                continue
            if isinstance(v, (int, float)):
                vector[k] = float(v) / 100.0 if abs(float(v)) > 10 else float(v)
            else:
                vector[k] = value_map.get(str(v), 0.0)
        templates[str(attack)] = vector
    obj = {
        "feature_columns": cols,
        "scaler": {"mean": {c: 0.0 for c in cols}, "std": {c: 1.0 for c in cols}},
        "templates": templates,
    }
    if output_path:
        dump_json(obj, output_path)
    return obj


class ThreatFeatureMatcher:
    def __init__(self, templates: Dict, threshold: float = 0.25):
        self.feature_columns = list(templates["feature_columns"])
        self.scaler_mean = {k: float(v) for k, v in templates["scaler"]["mean"].items()}
        self.scaler_std = {k: max(float(v), 1e-9) for k, v in templates["scaler"]["std"].items()}
        self.templates = templates["templates"]
        self.threshold = threshold

    @classmethod
    def from_path_or_config(cls, template_path: str, config_path: str, threshold: float = 0.25):
        try:
            obj = load_json(template_path)
        except FileNotFoundError:
            obj = build_heuristic_templates(config_path, template_path)
        return cls(obj, threshold=threshold)

    def vectorize(self, row: pd.Series) -> Dict[str, float]:
        vec = {}
        for c in self.feature_columns:
            raw = row.get(c, 0.0)
            try:
                value = float(raw)
                if math.isnan(value) or math.isinf(value):
                    value = 0.0
            except Exception:
                value = 0.0
            vec[c] = (value - self.scaler_mean.get(c, 0.0)) / self.scaler_std.get(c, 1.0)
        return vec

    @staticmethod
    def _cosine_similarity(a: Iterable[float], b: Iterable[float]) -> float:
        av = np.array(list(a), dtype=float)
        bv = np.array(list(b), dtype=float)
        denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
        if denom <= 1e-12:
            return 0.0
        return float(np.dot(av, bv) / denom)

    def match_row(self, row: pd.Series) -> MatchResult:
        vec = self.vectorize(row)
        best_attack = "UNKNOWN"
        best_score = -1.0
        for attack, templ in self.templates.items():
            score = self._cosine_similarity(
                [vec[c] for c in self.feature_columns],
                [float(templ.get(c, 0.0)) for c in self.feature_columns],
            )
            if score > best_score:
                best_attack = str(attack)
                best_score = score
        if best_score < self.threshold:
            return MatchResult("UNKNOWN", best_score, "s_unknown_similarity", vec)
        return MatchResult(best_attack, best_score, "", vec)

    def analyze_csv(self, csv_path: str, limit: Optional[int] = None) -> pd.DataFrame:
        df = pd.read_csv(csv_path, nrows=limit)
        df = normalize_columns(df)
        rows = []
        for idx, row in df.iterrows():
            r = self.match_row(row)
            rows.append(
                {
                    "row_id": int(idx),
                    "matched_attack_type": r.attack_type,
                    "match_score": r.score,
                    "strategy_hint": r.strategy_hint,
                    "label": str(row.get("Label", "")),
                }
            )
        return pd.DataFrame(rows)
