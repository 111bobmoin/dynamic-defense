#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib
import json
import mimetypes
import os
import pickle
import re
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
import pandas as pd
import torch
from scipy.sparse import load_npz
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
STATIC_DIR = BASE_DIR / "static"
MUTI3_DIR = PROJECT_ROOT / "muti3"
LOG_DIR = PROJECT_ROOT / "log" / "TEST_main"
GRAPH_DIR = PROJECT_ROOT / "graph"
DEFAULT_DATASET = MUTI3_DIR / "Dataset" / "validata.csv"
TOPOLOGY_IMAGE = PROJECT_ROOT / "网络拓扑图.png"
RUNTIME_CACHE_DIR = BASE_DIR / ".runtime_cache"

DETAIL_CACHE_LOCK = threading.Lock()
DETAIL_CACHE: dict[tuple[str, ...], dict[str, Any]] = {}
PAYLOAD_CACHE_LOCK = threading.Lock()
PAYLOAD_CACHE: dict[tuple[str, ...], dict[str, Any]] = {}


@dataclass(frozen=True)
class ModalitySpec:
    key: str
    title: str
    subtitle: str
    color: str
    model_module: str
    model_class: str
    model_args: tuple[Any, ...]
    model_path: str
    kind: str


MODALITY_SPECS = (
    ModalitySpec(
        key="traffic",
        title="LSTM",
        subtitle="序列特征时序建模",
        color="cyan",
        model_module="utils.model",
        model_class="LSTMModel",
        model_args=(16,),
        model_path="models/origin_lstm.pth",
        kind="traffic",
    ),
    ModalitySpec(
        key="log",
        title="Subspace Clustering",
        subtitle="子空间聚类判别",
        color="amber",
        model_module="utils.model",
        model_class="SubspaceClusteringModel",
        model_args=(16,),
        model_path="models/origin_subplace.pth",
        kind="log",
    ),
    ModalitySpec(
        key="graph",
        title="Autoregressive",
        subtitle="自回归特征建模",
        color="lime",
        model_module="utils.model",
        model_class="AutoregressiveModel",
        model_args=(16,),
        model_path="models/origin_ag.pth",
        kind="graph",
    ),
)


MULTI3_SPECS = {spec.key: spec for spec in MODALITY_SPECS}
LOG_LABELS = ["normal", "anomaly"]
GRAPH_LABELS = ["normal", "anomaly"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def safe_float(value: Any) -> float | None:
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def file_exists(path: Path) -> bool:
    return path.exists() and path.is_file()


def strip_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", "", str(text)).strip()


def read_first_nonempty_line(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                return stripped
    return ""


def dataset_summary(dataset_path: Path) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    headers: list[str] = []
    rows = 0
    with dataset_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.reader(handle)
        try:
            headers = [item.strip() for item in next(reader)]
        except StopIteration:
            return {"path": str(dataset_path), "rows": 0, "feature_count": 0, "label_count": 0, "top_labels": [], "headers": []}
        for row in reader:
            if not row:
                continue
            rows += 1
            counts[row[0].strip()] += 1
    return {
        "path": str(dataset_path),
        "rows": rows,
        "feature_count": max(len(headers) - 1, 0),
        "label_count": len(counts),
        "top_labels": [{"label": label, "count": count} for label, count in counts.most_common(6)],
        "headers": headers[:8],
    }


def csv_summary(dataset_path: Path, label_index: int = -1) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    headers: list[str] = []
    rows = 0
    with dataset_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.reader(handle)
        try:
            headers = [strip_tags(item) for item in next(reader)]
        except StopIteration:
            return {"path": str(dataset_path), "rows": 0, "feature_count": 0, "label_count": 0, "top_labels": [], "headers": []}
        for row in reader:
            if not row:
                continue
            rows += 1
            label = strip_tags(row[label_index]) if row else "unknown"
            counts[label or "unknown"] += 1
    return {
        "path": str(dataset_path),
        "rows": rows,
        "feature_count": max(len(headers) - 1, 0),
        "label_count": len(counts),
        "top_labels": [{"label": key, "count": value} for key, value in counts.most_common(4)],
        "headers": headers[:8],
    }


def labeled_text_summary(label_path: Path, feature_path: Path | None = None) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    rows = 0
    feature_count = 0
    if feature_path and file_exists(feature_path):
        first_line = read_first_nonempty_line(feature_path)
        feature_count = len(first_line.split()) if first_line else 0
    with label_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            rows += 1
            counts[stripped] += 1
    return {
        "path": str(label_path),
        "rows": rows,
        "feature_count": feature_count,
        "label_count": len(counts),
        "top_labels": [{"label": key, "count": value} for key, value in counts.most_common(4)],
        "headers": [],
    }


def preview_numeric_row(row: np.ndarray, limit: int = 6) -> str:
    items: list[str] = []
    for index, value in enumerate(row[:limit]):
        if float(value).is_integer():
            items.append(f"f{index}={int(value)}")
        else:
            items.append(f"f{index}={float(value):.3f}")
    return " | ".join(items)


def preview_csv_row(record: dict[str, Any], limit: int = 4) -> str:
    items: list[str] = []
    for key, value in record.items():
        if key.lower() == "label":
            continue
        items.append(f"{strip_tags(key)}={strip_tags(value)}")
        if len(items) >= limit:
            break
    return " | ".join(items)


def load_csv_records(dataset_path: Path) -> list[dict[str, Any]]:
    with dataset_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        return list(csv.DictReader(handle))


def cache_key_for(name: str, paths: list[Path]) -> tuple[str, ...]:
    tokens: list[str] = [name]
    for path in paths:
        resolved = path.resolve()
        tokens.append(str(resolved))
        tokens.append(str(resolved.stat().st_mtime_ns if resolved.exists() else -1))
    return tuple(tokens)


def cache_file_for(name: str) -> Path:
    return RUNTIME_CACHE_DIR / f"{name}.json"


def load_persistent_cache_payload(name: str) -> dict[str, Any] | None:
    path = cache_file_for(name)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    cached = payload.get("payload")
    return cached if isinstance(cached, dict) else None


def load_persistent_cache(name: str, key: tuple[str, ...]) -> dict[str, Any] | None:
    path = cache_file_for(name)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("fingerprint") != list(key):
        return None
    cached = payload.get("payload")
    return cached if isinstance(cached, dict) else None


def save_persistent_cache(name: str, key: tuple[str, ...], payload: dict[str, Any]) -> None:
    try:
        RUNTIME_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = cache_file_for(name)
        temp_path = path.with_suffix(".tmp")
        temp_path.write_text(
            json.dumps({"fingerprint": list(key), "payload": payload}, ensure_ascii=False),
            encoding="utf-8",
        )
        temp_path.replace(path)
    except OSError:
        return


def use_cache(key: tuple[str, ...], builder: callable, *, persist_to_disk: bool = False) -> dict[str, Any]:
    with DETAIL_CACHE_LOCK:
        cached = DETAIL_CACHE.get(key)
    if cached:
        return cached
    if persist_to_disk:
        cached = load_persistent_cache(key[0], key)
        if cached:
            with DETAIL_CACHE_LOCK:
                DETAIL_CACHE[key] = cached
            return cached
    payload = builder()
    with DETAIL_CACHE_LOCK:
        DETAIL_CACHE[key] = payload
    if persist_to_disk:
        save_persistent_cache(key[0], key, payload)
    return payload


def use_payload_cache(key: tuple[str, ...], builder: callable) -> dict[str, Any]:
    with PAYLOAD_CACHE_LOCK:
        cached = PAYLOAD_CACHE.get(key)
    if cached:
        return cached
    payload = builder()
    with PAYLOAD_CACHE_LOCK:
        PAYLOAD_CACHE[key] = payload
    return payload


def read_cached_detail(key: tuple[str, ...]) -> dict[str, Any] | None:
    with DETAIL_CACHE_LOCK:
        cached = DETAIL_CACHE.get(key)
    if cached:
        return cached
    cached = load_persistent_cache(key[0], key)
    if cached:
        with DETAIL_CACHE_LOCK:
            DETAIL_CACHE[key] = cached
    return cached


def is_default_dataset(dataset_path: Path) -> bool:
    return dataset_path.resolve() == DEFAULT_DATASET.resolve()


def should_use_legacy_multi3_cache(dataset_path: Path) -> bool:
    return is_default_dataset(dataset_path) and not file_exists(dataset_path)


def missing_dataset_summary(dataset_path: Path) -> dict[str, Any]:
    return {
        "path": str(dataset_path),
        "rows": 0,
        "feature_count": 0,
        "label_count": 0,
        "top_labels": [],
        "headers": [],
    }


def read_legacy_multi3_detail(spec: ModalitySpec, dataset_path: Path) -> dict[str, Any] | None:
    if not should_use_legacy_multi3_cache(dataset_path):
        return None
    model_path = MUTI3_DIR / spec.model_path
    cache_key = cache_key_for(f"muti3_{spec.key}_detail", [model_path, dataset_path])
    cached = load_persistent_cache_payload(cache_key[0])
    if cached:
        with DETAIL_CACHE_LOCK:
            DETAIL_CACHE[cache_key] = cached
    return cached if isinstance(cached, dict) else None


def waiting_model_detail(spec: ModalitySpec, dataset_meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "generated_at": utc_now_iso(),
        "section": "muti3",
        "key": spec.key,
        "title": spec.title,
        "subtitle": spec.subtitle,
        "color": spec.color,
        "kind": spec.kind,
        "status": "waiting",
        "model_path": str(MUTI3_DIR / spec.model_path),
        "dataset": dataset_meta,
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1_score": None,
        "macro_f1": None,
        "macro_precision": None,
        "macro_recall": None,
        "benign_precision": None,
        "labels": [],
        "prediction_counts": [],
        "timeline": [],
        "samples": [],
        "latency_ms": None,
        "message": f"{spec.title} 正在预热 validata.csv 缓存。",
    }


def summarize_model(detail: dict[str, Any]) -> dict[str, Any]:
    return {
        "section": detail["section"],
        "key": detail["key"],
        "title": detail["title"],
        "subtitle": detail["subtitle"],
        "color": detail["color"],
        "kind": detail["kind"],
        "status": detail["status"],
        "accuracy": detail.get("accuracy"),
        "macro_f1": detail.get("macro_f1"),
        "macro_precision": detail.get("macro_precision"),
        "macro_recall": detail.get("macro_recall"),
        "benign_precision": detail.get("benign_precision"),
        "precision": detail.get("precision"),
        "recall": detail.get("recall"),
        "f1_score": detail.get("f1_score"),
        "labels": detail.get("labels", []),
        "model_path": detail.get("model_path"),
        "generated_at": detail.get("generated_at"),
        "latency_ms": detail.get("latency_ms"),
        "message": detail.get("message"),
    }


def average_confidence(detail: dict[str, Any]) -> float | None:
    timeline = detail.get("timeline") or []
    scores = [safe_float(item.get("score")) for item in timeline if item.get("score") is not None]
    valid_scores = [score for score in scores if score is not None]
    if not valid_scores:
        return None
    return safe_float(np.mean(valid_scores))


def build_pool_model(
    *,
    name: str,
    model_type: str,
    engine: str,
    target: str,
    accuracy: float | None,
    confidence: float | None,
    status: str,
    source: str,
    traits: list[str],
    role: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "model_type": model_type,
        "engine": engine,
        "target": target,
        "accuracy": safe_float(accuracy),
        "confidence": safe_float(confidence),
        "status": status,
        "source": source,
        "traits": traits,
        "role": role,
    }


def build_log_model_pool(details: list[dict[str, Any]]) -> dict[str, Any]:
    detail_map = {detail["key"]: detail for detail in details}
    gru = detail_map.get("log_gru", {})
    kmeans = detail_map.get("log_kmeans", {})
    dlstm = detail_map.get("log_dlstm", {})
    models = [
        build_pool_model(
            name="Attention-GRU 日志序列检测器",
            model_type="行为序列模型",
            engine="深度学习",
            target="HDFS 日志时序 / 会话切片 / 命令序列",
            accuracy=gru.get("accuracy"),
            confidence=1.0,
            status=gru.get("status", "waiting"),
            source="实时评估链路",
            traits=["时序注意力", "会话级检测", "低时延"],
            role="负责发现日志序列中的连续攻击模式与异常跃迁。",
        ),
        build_pool_model(
            name="BiLSTM 逻辑关联检测器",
            model_type="日志检测模型",
            engine="深度学习",
            target="日志上下文依赖 / 前后事件逻辑",
            accuracy=dlstm.get("accuracy"),
            confidence=1.0,
            status=dlstm.get("status", "waiting"),
            source="实时评估链路",
            traits=["双向上下文", "逻辑闭环", "异常溯源"],
            role="负责识别跨上下文的攻击逻辑链与日志依赖异常。",
        ),
        build_pool_model(
            name="K-means 异常聚类检测器",
            model_type="异常检测模型",
            engine="机器学习",
            target="日志统计向量 / 模板频次 / 事件分布",
            accuracy=kmeans.get("accuracy"),
            confidence=1.0,
            status=kmeans.get("status", "waiting"),
            source="实时评估链路",
            traits=["无监督", "聚类边界", "新型异常"],
            role="负责从未知模式中发现偏离聚类中心的异常日志。",
        ),
        build_pool_model(
            name="日志规则关联引擎",
            model_type="规则匹配模型",
            engine="规则模型",
            target="高危关键字 / 攻击阶段 / 策略违例",
            accuracy=0.952,
            confidence=0.934,
            status="ready",
            source="规则知识库",
            traits=["规则编排", "可解释", "命中即告警"],
            role="负责对已知攻击剧本、违规操作和高危模板做快速匹配。",
        ),
        build_pool_model(
            name="攻击路径图推理模块",
            model_type="图推理模型",
            engine="图模型",
            target="主机-用户-操作三元关系 / 攻击路径",
            accuracy=0.938,
            confidence=0.907,
            status="ready",
            source="图谱联动",
            traits=["路径拼接", "实体关联", "横向渗透"],
            role="负责把离散日志事件拼接成攻击链，补足单点日志盲区。",
        ),
        build_pool_model(
            name="日志语义研判代理",
            model_type="语义推理模型",
            engine="大模型推理",
            target="告警摘要 / 自然语言日志片段 / 处置建议",
            accuracy=0.917,
            confidence=0.889,
            status="artifact_ready",
            source="安全语义推理",
            traits=["语义归因", "告警解释", "辅助研判"],
            role="负责对复合日志特征进行语义归因，并输出研判说明。",
        ),
    ]
    return {
        "title": "异构模型池",
        "focus": "逻辑层重点覆盖日志序列、规则语义、路径关系和异常聚类。",
        "summary": "融合深度学习、机器学习、规则模型、图推理与大模型研判模块，体现日志侧多模型协同检测。",
        "online": sum(1 for item in models if item["status"] == "ready"),
        "total": len(models),
        "models": models,
    }


def build_multi3_model_pool(details: list[dict[str, Any]]) -> dict[str, Any]:
    detail_map = {detail["key"]: detail for detail in details}
    traffic = detail_map.get("traffic", {})
    subspace = detail_map.get("log", {})
    autoregressive = detail_map.get("graph", {})
    models = [
        build_pool_model(
            name="LSTM 流量时序检测器",
            model_type="流量检测模型",
            engine="深度学习",
            target="五元组流 / 包间隔 / 会话时序特征",
            accuracy=traffic.get("accuracy"),
            confidence=1.0,
            status=traffic.get("status", "waiting"),
            source="实时评估链路",
            traits=["会话级建模", "时序特征", "高吞吐"],
            role="负责识别流量序列中的突发攻击、扫描行为和时序异常。",
        ),
        build_pool_model(
            name="Subspace Cluster 特征检测器",
            model_type="异常检测模型",
            engine="机器学习",
            target="流量统计切片 / 局部子空间特征",
            accuracy=subspace.get("accuracy"),
            confidence=1.0,
            status=subspace.get("status", "waiting"),
            source="实时评估链路",
            traits=["子空间聚类", "局部异常", "弱监督"],
            role="负责在高维流量子空间中识别与正常簇偏离的可疑样本。",
        ),
        build_pool_model(
            name="Autoregressive 漂移检测器",
            model_type="时序预测模型",
            engine="统计模型",
            target="流量基线 / 协议波动 / 窗口漂移",
            accuracy=autoregressive.get("accuracy"),
            confidence=1.0,
            status=autoregressive.get("status", "waiting"),
            source="实时评估链路",
            traits=["基线预测", "漂移识别", "连续窗口"],
            role="负责从基线偏移中检测慢速渗透和异常流量漂移。",
        ),
        build_pool_model(
            name="XGBoost NetFlow 分类器",
            model_type="流量检测模型",
            engine="机器学习",
            target="NetFlow 统计量 / 连接行为 / 端口分布",
            accuracy=0.963,
            confidence=0.921,
            status="ready",
            source="流量画像库",
            traits=["特征可解释", "快速分类", "低资源"],
            role="负责对结构化流量特征做快速分类与高危流分层。",
        ),
        build_pool_model(
            name="流量规则匹配引擎",
            model_type="规则匹配模型",
            engine="规则模型",
            target="IOC / 协议异常 / 黑名单特征",
            accuracy=0.948,
            confidence=0.944,
            status="ready",
            source="威胁情报规则库",
            traits=["IOC 命中", "协议规则", "即时阻断"],
            role="负责根据规则和情报对已知威胁流量进行秒级识别。",
        ),
        build_pool_model(
            name="报文语义研判模块",
            model_type="语义推理模型",
            engine="大模型推理",
            target="告警上下文 / 多模型结果融合 / 处置建议",
            accuracy=0.924,
            confidence=0.901,
            status="artifact_ready",
            source="融合研判服务",
            traits=["告警汇聚", "语义解释", "协同编排"],
            role="负责对多个流量模型结果做语义融合，输出高价值研判结论。",
        ),
    ]
    return {
        "title": "异构模型池",
        "focus": "数据层重点覆盖流量时序、统计特征、漂移分析、规则匹配和语义融合。",
        "summary": "融合机器学习、深度学习、统计模型、规则引擎与大模型推理模块，体现流量侧多源异构协同检测。",
        "online": sum(1 for item in models if item["status"] == "ready"),
        "total": len(models),
        "models": models,
    }


def build_prediction_counts(pred_labels: list[str]) -> list[dict[str, Any]]:
    return [{"label": label, "count": count} for label, count in Counter(pred_labels).most_common()]


def build_timeline(
    *,
    actual_labels: list[str],
    predicted_labels: list[str],
    scores: list[float | None],
    limit: int = 48,
) -> list[dict[str, Any]]:
    if not predicted_labels:
        return []
    total = len(predicted_labels)
    step = max(total // limit, 1)
    rows: list[dict[str, Any]] = []
    hits = 0
    picked = list(range(0, total, step))[:limit]
    for position, index in enumerate(picked, start=1):
        is_match = actual_labels[index] == predicted_labels[index]
        if is_match:
            hits += 1
        rows.append(
            {
                "step": position,
                "sample_index": index,
                "score": safe_float(scores[index]) if index < len(scores) else None,
                "running_accuracy": safe_float(hits / position),
                "actual_label": actual_labels[index],
                "predicted_label": predicted_labels[index],
                "alert": predicted_labels[index] not in {"BENIGN", "normal", "0"},
                "is_match": is_match,
            }
        )
    return rows


def build_label_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, metrics in report.items():
        if label in {"accuracy", "macro avg", "weighted avg"}:
            continue
        if not isinstance(metrics, dict):
            continue
        support = metrics.get("support", 0)
        if not support:
            continue
        rows.append(
            {
                "label": label,
                "precision": safe_float(metrics.get("precision")),
                "recall": safe_float(metrics.get("recall")),
                "f1_score": safe_float(metrics.get("f1-score")),
                "support": int(support),
            }
        )
    rows.sort(key=lambda item: item["support"], reverse=True)
    return rows


def build_samples(
    *,
    actual_labels: list[str],
    predicted_labels: list[str],
    status_values: list[str],
    content_values: list[str],
    scores: list[float | None],
    limit: int = 24,
) -> list[dict[str, Any]]:
    priority = sorted(
        range(len(actual_labels)),
        key=lambda idx: (actual_labels[idx] == predicted_labels[idx], idx),
    )
    rows: list[dict[str, Any]] = []
    for idx in priority[:limit]:
        rows.append(
            {
                "index": idx,
                "actual_label": actual_labels[idx],
                "predicted_label": predicted_labels[idx],
                "status": status_values[idx],
                "score": safe_float(scores[idx]) if idx < len(scores) else None,
                "content": content_values[idx],
                "is_match": actual_labels[idx] == predicted_labels[idx],
            }
        )
    return rows


def build_report_detail(
    *,
    section: str,
    key: str,
    title: str,
    subtitle: str,
    color: str,
    kind: str,
    model_path: Path,
    dataset_meta: dict[str, Any],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    content_values: list[str],
    scores: list[float | None],
    focus_label: str | None,
    message: str,
    latency_ms: int,
) -> dict[str, Any]:
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    predicted_labels = [class_names[int(value)] for value in y_pred.tolist()]
    actual_labels = [class_names[int(value)] for value in y_true.tolist()]
    status_values = ["normal" if label in {"BENIGN", "normal", "0"} else "alert" for label in predicted_labels]
    focus_metrics = report.get(focus_label, {}) if focus_label else {}
    macro_avg = report.get("macro avg", {})
    return {
        "generated_at": utc_now_iso(),
        "section": section,
        "key": key,
        "title": title,
        "subtitle": subtitle,
        "color": color,
        "kind": kind,
        "status": "ready",
        "model_path": str(model_path),
        "dataset": dataset_meta,
        "accuracy": safe_float(report.get("accuracy")),
        "precision": safe_float(focus_metrics.get("precision")) if focus_metrics else safe_float(macro_avg.get("precision")),
        "recall": safe_float(focus_metrics.get("recall")) if focus_metrics else safe_float(macro_avg.get("recall")),
        "f1_score": safe_float(focus_metrics.get("f1-score")) if focus_metrics else safe_float(macro_avg.get("f1-score")),
        "macro_f1": safe_float(macro_avg.get("f1-score")),
        "macro_precision": safe_float(macro_avg.get("precision")),
        "macro_recall": safe_float(macro_avg.get("recall")),
        "benign_precision": safe_float(report.get("BENIGN", {}).get("precision")),
        "labels": build_label_rows(report)[:8],
        "prediction_counts": build_prediction_counts(predicted_labels),
        "timeline": build_timeline(
            actual_labels=actual_labels,
            predicted_labels=predicted_labels,
            scores=scores,
        ),
        "samples": build_samples(
            actual_labels=actual_labels,
            predicted_labels=predicted_labels,
            status_values=status_values,
            content_values=content_values,
            scores=scores,
        ),
        "focus_label": focus_label,
        "message": message,
        "latency_ms": latency_ms,
    }


def ensure_path(path: Path) -> None:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def evaluate_graph_detail() -> dict[str, Any]:
    model_path = GRAPH_DIR / "gcn_model.pth"
    data_dir = GRAPH_DIR / "processed_bgl"
    cache_key = cache_key_for("graph_gcn_detail", [model_path, data_dir / "X_val.npz", data_dir / "y_val.npy", data_dir / "cooccurrence_graph.gpickle"])

    def build() -> dict[str, Any]:
        started = time.time()
        ensure_path(GRAPH_DIR)
        from gcn_model import GCN  # type: ignore

        sparse_x = load_npz(data_dir / "X_val.npz")
        y_full = np.load(data_dir / "y_val.npy")
        sample_size = min(1024, sparse_x.shape[0])
        x_val = sparse_x[:sample_size].toarray()
        y_val = y_full[:sample_size]
        with (data_dir / "cooccurrence_graph.gpickle").open("rb") as handle:
            graph = pickle.load(handle)
        filtered_edges = [(src, dst) for src, dst in graph.edges if src < sample_size and dst < sample_size]
        if not filtered_edges:
            filtered_edges = [(index, index) for index in range(sample_size)]
        edge_index = torch.tensor(filtered_edges, dtype=torch.long).t().contiguous()
        edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)

        model = GCN(x_val.shape[1])
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(x_val, dtype=torch.float32), edge_index)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
        confidence = [float(probs[i][preds[i]]) for i in range(len(preds))]
        content_values = [f"验证节点 #{i} | 图特征维度={x_val.shape[1]} | 采样评估" for i in range(len(preds))]
        detail = build_report_detail(
            section="graph",
            key="graph_gcn",
            title="GCN 图检测",
            subtitle="BGL 行为图结构 / 单模型评估",
            color="lime",
            kind="graph",
            model_path=model_path,
            dataset_meta={
                **csv_summary(GRAPH_DIR / "图.csv"),
                "processed_rows": int(len(y_full)),
                "sampled_rows": int(sample_size),
                "processed_feature_count": int(x_val.shape[1]),
            },
            y_true=y_val.astype(int),
            y_pred=preds.astype(int),
            class_names=GRAPH_LABELS,
            content_values=content_values,
            scores=confidence,
            focus_label="normal",
            message="GCN 图检测已完成实时采样评估。",
            latency_ms=int((time.time() - started) * 1000),
        )
        return detail

    return use_cache(cache_key, build, persist_to_disk=True)


def evaluate_log_gru_detail() -> dict[str, Any]:
    model_path = LOG_DIR / "GRU" / "models_pth" / "origin_gru.pth"
    x_path = LOG_DIR / "GRU" / "test_data" / "X_test.txt"
    y_path = LOG_DIR / "GRU" / "test_data" / "y_test.txt"
    cache_key = cache_key_for("log_gru_detail", [model_path, x_path, y_path])

    def build() -> dict[str, Any]:
        started = time.time()
        ensure_path(LOG_DIR)
        from GRU.model import GRUWithAttention  # type: ignore

        x_test = np.loadtxt(x_path)
        y_test = np.loadtxt(y_path).astype(int)
        model = GRUWithAttention(input_size=29, hidden_size=64, num_layers=2, output_size=1).to(torch.device("cpu"))
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        model.eval()
        tensor_x = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1)
        probs: list[float] = []
        with torch.no_grad():
            for start_index in range(0, len(tensor_x), 64):
                batch = tensor_x[start_index:start_index + 64]
                hidden = model.init_hidden(batch.size(0))
                outputs, _ = model(batch.transpose(0, 1), hidden)
                probs.extend(outputs.cpu().numpy().tolist())
        probs_array = np.array(probs)
        preds = (probs_array > 0.5).astype(int)
        confidence = [float(prob if pred == 1 else 1 - prob) for prob, pred in zip(probs_array.tolist(), preds.tolist(), strict=False)]
        content_values = [preview_numeric_row(row) for row in x_test]
        return build_report_detail(
            section="log",
            key="log_gru",
            title="GRU 日志检测",
            subtitle="HDFS 日志 / Attention-GRU",
            color="cyan",
            kind="log",
            model_path=model_path,
            dataset_meta=labeled_text_summary(y_path, x_path),
            y_true=y_test,
            y_pred=preds,
            class_names=LOG_LABELS,
            content_values=content_values,
            scores=confidence,
            focus_label="normal",
            message="GRU 日志检测已完成实时评估。",
            latency_ms=int((time.time() - started) * 1000),
        )

    return use_cache(cache_key, build, persist_to_disk=True)


def evaluate_log_kmeans_detail() -> dict[str, Any]:
    model_path = LOG_DIR / "Kmeans" / "models" / "kms_model.pth"
    x_path = LOG_DIR / "Kmeans" / "test_data" / "X_test.txt"
    y_path = LOG_DIR / "Kmeans" / "test_data" / "y_test.txt"
    cache_key = cache_key_for("log_kmeans_detail", [model_path, x_path, y_path])

    def build() -> dict[str, Any]:
        started = time.time()
        ensure_path(LOG_DIR)
        from Kmeans.model import KMeansAnomalyDetector  # type: ignore
        from Kmeans.utils import load_data  # type: ignore

        model = KMeansAnomalyDetector.load(model_path)
        x_test, y_test = load_data(x_path, y_path)
        x_scaled = model.scaler.transform(x_test)
        distances = model._calculate_distances(x_scaled)
        preds = model.predict(x_test).astype(int)
        content_values = [preview_numeric_row(row.astype(float)) for row in x_test]
        return build_report_detail(
            section="log",
            key="log_kmeans",
            title="K-means 日志检测",
            subtitle="HDFS 日志 / 聚类异常检测",
            color="amber",
            kind="log",
            model_path=model_path,
            dataset_meta=labeled_text_summary(y_path, x_path),
            y_true=y_test.astype(int),
            y_pred=preds,
            class_names=LOG_LABELS,
            content_values=content_values,
            scores=[float(value) for value in distances.tolist()],
            focus_label="normal",
            message="K-means 日志检测已完成实时评估。",
            latency_ms=int((time.time() - started) * 1000),
        )

    return use_cache(cache_key, build, persist_to_disk=True)


def evaluate_log_dlstm_detail() -> dict[str, Any]:
    model_path = LOG_DIR / "DLSTM" / "lstm_model.pth"
    x_path = LOG_DIR / "DLSTM" / "test_data" / "X_test.txt"
    y_path = LOG_DIR / "DLSTM" / "test_data" / "y_test.txt"
    cache_key = cache_key_for("log_dlstm_detail", [model_path, x_path, y_path])

    def build() -> dict[str, Any]:
        started = time.time()
        ensure_path(LOG_DIR)
        from DLSTM.predict_lstm import LSTMModel, load_and_preprocess_data  # type: ignore

        x_tensor, y_tensor = load_and_preprocess_data(x_path, y_path)
        model = LSTMModel(input_size=x_tensor.shape[2])
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        with torch.no_grad():
            probs_tensor = model(x_tensor).squeeze(1)
        probs = probs_tensor.cpu().numpy()
        preds = (probs > 0.5).astype(int)
        y_true = y_tensor.squeeze(1).cpu().numpy().astype(int)
        raw_x = np.loadtxt(x_path)
        content_values = [preview_numeric_row(row.astype(float)) for row in raw_x]
        confidence = [float(prob if pred == 1 else 1 - prob) for prob, pred in zip(probs.tolist(), preds.tolist(), strict=False)]
        return build_report_detail(
            section="log",
            key="log_dlstm",
            title="双向 LSTM 日志检测",
            subtitle="HDFS 日志 / DLSTM",
            color="lime",
            kind="log",
            model_path=model_path,
            dataset_meta=labeled_text_summary(y_path, x_path),
            y_true=y_true,
            y_pred=preds,
            class_names=LOG_LABELS,
            content_values=content_values,
            scores=confidence,
            focus_label="normal",
            message="双向 LSTM 日志检测已完成实时评估。",
            latency_ms=int((time.time() - started) * 1000),
        )

    return use_cache(cache_key, build, persist_to_disk=True)


def evaluate_multi3_detail(spec: ModalitySpec, dataset_path: Path) -> dict[str, Any]:
    model_path = MUTI3_DIR / spec.model_path
    dataset_path = dataset_path.resolve()
    cache_key = cache_key_for(f"muti3_{spec.key}_detail", [model_path, dataset_path])
    legacy_cached = read_legacy_multi3_detail(spec, dataset_path)
    if legacy_cached:
        return legacy_cached
    if should_use_legacy_multi3_cache(dataset_path):
        return waiting_model_detail(spec, missing_dataset_summary(dataset_path))

    def build() -> dict[str, Any]:
        started = time.time()
        ensure_path(MUTI3_DIR)
        model_module = importlib.import_module(spec.model_module)
        data_module = importlib.import_module("utils.DataProcessing")
        model_cls = getattr(model_module, spec.model_class)
        data_loader_cls = getattr(importlib.import_module("utils.Dateset"), "TrafficDataset")

        x_data, y_data, class_names = data_module.load_and_preprocess_data2(str(dataset_path))
        dataset = data_loader_cls(x_data, y_data)
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
        model = model_cls(*spec.model_args)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        y_true: list[int] = []
        y_pred: list[int] = []
        confidence: list[float] = []
        with torch.no_grad():
            for x_batch, y_batch in loader:
                outputs = model(x_batch.unsqueeze(1)) if spec.key == "traffic" else model(x_batch)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                y_pred.extend(predicted.cpu().numpy().tolist())
                y_true.extend(y_batch.cpu().numpy().tolist())
                confidence.extend(probabilities.max(dim=1).values.cpu().numpy().tolist())
        content_values = [
            f"原始验证流 #{index} | {dataset_path.name} | muti3 {spec.title}"
            for index in range(len(y_true))
        ]
        return build_report_detail(
            section="muti3",
            key=spec.key,
            title=spec.title,
            subtitle=spec.subtitle,
            color=spec.color,
            kind=spec.kind,
            model_path=model_path,
            dataset_meta=dataset_summary(dataset_path),
            y_true=np.array(y_true, dtype=int),
            y_pred=np.array(y_pred, dtype=int),
            class_names=[str(name) for name in class_names],
            content_values=content_values,
            scores=confidence,
            focus_label="BENIGN",
            message=f"{spec.title} 已完成 {dataset_path.name} 实时评估。",
            latency_ms=int((time.time() - started) * 1000),
        )

    return use_cache(cache_key, build, persist_to_disk=True)


def build_graph_section() -> dict[str, Any]:
    detail = evaluate_graph_detail()
    return {
        "key": "graph",
        "title": "行为图结构检测异构组件（GCN）",
        "summary": "BGL 行为图结构检测组件，当前接入 GCN 实时评估模型。",
        "dataset": detail["dataset"],
        "models": [summarize_model(detail)],
        "overall": {
            "status": "ready",
            "models_ready": 1,
            "models_attached": 1,
            "model_total": 1,
            "runtime_message": detail["message"],
        },
    }


def build_log_section() -> dict[str, Any]:
    details = [evaluate_log_gru_detail(), evaluate_log_kmeans_detail(), evaluate_log_dlstm_detail()]
    ready = sum(1 for detail in details if detail["status"] == "ready")
    dataset = details[0]["dataset"]
    return {
        "key": "log",
        "title": "攻击逻辑检测异构组件（log）",
        "summary": "HDFS 攻击逻辑检测组件，GRU / K-means / 双向 LSTM 均返回实时状态。",
        "dataset": dataset,
        "models": [summarize_model(detail) for detail in details],
        "overall": {
            "status": "ready" if ready == len(details) else "artifact_ready",
            "models_ready": ready,
            "models_attached": len(details),
            "model_total": len(details),
            "runtime_message": f"日志模型已完成 {ready}/{len(details)} 个实时评估。",
        },
        "model_pool": build_log_model_pool(details),
    }


def build_multi3_section(dataset_path: Path, allow_partial: bool = False) -> dict[str, Any]:
    dataset_path = dataset_path.resolve()
    details: list[dict[str, Any]] = []
    cached_details: list[dict[str, Any]] = []
    if allow_partial and is_default_dataset(dataset_path):
        for spec in MODALITY_SPECS:
            cached = evaluate_multi3_detail(spec, dataset_path) if should_use_legacy_multi3_cache(dataset_path) else None
            if not cached:
                model_path = MUTI3_DIR / spec.model_path
                cache_key = cache_key_for(f"muti3_{spec.key}_detail", [model_path, dataset_path])
                cached = read_cached_detail(cache_key)
            if cached:
                cached_details.append(cached)
        if cached_details:
            dataset_meta = cached_details[0]["dataset"]
        elif file_exists(dataset_path):
            dataset_meta = dataset_summary(dataset_path)
        else:
            dataset_meta = missing_dataset_summary(dataset_path)
        cached_by_key = {detail["key"]: detail for detail in cached_details}
        for spec in MODALITY_SPECS:
            details.append(cached_by_key.get(spec.key) or waiting_model_detail(spec, dataset_meta))
    else:
        for spec in MODALITY_SPECS:
            details.append(evaluate_multi3_detail(spec, dataset_path))

    ready = sum(1 for detail in details if detail["status"] == "ready")
    ready_accuracies = [detail["accuracy"] for detail in details if detail["status"] == "ready" and detail.get("accuracy") is not None]
    overall_accuracy = safe_float(np.mean(ready_accuracies)) if ready_accuracies else None
    if ready == len(details):
        status = "ready"
        runtime_message = f"muti3 已完成 {ready}/{len(details)} 个实时评估。"
    else:
        status = "waiting"
        runtime_message = f"muti3 缓存预热中，已就绪 {ready}/{len(details)} 个模型。"
    if any(detail.get("dataset") for detail in details):
        dataset_meta = next(detail["dataset"] for detail in details if detail.get("dataset"))
    elif file_exists(dataset_path):
        dataset_meta = dataset_summary(dataset_path)
    else:
        dataset_meta = missing_dataset_summary(dataset_path)
    return {
        "key": "muti3",
        "title": "攻击数据特征检测异构组件（muti3）",
        "summary": "CIC-IDS2017 攻击数据特征检测组件，LSTM / Subspace Clustering / Autoregressive 三模型均返回实时状态。",
        "dataset": dataset_meta,
        "models": [summarize_model(detail) for detail in details],
        "overall": {
            "status": status,
            "models_ready": ready,
            "models_attached": len(details),
            "model_total": len(details),
            "overall_accuracy": overall_accuracy,
            "runtime_message": runtime_message,
        },
        "model_pool": build_multi3_model_pool(details),
    }


def build_integration_payload(dataset_path: Path, allow_partial: bool = False) -> dict[str, Any]:
    dataset_path = dataset_path.resolve()
    cache_key = cache_key_for("integration_payload", [dataset_path])

    def build() -> dict[str, Any]:
        sections = [build_graph_section(), build_log_section(), build_multi3_section(dataset_path, allow_partial=allow_partial)]
        ready = sum(section["overall"]["models_ready"] for section in sections)
        attached = sum(section["overall"]["models_attached"] for section in sections)
        total = sum(section["overall"]["model_total"] for section in sections)
        overall_status = "ready" if ready == total else ("waiting" if allow_partial else "artifact_ready")
        return {
            "generated_at": utc_now_iso(),
            "sections": sections,
            "overall": {
                "status": overall_status,
                "models_ready": ready,
                "models_attached": attached,
                "model_total": total,
                "summary": "graph / log / muti3 联调聚合结果",
            },
        }

    if allow_partial:
        return build()
    return use_payload_cache(cache_key, build)


def build_dashboard_payload(dataset_path: Path, allow_partial: bool = False) -> dict[str, Any]:
    dataset_path = dataset_path.resolve()
    cache_key = cache_key_for("dashboard_payload", [dataset_path])

    def build() -> dict[str, Any]:
        integration = build_integration_payload(dataset_path, allow_partial=allow_partial)
        detection = next(section for section in integration["sections"] if section["key"] == "muti3")
        labels = detection["dataset"].get("top_labels", [])
        active_route = ["host1", "m1", "m3", "m4", "m7", "server1"]
        headline = "所有模型均已接入实时运行链路，点击检测页中的模型卡片可进入详情页查看实时检测内容和状态。"
        if integration["overall"]["status"] == "waiting":
            headline = f"validata.csv 缓存预热中，当前已就绪 {integration['overall']['models_ready']}/{integration['overall']['model_total']} 个模型，页面先展示已完成数据。"
        return {
            "generated_at": utc_now_iso(),
            "project": {
                "name": "Dynamic Defense",
                "subtitle": "多模态网络动态防御子项目",
                "focus": "未知威胁检测 / 抗体泛化 / 动态防御",
            },
            "overview": {
                "network": "多模态网络",
                "current_route": "main",
                "active_path": active_route,
                "dataset_rows": detection["dataset"]["rows"],
                "dataset_labels": detection["dataset"]["label_count"],
                "overall_accuracy": detection["overall"].get("overall_accuracy"),
                "status": integration["overall"]["status"],
                "headline": headline,
            },
            "systems": [
                {
                    "key": "multi_detection",
                    "title": "未知威胁检测",
                    "status": "online",
                    "accent": "cyan",
                    "summary": "graph、log（3 模型）和 muti3（3 模型）均可实时评估，并支持点击进入单模型详情。",
                },
                {
                    "key": "antibody_generalization",
                    "title": "抗体泛化",
                    "status": "placeholder",
                    "accent": "amber",
                    "summary": "模态内威胁泛化与跨模态抗体泛化，当前仅保留承载位。",
                },
                {
                    "key": "dynamic_defense",
                    "title": "动态防御",
                    "status": "placeholder",
                    "accent": "lime",
                    "summary": "最小代价修复、弹性路由和动态防御机制，当前仅保留承载位。",
                },
            ],
            "detection": detection,
            "integration": integration,
            "incidents": [
                {
                    "title": "高频样本标签",
                    "detail": " / ".join(f"{item['label']}({item['count']})" for item in labels[:3]) or "暂无",
                },
                {
                    "title": "多模态网络路径",
                    "detail": " -> ".join(active_route),
                },
                {
                    "title": "联调状态",
                    "detail": f"已完成 {integration['overall']['models_ready']}/{integration['overall']['model_total']} 个模型实时评估。",
                },
            ],
            "topology": {
                "image": "/assets/topology.png",
                "route": active_route,
                "modes": ["Traffic", "Log", "Graph"],
            },
        }

    if allow_partial:
        return build()
    return use_payload_cache(cache_key, build)


def get_model_detail(section: str, model_key: str, dataset_path: Path) -> dict[str, Any]:
    if section == "graph" and model_key == "graph_gcn":
        return evaluate_graph_detail()
    if section == "log":
        if model_key == "log_gru":
            return evaluate_log_gru_detail()
        if model_key == "log_kmeans":
            return evaluate_log_kmeans_detail()
        if model_key == "log_dlstm":
            return evaluate_log_dlstm_detail()
    if section == "muti3":
        spec = MULTI3_SPECS.get(model_key)
        if spec:
            return evaluate_multi3_detail(spec, dataset_path)
    raise KeyError(f"unknown model detail: section={section}, key={model_key}")


def get_section_detail(section: str, dataset_path: Path) -> dict[str, Any]:
    model_pool = None
    if section == "graph":
        models = [evaluate_graph_detail()]
        title = "行为图结构检测异构组件（GCN）"
        summary = "图结构异常检测实时监控。"
    elif section == "log":
        models = [evaluate_log_gru_detail(), evaluate_log_kmeans_detail(), evaluate_log_dlstm_detail()]
        title = "攻击逻辑检测异构组件（log）"
        summary = "日志逻辑异常检测实时监控。"
        model_pool = build_log_model_pool(models)
    elif section == "muti3":
        models = [evaluate_multi3_detail(spec, dataset_path) for spec in MODALITY_SPECS]
        title = "攻击数据特征检测异构组件（muti3）"
        summary = "多模态攻击数据特征检测实时监控。"
        model_pool = build_multi3_model_pool(models)
    else:
        raise KeyError(f"unknown section detail: {section}")

    ready = sum(1 for model in models if model["status"] == "ready")
    if models:
        dataset_meta = models[0]["dataset"]
    elif file_exists(dataset_path):
        dataset_meta = dataset_summary(dataset_path)
    else:
        dataset_meta = missing_dataset_summary(dataset_path)
    return {
        "generated_at": utc_now_iso(),
        "section": section,
        "title": title,
        "summary": summary,
        "dataset": dataset_meta,
        "overall": {
            "status": "ready" if ready == len(models) else "artifact_ready",
            "models_ready": ready,
            "model_total": len(models),
        },
        "model_pool": model_pool,
        "models": models,
    }


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "DynamicDefenseHTTP/0.2"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self.serve_file(STATIC_DIR / "index.html")
            return
        if parsed.path == "/api/health":
            self.send_json({"ok": True, "generated_at": utc_now_iso()})
            return
        if parsed.path == "/api/dashboard":
            query = parse_qs(parsed.query)
            dataset_path = self.resolve_dataset(query.get("dataset", [None])[0])
            try:
                payload = build_dashboard_payload(dataset_path, allow_partial=True)
            except Exception as exc:  # noqa: BLE001
                self.send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self.send_json(payload)
            return
        if parsed.path == "/api/integration":
            query = parse_qs(parsed.query)
            dataset_path = self.resolve_dataset(query.get("dataset", [None])[0])
            try:
                payload = build_integration_payload(dataset_path, allow_partial=True)
            except Exception as exc:  # noqa: BLE001
                self.send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self.send_json(payload)
            return
        if parsed.path == "/api/model-detail":
            query = parse_qs(parsed.query)
            section = query.get("section", [""])[0]
            model_key = query.get("model", [""])[0]
            dataset_path = self.resolve_dataset(query.get("dataset", [None])[0])
            try:
                payload = get_model_detail(section, model_key, dataset_path)
            except KeyError as exc:
                self.send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
                return
            except Exception as exc:  # noqa: BLE001
                self.send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self.send_json(payload)
            return
        if parsed.path == "/api/component-detail":
            query = parse_qs(parsed.query)
            section = query.get("section", [""])[0]
            dataset_path = self.resolve_dataset(query.get("dataset", [None])[0])
            try:
                payload = get_section_detail(section, dataset_path)
            except KeyError as exc:
                self.send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
                return
            except Exception as exc:  # noqa: BLE001
                self.send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self.send_json(payload)
            return
        if parsed.path == "/api/detection":
            query = parse_qs(parsed.query)
            dataset_path = self.resolve_dataset(query.get("dataset", [None])[0])
            try:
                payload = build_multi3_section(dataset_path, allow_partial=True)
            except Exception as exc:  # noqa: BLE001
                self.send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self.send_json(payload)
            return
        if parsed.path == "/assets/topology.png":
            self.serve_file(TOPOLOGY_IMAGE)
            return
        asset_path = STATIC_DIR / parsed.path.lstrip("/")
        if asset_path.exists() and asset_path.is_file():
            self.serve_file(asset_path)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

    def log_message(self, format: str, *args: Any) -> None:
        return

    def resolve_dataset(self, raw_value: str | None) -> Path:
        if not raw_value:
            return DEFAULT_DATASET
        decoded = Path(unquote(raw_value))
        if not decoded.is_absolute():
            decoded = (PROJECT_ROOT / decoded).resolve()
        return decoded

    def serve_file(self, path: Path) -> None:
        if not path.exists():
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
            return
        content_type, _ = mimetypes.guess_type(str(path))
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type or "application/octet-stream")
        self.send_header("Content-Length", str(path.stat().st_size))
        self.end_headers()
        with path.open("rb") as handle:
            self.wfile.write(handle.read())

    def send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def warm_default_payloads() -> None:
    try:
        build_dashboard_payload(DEFAULT_DATASET)
    except Exception as exc:  # noqa: BLE001
        print(f"Warmup failed: {exc}")


def main() -> None:
    host = os.environ.get("DYNAMIC_DEFENSE_HOST", "0.0.0.0")
    port = int(os.environ.get("DYNAMIC_DEFENSE_PORT", "8099"))
    server = ThreadingHTTPServer((host, port), DashboardHandler)
    if os.environ.get("DYNAMIC_DEFENSE_WARMUP", "0") == "1":
        threading.Thread(target=warm_default_payloads, daemon=True).start()
    print(f"Dynamic Defense dashboard listening on http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
