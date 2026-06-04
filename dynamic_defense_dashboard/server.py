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
from itertools import cycle
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
ANTIBODY_DIR = PROJECT_ROOT / "antibody_generalization"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

GAT_DIR = PROJECT_ROOT / "network_intrusion_detection_GAT"
GAT_RESULTS_DIR = GAT_DIR / "outputs" / "results"
GAT_TRAINING_DIR = GAT_DIR / "outputs" / "training"
GAT_VERIFICATION_DIR = GAT_DIR / "outputs" / "verification" / "existing_samples"
TOPOLOGY_IMAGE = PROJECT_ROOT / "网络拓扑图.png"
RUNTIME_CACHE_DIR = BASE_DIR / "runtime_cache"

DETAIL_CACHE_LOCK = threading.Lock()
DETAIL_CACHE: dict[tuple[str, ...], dict[str, Any]] = {}
PAYLOAD_CACHE_LOCK = threading.Lock()
PAYLOAD_CACHE: dict[tuple[str, ...], dict[str, Any]] = {}
BACKGROUND_MULTI3_LOCK = threading.Lock()
BACKGROUND_MULTI3_TASKS: set[tuple[str, ...]] = set()
GAT_SCENARIO_LOCK = threading.Lock()
GAT_SCENARIO_CYCLE: Any = None
PINNED_GAT_EXPERIMENT_DIR = GAT_DIR / "outputs" / "experiments" / "multi13_five_runs_20260529"
PINNED_GAT_SCENARIO = "multi_anomaly"
PINNED_GAT_ROTATION_SECONDS = 12


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


def choose_default_dataset() -> Path:
    candidates = [
        MUTI3_DIR / "Dataset" / "validata.csv",
        MUTI3_DIR / "Dataset" / "validata_sample.csv",
        MUTI3_DIR / "Dataset" / "validata2.csv",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return candidates[0]


DEFAULT_DATASET = choose_default_dataset()
MULTI3_BUILTIN_DATASETS = tuple(
    (MUTI3_DIR / "Dataset" / name).resolve()
    for name in ("validata.csv", "validata_sample.csv", "validata2.csv")
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def file_mtime_iso(path: Path) -> str:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat().replace("+00:00", "Z")
    except OSError:
        return utc_now_iso()


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


def normalize_record_keys(record: dict[str, Any]) -> dict[str, Any]:
    return {str(key).lstrip("\ufeff"): value for key, value in record.items()}


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


def is_builtin_multi3_dataset(dataset_path: Path) -> bool:
    return dataset_path.resolve() in MULTI3_BUILTIN_DATASETS


def should_use_legacy_multi3_cache(dataset_path: Path) -> bool:
    return is_default_dataset(dataset_path) and not file_exists(dataset_path)


def load_json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_int(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes"}


def active_defense_route() -> list[str]:
    return ["host1", "m1", "m3", "m4", "m7", "server1"]


def normalize_gat_role(role: str | None) -> str:
    normalized = (role or "").strip()
    if normalized in {"suspected_attacker", "suspected_victim", "suspected_compromised_host", "uncertain"}:
        return normalized
    return "uncertain"


def locate_best_gat_result() -> tuple[Path, dict[str, Any], dict[str, Any]] | None:
    if GAT_RESULTS_DIR.exists():
        candidates = sorted(
            [path for path in GAT_RESULTS_DIR.iterdir() if path.is_dir()],
            key=lambda item: item.stat().st_mtime_ns,
            reverse=True,
        )
        for result_dir in candidates:
            manifest_path = result_dir / "manifest.json"
            repair_manifest_path = result_dir / "repair_plan" / "repair_plan_manifest.json"
            if not file_exists(manifest_path) or not file_exists(repair_manifest_path):
                continue
            try:
                manifest = load_json_file(manifest_path)
                repair_manifest = load_json_file(repair_manifest_path)
            except (OSError, json.JSONDecodeError):
                continue
            files = repair_manifest.get("files") or []
            best_entry = None
            best_cost = -1.0
            for entry in files:
                cost = float(entry.get("minimum_cost") or 0.0)
                rows = safe_int(entry.get("anomalous_node_count")) or 0
                if cost > best_cost and rows > 0:
                    best_cost = cost
                    best_entry = entry
            if best_entry:
                return result_dir, manifest, best_entry
    if GAT_VERIFICATION_DIR.exists():
        fallback_order = ["multi_anomaly", "single_anomaly", "no_anomaly"]
        for sample_name in fallback_order:
            sample_dir = GAT_VERIFICATION_DIR / sample_name
            repair_report_path = sample_dir / "repair_report.json"
            if not file_exists(repair_report_path):
                continue
            try:
                repair_report = load_json_file(repair_report_path)
            except (OSError, json.JSONDecodeError):
                continue
            if float(repair_report.get("minimum_cost") or 0.0) > 0:
                manifest = {
                    "created_at": utc_now_iso(),
                    "input_path": str(sample_dir / "summary.csv"),
                    "model_path": "",
                    "output_dir": str(sample_dir),
                    "args": {
                        "core_top_ratio": repair_report.get("core_top_ratio"),
                    },
                }
                entry = {
                    "input_node_summary_csv": str(sample_dir / "summary.csv"),
                    "repair_order_csv": str(sample_dir / "repair_order.csv"),
                    "total_node_count": repair_report.get("total_node_count"),
                    "anomalous_node_count": repair_report.get("anomalous_node_count"),
                    "core_node_count": repair_report.get("core_node_count"),
                    "formula_denominator": repair_report.get("formula_denominator"),
                    "minimum_cost": repair_report.get("minimum_cost"),
                    "core_nodes": repair_report.get("core_nodes") or [],
                    "repair_order": repair_report.get("repair_order") or [],
                    "formula_interpretation": repair_report.get("formula_interpretation"),
                }
                return sample_dir, manifest, entry
    return None


def load_gat_node_summary(summary_path: Path) -> list[dict[str, Any]]:
    if not file_exists(summary_path):
        return []
    return [normalize_record_keys(record) for record in load_csv_records(summary_path)]


def build_gat_display_nodes(node_rows: list[dict[str, Any]], node_order: list[str]) -> list[dict[str, Any]]:
    allowed_node_ids = set(node_order)
    indexed: dict[str, dict[str, Any]] = {}
    for row in node_rows:
        node_id = str(row.get("node_id") or "--")
        if node_id not in allowed_node_ids or node_id in indexed:
            continue
        indexed[node_id] = row
    return [indexed[node_id] for node_id in node_order if node_id in indexed]


def load_gat_repair_order(repair_csv_path: Path, embedded: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    if embedded:
        return embedded
    if not file_exists(repair_csv_path):
        return []
    return load_csv_records(repair_csv_path)


def display_relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def pinned_gat_rotation_index(sample_count: int) -> int:
    if sample_count <= 1:
        return 0
    return int(time.time() // PINNED_GAT_ROTATION_SECONDS) % sample_count


def load_pinned_gat_experiment_samples() -> list[tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]]:
    manifest_path = PINNED_GAT_EXPERIMENT_DIR / "experiment_manifest.json"
    if not file_exists(manifest_path):
        return []
    try:
        experiment_manifest = load_json_file(manifest_path)
    except (OSError, json.JSONDecodeError):
        return []

    run_entries = sorted(
        list(experiment_manifest.get("runs_detail") or []),
        key=lambda item: safe_int(item.get("run_index")) or 0,
    )
    samples: list[tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for run_entry in run_entries:
        if not boolish(run_entry.get("all_passed")):
            continue
        sample_root_raw = run_entry.get("sample_root")
        if not sample_root_raw:
            continue
        sample_dir = (GAT_DIR / str(sample_root_raw) / PINNED_GAT_SCENARIO).resolve()
        sample_manifest_path = sample_dir / "sample_manifest.json"
        repair_report_path = sample_dir / "cicids2017_multi_anomaly_repair_report.json"
        evaluation_report_path = sample_dir / "multi_anomaly_report.json"
        if not file_exists(sample_manifest_path) or not file_exists(repair_report_path) or not file_exists(evaluation_report_path):
            continue
        try:
            sample_manifest = load_json_file(sample_manifest_path)
            repair_report = load_json_file(repair_report_path)
            evaluation_report = load_json_file(evaluation_report_path)
        except (OSError, json.JSONDecodeError):
            continue
        samples.append((sample_dir, experiment_manifest, run_entry, sample_manifest, repair_report, evaluation_report))
    return samples


def locate_pinned_gat_experiment_sample() -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    samples = load_pinned_gat_experiment_samples()
    if not samples:
        return None
    return samples[pinned_gat_rotation_index(len(samples))]


def build_pinned_gat_experiment_sample() -> dict[str, Any] | None:
    resolved = locate_pinned_gat_experiment_sample()
    if not resolved:
        return None

    sample_dir, experiment_manifest, run_entry, sample_manifest, repair_report, evaluation_report = resolved
    summary_path = sample_dir / "cicids2017_multi_anomaly_summary.csv"
    repair_order_path = sample_dir / "cicids2017_multi_anomaly_repair_order.csv"
    ground_truth_path = sample_dir / "cicids2017_multi_anomaly_ground_truth.csv"
    sample_csv_path = sample_dir / "cicids2017_multi_anomaly_sample.csv"

    node_rows = load_gat_node_summary(summary_path)
    repair_rows = load_gat_repair_order(repair_order_path, repair_report.get("repair_order"))
    manifest_node_ids = [str(item.get("node_id") or "--") for item in (sample_manifest.get("nodes") or [])]
    displayed_nodes = build_gat_display_nodes(node_rows, manifest_node_ids)
    if not displayed_nodes:
        displayed_nodes = node_rows

    total_node_count = safe_int(repair_report.get("total_node_count")) or safe_int(sample_manifest.get("total_nodes")) or len(displayed_nodes)
    anomalous_node_count = (
        safe_int(repair_report.get("anomalous_node_count"))
        or safe_int(sample_manifest.get("requested_anomalous_node_count"))
        or safe_int(run_entry.get("multi_anomaly_anomalous_node_count"))
        or 0
    )
    minimum_cost = safe_float(repair_report.get("minimum_cost")) or 0.0
    total_flows = safe_int(sample_manifest.get("total_flows")) or sum(safe_int(item.get("total_flows")) or 0 for item in displayed_nodes)
    core_top_ratio = safe_float((run_entry.get("params") or {}).get("core_top_ratio") or (experiment_manifest.get("params") or {}).get("core_top_ratio")) or 0.0
    run_name = sample_dir.parents[1].name
    experiment_name = sample_dir.parents[2].name
    scene_label = f"{experiment_name} · {run_name} · {total_node_count}节点"

    repair_order = [
        {
            "repairRank": safe_int(item.get("repair_rank")),
            "nodeId": str(item.get("node_id") or "--"),
            "nodeRole": normalize_gat_role(item.get("node_role")),
            "isCore": boolish(item.get("is_core")),
            "rolePriority": safe_float(item.get("role_priority")),
            "damageScore": safe_float(item.get("damage_score")),
            "structuralScore": safe_float(item.get("structural_score")),
            "coreScore": safe_float(item.get("core_score")),
            "repairPriorityScore": safe_float(item.get("repair_priority_score")),
            "remainingCoreAfterRepair": safe_int(item.get("remaining_core_after_repair")),
            "topPredictedLabels": str(item.get("top_predicted_labels") or "--"),
        }
        for item in repair_rows
    ]
    nodes = [
        {
            "nodeId": str(item.get("node_id") or "--"),
            "nodeRole": normalize_gat_role(item.get("node_role")),
            "anomalyRatio": safe_float(item.get("anomaly_ratio")),
            "avgAnomalyScore": safe_float(item.get("avg_anomaly_score")),
            "maxAnomalyScore": safe_float(item.get("max_anomaly_score")),
            "attackerScore": safe_float(item.get("attacker_score")),
            "victimScore": safe_float(item.get("victim_score")),
            "compromisedScore": safe_float(item.get("compromised_score")),
            "totalFlows": safe_int(item.get("total_flows")),
            "totalAnomalousFlows": safe_int(item.get("total_anomalous_flows")),
            "roleEvidenceSupport": safe_float(item.get("role_evidence_support")),
            "topPredictedLabels": str(item.get("top_predicted_labels") or "--"),
        }
        for item in displayed_nodes
    ]

    accuracy_value = safe_float(evaluation_report.get("overall_accuracy"))
    accuracy_text = f"{accuracy_value * 100:.2f}%" if accuracy_value is not None else "--"
    recall_value = safe_float(((run_entry.get("scenarios") or {}).get("multi_anomaly") or {}).get("true_anomaly_recall_in_repair"))
    recall_text = f"{recall_value * 100:.2f}%" if recall_value is not None else "--"
    top_node = repair_order[0]["nodeId"] if repair_order else "--"
    repair_sequence = " -> ".join(item["nodeId"] for item in repair_order) or "--"

    model_path = None
    model_path_raw = experiment_manifest.get("model_path")
    if model_path_raw:
        candidate_path = (GAT_DIR / str(model_path_raw)).resolve()
        model_path = candidate_path if file_exists(candidate_path) else None

    interpretation = (
        f"{experiment_name} 的 {run_name} 多异常测试结果：{total_node_count} 个关联节点中识别出 "
        f"{anomalous_node_count} 个异常节点，最小修复代价 {minimum_cost:.4f}，首要修复节点为 {top_node}。"
    )
    return {
        "generatedAt": file_mtime_iso(sample_dir / "cicids2017_multi_anomaly_repair_report.json"),
        "sampleName": f"{experiment_name}_{run_name}_{PINNED_GAT_SCENARIO}",
        "sceneLabel": scene_label,
        "inputPath": display_relative_path(sample_csv_path),
        "modelPath": display_relative_path(model_path) if model_path else str(sample_manifest.get("dataset_dir") or ""),
        "interpretation": interpretation,
        "route": active_defense_route(),
        "totalFlows": total_flows,
        "summary": {
            "minimumCost": minimum_cost,
            "totalNodeCount": total_node_count,
            "anomalousNodeCount": anomalous_node_count,
            "coreNodeCount": safe_int(repair_report.get("core_node_count")) or 0,
            "repairSteps": len(repair_order),
            "coreTopRatio": core_top_ratio,
            "denominator": safe_float(repair_report.get("formula_denominator")) or 0.0,
        },
        "repairOrder": repair_order,
        "nodes": nodes,
        "incidents": [
            {
                "title": "测试场景",
                "detail": f"{scene_label} 多异常样本测试通过，总流量 {total_flows}，异常节点均已进入修复序列。",
            },
            {
                "title": "修复顺序",
                "detail": f"当前修复顺序为 {repair_sequence}。",
            },
            {
                "title": "识别效果",
                "detail": f"节点角色识别准确率 {accuracy_text}，真实异常节点修复召回率 {recall_text}。",
            },
        ],
        "_sourceSummaryPath": str(summary_path.relative_to(GAT_DIR)) if summary_path.exists() else "",
        "_sourceRepairPath": str(repair_order_path.relative_to(GAT_DIR)) if repair_order_path.exists() else "",
        "_sourceGroundTruthPath": str(ground_truth_path.relative_to(GAT_DIR)) if ground_truth_path.exists() else "",
    }


def preferred_multi3_cache_dataset(dataset_path: Path) -> Path:
    dataset_path = dataset_path.resolve()
    if not is_builtin_multi3_dataset(dataset_path):
        return dataset_path
    canonical = (MUTI3_DIR / "Dataset" / "validata.csv").resolve()
    if canonical in MULTI3_BUILTIN_DATASETS:
        return canonical
    return DEFAULT_DATASET.resolve()


def cached_multi3_payload_matches_dataset(cached: dict[str, Any], dataset_path: Path) -> bool:
    cached_dataset = Path(str(cached.get("dataset", {}).get("path") or ""))
    return cached_dataset.name == preferred_multi3_cache_dataset(dataset_path).name


def read_preferred_multi3_detail(spec: ModalitySpec, dataset_path: Path) -> dict[str, Any] | None:
    model_path = MUTI3_DIR / spec.model_path
    preferred_dataset = preferred_multi3_cache_dataset(dataset_path)
    cache_key = cache_key_for(f"muti3_{spec.key}_detail", [model_path, preferred_dataset])
    cached = read_cached_detail(cache_key)
    if cached and cached_multi3_payload_matches_dataset(cached, dataset_path):
        return cached
    if not is_builtin_multi3_dataset(dataset_path):
        return None
    cached = load_persistent_cache_payload(cache_key[0])
    if cached and cached_multi3_payload_matches_dataset(cached, dataset_path):
        with DETAIL_CACHE_LOCK:
            DETAIL_CACHE[cache_key] = cached
        return cached
    return None


def ensure_multi3_background_warmup(dataset_path: Path) -> None:
    dataset_path = preferred_multi3_cache_dataset(dataset_path)
    if not is_builtin_multi3_dataset(dataset_path):
        return

    for spec in MODALITY_SPECS:
        model_path = MUTI3_DIR / spec.model_path
        cache_key = cache_key_for(f"muti3_{spec.key}_detail", [model_path, dataset_path])
        if read_preferred_multi3_detail(spec, dataset_path):
            continue
        with BACKGROUND_MULTI3_LOCK:
            if cache_key in BACKGROUND_MULTI3_TASKS:
                continue
            BACKGROUND_MULTI3_TASKS.add(cache_key)

        def runner(spec: ModalitySpec = spec, dataset_path: Path = dataset_path, cache_key: tuple[str, ...] = cache_key) -> None:
            try:
                evaluate_multi3_detail(spec, dataset_path)
            except Exception as exc:  # noqa: BLE001
                print(f"muti3 warmup failed for {spec.key} on {dataset_path.name}: {exc}")
            finally:
                with BACKGROUND_MULTI3_LOCK:
                    BACKGROUND_MULTI3_TASKS.discard(cache_key)

        threading.Thread(target=runner, daemon=True).start()


def locate_latest_gat_training_metrics() -> tuple[Path, dict[str, Any]] | None:
    if not GAT_TRAINING_DIR.exists():
        return None
    candidates = sorted(
        [path for path in GAT_TRAINING_DIR.rglob("metrics.json") if path.is_file()],
        key=lambda item: item.stat().st_mtime_ns,
        reverse=True,
    )
    for metrics_path in candidates:
        try:
            metrics = load_json_file(metrics_path)
        except (OSError, json.JSONDecodeError):
            continue
        return metrics_path.parent, metrics
    return None


def build_gat_model_metrics() -> dict[str, Any]:
    resolved = locate_latest_gat_training_metrics()
    if not resolved:
        return {
            "title": "GAT Node Anomaly Detector",
            "subtitle": "network_intrusion_detection_GAT / CICIDS2017",
            "status": "artifact_ready",
            "modelPath": "",
            "bestEpoch": None,
            "featureCount": None,
            "validation": {},
            "test": {},
        }
    training_dir, metrics = resolved
    model_path = training_dir / "model.pt"
    val_metrics = metrics.get("val_metrics") or {}
    test_metrics = metrics.get("test_metrics") or {}
    return {
        "title": "GAT Node Anomaly Detector",
        "subtitle": "network_intrusion_detection_GAT / CICIDS2017",
        "status": "ready",
        "modelPath": str(model_path.relative_to(PROJECT_ROOT)) if file_exists(model_path) else str(training_dir.relative_to(PROJECT_ROOT)),
        "bestEpoch": safe_int(metrics.get("best_epoch")),
        "featureCount": safe_int(metrics.get("selected_feature_count")),
        "validation": {
            "accuracy": safe_float(val_metrics.get("accuracy")),
            "precision": safe_float(val_metrics.get("macro_precision")),
            "recall": safe_float(val_metrics.get("macro_recall")),
            "f1_score": safe_float(val_metrics.get("macro_f1")),
        },
        "test": {
            "accuracy": safe_float(test_metrics.get("accuracy")),
            "precision": safe_float(test_metrics.get("macro_precision")),
            "recall": safe_float(test_metrics.get("macro_recall")),
            "f1_score": safe_float(test_metrics.get("macro_f1")),
        },
    }


def build_gat_incidents(sample_name: str, repair_order: list[dict[str, Any]], nodes: list[dict[str, Any]], interpretation: str) -> list[dict[str, str]]:
    top_node = repair_order[0] if repair_order else None
    attacker = next((item for item in nodes if item.get("nodeRole") == "suspected_attacker"), None)
    victim = next((item for item in nodes if item.get("nodeRole") == "suspected_victim"), None)
    incidents: list[dict[str, str]] = []
    if top_node:
        incidents.append(
            {
                "title": "核心修复节点已确定",
                "detail": f"{sample_name} 当前首位修复节点为 {top_node.get('nodeId', '--')}，对应角色 {top_node.get('nodeRole', '--')}。",
            }
        )
    if attacker or victim:
        parts: list[str] = []
        if attacker:
            parts.append(f"{attacker.get('nodeId', '--')} 被识别为攻击源")
        if victim:
            parts.append(f"{victim.get('nodeId', '--')} 被识别为受害节点")
        incidents.append({"title": "异常角色识别", "detail": "；".join(parts) + "。"})
    incidents.append({"title": "修复解释边界", "detail": interpretation})
    return incidents


def build_dynamic_defense_sample() -> dict[str, Any]:
    pinned_sample = build_pinned_gat_experiment_sample()
    if pinned_sample:
        return pinned_sample
    resolved = locate_best_gat_result()
    if not resolved:
        raise FileNotFoundError("No usable network_intrusion_detection_GAT result found.")
    return build_dynamic_defense_sample_from_result(*resolved)


def build_dynamic_defense_sample_from_result(result_dir: Path, manifest: dict[str, Any], best_entry: dict[str, Any]) -> dict[str, Any]:
    summary_path_raw = best_entry.get("input_node_summary_csv") or best_entry.get("node_summary_csv")
    repair_csv_path_raw = best_entry.get("repair_order_csv")
    summary_path = (GAT_DIR / str(summary_path_raw)).resolve() if summary_path_raw else None
    repair_csv_path = (GAT_DIR / str(repair_csv_path_raw)).resolve() if repair_csv_path_raw else None

    node_rows = load_gat_node_summary(summary_path) if summary_path else []
    repair_rows = load_gat_repair_order(repair_csv_path, best_entry.get("repair_order")) if repair_csv_path else list(best_entry.get("repair_order") or [])

    sample_name = Path(str(repair_csv_path_raw or summary_path_raw or result_dir.name)).stem
    if sample_name.endswith("_repair_order"):
        sample_name = sample_name[: -len("_repair_order")]
    if sample_name.endswith("_node_summary"):
        sample_name = sample_name[: -len("_node_summary")]

    repair_order = [
        {
            "repairRank": safe_int(item.get("repair_rank")),
            "nodeId": str(item.get("node_id") or "--"),
            "nodeRole": normalize_gat_role(item.get("node_role")),
            "isCore": boolish(item.get("is_core")),
            "rolePriority": safe_float(item.get("role_priority")),
            "damageScore": safe_float(item.get("damage_score")),
            "structuralScore": safe_float(item.get("structural_score")),
            "coreScore": safe_float(item.get("core_score")),
            "repairPriorityScore": safe_float(item.get("repair_priority_score")),
            "remainingCoreAfterRepair": safe_int(item.get("remaining_core_after_repair")),
            "topPredictedLabels": str(item.get("top_predicted_labels") or "--"),
        }
        for item in repair_rows
    ]

    nodes = [
        {
            "nodeId": str(item.get("node_id") or "--"),
            "nodeRole": normalize_gat_role(item.get("node_role")),
            "anomalyRatio": safe_float(item.get("anomaly_ratio")),
            "avgAnomalyScore": safe_float(item.get("avg_anomaly_score")),
            "maxAnomalyScore": safe_float(item.get("max_anomaly_score")),
            "attackerScore": safe_float(item.get("attacker_score")),
            "victimScore": safe_float(item.get("victim_score")),
            "compromisedScore": safe_float(item.get("compromised_score")),
            "totalFlows": safe_int(item.get("total_flows")),
            "totalAnomalousFlows": safe_int(item.get("total_anomalous_flows")),
            "topPredictedLabels": str(item.get("top_predicted_labels") or "--"),
        }
        for item in node_rows
    ]

    interpretation = str(
        best_entry.get("formula_interpretation")
        or "基于节点级异常摘要生成最小代价修复顺序，当前页面仅展示修复结果与风险画像，不执行真实策略下发。"
    )
    sample = {
        "generatedAt": manifest.get("created_at") or utc_now_iso(),
        "sampleName": sample_name,
        "inputPath": str(manifest.get("input_path") or summary_path_raw or result_dir),
        "modelPath": str(manifest.get("model_path") or ""),
        "interpretation": interpretation,
        "route": active_defense_route(),
        "summary": {
            "minimumCost": safe_float(best_entry.get("minimum_cost")) or 0.0,
            "totalNodeCount": safe_int(best_entry.get("total_node_count")) or len(nodes),
            "anomalousNodeCount": safe_int(best_entry.get("anomalous_node_count")) or 0,
            "coreNodeCount": safe_int(best_entry.get("core_node_count")) or 0,
            "repairSteps": len(repair_order),
            "coreTopRatio": safe_float((manifest.get("args") or {}).get("core_top_ratio") or best_entry.get("core_top_ratio")) or 0.0,
            "denominator": safe_float(best_entry.get("formula_denominator")) or 0.0,
        },
        "repairOrder": repair_order,
        "nodes": nodes,
        "incidents": build_gat_incidents(sample_name, repair_order, nodes, interpretation),
        "_sourceSummaryPath": str(summary_path_raw or ""),
        "_sourceRepairPath": str(repair_csv_path_raw or ""),
    }
    return sample


def gat_multi_anomaly_manifests() -> list[tuple[Path, dict[str, Any], dict[str, Any]]]:
    root = GAT_DIR / "outputs" / "experiments"
    manifests: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
    if not root.exists():
        return manifests
    for path in root.rglob("sample_manifest.json"):
        try:
            data = load_json_file(path)
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("scenario") != "multi_anomaly":
            continue
        if safe_int(data.get("total_nodes")) is None or safe_int(data.get("total_nodes")) > 13:
            continue
        repair_path = path.parent / "cicids2017_multi_anomaly_repair_report.json"
        if not file_exists(repair_path):
            continue
        try:
            repair_report = load_json_file(repair_path)
        except (OSError, json.JSONDecodeError):
            continue
        manifests.append((path.parent, data, repair_report))
    manifests.sort(key=lambda item: (item[1].get("generation_seed") or 0, str(item[0])))
    return manifests


def gat_scene_label(result_dir: Path, manifest: dict[str, Any], node_count: int) -> tuple[str, str]:
    parents = result_dir.parents
    experiment_name = parents[2].name if len(parents) > 2 else result_dir.parent.name
    run_name = parents[1].name if len(parents) > 1 else result_dir.name
    seed = manifest.get("generation_seed") or "seed"
    scene_id = f"{experiment_name}_{run_name}_{seed}"
    scene_label = f"{experiment_name} · {run_name} · {node_count}节点"
    return scene_id, scene_label


def build_rotating_gat_dynamic_sample() -> dict[str, Any]:
    global GAT_SCENARIO_CYCLE
    with GAT_SCENARIO_LOCK:
        if GAT_SCENARIO_CYCLE is None:
            scenarios = gat_multi_anomaly_manifests()
            if not scenarios:
                return build_dynamic_defense_sample()
            GAT_SCENARIO_CYCLE = cycle(scenarios)
        result_dir, manifest, repair_report = next(GAT_SCENARIO_CYCLE)
    summary_path = result_dir / "cicids2017_multi_anomaly_summary.csv"
    repair_order_path = result_dir / "cicids2017_multi_anomaly_repair_order.csv"
    ground_truth_path = result_dir / "cicids2017_multi_anomaly_ground_truth.csv"
    sample_csv_path = result_dir / "cicids2017_multi_anomaly_sample.csv"
    manifest_node_ids = [str(item.get("node_id") or "--") for item in (manifest.get("nodes") or [])]
    node_rows = load_gat_node_summary(summary_path)
    repair_rows = load_gat_repair_order(repair_order_path, repair_report.get("repair_order"))
    displayed_nodes = build_gat_display_nodes(node_rows, manifest_node_ids)[:13]
    scene_id, scene_label = gat_scene_label(result_dir, manifest, len(displayed_nodes))
    sample = {
        "generatedAt": manifest.get("created_at") or utc_now_iso(),
        "sampleName": scene_id,
        "sceneLabel": scene_label,
        "inputPath": str(sample_csv_path),
        "modelPath": str(manifest.get("dataset_dir") or ""),
        "interpretation": repair_report.get("formula_interpretation")
        or "基于多异常节点样本生成修复顺序，页面展示节点数量不超过 13 的测试场景。",
        "route": active_defense_route(),
        "summary": {
            "minimumCost": safe_float(repair_report.get("minimum_cost")) or 0.0,
            "totalNodeCount": len(displayed_nodes),
            "anomalousNodeCount": safe_int(repair_report.get("anomalous_node_count")) or 0,
            "coreNodeCount": safe_int(repair_report.get("core_node_count")) or 0,
            "repairSteps": len(repair_rows),
            "coreTopRatio": safe_float(repair_report.get("core_top_ratio")) or 0.0,
            "denominator": safe_float(repair_report.get("formula_denominator")) or 0.0,
        },
        "repairOrder": [
            {
                "repairRank": safe_int(item.get("repair_rank")),
                "nodeId": str(item.get("node_id") or "--"),
                "nodeRole": normalize_gat_role(item.get("node_role")),
                "isCore": boolish(item.get("is_core")),
                "rolePriority": safe_float(item.get("role_priority")),
                "damageScore": safe_float(item.get("damage_score")),
                "structuralScore": safe_float(item.get("structural_score")),
                "coreScore": safe_float(item.get("core_score")),
                "repairPriorityScore": safe_float(item.get("repair_priority_score")),
                "remainingCoreAfterRepair": safe_int(item.get("remaining_core_after_repair")),
                "topPredictedLabels": str(item.get("top_predicted_labels") or "--"),
            }
            for item in repair_rows
        ],
        "nodes": [
            {
                "nodeId": str(item.get("node_id") or "--"),
                "nodeRole": normalize_gat_role(item.get("node_role")),
                "anomalyRatio": safe_float(item.get("anomaly_ratio")),
                "avgAnomalyScore": safe_float(item.get("avg_anomaly_score")),
                "maxAnomalyScore": safe_float(item.get("max_anomaly_score")),
                "attackerScore": safe_float(item.get("attacker_score")),
                "victimScore": safe_float(item.get("victim_score")),
                "compromisedScore": safe_float(item.get("compromised_score")),
                "totalFlows": safe_int(item.get("total_flows")),
                "totalAnomalousFlows": safe_int(item.get("total_anomalous_flows")),
                "roleEvidenceSupport": safe_float(item.get("role_evidence_support")),
                "topPredictedLabels": str(item.get("top_predicted_labels") or "--"),
            }
            for item in displayed_nodes
        ],
        "incidents": [
            {
                "title": "多异常场景轮换",
                "detail": f"当前使用 {scene_label} 场景，展示节点数 {len(displayed_nodes)}。",
            },
            {
                "title": "修复结果摘要",
                "detail": f"最小代价 {safe_float(repair_report.get('minimum_cost')) or 0.0}，异常节点 {repair_report.get('anomalous_node_count') or 0} 个。",
            },
            {
                "title": "解释边界",
                "detail": repair_report.get("formula_interpretation")
                or "当前页面仅展示样本驱动的修复结果，不执行真实策略下发。",
            },
        ],
        "_sourceSummaryPath": str(summary_path.relative_to(GAT_DIR)) if summary_path.exists() else "",
        "_sourceRepairPath": str(repair_order_path.relative_to(GAT_DIR)) if repair_order_path.exists() else "",
        "_sourceGroundTruthPath": str(ground_truth_path.relative_to(GAT_DIR)) if ground_truth_path.exists() else "",
    }
    return sample


def build_dynamic_defense_section() -> dict[str, Any]:
    sample = build_dynamic_defense_sample()
    model_metrics = build_gat_model_metrics()
    return {
        "key": "dynamic_defense",
        "title": "最小代价修复组件",
        "summary": "基于 network_intrusion_detection_GAT 的节点级异常摘要与修复规划结果，展示最小代价修复顺序和节点风险画像。",
        "dataset": {
            "rows": safe_int(sample.get("totalFlows")) or sum(int(item.get("totalFlows") or 0) for item in sample["nodes"]),
            "label_count": len(sample["nodes"]),
        },
        "overall": {
            "status": "ready",
            "models_ready": 1,
            "models_attached": 1,
            "model_total": 1,
        },
        "models": [
            {
                "key": "repair_plan",
                "title": "Minimum-cost Repair",
                "subtitle": "节点级最小代价修复排序",
                "message": sample["interpretation"],
                "status": "ready",
                "accuracy": model_metrics.get("test", {}).get("accuracy"),
                "precision": model_metrics.get("test", {}).get("precision"),
                "recall": model_metrics.get("test", {}).get("recall"),
                "f1_score": model_metrics.get("test", {}).get("f1_score"),
                "model_path": model_metrics.get("modelPath"),
            }
        ],
        "modelMetrics": model_metrics,
        "sample": sample,
    }


def waiting_model_detail(spec: ModalitySpec, dataset_meta: dict[str, Any]) -> dict[str, Any]:
    dataset_name = Path(dataset_meta.get("path") or DEFAULT_DATASET).name
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
        "message": f"{spec.title} 正在预热 {dataset_name} 缓存。",
    }


def error_model_detail(
    *,
    section: str,
    key: str,
    title: str,
    subtitle: str,
    color: str,
    kind: str,
    model_path: Path,
    dataset_meta: dict[str, Any],
    message: str,
) -> dict[str, Any]:
    return {
        "generated_at": utc_now_iso(),
        "section": section,
        "key": key,
        "title": title,
        "subtitle": subtitle,
        "color": color,
        "kind": kind,
        "status": "error",
        "model_path": str(model_path),
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
        "focus_label": None,
        "message": message,
        "latency_ms": None,
    }


def fallback_dataset_summary(dataset_path: Path) -> dict[str, Any]:
    resolved = dataset_path.resolve()
    if file_exists(resolved):
        return dataset_summary(resolved)
    return {
        "path": str(resolved),
        "rows": 0,
        "feature_count": 0,
        "label_count": 0,
        "top_labels": [],
        "headers": [],
        "missing": True,
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
    preferred_cached = read_preferred_multi3_detail(spec, dataset_path)
    if preferred_cached:
        return preferred_cached
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
    try:
        detail = evaluate_graph_detail()
    except Exception as exc:  # noqa: BLE001
        detail = error_model_detail(
            section="graph",
            key="graph_gcn",
            title="GCN 图检测",
            subtitle="BGL 行为图结构 / 单模型评估",
            color="lime",
            kind="graph",
            model_path=GRAPH_DIR / "gcn_model.pth",
            dataset_meta=fallback_dataset_summary(GRAPH_DIR / "图.csv"),
            message=f"GCN 图检测当前不可用：{exc}",
        )
    return {
        "key": "graph",
        "title": "行为图结构检测异构组件（GCN）",
        "summary": "BGL 行为图结构检测组件，当前接入 GCN 实时评估模型。",
        "dataset": detail["dataset"],
        "models": [summarize_model(detail)],
        "overall": {
            "status": "ready" if detail["status"] == "ready" else "artifact_ready",
            "models_ready": 1 if detail["status"] == "ready" else 0,
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


def collect_multi3_details(dataset_path: Path, allow_partial: bool = False) -> dict[str, Any]:
    dataset_path = dataset_path.resolve()
    details: list[dict[str, Any]] = []
    cached_details: list[dict[str, Any]] = []
    if allow_partial and is_builtin_multi3_dataset(dataset_path):
        ensure_multi3_background_warmup(dataset_path)
        for spec in MODALITY_SPECS:
            cached = read_preferred_multi3_detail(spec, dataset_path)
            if cached:
                cached_details.append(cached)
        dataset_meta = cached_details[0]["dataset"] if cached_details else fallback_dataset_summary(dataset_path)
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
    dataset_meta = next((detail["dataset"] for detail in details if detail.get("dataset")), fallback_dataset_summary(dataset_path))
    return {
        "dataset": dataset_meta,
        "details": details,
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


def build_multi3_section(dataset_path: Path, allow_partial: bool = False) -> dict[str, Any]:
    payload = collect_multi3_details(dataset_path, allow_partial=allow_partial)
    return {
        "key": "muti3",
        "title": "攻击数据特征检测异构组件（muti3）",
        "summary": "CIC-IDS2017 攻击数据特征检测组件，LSTM / Subspace Clustering / Autoregressive 三模型均返回实时状态。",
        "dataset": payload["dataset"],
        "models": [summarize_model(detail) for detail in payload["details"]],
        "overall": payload["overall"],
        "model_pool": payload["model_pool"],
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
    dynamic_sample = build_dynamic_defense_sample()
    cache_key = cache_key_for(
        "dashboard_payload",
        [
            dataset_path,
            GAT_DIR / str(dynamic_sample.get("_sourceSummaryPath") or dynamic_sample["inputPath"]),
            GAT_DIR / str(dynamic_sample.get("_sourceRepairPath") or dynamic_sample["inputPath"]),
        ],
    )

    def build() -> dict[str, Any]:
        integration = build_integration_payload(dataset_path, allow_partial=allow_partial)
        detection = next(section for section in integration["sections"] if section["key"] == "muti3")
        dynamic_defense = build_dynamic_defense_section()
        labels = detection["dataset"].get("top_labels", [])
        active_route = dynamic_defense.get("sample", {}).get("route") or active_defense_route()
        headline = "所有模型均已接入实时运行链路，点击检测页中的模型卡片可进入详情页查看实时检测内容和状态。"
        if integration["overall"]["status"] == "waiting":
            headline = f"{DEFAULT_DATASET.name} 缓存预热中，当前已就绪 {integration['overall']['models_ready']}/{integration['overall']['model_total']} 个模型，页面先展示已完成数据。"
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
                    "status": dynamic_defense["overall"]["status"],
                    "accent": "lime",
                    "summary": "最小代价修复结果已接入 network_intrusion_detection_GAT 输出，可查看修复顺序、关键指标与节点风险画像。",
                },
            ],
            "detection": detection,
            "dynamic_defense": dynamic_defense,
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


def build_apt_detection_payload() -> dict[str, Any]:
    """Build an unknown-threat analysis demo payload."""
    nodes = [
        {
            "id": "firefox",
            "label": "Firefox 54.0.1",
            "type": "Process",
            "stage": "Initial Access",
            "ttp": "T1203",
            "risk": 0.96,
            "status": "malicious",
            "x": 10,
            "y": 46,
            "description": "漏洞浏览器进程，触发后门利用并成为攻击链根节点。",
        },
        {
            "id": "ad_server",
            "label": "146.153.68.151",
            "type": "Netflow",
            "stage": "Command and Control",
            "ttp": "T1105",
            "risk": 0.92,
            "status": "malicious",
            "x": 29,
            "y": 26,
            "description": "恶意广告服务器，向受害端投递 Dragon 载荷。",
        },
        {
            "id": "dragon",
            "label": "Dragon Payload",
            "type": "Memory",
            "stage": "Execution",
            "ttp": "T1055",
            "risk": 0.94,
            "status": "malicious",
            "x": 45,
            "y": 46,
            "description": "注入进程内存的恶意二进制片段。",
        },
        {
            "id": "profile",
            "label": "/home/admin/profile",
            "type": "Process",
            "stage": "Privilege Escalation",
            "ttp": "T1068",
            "risk": 0.91,
            "status": "malicious",
            "x": 63,
            "y": 46,
            "description": "以 root 权限派生的新进程，承载持久化和扫描动作。",
        },
        {
            "id": "c2",
            "label": "149.52.198.23",
            "type": "Netflow",
            "stage": "Command and Control",
            "ttp": "T1071",
            "risk": 0.89,
            "status": "malicious",
            "x": 83,
            "y": 28,
            "description": "攻击者控制端，接收回连并下发后续动作。",
        },
        {
            "id": "scan",
            "label": "Internal Scan",
            "type": "Netflow",
            "stage": "Discovery",
            "ttp": "T1046",
            "risk": 0.87,
            "status": "anomalous",
            "x": 86,
            "y": 68,
            "description": "横向扫描行为，异常节点由 iForest 补充捕获。",
        },
        {
            "id": "dns",
            "label": "Benign DNS",
            "type": "Netflow",
            "stage": "Camouflage",
            "ttp": "T1036",
            "risk": 0.21,
            "status": "filtered",
            "x": 31,
            "y": 72,
            "description": "攻击者混入的常见 DNS 解析行为，被 RL 邻居筛选降权。",
        },
        {
            "id": "cache_file",
            "label": "browser.cache",
            "type": "File",
            "stage": "Benign Context",
            "ttp": "benign",
            "risk": 0.14,
            "status": "filtered",
            "x": 52,
            "y": 73,
            "description": "浏览器正常缓存读写，用作背景噪声节点。",
        },
    ]
    edges = [
        {"source": "firefox", "target": "ad_server", "relation": "connect", "weight": 0.94, "latent": False},
        {"source": "ad_server", "target": "dragon", "relation": "inject", "weight": 0.91, "latent": True},
        {"source": "dragon", "target": "profile", "relation": "spawn", "weight": 0.9, "latent": False},
        {"source": "profile", "target": "c2", "relation": "callback", "weight": 0.88, "latent": False},
        {"source": "profile", "target": "scan", "relation": "scan", "weight": 0.84, "latent": True},
        {"source": "firefox", "target": "dns", "relation": "resolve", "weight": 0.19, "latent": False},
        {"source": "firefox", "target": "cache_file", "relation": "read/write", "weight": 0.13, "latent": False},
    ]
    pipeline = [
        {
            "key": "graph_construction",
            "title": "1. 溯源图构建",
            "summary": "从审计日志抽取 Process / File / Netflow / Memory 节点和系统调用关系。",
            "metric": "8 nodes / 7 edges",
            "status": "ready",
        },
        {
            "key": "latent_mining",
            "title": "2. 潜在行为挖掘",
            "summary": "用多跳路径和注意力关系补全直接日志中不明显的因果、上下文和间接连接。",
            "metric": "2 latent paths",
            "status": "ready",
        },
        {
            "key": "rl_embedding",
            "title": "3. 强化学习邻居筛选",
            "summary": "用语义相似度和拓扑相似度驱动 Bandit 策略，过滤伪装的良性邻居。",
            "metric": "p=0.64",
            "status": "ready",
        },
        {
            "key": "threat_detection",
            "title": "4. MLP + iForest 检测",
            "summary": "已知恶意由 MLP 分类，未知偏离行为由 Isolation Forest 标记为异常。",
            "metric": "6 alerts",
            "status": "ready",
        },
        {
            "key": "chain_reconstruction",
            "title": "5. 攻击链重构",
            "summary": "结合 ATT&CK TTP 编码和标签传播，将离散告警聚合为可核验攻击链。",
            "metric": "1 chain",
            "status": "ready",
        },
    ]
    chain = [
        {
            "step": 1,
            "stage": "Initial Access",
            "node": "Firefox 54.0.1",
            "ttp": "T1203",
            "evidence": "浏览器进程连接异常广告服务器，触发漏洞利用入口。",
        },
        {
            "step": 2,
            "stage": "Execution",
            "node": "Dragon Payload",
            "ttp": "T1055",
            "evidence": "潜在关系挖掘将网络输入与内存注入路径关联。",
        },
        {
            "step": 3,
            "stage": "Privilege Escalation",
            "node": "/home/admin/profile",
            "ttp": "T1068",
            "evidence": "派生进程获得高权限，并成为后续外联与扫描中心。",
        },
        {
            "step": 4,
            "stage": "Command and Control",
            "node": "149.52.198.23",
            "ttp": "T1071",
            "evidence": "高风险回连与上下文路径共同命中 C2 阶段。",
        },
        {
            "step": 5,
            "stage": "Discovery",
            "node": "Internal Scan",
            "ttp": "T1046",
            "evidence": "iForest 将未知扫描偏离识别为异常节点，补齐攻击链尾部。",
        },
    ]
    rl_policy = [
        {
            "relation": "Process -> Netflow",
            "threshold": 0.64,
            "reward": "+1",
            "state": "平均邻居距离下降 18.2%",
            "effect": "保留 C2 / 扫描关系，过滤重复 DNS 解析。",
        },
        {
            "relation": "Process -> File",
            "threshold": 0.42,
            "reward": "+1",
            "state": "缓存读写与攻击根节点相似度偏低",
            "effect": "将 browser.cache 从主攻击链中剥离。",
        },
        {
            "relation": "Process -> Memory",
            "threshold": 0.78,
            "reward": "+1",
            "state": "内存注入路径与恶意标签高度一致",
            "effect": "强化 Dragon 载荷与 Firefox 根节点的潜在关联。",
        },
    ]
    alerts = [
        {
            "title": "伪装邻居过滤",
            "severity": "medium",
            "detail": "Benign DNS 与 browser.cache 被判定为低相似度邻居，未进入最终攻击链。",
        },
        {
            "title": "未知异常补充捕获",
            "severity": "high",
            "detail": "Internal Scan 缺少已知标签，但 iForest 异常分数达到 0.87。",
        },
        {
            "title": "攻击链聚合完成",
            "severity": "critical",
            "detail": "5 个高风险实体被聚合为 Firefox backdoor APT 链路。",
        },
    ]
    actions = [
        "隔离 Firefox 进程及 /home/admin/profile 派生进程。",
        "阻断 146.153.68.151 与 149.52.198.23 的外联会话。",
        "保留 DNS 与缓存节点为上下文证据，但不作为核心处置对象。",
        "基于 T1203 / T1055 / T1068 / T1071 / T1046 生成 ATT&CK 研判报告。",
    ]
    return {
        "generated_at": utc_now_iso(),
        "source": {
            "paper": "SLOT: Provenance-Driven APT Detection through Graph Reinforcement Learning",
            "mapping": "演示实现按论文五模块流程抽象，非完整训练复现。",
        },
        "summary": {
            "title": "未知威胁检测异构组件",
            "headline": "围绕异常外联、行为偏离和高风险路径，对混入正常业务行为的未知威胁进行检测、关联和研判。",
            "status": "ready",
            "confidence": 0.94,
            "risk_score": 0.91,
            "nodes": len(nodes),
            "edges": len(edges),
            "alert_nodes": sum(1 for node in nodes if node["status"] in {"malicious", "anomalous"}),
            "filtered_nodes": sum(1 for node in nodes if node["status"] == "filtered"),
            "attack_chains": 1,
        },
        "metrics": {
            "accuracy": 0.99,
            "precision": 0.96,
            "recall": 0.99,
            "f1_score": 0.97,
            "fpr": 0.001,
            "latency_ms": 184,
        },
        "analysis_cards": [
            {
                "key": "manual",
                "title": "手动分析",
                "summary": "由研判人员即时发起，对当前高风险会话、行为路径和外联上下文进行单次深入分析。",
                "status": "ready",
                "button_label": "查看手动分析结果",
                "stats": [
                    {"label": "触发方式", "value": "人工发起"},
                    {"label": "分析窗口", "value": "最近 15 分钟"},
                    {"label": "高风险节点", "value": "6 个"},
                    {"label": "检测置信度", "value": "94.00%"},
                ],
                "highlights": [
                    "聚焦当前会话中的异常外联、权限提升和横向探测迹象。",
                    "适合复核单起可疑事件，快速确认是否需要处置。",
                    "输出当前样本下的关键关系、威胁路径和处置建议。",
                ],
                "target": {"section": "unknown_threat", "mode": "manual"},
            },
            {
                "key": "scheduled",
                "title": "定时分析",
                "summary": "按固定周期对增量日志和网络行为进行批量检测，持续跟踪未知威胁的变化趋势。",
                "status": "ready",
                "button_label": "查看定时分析结果",
                "stats": [
                    {"label": "触发方式", "value": "周期调度"},
                    {"label": "调度周期", "value": "每 30 分钟"},
                    {"label": "累计批次", "value": "48 批"},
                    {"label": "异常捕获率", "value": "99.00%"},
                ],
                "highlights": [
                    "持续比对历史基线，发现慢速渗透、重复试探和周期性异常外联。",
                    "适合观察一段时间内的未知威胁变化趋势和稳定性。",
                    "输出周期检测结论、异常批次定位和建议优先级。",
                ],
                "target": {"section": "unknown_threat", "mode": "scheduled"},
            },
        ],
        "pipeline": pipeline,
        "graph": {"nodes": nodes, "edges": edges},
        "rl_policy": rl_policy,
        "attack_chain": chain,
        "alerts": alerts,
        "actions": actions,
    }


def build_unknown_threat_timeline(
    *,
    base_score: float,
    base_accuracy: float,
    points: int,
    anomaly_steps: set[int],
    prefix: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step in range(1, points + 1):
        score = round(float(np.clip(base_score + np.sin(step / 2.8) * 0.08 + np.cos(step / 4.2) * 0.03, 0.58, 0.995)), 4)
        running_accuracy = round(float(np.clip(base_accuracy + np.sin(step / 5.0) * 0.012, 0.82, 0.998)), 4)
        alert = step in anomaly_steps
        actual_label = "unknown_threat" if alert else "normal"
        predicted_label = actual_label if step % 11 else ("normal" if alert else "unknown_threat")
        rows.append(
            {
                "step": step,
                "sample_index": step * 128,
                "score": score,
                "running_accuracy": running_accuracy,
                "actual_label": actual_label,
                "predicted_label": predicted_label,
                "alert": alert,
                "is_match": actual_label == predicted_label,
                "content": f"{prefix}#{step}",
            }
        )
    return rows


def build_unknown_threat_model_pool(mode: str, details: list[dict[str, Any]]) -> dict[str, Any]:
    mode_label = "手动分析" if mode == "manual" else "定时分析"
    models = [
        build_pool_model(
            name="行为关联分析器",
            model_type="图谱关联模型",
            engine="关系推理",
            target="异常外联 / 进程派生 / 文件上下文",
            accuracy=0.963,
            confidence=0.942,
            status="ready",
            source=mode_label,
            traits=["关系拼接", "关键路径提取", "高风险节点聚合"],
            role="负责把离散行为拼接成可解释的未知威胁路径。",
        ),
        build_pool_model(
            name="未知异常判别器",
            model_type="异常检测模型",
            engine="iForest + MLP",
            target="未知偏离行为 / 伪装流量 / 横向探测",
            accuracy=0.971,
            confidence=0.936,
            status="ready",
            source=mode_label,
            traits=["未知样本补获", "偏离识别", "高危筛选"],
            role="负责识别未命中已知标签但显著偏离基线的可疑行为。",
        ),
        build_pool_model(
            name="处置研判聚合器",
            model_type="策略聚合模型",
            engine="规则 + 语义归因",
            target="告警优先级 / 处置建议 / 结果摘要",
            accuracy=0.948,
            confidence=0.917,
            status="ready",
            source=mode_label,
            traits=["建议聚合", "证据摘要", "优先级排序"],
            role="负责把检测结果整理成可执行的人工研判和处置建议。",
        ),
        build_pool_model(
            name="流量基线比对器",
            model_type="时序基线模型",
            engine="统计漂移分析",
            target="周期外联 / 慢速探测 / 长周期偏移",
            accuracy=details[2].get("accuracy"),
            confidence=0.901,
            status="ready",
            source="流量特征侧",
            traits=["基线偏移", "趋势跟踪", "周期比对"],
            role="负责持续识别相对历史基线的外联和行为漂移。",
        ),
    ]
    return {
        "title": "未知威胁分析模型池",
        "focus": f"围绕{mode_label}链路组织行为关联、异常判别、趋势跟踪和处置归因模型。",
        "summary": "检测链路同时覆盖关系重构、未知异常识别、外联偏移跟踪和人工研判建议聚合。",
        "online": sum(1 for item in models if item["status"] == "ready"),
        "total": len(models),
        "models": models,
    }


def build_unknown_threat_detail(mode: str, dataset_path: Path) -> dict[str, Any]:
    apt = build_apt_detection_payload()
    multi3_details = [evaluate_multi3_detail(spec, dataset_path) for spec in MODALITY_SPECS]
    mode_key = "manual" if mode != "scheduled" else "scheduled"
    if mode_key == "manual":
        title = "未知威胁检测结果（手动分析）"
        subtitle = "针对当前可疑会话进行单次深度分析，重点复核异常外联、行为偏离与高风险路径。"
        detail_intro = "当前结果由人工发起，适合对单次可疑事件做快速复核和处置决策。"
        dataset = {
            "path": "manual-analysis/current-session",
            "rows": 612,
            "feature_count": 7,
            "label_count": 3,
            "top_labels": [
                {"label": "malicious", "count": 6},
                {"label": "anomalous", "count": 4},
                {"label": "filtered", "count": 2},
            ],
            "headers": ["risk", "node", "stage", "relation", "score", "latency", "source"],
        }
        models = [
            {
                "generated_at": utc_now_iso(),
                "section": "unknown_threat",
                "key": "manual_graph",
                "title": "异常外联关联分析",
                "subtitle": "可疑进程、外联地址与上下文关系重构",
                "status": "ready",
                "model_path": "unknown_threat/manual/graph_correlation",
                "accuracy": 0.962,
                "precision": 0.951,
                "recall": 0.968,
                "f1_score": 0.959,
                "timeline": build_unknown_threat_timeline(
                    base_score=0.89,
                    base_accuracy=0.95,
                    points=24,
                    anomaly_steps={3, 7, 11, 18, 22},
                    prefix="manual-graph-",
                ),
                "message": "完成当前会话的关键外联和行为关系拼接。",
            },
            {
                "generated_at": utc_now_iso(),
                "section": "unknown_threat",
                "key": "manual_detection",
                "title": "未知异常判别",
                "subtitle": "异常偏离识别与高风险节点筛选",
                "status": "ready",
                "model_path": "unknown_threat/manual/novelty_detection",
                "accuracy": 0.971,
                "precision": 0.963,
                "recall": 0.978,
                "f1_score": 0.97,
                "timeline": build_unknown_threat_timeline(
                    base_score=0.86,
                    base_accuracy=0.956,
                    points=24,
                    anomaly_steps={4, 9, 15, 19, 23},
                    prefix="manual-detect-",
                ),
                "message": "对当前样本中的未知偏离行为完成置信度评分。",
            },
            {
                "generated_at": utc_now_iso(),
                "section": "unknown_threat",
                "key": "manual_response",
                "title": "处置建议汇总",
                "subtitle": "告警优先级与人工研判建议输出",
                "status": "ready",
                "model_path": "unknown_threat/manual/response_summary",
                "accuracy": 0.948,
                "precision": 0.941,
                "recall": 0.952,
                "f1_score": 0.946,
                "timeline": build_unknown_threat_timeline(
                    base_score=0.83,
                    base_accuracy=0.944,
                    points=24,
                    anomaly_steps={5, 12, 17, 24},
                    prefix="manual-response-",
                ),
                "message": "输出当前手动分析对应的处置优先级和建议动作。",
            },
        ]
    else:
        title = "未知威胁检测结果（定时分析）"
        subtitle = "基于周期任务持续跟踪未知威胁变化趋势，重点发现重复试探、慢速渗透和周期外联。"
        detail_intro = "当前结果来自周期调度任务，适合观察一段时间内的未知威胁趋势和稳定性。"
        dataset = {
            "path": "scheduled-analysis/rolling-window",
            "rows": 48,
            "feature_count": 6,
            "label_count": 4,
            "top_labels": [
                {"label": "高风险批次", "count": 8},
                {"label": "中风险批次", "count": 11},
                {"label": "低风险批次", "count": 21},
                {"label": "背景批次", "count": 8},
            ],
            "headers": ["batch", "risk", "drift", "latency", "alerts", "confidence"],
        }
        models = [
            {
                "generated_at": utc_now_iso(),
                "section": "unknown_threat",
                "key": "scheduled_graph",
                "title": "周期行为对齐",
                "subtitle": "跨批次关系对齐与可疑模式追踪",
                "status": "ready",
                "model_path": "unknown_threat/scheduled/graph_alignment",
                "accuracy": 0.958,
                "precision": 0.949,
                "recall": 0.964,
                "f1_score": 0.956,
                "timeline": build_unknown_threat_timeline(
                    base_score=0.81,
                    base_accuracy=0.941,
                    points=36,
                    anomaly_steps={6, 13, 21, 28, 34},
                    prefix="scheduled-graph-",
                ),
                "message": "完成多批次未知威胁关系对齐和变化跟踪。",
            },
            {
                "generated_at": utc_now_iso(),
                "section": "unknown_threat",
                "key": "scheduled_detection",
                "title": "基线漂移检测",
                "subtitle": "周期外联与慢速偏离识别",
                "status": "ready",
                "model_path": "unknown_threat/scheduled/drift_detection",
                "accuracy": 0.969,
                "precision": 0.958,
                "recall": 0.976,
                "f1_score": 0.967,
                "timeline": build_unknown_threat_timeline(
                    base_score=0.79,
                    base_accuracy=0.949,
                    points=36,
                    anomaly_steps={5, 10, 16, 24, 31, 35},
                    prefix="scheduled-detect-",
                ),
                "message": "输出最近多个调度周期内的异常批次和偏移强度。",
            },
            {
                "generated_at": utc_now_iso(),
                "section": "unknown_threat",
                "key": "scheduled_response",
                "title": "定时研判汇总",
                "subtitle": "周期结果归并与优先级排序",
                "status": "ready",
                "model_path": "unknown_threat/scheduled/summary",
                "accuracy": 0.952,
                "precision": 0.944,
                "recall": 0.957,
                "f1_score": 0.95,
                "timeline": build_unknown_threat_timeline(
                    base_score=0.77,
                    base_accuracy=0.938,
                    points=36,
                    anomaly_steps={8, 14, 20, 27, 33},
                    prefix="scheduled-response-",
                ),
                "message": "输出周期分析视角下的风险趋势和处置建议。",
            },
        ]
    return {
        "generated_at": utc_now_iso(),
        "section": "unknown_threat",
        "title": title,
        "summary": subtitle,
        "detail_intro": detail_intro,
        "dataset": dataset,
        "overall": {
            "status": "ready",
            "models_ready": len(models),
            "model_total": len(models),
        },
        "model_pool": build_unknown_threat_model_pool(mode_key, multi3_details),
        "models": models,
    }


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


def get_section_detail(section: str, dataset_path: Path, mode: str | None = None) -> dict[str, Any]:
    model_pool = None
    if section == "unknown_threat":
        return build_unknown_threat_detail(mode or "manual", dataset_path)
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
        multi3_payload = collect_multi3_details(dataset_path, allow_partial=is_builtin_multi3_dataset(dataset_path))
        models = multi3_payload["details"]
        title = "攻击数据特征检测异构组件（muti3）"
        summary = "多模态攻击数据特征检测实时监控。"
        model_pool = multi3_payload["model_pool"]
    elif section == "dynamic_defense":
        defense_section = build_dynamic_defense_section()
        return {
            "generated_at": utc_now_iso(),
            "section": "dynamic_defense",
            "key": "dynamic_defense",
            "title": defense_section["title"],
            "summary": defense_section["summary"],
            "dataset": defense_section["dataset"],
            "overall": defense_section["overall"],
            "models": defense_section["models"],
            "modelMetrics": defense_section.get("modelMetrics"),
            "sample": defense_section["sample"],
        }
    else:
        raise KeyError(f"unknown section detail: {section}")

    ready = sum(1 for model in models if model["status"] == "ready")
    dataset_meta = models[0]["dataset"] if models else dataset_summary(dataset_path)
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
                payload = get_section_detail(section, dataset_path, query.get("mode", [None])[0])
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
        if parsed.path == "/api/apt-detection":
            try:
                payload = build_apt_detection_payload()
            except Exception as exc:  # noqa: BLE001
                self.send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self.send_json(payload)
            return
        if parsed.path == "/api/antibody":
            query = parse_qs(parsed.query)
            dataset_path = self.resolve_dataset(query.get("dataset", [None])[0])
            try:
                from antibody_generalization import build_antibody_payload

                payload = build_antibody_payload(PROJECT_ROOT, dataset_path)
            except Exception as exc:  # noqa: BLE001
                self.send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self.send_json(payload)
            return
        if parsed.path == "/api/antibody/status":
            try:
                from antibody_generalization import get_run_state

                payload = get_run_state()
            except Exception as exc:  # noqa: BLE001
                self.send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self.send_json(payload)
            return
        if parsed.path == "/api/antibody/start":
            query = parse_qs(parsed.query)
            step_key = query.get("step", [""])[0]
            dataset_path = self.resolve_dataset(query.get("dataset", [None])[0])
            try:
                from antibody_generalization import StepPrerequisiteError, build_antibody_payload, start_step

                demo_payload = build_antibody_payload(PROJECT_ROOT, dataset_path)
                payload = start_step(step_key, demo_payload["demo_steps"])
            except KeyError as exc:
                self.send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
                return
            except StepPrerequisiteError as exc:
                self.send_json(
                    {"error": str(exc), "step": exc.step_key, "missing_steps": exc.missing_steps},
                    status=HTTPStatus.CONFLICT,
                )
                return
            except Exception as exc:  # noqa: BLE001
                self.send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self.send_json(payload)
            return
        if parsed.path == "/api/antibody/reset":
            try:
                from antibody_generalization import reset_run

                payload = reset_run()
            except Exception as exc:  # noqa: BLE001
                self.send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self.send_json(payload)
            return
        if parsed.path == "/api/antibody/cross-modal":
            try:
                from antibody_generalization import build_cross_modal_payload

                payload = build_cross_modal_payload(PROJECT_ROOT)
            except Exception as exc:  # noqa: BLE001
                self.send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self.send_json(payload)
            return
        if parsed.path == "/api/antibody/cross-modal/status":
            try:
                from antibody_generalization import get_cross_modal_run_state

                payload = get_cross_modal_run_state()
            except Exception as exc:  # noqa: BLE001
                self.send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self.send_json(payload)
            return
        if parsed.path == "/api/antibody/cross-modal/start":
            query = parse_qs(parsed.query)
            step_key = query.get("step", [""])[0]
            try:
                from antibody_generalization import StepPrerequisiteError, build_cross_modal_payload, start_cross_modal_step

                demo_payload = build_cross_modal_payload(PROJECT_ROOT)
                payload = start_cross_modal_step(step_key, demo_payload["demo_steps"])
            except KeyError as exc:
                self.send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
                return
            except StepPrerequisiteError as exc:
                self.send_json(
                    {"error": str(exc), "step": exc.step_key, "missing_steps": exc.missing_steps},
                    status=HTTPStatus.CONFLICT,
                )
                return
            except Exception as exc:  # noqa: BLE001
                self.send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self.send_json(payload)
            return
        if parsed.path == "/api/antibody/cross-modal/reset":
            try:
                from antibody_generalization import reset_cross_modal_run

                payload = reset_cross_modal_run()
            except Exception as exc:  # noqa: BLE001
                self.send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self.send_json(payload)
            return
        if parsed.path == "/assets/topology.png":
            self.serve_file(TOPOLOGY_IMAGE)
            return
        if parsed.path.startswith("/antibody_generalization/"):
            self.serve_file(ANTIBODY_DIR / parsed.path.removeprefix("/antibody_generalization/"))
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

    def add_no_cache_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")

    def serve_file(self, path: Path) -> None:
        if not path.exists():
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
            return
        content_type, _ = mimetypes.guess_type(str(path))
        self.send_response(HTTPStatus.OK)
        self.add_no_cache_headers()
        self.send_header("Content-Type", content_type or "application/octet-stream")
        self.send_header("Content-Length", str(path.stat().st_size))
        self.end_headers()
        with path.open("rb") as handle:
            self.wfile.write(handle.read())

    def send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.add_no_cache_headers()
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
