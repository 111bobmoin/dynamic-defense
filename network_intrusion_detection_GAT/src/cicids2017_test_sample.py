from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.data import LABEL_COLUMN, clean_columns


DEFAULT_DATASET_DIR = Path(r"E:\dataset\MachineLearningCVE")
NO_ANOMALY_SCENARIO = "no_anomaly"
SINGLE_ANOMALY_SCENARIO = "single_anomaly"
MULTI_ANOMALY_SCENARIO = "multi_anomaly"
BENIGN_FPR_SCENARIO = NO_ANOMALY_SCENARIO
NODE_EVAL_SCENARIO = MULTI_ANOMALY_SCENARIO
SUPPORTED_SCENARIOS = (
    NO_ANOMALY_SCENARIO,
    SINGLE_ANOMALY_SCENARIO,
    MULTI_ANOMALY_SCENARIO,
)
SCENARIO_DISPLAY_NAMES = {
    NO_ANOMALY_SCENARIO: "无异常节点",
    SINGLE_ANOMALY_SCENARIO: "单异常节点",
    MULTI_ANOMALY_SCENARIO: "多异常节点",
}


@dataclass(frozen=True)
class ScenarioFilenames:
    sample_csv: str
    ground_truth_csv: str
    manifest_json: str
    predictions_csv: str
    summary_csv: str
    report_json: str


SCENARIO_FILENAMES = {
    NO_ANOMALY_SCENARIO: ScenarioFilenames(
        sample_csv="cicids2017_no_anomaly_sample.csv",
        ground_truth_csv="cicids2017_no_anomaly_ground_truth.csv",
        manifest_json="sample_manifest.json",
        predictions_csv="cicids2017_no_anomaly_predictions.csv",
        summary_csv="cicids2017_no_anomaly_summary.csv",
        report_json="no_anomaly_report.json",
    ),
    SINGLE_ANOMALY_SCENARIO: ScenarioFilenames(
        sample_csv="cicids2017_single_anomaly_sample.csv",
        ground_truth_csv="cicids2017_single_anomaly_ground_truth.csv",
        manifest_json="sample_manifest.json",
        predictions_csv="cicids2017_single_anomaly_predictions.csv",
        summary_csv="cicids2017_single_anomaly_summary.csv",
        report_json="single_anomaly_report.json",
    ),
    MULTI_ANOMALY_SCENARIO: ScenarioFilenames(
        sample_csv="cicids2017_multi_anomaly_sample.csv",
        ground_truth_csv="cicids2017_multi_anomaly_ground_truth.csv",
        manifest_json="sample_manifest.json",
        predictions_csv="cicids2017_multi_anomaly_predictions.csv",
        summary_csv="cicids2017_multi_anomaly_summary.csv",
        report_json="multi_anomaly_report.json",
    ),
}


ROLE_SEVERITY = {
    "uncertain": 0,
    "suspected_victim": 1,
    "suspected_attacker": 2,
    "suspected_compromised_host": 3,
}

ATTACKER_LABELS = ["DDoS", "PortScan", "DoS Hulk", "DoS GoldenEye"]
COMPROMISED_LABELS = ["Infiltration", "Bot", "FTP-Patator", "SSH-Patator"]
VICTIM_LABELS = ["DoS Slowhttptest", "DoS slowloris", "DDoS", "PortScan"]
SINGLE_ANOMALY_LABELS = ["Infiltration", "Bot", "FTP-Patator", "SSH-Patator", "PortScan"]
BENIGN_LABEL = "BENIGN"

NO_ANOMALY_TOTAL_FLOWS = 1440


@dataclass(frozen=True)
class NodeSpec:
    node_id: str
    ground_truth_role: str


ATTACKER_NODES = [
    NodeSpec("172.16.10.10", "suspected_attacker"),
    NodeSpec("172.16.10.11", "suspected_attacker"),
]
VICTIM_NODES = [
    NodeSpec("172.16.20.20", "suspected_victim"),
    NodeSpec("172.16.20.21", "suspected_victim"),
]
COMPROMISED_NODES = [
    NodeSpec("172.16.30.30", "suspected_compromised_host"),
    NodeSpec("172.16.30.31", "suspected_compromised_host"),
]
MULTI_ANOMALY_BENIGN_NODES = [
    NodeSpec("172.16.40.40", "uncertain"),
    NodeSpec("172.16.40.41", "uncertain"),
    NodeSpec("172.16.40.42", "uncertain"),
    NodeSpec("172.16.40.43", "uncertain"),
]
SINGLE_ANOMALY_NODE = NodeSpec("172.16.50.50", "suspected_compromised_host")
SINGLE_ANOMALY_BENIGN_NODES = [
    NodeSpec("172.16.50.60", "uncertain"),
    NodeSpec("172.16.50.61", "uncertain"),
    NodeSpec("172.16.50.62", "uncertain"),
    NodeSpec("172.16.50.63", "uncertain"),
]
INTERNET_NODES = [
    "61.135.169.121",
    "8.8.8.8",
    "114.114.114.114",
    "203.119.144.80",
    "104.16.132.229",
    "151.101.1.69",
    "23.45.119.216",
    "52.84.217.72",
]

INTERNAL_CLIENTS = [
    "10.10.10.11",
    "10.10.10.12",
    "10.10.10.13",
    "10.10.10.14",
    "10.10.10.15",
    "10.10.10.16",
    "10.10.10.17",
    "10.10.10.18",
]
INTERNAL_SERVICES = [
    "10.10.20.21",
    "10.10.20.22",
    "10.10.20.23",
    "10.10.20.24",
]
EXTERNAL_SERVICES = [
    "8.8.8.8",
    "1.1.1.1",
    "114.114.114.114",
    "151.101.1.69",
    "104.16.132.229",
    "23.45.119.216",
    "61.135.169.121",
    "52.84.217.72",
]
ALL_BENIGN_NODES = INTERNAL_CLIENTS + INTERNAL_SERVICES + EXTERNAL_SERVICES
DEFAULT_LARGE_NODE_COUNTS = {
    NO_ANOMALY_SCENARIO: 72,
    SINGLE_ANOMALY_SCENARIO: 64,
    MULTI_ANOMALY_SCENARIO: 88,
}


def ensure_supported_scenario(scenario: str) -> str:
    normalized = str(scenario).strip().lower()
    if normalized not in SUPPORTED_SCENARIOS:
        raise ValueError(f"Unsupported scenario: {scenario}")
    return normalized


def get_scenario_display_name(scenario: str) -> str:
    return SCENARIO_DISPLAY_NAMES[ensure_supported_scenario(scenario)]


def get_scenario_paths(output_dir: Path, scenario: str) -> dict[str, Path]:
    names = SCENARIO_FILENAMES[ensure_supported_scenario(scenario)]
    return {
        "output_dir": output_dir,
        "sample_csv": output_dir / names.sample_csv,
        "ground_truth_csv": output_dir / names.ground_truth_csv,
        "manifest_json": output_dir / names.manifest_json,
        "predictions_csv": output_dir / names.predictions_csv,
        "summary_csv": output_dir / names.summary_csv,
        "report_json": output_dir / names.report_json,
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def build_rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def stream_rows_for_labels(
    dataset_dir: Path,
    labels: set[str],
    rng: np.random.Generator | None = None,
) -> dict[str, list[dict[str, object]]]:
    buckets: dict[str, list[dict[str, object]]] = defaultdict(list)
    for csv_path in sorted(dataset_dir.glob("*.csv")):
        for chunk in pd.read_csv(csv_path, chunksize=20000, low_memory=False):
            chunk = chunk.copy()
            chunk.columns = clean_columns(chunk.columns)
            if LABEL_COLUMN not in chunk.columns:
                continue
            chunk[LABEL_COLUMN] = chunk[LABEL_COLUMN].astype(str).str.strip()
            filtered = chunk[chunk[LABEL_COLUMN].isin(labels)]
            if filtered.empty:
                continue
            for record in filtered.to_dict(orient="records"):
                buckets[str(record[LABEL_COLUMN])].append(record)
    if rng is not None:
        for rows in buckets.values():
            rng.shuffle(rows)
    return buckets


def load_benign_rows(
    dataset_dir: Path,
    total_rows: int,
    rng: np.random.Generator | None = None,
) -> list[dict[str, object]]:
    csv_files = sorted(dataset_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {dataset_dir}")

    base_quota = total_rows // len(csv_files)
    remainder = total_rows % len(csv_files)
    quotas = {
        path: base_quota + (1 if index < remainder else 0)
        for index, path in enumerate(csv_files)
    }

    selected: list[dict[str, object]] = []
    for csv_path in csv_files:
        target = quotas[csv_path]
        if target <= 0:
            continue
        reservoir: list[dict[str, object]] = []
        seen = 0
        for chunk in pd.read_csv(csv_path, chunksize=20000, low_memory=False):
            chunk = chunk.copy()
            chunk.columns = clean_columns(chunk.columns)
            if LABEL_COLUMN not in chunk.columns:
                continue
            chunk[LABEL_COLUMN] = chunk[LABEL_COLUMN].astype(str).str.strip()
            benign = chunk[chunk[LABEL_COLUMN] == BENIGN_LABEL]
            if benign.empty:
                continue
            for record in benign.to_dict(orient="records"):
                seen += 1
                if len(reservoir) < target:
                    reservoir.append(record)
                    continue
                if rng is None:
                    continue
                replacement_index = int(rng.integers(0, seen))
                if replacement_index < target:
                    reservoir[replacement_index] = record
        if rng is not None:
            rng.shuffle(reservoir)
        selected.extend(reservoir)

    if len(selected) < total_rows:
        raise RuntimeError(f"Collected {len(selected)} BENIGN rows, need {total_rows}")
    if rng is not None:
        rng.shuffle(selected)
    return selected[:total_rows]


def select_records(
    buckets: dict[str, list[dict[str, object]]],
    label_plan: Iterable[tuple[str, int]],
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    offsets: dict[str, int] = defaultdict(int)
    for label, count in label_plan:
        rows = buckets.get(label, [])
        if len(rows) < count:
            raise RuntimeError(f"Not enough rows for label {label}: need {count}, have {len(rows)}")
        start = offsets[label]
        end = start + count
        if end > len(rows):
            raise RuntimeError(f"Label {label} exhausted while sampling.")
        selected.extend(rows[start:end])
        offsets[label] = end
    return selected


def assign_network_fields(
    rows: list[dict[str, object]],
    *,
    src_ip: str,
    dst_ips: list[str],
    src_port_start: int,
    protocol: int,
    start_timestamp: pd.Timestamp,
    spacing_seconds: int,
) -> list[dict[str, object]]:
    assigned: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        record = dict(row)
        record["src_ip"] = src_ip
        record["dst_ip"] = dst_ips[index % len(dst_ips)]
        record["src_port"] = src_port_start + (index % 20000)
        record["protocol"] = protocol
        record["timestamp"] = (start_timestamp + pd.Timedelta(seconds=index * spacing_seconds)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        assigned.append(record)
    return assigned


def assign_inbound_fields(
    rows: list[dict[str, object]],
    *,
    dst_ip: str,
    src_ips: list[str],
    src_port_start: int,
    protocol: int,
    start_timestamp: pd.Timestamp,
    spacing_seconds: int,
) -> list[dict[str, object]]:
    assigned: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        record = dict(row)
        record["src_ip"] = src_ips[index % len(src_ips)]
        record["dst_ip"] = dst_ip
        record["src_port"] = src_port_start + (index % 20000)
        record["protocol"] = protocol
        record["timestamp"] = (start_timestamp + pd.Timedelta(seconds=index * spacing_seconds)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        assigned.append(record)
    return assigned


def assign_profile(
    rows: list[dict[str, object]],
    *,
    src_nodes: list[str],
    dst_nodes: list[str],
    src_port_start: int,
    protocol: int,
    start_timestamp: pd.Timestamp,
    spacing_seconds: int,
    src_stride: int = 1,
    dst_stride: int = 1,
) -> list[dict[str, object]]:
    assigned: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        record = dict(row)
        src_index = (index * src_stride) % len(src_nodes)
        dst_index = (index * dst_stride) % len(dst_nodes)
        record["src_ip"] = src_nodes[src_index]
        record["dst_ip"] = dst_nodes[dst_index]
        if record["src_ip"] == record["dst_ip"]:
            dst_index = (dst_index + 1) % len(dst_nodes)
            record["dst_ip"] = dst_nodes[dst_index]
        record["src_port"] = src_port_start + (index % 20000)
        record["protocol"] = protocol
        record["timestamp"] = (start_timestamp + pd.Timedelta(seconds=index * spacing_seconds)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        assigned.append(record)
    return assigned


def order_flow_dataframe(flow_rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(flow_rows)
    ordered_columns = ["src_ip", "dst_ip", "src_port", "protocol", "timestamp"] + [
        column
        for column in frame.columns
        if column not in {"src_ip", "dst_ip", "src_port", "protocol", "timestamp"}
    ]
    return frame[ordered_columns]


def build_weighted_counts(
    total_count: int,
    weights: dict[str, float],
    minimums: dict[str, int] | None = None,
) -> dict[str, int]:
    if total_count <= 0:
        raise ValueError("total_count must be positive")
    minimums = minimums or {}
    base_total = sum(int(minimums.get(key, 0)) for key in weights)
    if total_count < base_total:
        raise ValueError(f"total_count={total_count} is smaller than minimum required count={base_total}")

    counts = {key: int(minimums.get(key, 0)) for key in weights}
    remaining = total_count - base_total
    if remaining == 0:
        return counts

    weight_sum = sum(float(value) for value in weights.values())
    remainders: dict[str, float] = {}
    for key, weight in weights.items():
        raw = remaining * float(weight) / weight_sum
        whole = int(raw)
        counts[key] += whole
        remainders[key] = raw - whole

    leftover = total_count - sum(counts.values())
    for key in sorted(weights, key=lambda item: (-remainders[item], item)):
        if leftover <= 0:
            break
        counts[key] += 1
        leftover -= 1
    return counts


def generate_host_pool(prefix: str, count: int, start_host: int) -> list[str]:
    if count <= 0:
        return []
    if start_host + count - 1 > 254:
        raise ValueError(f"Host pool exceeds subnet capacity for {prefix}.0/24")
    return [f"{prefix}.{start_host + index}" for index in range(count)]


def flatten_node_groups(node_groups: dict[str, list[str]]) -> list[str]:
    ordered_keys = ("internal_clients", "internal_services", "external_services")
    flattened: list[str] = []
    for key in ordered_keys:
        flattened.extend(node_groups.get(key, []))
    return flattened


def build_large_benign_node_groups(
    total_nodes: int,
    *,
    client_prefix: str,
    service_prefix: str,
    external_prefix: str,
    weights: dict[str, float],
    minimums: dict[str, int],
) -> dict[str, list[str]]:
    counts = build_weighted_counts(total_nodes, weights=weights, minimums=minimums)
    return {
        "internal_clients": generate_host_pool(client_prefix, counts["internal_clients"], start_host=11),
        "internal_services": generate_host_pool(service_prefix, counts["internal_services"], start_host=21),
        "external_services": generate_host_pool(external_prefix, counts["external_services"], start_host=31),
    }


def build_benign_background_flows(
    benign_rows: list[dict[str, object]],
    *,
    internal_clients: list[str],
    internal_services: list[str],
    external_services: list[str],
    start_timestamp: pd.Timestamp,
    base_src_port: int,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    profile_sizes = build_weighted_counts(
        len(benign_rows),
        weights={
            "client_to_external": 0.33,
            "external_to_client": 0.17,
            "client_to_service": 0.21,
            "service_to_client": 0.13,
            "service_to_service": 0.08,
            "external_to_service": 0.08,
        },
    )

    offsets: dict[str, tuple[int, int]] = {}
    cursor = 0
    for name, size in profile_sizes.items():
        offsets[name] = (cursor, cursor + size)
        cursor += size

    def take(profile_name: str) -> list[dict[str, object]]:
        start, end = offsets[profile_name]
        return benign_rows[start:end]

    flow_rows: list[dict[str, object]] = []
    flow_rows.extend(
        assign_profile(
            take("client_to_external"),
            src_nodes=internal_clients,
            dst_nodes=external_services,
            src_port_start=base_src_port,
            protocol=6,
            start_timestamp=start_timestamp,
            spacing_seconds=8,
            src_stride=1,
            dst_stride=3,
        )
    )
    flow_rows.extend(
        assign_profile(
            take("external_to_client"),
            src_nodes=external_services,
            dst_nodes=internal_clients,
            src_port_start=base_src_port + 6000,
            protocol=6,
            start_timestamp=start_timestamp + pd.Timedelta(minutes=20),
            spacing_seconds=10,
            src_stride=2,
            dst_stride=1,
        )
    )
    flow_rows.extend(
        assign_profile(
            take("client_to_service"),
            src_nodes=internal_clients,
            dst_nodes=internal_services,
            src_port_start=base_src_port + 12000,
            protocol=6,
            start_timestamp=start_timestamp + pd.Timedelta(minutes=45),
            spacing_seconds=7,
            src_stride=3,
            dst_stride=1,
        )
    )
    flow_rows.extend(
        assign_profile(
            take("service_to_client"),
            src_nodes=internal_services,
            dst_nodes=internal_clients,
            src_port_start=base_src_port + 18000,
            protocol=6,
            start_timestamp=start_timestamp + pd.Timedelta(hours=1, minutes=5),
            spacing_seconds=8,
            src_stride=1,
            dst_stride=2,
        )
    )
    flow_rows.extend(
        assign_profile(
            take("service_to_service"),
            src_nodes=internal_services,
            dst_nodes=internal_services,
            src_port_start=base_src_port + 24000,
            protocol=17,
            start_timestamp=start_timestamp + pd.Timedelta(hours=1, minutes=28),
            spacing_seconds=12,
            src_stride=1,
            dst_stride=3,
        )
    )
    flow_rows.extend(
        assign_profile(
            take("external_to_service"),
            src_nodes=external_services,
            dst_nodes=internal_services,
            src_port_start=base_src_port + 30000,
            protocol=17,
            start_timestamp=start_timestamp + pd.Timedelta(hours=1, minutes=52),
            spacing_seconds=11,
            src_stride=1,
            dst_stride=1,
        )
    )
    return flow_rows, profile_sizes


def build_multi_anomaly_sample_dataframe(
    buckets: dict[str, list[dict[str, object]]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    flow_rows: list[dict[str, object]] = []
    node_truth_rows: list[dict[str, object]] = []

    monitored_nodes = ATTACKER_NODES + VICTIM_NODES + COMPROMISED_NODES + MULTI_ANOMALY_BENIGN_NODES
    for node in monitored_nodes:
        node_truth_rows.append(
            {
                "node_id": node.node_id,
                "ground_truth_role": node.ground_truth_role,
            }
        )

    start = pd.Timestamp("2026-05-10 09:00:00")

    attacker_plan = [("DDoS", 80), ("PortScan", 70), ("DoS Hulk", 40), (BENIGN_LABEL, 20)]
    attacker_records = select_records(buckets, attacker_plan)
    split_a = 105
    flow_rows.extend(
        assign_network_fields(
            attacker_records[:split_a],
            src_ip=ATTACKER_NODES[0].node_id,
            dst_ips=[node.node_id for node in VICTIM_NODES + COMPROMISED_NODES],
            src_port_start=20000,
            protocol=6,
            start_timestamp=start,
            spacing_seconds=12,
        )
    )
    flow_rows.extend(
        assign_network_fields(
            attacker_records[split_a:],
            src_ip=ATTACKER_NODES[1].node_id,
            dst_ips=[node.node_id for node in VICTIM_NODES + COMPROMISED_NODES],
            src_port_start=24000,
            protocol=6,
            start_timestamp=start + pd.Timedelta(minutes=8),
            spacing_seconds=14,
        )
    )

    victim_plan = [("DoS Slowhttptest", 45), ("DoS slowloris", 45), ("DDoS", 20), (BENIGN_LABEL, 30)]
    victim_records = select_records(buckets, victim_plan)
    split_v = 70
    flow_rows.extend(
        assign_inbound_fields(
            victim_records[:split_v],
            dst_ip=VICTIM_NODES[0].node_id,
            src_ips=INTERNET_NODES + [node.node_id for node in ATTACKER_NODES],
            src_port_start=30000,
            protocol=6,
            start_timestamp=start + pd.Timedelta(minutes=20),
            spacing_seconds=11,
        )
    )
    flow_rows.extend(
        assign_inbound_fields(
            victim_records[split_v:],
            dst_ip=VICTIM_NODES[1].node_id,
            src_ips=list(reversed(INTERNET_NODES)) + [node.node_id for node in ATTACKER_NODES],
            src_port_start=34000,
            protocol=6,
            start_timestamp=start + pd.Timedelta(minutes=30),
            spacing_seconds=13,
        )
    )

    compromised_out_plan = [
        ("Infiltration", 18),
        ("Bot", 24),
        ("FTP-Patator", 18),
        ("SSH-Patator", 18),
        (BENIGN_LABEL, 22),
    ]
    compromised_out = select_records(buckets, compromised_out_plan)
    split_co = 50
    flow_rows.extend(
        assign_network_fields(
            compromised_out[:split_co],
            src_ip=COMPROMISED_NODES[0].node_id,
            dst_ips=INTERNET_NODES + [node.node_id for node in VICTIM_NODES],
            src_port_start=38000,
            protocol=6,
            start_timestamp=start + pd.Timedelta(minutes=40),
            spacing_seconds=10,
        )
    )
    flow_rows.extend(
        assign_network_fields(
            compromised_out[split_co:],
            src_ip=COMPROMISED_NODES[1].node_id,
            dst_ips=list(reversed(INTERNET_NODES)) + [node.node_id for node in VICTIM_NODES],
            src_port_start=42000,
            protocol=6,
            start_timestamp=start + pd.Timedelta(minutes=48),
            spacing_seconds=9,
        )
    )

    compromised_in_plan = [("Infiltration", 18), ("Bot", 16), ("PortScan", 18), (BENIGN_LABEL, 18)]
    compromised_in = select_records(buckets, compromised_in_plan)
    split_ci = 35
    flow_rows.extend(
        assign_inbound_fields(
            compromised_in[:split_ci],
            dst_ip=COMPROMISED_NODES[0].node_id,
            src_ips=INTERNET_NODES + [node.node_id for node in ATTACKER_NODES],
            src_port_start=46000,
            protocol=17,
            start_timestamp=start + pd.Timedelta(minutes=56),
            spacing_seconds=8,
        )
    )
    flow_rows.extend(
        assign_inbound_fields(
            compromised_in[split_ci:],
            dst_ip=COMPROMISED_NODES[1].node_id,
            src_ips=list(reversed(INTERNET_NODES)) + [node.node_id for node in ATTACKER_NODES],
            src_port_start=50000,
            protocol=17,
            start_timestamp=start + pd.Timedelta(hours=1, minutes=2),
            spacing_seconds=8,
        )
    )

    benign_plan = [(BENIGN_LABEL, 220)]
    benign_records = select_records(buckets, benign_plan)
    benign_targets = [node.node_id for node in MULTI_ANOMALY_BENIGN_NODES]
    for node_index, node in enumerate(MULTI_ANOMALY_BENIGN_NODES):
        chunk = benign_records[node_index * 55 : (node_index + 1) * 55]
        flow_rows.extend(
            assign_network_fields(
                chunk[:28],
                src_ip=node.node_id,
                dst_ips=INTERNET_NODES,
                src_port_start=54000 + node_index * 1000,
                protocol=6,
                start_timestamp=start + pd.Timedelta(hours=1, minutes=10 + node_index * 3),
                spacing_seconds=15,
            )
        )
        flow_rows.extend(
            assign_inbound_fields(
                chunk[28:],
                dst_ip=node.node_id,
                src_ips=[ip for ip in INTERNET_NODES if ip not in benign_targets],
                src_port_start=58000 + node_index * 1000,
                protocol=17,
                start_timestamp=start + pd.Timedelta(hours=1, minutes=12 + node_index * 3),
                spacing_seconds=16,
            )
        )

    frame = order_flow_dataframe(flow_rows)
    ground_truth_df = pd.DataFrame(node_truth_rows)
    manifest = {
        "scenario": MULTI_ANOMALY_SCENARIO,
        "scenario_name": get_scenario_display_name(MULTI_ANOMALY_SCENARIO),
        "total_flows": int(len(frame)),
        "total_nodes": int(len(ground_truth_df)),
        "labels": {
            str(label): int(count)
            for label, count in frame[LABEL_COLUMN].value_counts().sort_index().items()
        },
        "nodes": node_truth_rows,
    }
    return frame, ground_truth_df, manifest


def build_single_anomaly_sample_dataframe(
    buckets: dict[str, list[dict[str, object]]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    flow_rows: list[dict[str, object]] = []
    monitored_nodes = [SINGLE_ANOMALY_NODE] + SINGLE_ANOMALY_BENIGN_NODES
    node_truth_rows = [
        {
            "node_id": node.node_id,
            "ground_truth_role": node.ground_truth_role,
        }
        for node in monitored_nodes
    ]

    start = pd.Timestamp("2026-05-10 15:00:00")

    compromised_out_plan = [
        ("Infiltration", 24),
        ("Bot", 26),
        ("FTP-Patator", 18),
        ("SSH-Patator", 18),
        (BENIGN_LABEL, 24),
    ]
    compromised_out = select_records(buckets, compromised_out_plan)
    split_out = 58
    flow_rows.extend(
        assign_network_fields(
            compromised_out[:split_out],
            src_ip=SINGLE_ANOMALY_NODE.node_id,
            dst_ips=INTERNET_NODES,
            src_port_start=21000,
            protocol=6,
            start_timestamp=start,
            spacing_seconds=12,
        )
    )
    flow_rows.extend(
        assign_network_fields(
            compromised_out[split_out:],
            src_ip=SINGLE_ANOMALY_NODE.node_id,
            dst_ips=list(reversed(INTERNET_NODES)),
            src_port_start=25000,
            protocol=17,
            start_timestamp=start + pd.Timedelta(minutes=12),
            spacing_seconds=11,
        )
    )

    compromised_in_plan = [("Infiltration", 18), ("Bot", 16), ("PortScan", 16), (BENIGN_LABEL, 20)]
    compromised_in = select_records(buckets, compromised_in_plan)
    split_in = 34
    flow_rows.extend(
        assign_inbound_fields(
            compromised_in[:split_in],
            dst_ip=SINGLE_ANOMALY_NODE.node_id,
            src_ips=INTERNET_NODES,
            src_port_start=30000,
            protocol=6,
            start_timestamp=start + pd.Timedelta(minutes=24),
            spacing_seconds=10,
        )
    )
    flow_rows.extend(
        assign_inbound_fields(
            compromised_in[split_in:],
            dst_ip=SINGLE_ANOMALY_NODE.node_id,
            src_ips=list(reversed(INTERNET_NODES)),
            src_port_start=34000,
            protocol=17,
            start_timestamp=start + pd.Timedelta(minutes=31),
            spacing_seconds=10,
        )
    )

    benign_plan = [(BENIGN_LABEL, 200)]
    benign_records = select_records(buckets, benign_plan)
    for node_index, node in enumerate(SINGLE_ANOMALY_BENIGN_NODES):
        chunk = benign_records[node_index * 50 : (node_index + 1) * 50]
        flow_rows.extend(
            assign_network_fields(
                chunk[:26],
                src_ip=node.node_id,
                dst_ips=EXTERNAL_SERVICES,
                src_port_start=40000 + node_index * 1000,
                protocol=6,
                start_timestamp=start + pd.Timedelta(minutes=45 + node_index * 4),
                spacing_seconds=16,
            )
        )
        flow_rows.extend(
            assign_inbound_fields(
                chunk[26:],
                dst_ip=node.node_id,
                src_ips=list(reversed(EXTERNAL_SERVICES)),
                src_port_start=43000 + node_index * 1000,
                protocol=17,
                start_timestamp=start + pd.Timedelta(minutes=47 + node_index * 4),
                spacing_seconds=18,
            )
        )

    frame = order_flow_dataframe(flow_rows)
    ground_truth_df = pd.DataFrame(node_truth_rows)
    manifest = {
        "scenario": SINGLE_ANOMALY_SCENARIO,
        "scenario_name": get_scenario_display_name(SINGLE_ANOMALY_SCENARIO),
        "total_flows": int(len(frame)),
        "total_nodes": int(len(ground_truth_df)),
        "labels": {
            str(label): int(count)
            for label, count in frame[LABEL_COLUMN].value_counts().sort_index().items()
        },
        "primary_anomaly_node": SINGLE_ANOMALY_NODE.node_id,
        "nodes": node_truth_rows,
    }
    return frame, ground_truth_df, manifest


def build_no_anomaly_sample_dataframe(
    benign_rows: list[dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    profile_sizes = {
        "client_to_external": 480,
        "external_to_client": 240,
        "client_to_service": 300,
        "service_to_client": 180,
        "service_to_service": 120,
        "external_to_service": 120,
    }
    if sum(profile_sizes.values()) != len(benign_rows):
        raise RuntimeError("Profile sizes do not match benign sample size.")

    offsets: dict[str, tuple[int, int]] = {}
    cursor = 0
    for name, size in profile_sizes.items():
        offsets[name] = (cursor, cursor + size)
        cursor += size

    def take(profile_name: str) -> list[dict[str, object]]:
        start, end = offsets[profile_name]
        return benign_rows[start:end]

    flow_rows: list[dict[str, object]] = []
    start = pd.Timestamp("2026-05-11 09:00:00")

    flow_rows.extend(
        assign_profile(
            take("client_to_external"),
            src_nodes=INTERNAL_CLIENTS,
            dst_nodes=EXTERNAL_SERVICES,
            src_port_start=20000,
            protocol=6,
            start_timestamp=start,
            spacing_seconds=12,
            src_stride=1,
            dst_stride=3,
        )
    )
    flow_rows.extend(
        assign_profile(
            take("external_to_client"),
            src_nodes=EXTERNAL_SERVICES,
            dst_nodes=INTERNAL_CLIENTS,
            src_port_start=26000,
            protocol=6,
            start_timestamp=start + pd.Timedelta(minutes=22),
            spacing_seconds=15,
            src_stride=2,
            dst_stride=1,
        )
    )
    flow_rows.extend(
        assign_profile(
            take("client_to_service"),
            src_nodes=INTERNAL_CLIENTS,
            dst_nodes=INTERNAL_SERVICES,
            src_port_start=32000,
            protocol=6,
            start_timestamp=start + pd.Timedelta(minutes=50),
            spacing_seconds=10,
            src_stride=3,
            dst_stride=1,
        )
    )
    flow_rows.extend(
        assign_profile(
            take("service_to_client"),
            src_nodes=INTERNAL_SERVICES,
            dst_nodes=INTERNAL_CLIENTS,
            src_port_start=38000,
            protocol=6,
            start_timestamp=start + pd.Timedelta(hours=1, minutes=16),
            spacing_seconds=11,
            src_stride=1,
            dst_stride=2,
        )
    )
    flow_rows.extend(
        assign_profile(
            take("service_to_service"),
            src_nodes=INTERNAL_SERVICES,
            dst_nodes=INTERNAL_SERVICES,
            src_port_start=44000,
            protocol=17,
            start_timestamp=start + pd.Timedelta(hours=1, minutes=42),
            spacing_seconds=18,
            src_stride=1,
            dst_stride=3,
        )
    )
    flow_rows.extend(
        assign_profile(
            take("external_to_service"),
            src_nodes=EXTERNAL_SERVICES,
            dst_nodes=INTERNAL_SERVICES,
            src_port_start=50000,
            protocol=17,
            start_timestamp=start + pd.Timedelta(hours=2, minutes=8),
            spacing_seconds=20,
            src_stride=1,
            dst_stride=1,
        )
    )

    frame = order_flow_dataframe(flow_rows)
    ground_truth_df = pd.DataFrame(
        [{"node_id": node_id, "ground_truth_role": "uncertain"} for node_id in ALL_BENIGN_NODES]
    )
    manifest = {
        "scenario": NO_ANOMALY_SCENARIO,
        "scenario_name": get_scenario_display_name(NO_ANOMALY_SCENARIO),
        "total_flows": int(len(frame)),
        "total_nodes": int(len(ground_truth_df)),
        "labels": {
            str(label): int(count)
            for label, count in frame[LABEL_COLUMN].value_counts().sort_index().items()
        },
        "profiles": profile_sizes,
        "nodes": {
            "internal_clients": INTERNAL_CLIENTS,
            "internal_services": INTERNAL_SERVICES,
            "external_services": EXTERNAL_SERVICES,
        },
    }
    return frame, ground_truth_df, manifest


def build_large_no_anomaly_sample_dataframe(
    benign_rows: list[dict[str, object]],
    target_node_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    node_groups = build_large_benign_node_groups(
        target_node_count,
        client_prefix="10.10.10",
        service_prefix="10.10.20",
        external_prefix="198.18.10",
        weights={
            "internal_clients": 0.45,
            "internal_services": 0.20,
            "external_services": 0.35,
        },
        minimums={
            "internal_clients": 8,
            "internal_services": 4,
            "external_services": 8,
        },
    )
    flow_rows, profile_sizes = build_benign_background_flows(
        benign_rows,
        internal_clients=node_groups["internal_clients"],
        internal_services=node_groups["internal_services"],
        external_services=node_groups["external_services"],
        start_timestamp=pd.Timestamp("2026-05-11 09:00:00"),
        base_src_port=20000,
    )
    frame = order_flow_dataframe(flow_rows)
    all_nodes = flatten_node_groups(node_groups)
    ground_truth_df = pd.DataFrame(
        [{"node_id": node_id, "ground_truth_role": "uncertain"} for node_id in all_nodes]
    )
    manifest = {
        "scenario": NO_ANOMALY_SCENARIO,
        "scenario_name": get_scenario_display_name(NO_ANOMALY_SCENARIO),
        "sample_profile": "large",
        "requested_node_count": int(target_node_count),
        "total_flows": int(len(frame)),
        "total_nodes": int(len(ground_truth_df)),
        "labels": {
            str(label): int(count)
            for label, count in frame[LABEL_COLUMN].value_counts().sort_index().items()
        },
        "profiles": profile_sizes,
        "node_group_counts": {key: int(len(value)) for key, value in node_groups.items()},
        "nodes": node_groups,
    }
    return frame, ground_truth_df, manifest


def build_large_single_anomaly_sample_dataframe(
    buckets: dict[str, list[dict[str, object]]],
    target_node_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    benign_node_count = target_node_count - 1
    node_groups = build_large_benign_node_groups(
        benign_node_count,
        client_prefix="10.50.10",
        service_prefix="10.50.20",
        external_prefix="198.18.50",
        weights={
            "internal_clients": 0.30,
            "internal_services": 0.15,
            "external_services": 0.55,
        },
        minimums={
            "internal_clients": 8,
            "internal_services": 4,
            "external_services": 12,
        },
    )
    benign_nodes = flatten_node_groups(node_groups)
    monitored_nodes = [SINGLE_ANOMALY_NODE] + [NodeSpec(node_id, "uncertain") for node_id in benign_nodes]
    node_truth_rows = [
        {
            "node_id": node.node_id,
            "ground_truth_role": node.ground_truth_role,
        }
        for node in monitored_nodes
    ]

    start = pd.Timestamp("2026-05-10 15:00:00")
    flow_rows: list[dict[str, object]] = []

    compromised_out_plan = [
        ("Infiltration", 30),
        ("Bot", 220),
        ("FTP-Patator", 160),
        ("SSH-Patator", 160),
        (BENIGN_LABEL, 220),
    ]
    compromised_out = select_records(buckets, compromised_out_plan)
    split_out = len(compromised_out) // 2
    flow_rows.extend(
        assign_network_fields(
            compromised_out[:split_out],
            src_ip=SINGLE_ANOMALY_NODE.node_id,
            dst_ips=node_groups["external_services"],
            src_port_start=21000,
            protocol=6,
            start_timestamp=start,
            spacing_seconds=6,
        )
    )
    flow_rows.extend(
        assign_network_fields(
            compromised_out[split_out:],
            src_ip=SINGLE_ANOMALY_NODE.node_id,
            dst_ips=list(reversed(node_groups["external_services"])),
            src_port_start=27000,
            protocol=17,
            start_timestamp=start + pd.Timedelta(minutes=18),
            spacing_seconds=6,
        )
    )

    compromised_in_plan = [
        ("Infiltration", 30),
        ("Bot", 180),
        ("PortScan", 180),
        (BENIGN_LABEL, 220),
    ]
    compromised_in = select_records(buckets, compromised_in_plan)
    split_in = len(compromised_in) // 2
    flow_rows.extend(
        assign_inbound_fields(
            compromised_in[:split_in],
            dst_ip=SINGLE_ANOMALY_NODE.node_id,
            src_ips=node_groups["external_services"],
            src_port_start=33000,
            protocol=6,
            start_timestamp=start + pd.Timedelta(minutes=36),
            spacing_seconds=6,
        )
    )
    flow_rows.extend(
        assign_inbound_fields(
            compromised_in[split_in:],
            dst_ip=SINGLE_ANOMALY_NODE.node_id,
            src_ips=list(reversed(node_groups["external_services"])),
            src_port_start=39000,
            protocol=17,
            start_timestamp=start + pd.Timedelta(minutes=52),
            spacing_seconds=6,
        )
    )

    benign_background_count = max(3600, benign_node_count * 72)
    benign_rows = select_records(buckets, [(BENIGN_LABEL, benign_background_count)])
    background_flows, background_profiles = build_benign_background_flows(
        benign_rows,
        internal_clients=node_groups["internal_clients"],
        internal_services=node_groups["internal_services"],
        external_services=node_groups["external_services"],
        start_timestamp=start + pd.Timedelta(hours=2),
        base_src_port=45000,
    )
    flow_rows.extend(background_flows)

    frame = order_flow_dataframe(flow_rows)
    ground_truth_df = pd.DataFrame(node_truth_rows)
    manifest = {
        "scenario": SINGLE_ANOMALY_SCENARIO,
        "scenario_name": get_scenario_display_name(SINGLE_ANOMALY_SCENARIO),
        "sample_profile": "large",
        "requested_node_count": int(target_node_count),
        "total_flows": int(len(frame)),
        "total_nodes": int(len(ground_truth_df)),
        "labels": {
            str(label): int(count)
            for label, count in frame[LABEL_COLUMN].value_counts().sort_index().items()
        },
        "primary_anomaly_node": SINGLE_ANOMALY_NODE.node_id,
        "background_profiles": background_profiles,
        "node_group_counts": {key: int(len(value)) for key, value in node_groups.items()},
        "nodes": node_truth_rows,
    }
    return frame, ground_truth_df, manifest


def build_large_multi_anomaly_sample_dataframe(
    buckets: dict[str, list[dict[str, object]]],
    target_node_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    benign_node_count = target_node_count - len(ATTACKER_NODES) - len(VICTIM_NODES) - len(COMPROMISED_NODES)
    node_groups = build_large_benign_node_groups(
        benign_node_count,
        client_prefix="10.40.10",
        service_prefix="10.40.20",
        external_prefix="198.18.40",
        weights={
            "internal_clients": 0.30,
            "internal_services": 0.15,
            "external_services": 0.55,
        },
        minimums={
            "internal_clients": 8,
            "internal_services": 4,
            "external_services": 12,
        },
    )
    benign_nodes = flatten_node_groups(node_groups)
    monitored_nodes = (
        ATTACKER_NODES
        + VICTIM_NODES
        + COMPROMISED_NODES
        + [NodeSpec(node_id, "uncertain") for node_id in benign_nodes]
    )
    node_truth_rows = [
        {
            "node_id": node.node_id,
            "ground_truth_role": node.ground_truth_role,
        }
        for node in monitored_nodes
    ]

    start = pd.Timestamp("2026-05-10 09:00:00")
    flow_rows: list[dict[str, object]] = []

    attacker_plan = [("DDoS", 240), ("PortScan", 220), ("DoS Hulk", 120), (BENIGN_LABEL, 120)]
    attacker_records = select_records(buckets, attacker_plan)
    split_a = len(attacker_records) // 2
    attacker_targets = [node.node_id for node in VICTIM_NODES + COMPROMISED_NODES]
    flow_rows.extend(
        assign_network_fields(
            attacker_records[:split_a],
            src_ip=ATTACKER_NODES[0].node_id,
            dst_ips=attacker_targets,
            src_port_start=20000,
            protocol=6,
            start_timestamp=start,
            spacing_seconds=5,
        )
    )
    flow_rows.extend(
        assign_network_fields(
            attacker_records[split_a:],
            src_ip=ATTACKER_NODES[1].node_id,
            dst_ips=list(reversed(attacker_targets)),
            src_port_start=26000,
            protocol=6,
            start_timestamp=start + pd.Timedelta(minutes=10),
            spacing_seconds=5,
        )
    )

    victim_plan = [("DoS Slowhttptest", 180), ("DoS slowloris", 180), ("DDoS", 120), (BENIGN_LABEL, 140)]
    victim_records = select_records(buckets, victim_plan)
    split_v = len(victim_records) // 2
    victim_sources = node_groups["external_services"] + [node.node_id for node in ATTACKER_NODES]
    flow_rows.extend(
        assign_inbound_fields(
            victim_records[:split_v],
            dst_ip=VICTIM_NODES[0].node_id,
            src_ips=victim_sources,
            src_port_start=32000,
            protocol=6,
            start_timestamp=start + pd.Timedelta(minutes=24),
            spacing_seconds=5,
        )
    )
    flow_rows.extend(
        assign_inbound_fields(
            victim_records[split_v:],
            dst_ip=VICTIM_NODES[1].node_id,
            src_ips=list(reversed(victim_sources)),
            src_port_start=38000,
            protocol=6,
            start_timestamp=start + pd.Timedelta(minutes=36),
            spacing_seconds=5,
        )
    )

    compromised_out_plan = [
        ("Infiltration", 30),
        ("Bot", 220),
        ("FTP-Patator", 180),
        ("SSH-Patator", 180),
        (BENIGN_LABEL, 180),
    ]
    compromised_out = select_records(buckets, compromised_out_plan)
    split_co = len(compromised_out) // 2
    compromised_targets = node_groups["external_services"] + [node.node_id for node in VICTIM_NODES]
    flow_rows.extend(
        assign_network_fields(
            compromised_out[:split_co],
            src_ip=COMPROMISED_NODES[0].node_id,
            dst_ips=compromised_targets,
            src_port_start=44000,
            protocol=6,
            start_timestamp=start + pd.Timedelta(minutes=48),
            spacing_seconds=5,
        )
    )
    flow_rows.extend(
        assign_network_fields(
            compromised_out[split_co:],
            src_ip=COMPROMISED_NODES[1].node_id,
            dst_ips=list(reversed(compromised_targets)),
            src_port_start=50000,
            protocol=6,
            start_timestamp=start + pd.Timedelta(minutes=58),
            spacing_seconds=5,
        )
    )

    compromised_in_plan = [("Infiltration", 30), ("Bot", 180), ("PortScan", 180), (BENIGN_LABEL, 180)]
    compromised_in = select_records(buckets, compromised_in_plan)
    split_ci = len(compromised_in) // 2
    compromised_sources = node_groups["external_services"] + [node.node_id for node in ATTACKER_NODES]
    flow_rows.extend(
        assign_inbound_fields(
            compromised_in[:split_ci],
            dst_ip=COMPROMISED_NODES[0].node_id,
            src_ips=compromised_sources,
            src_port_start=56000,
            protocol=17,
            start_timestamp=start + pd.Timedelta(hours=1, minutes=10),
            spacing_seconds=5,
        )
    )
    flow_rows.extend(
        assign_inbound_fields(
            compromised_in[split_ci:],
            dst_ip=COMPROMISED_NODES[1].node_id,
            src_ips=list(reversed(compromised_sources)),
            src_port_start=62000,
            protocol=17,
            start_timestamp=start + pd.Timedelta(hours=1, minutes=20),
            spacing_seconds=5,
        )
    )

    benign_background_count = max(4800, benign_node_count * 75)
    benign_rows = select_records(buckets, [(BENIGN_LABEL, benign_background_count)])
    background_flows, background_profiles = build_benign_background_flows(
        benign_rows,
        internal_clients=node_groups["internal_clients"],
        internal_services=node_groups["internal_services"],
        external_services=node_groups["external_services"],
        start_timestamp=start + pd.Timedelta(hours=2),
        base_src_port=18000,
    )
    flow_rows.extend(background_flows)

    frame = order_flow_dataframe(flow_rows)
    ground_truth_df = pd.DataFrame(node_truth_rows)
    manifest = {
        "scenario": MULTI_ANOMALY_SCENARIO,
        "scenario_name": get_scenario_display_name(MULTI_ANOMALY_SCENARIO),
        "sample_profile": "large",
        "requested_node_count": int(target_node_count),
        "total_flows": int(len(frame)),
        "total_nodes": int(len(ground_truth_df)),
        "labels": {
            str(label): int(count)
            for label, count in frame[LABEL_COLUMN].value_counts().sort_index().items()
        },
        "background_profiles": background_profiles,
        "node_group_counts": {key: int(len(value)) for key, value in node_groups.items()},
        "nodes": node_truth_rows,
    }
    return frame, ground_truth_df, manifest


def write_generated_sample(
    *,
    scenario: str,
    dataset_dir: Path,
    output_dir: Path,
    sample_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    manifest: dict[str, object],
    seed: int | None,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = get_scenario_paths(output_dir, scenario)
    sample_df.to_csv(paths["sample_csv"], index=False, encoding="utf-8-sig")
    ground_truth_df.to_csv(paths["ground_truth_csv"], index=False, encoding="utf-8-sig")
    payload = {
        **manifest,
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "output_csv": str(paths["sample_csv"]),
        "ground_truth_csv": str(paths["ground_truth_csv"]),
        "generation_seed": seed,
    }
    write_json(paths["manifest_json"], payload)
    return {
        "scenario": ensure_supported_scenario(scenario),
        "scenario_name": get_scenario_display_name(scenario),
        "output_dir": str(output_dir),
        "sample_csv": str(paths["sample_csv"]),
        "ground_truth_csv": str(paths["ground_truth_csv"]),
        "manifest_json": str(paths["manifest_json"]),
        "total_flows": int(len(sample_df)),
        "total_nodes": int(len(ground_truth_df)),
        "generation_seed": seed,
    }


def generate_multi_anomaly_sample(
    dataset_dir: Path,
    output_dir: Path,
    seed: int | None = None,
    target_node_count: int | None = None,
) -> dict[str, object]:
    labels_needed = {BENIGN_LABEL, *ATTACKER_LABELS, *COMPROMISED_LABELS, *VICTIM_LABELS}
    rng = build_rng(seed)
    buckets = stream_rows_for_labels(dataset_dir, labels_needed, rng=rng)
    if target_node_count is None:
        sample_df, ground_truth_df, manifest = build_multi_anomaly_sample_dataframe(buckets)
    else:
        sample_df, ground_truth_df, manifest = build_large_multi_anomaly_sample_dataframe(
            buckets,
            target_node_count=target_node_count,
        )
    return write_generated_sample(
        scenario=MULTI_ANOMALY_SCENARIO,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        sample_df=sample_df,
        ground_truth_df=ground_truth_df,
        manifest=manifest,
        seed=seed,
    )


def generate_single_anomaly_sample(
    dataset_dir: Path,
    output_dir: Path,
    seed: int | None = None,
    target_node_count: int | None = None,
) -> dict[str, object]:
    labels_needed = {BENIGN_LABEL, *SINGLE_ANOMALY_LABELS}
    rng = build_rng(seed)
    buckets = stream_rows_for_labels(dataset_dir, labels_needed, rng=rng)
    if target_node_count is None:
        sample_df, ground_truth_df, manifest = build_single_anomaly_sample_dataframe(buckets)
    else:
        sample_df, ground_truth_df, manifest = build_large_single_anomaly_sample_dataframe(
            buckets,
            target_node_count=target_node_count,
        )
    return write_generated_sample(
        scenario=SINGLE_ANOMALY_SCENARIO,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        sample_df=sample_df,
        ground_truth_df=ground_truth_df,
        manifest=manifest,
        seed=seed,
    )


def generate_no_anomaly_sample(
    dataset_dir: Path,
    output_dir: Path,
    seed: int | None = None,
    target_node_count: int | None = None,
) -> dict[str, object]:
    rng = build_rng(seed)
    total_flows = NO_ANOMALY_TOTAL_FLOWS if target_node_count is None else max(4200, target_node_count * 90)
    benign_rows = load_benign_rows(dataset_dir, total_flows, rng=rng)
    if target_node_count is None:
        sample_df, ground_truth_df, manifest = build_no_anomaly_sample_dataframe(benign_rows)
    else:
        sample_df, ground_truth_df, manifest = build_large_no_anomaly_sample_dataframe(
            benign_rows,
            target_node_count=target_node_count,
        )
    return write_generated_sample(
        scenario=NO_ANOMALY_SCENARIO,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        sample_df=sample_df,
        ground_truth_df=ground_truth_df,
        manifest=manifest,
        seed=seed,
    )


def generate_test_samples(
    dataset_dir: Path,
    output_root: Path,
    scenarios: list[str],
    seed: int | None = None,
    node_count_overrides: dict[str, int] | None = None,
) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    master_rng = build_rng(seed)
    node_count_overrides = node_count_overrides or {}
    generated: list[dict[str, object]] = []
    for scenario in scenarios:
        normalized = ensure_supported_scenario(scenario)
        scenario_output_dir = output_root / normalized
        scenario_seed = int(master_rng.integers(0, 2**32 - 1))
        target_node_count = node_count_overrides.get(normalized)
        if normalized == NO_ANOMALY_SCENARIO:
            generated.append(
                generate_no_anomaly_sample(
                    dataset_dir,
                    scenario_output_dir,
                    seed=scenario_seed,
                    target_node_count=target_node_count,
                )
            )
        elif normalized == SINGLE_ANOMALY_SCENARIO:
            generated.append(
                generate_single_anomaly_sample(
                    dataset_dir,
                    scenario_output_dir,
                    seed=scenario_seed,
                    target_node_count=target_node_count,
                )
            )
        elif normalized == MULTI_ANOMALY_SCENARIO:
            generated.append(
                generate_multi_anomaly_sample(
                    dataset_dir,
                    scenario_output_dir,
                    seed=scenario_seed,
                    target_node_count=target_node_count,
                )
            )
    suite_manifest = {
        "dataset_dir": str(dataset_dir),
        "output_root": str(output_root),
        "generation_seed": seed,
        "node_count_overrides": {str(key): int(value) for key, value in node_count_overrides.items()},
        "scenarios": generated,
    }
    write_json(output_root / "suite_manifest.json", suite_manifest)
    return suite_manifest


def choose_final_role(group: pd.DataFrame) -> str:
    ordered = group.sort_values(
        by=["compromised_score", "attacker_score", "victim_score", "anomaly_ratio"],
        ascending=[False, False, False, False],
    )
    role = str(ordered.iloc[0]["node_role"])
    return role if role in ROLE_SEVERITY else "uncertain"


def evaluate_role_summary(
    summary_path: Path,
    ground_truth_path: Path,
    output_path: Path,
) -> dict[str, object]:
    summary_df = pd.read_csv(summary_path, low_memory=False)
    truth_df = pd.read_csv(ground_truth_path, low_memory=False)

    predicted = (
        summary_df.groupby("node_id", sort=False)
        .apply(choose_final_role, include_groups=False)
        .reset_index(name="predicted_role")
    )
    merged = truth_df.merge(predicted, on="node_id", how="left")
    merged["predicted_role"] = merged["predicted_role"].fillna("missing")
    merged["correct"] = merged["ground_truth_role"] == merged["predicted_role"]

    confusion = merged.groupby(["ground_truth_role", "predicted_role"]).size().reset_index(name="count")
    role_accuracy = merged.groupby("ground_truth_role")["correct"].mean().reset_index(name="accuracy")
    report = {
        "summary_path": str(summary_path),
        "ground_truth_path": str(ground_truth_path),
        "nodes_evaluated": int(len(merged)),
        "overall_accuracy": float(merged["correct"].mean() if len(merged) else 0.0),
        "role_accuracy": [
            {
                "ground_truth_role": str(row["ground_truth_role"]),
                "accuracy": float(row["accuracy"]),
            }
            for _, row in role_accuracy.iterrows()
        ],
        "confusion": [
            {
                "ground_truth_role": str(row["ground_truth_role"]),
                "predicted_role": str(row["predicted_role"]),
                "count": int(row["count"]),
            }
            for _, row in confusion.iterrows()
        ],
        "details": merged.to_dict(orient="records"),
    }
    write_json(output_path, report)
    return report


def evaluate_node_eval_summary(
    summary_path: Path,
    ground_truth_path: Path,
    output_path: Path,
) -> dict[str, object]:
    return evaluate_role_summary(
        summary_path=summary_path,
        ground_truth_path=ground_truth_path,
        output_path=output_path,
    )


def choose_benign_final_role(group: pd.DataFrame) -> str:
    ordered = group.assign(
        role_severity=group["node_role"].astype(str).map(lambda value: ROLE_SEVERITY.get(value, 0))
    ).sort_values(
        by=["role_severity", "compromised_score", "attacker_score", "victim_score", "anomaly_ratio"],
        ascending=[False, False, False, False, False],
    )
    return str(ordered.iloc[0]["node_role"])


def evaluate_flow_fpr(predictions_path: Path) -> dict[str, float] | None:
    if not predictions_path.exists():
        return None
    df = pd.read_csv(predictions_path, low_memory=False)
    if LABEL_COLUMN not in df.columns or "is_anomaly" not in df.columns:
        return None
    labels = df[LABEL_COLUMN].astype(str).str.strip()
    benign_df = df[labels == BENIGN_LABEL].copy()
    if benign_df.empty:
        return None
    benign_df["is_anomaly"] = benign_df["is_anomaly"].astype(str).str.lower().map(
        {"true": True, "false": False}
    ).fillna(False)
    false_positive_flows = int(benign_df["is_anomaly"].sum())
    return {
        "benign_flows": int(len(benign_df)),
        "false_positive_flows": false_positive_flows,
        "flow_false_positive_rate": float(false_positive_flows / len(benign_df)),
    }


def evaluate_no_anomaly_summary(
    predictions_path: Path,
    summary_path: Path,
    ground_truth_path: Path,
    output_path: Path,
) -> dict[str, object]:
    truth_df = pd.read_csv(ground_truth_path, low_memory=False)
    summary_df = pd.read_csv(summary_path, low_memory=False)

    summary_subset = summary_df[
        [
            "node_id",
            "window_start",
            "node_role",
            "attacker_score",
            "victim_score",
            "compromised_score",
            "anomaly_ratio",
        ]
    ].copy()
    truth_nodes = set(truth_df["node_id"].astype(str))
    summary_subset = summary_subset[summary_subset["node_id"].astype(str).isin(truth_nodes)].copy()
    summary_subset["is_fp_window"] = summary_subset["node_role"].astype(str) != "uncertain"

    window_false_positive_rate = float(summary_subset["is_fp_window"].mean()) if len(summary_subset) else 0.0
    false_positive_windows = int(summary_subset["is_fp_window"].sum())

    node_roles = (
        summary_subset.groupby("node_id", sort=False)
        .apply(choose_benign_final_role, include_groups=False)
        .reset_index(name="predicted_role")
    )
    node_fp = (
        summary_subset.groupby("node_id", sort=False)["is_fp_window"]
        .any()
        .reset_index(name="has_false_positive_window")
    )
    node_report = truth_df.merge(node_roles, on="node_id", how="left").merge(node_fp, on="node_id", how="left")
    node_report["predicted_role"] = node_report["predicted_role"].fillna("missing")
    node_report["has_false_positive_window"] = node_report["has_false_positive_window"].fillna(False)
    node_report["is_false_positive_node"] = node_report["has_false_positive_window"].astype(bool)

    node_false_positive_rate_any_window = float(node_report["is_false_positive_node"].mean())
    nodes_with_any_fp = int(node_report["is_false_positive_node"].sum())

    role_counts = (
        summary_subset["node_role"].astype(str).value_counts().sort_index().to_dict()
        if not summary_subset.empty
        else {}
    )
    node_level_role_counts = (
        node_report["predicted_role"].astype(str).value_counts().sort_index().to_dict()
        if not node_report.empty
        else {}
    )

    report = {
        "ground_truth_path": str(ground_truth_path),
        "summary_path": str(summary_path),
        "predictions_path": str(predictions_path),
        "benign_nodes": int(len(node_report)),
        "benign_node_windows": int(len(summary_subset)),
        "false_positive_windows": false_positive_windows,
        "window_false_positive_rate": window_false_positive_rate,
        "nodes_with_any_false_positive_window": nodes_with_any_fp,
        "node_false_positive_rate_any_window": node_false_positive_rate_any_window,
        "window_level_role_counts": {str(key): int(value) for key, value in role_counts.items()},
        "node_level_role_counts": {str(key): int(value) for key, value in node_level_role_counts.items()},
        "false_positive_nodes": [
            {
                "node_id": str(row["node_id"]),
                "predicted_role": str(row["predicted_role"]),
            }
            for _, row in node_report[node_report["is_false_positive_node"]].iterrows()
        ],
        "node_details": node_report.to_dict(orient="records"),
        "flow_fpr": evaluate_flow_fpr(predictions_path),
    }
    write_json(output_path, report)
    return report


def evaluate_benign_fpr_summary(
    predictions_path: Path,
    summary_path: Path,
    ground_truth_path: Path,
    output_path: Path,
) -> dict[str, object]:
    return evaluate_no_anomaly_summary(
        predictions_path=predictions_path,
        summary_path=summary_path,
        ground_truth_path=ground_truth_path,
        output_path=output_path,
    )
