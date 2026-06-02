#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AFFECTED_LINKS = ["s3-s4", "s4-s7"]
BENIGN_TYPES = {"", "BENIGN", "NORMAL", "UNKNOWN"}
ATTACK_WEIGHTS = {
    "DDoS": 35,
    "DoS Hulk": 35,
    "DoS GoldenEye": 35,
    "DoS slowloris": 35,
    "DoS Slowhttptest": 35,
    "Heartbleed": 40,
    "Web Attack": 32,
    "Web Attack Brute Force": 32,
    "Web Attack Sql Injection": 38,
    "Web Attack XSS": 32,
    "PortScan": 25,
    "SSH-Patator": 28,
    "FTP-Patator": 28,
    "Brute Force": 28,
}


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_json_optional(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_events(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except ValueError:
                continue
    return records


def safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def attack_types_from_events(events: Iterable[Dict]) -> List[str]:
    seen = []
    for event in events:
        attack_type = str(event.get("attack_type") or event.get("raw_matched_attack_type") or "").strip()
        if attack_type in BENIGN_TYPES:
            continue
        if attack_type not in seen:
            seen.append(attack_type)
    return seen


def latest_attack_event(events: List[Dict]) -> Optional[Dict]:
    for event in reversed(events):
        attack_type = str(event.get("attack_type") or "").strip()
        if attack_type not in BENIGN_TYPES:
            return event
    return events[-1] if events else None


def infer_links_from_active_path(active_path) -> List[str]:
    if not active_path:
        return []
    if isinstance(active_path, str):
        if "," in active_path and "-" in active_path and "->" not in active_path:
            links = [part.strip() for part in active_path.split(",") if part.strip()]
            return links
        nodes = [part.strip() for part in active_path.replace("->", ",").replace("-", ",").split(",") if part.strip()]
    elif isinstance(active_path, list):
        if all(isinstance(item, str) and "-" in item for item in active_path):
            return [str(item).strip() for item in active_path if str(item).strip()]
        if all(isinstance(item, dict) for item in active_path):
            links = []
            for item in active_path:
                src = item.get("src") or item.get("source") or item.get("from")
                dst = item.get("dst") or item.get("target") or item.get("to")
                if src and dst:
                    links.append("%s-%s" % (src, dst))
            return links
        nodes = [str(item).strip() for item in active_path if str(item).strip()]
    else:
        return []
    if len(nodes) < 2:
        return []
    return ["%s-%s" % (nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]


def affected_links_from_network_status(network_status: Optional[Dict]) -> List[str]:
    if not network_status:
        return list(DEFAULT_AFFECTED_LINKS)
    for key in ["active_path", "path", "current_path"]:
        links = infer_links_from_active_path(network_status.get(key))
        if links:
            return links
    return list(DEFAULT_AFFECTED_LINKS)


def affected_nodes_from_links(links: Iterable[str]) -> List[str]:
    nodes = []
    for link in links:
        for node in str(link).split("-"):
            node = node.strip()
            if node and node not in nodes:
                nodes.append(node)
    return nodes


def actions_from_plan(records: Iterable[Dict]) -> List[str]:
    actions = []
    for record in records:
        action_type = str(record.get("action_type") or "").strip()
        if action_type and action_type not in actions:
            actions.append(action_type)
    return actions


def compute_risk_score(adjustment_events: int, attack_types: List[str], defense_success_rate: float) -> int:
    adjustment_component = min(40, max(0, adjustment_events) * 8)
    attack_component = 0
    for attack_type in attack_types:
        attack_component = max(attack_component, ATTACK_WEIGHTS.get(attack_type, 24))
    defense_gap_component = int(round(max(0.0, 1.0 - defense_success_rate) * 30))
    return max(0, min(100, adjustment_component + attack_component + defense_gap_component))


def status_and_severity(
    adjustment_events: int,
    detection_success_rate: float,
    defense_success_rate: float,
    risk_score: int,
) -> Tuple[str, str]:
    close_to_one = detection_success_rate >= 0.95 and defense_success_rate >= 0.95
    if adjustment_events > 0 and close_to_one:
        return "attack_detected", "critical" if risk_score >= 70 else "warning"
    if adjustment_events > 0:
        return "attack_detected", "critical" if risk_score >= 65 else "warning"
    if detection_success_rate < 0.8 or defense_success_rate < 0.8:
        return "degraded", "warning"
    return "normal", "info"


def build_metrics(summary: Dict) -> Dict:
    return {
        "detector": summary.get("detector"),
        "optimizer": summary.get("optimizer"),
        "windows": safe_int(summary.get("windows")),
        "adjustment_events": safe_int(summary.get("adjustment_events")),
        "detection_success_rate": safe_float(summary.get("detection_success_rate")),
        "defense_success_rate": safe_float(summary.get("defense_success_rate")),
        "strategy_counts": summary.get("strategy_counts", {}),
    }


def build_payload(
    summary: Dict,
    events: List[Dict],
    controller_state: Optional[Dict],
    execution_plan: List[Dict],
    network_status: Optional[Dict],
) -> Dict:
    updated_at = datetime.utcnow().isoformat() + "Z"
    metrics = build_metrics(summary)
    adjustment_events = metrics["adjustment_events"]
    detection_success_rate = metrics["detection_success_rate"]
    defense_success_rate = metrics["defense_success_rate"]
    attack_types = attack_types_from_events(events)
    risk_score = compute_risk_score(adjustment_events, attack_types, defense_success_rate)
    status, severity = status_and_severity(adjustment_events, detection_success_rate, defense_success_rate, risk_score)
    links = affected_links_from_network_status(network_status)
    nodes = affected_nodes_from_links(links)
    actions = actions_from_plan(execution_plan)
    latest_event = latest_attack_event(events)
    attack_counts = Counter(str(event.get("attack_type") or "UNKNOWN") for event in events)
    scenario = "dynamic_defense_hybrid_actor_critic"
    if attack_types:
        scenario = "dynamic_defense_" + "_".join(attack_types[:3]).replace(" ", "_").replace("/", "_")

    message = (
        "Dynamic defense detected %d adjustment event(s); attacks=%s; risk_score=%d"
        % (adjustment_events, ",".join(attack_types) if attack_types else "none", risk_score)
    )
    summary_text = "动态防御检测到 %d 个策略调整事件，风险评分 %d，已触发动态防御响应。" % (
        adjustment_events,
        risk_score,
    )
    recommendation = (
        "Review generated controller execution plan and keep ActionExecutor in stateful/simulated mode until CENI controller validation is complete."
    )
    if severity == "critical":
        recommendation = "Prioritize affected links and validate rate-limit/isolation intents before enabling shell execution."

    alert = {
        "type": "dynamic_defense",
        "severity": severity,
        "status": status,
        "message": message,
        "risk_score": risk_score,
        "attack_types": attack_types,
        "latest_window_id": latest_event.get("window_id") if latest_event else None,
        "strategy_id": latest_event.get("strategy_id") if latest_event else None,
        "affected_links": links,
        "affected_nodes": nodes,
        "updated_at": updated_at,
    }

    metrics["event_summary"] = {
        "attack_types": attack_types,
        "attack_counts": dict(attack_counts),
        "latest_event": latest_event or {},
    }

    if controller_state:
        metrics["controller_current_model"] = controller_state.get("current_model")
        metrics["controller_thresholds"] = controller_state.get("thresholds", {})
        metrics["controller_execution_mode"] = controller_state.get("execution_mode")

    return {
        "status": status,
        "severity": severity,
        "updated_at": updated_at,
        "summary": summary_text,
        "message": message,
        "metrics": metrics,
        "alerts": [alert],
        "risk_score": risk_score,
        "scenario": scenario,
        "event_type": "dynamic_defense_status",
        "affected_links": links,
        "affected_nodes": nodes,
        "recommendation": recommendation,
        "actions": actions,
        "version": "1.0",
        "source": "dynamic_defense",
    }


def atomic_write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(str(tmp_path), str(path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export dynamic defense status to the CENI controller file interface.")
    parser.add_argument("--summary", default="reports/dynamic_defense_summary.json")
    parser.add_argument("--events", default="reports/dynamic_defense_events.csv")
    parser.add_argument("--controller-state", default="runtime/controller_state.json")
    parser.add_argument("--execution-plan", default="reports/controller_execution_plan.jsonl")
    parser.add_argument("--network-status", default="/tmp/optimize_multi_vm_runtime/defense_feeds/network_status.json")
    parser.add_argument("--out-json", default="/tmp/optimize_multi_vm_runtime/defense_inputs/dynamic_defense.json")
    args = parser.parse_args()

    summary_path = resolve_path(args.summary)
    if not summary_path.exists():
        raise FileNotFoundError("missing dynamic defense summary: %s" % summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    events = load_events(resolve_path(args.events))
    controller_state = load_json_optional(resolve_path(args.controller_state))
    execution_plan = load_jsonl(resolve_path(args.execution_plan))
    network_status = load_json_optional(resolve_path(args.network_status))
    payload = build_payload(summary, events, controller_state, execution_plan, network_status)
    out_path = resolve_path(args.out_json)
    atomic_write_json(out_path, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
