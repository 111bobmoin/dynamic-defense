#!/usr/bin/env python3
"""Sidecar bridge for adapting dynamic_defense_ceni output."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from action_logger import log_action, utc_timestamp


VERSION = "v1.0.1-dynamic-defense-ceni"
SOURCE = "dynamic_defense_ceni"
DEFAULT_OUTPUT_DIR = "/tmp/optimize_multi_vm_runtime/defense_inputs"
DEFAULT_OUTPUT_FILE = "dynamic_defense.json"
DEFAULT_LOG_FILE = "logs/dynamic_defense_ceni_actions.jsonl"
STATIC_DEMO_MODE = {
    "data_mode": "static-demo",
    "data_mode_label": "演示数据 / static-demo",
    "data_mode_note": "当前页面展示的是 p4 最终验证结果的静态演示载荷，未代表 attack_defender.py 正在实时运行。",
}
REAL_REPORTS_MODE = {
    "data_mode": "real-reports",
    "data_mode_label": "真实运行结果 / real-reports",
    "data_mode_note": "当前页面展示的是从 dynamic_defense_ceni reports/runtime 读取并转换的真实运行结果。",
}
DEFAULT_MODEL_INFO = {
    "detector_model_name": "FlowMLP family_v3 策略族分类器",
    "detector_model_path": "models/torch_flow_classifier_expanded_family_v3.pt",
    "detector_meta_path": "models/torch_flow_classifier_expanded_family_v3_meta.json",
    "label_mode": "family",
    "detector_mode": "hybrid",
    "optimizer": "actor_critic",
    "optimizer_model_path": "models/actor_critic_policy.pt",
    "execution_mode": "REST/stateful 计划生成与状态更新",
    "version": VERSION,
}
DEFAULT_EXECUTION_RESULT = {
    "runtime_status": "发现攻击 / attack_detected",
    "controller_execution_mode": "stateful",
    "windows": 11,
    "adjustment_events": 11,
    "detection_success_rate": 1.0,
    "defense_success_rate": 1.0,
    "attack_family_accuracy": 1.0,
    "strategy_match_accuracy": 1.0,
    "execution_plan_lines": 30,
    "latest_strategy_id": "s_web_attack_strict",
    "latest_window_id": "10",
    "result_summary": "已完成 hybrid 检测、actor_critic 策略优化、REST/stateful 动作计划生成和 CENI JSON 输出。",
}
DEFAULT_STRATEGY_SWITCH_VISUALIZATION = {
    "pipeline": [
        {"name": "流量窗口输入", "status": "completed"},
        {"name": "攻击特征提取", "status": "completed"},
        {"name": "hybrid 检测源选择", "status": "completed"},
        {"name": "攻击族识别", "status": "completed"},
        {"name": "actor_critic 策略优化", "status": "completed"},
        {"name": "防御动作计划生成", "status": "completed"},
        {"name": "CENI dynamic_defense.json 输出", "status": "completed"},
    ],
    "detector_switch": {
        "mode": "hybrid",
        "active_detector": "FlowMLP family_v3",
        "fallback_detector": "template_fallback",
        "detector_source_counts": {"torch": 11},
    },
    "strategy_actions": [
        {"action": "monitor_only", "label": "监控保持", "status": "planned"},
        {"action": "log_enrich", "label": "日志增强", "status": "planned"},
        {"action": "switch_model", "label": "检测模型切换", "status": "planned"},
        {"action": "raise_threshold", "label": "阈值提升", "status": "planned"},
        {"action": "rate_limit", "label": "限速", "status": "planned"},
        {"action": "isolate_flow", "label": "流隔离", "status": "planned"},
    ],
    "strategy_counts": {
        "s_ddos_vote_rate_limit": 5,
        "s_bruteforce_ssh_ftp": 2,
        "s_web_attack_strict": 2,
        "s_benign_monitor": 1,
        "s_portscan_isolate": 1,
    },
}

REQUIRED_INPUTS = {
    "summary": Path("reports/dynamic_defense_summary.json"),
    "events": Path("reports/dynamic_defense_events.csv"),
    "controller_state": Path("runtime/controller_state.json"),
    "execution_plan": Path("reports/controller_execution_plan.jsonl"),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adapt dynamic_defense_ceni output.")
    parser.add_argument("--ceni-project-root", default=".", help="Root of the dynamic_defense_ceni project.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for bridge output JSON.")
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE, help="Output JSON file name.")
    parser.add_argument("--log-file", default=DEFAULT_LOG_FILE, help="JSONL action log path.")
    parser.add_argument("--static-demo", action="store_true", help="Emit the built-in P4 final validation payload.")
    parser.add_argument("--once", action="store_true", help="Run one bridge cycle and exit.")
    parser.add_argument("--watch-interval", type=float, default=5.0, help="Seconds between bridge cycles.")
    return parser.parse_args(argv)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {"value": data}


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError as exc:
                parsed = {"line_number": line_number, "raw": stripped, "parse_error": str(exc)}
            records.append(parsed if isinstance(parsed, dict) else {"value": parsed})
    return records


def read_ceni_outputs(project_root: Path) -> dict[str, Any]:
    paths = {name: project_root / relative_path for name, relative_path in REQUIRED_INPUTS.items()}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing CENI output files: " + ", ".join(missing))

    return {
        "summary": load_json(paths["summary"]),
        "events": load_csv(paths["events"]),
        "controller_state": load_json(paths["controller_state"]),
        "execution_plan": load_jsonl(paths["execution_plan"]),
        "paths": {name: str(path) for name, path in paths.items()},
    }


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _as_number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return default
    return default


def _as_int(value: Any, default: int) -> int:
    if value is None or isinstance(value, bool):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _risk_score(value: Any) -> int:
    return int(max(0, min(100, round(_as_number(value)))))


def _severity_from_risk(risk_score: int) -> str:
    if risk_score >= 75:
        return "critical"
    if risk_score >= 50:
        return "high"
    if risk_score >= 25:
        return "medium"
    return "low"


def _coerce_list(value: Any) -> list[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                return parsed
        return [item.strip() for item in stripped.split(",") if item.strip()]
    return [value]


def _unique_strings(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        for item in _coerce_list(value):
            text = str(item).strip()
            if text and text not in seen:
                result.append(text)
                seen.add(text)
    return result


def _dict_value(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return None


def _model_info_from(*sources: dict[str, Any]) -> dict[str, Any]:
    model_info = copy.deepcopy(DEFAULT_MODEL_INFO)
    for source in sources:
        candidate = source.get("model_info")
        if isinstance(candidate, dict):
            model_info.update(candidate)
    model_info["version"] = str(model_info.get("version") or VERSION)
    return model_info


def _status_label(status: str) -> str:
    if status == "attack_detected":
        return "发现攻击 / attack_detected"
    return status


def _execution_result_from(
    status: str,
    metrics: dict[str, Any],
    execution_plan: list[dict[str, Any]],
    *sources: dict[str, Any],
) -> dict[str, Any]:
    result = copy.deepcopy(DEFAULT_EXECUTION_RESULT)
    for source in sources:
        candidate = source.get("execution_result")
        if isinstance(candidate, dict):
            result.update(candidate)

    attack_type_accuracy = metrics.get("attack_type_accuracy")
    if not isinstance(attack_type_accuracy, dict):
        attack_type_accuracy = {}

    latest_plan = execution_plan[-1] if execution_plan else {}
    result.update(
        {
            "runtime_status": _status_label(status),
            "windows": _as_int(metrics.get("windows"), result["windows"]),
            "adjustment_events": _as_int(metrics.get("adjustment_events"), result["adjustment_events"]),
            "attack_family_accuracy": _as_number(attack_type_accuracy.get("family"), result["attack_family_accuracy"]),
            "strategy_match_accuracy": _as_number(metrics.get("strategy_match_accuracy"), result["strategy_match_accuracy"]),
            "execution_plan_lines": len(execution_plan) if execution_plan else result["execution_plan_lines"],
        }
    )

    latest_strategy_id = _first_present(
        _dict_value(latest_plan, "strategy_id", "strategy", "latest_strategy_id"),
        result.get("latest_strategy_id"),
    )
    latest_window_id = _first_present(
        _dict_value(latest_plan, "window_id", "window", "latest_window_id"),
        result.get("latest_window_id"),
    )
    result["latest_strategy_id"] = str(latest_strategy_id)
    result["latest_window_id"] = str(latest_window_id)
    return result


def _strategy_switch_visualization_from(metrics: dict[str, Any], *sources: dict[str, Any]) -> dict[str, Any]:
    visualization = copy.deepcopy(DEFAULT_STRATEGY_SWITCH_VISUALIZATION)
    for source in sources:
        candidate = source.get("strategy_switch_visualization")
        if isinstance(candidate, dict):
            visualization.update(candidate)

    detector_switch = visualization.get("detector_switch")
    if not isinstance(detector_switch, dict):
        detector_switch = {}
    detector_switch.setdefault("mode", metrics.get("detector") or "hybrid")
    detector_switch.setdefault("active_detector", "FlowMLP family_v3")
    detector_switch.setdefault("fallback_detector", "template_fallback")
    detector_switch["detector_source_counts"] = metrics.get("detector_source_counts") or detector_switch.get(
        "detector_source_counts",
        {"torch": 11},
    )
    visualization["detector_switch"] = detector_switch
    return visualization


def _event_alert(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "timestamp": _dict_value(row, "timestamp", "time", "updated_at"),
        "event_type": _dict_value(row, "event_type", "type", "attack_type") or "dynamic_defense_event",
        "severity": _dict_value(row, "severity", "level") or "medium",
        "message": _dict_value(row, "message", "description", "summary") or "CENI event detected.",
    }


def _derive_links(events: list[dict[str, Any]]) -> list[str]:
    links: list[Any] = []
    for row in events:
        direct = _dict_value(row, "affected_links", "affected_link", "link", "links")
        if direct:
            links.append(direct)
            continue
        src = _dict_value(row, "src", "source", "source_node")
        dst = _dict_value(row, "dst", "target", "target_node")
        if src and dst:
            links.append(f"{src}-{dst}")
    return _unique_strings(links)


def _derive_nodes(events: list[dict[str, Any]]) -> list[str]:
    nodes: list[Any] = []
    for row in events:
        nodes.append(_dict_value(row, "affected_nodes", "affected_node", "node", "nodes"))
        nodes.append(_dict_value(row, "src", "source", "source_node"))
        nodes.append(_dict_value(row, "dst", "target", "target_node"))
    return _unique_strings(nodes)


def _derive_actions(plan: list[dict[str, Any]]) -> list[str]:
    actions: list[Any] = []
    for record in plan:
        actions.append(_dict_value(record, "action", "strategy", "command", "name", "mitigation"))
    return _unique_strings(actions)


def build_static_demo_payload() -> dict[str, Any]:
    return {
        "status": "attack_detected",
        "severity": "critical",
        **STATIC_DEMO_MODE,
        "updated_at": utc_timestamp(),
        "summary": "P4 final validation payload generated by the dynamic_defense_ceni bridge.",
        "message": "Hybrid CENI detection selected actor-critic defense actions for affected P4 links.",
        "metrics": {
            "detector": "hybrid",
            "optimizer": "actor_critic",
            "windows": 11,
            "adjustment_events": 11,
            "attack_type_accuracy": {"family": 1.0},
            "strategy_match_accuracy": 1.0,
            "detector_source_counts": {"torch": 11},
        },
        "alerts": [
            {
                "event_type": "p4_final_validation",
                "severity": "critical",
                "message": "Attack detected on s3-s4 and s4-s7.",
            }
        ],
        "risk_score": 75,
        "scenario": "p4_final_validation",
        "event_type": "dynamic_defense_attack",
        "affected_links": ["s3-s4", "s4-s7"],
        "affected_nodes": ["s3", "s4", "s7"],
        "recommendation": "Apply staged mitigation and continue enriched monitoring.",
        "actions": [
            "monitor_only",
            "log_enrich",
            "switch_model",
            "raise_threshold",
            "rate_limit",
            "isolate_flow",
        ],
        "model_info": copy.deepcopy(DEFAULT_MODEL_INFO),
        "execution_result": copy.deepcopy(DEFAULT_EXECUTION_RESULT),
        "strategy_switch_visualization": copy.deepcopy(DEFAULT_STRATEGY_SWITCH_VISUALIZATION),
        "version": VERSION,
        "source": SOURCE,
    }


def transform_ceni_outputs(data: dict[str, Any]) -> dict[str, Any]:
    summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
    controller_state = data.get("controller_state") if isinstance(data.get("controller_state"), dict) else {}
    events = data.get("events") if isinstance(data.get("events"), list) else []
    execution_plan = data.get("execution_plan") if isinstance(data.get("execution_plan"), list) else []
    latest_event = events[-1] if events and isinstance(events[-1], dict) else {}

    risk = _risk_score(
        _first_present(
            _dict_value(summary, "risk_score", "risk", "score"),
            _dict_value(controller_state, "risk_score", "risk", "score"),
            _dict_value(latest_event, "risk_score", "risk", "score"),
        )
    )
    severity = str(
        _first_present(
            _dict_value(summary, "severity"),
            _dict_value(controller_state, "severity"),
            _dict_value(latest_event, "severity"),
            _severity_from_risk(risk),
        )
    )
    status = str(
        _first_present(
            _dict_value(summary, "status"),
            _dict_value(controller_state, "status"),
            "attack_detected" if risk >= 50 else "normal",
        )
    )

    metrics: dict[str, Any] = {}
    for candidate in (_dict_value(summary, "metrics"), _dict_value(controller_state, "metrics")):
        if isinstance(candidate, dict):
            metrics.update(candidate)
    metrics.setdefault("event_count", len(events))
    metrics.setdefault("execution_plan_count", len(execution_plan))

    for key in (
        "detector",
        "optimizer",
        "windows",
        "adjustment_events",
        "attack_type_accuracy",
        "strategy_match_accuracy",
        "detector_source_counts",
    ):
        value = _first_present(_dict_value(summary, key), _dict_value(controller_state, key))
        if value is not None:
            metrics[key] = value

    alerts = _coerce_list(_first_present(_dict_value(summary, "alerts"), _dict_value(controller_state, "alerts")))
    if not alerts:
        alerts = [_event_alert(row) for row in events[-10:] if isinstance(row, dict)]

    affected_links = _unique_strings(
        [
            _first_present(_dict_value(summary, "affected_links"), _dict_value(controller_state, "affected_links")),
            _derive_links([row for row in events if isinstance(row, dict)]),
        ]
    )
    affected_nodes = _unique_strings(
        [
            _first_present(_dict_value(summary, "affected_nodes"), _dict_value(controller_state, "affected_nodes")),
            _derive_nodes([row for row in events if isinstance(row, dict)]),
        ]
    )

    actions = _unique_strings(
        [
            _first_present(_dict_value(summary, "actions"), _dict_value(controller_state, "actions")),
            _derive_actions([row for row in execution_plan if isinstance(row, dict)]),
        ]
    )

    return {
        "status": status,
        "severity": severity,
        **REAL_REPORTS_MODE,
        "updated_at": str(_first_present(_dict_value(summary, "updated_at"), utc_timestamp())),
        "summary": str(_first_present(_dict_value(summary, "summary"), "CENI dynamic defense output converted.")),
        "message": str(_first_present(_dict_value(summary, "message"), "Converted dynamic_defense_ceni results.")),
        "metrics": metrics,
        "alerts": alerts,
        "risk_score": risk,
        "scenario": str(_first_present(_dict_value(summary, "scenario"), _dict_value(controller_state, "scenario"), "dynamic_defense_ceni")),
        "event_type": str(_first_present(_dict_value(summary, "event_type"), _dict_value(latest_event, "event_type"), "dynamic_defense_event")),
        "affected_links": affected_links,
        "affected_nodes": affected_nodes,
        "recommendation": str(
            _first_present(
                _dict_value(summary, "recommendation"),
                _dict_value(controller_state, "recommendation"),
                "Review mitigation plan and keep monitoring active.",
            )
        ),
        "actions": actions,
        "model_info": _model_info_from(summary, controller_state),
        "execution_result": _execution_result_from(status, metrics, [row for row in execution_plan if isinstance(row, dict)], summary, controller_state),
        "strategy_switch_visualization": _strategy_switch_visualization_from(metrics, summary, controller_state),
        "version": str(_first_present(_dict_value(summary, "version"), VERSION)),
        "source": SOURCE,
    }


def atomic_write_json(payload: dict[str, Any], output_dir: str | Path, output_file: str | Path) -> Path:
    output_file_path = Path(output_file)
    target = output_file_path if output_file_path.is_absolute() else Path(output_dir) / output_file_path
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(target.name + ".tmp")

    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())

    os.replace(tmp_path, target)
    return target


def run_once(args: argparse.Namespace) -> Path:
    if args.static_demo:
        log_action(
            args.log_file,
            "read",
            "success",
            "Using static demo payload; CENI report files are not required.",
            inputs={"static_demo": True},
        )
        payload = build_static_demo_payload()
    else:
        project_root = Path(args.ceni_project_root)
        try:
            source_data = read_ceni_outputs(project_root)
        except Exception as exc:
            log_action(
                args.log_file,
                "read",
                "failure",
                "Failed to read CENI output files.",
                inputs={"ceni_project_root": str(project_root)},
                details={"error": str(exc)},
            )
            raise
        log_action(
            args.log_file,
            "read",
            "success",
            "Read CENI output files.",
            inputs={"ceni_project_root": str(project_root)},
            outputs={"paths": source_data.get("paths", {})},
        )
        try:
            payload = transform_ceni_outputs(source_data)
        except Exception as exc:
            log_action(
                args.log_file,
                "transform",
                "failure",
                "Failed to transform CENI outputs.",
                details={"error": str(exc)},
            )
            raise

    log_action(
        args.log_file,
        "transform",
        "success",
        "Converted CENI data to platform payload.",
        outputs={"source": payload.get("source"), "version": payload.get("version")},
    )

    try:
        target = atomic_write_json(payload, args.output_dir, args.output_file)
    except Exception as exc:
        log_action(
            args.log_file,
            "write",
            "failure",
            "Failed to write bridge output.",
            inputs={"output_dir": args.output_dir, "output_file": args.output_file},
            details={"error": str(exc)},
        )
        raise

    log_action(
        args.log_file,
        "write",
        "success",
        "Wrote bridge output atomically.",
        inputs={"output_dir": args.output_dir, "output_file": args.output_file},
        outputs={"path": str(target)},
    )
    return target


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    log_action(
        args.log_file,
        "startup",
        "success",
        "dynamic_defense_ceni bridge starting.",
        inputs=vars(args),
    )

    exit_status = "success"
    exit_message = "dynamic_defense_ceni bridge exited."
    exit_details: dict[str, Any] = {}

    try:
        while True:
            run_once(args)
            if args.once:
                break
            time.sleep(max(0.1, args.watch_interval))
    except KeyboardInterrupt:
        exit_status = "interrupted"
        exit_message = "dynamic_defense_ceni bridge interrupted."
        return_code = 130
    except Exception as exc:
        exit_status = "failure"
        exit_message = "dynamic_defense_ceni bridge failed."
        exit_details = {"error": str(exc)}
        log_action(args.log_file, "failure", "failure", exit_message, details=exit_details)
        return_code = 1
    else:
        return_code = 0
    finally:
        log_action(args.log_file, "exit", exit_status, exit_message, details=exit_details)

    return return_code


if __name__ == "__main__":
    sys.exit(main())
