#!/usr/bin/env python3
"""Sidecar bridge for adapting dynamic_defense_ceni output."""

from __future__ import annotations

import argparse
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
