#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import socket
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


ROOT = Path(__file__).resolve().parents[1]


def _check(ok: bool, status: str, detail: str = "", **extra) -> Dict:
    item = {"ok": bool(ok), "status": status, "detail": detail}
    item.update(extra)
    return item


def check_python_version(min_major: int = 3, min_minor: int = 7) -> Dict:
    version = {
        "major": sys.version_info.major,
        "minor": sys.version_info.minor,
        "micro": sys.version_info.micro,
        "executable": sys.executable,
    }
    ok = (sys.version_info.major, sys.version_info.minor) >= (min_major, min_minor)
    detail = "Python %d.%d.%d" % (sys.version_info.major, sys.version_info.minor, sys.version_info.micro)
    return _check(ok, "OK" if ok else "FAIL", detail, version=version, minimum="%d.%d" % (min_major, min_minor))


def check_import(module_name: str) -> Dict:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return _check(False, "MISSING", repr(exc), module=module_name)
    version = getattr(module, "__version__", None)
    return _check(True, "OK", "%s importable" % module_name, module=module_name, version=version)


def check_file(path: str) -> Dict:
    file_path = ROOT / path
    exists = file_path.exists()
    size = file_path.stat().st_size if exists and file_path.is_file() else None
    return _check(exists, "OK" if exists else "MISSING", path, path=path, absolute_path=str(file_path), size_bytes=size)


def check_source_flag(script_path: str, flag: str) -> Dict:
    file_path = ROOT / script_path
    if not file_path.exists():
        return _check(False, "MISSING_SCRIPT", "%s not found" % script_path, path=script_path, flag=flag)
    text = file_path.read_text(encoding="utf-8")
    ok = flag in text
    return _check(ok, "OK" if ok else "MISSING_FLAG", "%s contains %s" % (script_path, flag), path=script_path, flag=flag)


def check_port_available(port: int, host: str = "127.0.0.1") -> Dict:
    sock: Optional[socket.socket] = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
    except OSError as exc:
        return _check(False, "UNAVAILABLE", repr(exc), host=host, port=port)
    finally:
        if sock is not None:
            sock.close()
    return _check(True, "OK", "%s:%s is available" % (host, port), host=host, port=port)


def build_summary(port: int = 18082) -> Dict:
    checks = {
        "python_version": check_python_version(),
        "torch_import": check_import("torch"),
        "strategies_yaml": check_file("configs/strategies.yaml"),
        "torch_model": check_file("models/torch_flow_classifier.pt"),
        "torch_model_meta": check_file("models/torch_flow_classifier_meta.json"),
        "actor_critic_model": check_file("models/actor_critic_policy.pt"),
        "actor_critic_meta": check_file("models/actor_critic_policy_meta.json"),
        "attack_defender_detector_flag": check_source_flag("attack_defender.py", "--detector"),
        "attack_defender_optimizer_flag": check_source_flag("attack_defender.py", "--optimizer"),
        "controller_execution_mode_flag": check_source_flag("scripts/translating_defense_controller.py", "--execution-mode"),
        "scenario_csv": check_file("data/cicids2017_subset/cicids2017_scenario_ordered.csv"),
        "port_18082": check_port_available(port),
    }
    required = list(checks.keys())
    ready = all(checks[name]["ok"] for name in required)
    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "project_root": str(ROOT),
        "overall_ready": ready,
        "checks": checks,
        "required_checks": required,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check dynamic_defense deployment readiness for Ubuntu/CENI VM.")
    parser.add_argument("--port", type=int, default=18082, help="Controller port to check for availability")
    parser.add_argument("--out-json", default="reports/deployment_readiness.json")
    args = parser.parse_args()

    summary = build_summary(port=args.port)
    output_path = ROOT / args.out_json
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
