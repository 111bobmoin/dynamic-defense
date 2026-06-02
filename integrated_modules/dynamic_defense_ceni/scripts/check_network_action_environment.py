#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


ROOT = Path(__file__).resolve().parents[1]
COMMON_BIN_DIRS = ["/usr/sbin", "/sbin", "/usr/bin", "/bin"]


def find_binary(name: str) -> Optional[str]:
    found = shutil.which(name)
    if found:
        return found
    for directory in COMMON_BIN_DIRS:
        candidate = Path(directory) / name
        if candidate.exists() and os.access(str(candidate), os.X_OK):
            return str(candidate)
    return None


def list_interfaces() -> List[str]:
    sys_net = Path("/sys/class/net")
    if sys_net.exists():
        try:
            return sorted(path.name for path in sys_net.iterdir())
        except OSError:
            pass
    try:
        return sorted(name for _, name in socket.if_nameindex())
    except OSError:
        return []


def check_sudo_available() -> Dict:
    sudo_path = find_binary("sudo")
    if not sudo_path:
        return {
            "available": False,
            "path": None,
            "status": "MISSING",
            "detail": "sudo binary not found",
        }
    try:
        completed = subprocess.run(
            [sudo_path, "-n", "true"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=2,
        )
    except Exception as exc:
        return {
            "available": False,
            "path": sudo_path,
            "status": "ERROR",
            "detail": repr(exc),
        }
    stderr = completed.stderr.decode("utf-8", "replace").strip()
    return {
        "available": completed.returncode == 0,
        "path": sudo_path,
        "status": "OK" if completed.returncode == 0 else "UNAVAILABLE",
        "detail": stderr,
        "returncode": completed.returncode,
    }


def check_port_available(port: int, host: str = "127.0.0.1") -> Dict:
    sock = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
    except OSError as exc:
        return {
            "available": False,
            "host": host,
            "port": port,
            "status": "UNAVAILABLE",
            "detail": repr(exc),
        }
    finally:
        if sock is not None:
            sock.close()
    return {
        "available": True,
        "host": host,
        "port": port,
        "status": "OK",
        "detail": "%s:%s can be bound" % (host, port),
    }


def recommend_executor(
    sudo_available: bool,
    has_tc: bool,
    has_iptables: bool,
    has_ovs_ofctl: bool,
    has_br0: bool,
    can_bind_18082: bool,
) -> str:
    shell_ready = sudo_available and has_tc and has_iptables and has_ovs_ofctl and has_br0 and can_bind_18082
    if shell_ready:
        return "shell"
    if can_bind_18082:
        return "stateful"
    return "simulated"


def build_summary(port: int = 18082) -> Dict:
    tc_path = find_binary("tc")
    iptables_path = find_binary("iptables")
    ovs_ofctl_path = find_binary("ovs-ofctl")
    interfaces = list_interfaces()
    sudo = check_sudo_available()
    bind = check_port_available(port)

    has_tc = tc_path is not None
    has_iptables = iptables_path is not None
    has_ovs_ofctl = ovs_ofctl_path is not None
    has_br0 = "br0" in interfaces
    sudo_available = bool(sudo["available"])
    can_bind_18082 = bool(bind["available"])

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "project_root": str(ROOT),
        "has_tc": has_tc,
        "has_iptables": has_iptables,
        "has_ovs_ofctl": has_ovs_ofctl,
        "interfaces": interfaces,
        "has_br0": has_br0,
        "can_bind_18082": can_bind_18082,
        "sudo_available": sudo_available,
        "recommended_executor": recommend_executor(
            sudo_available=sudo_available,
            has_tc=has_tc,
            has_iptables=has_iptables,
            has_ovs_ofctl=has_ovs_ofctl,
            has_br0=has_br0,
            can_bind_18082=can_bind_18082,
        ),
        "details": {
            "tc_path": tc_path,
            "iptables_path": iptables_path,
            "ovs_ofctl_path": ovs_ofctl_path,
            "sudo": sudo,
            "port_18082": bind,
            "safety": "read-only probe; no tc, iptables, or ovs-ofctl modification commands executed",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check network action execution prerequisites without changing networking.")
    parser.add_argument("--port", type=int, default=18082, help="Controller port to test for listen availability")
    parser.add_argument("--out-json", default="reports/network_action_environment.json")
    args = parser.parse_args()

    summary = build_summary(port=args.port)
    output_path = ROOT / args.out_json
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
