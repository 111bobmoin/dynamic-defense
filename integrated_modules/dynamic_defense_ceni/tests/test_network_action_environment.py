import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_check_network_action_environment_outputs_json(tmp_path):
    out_json = tmp_path / "network_action_environment.json"
    subprocess.check_call(
        [
            sys.executable,
            "scripts/check_network_action_environment.py",
            "--out-json",
            str(out_json),
        ],
        cwd=str(ROOT),
    )

    assert out_json.exists()
    summary = json.loads(out_json.read_text(encoding="utf-8"))
    for field in [
        "has_tc",
        "has_iptables",
        "has_ovs_ofctl",
        "interfaces",
        "has_br0",
        "can_bind_18082",
        "sudo_available",
        "recommended_executor",
    ]:
        assert field in summary

    assert isinstance(summary["has_tc"], bool)
    assert isinstance(summary["has_iptables"], bool)
    assert isinstance(summary["has_ovs_ofctl"], bool)
    assert isinstance(summary["interfaces"], list)
    assert isinstance(summary["has_br0"], bool)
    assert isinstance(summary["can_bind_18082"], bool)
    assert isinstance(summary["sudo_available"], bool)
    assert summary["recommended_executor"] in {"simulated", "stateful", "shell"}
    assert "details" in summary
    assert "safety" in summary["details"]
