import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FIELDS = [
    "status",
    "severity",
    "updated_at",
    "summary",
    "message",
    "metrics",
    "alerts",
    "risk_score",
    "scenario",
    "event_type",
    "affected_links",
    "affected_nodes",
    "recommendation",
    "actions",
    "version",
    "source",
]


def test_export_ceni_dynamic_defense_status_generates_payload(tmp_path):
    summary_path = tmp_path / "dynamic_defense_summary.json"
    events_path = tmp_path / "dynamic_defense_events.csv"
    state_path = tmp_path / "controller_state.json"
    plan_path = tmp_path / "controller_execution_plan.jsonl"
    network_status_path = tmp_path / "network_status.json"
    out_path = tmp_path / "dynamic_defense.json"

    summary_path.write_text(
        json.dumps(
            {
                "detector": "hybrid",
                "optimizer": "actor_critic",
                "windows": 3,
                "adjustment_events": 2,
                "detection_success_rate": 1.0,
                "defense_success_rate": 1.0,
                "strategy_counts": {"s_ddos_vote_rate_limit": 2},
            }
        ),
        encoding="utf-8",
    )
    with events_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["window_id", "attack_type", "strategy_id", "adjustment_triggered", "detection_success", "defense_success"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "window_id": "0",
                "attack_type": "BENIGN",
                "strategy_id": "s_benign_monitor",
                "adjustment_triggered": "True",
                "detection_success": "True",
                "defense_success": "True",
            }
        )
        writer.writerow(
            {
                "window_id": "1",
                "attack_type": "DDoS",
                "strategy_id": "s_ddos_vote_rate_limit",
                "adjustment_triggered": "True",
                "detection_success": "True",
                "defense_success": "True",
            }
        )
    state_path.write_text(
        json.dumps({"current_model": "ensemble_lstm_ar_subspace", "thresholds": {"flow_rate": 0.85}, "execution_mode": "stateful"}),
        encoding="utf-8",
    )
    plan_records = [
        {"action_type": "switch_model"},
        {"action_type": "rate_limit"},
        {"action_type": "rate_limit"},
    ]
    plan_path.write_text("\n".join(json.dumps(record) for record in plan_records) + "\n", encoding="utf-8")
    network_status_path.write_text(json.dumps({"active_path": ["s1", "s3", "s4"]}), encoding="utf-8")

    subprocess.check_call(
        [
            sys.executable,
            "scripts/export_ceni_dynamic_defense_status.py",
            "--summary",
            str(summary_path),
            "--events",
            str(events_path),
            "--controller-state",
            str(state_path),
            "--execution-plan",
            str(plan_path),
            "--network-status",
            str(network_status_path),
            "--out-json",
            str(out_path),
        ],
        cwd=str(ROOT),
    )

    assert out_path.exists()
    assert not out_path.with_name(out_path.name + ".tmp").exists()
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    for field in REQUIRED_FIELDS:
        assert field in payload

    assert payload["status"] == "attack_detected"
    assert payload["severity"] in {"critical", "warning"}
    assert isinstance(payload["summary"], str)
    assert "策略调整事件" in payload["summary"]
    assert 0 <= payload["risk_score"] <= 100
    assert payload["metrics"]["detector"] == "hybrid"
    assert payload["metrics"]["optimizer"] == "actor_critic"
    assert payload["metrics"]["windows"] == 3
    assert payload["metrics"]["adjustment_events"] == 2
    assert payload["metrics"]["strategy_counts"] == {"s_ddos_vote_rate_limit": 2}
    assert payload["metrics"]["event_summary"]["attack_types"] == ["DDoS"]
    assert payload["metrics"]["event_summary"]["latest_event"]["attack_type"] == "DDoS"
    assert payload["affected_links"] == ["s1-s3", "s3-s4"]
    assert payload["affected_nodes"] == ["s1", "s3", "s4"]
    assert payload["actions"] == ["switch_model", "rate_limit"]
    assert isinstance(payload["alerts"], list)
    assert payload["alerts"]
    assert isinstance(payload["alerts"][0], dict)
    assert payload["alerts"][0]["affected_links"] == ["s1-s3", "s3-s4"]
    assert payload["alerts"][0]["affected_nodes"] == ["s1", "s3", "s4"]
    assert payload["source"] == "dynamic_defense"
