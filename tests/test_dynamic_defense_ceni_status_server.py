import json
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXTENSION_DIR = REPO_ROOT / "extensions" / "dynamic_defense_ceni"
STATIC_STATUS_HTML = EXTENSION_DIR / "static_status.html"
sys.path.insert(0, str(EXTENSION_DIR))

import status_server  # noqa: E402


def sample_payload() -> dict:
    return {
        "status": "attack_detected",
        "severity": "critical",
        "risk_score": 75,
        "source": "dynamic_defense_ceni",
        "version": "v1.0.1-dynamic-defense-ceni",
        "affected_links": ["s3-s4", "s4-s7"],
        "affected_nodes": ["s3", "s4", "s7"],
        "actions": ["monitor_only", "log_enrich"],
        "metrics": {
            "detector": "hybrid",
            "optimizer": "actor_critic",
            "windows": 11,
            "adjustment_events": 11,
            "attack_type_accuracy": {"family": 1.0},
            "strategy_match_accuracy": 1.0,
            "detector_source_counts": {"torch": 11},
        },
    }


def call_api(route: str, input_path: Path, log_file: Path):
    status_code, payload = status_server.handle_api_request(route, input_path, log_file)
    return int(status_code), payload


def write_payload(path: Path) -> None:
    path.write_text(json.dumps(sample_payload()), encoding="utf-8")


def test_api_health():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        input_path = root / "dynamic_defense.json"
        write_payload(input_path)
        log_file = root / "actions.jsonl"
        status_code, payload = call_api("/api/health", input_path, log_file)

    assert status_code == 200
    assert payload == {"status": "ok", "service": "dynamic_defense_ceni_status_server"}


def test_api_status_reads_temp_dynamic_defense_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        input_path = root / "dynamic_defense.json"
        write_payload(input_path)
        log_file = root / "actions.jsonl"
        status_code, payload = call_api("/api/status", input_path, log_file)

    assert status_code == 200
    assert payload["status"] == "attack_detected"
    assert payload["risk_score"] == 75
    assert payload["source"] == "dynamic_defense_ceni"
    assert payload["affected_links"] == ["s3-s4", "s4-s7"]
    assert payload["detector"] == "hybrid"
    assert payload["optimizer"] == "actor_critic"
    assert payload["attack_type_accuracy_family"] == 1.0
    assert payload["detector_source_counts"] == {"torch": 11}
    assert payload["model_info"] == {
        "detector_model_name": "FlowMLP family_v3 策略族分类器",
        "detector_model_path": "models/torch_flow_classifier_expanded_family_v3.pt",
        "detector_meta_path": "models/torch_flow_classifier_expanded_family_v3_meta.json",
        "label_mode": "family",
        "detector_mode": "hybrid",
        "optimizer": "actor_critic",
        "optimizer_model_path": "models/actor_critic_policy.pt",
        "execution_mode": "REST/stateful 计划生成与状态更新",
        "version": "v1.0.1-dynamic-defense-ceni",
    }
    assert payload["execution_result"]["runtime_status"] == "发现攻击 / attack_detected"
    assert payload["execution_result"]["controller_execution_mode"] == "stateful"
    assert payload["execution_result"]["windows"] == 11
    assert payload["execution_result"]["adjustment_events"] == 11
    assert payload["execution_result"]["attack_family_accuracy"] == 1.0
    assert payload["execution_result"]["strategy_match_accuracy"] == 1.0
    assert payload["strategy_switch_visualization"]["pipeline"][4] == {
        "name": "actor_critic 策略优化",
        "status": "completed",
    }
    assert payload["strategy_switch_visualization"]["detector_switch"]["active_detector"] == "FlowMLP family_v3"
    assert payload["strategy_switch_visualization"]["strategy_actions"][2]["action"] == "switch_model"


def test_api_status_returns_error_json_when_input_missing():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        input_path = root / "missing_dynamic_defense.json"
        log_file = root / "actions.jsonl"
        status_code, payload = call_api("/api/status", input_path, log_file)

    assert status_code == 404
    assert payload["status"] == "error"
    assert payload["service"] == "dynamic_defense_ceni_status_server"
    assert payload["error"] == "input_not_found"
    assert str(input_path) == payload["input"]


def test_api_logs_returns_list():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        input_path = root / "dynamic_defense.json"
        write_payload(input_path)
        log_file = root / "actions.jsonl"
        status_server.safe_log_action(log_file, "startup", "success", "test startup")
        status_code, payload = call_api("/api/logs", input_path, log_file)

    assert status_code == 200
    assert isinstance(payload, list)
    assert any(entry.get("action") == "startup" for entry in payload)
    assert any(entry.get("action") == "request" for entry in payload)


def test_static_status_html_contains_model_panel_title():
    html = STATIC_STATUS_HTML.read_text(encoding="utf-8")

    assert "<h2>使用模型</h2>" in html
    assert "动态防御指标" in html
    assert "最近动作日志" in html
    assert "发现攻击 / attack_detected" in html
    assert "严重 / critical" in html
    assert "动态防御执行结果" in html
    assert "体系策略切换可视化" in html
    assert "检测模型切换动作计划" in html
    assert "REST/stateful 动作计划生成与状态更新" in html
