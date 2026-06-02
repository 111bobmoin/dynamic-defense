import json
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXTENSION_DIR = REPO_ROOT / "extensions" / "dynamic_defense_ceni"
BRIDGE = EXTENSION_DIR / "bridge.py"
VALIDATOR = EXTENSION_DIR / "validate_bridge_output.py"


def run_bridge(tmpdir: str, *, static_demo: bool = True, ceni_project_root: Path | None = None) -> tuple[Path, Path]:
    root = Path(tmpdir)
    output_dir = root / "defense_inputs"
    log_file = root / "logs" / "dynamic_defense_ceni_actions.jsonl"
    command = [
        sys.executable,
        str(BRIDGE),
        "--once",
        "--output-dir",
        str(output_dir),
        "--output-file",
        "dynamic_defense.json",
        "--log-file",
        str(log_file),
    ]
    if static_demo:
        command.append("--static-demo")
    if ceni_project_root is not None:
        command.extend(["--ceni-project-root", str(ceni_project_root)])

    subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return output_dir / "dynamic_defense.json", log_file


def write_fake_ceni_project(root: Path) -> None:
    reports = root / "reports"
    runtime = root / "runtime"
    reports.mkdir(parents=True)
    runtime.mkdir(parents=True)

    summary = {
        "detector": "hybrid",
        "optimizer": "actor_critic",
        "windows": 11,
        "adjustment_events": 11,
        "detection_success_rate": 1.0,
        "defense_success_rate": 1.0,
        "strategy_counts": {
            "s_ddos_vote_rate_limit": 5,
            "s_bruteforce_ssh_ftp": 2,
            "s_web_attack_strict": 2,
            "s_benign_monitor": 1,
            "s_portscan_isolate": 1,
        },
        "attack_type_accuracy": {"family": 1.0},
        "strategy_match_accuracy": {"accuracy": 1.0},
        "detector_source_counts": {"torch": 11},
    }
    (reports / "dynamic_defense_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (reports / "dynamic_defense_events.csv").write_text(
        "timestamp,event_type,severity,message,src,dst\n"
        "2026-05-29T00:00:00Z,dynamic_defense_event,critical,attack detected,s3,s4\n",
        encoding="utf-8",
    )
    (runtime / "controller_state.json").write_text(
        json.dumps({"controller_execution_mode": "stateful"}),
        encoding="utf-8",
    )
    plan_lines = [
        json.dumps({"window_id": str(index), "strategy_id": "s_web_attack_strict", "action": "rate_limit"})
        for index in range(30)
    ]
    (reports / "controller_execution_plan.jsonl").write_text("\n".join(plan_lines) + "\n", encoding="utf-8")


def test_static_demo_generates_dynamic_defense_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file, _ = run_bridge(tmpdir)

        payload = json.loads(output_file.read_text(encoding="utf-8"))

        assert payload["status"] == "attack_detected"
        assert payload["severity"] == "critical"
        assert payload["data_mode"] == "static-demo"
        assert payload["data_mode_label"] == "演示数据 / static-demo"
        assert "未代表 attack_defender.py 正在实时运行" in payload["data_mode_note"]
        assert payload["risk_score"] == 75
        assert payload["source"] == "dynamic_defense_ceni"
        assert payload["version"] == "v1.0.1-dynamic-defense-ceni"
        assert payload["metrics"]["detector"] == "hybrid"
        assert payload["metrics"]["optimizer"] == "actor_critic"
        assert payload["metrics"]["windows"] == 11
        assert payload["metrics"]["adjustment_events"] == 11
        assert payload["metrics"]["attack_type_accuracy"]["family"] == 1.0
        assert payload["metrics"]["strategy_match_accuracy"] == 1.0
        assert payload["metrics"]["detector_source_counts"] == {"torch": 11}
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
        assert payload["execution_result"]["detection_success_rate"] == 1.0
        assert payload["execution_result"]["defense_success_rate"] == 1.0
        assert payload["execution_result"]["attack_family_accuracy"] == 1.0
        assert payload["execution_result"]["strategy_match_accuracy"] == 1.0
        assert payload["execution_result"]["execution_plan_lines"] == 30
        assert payload["execution_result"]["latest_strategy_id"] == "s_web_attack_strict"
        assert payload["execution_result"]["latest_window_id"] == "10"
        assert "REST/stateful 动作计划生成" in payload["execution_result"]["result_summary"]
        assert payload["strategy_switch_visualization"]["pipeline"][2] == {
            "name": "hybrid 检测源选择",
            "status": "completed",
        }
        assert payload["strategy_switch_visualization"]["detector_switch"] == {
            "mode": "hybrid",
            "active_detector": "FlowMLP family_v3",
            "fallback_detector": "template_fallback",
            "detector_source_counts": {"torch": 11},
        }
        assert payload["strategy_switch_visualization"]["strategy_actions"][2] == {
            "action": "switch_model",
            "label": "检测模型切换",
            "status": "planned",
        }
        assert payload["strategy_switch_visualization"]["strategy_counts"]["s_web_attack_strict"] == 2
        assert payload["affected_links"] == ["s3-s4", "s4-s7"]
        assert payload["affected_nodes"] == ["s3", "s4", "s7"]
        assert payload["actions"] == [
            "monitor_only",
            "log_enrich",
            "switch_model",
            "raise_threshold",
            "rate_limit",
            "isolate_flow",
        ]


def test_real_reports_maps_adjustment_events_to_attack_status():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        ceni_project = root / "dynamic_defense_ceni"
        write_fake_ceni_project(ceni_project)
        output_file, _ = run_bridge(tmpdir, static_demo=False, ceni_project_root=ceni_project)

        payload = json.loads(output_file.read_text(encoding="utf-8"))

        assert payload["data_mode"] == "real-reports"
        assert payload["data_mode_label"] == "真实运行结果 / real-reports"
        assert payload["status"] == "attack_detected"
        assert payload["severity"] == "critical"
        assert payload["risk_score"] == 75
        assert payload["summary"] == "动态防御检测到 11 个策略调整事件，风险评分 75，已触发动态防御响应。"
        assert payload["message"] == "Dynamic defense detected 11 adjustment event(s); risk_score=75"
        assert payload["metrics"]["adjustment_events"] == 11
        assert payload["metrics"]["execution_plan_lines"] == 30
        assert payload["metrics"]["detector"] == "hybrid"
        assert payload["metrics"]["optimizer"] == "actor_critic"
        assert payload["metrics"]["windows"] == 11
        assert payload["metrics"]["detection_success_rate"] == 1.0
        assert payload["metrics"]["defense_success_rate"] == 1.0
        assert payload["metrics"]["strategy_counts"]["s_ddos_vote_rate_limit"] == 5
        assert payload["metrics"]["attack_type_accuracy"]["family"] == 1.0
        assert payload["metrics"]["strategy_match_accuracy"] == 1.0
        assert payload["metrics"]["detector_source_counts"] == {"torch": 11}
        assert payload["metrics"]["controller_execution_mode"] == "stateful"
        assert payload["execution_result"]["runtime_status"] == "发现攻击 / attack_detected"
        assert payload["execution_result"]["controller_execution_mode"] == "stateful"
        assert payload["execution_result"]["windows"] == 11
        assert payload["execution_result"]["adjustment_events"] == 11
        assert payload["execution_result"]["detection_success_rate"] == 1.0
        assert payload["execution_result"]["defense_success_rate"] == 1.0
        assert payload["execution_result"]["attack_family_accuracy"] == 1.0
        assert payload["execution_result"]["strategy_match_accuracy"] == 1.0
        assert payload["execution_result"]["execution_plan_lines"] == 30
        assert "真实 reports/runtime 转换" in payload["execution_result"]["result_summary"]
        assert payload["strategy_switch_visualization"]["detector_switch"]["detector_source_counts"] == {"torch": 11}


def test_action_log_generates_jsonl():
    with tempfile.TemporaryDirectory() as tmpdir:
        _, log_file = run_bridge(tmpdir)

        lines = log_file.read_text(encoding="utf-8").strip().splitlines()
        entries = [json.loads(line) for line in lines]
        actions = {entry["action"] for entry in entries}

        assert {"startup", "read", "transform", "write", "exit"}.issubset(actions)
        for entry in entries:
            assert set(entry) == {"timestamp", "action", "status", "message", "inputs", "outputs", "details"}


def test_validate_bridge_output_passes():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file, log_file = run_bridge(tmpdir)

        result = subprocess.run(
            [
                sys.executable,
                str(VALIDATOR),
                "--input",
                str(output_file),
                "--log-file",
                str(log_file),
            ],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        assert result.stdout.splitlines()[0] == "PASS"
        entries = [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines()]
        assert entries[-1]["action"] == "validate"
        assert entries[-1]["status"] == "success"


def test_atomic_write_target_json_is_parseable():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file, _ = run_bridge(tmpdir)

        assert output_file.exists()
        assert not output_file.with_name(output_file.name + ".tmp").exists()
        with output_file.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        assert isinstance(payload, dict)
