import json
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXTENSION_DIR = REPO_ROOT / "extensions" / "dynamic_defense_ceni"
BRIDGE = EXTENSION_DIR / "bridge.py"
VALIDATOR = EXTENSION_DIR / "validate_bridge_output.py"


def run_bridge(tmpdir: str) -> tuple[Path, Path]:
    root = Path(tmpdir)
    output_dir = root / "defense_inputs"
    log_file = root / "logs" / "dynamic_defense_ceni_actions.jsonl"

    subprocess.run(
        [
            sys.executable,
            str(BRIDGE),
            "--static-demo",
            "--once",
            "--output-dir",
            str(output_dir),
            "--output-file",
            "dynamic_defense.json",
            "--log-file",
            str(log_file),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return output_dir / "dynamic_defense.json", log_file


def test_static_demo_generates_dynamic_defense_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file, _ = run_bridge(tmpdir)

        payload = json.loads(output_file.read_text(encoding="utf-8"))

        assert payload["status"] == "attack_detected"
        assert payload["severity"] == "critical"
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
