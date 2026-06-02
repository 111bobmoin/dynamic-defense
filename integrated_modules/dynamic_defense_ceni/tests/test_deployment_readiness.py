import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_check_deployment_readiness_outputs_summary(tmp_path):
    out_json = tmp_path / "deployment_readiness.json"
    subprocess.check_call(
        [
            sys.executable,
            "scripts/check_deployment_readiness.py",
            "--out-json",
            str(out_json),
        ],
        cwd=str(ROOT),
    )

    assert out_json.exists()
    summary = json.loads(out_json.read_text(encoding="utf-8"))
    assert "generated_at" in summary
    assert "project_root" in summary
    assert "overall_ready" in summary
    assert "checks" in summary
    assert "required_checks" in summary

    checks = summary["checks"]
    for key in [
        "python_version",
        "torch_import",
        "strategies_yaml",
        "torch_model",
        "torch_model_meta",
        "actor_critic_model",
        "actor_critic_meta",
        "attack_defender_detector_flag",
        "attack_defender_optimizer_flag",
        "controller_execution_mode_flag",
        "scenario_csv",
        "port_18082",
    ]:
        assert key in checks
        assert "ok" in checks[key]
        assert "status" in checks[key]
        assert "detail" in checks[key]
