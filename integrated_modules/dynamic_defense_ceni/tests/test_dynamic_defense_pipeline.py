import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.dynamic_defense.policy_store import DefensePolicy, PolicyStore, load_policies_from_yaml


ROOT = Path(__file__).resolve().parents[1]


def run_cmd(args):
    subprocess.check_call([sys.executable] + args, cwd=str(ROOT))


def test_strategy_loader_outputs_metadata():
    run_cmd([
        "strategy_loader.py",
        "--config", "configs/strategies.yaml",
        "--db", "data/policies.sqlite",
    ])

    output = ROOT / "reports" / "strategy_metadata.json"
    assert output.exists()

    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["loaded"] >= 3

    first = data["policies"][0]
    assert "strategy_id" in first
    assert "model_type" in first
    assert "last_updated_at" in first


def test_feature_analyzer_outputs_matches():
    run_cmd([
        "scripts/make_sample_cicids.py",
        "--out", "data/sample_cicids.csv",
    ])

    run_cmd([
        "feature_analyzer.py",
        "--input", "data/sample_cicids.csv",
        "--build-templates",
        "--limit", "300",
    ])

    summary_file = ROOT / "reports" / "feature_match_summary.json"
    report_file = ROOT / "reports" / "feature_match_report.csv"

    assert summary_file.exists()
    assert report_file.exists()

    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    assert summary["rows"] == 300
    assert len(summary["matched_attack_counts"]) >= 1


def test_attack_defender_outputs_adjustment_events():
    run_cmd([
        "scripts/make_sample_cicids.py",
        "--out", "data/sample_cicids.csv",
    ])

    run_cmd([
        "attack_defender.py",
        "--input", "data/sample_cicids.csv",
        "--build-templates",
        "--window-size", "100",
        "--limit", "700",
    ])

    summary_file = ROOT / "reports" / "dynamic_defense_summary.json"
    events_file = ROOT / "reports" / "dynamic_defense_events.csv"

    assert summary_file.exists()
    assert events_file.exists()

    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    assert summary["windows"] >= 1
    assert summary["adjustment_events"] >= 1
    assert summary["defense_success_rate"] >= 0.8


def test_actor_critic_optimizer_select_observe_and_persist(tmp_path):
    pytest.importorskip("torch")
    from src.dynamic_defense.ac_optimizer import TorchActorCriticOptimizer

    store = PolicyStore(str(tmp_path / "policies.sqlite"))
    store.upsert_many(load_policies_from_yaml(str(ROOT / "configs" / "strategies.yaml")))
    optimizer = TorchActorCriticOptimizer(store=store, lr=0.001, gamma=0.95)
    context = {
        "attack_type": "DDoS",
        "detector_source": "template",
        "avg_match_score": 0.9,
        "template_score": 0.9,
        "torch_confidence": 0.0,
        "attack_present_by_label": True,
    }

    policy = optimizer.select("DDoS", state_context=context)
    assert isinstance(policy, DefensePolicy)

    optimizer.observe(policy.strategy_id, reward=1.0, success=True, state_context=context)

    model_path = tmp_path / "actor_critic_policy.pt"
    meta_path = tmp_path / "actor_critic_policy_meta.json"
    optimizer.save(str(model_path), str(meta_path))
    assert model_path.exists()
    assert meta_path.exists()

    loaded = TorchActorCriticOptimizer(store=store, model_path=str(model_path), meta_path=str(meta_path))
    assert loaded.select("DDoS", state_context=context).strategy_id


def test_attack_defender_actor_critic_outputs_model_and_summary():
    pytest.importorskip("torch")
    run_cmd([
        "strategy_loader.py",
        "--config", "configs/strategies.yaml",
        "--db", "data/policies.sqlite",
    ])
    run_cmd([
        "scripts/make_sample_cicids.py",
        "--out", "data/sample_cicids.csv",
    ])

    run_cmd([
        "attack_defender.py",
        "--input", "data/sample_cicids.csv",
        "--build-templates",
        "--window-size", "100",
        "--limit", "700",
        "--optimizer", "actor_critic",
    ])

    summary_file = ROOT / "reports" / "dynamic_defense_summary.json"
    events_file = ROOT / "reports" / "dynamic_defense_events.csv"
    model_file = ROOT / "models" / "actor_critic_policy.pt"
    meta_file = ROOT / "models" / "actor_critic_policy_meta.json"

    assert summary_file.exists()
    assert events_file.exists()
    assert model_file.exists()
    assert meta_file.exists()

    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    assert summary["detector"] == "template"
    assert summary["optimizer"] == "actor_critic"
    assert summary["windows"] >= 1
    assert summary["adjustment_events"] >= 1
    assert "detection_success_rate" in summary
    assert "defense_success_rate" in summary
    assert summary["strategy_counts"]

    header = events_file.read_text(encoding="utf-8").splitlines()[0].split(",")
    for field in ["strategy_id", "reward", "detector_source", "torch_label", "torch_confidence"]:
        assert field in header
