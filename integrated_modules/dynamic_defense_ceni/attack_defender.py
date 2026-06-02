#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.dynamic_defense.ceni_adapter import CeniActionAdapter
from src.dynamic_defense.ac_optimizer import TorchActorCriticOptimizer
from src.dynamic_defense.defense_engine import DynamicDefenseEngine
from src.dynamic_defense.evaluation import build_evaluation_summary
from src.dynamic_defense.torch_detector import TorchFlowDetector
from src.dynamic_defense.feature_extractor import ThreatFeatureMatcher, build_heuristic_templates, build_templates_from_labeled_csv
from src.dynamic_defense.policy_store import PolicyStore
from src.dynamic_defense.utils import dump_json


def main() -> None:
    parser = argparse.ArgumentParser(description="测试用例36：持续攻击场景下的动态防御策略调整")
    parser.add_argument("--input", required=True, help="CICIDS 2017 CSV 或等价流量特征 CSV")
    parser.add_argument("--db", default="data/policies.sqlite", help="strategy_loader.py 生成的策略库")
    parser.add_argument("--config", default="configs/feature_templates.yaml", help="特征模板配置")
    parser.add_argument("--templates", default="data/threat_templates.json", help="模板 JSON 路径")
    parser.add_argument("--build-templates", action="store_true", help="使用带 Label 的输入数据重建模板")
    parser.add_argument("--window-size", type=int, default=200, help="每个动态决策窗口的流记录数")
    parser.add_argument("--limit", type=int, default=2000, help="最多处理行数")
    parser.add_argument("--adapter", choices=["dry_run", "rest", "local"], default="dry_run", help="防御动作执行模式")
    parser.add_argument("--controller-endpoint", default=None, help="REST 模式下的控制器地址")
    parser.add_argument("--detector", choices=["template", "torch", "hybrid"], default="template")
    parser.add_argument("--torch-model", default="models/torch_flow_classifier.pt")
    parser.add_argument("--torch-meta", default="models/torch_flow_classifier_meta.json")
    parser.add_argument("--torch-threshold", type=float, default=0.70)
    parser.add_argument("--optimizer", choices=["heuristic", "actor_critic"], default="heuristic")
    parser.add_argument("--ac-model", default="models/actor_critic_policy.pt")
    parser.add_argument("--ac-meta", default="models/actor_critic_policy_meta.json")
    parser.add_argument("--ac-lr", type=float, default=0.001)
    parser.add_argument("--ac-gamma", type=float, default=0.95)
    parser.add_argument("--out-csv", default="reports/dynamic_defense_events.csv")
    parser.add_argument("--out-json", default="reports/dynamic_defense_summary.json")
    args = parser.parse_args()

    if args.build_templates:
        try:
            build_templates_from_labeled_csv(args.input, args.config, args.templates)
        except Exception as exc:
            print(f"自动构建模板失败，改用启发式模板：{exc}")
            build_heuristic_templates(args.config, args.templates)

    store = PolicyStore(args.db)
    matcher = ThreatFeatureMatcher.from_path_or_config(args.templates, args.config)
    adapter = CeniActionAdapter(mode=args.adapter, endpoint=args.controller_endpoint)

    torch_detector = None
    if args.detector in {"torch", "hybrid"}:
        torch_detector = TorchFlowDetector(
            model_path=args.torch_model,
            meta_path=args.torch_meta,
            device="cpu",
        )

    optimizer = None
    if args.optimizer == "actor_critic":
        optimizer = TorchActorCriticOptimizer(
            store=store,
            lr=args.ac_lr,
            gamma=args.ac_gamma,
            device="cpu",
            model_path=args.ac_model,
            meta_path=args.ac_meta,
        )

    engine = DynamicDefenseEngine(
        store=store,
        matcher=matcher,
        adapter=adapter,
        detector_mode=args.detector,
        torch_detector=torch_detector,
        torch_confidence_threshold=args.torch_threshold,
        optimizer=optimizer,
    )
    events = engine.run_on_csv(args.input, window_size=args.window_size, limit=args.limit)
    if args.optimizer == "actor_critic":
        engine.optimizer.save(args.ac_model, args.ac_meta)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(args.out_csv, index=False)
    summary = {
        "detector": args.detector,
        "optimizer": args.optimizer,
        "windows": int(len(events)),
        "adjustment_events": int(events["adjustment_triggered"].sum()) if len(events) else 0,
        "detection_success_rate": float(events["detection_success"].mean()) if len(events) and "detection_success" in events.columns else 0.0,
        "defense_success_rate": float(events["defense_success"].mean()) if len(events) else 0.0,
        "strategy_counts": events["strategy_id"].value_counts().to_dict() if len(events) else {},
        "output_csv": args.out_csv,
    }
    summary.update(build_evaluation_summary(events))
    dump_json(summary, args.out_json)
    print(json.dumps({"status": "OK", **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
