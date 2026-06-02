#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.dynamic_defense.feature_extractor import (
    ThreatFeatureMatcher,
    build_heuristic_templates,
    build_templates_from_labeled_csv,
)
from src.dynamic_defense.utils import dump_json


def main() -> None:
    parser = argparse.ArgumentParser(description="测试用例35：攻击特征向量化与特征库匹配")
    parser.add_argument("--input", required=True, help="CICIDS 2017 CSV 或等价流量特征 CSV")
    parser.add_argument("--config", default="configs/feature_templates.yaml", help="特征模板配置")
    parser.add_argument("--templates", default="data/threat_templates.json", help="模板 JSON 路径")
    parser.add_argument("--build-templates", action="store_true", help="使用带 Label 的输入数据重建模板")
    parser.add_argument("--limit", type=int, default=1000, help="最多分析行数")
    parser.add_argument("--threshold", type=float, default=0.25, help="相似度匹配阈值")
    parser.add_argument("--out-csv", default="reports/feature_match_report.csv")
    parser.add_argument("--out-json", default="reports/feature_match_summary.json")
    args = parser.parse_args()

    Path(args.templates).parent.mkdir(parents=True, exist_ok=True)
    if args.build_templates:
        try:
            build_templates_from_labeled_csv(args.input, args.config, args.templates)
        except Exception as exc:
            print(f"自动构建模板失败，改用启发式模板：{exc}")
            build_heuristic_templates(args.config, args.templates)

    matcher = ThreatFeatureMatcher.from_path_or_config(args.templates, args.config, threshold=args.threshold)
    report = matcher.analyze_csv(args.input, limit=args.limit)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.out_csv, index=False)

    summary = {
        "rows": int(len(report)),
        "matched_attack_counts": report["matched_attack_type"].value_counts().to_dict(),
        "avg_match_score": float(report["match_score"].mean()) if len(report) else 0.0,
        "output_csv": args.out_csv,
    }
    dump_json(summary, args.out_json)
    print(json.dumps({"status": "OK", **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
