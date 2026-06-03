from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cicids2017_test_sample import (
    NO_ANOMALY_SCENARIO,
    SUPPORTED_SCENARIOS,
    evaluate_no_anomaly_summary,
    evaluate_role_summary,
    get_scenario_display_name,
    get_scenario_paths,
)


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "generated"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="评估生成后的 CICIDS2017 流程测试样本，适用于 inference -> node_summary -> repair_plan 流程。"
    )
    parser.add_argument(
        "--scenario",
        required=True,
        choices=SUPPORTED_SCENARIOS,
        help="要评估的测试场景。",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="包含测试场景输出的根目录。",
    )
    parser.add_argument(
        "--predictions-path",
        type=Path,
        default=None,
        help="inference.py 生成的预测 CSV。无异常节点场景下可额外统计流级误报率。",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="node_summary.py 生成的节点摘要 CSV。默认使用场景标准输出路径。",
    )
    parser.add_argument(
        "--ground-truth-path",
        type=Path,
        default=None,
        help="节点真值 CSV 路径。默认使用场景标准输出路径。",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="评估报告 JSON 路径。默认使用场景标准输出路径。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario_dir = args.output_root / args.scenario
    defaults = get_scenario_paths(scenario_dir, args.scenario)
    summary_path = args.summary_path or defaults["summary_csv"]
    ground_truth_path = args.ground_truth_path or defaults["ground_truth_csv"]
    output_path = args.output_path or defaults["report_json"]

    if args.scenario != NO_ANOMALY_SCENARIO:
        report = evaluate_role_summary(
            summary_path=summary_path,
            ground_truth_path=ground_truth_path,
            output_path=output_path,
        )
        print(f"Scenario: {args.scenario} ({get_scenario_display_name(args.scenario)})")
        print(f"Nodes evaluated: {report['nodes_evaluated']}")
        print(f"Overall accuracy: {report['overall_accuracy']:.4f}")
        print(f"Report: {output_path}")
        return

    predictions_path = args.predictions_path or defaults["predictions_csv"]
    report = evaluate_no_anomaly_summary(
        predictions_path=predictions_path,
        summary_path=summary_path,
        ground_truth_path=ground_truth_path,
        output_path=output_path,
    )
    print(f"Scenario: {args.scenario} ({get_scenario_display_name(args.scenario)})")
    print(f"Benign nodes: {report['benign_nodes']}")
    print(f"Window FPR: {report['window_false_positive_rate']:.4f}")
    print(f"Node FPR (any window): {report['node_false_positive_rate_any_window']:.4f}")
    if report["flow_fpr"] is not None:
        print(f"Flow FPR: {report['flow_fpr']['flow_false_positive_rate']:.4f}")
    print(f"Report: {output_path}")


if __name__ == "__main__":
    main()
