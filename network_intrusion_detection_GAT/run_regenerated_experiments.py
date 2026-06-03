from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import pandas as pd

from inference import (
    build_model_from_checkpoint,
    clean_inference_chunk,
    infer_task_from_class_names,
    load_checkpoint,
    predict_dataframe,
)
from node_summary import (
    DEFAULT_MIN_DIRECTIONAL_HIGH_CONF_FLOWS,
    DEFAULT_MIN_ROLE_ANOMALY_RATIO,
    DEFAULT_MIN_ROLE_HIGH_CONF_FLOWS,
    DEFAULT_MIN_ROLE_TOTAL_FLOWS,
    normalize_inference_dataframe,
    summarize_nodes,
)
from src.cicids2017_test_sample import (
    DEFAULT_DATASET_DIR,
    MULTI_ANOMALY_SCENARIO,
    NO_ANOMALY_SCENARIO,
    SINGLE_ANOMALY_SCENARIO,
    evaluate_no_anomaly_summary,
    evaluate_role_summary,
    generate_test_samples,
    get_scenario_paths,
)
from src.repair import build_repair_plan


DEFAULT_MODEL_PATH = Path("outputs") / "training" / "20260511_182846" / "model.pt"
DEFAULT_OUTPUT_ROOT = Path("outputs") / "experiments" / "regenerated_samples"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run repeated end-to-end experiments with freshly regenerated three-scenario test samples."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--runs", type=int, default=5, help="Number of experiments to run.")
    parser.add_argument(
        "--seed-start",
        type=int,
        default=2026051201,
        help="Base seed. Each experiment uses seed_start + run_index.",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--decision-threshold", type=float, default=0.55)
    parser.add_argument("--anomaly-threshold", type=float, default=0.65)
    parser.add_argument("--role-threshold", type=float, default=0.65)
    parser.add_argument("--min-role-total-flows", type=int, default=DEFAULT_MIN_ROLE_TOTAL_FLOWS)
    parser.add_argument("--min-role-anomaly-ratio", type=float, default=DEFAULT_MIN_ROLE_ANOMALY_RATIO)
    parser.add_argument("--min-role-high-conf-flows", type=int, default=DEFAULT_MIN_ROLE_HIGH_CONF_FLOWS)
    parser.add_argument(
        "--min-directional-high-conf-flows",
        type=int,
        default=DEFAULT_MIN_DIRECTIONAL_HIGH_CONF_FLOWS,
    )
    parser.add_argument("--no-anomaly-nodes", type=int, default=None)
    parser.add_argument("--single-anomaly-nodes", type=int, default=None)
    parser.add_argument("--multi-anomaly-nodes", type=int, default=None)
    parser.add_argument(
        "--randomize-large-node-counts",
        action="store_true",
        help="Randomize each scenario's node count per run within [min-large-nodes, max-large-nodes].",
    )
    parser.add_argument("--min-large-nodes", type=int, default=50)
    parser.add_argument("--max-large-nodes", type=int, default=100)
    parser.add_argument("--core-top-ratio", type=float, default=0.30)
    return parser.parse_args()


def ensure_cleaned_dataframe(sample_csv: Path, task: str) -> pd.DataFrame:
    frame = pd.read_csv(sample_csv, low_memory=False)
    return clean_inference_chunk(frame, task=task)


def scenario_benign_fp(report: dict[str, Any]) -> float:
    details = report.get("details") or report.get("node_details") or []
    if not details:
        return 0.0
    df = pd.DataFrame(details)
    if "ground_truth_role" not in df.columns or "predicted_role" not in df.columns:
        return 0.0
    benign = df[df["ground_truth_role"].astype(str) == "uncertain"]
    if benign.empty:
        return 0.0
    return float((benign["predicted_role"].astype(str) != "uncertain").mean())


def single_anomaly_rank_metrics(repair_order: pd.DataFrame, ground_truth_csv: Path) -> dict[str, Any]:
    truth = pd.read_csv(ground_truth_csv, low_memory=False)
    true_nodes = set(truth[truth["ground_truth_role"].astype(str) != "uncertain"]["node_id"].astype(str))
    if repair_order.empty:
        return {"true_anomaly_rank": None, "true_anomaly_rank_ratio": None, "true_anomaly_in_top_20pct": False}
    hits = repair_order[repair_order["node_id"].astype(str).isin(true_nodes)]
    if hits.empty:
        return {"true_anomaly_rank": None, "true_anomaly_rank_ratio": None, "true_anomaly_in_top_20pct": False}
    best_rank = int(hits["repair_rank"].min())
    plan_len = int(len(repair_order))
    top_k = max(1, math.ceil(plan_len * 0.20))
    return {
        "true_anomaly_rank": best_rank,
        "true_anomaly_rank_ratio": best_rank / plan_len,
        "true_anomaly_in_top_20pct": best_rank <= top_k,
    }


def multi_anomaly_recall_metrics(repair_order: pd.DataFrame, ground_truth_csv: Path) -> dict[str, Any]:
    truth = pd.read_csv(ground_truth_csv, low_memory=False)
    true_nodes = set(truth[truth["ground_truth_role"].astype(str) != "uncertain"]["node_id"].astype(str))
    predicted_nodes = set(repair_order["node_id"].astype(str)) if not repair_order.empty else set()
    hit_count = len(predicted_nodes & true_nodes)
    total = len(true_nodes)
    recall = hit_count / total if total else 0.0
    return {
        "true_anomaly_nodes": total,
        "true_anomaly_nodes_in_repair": hit_count,
        "true_anomaly_recall_in_repair": recall,
    }


def validate_node_count_range(args: argparse.Namespace) -> None:
    if args.min_large_nodes < 1:
        raise ValueError("--min-large-nodes must be positive.")
    if args.max_large_nodes < args.min_large_nodes:
        raise ValueError("--max-large-nodes must be greater than or equal to --min-large-nodes.")


def resolve_node_count_overrides(args: argparse.Namespace, run_seed: int) -> dict[str, int]:
    explicit = {
        NO_ANOMALY_SCENARIO: args.no_anomaly_nodes,
        SINGLE_ANOMALY_SCENARIO: args.single_anomaly_nodes,
        MULTI_ANOMALY_SCENARIO: args.multi_anomaly_nodes,
    }
    overrides = {key: int(value) for key, value in explicit.items() if value is not None}

    if args.randomize_large_node_counts:
        rng = random.Random(run_seed)
        randomized = {
            NO_ANOMALY_SCENARIO: rng.randint(args.min_large_nodes, args.max_large_nodes),
            SINGLE_ANOMALY_SCENARIO: rng.randint(args.min_large_nodes, args.max_large_nodes),
            MULTI_ANOMALY_SCENARIO: rng.randint(args.min_large_nodes, args.max_large_nodes),
        }
        randomized.update(overrides)
        return randomized

    if overrides:
        return overrides

    return {}


def run_one_experiment(
    *,
    run_index: int,
    run_seed: int,
    output_root: Path,
    dataset_dir: Path,
    model: Any,
    checkpoint: dict[str, Any],
    task: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    run_dir = output_root / f"run_{run_index:02d}_seed_{run_seed}"
    sample_root = run_dir / "samples"
    node_count_overrides = resolve_node_count_overrides(args, run_seed)
    generate_test_samples(
        dataset_dir=dataset_dir,
        output_root=sample_root,
        scenarios=[NO_ANOMALY_SCENARIO, SINGLE_ANOMALY_SCENARIO, MULTI_ANOMALY_SCENARIO],
        seed=run_seed,
        node_count_overrides=node_count_overrides or None,
    )

    scenario_results: dict[str, dict[str, Any]] = {}
    for scenario in (NO_ANOMALY_SCENARIO, SINGLE_ANOMALY_SCENARIO, MULTI_ANOMALY_SCENARIO):
        scenario_dir = sample_root / scenario
        paths = get_scenario_paths(scenario_dir, scenario)
        sample_csv = paths["sample_csv"]
        predictions_csv = paths["predictions_csv"]
        summary_csv = paths["summary_csv"]
        report_json = paths["report_json"]
        repair_csv = scenario_dir / f"{sample_csv.stem.replace('_sample', '')}_repair_order.csv"
        repair_json = scenario_dir / f"{sample_csv.stem.replace('_sample', '')}_repair_report.json"

        cleaned = ensure_cleaned_dataframe(sample_csv, task=task)
        predicted = predict_dataframe(
            df=cleaned,
            model=model,
            checkpoint=checkpoint,
            device=args.device,
            batch_size=args.batch_size,
            decision_threshold=args.decision_threshold,
        )
        predicted.insert(0, "source_file", sample_csv.name)
        predictions_csv.parent.mkdir(parents=True, exist_ok=True)
        predicted.to_csv(predictions_csv, index=False, encoding="utf-8-sig")

        normalized = normalize_inference_dataframe(predicted)
        summary = summarize_nodes(
            normalized,
            window="1h",
            anomaly_threshold=args.anomaly_threshold,
            role_threshold=args.role_threshold,
            min_role_total_flows=args.min_role_total_flows,
            min_role_anomaly_ratio=args.min_role_anomaly_ratio,
            min_role_high_conf_flows=args.min_role_high_conf_flows,
            min_directional_high_conf_flows=args.min_directional_high_conf_flows,
        )
        summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")

        repair_result = build_repair_plan(summary, core_top_ratio=args.core_top_ratio)
        repair_result.repair_order.to_csv(repair_csv, index=False, encoding="utf-8-sig")
        repair_payload = {
            "total_node_count": int(repair_result.total_node_count),
            "anomalous_node_count": int(repair_result.anomalous_node_count),
            "core_node_count": int(repair_result.core_node_count),
            "minimum_cost": float(repair_result.cost),
            "repair_order": repair_result.repair_order.to_dict(orient="records"),
        }
        with repair_json.open("w", encoding="utf-8") as handle:
            json.dump(repair_payload, handle, ensure_ascii=False, indent=2)

        if scenario == NO_ANOMALY_SCENARIO:
            report = evaluate_no_anomaly_summary(
                predictions_path=predictions_csv,
                summary_path=summary_csv,
                ground_truth_path=paths["ground_truth_csv"],
                output_path=report_json,
            )
        else:
            report = evaluate_role_summary(
                summary_path=summary_csv,
                ground_truth_path=paths["ground_truth_csv"],
                output_path=report_json,
            )

        result = {
            "sample_csv": str(sample_csv),
            "predictions_csv": str(predictions_csv),
            "summary_csv": str(summary_csv),
            "report_json": str(report_json),
            "repair_csv": str(repair_csv),
            "repair_json": str(repair_json),
            "benign_fp": scenario_benign_fp(report),
            "repair_order_len": int(len(repair_result.repair_order)),
        }
        if scenario == NO_ANOMALY_SCENARIO:
            result.update(
                {
                    "window_false_positive_rate": float(report["window_false_positive_rate"]),
                    "node_false_positive_rate_any_window": float(report["node_false_positive_rate_any_window"]),
                    "flow_false_positive_rate": float(report["flow_fpr"]["flow_false_positive_rate"])
                    if report.get("flow_fpr")
                    else None,
                }
            )
        elif scenario == SINGLE_ANOMALY_SCENARIO:
            result.update(
                {
                    "overall_accuracy": float(report["overall_accuracy"]),
                    **single_anomaly_rank_metrics(repair_result.repair_order, paths["ground_truth_csv"]),
                }
            )
        else:
            result.update(
                {
                    "overall_accuracy": float(report["overall_accuracy"]),
                    **multi_anomaly_recall_metrics(repair_result.repair_order, paths["ground_truth_csv"]),
                }
            )
        scenario_results[scenario] = result

    pass_flags = {
        "no_anomaly_fp": scenario_results[NO_ANOMALY_SCENARIO]["node_false_positive_rate_any_window"] < 0.05,
        "single_anomaly_fp": scenario_results[SINGLE_ANOMALY_SCENARIO]["benign_fp"] < 0.05,
        "single_anomaly_rank": bool(scenario_results[SINGLE_ANOMALY_SCENARIO]["true_anomaly_in_top_20pct"]),
        "multi_anomaly_fp": scenario_results[MULTI_ANOMALY_SCENARIO]["benign_fp"] < 0.05,
        "multi_anomaly_recall": scenario_results[MULTI_ANOMALY_SCENARIO]["true_anomaly_recall_in_repair"] >= 0.80,
    }

    run_summary = {
        "run_index": run_index,
        "run_seed": run_seed,
        "sample_root": str(sample_root),
        "params": {
            "decision_threshold": args.decision_threshold,
            "anomaly_threshold": args.anomaly_threshold,
            "role_threshold": args.role_threshold,
            "min_role_total_flows": args.min_role_total_flows,
            "min_role_anomaly_ratio": args.min_role_anomaly_ratio,
            "min_role_high_conf_flows": args.min_role_high_conf_flows,
            "min_directional_high_conf_flows": args.min_directional_high_conf_flows,
            "core_top_ratio": args.core_top_ratio,
        },
        "node_count_overrides": node_count_overrides,
        "scenarios": scenario_results,
        "pass_flags": pass_flags,
        "all_passed": all(pass_flags.values()),
    }
    with (run_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(run_summary, handle, ensure_ascii=False, indent=2)
    return run_summary


def main() -> None:
    args = parse_args()
    validate_node_count_range(args)
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint(args.model_path, args.device)
    model = build_model_from_checkpoint(checkpoint, args.device)
    class_names = [str(name) for name in checkpoint["class_names"]]
    task = infer_task_from_class_names(class_names)

    all_runs: list[dict[str, Any]] = []
    for run_index in range(1, args.runs + 1):
        run_seed = args.seed_start + run_index - 1
        summary = run_one_experiment(
            run_index=run_index,
            run_seed=run_seed,
            output_root=output_root,
            dataset_dir=args.dataset_dir,
            model=model,
            checkpoint=checkpoint,
            task=task,
            args=args,
        )
        all_runs.append(summary)
        print(
            f"Run {run_index}/{args.runs} seed={run_seed}: "
            f"all_passed={summary['all_passed']}, "
            f"no_fp={summary['scenarios'][NO_ANOMALY_SCENARIO]['node_false_positive_rate_any_window']:.4f}, "
            f"single_fp={summary['scenarios'][SINGLE_ANOMALY_SCENARIO]['benign_fp']:.4f}, "
            f"single_top20={summary['scenarios'][SINGLE_ANOMALY_SCENARIO]['true_anomaly_in_top_20pct']}, "
            f"multi_fp={summary['scenarios'][MULTI_ANOMALY_SCENARIO]['benign_fp']:.4f}, "
            f"multi_recall={summary['scenarios'][MULTI_ANOMALY_SCENARIO]['true_anomaly_recall_in_repair']:.4f}"
        )

    rows: list[dict[str, Any]] = []
    for summary in all_runs:
        rows.append(
            {
                "run_index": summary["run_index"],
                "run_seed": summary["run_seed"],
                "all_passed": summary["all_passed"],
                "no_anomaly_nodes": summary["node_count_overrides"].get(NO_ANOMALY_SCENARIO),
                "single_anomaly_nodes": summary["node_count_overrides"].get(SINGLE_ANOMALY_SCENARIO),
                "multi_anomaly_nodes": summary["node_count_overrides"].get(MULTI_ANOMALY_SCENARIO),
                "no_anomaly_node_fpr_any": summary["scenarios"][NO_ANOMALY_SCENARIO]["node_false_positive_rate_any_window"],
                "no_anomaly_flow_fpr": summary["scenarios"][NO_ANOMALY_SCENARIO]["flow_false_positive_rate"],
                "single_anomaly_benign_fp": summary["scenarios"][SINGLE_ANOMALY_SCENARIO]["benign_fp"],
                "single_anomaly_overall_accuracy": summary["scenarios"][SINGLE_ANOMALY_SCENARIO]["overall_accuracy"],
                "single_anomaly_true_rank": summary["scenarios"][SINGLE_ANOMALY_SCENARIO]["true_anomaly_rank"],
                "single_anomaly_top20": summary["scenarios"][SINGLE_ANOMALY_SCENARIO]["true_anomaly_in_top_20pct"],
                "multi_anomaly_benign_fp": summary["scenarios"][MULTI_ANOMALY_SCENARIO]["benign_fp"],
                "multi_anomaly_overall_accuracy": summary["scenarios"][MULTI_ANOMALY_SCENARIO]["overall_accuracy"],
                "multi_anomaly_recall_in_repair": summary["scenarios"][MULTI_ANOMALY_SCENARIO]["true_anomaly_recall_in_repair"],
            }
        )

    summary_df = pd.DataFrame(rows)
    summary_csv = output_root / "experiment_summary.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    aggregate = {
        "model_path": str(args.model_path),
        "runs": args.runs,
        "seed_start": args.seed_start,
        "params": {
            "decision_threshold": args.decision_threshold,
            "anomaly_threshold": args.anomaly_threshold,
            "role_threshold": args.role_threshold,
            "min_role_total_flows": args.min_role_total_flows,
            "min_role_anomaly_ratio": args.min_role_anomaly_ratio,
            "min_role_high_conf_flows": args.min_role_high_conf_flows,
            "min_directional_high_conf_flows": args.min_directional_high_conf_flows,
            "no_anomaly_nodes": args.no_anomaly_nodes,
            "single_anomaly_nodes": args.single_anomaly_nodes,
            "multi_anomaly_nodes": args.multi_anomaly_nodes,
            "randomize_large_node_counts": bool(args.randomize_large_node_counts),
            "min_large_nodes": args.min_large_nodes,
            "max_large_nodes": args.max_large_nodes,
            "core_top_ratio": args.core_top_ratio,
        },
        "all_pass_count": int(summary_df["all_passed"].sum()) if not summary_df.empty else 0,
        "summary_csv": str(summary_csv),
        "runs_detail": all_runs,
    }
    with (output_root / "experiment_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
