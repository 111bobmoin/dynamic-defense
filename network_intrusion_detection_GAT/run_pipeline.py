from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from inference import (
    build_empty_prediction_frame,
    build_model_from_checkpoint,
    clean_inference_chunk,
    evaluate_if_possible,
    find_latest_model,
    infer_task_from_class_names,
    load_checkpoint,
    predict_dataframe,
)
from node_summary import (
    NODE_OUTPUT_COLUMNS,
    DEFAULT_MIN_DIRECTIONAL_HIGH_CONF_FLOWS,
    DEFAULT_MIN_ROLE_ANOMALY_RATIO,
    DEFAULT_MIN_ROLE_HIGH_CONF_FLOWS,
    DEFAULT_MIN_ROLE_TOTAL_FLOWS,
    normalize_inference_dataframe,
    summarize_nodes,
)
from pcap_to_csv import convert_one_file, discover_pcap_files
from src.output_layout import make_result_dir, write_manifest
from src.repair import REPAIR_ORDER_COLUMNS, RepairPlanResult, build_repair_plan
from src.threat_scenarios import build_threat_node_summary, filter_threat_scenarios, load_threat_scenarios
from src.threat_triggers import load_threat_triggers, resolve_triggered_scenario_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full PCAP -> CSV -> inference -> node summary -> repair plan "
            "pipeline and store all artifacts in one dedicated result directory."
        )
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="A .pcap/.pcapng file or a directory that contains them.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to a trained model.pt. If omitted, use the latest model under outputs/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Dedicated result directory. Default: outputs/results/<timestamp>_<name>/",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Optional label injected into converted flow CSVs for later evaluation.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=20000,
        help="CSV chunk size when streaming converted flow files.",
    )
    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=0.55,
        help="Anomaly-score threshold used to derive flow-level is_anomaly. Default: 0.55",
    )
    parser.add_argument(
        "--window",
        default="1h",
        help="Pandas time window used for node aggregation. Default: 1h",
    )
    parser.add_argument(
        "--anomaly-threshold",
        type=float,
        default=0.65,
        help="Minimum anomaly score to count a flow as high-confidence anomalous for node scoring. Default: 0.65",
    )
    parser.add_argument(
        "--role-threshold",
        type=float,
        default=0.65,
        help="Minimum role score required before a node is labeled attacker/victim/compromised. Default: 0.65",
    )
    parser.add_argument(
        "--min-role-total-flows",
        type=int,
        default=DEFAULT_MIN_ROLE_TOTAL_FLOWS,
        help=(
            "Minimum flows required in a window before any non-uncertain role can be assigned. "
            f"Default: {DEFAULT_MIN_ROLE_TOTAL_FLOWS}"
        ),
    )
    parser.add_argument(
        "--min-role-anomaly-ratio",
        type=float,
        default=DEFAULT_MIN_ROLE_ANOMALY_RATIO,
        help=(
            "Minimum anomalous-flow ratio required in a window before any non-uncertain role can be assigned. "
            f"Default: {DEFAULT_MIN_ROLE_ANOMALY_RATIO}"
        ),
    )
    parser.add_argument(
        "--min-role-high-conf-flows",
        type=int,
        default=DEFAULT_MIN_ROLE_HIGH_CONF_FLOWS,
        help=(
            "Minimum high-confidence anomalous flows required in a window before any non-uncertain role can be assigned. "
            f"Default: {DEFAULT_MIN_ROLE_HIGH_CONF_FLOWS}"
        ),
    )
    parser.add_argument(
        "--min-directional-high-conf-flows",
        type=int,
        default=DEFAULT_MIN_DIRECTIONAL_HIGH_CONF_FLOWS,
        help=(
            "Minimum high-confidence anomalous flows required in the decisive direction "
            "(outbound for attacker, inbound for victim, both for compromised). "
            f"Default: {DEFAULT_MIN_DIRECTIONAL_HIGH_CONF_FLOWS}"
        ),
    )
    parser.add_argument(
        "--core-top-ratio",
        type=float,
        default=0.30,
        help="Top ratio of anomalous nodes marked as core nodes for repair planning.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose cicflowmeter logging during PCAP conversion.",
    )
    parser.add_argument(
        "--threat-scenarios-path",
        type=Path,
        default=Path("threat_scenarios.yaml"),
        help="Threat scenario description file used to generate a parallel node-summary/repair branch.",
    )
    parser.add_argument(
        "--threat-triggers-path",
        type=Path,
        default=None,
        help=(
            "Optional threat trigger file. If provided, the parallel threat-scenario branch only outputs "
            "scenarios triggered for the current input; otherwise all scenarios are emitted as before."
        ),
    )
    return parser.parse_args()


def run_inference_for_csv(
    input_csv: Path,
    output_csv: Path,
    model: Any,
    checkpoint: dict[str, Any],
    *,
    device: str,
    batch_size: int,
    chunksize: int,
    source_name: str,
    decision_threshold: float | None = None,
) -> tuple[pd.DataFrame, dict[str, float] | None]:
    class_names = [str(name) for name in checkpoint["class_names"]]
    task = infer_task_from_class_names(class_names)
    results: list[pd.DataFrame] = []

    for chunk in pd.read_csv(input_csv, chunksize=chunksize, low_memory=False):
        cleaned = clean_inference_chunk(chunk, task=task)
        predicted = predict_dataframe(
            df=cleaned,
            model=model,
            checkpoint=checkpoint,
            device=device,
            batch_size=batch_size,
            decision_threshold=decision_threshold,
        )
        predicted.insert(0, "source_file", source_name)
        results.append(predicted)

    if not results:
        empty_predicted = build_empty_prediction_frame(
            input_csv,
            task=task,
            model=model,
            checkpoint=checkpoint,
            device=device,
            batch_size=batch_size,
            decision_threshold=decision_threshold,
        )
        empty_predicted.insert(0, "source_file", source_name)
        results.append(empty_predicted)

    merged = pd.concat(results, ignore_index=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return merged, evaluate_if_possible(merged, class_names=class_names)


def run_node_summary_for_predictions(
    prediction_df: pd.DataFrame,
    output_csv: Path,
    *,
    window: str,
    anomaly_threshold: float,
    role_threshold: float,
    min_role_total_flows: int,
    min_role_anomaly_ratio: float,
    min_role_high_conf_flows: int,
    min_directional_high_conf_flows: int,
) -> pd.DataFrame:
    if prediction_df.empty:
        summary = pd.DataFrame(columns=NODE_OUTPUT_COLUMNS)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_csv, index=False, encoding="utf-8-sig")
        return summary

    normalized = normalize_inference_dataframe(prediction_df)
    summary = summarize_nodes(
        normalized,
        window=window,
        anomaly_threshold=anomaly_threshold,
        role_threshold=role_threshold,
        min_role_total_flows=min_role_total_flows,
        min_role_anomaly_ratio=min_role_anomaly_ratio,
        min_role_high_conf_flows=min_role_high_conf_flows,
        min_directional_high_conf_flows=min_directional_high_conf_flows,
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return summary


def run_repair_plan_for_summary(
    summary_df: pd.DataFrame,
    output_csv: Path,
    *,
    core_top_ratio: float,
) -> RepairPlanResult:
    if summary_df.empty:
        empty_result = build_repair_plan(summary_df, core_top_ratio=core_top_ratio)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        empty_result.repair_order.to_csv(output_csv, index=False, encoding="utf-8-sig")
        return empty_result

    result = build_repair_plan(summary_df, core_top_ratio=core_top_ratio)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.repair_order.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return result


def main() -> None:
    args = parse_args()
    input_files = discover_pcap_files(args.input_path)
    result_dir = make_result_dir(args.input_path, result_dir=args.output_dir)
    pcap_csv_dir = result_dir / "pcap_csv"
    inference_dir = result_dir / "inference"
    node_summary_dir = result_dir / "node_summary"
    repair_plan_dir = result_dir / "repair_plan"
    threat_node_summary_dir = result_dir / "threat_node_summary"
    threat_repair_plan_dir = result_dir / "threat_repair_plan"

    model_path = args.model_path or find_latest_model(Path("outputs"))
    checkpoint = load_checkpoint(model_path, args.device)
    model = build_model_from_checkpoint(checkpoint, args.device)

    file_results: list[dict[str, Any]] = []
    repair_manifest_entries: list[dict[str, Any]] = []
    for input_file in input_files:
        converted_csv = pcap_csv_dir / f"{input_file.stem}_flows.csv"
        prediction_csv = inference_dir / f"{input_file.stem}_predictions.csv"
        node_summary_csv = node_summary_dir / f"{input_file.stem}_node_summary.csv"
        repair_order_csv = repair_plan_dir / f"{input_file.stem}_repair_order.csv"

        converted_rows = convert_one_file(
            input_file=input_file,
            output_csv=converted_csv,
            label=args.label,
            verbose=args.verbose,
            keep_metadata=True,
        )
        prediction_df, metrics = run_inference_for_csv(
            input_csv=converted_csv,
            output_csv=prediction_csv,
            model=model,
            checkpoint=checkpoint,
            device=args.device,
            batch_size=args.batch_size,
            chunksize=args.chunksize,
            source_name=input_file.name,
            decision_threshold=args.decision_threshold,
        )
        node_summary_df = run_node_summary_for_predictions(
            prediction_df=prediction_df,
            output_csv=node_summary_csv,
            window=args.window,
            anomaly_threshold=args.anomaly_threshold,
            role_threshold=args.role_threshold,
            min_role_total_flows=args.min_role_total_flows,
            min_role_anomaly_ratio=args.min_role_anomaly_ratio,
            min_role_high_conf_flows=args.min_role_high_conf_flows,
            min_directional_high_conf_flows=args.min_directional_high_conf_flows,
        )
        repair_result = run_repair_plan_for_summary(
            summary_df=node_summary_df,
            output_csv=repair_order_csv,
            core_top_ratio=args.core_top_ratio,
        )
        core_nodes = (
            repair_result.repair_order[repair_result.repair_order["is_core"]]["node_id"].astype(str).tolist()
            if not repair_result.repair_order.empty
            else []
        )

        file_results.append(
            {
                "input_pcap": str(input_file),
                "converted_csv": str(converted_csv),
                "prediction_csv": str(prediction_csv),
                "node_summary_csv": str(node_summary_csv),
                "repair_order_csv": str(repair_order_csv),
                "converted_rows": converted_rows,
                "prediction_rows": int(len(prediction_df)),
                "node_summary_rows": int(len(node_summary_df)),
                "repair_order_rows": int(len(repair_result.repair_order)),
                "repair_total_nodes": int(repair_result.total_node_count),
                "repair_anomalous_nodes": int(repair_result.anomalous_node_count),
                "repair_core_nodes": int(repair_result.core_node_count),
                "repair_minimum_cost": float(repair_result.cost),
                "evaluation_metrics": metrics,
            }
        )
        repair_manifest_entries.append(
            {
                "input_pcap": str(input_file),
                "input_node_summary_csv": str(node_summary_csv),
                "repair_order_csv": str(repair_order_csv),
                "core_top_ratio": float(args.core_top_ratio),
                "total_node_count": int(repair_result.total_node_count),
                "anomalous_node_count": int(repair_result.anomalous_node_count),
                "core_node_count": int(repair_result.core_node_count),
                "core_ratio": float(repair_result.core_ratio),
                "formula_denominator": float(repair_result.denominator),
                "minimum_cost": float(repair_result.cost),
                "core_nodes": core_nodes,
                "repair_order": repair_result.repair_order.to_dict(orient="records"),
                "formula_interpretation": repair_result.interpretation,
            }
        )

    threat_manifest_payload: dict[str, Any] | None = None
    if args.threat_scenarios_path.exists():
        scenarios = load_threat_scenarios(args.threat_scenarios_path)
        selected_scenarios = scenarios
        matched_trigger_events: list[dict[str, Any]] = []
        triggered_scenario_ids: list[str] | None = None

        if args.threat_triggers_path is not None:
            if not args.threat_triggers_path.exists():
                raise FileNotFoundError(f"Threat trigger file does not exist: {args.threat_triggers_path}")
            trigger_events = load_threat_triggers(args.threat_triggers_path)
            input_names = [args.input_path.name, args.input_path.stem]
            input_names.extend(path.name for path in input_files)
            input_names.extend(path.stem for path in input_files)
            triggered_scenario_ids, matched_events = resolve_triggered_scenario_ids(
                trigger_events,
                input_names=input_names,
            )
            matched_trigger_events = [
                {
                    "input_name": event.input_name,
                    "scenario_ids": list(event.scenario_ids),
                    "triggered_at": event.triggered_at,
                }
                for event in matched_events
            ]
            selected_scenarios = filter_threat_scenarios(scenarios, triggered_scenario_ids)

        threat_summary = build_threat_node_summary(selected_scenarios)
        threat_input_name = Path(args.input_path).stem if args.input_path.is_file() else args.input_path.name
        threat_summary_csv = threat_node_summary_dir / f"{threat_input_name}_node_summary.csv"
        threat_summary_csv.parent.mkdir(parents=True, exist_ok=True)
        threat_summary.to_csv(threat_summary_csv, index=False, encoding="utf-8-sig")

        threat_repair_csv = threat_repair_plan_dir / f"{threat_input_name}_repair_order.csv"
        threat_repair_result = run_repair_plan_for_summary(
            summary_df=threat_summary,
            output_csv=threat_repair_csv,
            core_top_ratio=args.core_top_ratio,
        )
        threat_repair_manifest_path = threat_repair_plan_dir / "repair_plan_manifest.json"
        threat_manifest_payload = {
            "script": "run_pipeline.py",
            "stage": "threat_scenarios",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "threat_scenarios_path": str(args.threat_scenarios_path),
            "threat_triggers_path": str(args.threat_triggers_path) if args.threat_triggers_path is not None else None,
            "input_path": str(args.input_path),
            "output_root": str(result_dir),
            "node_summary_csv": str(threat_summary_csv),
            "repair_order_csv": str(threat_repair_csv),
            "core_top_ratio": float(args.core_top_ratio),
            "scenario_count": len(selected_scenarios),
            "triggered_scenario_ids": triggered_scenario_ids,
            "matched_trigger_events": matched_trigger_events,
            "node_summary_rows": int(len(threat_summary)),
            "repair_order_rows": int(len(threat_repair_result.repair_order)),
            "total_node_count": int(threat_repair_result.total_node_count),
            "anomalous_node_count": int(threat_repair_result.anomalous_node_count),
            "core_node_count": int(threat_repair_result.core_node_count),
            "minimum_cost": float(threat_repair_result.cost),
        }
        write_manifest(threat_repair_manifest_path, threat_manifest_payload)

    repair_plan_manifest_path = repair_plan_dir / "repair_plan_manifest.json"
    write_manifest(
        repair_plan_manifest_path,
        {
            "script": "run_pipeline.py",
            "stage": "repair_plan",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "input_path": str(args.input_path),
            "model_path": str(model_path),
            "output_root": str(result_dir),
            "core_top_ratio": float(args.core_top_ratio),
            "files": repair_manifest_entries,
        },
    )

    write_manifest(
        result_dir / "manifest.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "input_path": str(args.input_path),
            "model_path": str(model_path),
            "output_dir": str(result_dir),
            "args": {
                "label": args.label,
                "batch_size": args.batch_size,
                "device": args.device,
                "chunksize": args.chunksize,
                "window": args.window,
                "decision_threshold": args.decision_threshold,
                "anomaly_threshold": args.anomaly_threshold,
                "role_threshold": args.role_threshold,
                "min_role_total_flows": args.min_role_total_flows,
                "min_role_anomaly_ratio": args.min_role_anomaly_ratio,
                "min_role_high_conf_flows": args.min_role_high_conf_flows,
                "min_directional_high_conf_flows": args.min_directional_high_conf_flows,
                "core_top_ratio": args.core_top_ratio,
                "verbose": args.verbose,
                "threat_scenarios_path": str(args.threat_scenarios_path),
                "threat_triggers_path": str(args.threat_triggers_path) if args.threat_triggers_path is not None else None,
            },
            "repair_plan_manifest": str(repair_plan_manifest_path),
            "threat_scenarios_manifest": (
                str(threat_repair_plan_dir / "repair_plan_manifest.json") if threat_manifest_payload is not None else None
            ),
            "files": file_results,
        },
    )

    print(f"Result directory: {result_dir}")
    print(f"Model path: {model_path}")
    print(f"PCAP files processed: {len(file_results)}")
    for result in file_results:
        print(f"- {Path(result['input_pcap']).name}:")
        print(
            f"  flows={result['converted_rows']}, "
            f"predictions={result['prediction_rows']}, "
            f"nodes={result['node_summary_rows']}, "
            f"repairs={result['repair_order_rows']}"
        )
        print(
            f"  repair_core_nodes={result['repair_core_nodes']}, "
            f"repair_cost={result['repair_minimum_cost']:.10f}"
        )
        if result["evaluation_metrics"] is not None:
            print("  evaluation_metrics:")
            print(json.dumps(result["evaluation_metrics"], ensure_ascii=False, indent=2))
    if threat_manifest_payload is not None:
        print(
            "Threat-scenario branch: "
            f"nodes={threat_manifest_payload['node_summary_rows']}, "
            f"repairs={threat_manifest_payload['repair_order_rows']}, "
            f"cost={threat_manifest_payload['minimum_cost']:.10f}"
        )


if __name__ == "__main__":
    main()
