from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.output_layout import default_stage_filename, make_stage_dir, write_manifest
from src.repair import build_repair_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a minimum-cost repair order from predicted anomalous node summaries and "
            "compute the corresponding repair cost."
        )
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="node_summary CSV file or a directory of node_summary CSV files.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output CSV path for the repair order table.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Output JSON path for the repair plan report.",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=None,
        help="Dedicated result directory that stores repair outputs under <result-dir>/repair_plan/.",
    )
    parser.add_argument(
        "--core-top-ratio",
        type=float,
        default=0.30,
        help="Top ratio of anomalous nodes marked as core nodes. Default: 0.30",
    )
    return parser.parse_args()


def discover_csv_inputs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        files = sorted(input_path.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"No CSV files found in {input_path}")
        return files
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def validate_output_args(output_path: Path | None, result_dir: Path | None) -> None:
    if output_path is not None and result_dir is not None:
        raise ValueError("Use either --output-path or --result-dir, not both.")


def ensure_output_paths(
    output_path: Path | None,
    report_path: Path | None,
    input_path: Path,
    result_dir: Path | None,
) -> tuple[Path, Path]:
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_report = report_path or output_path.with_name(f"{output_path.stem}_report.json")
        resolved_report.parent.mkdir(parents=True, exist_ok=True)
        return output_path, resolved_report

    stage_dir = make_stage_dir(input_path=input_path, stage="repair_plan", result_dir=result_dir)
    csv_path = stage_dir / default_stage_filename(input_path, "repair_order")
    json_path = report_path or stage_dir / "repair_plan_manifest.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    return csv_path, json_path


def load_summary_dataframe(input_files: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path, low_memory=False) for path in input_files]
    if not frames:
        raise RuntimeError("No node summary CSVs were loaded.")
    return pd.concat(frames, ignore_index=True)


def build_report_payload(
    *,
    input_path: Path,
    input_files: list[Path],
    output_csv: Path,
    core_top_ratio: float,
    result: Any,
) -> dict[str, Any]:
    core_nodes = (
        result.repair_order[result.repair_order["is_core"]]["node_id"].astype(str).tolist()
        if not result.repair_order.empty
        else []
    )
    return {
        "script": "repair_plan.py",
        "input_path": str(input_path),
        "source_files": [str(path) for path in input_files],
        "repair_order_csv": str(output_csv),
        "core_top_ratio": float(core_top_ratio),
        "total_node_count": int(result.total_node_count),
        "anomalous_node_count": int(result.anomalous_node_count),
        "core_node_count": int(result.core_node_count),
        "core_ratio": float(result.core_ratio),
        "formula_denominator": float(result.denominator),
        "minimum_cost": float(result.cost),
        "core_nodes": core_nodes,
        "repair_order": result.repair_order.to_dict(orient="records"),
        "formula_interpretation": result.interpretation,
    }


def main() -> None:
    args = parse_args()
    validate_output_args(args.output_path, args.result_dir)
    input_files = discover_csv_inputs(args.input_path)
    output_csv, report_json = ensure_output_paths(
        args.output_path,
        args.report_path,
        input_path=args.input_path,
        result_dir=args.result_dir,
    )

    summary_df = load_summary_dataframe(input_files)
    result = build_repair_plan(summary_df, core_top_ratio=args.core_top_ratio)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.repair_order.to_csv(output_csv, index=False, encoding="utf-8-sig")

    report_payload = build_report_payload(
        input_path=args.input_path,
        input_files=input_files,
        output_csv=output_csv,
        core_top_ratio=args.core_top_ratio,
        result=result,
    )
    write_manifest(report_json, report_payload)

    print(f"Input files: {len(input_files)}")
    print(f"Repair order CSV: {output_csv}")
    print(f"Report JSON: {report_json}")
    print(f"Total nodes: {result.total_node_count}")
    print(f"Anomalous nodes: {result.anomalous_node_count}")
    print(f"Core nodes: {result.core_node_count}")
    print(f"Minimum cost: {result.cost:.10f}")


if __name__ == "__main__":
    main()
