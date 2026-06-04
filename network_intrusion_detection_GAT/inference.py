from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from pandas.errors import EmptyDataError

from src.cicflow_adapter import adapt_cicflowmeter_schema, split_metadata_and_feature_columns
from src.data import LABEL_COLUMN, clean_columns
from src.graph import build_knn_edge_index
from src.metrics import classification_metrics
from src.model import IntrusionGAT
from src.output_layout import default_stage_filename, make_stage_dir, write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run anomaly detection inference with a trained GAT/GATv2 model on CSV files."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to a trained model.pt. If omitted, use the latest model under outputs/.",
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="CSV file or directory of CSV files that follow the training column format.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output CSV path. Default: outputs/results/<timestamp>_<input>/inference/<input>_predictions.csv",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=None,
        help="Dedicated result directory that stores inference CSVs under <result-dir>/inference/.",
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
        help="CSV chunk size when streaming input files.",
    )
    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=0.55,
        help="Anomaly-score threshold used to derive is_anomaly. Default: 0.55",
    )
    return parser.parse_args()


def find_latest_model(outputs_root: Path) -> Path:
    candidates = sorted(outputs_root.rglob("model.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No model.pt found under {outputs_root}")
    return candidates[0]


def load_checkpoint(model_path: Path, device: str) -> dict[str, Any]:
    return torch.load(model_path, map_location=device)


def infer_task_from_class_names(class_names: list[str]) -> str:
    classes = set(class_names)
    if classes == {"ATTACK", "BENIGN"}:
        return "binary"
    return "multiclass"


def normalize_label(label: str, task: str) -> str:
    value = str(label).strip()
    if task == "binary":
        return "BENIGN" if value.upper() == "BENIGN" else "ATTACK"
    return value


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


def clean_inference_chunk(chunk: pd.DataFrame, task: str) -> pd.DataFrame:
    chunk = chunk.copy()
    chunk.columns = clean_columns(chunk.columns)
    chunk = adapt_cicflowmeter_schema(chunk, label_column=LABEL_COLUMN, reorder=True)
    if LABEL_COLUMN in chunk.columns:
        chunk[LABEL_COLUMN] = chunk[LABEL_COLUMN].astype(str).map(lambda value: normalize_label(value, task))
    _, feature_columns = split_metadata_and_feature_columns(chunk, label_column=LABEL_COLUMN)
    for column in feature_columns:
        chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
    chunk = chunk.replace([np.inf, -np.inf], np.nan)
    return chunk


def build_model_from_checkpoint(checkpoint: dict[str, Any], device: str) -> IntrusionGAT:
    config = checkpoint["config"]
    feature_names = checkpoint["selected_feature_names"]
    model = IntrusionGAT(
        input_dim=len(feature_names),
        hidden_dim=int(config["hidden_dim"]),
        num_classes=len(checkpoint["class_names"]),
        heads=int(config["heads"]),
        dropout=float(config["dropout"]),
        model_name=str(config.get("model_name", "gat")),
        use_residual=bool(config.get("use_residual", False)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def ensure_output_path(output_path: Path | None, input_path: Path, result_dir: Path | None) -> Path:
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path
    stage_dir = make_stage_dir(input_path=input_path, stage="inference", result_dir=result_dir)
    return stage_dir / default_stage_filename(input_path, "predictions")


def predict_dataframe(
    df: pd.DataFrame,
    model: IntrusionGAT,
    checkpoint: dict[str, Any],
    device: str,
    batch_size: int,
    decision_threshold: float | None = None,
) -> pd.DataFrame:
    class_names = [str(name) for name in checkpoint["class_names"]]
    feature_names = [str(name) for name in checkpoint["selected_feature_names"]]
    if df.empty:
        result = df.copy()
        result["predicted_label"] = pd.Series(dtype=object)
        result["anomaly_score"] = pd.Series(dtype=float)
        result["is_anomaly"] = pd.Series(dtype=bool)
        return result

    mean = checkpoint["mean"].to(device=device, dtype=torch.float32)
    std = checkpoint["std"].to(device=device, dtype=torch.float32)
    config = checkpoint["config"]
    graph_metric = str(config.get("graph_metric", "cosine"))
    graph_strategy = str(config.get("graph_strategy", "knn"))
    benign_index = class_names.index("BENIGN") if "BENIGN" in class_names else None

    missing = [name for name in feature_names if name not in df.columns]
    if missing:
        raise KeyError(f"Missing required feature columns: {missing}")

    feature_frame = df[feature_names].copy()
    medians = feature_frame.median()
    feature_frame = feature_frame.fillna(medians)
    x = torch.tensor(feature_frame.to_numpy(dtype=np.float32), dtype=torch.float32, device=device)
    x = (x - mean) / std

    all_pred_indices: list[torch.Tensor] = []
    all_anomaly_scores: list[torch.Tensor] = []

    with torch.no_grad():
        for start in range(0, x.size(0), batch_size):
            end = min(start + batch_size, x.size(0))
            x_batch = x[start:end]
            edge_index = build_knn_edge_index(
                x_batch,
                k=int(config.get("k_neighbors", 10)),
                metric=graph_metric,
                strategy=graph_strategy,
            ).to(device)
            logits = model(x_batch, edge_index)
            probs = torch.softmax(logits, dim=1)
            pred_indices = probs.argmax(dim=1)
            if benign_index is not None:
                anomaly_scores = 1.0 - probs[:, benign_index]
            elif "ATTACK" in class_names:
                anomaly_scores = probs[:, class_names.index("ATTACK")]
            else:
                anomaly_scores = probs.max(dim=1).values

            all_pred_indices.append(pred_indices.cpu())
            all_anomaly_scores.append(anomaly_scores.cpu())

    pred_indices = torch.cat(all_pred_indices, dim=0)
    anomaly_scores = torch.cat(all_anomaly_scores, dim=0)
    predicted_labels = [class_names[index] for index in pred_indices.tolist()]
    if decision_threshold is not None:
        is_anomaly = (anomaly_scores.numpy() >= decision_threshold).tolist()
    else:
        is_anomaly = [label != "BENIGN" if "BENIGN" in class_names else True for label in predicted_labels]

    result = df.copy()
    result["predicted_label"] = predicted_labels
    result["anomaly_score"] = anomaly_scores.numpy()
    result["is_anomaly"] = is_anomaly
    return result


def build_empty_prediction_frame(
    input_csv: Path,
    *,
    task: str,
    model: IntrusionGAT,
    checkpoint: dict[str, Any],
    device: str,
    batch_size: int,
    decision_threshold: float | None,
) -> pd.DataFrame:
    try:
        empty_chunk = pd.read_csv(input_csv, nrows=0, low_memory=False)
    except EmptyDataError:
        empty_chunk = pd.DataFrame()
    cleaned = clean_inference_chunk(empty_chunk, task=task)
    return predict_dataframe(
        df=cleaned,
        model=model,
        checkpoint=checkpoint,
        device=device,
        batch_size=batch_size,
        decision_threshold=decision_threshold,
    )


def evaluate_if_possible(df: pd.DataFrame, class_names: list[str]) -> dict[str, float] | None:
    if df.empty:
        return None
    if LABEL_COLUMN not in df.columns:
        return None
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    if not set(df[LABEL_COLUMN].astype(str)).issubset(class_to_idx):
        return None
    y_true = torch.tensor([class_to_idx[label] for label in df[LABEL_COLUMN].astype(str).tolist()], dtype=torch.long)
    y_pred = torch.tensor([class_to_idx[label] for label in df["predicted_label"].astype(str).tolist()], dtype=torch.long)
    return {
        key: float(value)
        for key, value in classification_metrics(y_true, y_pred, num_classes=len(class_names)).items()
    }


def main() -> None:
    args = parse_args()
    validate_output_args(args.output_path, args.result_dir)
    model_path = args.model_path or find_latest_model(Path("outputs"))
    checkpoint = load_checkpoint(model_path, args.device)
    model = build_model_from_checkpoint(checkpoint, args.device)
    class_names = [str(name) for name in checkpoint["class_names"]]
    task = infer_task_from_class_names(class_names)

    output_path = ensure_output_path(args.output_path, args.input_path, args.result_dir)
    csv_files = discover_csv_inputs(args.input_path)
    results: list[pd.DataFrame] = []

    for csv_path in csv_files:
        file_results: list[pd.DataFrame] = []
        for chunk in pd.read_csv(csv_path, chunksize=args.chunksize, low_memory=False):
            cleaned = clean_inference_chunk(chunk, task=task)
            predicted = predict_dataframe(
                df=cleaned,
                model=model,
                checkpoint=checkpoint,
                device=args.device,
                batch_size=args.batch_size,
                decision_threshold=args.decision_threshold,
            )
            predicted.insert(0, "source_file", csv_path.name)
            file_results.append(predicted)
        if not file_results:
            empty_predicted = build_empty_prediction_frame(
                csv_path,
                task=task,
                model=model,
                checkpoint=checkpoint,
                device=args.device,
                batch_size=args.batch_size,
                decision_threshold=args.decision_threshold,
            )
            empty_predicted.insert(0, "source_file", csv_path.name)
            file_results.append(empty_predicted)
        results.extend(file_results)

    if not results:
        raise RuntimeError(f"No rows produced from input path: {args.input_path}")

    merged = pd.concat(results, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")

    metrics = evaluate_if_possible(merged, class_names=class_names)
    if args.output_path is None:
        write_manifest(
            output_path.parent.parent / "inference_manifest.json",
            {
                "script": "inference.py",
                "input_path": str(args.input_path),
                "model_path": str(model_path),
                "output_root": str(output_path.parent.parent),
                "prediction_csv": str(output_path),
                "source_files": [str(path) for path in csv_files],
                "rows_processed": int(len(merged)),
                "decision_threshold": args.decision_threshold,
                "evaluation_metrics": metrics,
            },
        )
    print(f"Model path: {model_path}")
    print(f"Output CSV: {output_path}")
    print(f"Rows processed: {len(merged)}")
    if args.decision_threshold is not None:
        print(f"Decision threshold: {args.decision_threshold}")
    if metrics is not None:
        print("Evaluation metrics:")
        print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
