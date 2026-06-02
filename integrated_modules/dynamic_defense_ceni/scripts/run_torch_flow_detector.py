import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_recall_fscore_support

from src.dynamic_defense.evaluation import strategy_family_label
from src.dynamic_defense.torch_detector import TorchFlowDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", default="models/torch_flow_classifier.pt")
    parser.add_argument("--meta", default="models/torch_flow_classifier_meta.json")
    parser.add_argument("--out-csv", default="reports/torch_detector_report.csv")
    parser.add_argument("--out-json", default="reports/torch_detector_summary.json")
    args = parser.parse_args()

    df = pd.read_csv(args.input, nrows=args.limit)
    df.columns = [c.strip() for c in df.columns]

    detector = TorchFlowDetector(model_path=args.model, meta_path=args.meta, device="cpu")
    pred = detector.predict_dataframe(df)

    out = pd.concat([df[["Label"]].reset_index(drop=True), pred], axis=1)
    out["Label"] = out["Label"].astype(str).str.strip()

    label_mode = detector.meta.get("label_mode", "exact")
    metric_out = out
    y_true_column = "Label"
    dropped_unknown_rows = 0
    if label_mode == "family":
        out["evaluation_label"] = out["Label"].map(strategy_family_label)
        metric_out = out[out["evaluation_label"] != "UNKNOWN"].copy()
        y_true_column = "evaluation_label"
        dropped_unknown_rows = int(len(out) - len(metric_out))

    accuracy = accuracy_score(metric_out[y_true_column], metric_out["torch_predicted_label"]) if len(metric_out) else 0.0
    macro_f1 = f1_score(
        metric_out[y_true_column],
        metric_out["torch_predicted_label"],
        labels=detector.labels,
        average="macro",
        zero_division=0,
    ) if len(metric_out) else 0.0
    weighted_f1 = f1_score(
        metric_out[y_true_column],
        metric_out["torch_predicted_label"],
        labels=detector.labels,
        average="weighted",
        zero_division=0,
    ) if len(metric_out) else 0.0
    labels = detector.labels
    precision, recall, f1, support = precision_recall_fscore_support(
        metric_out[y_true_column],
        metric_out["torch_predicted_label"],
        labels=labels,
        zero_division=0,
    )
    per_class = {
        label: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, label in enumerate(labels)
    }
    cm = confusion_matrix(metric_out[y_true_column], metric_out["torch_predicted_label"], labels=labels)

    summary = {
        "rows": int(len(out)),
        "evaluation_rows": int(len(metric_out)),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "label_mode": label_mode,
        "labels": labels,
        "per_class": per_class,
        "classification_report": classification_report(
            metric_out[y_true_column],
            metric_out["torch_predicted_label"],
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": {
            "labels": labels,
            "matrix": cm.astype(int).tolist(),
        },
        "predicted_counts": out["torch_predicted_label"].value_counts().to_dict(),
        "true_counts": metric_out[y_true_column].value_counts().to_dict(),
        "model": args.model,
        "meta": args.meta,
        "output_csv": args.out_csv,
    }
    if label_mode == "family":
        summary["note"] = "family-level classifier output is intended for strategy routing, not exact CICIDS subclass reporting"
        summary["dropped_unknown_rows"] = dropped_unknown_rows

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    Path(args.out_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n=== classification report ===")
    print(classification_report(metric_out[y_true_column], metric_out["torch_predicted_label"], labels=labels, zero_division=0))
    print("\n=== confusion matrix ===")
    print(pd.DataFrame(
        cm,
        index=[f"true_{x}" for x in labels],
        columns=[f"pred_{x}" for x in labels],
    ))


if __name__ == "__main__":
    main()
