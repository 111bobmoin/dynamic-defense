from pathlib import Path
import argparse
import json
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dynamic_defense.evaluation import strategy_family_label


BASIC_FEATURE_COLUMNS = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Mean",
    "Flow Bytes/s",
    "Flow Packets/s",
]

EXTENDED_CANDIDATE_COLUMNS = BASIC_FEATURE_COLUMNS + [
    "Total Fwd Packets",
    "Total Backward Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWE Flag Count",
    "ECE Flag Count",
    "Down/Up Ratio",
    "Average Packet Size",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Packets",
    "Subflow Bwd Bytes",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
]

EXCLUDED_ALL_NUMERIC_COLUMNS = {
    "Flow ID",
    "Source IP",
    "Src IP",
    "Destination IP",
    "Dst IP",
    "Timestamp",
    "Label",
}


class FlowMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for _ in range(max(1, int(num_layers))):
            layers.append(nn.Linear(prev_dim, int(hidden_dim)))
            layers.append(nn.ReLU())
            if float(dropout) > 0.0:
                layers.append(nn.Dropout(float(dropout)))
            prev_dim = int(hidden_dim)
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _dedupe(items):
    out = []
    for item in items:
        if item not in out:
            out.append(item)
    return out


def select_feature_columns(df: pd.DataFrame, feature_set: str):
    if feature_set == "basic":
        missing = [c for c in BASIC_FEATURE_COLUMNS + ["Label"] if c not in df.columns]
        if missing:
            raise RuntimeError("Missing columns: %s" % missing)
        return list(BASIC_FEATURE_COLUMNS)

    if "Label" not in df.columns:
        raise RuntimeError("Missing columns: ['Label']")

    if feature_set == "extended":
        return [c for c in _dedupe(EXTENDED_CANDIDATE_COLUMNS) if c in df.columns]

    if feature_set == "all_numeric":
        columns = []
        excluded = {c.lower() for c in EXCLUDED_ALL_NUMERIC_COLUMNS}
        for col in df.columns:
            clean = str(col).strip()
            if clean.lower() in excluded:
                continue
            numeric = pd.to_numeric(df[col], errors="coerce")
            if numeric.notna().any():
                columns.append(clean)
        return columns

    raise RuntimeError("unsupported feature_set: %s" % feature_set)


def load_dataset(path: str, feature_set: str):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    feature_columns = select_feature_columns(df, feature_set)
    if not feature_columns:
        raise RuntimeError("no usable numeric feature columns for feature_set=%s" % feature_set)

    data = pd.DataFrame(index=df.index)
    for col in feature_columns:
        data[col] = pd.to_numeric(df[col], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    labels = df["Label"].astype(str).str.strip()
    keep = labels != ""
    data = data.loc[keep]
    labels = labels.loc[keep]

    x = data.astype("float32").values
    y = labels.values
    return x, y, feature_columns


def prepare_labels(y_raw, label_mode: str):
    labels = np.asarray([str(label).strip() for label in y_raw], dtype=object)
    original_label_count = {str(k): int(v) for k, v in pd.Series(labels).value_counts().sort_index().to_dict().items()}

    if label_mode == "exact":
        training_labels = labels
        keep_mask = np.ones(len(labels), dtype=bool)
        dropped_unknown_rows = 0
        label_mapping_summary = {
            label: {
                "mapped_label": label,
                "rows": count,
                "kept": True,
            }
            for label, count in original_label_count.items()
        }
    elif label_mode == "family":
        mapped_labels = np.asarray([strategy_family_label(label) for label in labels], dtype=object)
        keep_mask = mapped_labels != "UNKNOWN"
        dropped_unknown_rows = int((~keep_mask).sum())
        training_labels = mapped_labels[keep_mask]
        label_mapping_summary = {}
        for label, count in original_label_count.items():
            mapped = strategy_family_label(label)
            label_mapping_summary[label] = {
                "mapped_label": mapped,
                "rows": int(count),
                "kept": mapped != "UNKNOWN",
            }
    else:
        raise RuntimeError("unsupported label_mode: %s" % label_mode)

    if len(training_labels) == 0:
        raise RuntimeError("no training labels remain after label_mode=%s normalization" % label_mode)

    training_label_count = {
        str(k): int(v)
        for k, v in pd.Series(training_labels).value_counts().sort_index().to_dict().items()
    }
    return training_labels, keep_mask, {
        "original_label_count": original_label_count,
        "training_label_count": training_label_count,
        "label_mapping_summary": label_mapping_summary,
        "dropped_unknown_rows": dropped_unknown_rows,
    }


def make_class_weight(y_train, num_classes: int, mode: str):
    if mode == "none":
        return None
    counts = np.bincount(y_train, minlength=num_classes).astype("float32")
    counts[counts == 0] = 1.0
    weights = float(len(y_train)) / (float(num_classes) * counts)
    return torch.tensor(weights, dtype=torch.float32)


def predict_labels(model, x_tensor, device):
    model.eval()
    with torch.no_grad():
        logits = model(x_tensor.to(device))
        return logits.argmax(dim=1).cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/cicids2017_subset/cicids2017_expanded_scenario_ordered.csv",
        help="Training CSV, for example the expanded ordered CICIDS2017 scenario",
    )
    parser.add_argument("--model-out", default="models/torch_flow_classifier.pt")
    parser.add_argument("--meta-out", default="models/torch_flow_classifier_meta.json")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--feature-set", choices=["basic", "extended", "all_numeric"], default="basic")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="none")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-mode", choices=["exact", "family"], default="exact")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cpu")

    x, y_raw, feature_columns = load_dataset(args.input, args.feature_set)
    y_labels, label_keep_mask, label_meta = prepare_labels(y_raw, args.label_mode)
    if not label_keep_mask.all():
        x = x[label_keep_mask]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_labels)

    scaler = StandardScaler()
    x = scaler.fit_transform(x).astype("float32")

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=args.seed,
        stratify=y,
    )

    train_x = torch.tensor(x_train, dtype=torch.float32)
    train_y = torch.tensor(y_train, dtype=torch.long)
    test_x = torch.tensor(x_test, dtype=torch.float32)
    test_y = torch.tensor(y_test, dtype=torch.long)

    model = FlowMLP(
        input_dim=train_x.shape[1],
        num_classes=len(label_encoder.classes_),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    class_weight = make_class_weight(y_train, len(label_encoder.classes_), args.class_weight)
    criterion = nn.CrossEntropyLoss(weight=class_weight.to(device) if class_weight is not None else None)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    num_samples = train_x.shape[0]
    best_acc = -1.0
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        indices = torch.randperm(num_samples)
        total_loss = 0.0

        for start in range(0, num_samples, args.batch_size):
            batch_idx = indices[start:start + args.batch_size]
            bx = train_x[batch_idx].to(device)
            by = train_y[batch_idx].to(device)

            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_idx)

        pred = predict_labels(model, test_x, device)
        acc = accuracy_score(y_test, pred)
        if acc > best_acc:
            best_acc = float(acc)
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print("epoch=%03d loss=%.4f test_acc=%.4f best_acc=%.4f" % (epoch, total_loss / num_samples, acc, best_acc))

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print("early stopping at epoch=%03d best_epoch=%03d best_acc=%.4f" % (epoch, best_epoch, best_acc))
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_pred = predict_labels(model, train_x, device)
    test_pred = predict_labels(model, test_x, device)
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    print("\nBest epoch:", best_epoch)
    print("Train accuracy:", train_acc)
    print("Test accuracy:", test_acc)
    print(classification_report(y_test, test_pred, target_names=label_encoder.classes_))

    model_path = Path(args.model_out)
    meta_path = Path(args.meta_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), model_path)

    meta = {
        "model_type": "FlowMLP",
        "feature_set": args.feature_set,
        "label_mode": args.label_mode,
        "feature_columns": feature_columns,
        "labels": label_encoder.classes_.tolist(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "input_dim": len(feature_columns),
        "num_classes": len(label_encoder.classes_),
        "hidden_dim": int(args.hidden_dim),
        "num_layers": int(args.num_layers),
        "dropout": float(args.dropout),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "batch_size": int(args.batch_size),
        "class_weight": args.class_weight,
        "seed": int(args.seed),
        "best_epoch": int(best_epoch),
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "device": "cpu",
        "accuracy": float(test_acc),
    }
    meta.update(label_meta)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nSaved model: %s" % model_path)
    print("Saved meta: %s" % meta_path)


if __name__ == "__main__":
    main()
