import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.dynamic_defense.torch_detector import TorchFlowDetector


ROOT = Path(__file__).resolve().parents[1]


FEATURES = [
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
    "SYN Flag Count",
    "ACK Flag Count",
    "Packet Length Mean",
]


def _make_training_csv(path: Path):
    rows = []
    labels = [("BENIGN", 10.0), ("DDoS", 100.0), ("PortScan", 200.0)]
    for label, base in labels:
        for idx in range(18):
            row = {feature: base + idx * 0.1 + col_idx for col_idx, feature in enumerate(FEATURES)}
            row["Flow ID"] = "flow-%s-%s" % (label, idx)
            row["Source IP"] = "10.0.0.%s" % (idx + 1)
            row["Destination IP"] = "10.0.1.%s" % (idx + 1)
            row["Timestamp"] = "2026-01-01 00:00:%02d" % idx
            row["Label"] = label
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def _make_family_training_csv(path: Path):
    rows = []
    labels = [
        ("BENIGN", 10.0),
        ("DoS Hulk", 100.0),
        ("DDoS", 110.0),
        ("FTP-Patator", 200.0),
        ("SSH-Patator", 210.0),
        ("Web Attack Brute Force", 300.0),
        ("Web Attack XSS", 310.0),
        ("Web Attack Sql Injection", 320.0),
        ("Infiltration", 900.0),
    ]
    for label, base in labels:
        for idx in range(10):
            row = {feature: base + idx * 0.1 + col_idx for col_idx, feature in enumerate(FEATURES)}
            row["Flow ID"] = "flow-%s-%s" % (label, idx)
            row["Source IP"] = "10.0.0.%s" % (idx + 1)
            row["Destination IP"] = "10.0.1.%s" % (idx + 1)
            row["Timestamp"] = "2026-01-01 00:01:%02d" % idx
            row["Label"] = label
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_torch_detector_loads_legacy_basic_meta(tmp_path):
    torch = pytest.importorskip("torch")
    from src.dynamic_defense.torch_detector import FlowMLP

    model_path = tmp_path / "legacy.pt"
    meta_path = tmp_path / "legacy_meta.json"
    model = FlowMLP(input_dim=2, num_classes=2, legacy=True)
    torch.save(model.state_dict(), str(model_path))
    meta_path.write_text(
        json.dumps(
            {
                "model_type": "FlowMLP",
                "feature_columns": ["a", "b"],
                "labels": ["BENIGN", "DDoS"],
                "scaler_mean": [0.0, 0.0],
                "scaler_scale": [1.0, 1.0],
                "input_dim": 2,
                "num_classes": 2,
                "device": "cpu",
                "accuracy": 0.5,
            }
        ),
        encoding="utf-8",
    )

    detector = TorchFlowDetector(model_path=str(model_path), meta_path=str(meta_path), device="cpu")
    pred = detector.predict_dataframe(pd.DataFrame([{"a": 1.0, "b": 2.0}]))
    assert list(pred.columns) == ["row_id", "torch_predicted_label", "torch_confidence"]
    assert pred.iloc[0]["torch_predicted_label"] in {"BENIGN", "DDoS"}


def test_extended_training_meta_and_detector_metrics(tmp_path):
    pytest.importorskip("torch")
    csv_path = tmp_path / "flows.csv"
    model_path = tmp_path / "flow_model.pt"
    meta_path = tmp_path / "flow_meta.json"
    report_csv = tmp_path / "torch_detector_report.csv"
    report_json = tmp_path / "torch_detector_summary.json"
    _make_training_csv(csv_path)

    subprocess.check_call(
        [
            sys.executable,
            "scripts/train_torch_flow_classifier.py",
            "--input",
            str(csv_path),
            "--model-out",
            str(model_path),
            "--meta-out",
            str(meta_path),
            "--feature-set",
            "extended",
            "--hidden-dim",
            "16",
            "--num-layers",
            "1",
            "--dropout",
            "0.0",
            "--epochs",
            "8",
            "--patience",
            "4",
            "--batch-size",
            "16",
            "--lr",
            "0.01",
            "--class-weight",
            "balanced",
            "--seed",
            "7",
        ],
        cwd=str(ROOT),
    )

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["feature_set"] == "extended"
    assert meta["label_mode"] == "exact"
    assert meta["hidden_dim"] == 16
    assert meta["num_layers"] == 1
    assert meta["dropout"] == 0.0
    assert meta["lr"] == 0.01
    assert meta["weight_decay"] == 0.0
    assert meta["batch_size"] == 16
    assert meta["class_weight"] == "balanced"
    assert meta["seed"] == 7
    assert "best_epoch" in meta
    assert "train_accuracy" in meta
    assert "test_accuracy" in meta
    assert "SYN Flag Count" in meta["feature_columns"]

    subprocess.check_call(
        [
            sys.executable,
            "scripts/run_torch_flow_detector.py",
            "--input",
            str(csv_path),
            "--model",
            str(model_path),
            "--meta",
            str(meta_path),
            "--out-csv",
            str(report_csv),
            "--out-json",
            str(report_json),
        ],
        cwd=str(ROOT),
    )

    summary = json.loads(report_json.read_text(encoding="utf-8"))
    assert report_csv.exists()
    assert "macro_f1" in summary
    assert "weighted_f1" in summary
    assert "per_class" in summary
    assert "confusion_matrix" in summary


def test_family_label_mode_training_and_detector_summary(tmp_path):
    pytest.importorskip("torch")
    csv_path = tmp_path / "family_flows.csv"
    model_path = tmp_path / "family_flow_model.pt"
    meta_path = tmp_path / "family_flow_meta.json"
    report_csv = tmp_path / "family_torch_detector_report.csv"
    report_json = tmp_path / "family_torch_detector_summary.json"
    _make_family_training_csv(csv_path)

    subprocess.check_call(
        [
            sys.executable,
            "scripts/train_torch_flow_classifier.py",
            "--input",
            str(csv_path),
            "--model-out",
            str(model_path),
            "--meta-out",
            str(meta_path),
            "--feature-set",
            "extended",
            "--label-mode",
            "family",
            "--hidden-dim",
            "12",
            "--num-layers",
            "1",
            "--dropout",
            "0.0",
            "--epochs",
            "6",
            "--patience",
            "3",
            "--batch-size",
            "16",
            "--lr",
            "0.01",
            "--class-weight",
            "balanced",
            "--seed",
            "11",
        ],
        cwd=str(ROOT),
    )

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["label_mode"] == "family"
    assert meta["label_mapping_summary"]["Web Attack XSS"]["mapped_label"] == "Web Attack"
    assert meta["label_mapping_summary"]["DoS Hulk"]["mapped_label"] == "DDoS"
    assert meta["label_mapping_summary"]["Infiltration"]["mapped_label"] == "UNKNOWN"
    assert meta["dropped_unknown_rows"] == 10
    assert "Infiltration" in meta["original_label_count"]
    assert "UNKNOWN" not in meta["training_label_count"]
    assert "Web Attack XSS" not in meta["labels"]
    assert {"BENIGN", "DDoS", "Brute Force", "Web Attack"}.issubset(set(meta["labels"]))

    subprocess.check_call(
        [
            sys.executable,
            "scripts/run_torch_flow_detector.py",
            "--input",
            str(csv_path),
            "--model",
            str(model_path),
            "--meta",
            str(meta_path),
            "--out-csv",
            str(report_csv),
            "--out-json",
            str(report_json),
        ],
        cwd=str(ROOT),
    )

    summary = json.loads(report_json.read_text(encoding="utf-8"))
    assert report_csv.exists()
    assert summary["label_mode"] == "family"
    assert summary["dropped_unknown_rows"] == 10
    assert "strategy routing" in summary["note"]
    assert "Web Attack" in summary["true_counts"]
    assert "Web Attack XSS" not in summary["true_counts"]
