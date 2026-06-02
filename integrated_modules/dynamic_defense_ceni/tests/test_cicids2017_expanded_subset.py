import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TARGET_LABELS = [
    "BENIGN",
    "DDoS",
    "DoS Hulk",
    "DoS GoldenEye",
    "DoS slowloris",
    "DoS Slowhttptest",
    "PortScan",
    "FTP-Patator",
    "SSH-Patator",
    "Heartbleed",
    "Web Attack Brute Force",
    "Web Attack XSS",
    "Web Attack Sql Injection",
]

MINORITY_LABELS = [
    "BENIGN",
    "Heartbleed",
    "Web Attack Sql Injection",
    "Web Attack XSS",
    "Web Attack Brute Force",
]


def _row(label, port):
    return {
        "Destination Port": port,
        "Flow Duration": 1000 + port,
        "Total Fwd Packets": 10,
        "Total Backward Packets": 5,
        "Total Length of Fwd Packets": 500,
        "Total Length of Bwd Packets": 250,
        "Fwd Packet Length Mean": 50.0,
        "Bwd Packet Length Mean": 25.0,
        "Flow Bytes/s": 1000.0,
        "Flow Packets/s": 30.0,
        "Label": label,
    }


def test_make_cicids2017_expanded_summary_structure(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    rows = []
    for idx, label in enumerate(TARGET_LABELS):
        for offset in range(3):
            rows.append(_row(label, 1000 + idx * 10 + offset))
    pd.DataFrame(rows).to_csv(raw_dir / "synthetic_cicids.csv", index=False)

    out_csv = tmp_path / "cicids2017_expanded_scenario_ordered.csv"
    summary_json = tmp_path / "cicids2017_expanded_summary.json"
    subprocess.check_call(
        [
            sys.executable,
            "scripts/make_cicids2017_subset.py",
            "--raw-dir",
            str(raw_dir),
            "--rows-per-class",
            "2",
            "--out",
            str(out_csv),
            "--summary-out",
            str(summary_json),
        ],
        cwd=str(ROOT),
    )

    assert out_csv.exists()
    assert summary_json.exists()
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["scenario"] == "cicids2017_expanded"
    assert summary["rows_per_class"] == 2
    assert summary["target_labels"] == TARGET_LABELS
    assert summary["ordered_by_stage"] is True
    assert summary["missing_labels"] == []
    assert summary["partial_labels"] == []
    assert summary["total_rows"] == len(TARGET_LABELS) * 2
    for label in TARGET_LABELS:
        assert summary["label_counts"][label] == 2

    out = pd.read_csv(out_csv)
    assert out["Label"].tolist() == [label for label in TARGET_LABELS for _ in range(2)]


def test_make_cicids2017_minority_summary_and_oversampling(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    rows = []
    for offset in range(3):
        rows.append(_row("BENIGN", 1000 + offset))
        rows.append(_row("Web Attack XSS", 2000 + offset))
        rows.append(_row("Web Attack Brute Force", 3000 + offset))
    rows.append(_row("Heartbleed", 4000))
    rows.append(_row("Web Attack Sql Injection", 5000))
    pd.DataFrame(rows).to_csv(raw_dir / "synthetic_minority.csv", index=False)

    out_csv = tmp_path / "cicids2017_minority_scenario_ordered.csv"
    summary_json = tmp_path / "cicids2017_minority_summary.json"
    subprocess.check_call(
        [
            sys.executable,
            "scripts/make_cicids2017_subset.py",
            "--raw-dir",
            str(raw_dir),
            "--scenario",
            "minority",
            "--rows-per-class",
            "3",
            "--out",
            str(out_csv),
            "--summary-out",
            str(summary_json),
        ],
        cwd=str(ROOT),
    )

    assert out_csv.exists()
    assert summary_json.exists()
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["scenario"] == "cicids2017_minority"
    assert summary["rows_per_class"] == 3
    assert summary["target_labels"] == MINORITY_LABELS
    assert summary["ordered_by_stage"] is True
    assert summary["missing_labels"] == []
    assert summary["partial_labels"] == []
    assert summary["total_rows"] == len(MINORITY_LABELS) * 3

    stats = summary["sampling_stats"]
    assert stats["Heartbleed"] == {"original_rows": 1, "sampled_rows": 3, "oversampled": True}
    assert stats["Web Attack Sql Injection"] == {"original_rows": 1, "sampled_rows": 3, "oversampled": True}
    assert stats["BENIGN"] == {"original_rows": 3, "sampled_rows": 3, "oversampled": False}
    assert stats["Web Attack XSS"] == {"original_rows": 3, "sampled_rows": 3, "oversampled": False}
    assert stats["Web Attack Brute Force"] == {"original_rows": 3, "sampled_rows": 3, "oversampled": False}

    out = pd.read_csv(out_csv)
    assert out["Label"].tolist() == [label for label in MINORITY_LABELS for _ in range(3)]
