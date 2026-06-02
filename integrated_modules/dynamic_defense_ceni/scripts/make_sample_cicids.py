#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make_rows(label: str, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    base = {
        "BENIGN": dict(duration=70000, fwd=8, bwd=7, fwd_len=500, bwd_len=700, flow_bytes=3000, flow_pkts=80, syn=0, ack=1, port=80),
        "DDoS": dict(duration=2000, fwd=80, bwd=3, fwd_len=8000, bwd_len=200, flow_bytes=200000, flow_pkts=5000, syn=10, ack=0, port=80),
        "PortScan": dict(duration=500, fwd=2, bwd=1, fwd_len=80, bwd_len=40, flow_bytes=300, flow_pkts=300, syn=1, ack=0, port=1024),
        "SSH-Patator": dict(duration=5000, fwd=15, bwd=8, fwd_len=900, bwd_len=400, flow_bytes=5000, flow_pkts=200, syn=2, ack=1, port=22),
    }[label]
    rows = []
    for _ in range(n):
        rows.append(
            {
                "Destination Port": max(1, int(rng.normal(base["port"], 5 if label != "PortScan" else 1000))),
                "Flow Duration": max(1, rng.normal(base["duration"], base["duration"] * 0.1 + 1)),
                "Total Fwd Packets": max(0, int(rng.normal(base["fwd"], 3))),
                "Total Backward Packets": max(0, int(rng.normal(base["bwd"], 2))),
                "Total Length of Fwd Packets": max(0, rng.normal(base["fwd_len"], base["fwd_len"] * 0.15 + 1)),
                "Total Length of Bwd Packets": max(0, rng.normal(base["bwd_len"], base["bwd_len"] * 0.15 + 1)),
                "Fwd Packet Length Max": max(0, rng.normal(base["fwd_len"] / max(base["fwd"], 1), 20)),
                "Bwd Packet Length Max": max(0, rng.normal(base["bwd_len"] / max(base["bwd"], 1), 20)),
                "Flow Bytes/s": max(0, rng.normal(base["flow_bytes"], base["flow_bytes"] * 0.2 + 1)),
                "Flow Packets/s": max(0, rng.normal(base["flow_pkts"], base["flow_pkts"] * 0.2 + 1)),
                "SYN Flag Count": max(0, int(rng.normal(base["syn"], 1))),
                "ACK Flag Count": max(0, int(rng.normal(base["ack"], 1))),
                "Label": label,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/sample_cicids.csv")
    args = parser.parse_args()
    df = pd.concat(
        [
            make_rows("BENIGN", 200, 1),
            make_rows("DDoS", 200, 2),
            make_rows("PortScan", 160, 3),
            make_rows("SSH-Patator", 140, 4),
        ],
        ignore_index=True,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(args.out)


if __name__ == "__main__":
    main()
