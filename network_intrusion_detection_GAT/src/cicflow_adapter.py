from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


MACHINELEARNINGCVE_FEATURE_COLUMNS: list[str] = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow Bytes/s",
    "Flow Packets/s",
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
    "Fwd Header Length.1",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
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

RAW_CICFLOWMETER_TO_CANONICAL: dict[str, str] = {
    "dst_port": "Destination Port",
    "flow_duration": "Flow Duration",
    "tot_fwd_pkts": "Total Fwd Packets",
    "tot_bwd_pkts": "Total Backward Packets",
    "totlen_fwd_pkts": "Total Length of Fwd Packets",
    "totlen_bwd_pkts": "Total Length of Bwd Packets",
    "fwd_pkt_len_max": "Fwd Packet Length Max",
    "fwd_pkt_len_min": "Fwd Packet Length Min",
    "fwd_pkt_len_mean": "Fwd Packet Length Mean",
    "fwd_pkt_len_std": "Fwd Packet Length Std",
    "bwd_pkt_len_max": "Bwd Packet Length Max",
    "bwd_pkt_len_min": "Bwd Packet Length Min",
    "bwd_pkt_len_mean": "Bwd Packet Length Mean",
    "bwd_pkt_len_std": "Bwd Packet Length Std",
    "flow_byts_s": "Flow Bytes/s",
    "flow_pkts_s": "Flow Packets/s",
    "flow_iat_mean": "Flow IAT Mean",
    "flow_iat_std": "Flow IAT Std",
    "flow_iat_max": "Flow IAT Max",
    "flow_iat_min": "Flow IAT Min",
    "fwd_iat_tot": "Fwd IAT Total",
    "fwd_iat_mean": "Fwd IAT Mean",
    "fwd_iat_std": "Fwd IAT Std",
    "fwd_iat_max": "Fwd IAT Max",
    "fwd_iat_min": "Fwd IAT Min",
    "bwd_iat_tot": "Bwd IAT Total",
    "bwd_iat_mean": "Bwd IAT Mean",
    "bwd_iat_std": "Bwd IAT Std",
    "bwd_iat_max": "Bwd IAT Max",
    "bwd_iat_min": "Bwd IAT Min",
    "fwd_psh_flags": "Fwd PSH Flags",
    "bwd_psh_flags": "Bwd PSH Flags",
    "fwd_urg_flags": "Fwd URG Flags",
    "bwd_urg_flags": "Bwd URG Flags",
    "fwd_header_len": "Fwd Header Length",
    "bwd_header_len": "Bwd Header Length",
    "fwd_pkts_s": "Fwd Packets/s",
    "bwd_pkts_s": "Bwd Packets/s",
    "pkt_len_min": "Min Packet Length",
    "pkt_len_max": "Max Packet Length",
    "pkt_len_mean": "Packet Length Mean",
    "pkt_len_std": "Packet Length Std",
    "pkt_len_var": "Packet Length Variance",
    "fin_flag_cnt": "FIN Flag Count",
    "syn_flag_cnt": "SYN Flag Count",
    "rst_flag_cnt": "RST Flag Count",
    "psh_flag_cnt": "PSH Flag Count",
    "ack_flag_cnt": "ACK Flag Count",
    "urg_flag_cnt": "URG Flag Count",
    "cwr_flag_count": "CWE Flag Count",
    "ece_flag_cnt": "ECE Flag Count",
    "down_up_ratio": "Down/Up Ratio",
    "pkt_size_avg": "Average Packet Size",
    "fwd_seg_size_avg": "Avg Fwd Segment Size",
    "bwd_seg_size_avg": "Avg Bwd Segment Size",
    "fwd_byts_b_avg": "Fwd Avg Bytes/Bulk",
    "fwd_pkts_b_avg": "Fwd Avg Packets/Bulk",
    "fwd_blk_rate_avg": "Fwd Avg Bulk Rate",
    "bwd_byts_b_avg": "Bwd Avg Bytes/Bulk",
    "bwd_pkts_b_avg": "Bwd Avg Packets/Bulk",
    "bwd_blk_rate_avg": "Bwd Avg Bulk Rate",
    "subflow_fwd_pkts": "Subflow Fwd Packets",
    "subflow_fwd_byts": "Subflow Fwd Bytes",
    "subflow_bwd_pkts": "Subflow Bwd Packets",
    "subflow_bwd_byts": "Subflow Bwd Bytes",
    "init_fwd_win_byts": "Init_Win_bytes_forward",
    "init_bwd_win_byts": "Init_Win_bytes_backward",
    "fwd_act_data_pkts": "act_data_pkt_fwd",
    "fwd_seg_size_min": "min_seg_size_forward",
    "active_mean": "Active Mean",
    "active_std": "Active Std",
    "active_max": "Active Max",
    "active_min": "Active Min",
    "idle_mean": "Idle Mean",
    "idle_std": "Idle Std",
    "idle_max": "Idle Max",
    "idle_min": "Idle Min",
}

RAW_CICFLOWMETER_METADATA_COLUMNS: tuple[str, ...] = (
    "src_ip",
    "dst_ip",
    "src_port",
    "protocol",
    "timestamp",
)

NODE_METADATA_COLUMNS: tuple[str, ...] = RAW_CICFLOWMETER_METADATA_COLUMNS


def clean_column_names(columns: Iterable[object]) -> list[str]:
    return [str(column).strip() for column in columns]


def looks_like_cicflowmeter_export(columns: Iterable[object]) -> bool:
    normalized = set(clean_column_names(columns))
    return len(normalized.intersection(RAW_CICFLOWMETER_TO_CANONICAL)) >= 3


def adapt_cicflowmeter_schema(
    df: pd.DataFrame,
    *,
    label_column: str = "Label",
    drop_metadata: bool = False,
    drop_extra: bool = False,
    ensure_all_features: bool = False,
    reorder: bool = False,
) -> pd.DataFrame:
    frame = df.copy()
    frame.columns = clean_column_names(frame.columns)

    if ensure_all_features:
        for column in RAW_CICFLOWMETER_METADATA_COLUMNS:
            if column not in frame.columns and not drop_metadata:
                frame[column] = pd.Series(dtype=object)

    if label_column not in frame.columns:
        for alias in ("label", label_column.lower()):
            if alias in frame.columns:
                frame[label_column] = frame[alias]
                break

    for alias, canonical in RAW_CICFLOWMETER_TO_CANONICAL.items():
        if alias in frame.columns and canonical not in frame.columns:
            frame[canonical] = frame[alias]

    if "Fwd Header Length" in frame.columns and "Fwd Header Length.1" not in frame.columns:
        frame["Fwd Header Length.1"] = frame["Fwd Header Length"]

    if ensure_all_features:
        for column in MACHINELEARNINGCVE_FEATURE_COLUMNS:
            if column not in frame.columns:
                frame[column] = np.nan

    if drop_metadata:
        metadata_to_drop = [column for column in RAW_CICFLOWMETER_METADATA_COLUMNS if column in frame.columns]
        if metadata_to_drop:
            frame = frame.drop(columns=metadata_to_drop)

    metadata_columns = [
        column for column in RAW_CICFLOWMETER_METADATA_COLUMNS if column in frame.columns
    ]
    supported_columns = [
        column for column in MACHINELEARNINGCVE_FEATURE_COLUMNS if column in frame.columns
    ]
    ordered_columns = metadata_columns + supported_columns
    if label_column in frame.columns:
        ordered_columns = ordered_columns + [label_column]

    if drop_extra:
        return frame[ordered_columns]

    if reorder:
        extra_columns = [column for column in frame.columns if column not in ordered_columns]
        return frame[ordered_columns + extra_columns]

    return frame


def split_metadata_and_feature_columns(
    df: pd.DataFrame,
    *,
    label_column: str = "Label",
) -> tuple[list[str], list[str]]:
    metadata_columns = [column for column in NODE_METADATA_COLUMNS if column in df.columns]
    feature_columns = [
        column
        for column in df.columns
        if column not in metadata_columns and column != label_column
    ]
    return metadata_columns, feature_columns
