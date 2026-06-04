from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd


ROLE_SEVERITY = {
    "uncertain": 0,
    "suspected_victim": 1,
    "suspected_attacker": 2,
    "suspected_compromised_host": 3,
}
ROLE_PRIORITY = {
    "uncertain": 0.0,
    "suspected_victim": 0.75,
    "suspected_attacker": 0.85,
    "suspected_compromised_host": 1.0,
}
REQUIRED_SUMMARY_COLUMNS = {
    "node_id",
    "node_role",
    "attacker_score",
    "victim_score",
    "compromised_score",
    "anomaly_ratio",
    "avg_anomaly_score",
    "max_anomaly_score",
    "total_flows",
    "total_anomalous_flows",
    "outbound_unique_targets",
    "inbound_unique_sources",
    "outbound_unique_target_ports",
}
REPAIR_ORDER_COLUMNS = [
    "repair_rank",
    "node_id",
    "node_role",
    "is_core",
    "role_priority",
    "damage_score",
    "structural_score",
    "core_score",
    "repair_priority_score",
    "factorial_rank",
    "rank_contribution",
    "cumulative_contribution",
    "remaining_core_after_repair",
    "top_predicted_labels",
]


@dataclass
class RepairPlanResult:
    aggregated_nodes: pd.DataFrame
    repair_order: pd.DataFrame
    total_node_count: int
    anomalous_node_count: int
    core_node_count: int
    core_ratio: float
    cost: float
    denominator: float
    interpretation: str


def validate_summary_dataframe(df: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_SUMMARY_COLUMNS.difference(df.columns))
    if missing:
        raise KeyError(f"Missing required columns for repair planning: {missing}")


def normalize_series(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    if values.empty:
        return values
    min_value = float(values.min())
    max_value = float(values.max())
    if max_value - min_value <= 1e-12:
        fill_value = 1.0 if max_value > 0 else 0.0
        return pd.Series(fill_value, index=values.index, dtype=float)
    return (values - min_value) / (max_value - min_value)


def _best_node_row(group: pd.DataFrame) -> pd.Series:
    ranked = group.assign(
        role_severity=group["node_role"].astype(str).map(lambda value: ROLE_SEVERITY.get(value, 0))
    ).sort_values(
        by=[
            "role_severity",
            "compromised_score",
            "attacker_score",
            "victim_score",
            "anomaly_ratio",
            "total_anomalous_flows",
            "max_anomaly_score",
        ],
        ascending=[False, False, False, False, False, False, False],
    )
    return ranked.iloc[0]


def aggregate_node_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    validate_summary_dataframe(summary_df)
    if summary_df.empty:
        return pd.DataFrame(
            columns=[
                "node_id",
                "node_role",
                "role_severity",
                "role_priority",
                "window_count",
                "anomaly_window_count",
                "total_flows_sum",
                "total_anomalous_flows_sum",
                "max_anomaly_ratio",
                "mean_anomaly_ratio",
                "avg_anomaly_score",
                "max_anomaly_score",
                "attacker_score",
                "victim_score",
                "compromised_score",
                "outbound_unique_targets",
                "inbound_unique_sources",
                "outbound_unique_target_ports",
                "top_predicted_labels",
            ]
        )
    rows: list[dict[str, object]] = []
    for node_id, group in summary_df.groupby("node_id", sort=False):
        best = _best_node_row(group)
        rows.append(
            {
                "node_id": str(node_id),
                "node_role": str(best["node_role"]),
                "role_severity": int(ROLE_SEVERITY.get(str(best["node_role"]), 0)),
                "role_priority": float(ROLE_PRIORITY.get(str(best["node_role"]), 0.0)),
                "window_count": int(len(group)),
                "anomaly_window_count": int((group["node_role"].astype(str) != "uncertain").sum()),
                "total_flows_sum": float(pd.to_numeric(group["total_flows"], errors="coerce").fillna(0.0).sum()),
                "total_anomalous_flows_sum": float(
                    pd.to_numeric(group["total_anomalous_flows"], errors="coerce").fillna(0.0).sum()
                ),
                "max_anomaly_ratio": float(pd.to_numeric(group["anomaly_ratio"], errors="coerce").fillna(0.0).max()),
                "mean_anomaly_ratio": float(
                    pd.to_numeric(group["anomaly_ratio"], errors="coerce").fillna(0.0).mean()
                ),
                "avg_anomaly_score": float(
                    pd.to_numeric(group["avg_anomaly_score"], errors="coerce").fillna(0.0).mean()
                ),
                "max_anomaly_score": float(
                    pd.to_numeric(group["max_anomaly_score"], errors="coerce").fillna(0.0).max()
                ),
                "attacker_score": float(
                    pd.to_numeric(group["attacker_score"], errors="coerce").fillna(0.0).max()
                ),
                "victim_score": float(
                    pd.to_numeric(group["victim_score"], errors="coerce").fillna(0.0).max()
                ),
                "compromised_score": float(
                    pd.to_numeric(group["compromised_score"], errors="coerce").fillna(0.0).max()
                ),
                "outbound_unique_targets": int(
                    pd.to_numeric(group["outbound_unique_targets"], errors="coerce").fillna(0.0).max()
                ),
                "inbound_unique_sources": int(
                    pd.to_numeric(group["inbound_unique_sources"], errors="coerce").fillna(0.0).max()
                ),
                "outbound_unique_target_ports": int(
                    pd.to_numeric(group["outbound_unique_target_ports"], errors="coerce").fillna(0.0).max()
                ),
                "top_predicted_labels": str(best.get("top_predicted_labels", "")),
            }
        )
    return pd.DataFrame(rows)


def _score_nodes(nodes: pd.DataFrame, core_top_ratio: float) -> pd.DataFrame:
    scored = nodes.copy()
    scored["connectivity_proxy"] = (
        pd.to_numeric(scored["outbound_unique_targets"], errors="coerce").fillna(0.0)
        + pd.to_numeric(scored["inbound_unique_sources"], errors="coerce").fillna(0.0)
        + 0.5 * pd.to_numeric(scored["outbound_unique_target_ports"], errors="coerce").fillna(0.0)
    )
    scored["damage_score"] = 0.0
    scored["structural_score"] = 0.0
    scored["core_score"] = 0.0
    scored["repair_priority_score"] = 0.0
    scored["is_core"] = False

    anomalous_mask = scored["role_priority"] > 0.0
    anomalous = scored[anomalous_mask].copy()
    if anomalous.empty:
        return scored

    anomaly_intensity = normalize_series(anomalous["max_anomaly_score"])
    anomalous["damage_score"] = (
        0.40 * anomalous["compromised_score"]
        + 0.30 * anomalous["attacker_score"]
        + 0.20 * anomalous["victim_score"]
        + 0.05 * anomalous["max_anomaly_ratio"]
        + 0.05 * anomaly_intensity
    )

    anomalous["structural_score"] = (
        0.50 * normalize_series(anomalous["connectivity_proxy"])
        + 0.30 * normalize_series(anomalous["total_anomalous_flows_sum"])
        + 0.20 * normalize_series(anomalous["anomaly_window_count"])
    )
    anomalous["core_score"] = 0.60 * anomalous["damage_score"] + 0.40 * anomalous["structural_score"]

    ratio = min(max(core_top_ratio, 0.0), 1.0)
    core_count = max(1, int(math.ceil(len(anomalous) * ratio)))
    core_indices = anomalous.sort_values(
        by=[
            "core_score",
            "compromised_score",
            "attacker_score",
            "victim_score",
            "total_anomalous_flows_sum",
        ],
        ascending=[False, False, False, False, False],
    ).head(core_count).index
    anomalous.loc[core_indices, "is_core"] = True
    anomalous["repair_priority_score"] = (
        0.40 * anomalous["damage_score"]
        + 0.25 * anomalous["structural_score"]
        + 0.15 * anomalous["role_priority"]
        + 0.20 * anomalous["is_core"].astype(float)
    )

    scored.loc[anomalous.index, anomalous.columns] = anomalous
    return scored


def build_repair_plan(summary_df: pd.DataFrame, core_top_ratio: float = 0.30) -> RepairPlanResult:
    aggregated = aggregate_node_summary(summary_df)
    scored = _score_nodes(aggregated, core_top_ratio=core_top_ratio)
    anomalous = scored[scored["role_priority"] > 0.0].copy()
    total_node_count = int(len(scored))
    anomalous_node_count = int(len(anomalous))
    core_node_count = int(anomalous["is_core"].sum()) if not anomalous.empty else 0
    core_ratio = core_node_count / max(1, total_node_count)

    if anomalous.empty:
        empty_plan = pd.DataFrame(columns=REPAIR_ORDER_COLUMNS)
        interpretation = (
            "No anomalous nodes were found. Cost is set to 0 because there is no repair action to schedule."
        )
        return RepairPlanResult(
            aggregated_nodes=scored,
            repair_order=empty_plan,
            total_node_count=total_node_count,
            anomalous_node_count=0,
            core_node_count=0,
            core_ratio=0.0,
            cost=0.0,
            denominator=0.0,
            interpretation=interpretation,
        )

    anomalous = anomalous.sort_values(
        by=[
            "repair_priority_score",
            "is_core",
            "role_priority",
            "damage_score",
            "structural_score",
            "total_anomalous_flows_sum",
            "node_id",
        ],
        ascending=[False, False, False, False, False, False, True],
    ).reset_index(drop=True)

    cumulative_contribution = 0.0
    repaired_core_count = 0
    plan_rows: list[dict[str, object]] = []
    for index, row in anomalous.iterrows():
        rank = index + 1
        factorial_rank = math.factorial(rank)
        contribution = float(row["repair_priority_score"]) / float(factorial_rank)
        cumulative_contribution += contribution
        repaired_core_count += int(bool(row["is_core"]))
        plan_rows.append(
            {
                "repair_rank": rank,
                "node_id": str(row["node_id"]),
                "node_role": str(row["node_role"]),
                "is_core": bool(row["is_core"]),
                "role_priority": float(row["role_priority"]),
                "damage_score": float(row["damage_score"]),
                "structural_score": float(row["structural_score"]),
                "core_score": float(row["core_score"]),
                "repair_priority_score": float(row["repair_priority_score"]),
                "factorial_rank": factorial_rank,
                "rank_contribution": contribution,
                "cumulative_contribution": cumulative_contribution,
                "remaining_core_after_repair": max(0, core_node_count - repaired_core_count),
                "top_predicted_labels": str(row["top_predicted_labels"]),
            }
        )

    repair_order = pd.DataFrame(plan_rows, columns=REPAIR_ORDER_COLUMNS)
    denominator = float(repair_order["rank_contribution"].sum())
    cost = (core_ratio**2) / denominator if denominator > 0.0 else float("inf")
    interpretation = (
        "The original expression does not distinguish different permutations if repair order is treated only as "
        "1..n. To make ordering operational, the denominator is interpreted as the factorial-discounted sum of "
        "node repair priority scores derived from the predicted anomalous node set. Under this interpretation, "
        "sorting nodes by repair_priority_score in descending order yields the minimum cost."
    )
    return RepairPlanResult(
        aggregated_nodes=scored,
        repair_order=repair_order,
        total_node_count=total_node_count,
        anomalous_node_count=anomalous_node_count,
        core_node_count=core_node_count,
        core_ratio=core_ratio,
        cost=cost,
        denominator=denominator,
        interpretation=interpretation,
    )
