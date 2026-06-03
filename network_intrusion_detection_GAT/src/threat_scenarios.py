from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from node_summary import NODE_OUTPUT_COLUMNS


THREAT_NODE_OUTPUT_COLUMNS = NODE_OUTPUT_COLUMNS + [
    "threat_ids",
    "threat_names",
    "threat_severities",
    "threat_descriptions",
    "related_links",
    "related_systems",
    "recommended_actions",
]

SEVERITY_WEIGHT = {
    "info": 0.35,
    "warning": 0.70,
    "critical": 1.00,
}


@dataclass(frozen=True)
class ThreatScenario:
    scenario_id: str
    name: str
    severity: str
    description: str
    affected_links: tuple[str, ...]
    affected_nodes: tuple[str, ...]
    related_systems: tuple[str, ...]
    recommended_action: str


def _as_string_list(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    if not isinstance(values, list):
        raise ValueError(f"Expected a list, got: {type(values)!r}")
    return tuple(str(value).strip() for value in values if str(value).strip())


def load_threat_scenarios(path: Path) -> list[ThreatScenario]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list):
        raise ValueError(f"Invalid threat scenario file, missing 'scenarios' list: {path}")

    loaded: list[ThreatScenario] = []
    for index, raw in enumerate(scenarios, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"Scenario #{index} is not an object.")
        loaded.append(
            ThreatScenario(
                scenario_id=str(raw.get("id", f"scenario_{index:02d}")).strip(),
                name=str(raw.get("name", "")).strip(),
                severity=str(raw.get("severity", "warning")).strip().lower(),
                description=str(raw.get("description", "")).strip(),
                affected_links=_as_string_list(raw.get("affected_links")),
                affected_nodes=_as_string_list(raw.get("affected_nodes")),
                related_systems=_as_string_list(raw.get("related_systems")),
                recommended_action=str(raw.get("recommended_action", "")).strip(),
            )
        )
    return loaded


def filter_threat_scenarios(
    scenarios: list[ThreatScenario],
    scenario_ids: list[str] | tuple[str, ...] | set[str],
) -> list[ThreatScenario]:
    wanted = {str(value).strip() for value in scenario_ids if str(value).strip()}
    if not wanted:
        return []
    return [scenario for scenario in scenarios if scenario.scenario_id in wanted]


def _join_unique(values: list[str]) -> str:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalized = str(value).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return "; ".join(ordered)


def _role_from_scenario(scenario: ThreatScenario, node_id: str) -> str:
    node = str(node_id).strip()
    affected_nodes = list(scenario.affected_nodes)
    if not affected_nodes:
        return "uncertain"

    if scenario.scenario_id == "threat_09_traffic_attack":
        if node == affected_nodes[-1]:
            return "suspected_victim"
        return "suspected_attacker"

    if scenario.scenario_id in {"threat_02_file_tamper", "threat_05_os_vulnerability", "threat_06_middleware_vulnerability"}:
        return "suspected_compromised_host"

    if scenario.scenario_id in {"threat_03_unauthorized_device", "threat_07_application_attack", "threat_08_comm_spoofing"}:
        if node == affected_nodes[0]:
            return "suspected_attacker"
        if node == affected_nodes[-1]:
            return "suspected_victim"
        return "suspected_compromised_host"

    if scenario.scenario_id in {"threat_01_comm_hijack", "threat_04_data_forgery", "threat_10_illegal_operation"}:
        if node == affected_nodes[0]:
            return "suspected_attacker"
        if node == affected_nodes[-1]:
            return "suspected_victim"
        return "suspected_compromised_host"

    return "suspected_compromised_host"


def _score_triplet(role: str, severity_weight: float) -> tuple[float, float, float]:
    if role == "suspected_attacker":
        return severity_weight, 0.0, severity_weight * 0.55
    if role == "suspected_victim":
        return 0.0, severity_weight, severity_weight * 0.55
    if role == "suspected_compromised_host":
        return severity_weight * 0.75, severity_weight * 0.75, severity_weight
    return 0.0, 0.0, 0.0


def build_threat_node_summary(
    scenarios: list[ThreatScenario],
    *,
    window_start: str = "1970-01-01 08:00:00",
) -> pd.DataFrame:
    node_to_scenarios: dict[str, list[ThreatScenario]] = defaultdict(list)
    for scenario in scenarios:
        for node_id in scenario.affected_nodes:
            node_to_scenarios[str(node_id)].append(scenario)

    rows: list[dict[str, object]] = []
    for node_id in sorted(node_to_scenarios):
        associated = node_to_scenarios[node_id]
        attacker_score = 0.0
        victim_score = 0.0
        compromised_score = 0.0
        roles: list[str] = []
        names: list[str] = []
        severities: list[str] = []
        descriptions: list[str] = []
        links: list[str] = []
        systems: list[str] = []
        actions: list[str] = []
        ids: list[str] = []

        for scenario in associated:
            ids.append(scenario.scenario_id)
            names.append(scenario.name or scenario.scenario_id)
            severities.append(scenario.severity)
            descriptions.append(scenario.description)
            links.extend(scenario.affected_links)
            systems.extend(scenario.related_systems)
            actions.append(scenario.recommended_action)

            severity_weight = SEVERITY_WEIGHT.get(scenario.severity, SEVERITY_WEIGHT["warning"])
            role = _role_from_scenario(scenario, node_id)
            roles.append(role)
            a_score, v_score, c_score = _score_triplet(role, severity_weight)
            attacker_score = max(attacker_score, a_score)
            victim_score = max(victim_score, v_score)
            compromised_score = max(compromised_score, c_score)

        role_priority = {
            "suspected_compromised_host": 3,
            "suspected_attacker": 2,
            "suspected_victim": 1,
            "uncertain": 0,
        }
        final_role = max(roles, key=lambda value: role_priority.get(value, 0), default="uncertain")
        high_conf = len(associated)
        total_flows = max(60, high_conf * 60)
        total_anomalous_flows = total_flows

        rows.append(
            {
                "node_id": node_id,
                "window_start": window_start,
                "total_flows": total_flows,
                "total_anomalous_flows": total_anomalous_flows,
                "high_conf_anomalous_flows": high_conf,
                "anomaly_ratio": 1.0 if high_conf else 0.0,
                "avg_anomaly_score": max(attacker_score, victim_score, compromised_score),
                "max_anomaly_score": max(attacker_score, victim_score, compromised_score),
                "outbound_flows": total_flows if final_role != "suspected_victim" else total_flows // 2,
                "outbound_anomalous_flows": total_flows if final_role != "suspected_victim" else total_flows // 2,
                "outbound_high_conf_anomalous_flows": high_conf if final_role != "suspected_victim" else max(1, high_conf // 2),
                "outbound_anomaly_ratio": 1.0 if final_role != "suspected_victim" else 0.5,
                "inbound_flows": total_flows if final_role != "suspected_attacker" else total_flows // 2,
                "inbound_anomalous_flows": total_flows if final_role != "suspected_attacker" else total_flows // 2,
                "inbound_high_conf_anomalous_flows": high_conf if final_role != "suspected_attacker" else max(1, high_conf // 2),
                "inbound_anomaly_ratio": 1.0 if final_role != "suspected_attacker" else 0.5,
                "outbound_unique_targets": max(1, len({link for link in links if node_id in link})),
                "inbound_unique_sources": max(1, len({link for link in links if node_id in link})),
                "outbound_unique_target_ports": max(1, len(associated) * 2),
                "small_flow_ratio": 0.0,
                "single_packet_ratio": 0.0,
                "role_evidence_support": 1.0,
                "attacker_score": attacker_score,
                "victim_score": victim_score,
                "compromised_score": compromised_score,
                "node_role": final_role,
                "top_predicted_labels": _join_unique(names),
                "threat_ids": _join_unique(ids),
                "threat_names": _join_unique(names),
                "threat_severities": _join_unique(severities),
                "threat_descriptions": _join_unique(descriptions),
                "related_links": _join_unique(links),
                "related_systems": _join_unique(systems),
                "recommended_actions": _join_unique(actions),
            }
        )

    return pd.DataFrame(rows, columns=THREAT_NODE_OUTPUT_COLUMNS)
