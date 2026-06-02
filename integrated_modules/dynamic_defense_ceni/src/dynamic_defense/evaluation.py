from __future__ import annotations

from typing import Dict, Optional


NORMAL_LABELS = {"BENIGN", "NORMAL", "0"}
LABEL_ALIASES = {
    "Web Attack \ufffd Brute Force": "Web Attack Brute Force",
    "Web Attack � Brute Force": "Web Attack Brute Force",
    "Web Attack-Brute Force": "Web Attack Brute Force",
    "Web Attack \ufffd XSS": "Web Attack XSS",
    "Web Attack � XSS": "Web Attack XSS",
    "Web Attack-XSS": "Web Attack XSS",
    "Web Attack \ufffd Sql Injection": "Web Attack Sql Injection",
    "Web Attack � Sql Injection": "Web Attack Sql Injection",
    "Web Attack-Sql Injection": "Web Attack Sql Injection",
}

EXPECTED_STRATEGY_BY_FAMILY = {
    "BENIGN": "s_benign_monitor",
    "DDoS": "s_ddos_vote_rate_limit",
    "PortScan": "s_portscan_isolate",
    "Brute Force": "s_bruteforce_ssh_ftp",
    "Web Attack": "s_web_attack_strict",
    "Heartbleed": "s_heartbleed_deep_inspection",
    "UNKNOWN": "s_unknown_similarity",
}
KNOWN_STRATEGY_FAMILIES = set(EXPECTED_STRATEGY_BY_FAMILY.keys())


def normalize_label(label: Optional[str]) -> str:
    value = str(label or "UNKNOWN").strip()
    if not value:
        return "UNKNOWN"
    if value.upper() in NORMAL_LABELS:
        return "BENIGN"
    return LABEL_ALIASES.get(value, value)


def attack_family(label: Optional[str]) -> str:
    value = normalize_label(label)
    if value == "BENIGN":
        return "BENIGN"
    if value == "DDoS" or value.startswith("DoS "):
        return "DDoS"
    if value in {"FTP-Patator", "SSH-Patator", "Brute Force"}:
        return "Brute Force"
    if value.startswith("Web Attack"):
        return "Web Attack"
    if value == "PortScan":
        return "PortScan"
    if value == "Heartbleed":
        return "Heartbleed"
    if value == "UNKNOWN":
        return "UNKNOWN"
    return value


def strategy_family_label(label: Optional[str]) -> str:
    family = attack_family(label)
    if family in KNOWN_STRATEGY_FAMILIES:
        return family
    return "UNKNOWN"


def expected_strategy_for_label(label: Optional[str]) -> Optional[str]:
    return EXPECTED_STRATEGY_BY_FAMILY.get(attack_family(label))


def strategy_matches_label(strategy_id: str, label: Optional[str]) -> bool:
    expected = expected_strategy_for_label(label)
    return expected is not None and str(strategy_id) == expected


def _empty_attack_accuracy() -> Dict:
    return {
        "exact": 0.0,
        "family": 0.0,
        "evaluated_windows": 0,
        "exact_matches": 0,
        "family_matches": 0,
    }


def compute_attack_type_accuracy(events) -> Dict:
    if events is None or len(events) == 0 or "true_majority_label" not in events.columns:
        return _empty_attack_accuracy()

    evaluated = 0
    exact_matches = 0
    family_matches = 0
    for _, row in events.iterrows():
        true_label = normalize_label(row.get("true_majority_label"))
        predicted = normalize_label(row.get("attack_type"))
        evaluated += 1
        if predicted == true_label:
            exact_matches += 1
        if attack_family(predicted) == attack_family(true_label):
            family_matches += 1

    if evaluated == 0:
        return _empty_attack_accuracy()
    return {
        "exact": float(exact_matches / evaluated),
        "family": float(family_matches / evaluated),
        "evaluated_windows": int(evaluated),
        "exact_matches": int(exact_matches),
        "family_matches": int(family_matches),
    }


def compute_strategy_match_accuracy(events) -> Dict:
    if events is None or len(events) == 0 or "true_majority_label" not in events.columns:
        return {"accuracy": 0.0, "evaluated_windows": 0, "matches": 0}

    evaluated = 0
    matches = 0
    for _, row in events.iterrows():
        expected = expected_strategy_for_label(row.get("true_majority_label"))
        if expected is None:
            continue
        evaluated += 1
        if str(row.get("strategy_id")) == expected:
            matches += 1

    if evaluated == 0:
        return {"accuracy": 0.0, "evaluated_windows": 0, "matches": 0}
    return {"accuracy": float(matches / evaluated), "evaluated_windows": int(evaluated), "matches": int(matches)}


def detector_source_counts(events) -> Dict[str, int]:
    if events is None or len(events) == 0 or "detector_source" not in events.columns:
        return {}
    return {str(k): int(v) for k, v in events["detector_source"].value_counts().to_dict().items()}


def build_evaluation_summary(events) -> Dict:
    return {
        "attack_type_accuracy": compute_attack_type_accuracy(events),
        "strategy_match_accuracy": compute_strategy_match_accuracy(events),
        "detector_source_counts": detector_source_counts(events),
    }
