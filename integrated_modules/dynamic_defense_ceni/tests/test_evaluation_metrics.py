import pandas as pd

from src.dynamic_defense.evaluation import (
    attack_family,
    build_evaluation_summary,
    expected_strategy_for_label,
    normalize_label,
    strategy_family_label,
    strategy_matches_label,
)


def test_attack_family_normalization():
    assert normalize_label("NORMAL") == "BENIGN"
    assert normalize_label("BENIGN") == "BENIGN"
    assert attack_family("DoS Hulk") == "DDoS"
    assert attack_family("DoS GoldenEye") == "DDoS"
    assert attack_family("DoS slowloris") == "DDoS"
    assert attack_family("DoS Slowhttptest") == "DDoS"
    assert attack_family("Web Attack XSS") == "Web Attack"
    assert attack_family("Web Attack Sql Injection") == "Web Attack"
    assert attack_family("FTP-Patator") == "Brute Force"
    assert strategy_family_label("NORMAL") == "BENIGN"
    assert strategy_family_label("Web Attack Brute Force") == "Web Attack"
    assert strategy_family_label("Infiltration") == "UNKNOWN"


def test_strategy_match_for_label_families():
    assert expected_strategy_for_label("BENIGN") == "s_benign_monitor"
    assert strategy_matches_label("s_ddos_vote_rate_limit", "DoS Hulk")
    assert strategy_matches_label("s_portscan_isolate", "PortScan")
    assert strategy_matches_label("s_bruteforce_ssh_ftp", "SSH-Patator")
    assert strategy_matches_label("s_web_attack_strict", "Web Attack Brute Force")
    assert strategy_matches_label("s_heartbleed_deep_inspection", "Heartbleed")
    assert not strategy_matches_label("s_benign_monitor", "DDoS")


def test_build_evaluation_summary_counts_accuracy():
    events = pd.DataFrame(
        [
            {
                "true_majority_label": "BENIGN",
                "attack_type": "BENIGN",
                "strategy_id": "s_benign_monitor",
                "detector_source": "template",
            },
            {
                "true_majority_label": "DoS Hulk",
                "attack_type": "DDoS",
                "strategy_id": "s_ddos_vote_rate_limit",
                "detector_source": "torch",
            },
            {
                "true_majority_label": "Web Attack XSS",
                "attack_type": "Web Attack",
                "strategy_id": "s_web_attack_strict",
                "detector_source": "torch",
            },
        ]
    )

    summary = build_evaluation_summary(events)
    assert summary["attack_type_accuracy"]["exact"] == 1.0 / 3.0
    assert summary["attack_type_accuracy"]["family"] == 1.0
    assert summary["strategy_match_accuracy"]["accuracy"] == 1.0
    assert summary["detector_source_counts"] == {"torch": 2, "template": 1}
