import json
from pathlib import Path
import pandas as pd


def load_json(path):
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def main():
    strategy = load_json("reports/strategy_metadata.json")
    feature = load_json("reports/feature_match_summary.json")
    defense = load_json("reports/dynamic_defense_summary.json")

    print("\n=== Test Case 34: Strategy Library ===")
    if strategy:
        print(f"Loaded policies: {strategy.get('loaded')}")
        for p in strategy.get("policies", []):
            print(f"- {p.get('strategy_id')} | model={p.get('model_type')} | attacks={p.get('attack_types')}")

    print("\n=== Test Case 35: Feature Matching ===")
    if feature:
        print(f"Rows: {feature.get('rows')}")
        print(f"Average match score: {feature.get('avg_match_score')}")
        print("Matched attack counts:")
        for k, v in feature.get("matched_attack_counts", {}).items():
            print(f"- {k}: {v}")

    print("\n=== Test Case 36: Dynamic Defense ===")
    if defense:
        print(f"Windows: {defense.get('windows')}")
        print(f"Adjustment events: {defense.get('adjustment_events')}")
        print(f"Detection success rate: {defense.get('detection_success_rate')}")
        print(f"Defense success rate: {defense.get('defense_success_rate')}")
        print("Strategy counts:")
        for k, v in defense.get("strategy_counts", {}).items():
            print(f"- {k}: {v}")

    events_path = Path("reports/dynamic_defense_events.csv")
    if events_path.exists():
        events = pd.read_csv(events_path)
        print("\n=== Dynamic Defense Events Preview ===")
        cols = [
            "window_id",
            "attack_type",
            "raw_matched_attack_type",
            "strategy_id",
            "adjustment_triggered",
            "detection_success",
            "defense_success",
            "reward",
        ]
        cols = [c for c in cols if c in events.columns]
        print(events[cols].to_string(index=False))


if __name__ == "__main__":
    main()
