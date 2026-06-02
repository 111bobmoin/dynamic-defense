#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.dynamic_defense.policy_store import PolicyStore, load_policies_from_yaml
from src.dynamic_defense.utils import dump_json


def main() -> None:
    parser = argparse.ArgumentParser(description="测试用例34：加载防御策略库并输出元数据")
    parser.add_argument("--config", default="configs/strategies.yaml", help="策略配置文件")
    parser.add_argument("--db", default="data/policies.sqlite", help="策略库 SQLite 路径")
    parser.add_argument("--out", default="reports/strategy_metadata.json", help="元数据输出路径")
    args = parser.parse_args()

    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    policies = load_policies_from_yaml(args.config)
    store = PolicyStore(args.db)
    store.upsert_many(policies)
    metadata = [
        {
            "strategy_id": p.strategy_id,
            "name": p.name,
            "model_type": p.model_type,
            "attack_types": p.attack_types,
            "priority": p.priority,
            "cost": p.cost,
            "last_updated_at": p.updated_at,
        }
        for p in store.list_policies()
    ]
    dump_json({"loaded": len(metadata), "policies": metadata}, args.out)
    print(json.dumps({"status": "OK", "loaded": len(metadata), "output": args.out}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
