from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional

from .utils import load_yaml


@dataclass
class DefensePolicy:
    strategy_id: str
    name: str
    model_type: str
    attack_types: List[str]
    priority: int
    cost: float
    actions: List[Dict]
    updated_at: str


class PolicyStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS policies (
                    strategy_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    attack_types TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    cost REAL NOT NULL,
                    actions TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS policy_stats (
                    strategy_id TEXT PRIMARY KEY,
                    selected_count INTEGER NOT NULL DEFAULT 0,
                    success_count INTEGER NOT NULL DEFAULT 0,
                    avg_reward REAL NOT NULL DEFAULT 0.0,
                    last_selected_at TEXT,
                    FOREIGN KEY(strategy_id) REFERENCES policies(strategy_id)
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def upsert_many(self, policies: Iterable[DefensePolicy]) -> None:
        import json

        conn = self._connect()
        try:
            cur = conn.cursor()
            for p in policies:
                cur.execute(
                    """
                    INSERT INTO policies(strategy_id, name, model_type, attack_types, priority, cost, actions, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(strategy_id) DO UPDATE SET
                        name=excluded.name,
                        model_type=excluded.model_type,
                        attack_types=excluded.attack_types,
                        priority=excluded.priority,
                        cost=excluded.cost,
                        actions=excluded.actions,
                        updated_at=excluded.updated_at
                    """,
                    (
                        p.strategy_id,
                        p.name,
                        p.model_type,
                        json.dumps(p.attack_types, ensure_ascii=False),
                        p.priority,
                        p.cost,
                        json.dumps(p.actions, ensure_ascii=False),
                        p.updated_at,
                    ),
                )
                cur.execute(
                    "INSERT OR IGNORE INTO policy_stats(strategy_id) VALUES (?)",
                    (p.strategy_id,),
                )
            conn.commit()
        finally:
            conn.close()

    def list_policies(self) -> List[DefensePolicy]:
        import json

        conn = self._connect()
        try:
            cur = conn.cursor()
            rows = cur.execute(
                "SELECT strategy_id, name, model_type, attack_types, priority, cost, actions, updated_at FROM policies"
            ).fetchall()
            return [
                DefensePolicy(
                    strategy_id=row[0],
                    name=row[1],
                    model_type=row[2],
                    attack_types=json.loads(row[3]),
                    priority=int(row[4]),
                    cost=float(row[5]),
                    actions=json.loads(row[6]),
                    updated_at=row[7],
                )
                for row in rows
            ]
        finally:
            conn.close()

    def get_by_attack_type(self, attack_type: str) -> List[DefensePolicy]:
        attack_type_norm = (attack_type or "UNKNOWN").strip()
        policies = self.list_policies()
        matched = [p for p in policies if attack_type_norm in p.attack_types]
        if not matched:
            matched = [p for p in policies if "UNKNOWN" in p.attack_types]
        return sorted(matched, key=lambda p: (p.priority, -p.cost), reverse=True)

    def get_stats(self, strategy_id: str) -> Dict:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT selected_count, success_count, avg_reward, last_selected_at FROM policy_stats WHERE strategy_id=?",
                (strategy_id,),
            ).fetchone()
            if row is None:
                return {"selected_count": 0, "success_count": 0, "avg_reward": 0.0, "last_selected_at": None}
            return {
                "selected_count": int(row[0]),
                "success_count": int(row[1]),
                "avg_reward": float(row[2]),
                "last_selected_at": row[3],
            }
        finally:
            conn.close()

    def update_reward(self, strategy_id: str, reward: float, success: bool) -> None:
        stats = self.get_stats(strategy_id)
        n = stats["selected_count"] + 1
        old_avg = stats["avg_reward"]
        new_avg = old_avg + (reward - old_avg) / max(n, 1)
        success_count = stats["success_count"] + (1 if success else 0)
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO policy_stats(strategy_id, selected_count, success_count, avg_reward, last_selected_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(strategy_id) DO UPDATE SET
                    selected_count=excluded.selected_count,
                    success_count=excluded.success_count,
                    avg_reward=excluded.avg_reward,
                    last_selected_at=excluded.last_selected_at
                """,
                (strategy_id, n, success_count, float(new_avg), datetime.utcnow().isoformat()),
            )
            conn.commit()
        finally:
            conn.close()


def load_policies_from_yaml(path: str) -> List[DefensePolicy]:
    raw = load_yaml(path)
    now = datetime.utcnow().isoformat()
    policies = []
    for item in raw.get("strategies", []):
        policies.append(
            DefensePolicy(
                strategy_id=str(item["strategy_id"]),
                name=str(item.get("name", item["strategy_id"])),
                model_type=str(item.get("model_type", "unknown")),
                attack_types=[str(x) for x in item.get("attack_types", ["UNKNOWN"])],
                priority=int(item.get("priority", 0)),
                cost=float(item.get("cost", 0.0)),
                actions=list(item.get("actions", [])),
                updated_at=now,
            )
        )
    return policies
