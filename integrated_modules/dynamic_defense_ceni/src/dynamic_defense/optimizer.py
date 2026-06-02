from __future__ import annotations

import random
from typing import List, Optional

from .policy_store import DefensePolicy, PolicyStore


class ActorCriticLikeOptimizer:
    """轻量级在线策略优化器。

    为了与测试大纲的“演员-评论家”说法对齐，这里采用可解释的 bandit 近似：
    actor = 按 attack_type 给出候选策略；critic = policy_stats.avg_reward 评估策略收益。
    后续可以替换成真正的 A2C/PPO，但测试脚本接口不需要改。
    """

    def __init__(self, store: PolicyStore, epsilon: float = 0.05):
        self.store = store
        self.epsilon = epsilon

    def select(self, attack_type: str) -> DefensePolicy:
        candidates = self.store.get_by_attack_type(attack_type)
        if not candidates:
            raise RuntimeError("策略库为空，无法执行动态防御")
        if random.random() < self.epsilon:
            return random.choice(candidates)

        def score(policy: DefensePolicy) -> float:
            stats = self.store.get_stats(policy.strategy_id)
            return float(stats["avg_reward"]) + policy.priority / 100.0 - policy.cost

        return max(candidates, key=score)

    def observe(self, strategy_id: str, reward: float, success: bool) -> None:
        self.store.update_reward(strategy_id, reward=reward, success=success)
