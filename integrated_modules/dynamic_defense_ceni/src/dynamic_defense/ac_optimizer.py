from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .policy_store import DefensePolicy, PolicyStore


DEFAULT_ATTACK_LABELS = [
    "BENIGN",
    "NORMAL",
    "UNKNOWN",
    "DDoS",
    "DoS Hulk",
    "DoS GoldenEye",
    "DoS slowloris",
    "DoS Slowhttptest",
    "PortScan",
    "SSH-Patator",
    "FTP-Patator",
    "Brute Force",
    "Web Attack",
    "Web Attack Brute Force",
    "Web Attack Sql Injection",
    "Web Attack XSS",
    "Heartbleed",
]

DEFAULT_DETECTOR_SOURCES = ["template", "torch", "template_fallback", "unknown"]


class ActorCriticPolicyNet(nn.Module):
    def __init__(self, input_dim: int, num_actions: int, hidden_dim: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_dim, num_actions)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        hidden = self.shared(x)
        logits = self.actor(hidden)
        value = self.critic(hidden).squeeze(-1)
        return logits, value


class TorchActorCriticOptimizer:
    """CPU PyTorch Actor-Critic optimizer for dynamic defense policies."""

    def __init__(
        self,
        store: PolicyStore,
        lr: float = 0.001,
        gamma: float = 0.95,
        device: str = "cpu",
        hidden_dim: int = 64,
        attack_labels: Optional[List[str]] = None,
        detector_sources: Optional[List[str]] = None,
        model_path: Optional[str] = None,
        meta_path: Optional[str] = None,
    ):
        self.store = store
        self.lr = float(lr)
        self.gamma = float(gamma)
        self.device = torch.device(device or "cpu")
        self.hidden_dim = int(hidden_dim)
        self.attack_labels = list(attack_labels or DEFAULT_ATTACK_LABELS)
        self.detector_sources = list(detector_sources or DEFAULT_DETECTOR_SOURCES)
        self._refresh_policy_index()

        self.state_feature_names = (
            [f"attack_type={label}" for label in self.attack_labels]
            + [f"detector_source={source}" for source in self.detector_sources]
            + [
                "avg_match_score",
                "template_score",
                "torch_confidence",
                "attack_present_by_label",
                "candidate_priority_norm",
                "candidate_cost",
            ]
        )
        self.input_dim = len(self.state_feature_names)
        self.model = ActorCriticPolicyNet(self.input_dim, len(self.strategy_ids), hidden_dim=self.hidden_dim).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self._last_selection: Optional[Dict] = None

        if model_path and meta_path and Path(model_path).exists() and Path(meta_path).exists():
            self.load(model_path, meta_path)

    def _refresh_policy_index(self) -> None:
        self.policies = sorted(self.store.list_policies(), key=lambda p: p.strategy_id)
        if not self.policies:
            raise RuntimeError("策略库为空，无法初始化 Actor-Critic 优化器")
        self.policy_by_id = {p.strategy_id: p for p in self.policies}
        self.strategy_ids = [p.strategy_id for p in self.policies]
        self.strategy_to_idx = {sid: i for i, sid in enumerate(self.strategy_ids)}

    @staticmethod
    def _safe_float(value, default: float = 0.0) -> float:
        if value is None:
            return default
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return default
        return numeric if math.isfinite(numeric) else default

    @staticmethod
    def _safe_bool(value) -> float:
        if isinstance(value, str):
            return 1.0 if value.strip().lower() in {"1", "true", "yes", "y"} else 0.0
        return 1.0 if bool(value) else 0.0

    @staticmethod
    def _one_hot(value: str, labels: List[str], fallback: str) -> List[float]:
        normalized = str(value or fallback).strip()
        if normalized not in labels:
            normalized = fallback
        return [1.0 if label == normalized else 0.0 for label in labels]

    def build_state(
        self,
        attack_type: str,
        state_context: Optional[Dict] = None,
        candidate_policy: Optional[DefensePolicy] = None,
    ) -> torch.Tensor:
        context = state_context or {}
        effective_attack_type = str(context.get("attack_type") or attack_type or "UNKNOWN")
        detector_source = str(context.get("detector_source") or "unknown")
        if detector_source not in self.detector_sources:
            detector_source = "unknown"

        priority = candidate_policy.priority if candidate_policy is not None else context.get("priority", 0.0)
        cost = candidate_policy.cost if candidate_policy is not None else context.get("cost", 0.0)
        values = (
            self._one_hot(effective_attack_type, self.attack_labels, "UNKNOWN")
            + self._one_hot(detector_source, self.detector_sources, "unknown")
            + [
                self._safe_float(context.get("avg_match_score")),
                self._safe_float(context.get("template_score")),
                self._safe_float(context.get("torch_confidence")),
                self._safe_bool(context.get("attack_present_by_label", False)),
                self._safe_float(priority) / 100.0,
                self._safe_float(cost),
            ]
        )
        return torch.tensor(values, dtype=torch.float32, device=self.device)

    def _candidate_distribution(self, attack_type: str, state_context: Optional[Dict]):
        candidates = self.store.get_by_attack_type(attack_type)
        if not candidates:
            raise RuntimeError("策略库为空，无法执行动态防御")

        probs = []
        states = []
        self.model.eval()
        with torch.no_grad():
            for policy in candidates:
                state = self.build_state(attack_type, state_context, policy)
                logits, _ = self.model(state.unsqueeze(0))
                all_probs = torch.softmax(logits, dim=-1).squeeze(0)
                probs.append(all_probs[self.strategy_to_idx[policy.strategy_id]])
                states.append(state)

        candidate_probs = torch.stack(probs)
        total = float(candidate_probs.sum().item())
        if math.isfinite(total) and total > 0.0:
            candidate_probs = candidate_probs / total
        else:
            candidate_probs = torch.full_like(candidate_probs, 1.0 / len(candidates))
        return candidates, states, candidate_probs

    def select(self, attack_type: str, state_context: Optional[Dict] = None) -> DefensePolicy:
        candidates, states, candidate_probs = self._candidate_distribution(attack_type, state_context)
        sampled_idx = int(torch.multinomial(candidate_probs.cpu(), num_samples=1).item())
        policy = candidates[sampled_idx]
        self._last_selection = {
            "strategy_id": policy.strategy_id,
            "action_idx": self.strategy_to_idx[policy.strategy_id],
            "attack_type": attack_type,
            "state_context": dict(state_context or {}),
            "state": states[sampled_idx].detach().cpu(),
        }
        return policy

    def observe(
        self,
        strategy_id: str,
        reward: float,
        success: bool,
        state_context: Optional[Dict] = None,
        next_state_context: Optional[Dict] = None,
    ) -> None:
        self.store.update_reward(strategy_id, reward=reward, success=success)
        policy = self.policy_by_id.get(strategy_id)
        if policy is None:
            return

        if self._last_selection and self._last_selection.get("strategy_id") == strategy_id and state_context is None:
            attack_type = str(self._last_selection.get("attack_type") or "UNKNOWN")
            state = self._last_selection["state"].to(self.device)
            action_idx = int(self._last_selection["action_idx"])
        else:
            context = state_context or {}
            attack_type = str(context.get("attack_type") or "UNKNOWN")
            state = self.build_state(attack_type, context, policy)
            action_idx = self.strategy_to_idx[strategy_id]

        self.model.train()
        logits, value = self.model(state.unsqueeze(0))
        log_prob = F.log_softmax(logits, dim=-1)[0, action_idx]

        reward_tensor = torch.tensor(float(reward), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            if next_state_context:
                next_attack_type = str(next_state_context.get("attack_type") or attack_type)
                next_state = self.build_state(next_attack_type, next_state_context, policy)
                _, next_value = self.model(next_state.unsqueeze(0))
                target = reward_tensor + self.gamma * next_value.squeeze(0)
            else:
                target = reward_tensor

        td_error = target - value.squeeze(0)
        actor_loss = -log_prob * td_error.detach()
        critic_loss = F.smooth_l1_loss(value.squeeze(0), target)
        loss = actor_loss + critic_loss

        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.optim.step()
        self._last_selection = None

    def save(self, model_path: str, meta_path: str) -> None:
        model_file = Path(model_path)
        meta_file = Path(meta_path)
        model_file.parent.mkdir(parents=True, exist_ok=True)
        meta_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), str(model_file))
        meta = {
            "version": 1,
            "created_at": datetime.utcnow().isoformat(),
            "strategy_ids": self.strategy_ids,
            "attack_labels": self.attack_labels,
            "detector_sources": self.detector_sources,
            "state_feature_names": self.state_feature_names,
            "input_dim": self.input_dim,
            "num_actions": len(self.strategy_ids),
            "hidden_dim": self.hidden_dim,
            "lr": self.lr,
            "gamma": self.gamma,
            "device": "cpu",
        }
        meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, model_path: str, meta_path: str) -> None:
        meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        saved_strategy_ids = list(meta.get("strategy_ids", []))
        saved_attack_labels = list(meta.get("attack_labels", []))
        saved_detector_sources = list(meta.get("detector_sources", []))
        if saved_strategy_ids != self.strategy_ids:
            raise ValueError("Actor-Critic meta strategy_ids 与当前策略库不一致")
        if saved_attack_labels and saved_attack_labels != self.attack_labels:
            raise ValueError("Actor-Critic meta attack_labels 与当前状态编码不一致")
        if saved_detector_sources and saved_detector_sources != self.detector_sources:
            raise ValueError("Actor-Critic meta detector_sources 与当前状态编码不一致")
        if int(meta.get("input_dim", self.input_dim)) != self.input_dim:
            raise ValueError("Actor-Critic meta input_dim 与当前状态编码不一致")

        state = torch.load(str(model_path), map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
