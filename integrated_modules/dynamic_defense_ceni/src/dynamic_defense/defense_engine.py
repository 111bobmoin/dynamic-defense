from __future__ import annotations

import inspect
from dataclasses import asdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .ceni_adapter import CeniActionAdapter
from .feature_extractor import ThreatFeatureMatcher, normalize_columns
from .optimizer import ActorCriticLikeOptimizer
from .policy_store import PolicyStore


BENIGN_LABELS = {"BENIGN", "NORMAL", "0"}


class DynamicDefenseEngine:
    def __init__(
        self,
        store: PolicyStore,
        matcher: ThreatFeatureMatcher,
        adapter: CeniActionAdapter,
        epsilon: float = 0.05,
        confidence_threshold: float = 0.70,
        detector_mode: str = "template",
        torch_detector=None,
        torch_confidence_threshold: float = 0.70,
        optimizer=None,
    ):
        self.store = store
        self.matcher = matcher
        self.adapter = adapter
        self.optimizer = optimizer if optimizer is not None else ActorCriticLikeOptimizer(store, epsilon=epsilon)
        self.confidence_threshold = confidence_threshold
        self.detector_mode = detector_mode
        self.torch_detector = torch_detector
        self.torch_confidence_threshold = torch_confidence_threshold
        self.current_strategy_id = None

    @staticmethod
    def _is_attack_label(label: str) -> bool:
        return str(label).strip().upper() not in BENIGN_LABELS and str(label).strip() != ""

    @staticmethod
    def _supports_parameter(func, name: str) -> bool:
        try:
            sig = inspect.signature(func)
        except (TypeError, ValueError):
            return False
        return any(
            param.kind == inspect.Parameter.VAR_KEYWORD or param.name == name
            for param in sig.parameters.values()
        )

    def _select_policy(self, attack_type: str, state_context: Dict):
        if self._supports_parameter(self.optimizer.select, "state_context"):
            return self.optimizer.select(attack_type, state_context=state_context)
        return self.optimizer.select(attack_type)

    def _observe_policy(self, strategy_id: str, reward: float, success: bool, state_context: Dict) -> None:
        kwargs = {"reward": reward, "success": success}
        if self._supports_parameter(self.optimizer.observe, "state_context"):
            kwargs["state_context"] = state_context
        if self._supports_parameter(self.optimizer.observe, "next_state_context"):
            kwargs["next_state_context"] = None
        self.optimizer.observe(strategy_id, **kwargs)

    def run_on_csv(self, csv_path: str, window_size: int = 200, limit: Optional[int] = None) -> pd.DataFrame:
        df = pd.read_csv(csv_path, nrows=limit)
        df = normalize_columns(df)
        events = []
        for start in range(0, len(df), window_size):
            window = df.iloc[start : start + window_size]
            if window.empty:
                continue
            events.extend(self._handle_window(window, window_id=start // window_size))
        return pd.DataFrame(events)

    def _handle_window(self, window: pd.DataFrame, window_id: int) -> List[Dict]:
        matches = [self.matcher.match_row(row) for _, row in window.iterrows()]
        attack_votes = pd.Series([m.attack_type for m in matches]).value_counts()
        template_attack_type = str(attack_votes.index[0]) if not attack_votes.empty else "UNKNOWN"
        template_score = float(np.mean([m.score for m in matches])) if matches else 0.0

        torch_label = None
        torch_confidence = None
        detector_source = "template"

        if self.detector_mode in {"torch", "hybrid"}:
            if self.torch_detector is None:
                raise RuntimeError("detector_mode requires torch_detector, but torch_detector is None")
            torch_result = self.torch_detector.predict_window(window)
            torch_label = torch_result.predicted_label
            torch_confidence = float(torch_result.confidence)

        if self.detector_mode == "template":
            attack_type = template_attack_type
            avg_score = template_score
            detector_source = "template"
        elif self.detector_mode == "torch":
            attack_type = torch_label or "UNKNOWN"
            avg_score = float(torch_confidence or 0.0)
            detector_source = "torch"
        elif self.detector_mode == "hybrid":
            if torch_confidence is not None and torch_confidence >= self.torch_confidence_threshold:
                attack_type = torch_label or "UNKNOWN"
                avg_score = float(torch_confidence)
                detector_source = "torch"
            else:
                attack_type = template_attack_type
                avg_score = template_score
                detector_source = "template_fallback"
        else:
            raise RuntimeError(f"unsupported detector_mode: {self.detector_mode}")

        labels = window["Label"].astype(str) if "Label" in window.columns else pd.Series([""] * len(window))
        label_counts = labels.fillna("").astype(str).str.strip().value_counts()
        true_majority_label = str(label_counts.index[0]) if not label_counts.empty and str(label_counts.index[0]).strip() else "UNKNOWN"
        attack_present = any(self._is_attack_label(x) for x in labels)

        # detection_success：检测结果是否与数据标签一致。
        detected_as_attack = attack_type not in {"UNKNOWN", "BENIGN", "NORMAL"} and avg_score >= self.confidence_threshold
        detection_success = (attack_present and detected_as_attack) or ((not attack_present) and (not detected_as_attack))

        raw_attack_type = attack_type
        effective_attack_type = attack_type if detected_as_attack else "BENIGN"
        context = {
            "window_id": window_id,
            "attack_type": effective_attack_type,
            "raw_matched_attack_type": raw_attack_type,
            "avg_match_score": avg_score,
            "attack_present_by_label": attack_present,
            "rows": int(len(window)),
            "detector_source": detector_source,
            "template_attack_type": template_attack_type,
            "template_score": template_score,
            "torch_label": torch_label,
            "torch_confidence": torch_confidence,
        }
        policy = self._select_policy(effective_attack_type, context)

        # defense_success：动态防御是否触发了合理响应。
        # 这个指标关注策略选择、模型切换、限速、隔离、日志增强等动作是否被调度。
        has_policy = policy is not None
        has_actions = bool(policy.actions)
        benign_policy = effective_attack_type == "BENIGN" and policy.strategy_id == "s_benign_monitor"
        known_attack_policy = effective_attack_type not in {"UNKNOWN", "BENIGN"} and policy.strategy_id not in {"s_unknown_similarity", "s_benign_monitor"}
        fallback_policy = effective_attack_type == "UNKNOWN" and policy.strategy_id == "s_unknown_similarity"
        defense_success = has_policy and has_actions and (benign_policy or known_attack_policy or fallback_policy)

        # 奖励函数：
        # 1. 检测正确且防御响应成功：高奖励；
        # 2. 正常流量被正确保持：中等奖励；
        # 3. 检测不确定但已触发合理防御响应：中等奖励；
        # 4. 无有效响应：惩罚。
        if attack_present and detected_as_attack and defense_success:
            reward = 1.0 + min(avg_score, 1.0) - policy.cost
        elif not attack_present and not detected_as_attack:
            reward = 0.6 - policy.cost
        elif defense_success:
            reward = 0.7 + min(avg_score, 0.5) - policy.cost
        else:
            reward = -1.0 - policy.cost

        self._observe_policy(policy.strategy_id, reward=reward, success=defense_success, state_context=context)

        adjustment_triggered = self.current_strategy_id != policy.strategy_id or attack_present
        self.current_strategy_id = policy.strategy_id
        action_results = self.adapter.execute_actions(policy.strategy_id, policy.actions, context) if adjustment_triggered else []
        return [
            {
                "window_id": window_id,
                "rows": int(len(window)),
                "attack_type": effective_attack_type,
                "raw_matched_attack_type": raw_attack_type,
                "true_majority_label": true_majority_label,
                "avg_match_score": avg_score,
                "detector_source": detector_source,
                "template_attack_type": template_attack_type,
                "template_score": float(template_score),
                "torch_label": torch_label,
                "torch_confidence": torch_confidence,
                "strategy_id": policy.strategy_id,
                "model_type": policy.model_type,
                "adjustment_triggered": bool(adjustment_triggered),
                "detection_success": bool(detection_success),
                "defense_success": bool(defense_success),
                "reward": float(reward),
                "actions": [asdict(r) for r in action_results],
            }
        ]
