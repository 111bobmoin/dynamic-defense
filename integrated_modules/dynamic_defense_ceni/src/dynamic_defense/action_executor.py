from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union


SUPPORTED_EXECUTION_MODES = {"simulated", "stateful", "shell"}
NETWORK_ACTIONS = {"rate_limit", "isolate_flow"}


class ActionExecutor:
    """Controller-side action execution framework.

    The executor is intentionally conservative: it records state transitions
    and executable plans, but it never mutates host networking directly.
    """

    def __init__(
        self,
        execution_mode: str = "simulated",
        state_path: str = "runtime/controller_state.json",
        plan_path: str = "reports/controller_execution_plan.jsonl",
    ):
        if execution_mode not in SUPPORTED_EXECUTION_MODES:
            raise ValueError("unsupported execution_mode: %s" % execution_mode)
        self.execution_mode = execution_mode
        self.state_path = Path(state_path)
        self.plan_path = Path(plan_path)

    @staticmethod
    def _timestamp() -> str:
        return datetime.utcnow().isoformat() + "Z"

    def execute(self, strategy_id: Union[str, Dict], action: Optional[Dict] = None, context: Optional[Dict] = None) -> Dict:
        if isinstance(strategy_id, dict) and action is None:
            payload = strategy_id
            strategy_id = str(payload.get("strategy_id", ""))
            action = payload.get("action", {})
            context = payload.get("context", {})
        strategy_id = str(strategy_id)
        context = context or {}
        action = action or {}
        action_type = str(action.get("type", "unknown"))
        timestamp = self._timestamp()

        state_updates = self._state_updates(strategy_id, action, context, timestamp)
        plan = self._build_plan(strategy_id, action, context, timestamp)
        status = self._status_for(action_type)
        safety = {
            "system_command_executed": False,
            "reason": "network-changing commands are planned only",
        }
        if self.execution_mode == "shell" and action_type in NETWORK_ACTIONS:
            safety["reason"] = "blocked by default shell-mode safety guard"

        self._append_plan(plan, status)
        state = self._load_state()
        self._apply_state_updates(state, state_updates, strategy_id, action, context, status, timestamp)
        self._write_state(state)

        return {
            "timestamp": timestamp,
            "execution_mode": self.execution_mode,
            "strategy_id": strategy_id,
            "action_type": action_type,
            "status": status,
            "state_path": str(self.state_path),
            "plan_path": str(self.plan_path),
            "planned_commands": plan["planned_commands"],
            "intents": plan["intents"],
            "state_updates": state_updates,
            "safety": safety,
        }

    def _status_for(self, action_type: str) -> str:
        if action_type not in {
            "monitor_only",
            "log_enrich",
            "switch_model",
            "raise_threshold",
            "rate_limit",
            "isolate_flow",
        }:
            return "UNSUPPORTED_ACTION"
        if self.execution_mode == "shell" and action_type in NETWORK_ACTIONS:
            return "BLOCKED_FOR_SAFETY"
        if action_type in NETWORK_ACTIONS:
            return "PLANNED"
        if self.execution_mode == "simulated":
            return "SIMULATED_STATE_RECORDED"
        return "STATE_UPDATED"

    def _state_updates(self, strategy_id: str, action: Dict, context: Dict, timestamp: str) -> Dict:
        action_type = str(action.get("type", "unknown"))
        if action_type == "monitor_only":
            return {
                "last_monitored_window": {
                    "window_id": context.get("window_id"),
                    "attack_type": context.get("attack_type", "UNKNOWN"),
                    "rows": context.get("rows"),
                    "detector_source": context.get("detector_source"),
                    "strategy_id": strategy_id,
                    "timestamp": timestamp,
                }
            }
        if action_type == "log_enrich":
            return {"log_fields_add": [str(field) for field in action.get("fields", [])]}
        if action_type == "switch_model":
            return {"current_model": str(action.get("target_model", "unknown_model"))}
        if action_type == "raise_threshold":
            return {
                "threshold": {
                    "metric": str(action.get("metric", "unknown_metric")),
                    "value": action.get("value"),
                }
            }
        return {}

    def _build_plan(self, strategy_id: str, action: Dict, context: Dict, timestamp: str) -> Dict:
        action_type = str(action.get("type", "unknown"))
        planned_commands: List[str] = []
        intents: List[Dict] = []

        if action_type == "monitor_only":
            planned_commands.append(
                "monitor window=%s attack_type=%s"
                % (context.get("window_id", -1), context.get("attack_type", "UNKNOWN"))
            )
            intents.append({"type": "monitor", "window_id": context.get("window_id")})
        elif action_type == "log_enrich":
            fields = [str(field) for field in action.get("fields", [])]
            planned_commands.append("enable_log_enrichment fields=%s" % ",".join(fields))
            intents.append({"type": "log_enrich", "fields": fields})
        elif action_type == "switch_model":
            target_model = str(action.get("target_model", "unknown_model"))
            planned_commands.append("set_detection_model --model %s" % target_model)
            intents.append({"type": "switch_model", "target_model": target_model})
        elif action_type == "raise_threshold":
            metric = str(action.get("metric", "unknown_metric"))
            value = action.get("value", "unknown_value")
            planned_commands.append("set_threshold --metric %s --value %s" % (metric, value))
            intents.append({"type": "raise_threshold", "metric": metric, "value": value})
        elif action_type == "rate_limit":
            scope = str(action.get("scope", "suspicious_flow"))
            value = str(action.get("value", "20mbit"))
            planned_commands.append(
                "tc qdisc replace dev eth0 root tbf rate %s burst 32kbit latency 400ms # scope=%s"
                % (value, scope)
            )
            intents.append({"type": "openflow_meter", "scope": scope, "rate": value})
        elif action_type == "isolate_flow":
            scope = str(action.get("scope", "suspicious_flow"))
            planned_commands.append("ovs-ofctl add-flow br0 priority=100,ip,actions=drop # scope=%s" % scope)
            intents.append({"type": "acl_drop", "scope": scope})
        else:
            planned_commands.append("unsupported_action")
            intents.append({"type": "unsupported", "action_type": action_type})

        return {
            "timestamp": timestamp,
            "execution_mode": self.execution_mode,
            "strategy_id": strategy_id,
            "action_type": action_type,
            "action": action,
            "context": context,
            "planned_commands": planned_commands,
            "intents": intents,
        }

    def _load_state(self) -> Dict:
        if not self.state_path.exists():
            return {
                "last_monitored_window": None,
                "log_fields": [],
                "current_model": None,
                "thresholds": {},
                "last_actions": [],
            }
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return {
                "last_monitored_window": None,
                "log_fields": [],
                "current_model": None,
                "thresholds": {},
                "last_actions": [],
            }

    def _write_state(self, state: Dict) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    def _append_plan(self, plan: Dict, status: str) -> None:
        self.plan_path.parent.mkdir(parents=True, exist_ok=True)
        record = dict(plan)
        record["status"] = status
        record["system_command_executed"] = False
        with self.plan_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _apply_state_updates(
        self,
        state: Dict,
        updates: Dict,
        strategy_id: str,
        action: Dict,
        context: Dict,
        status: str,
        timestamp: str,
    ) -> None:
        state.setdefault("log_fields", [])
        state.setdefault("thresholds", {})
        state.setdefault("last_actions", [])

        if "last_monitored_window" in updates:
            state["last_monitored_window"] = updates["last_monitored_window"]
        if "log_fields_add" in updates:
            existing = [str(field) for field in state.get("log_fields", [])]
            for field in updates["log_fields_add"]:
                if field not in existing:
                    existing.append(field)
            state["log_fields"] = existing
        if "current_model" in updates:
            state["current_model"] = updates["current_model"]
        if "threshold" in updates:
            threshold = updates["threshold"]
            metric = str(threshold.get("metric", "unknown_metric"))
            state["thresholds"][metric] = threshold.get("value")

        state["updated_at"] = timestamp
        state["execution_mode"] = self.execution_mode
        state["last_actions"].append(
            {
                "timestamp": timestamp,
                "strategy_id": strategy_id,
                "action_type": str(action.get("type", "unknown")),
                "status": status,
                "window_id": context.get("window_id"),
            }
        )
        state["last_actions"] = state["last_actions"][-100:]
