from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import requests


@dataclass
class ActionExecutionResult:
    action_type: str
    status: str
    detail: str
    timestamp: str


class CeniActionAdapter:
    """防御动作适配层。

    默认 dry-run，只记录动作，不改变网络。部署到 CENI VM 后，可使用：
    1) REST 模式：向你们自己的 Ryu/ONOS/控制器 API POST 动作；
    2) local 模式：调用本机白名单脚本，例如 tc/ovs/iptables 封装脚本。

    注意：公开 CENI 文档主要描述门户式资源编排，未承诺通用 REST API。
    因此这里不直接假设 CENI 官方 API，而是把 CENI 上的控制器或 VM 作为执行端。
    """

    def __init__(self, mode: str = "dry_run", endpoint: Optional[str] = None, token: Optional[str] = None):
        self.mode = mode
        self.endpoint = endpoint or os.environ.get("CENI_CONTROLLER_ENDPOINT")
        self.token = token or os.environ.get("CENI_CONTROLLER_TOKEN")

    def execute_actions(self, strategy_id: str, actions: List[Dict], context: Dict) -> List[ActionExecutionResult]:
        results = []
        for action in actions:
            if self.mode == "rest":
                results.append(self._execute_rest(strategy_id, action, context))
            elif self.mode == "local":
                results.append(self._execute_local(strategy_id, action, context))
            else:
                results.append(
                    ActionExecutionResult(
                        action_type=str(action.get("type", "unknown")),
                        status="DRY_RUN",
                        detail=json.dumps({"strategy_id": strategy_id, "action": action, "context": context}, ensure_ascii=False),
                        timestamp=datetime.utcnow().isoformat(),
                    )
                )
        return results

    def _execute_rest(self, strategy_id: str, action: Dict, context: Dict) -> ActionExecutionResult:
        if not self.endpoint:
            return ActionExecutionResult(str(action.get("type", "unknown")), "SKIPPED", "missing CENI_CONTROLLER_ENDPOINT", datetime.utcnow().isoformat())
        payload = {"strategy_id": strategy_id, "action": action, "context": context}
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        try:
            resp = requests.post(self.endpoint.rstrip("/") + "/defense/action", json=payload, headers=headers, timeout=3)
            return ActionExecutionResult(str(action.get("type", "unknown")), "OK" if resp.ok else "ERROR", resp.text[:500], datetime.utcnow().isoformat())
        except Exception as exc:
            return ActionExecutionResult(str(action.get("type", "unknown")), "ERROR", repr(exc), datetime.utcnow().isoformat())

    def _execute_local(self, strategy_id: str, action: Dict, context: Dict) -> ActionExecutionResult:
        # 只允许调用明确白名单脚本，防止策略配置变成任意命令执行。
        action_type = str(action.get("type", "unknown"))
        script_map = {
            "rate_limit": os.environ.get("DD_RATE_LIMIT_SCRIPT"),
            "isolate_flow": os.environ.get("DD_ISOLATE_SCRIPT"),
            "switch_model": os.environ.get("DD_SWITCH_MODEL_SCRIPT"),
        }
        script = script_map.get(action_type)
        if not script:
            return ActionExecutionResult(action_type, "SKIPPED", "no whitelist script configured", datetime.utcnow().isoformat())
        try:
            payload = json.dumps({"strategy_id": strategy_id, "action": action, "context": context}, ensure_ascii=False)
            completed = subprocess.run([script, payload], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
            status = "OK" if completed.returncode == 0 else "ERROR"
            detail = (completed.stdout + completed.stderr)[:500]
            return ActionExecutionResult(action_type, status, detail, datetime.utcnow().isoformat())
        except Exception as exc:
            return ActionExecutionResult(action_type, "ERROR", repr(exc), datetime.utcnow().isoformat())
