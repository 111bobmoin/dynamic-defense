from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from datetime import datetime
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dynamic_defense.action_executor import ActionExecutor


def translate_action(payload):
    strategy_id = payload.get("strategy_id", "")
    action = payload.get("action", {})
    context = payload.get("context", {})

    action_type = action.get("type")
    attack_type = context.get("attack_type", "UNKNOWN")
    window_id = context.get("window_id", -1)

    if action_type == "monitor_only":
        translated = {
            "execution_mode": "SIMULATED",
            "target_subsystem": "monitoring",
            "translated_command": f"monitor window={window_id} attack_type={attack_type}",
            "description": "保持监控，不改变网络转发行为。"
        }

    elif action_type == "log_enrich":
        fields = ",".join(action.get("fields", []))
        translated = {
            "execution_mode": "SIMULATED",
            "target_subsystem": "logging",
            "translated_command": f"enable_log_enrichment fields={fields}",
            "description": "开启增强日志采集，用于攻击溯源和后续特征分析。"
        }

    elif action_type == "switch_model":
        target_model = action.get("target_model", "unknown_model")
        translated = {
            "execution_mode": "SIMULATED",
            "target_subsystem": "detector",
            "translated_command": f"set_detection_model --model {target_model}",
            "description": "通知检测服务切换或启用指定检测模型。"
        }

    elif action_type == "raise_threshold":
        metric = action.get("metric", "unknown_metric")
        value = action.get("value", "unknown_value")
        translated = {
            "execution_mode": "SIMULATED",
            "target_subsystem": "detector",
            "translated_command": f"set_threshold --metric {metric} --value {value}",
            "description": "调整检测阈值，用于持续攻击下的策略自适应优化。"
        }

    elif action_type == "rate_limit":
        scope = action.get("scope", "suspicious_flow")
        value = action.get("value", "20mbit")
        translated = {
            "execution_mode": "SIMULATED",
            "target_subsystem": "sdn_qos",
            "translated_command": f"tc qdisc replace dev eth0 root tbf rate {value} burst 32kbit latency 400ms # scope={scope}",
            "alternative_openflow": f"install OpenFlow meter for scope={scope}, rate={value}",
            "description": "对可疑源、可疑流或链路执行限速。"
        }

    elif action_type == "isolate_flow":
        scope = action.get("scope", "suspicious_flow")
        translated = {
            "execution_mode": "SIMULATED",
            "target_subsystem": "sdn_acl",
            "translated_command": f"ovs-ofctl add-flow br0 priority=100,ip,actions=drop # scope={scope}",
            "alternative_acl": f"install ACL drop rule for scope={scope}",
            "description": "隔离可疑流量，可映射为 OpenFlow drop、ACL 或重定向到隔离区。"
        }

    else:
        translated = {
            "execution_mode": "SIMULATED",
            "target_subsystem": "unknown",
            "translated_command": "unsupported_action",
            "description": f"不支持的动作类型: {action_type}"
        }

    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "strategy_id": strategy_id,
        "action_type": action_type,
        "context": context,
        "translated": translated
    }


class Handler(BaseHTTPRequestHandler):
    raw_log_file = Path("reports/controller_actions.jsonl")
    translated_log_file = Path("reports/controller_translated_actions.jsonl")
    execution_mode = "simulated"

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8") if length else "{}"

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"raw": raw}

        raw_record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "path": self.path,
            "payload": payload
        }

        translated_record = translate_action(payload)
        executor = ActionExecutor(execution_mode=self.execution_mode)
        execution_result = executor.execute(
            strategy_id=str(payload.get("strategy_id", "")),
            action=payload.get("action", {}),
            context=payload.get("context", {}),
        )

        self.raw_log_file.parent.mkdir(parents=True, exist_ok=True)

        with self.raw_log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(raw_record, ensure_ascii=False) + "\n")

        with self.translated_log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(translated_record, ensure_ascii=False) + "\n")

        response = {
            "status": "OK",
            "message": "action translated",
            "translated": translated_record["translated"],
            "execution_result": execution_result,
        }

        data = json.dumps(response, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        print("[%s] %s" % (datetime.now().isoformat(timespec="seconds"), fmt % args))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--execution-mode", choices=["simulated", "stateful", "shell"], default="simulated")
    args = parser.parse_args()

    Handler.execution_mode = args.execution_mode
    server = HTTPServer((args.host, args.port), Handler)
    print(f"translating defense controller listening on http://{args.host}:{args.port}")
    print(f"execution mode: {args.execution_mode}")
    print("raw logs: reports/controller_actions.jsonl")
    print("translated logs: reports/controller_translated_actions.jsonl")
    print("state: runtime/controller_state.json")
    print("execution plan: reports/controller_execution_plan.jsonl")
    server.serve_forever()


if __name__ == "__main__":
    main()
