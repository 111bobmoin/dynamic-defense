import importlib.util
import io
import json
from pathlib import Path

from src.dynamic_defense.action_executor import ActionExecutor


ROOT = Path(__file__).resolve().parents[1]


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_action_executor_updates_state_actions(tmp_path):
    state_path = tmp_path / "runtime" / "controller_state.json"
    plan_path = tmp_path / "reports" / "controller_execution_plan.jsonl"
    executor = ActionExecutor(
        execution_mode="stateful",
        state_path=str(state_path),
        plan_path=str(plan_path),
    )
    context = {"window_id": 7, "attack_type": "DDoS", "rows": 200, "detector_source": "hybrid"}

    monitor = executor.execute("s_test", {"type": "monitor_only"}, context)
    enrich = executor.execute("s_test", {"type": "log_enrich", "fields": ["src_ip", "dst_ip"]}, context)
    switch = executor.execute("s_test", {"type": "switch_model", "target_model": "mlp_v2"}, context)
    threshold = executor.execute("s_test", {"type": "raise_threshold", "metric": "flow_rate", "value": 0.85}, context)

    assert monitor["status"] == "STATE_UPDATED"
    assert enrich["status"] == "STATE_UPDATED"
    assert switch["status"] == "STATE_UPDATED"
    assert threshold["status"] == "STATE_UPDATED"

    state = _read_json(state_path)
    assert state["last_monitored_window"]["window_id"] == 7
    assert state["log_fields"] == ["src_ip", "dst_ip"]
    assert state["current_model"] == "mlp_v2"
    assert state["thresholds"]["flow_rate"] == 0.85
    assert len(plan_path.read_text(encoding="utf-8").splitlines()) == 4


def test_network_actions_only_generate_plans(tmp_path):
    for mode in ["simulated", "stateful"]:
        state_path = tmp_path / mode / "runtime" / "controller_state.json"
        plan_path = tmp_path / mode / "reports" / "controller_execution_plan.jsonl"
        executor = ActionExecutor(
            execution_mode=mode,
            state_path=str(state_path),
            plan_path=str(plan_path),
        )

        rate = executor.execute("s_test", {"type": "rate_limit", "scope": "suspicious_src", "value": "10mbit"}, {})
        isolate = executor.execute("s_test", {"type": "isolate_flow", "scope": "suspicious_src_dst"}, {})

        assert rate["status"] == "PLANNED"
        assert isolate["status"] == "PLANNED"
        assert rate["safety"]["system_command_executed"] is False
        assert isolate["safety"]["system_command_executed"] is False
        assert "tc qdisc replace" in rate["planned_commands"][0]
        assert "ovs-ofctl add-flow" in isolate["planned_commands"][0]

        records = [json.loads(line) for line in plan_path.read_text(encoding="utf-8").splitlines()]
        assert len(records) == 2
        assert all(record["system_command_executed"] is False for record in records)


def test_shell_network_actions_blocked_for_safety(tmp_path):
    executor = ActionExecutor(
        execution_mode="shell",
        state_path=str(tmp_path / "runtime" / "controller_state.json"),
        plan_path=str(tmp_path / "reports" / "controller_execution_plan.jsonl"),
    )

    result = executor.execute("s_test", {"type": "isolate_flow", "scope": "suspicious_src"}, {})

    assert result["status"] == "BLOCKED_FOR_SAFETY"
    assert result["safety"]["system_command_executed"] is False


class _FakeSocket:
    def __init__(self, payload):
        self._request = io.BytesIO(payload)
        self.response = io.BytesIO()

    def makefile(self, mode, buffering=None):
        if "r" in mode:
            return self._request
        return self.response

    def sendall(self, data):
        self.response.write(data)


def test_translating_controller_keeps_translation_and_executes(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "translating_defense_controller",
        str(ROOT / "scripts" / "translating_defense_controller.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.chdir(str(tmp_path))
    module.Handler.execution_mode = "stateful"

    payload = {
        "strategy_id": "s_test",
        "action": {"type": "rate_limit", "scope": "suspicious_src", "value": "10mbit"},
        "context": {"window_id": 3, "attack_type": "DDoS"},
    }
    body = json.dumps(payload).encode("utf-8")
    request = (
        b"POST /defense/action HTTP/1.1\r\n"
        b"Host: localhost\r\n"
        b"Content-Type: application/json\r\n"
        b"Content-Length: " + str(len(body)).encode("ascii") + b"\r\n"
        b"\r\n" + body
    )
    sock = _FakeSocket(request)

    module.Handler(sock, ("127.0.0.1", 1), object())
    raw_response = sock.response.getvalue()
    response_body = raw_response.split(b"\r\n\r\n", 1)[1]
    response = json.loads(response_body.decode("utf-8"))

    assert response["status"] == "OK"
    assert response["message"] == "action translated"
    assert response["translated"]["target_subsystem"] == "sdn_qos"
    assert response["execution_result"]["status"] == "PLANNED"
    assert response["execution_result"]["safety"]["system_command_executed"] is False

    assert (tmp_path / "reports" / "controller_actions.jsonl").exists()
    assert (tmp_path / "reports" / "controller_translated_actions.jsonl").exists()
    assert (tmp_path / "reports" / "controller_execution_plan.jsonl").exists()
    assert (tmp_path / "runtime" / "controller_state.json").exists()
