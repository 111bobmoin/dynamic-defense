#!/usr/bin/env python3
"""Standalone status server for the dynamic_defense_ceni sidecar."""

from __future__ import annotations

import argparse
import json
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from action_logger import log_action


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 18188
DEFAULT_INPUT = "/tmp/optimize_multi_vm_runtime/defense_inputs/dynamic_defense.json"
DEFAULT_LOG_FILE = "logs/dynamic_defense_ceni_actions.jsonl"
SERVICE_NAME = "dynamic_defense_ceni_status_server"
STATIC_HTML = Path(__file__).resolve().parent / "static_status.html"
VERSION = "v1.0.1-dynamic-defense-ceni"
DEFAULT_MODEL_INFO = {
    "detector_model_name": "FlowMLP family_v3 策略族分类器",
    "detector_model_path": "models/torch_flow_classifier_expanded_family_v3.pt",
    "detector_meta_path": "models/torch_flow_classifier_expanded_family_v3_meta.json",
    "label_mode": "family",
    "detector_mode": "hybrid",
    "optimizer": "actor_critic",
    "optimizer_model_path": "models/actor_critic_policy.pt",
    "execution_mode": "REST/stateful 计划生成与状态更新",
    "version": VERSION,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve dynamic_defense_ceni sidecar status.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Listen host.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Listen port.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to dynamic_defense.json.")
    parser.add_argument("--log-file", default=DEFAULT_LOG_FILE, help="JSONL action log path.")
    return parser.parse_args(argv)


def safe_log_action(
    log_file: str | Path,
    action: str,
    status: str,
    message: str = "",
    inputs: dict[str, Any] | None = None,
    outputs: dict[str, Any] | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    try:
        log_action(log_file, action, status, message, inputs=inputs, outputs=outputs, details=details)
    except Exception as exc:  # pragma: no cover - logging must not crash the server
        print(f"{SERVICE_NAME}: failed to write action log: {exc}", file=sys.stderr)


def load_status_payload(input_path: str | Path) -> dict[str, Any]:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"input file does not exist: {path}")

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError("status input must be a JSON object")

    return payload


def normalize_model_info(payload: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    model_info = dict(DEFAULT_MODEL_INFO)
    candidate = payload.get("model_info")
    if isinstance(candidate, dict):
        model_info.update(candidate)

    if metrics.get("detector") and not model_info.get("detector_mode"):
        model_info["detector_mode"] = metrics["detector"]
    if metrics.get("optimizer") and not model_info.get("optimizer"):
        model_info["optimizer"] = metrics["optimizer"]
    model_info["version"] = str(model_info.get("version") or payload.get("version") or VERSION)
    return model_info


def extract_status(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}

    attack_type_accuracy = metrics.get("attack_type_accuracy")
    if not isinstance(attack_type_accuracy, dict):
        attack_type_accuracy = {}

    model_info = normalize_model_info(payload, metrics)

    return {
        "status": payload.get("status"),
        "severity": payload.get("severity"),
        "risk_score": payload.get("risk_score"),
        "source": payload.get("source"),
        "version": payload.get("version"),
        "updated_at": payload.get("updated_at"),
        "summary": payload.get("summary"),
        "message": payload.get("message"),
        "scenario": payload.get("scenario"),
        "event_type": payload.get("event_type"),
        "recommendation": payload.get("recommendation"),
        "affected_links": payload.get("affected_links", []),
        "affected_nodes": payload.get("affected_nodes", []),
        "actions": payload.get("actions", []),
        "alerts": payload.get("alerts", []),
        "metrics": metrics,
        "detector": metrics.get("detector"),
        "optimizer": metrics.get("optimizer"),
        "windows": metrics.get("windows"),
        "adjustment_events": metrics.get("adjustment_events"),
        "attack_type_accuracy": attack_type_accuracy,
        "attack_type_accuracy_family": attack_type_accuracy.get("family"),
        "strategy_match_accuracy": metrics.get("strategy_match_accuracy"),
        "detector_source_counts": metrics.get("detector_source_counts", {}),
        "model_info": model_info,
    }


def read_recent_logs(log_file: str | Path, limit: int = 50) -> list[dict[str, Any]]:
    path = Path(log_file)
    if not path.exists():
        return []

    lines = path.read_text(encoding="utf-8").splitlines()
    entries: list[dict[str, Any]] = []
    for line in lines[-limit:]:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            parsed = {"raw": stripped, "parse_error": str(exc)}
        entries.append(parsed if isinstance(parsed, dict) else {"value": parsed})
    return entries


def handle_api_request(
    route: str,
    input_path: str | Path,
    log_file: str | Path,
    method: str = "GET",
) -> tuple[HTTPStatus, Any]:
    safe_log_action(
        log_file,
        "request",
        "success",
        "Received status server request.",
        inputs={"method": method, "path": route},
    )

    if route == "/api/health":
        return HTTPStatus.OK, {"status": "ok", "service": SERVICE_NAME}

    if route == "/api/logs":
        return HTTPStatus.OK, read_recent_logs(log_file, limit=50)

    if route == "/api/status":
        try:
            payload = load_status_payload(input_path)
        except FileNotFoundError as exc:
            safe_log_action(
                log_file,
                "error",
                "failure",
                "Status input file is missing.",
                inputs={"input": str(input_path)},
                details={"error": str(exc)},
            )
            return HTTPStatus.NOT_FOUND, {
                "status": "error",
                "service": SERVICE_NAME,
                "error": "input_not_found",
                "message": str(exc),
                "input": str(input_path),
            }
        except Exception as exc:
            safe_log_action(
                log_file,
                "error",
                "failure",
                "Failed to read status input.",
                inputs={"input": str(input_path)},
                details={"error": str(exc)},
            )
            return HTTPStatus.INTERNAL_SERVER_ERROR, {
                "status": "error",
                "service": SERVICE_NAME,
                "error": "input_invalid",
                "message": str(exc),
                "input": str(input_path),
            }

        return HTTPStatus.OK, extract_status(payload)

    return HTTPStatus.NOT_FOUND, {"error": "not found", "path": route}


class DynamicDefenseCENIStatusHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address: tuple[str, int], input_path: str | Path, log_file: str | Path) -> None:
        super().__init__(server_address, DynamicDefenseCENIStatusHandler)
        self.input_path = str(input_path)
        self.log_file = str(log_file)


class DynamicDefenseCENIStatusHandler(BaseHTTPRequestHandler):
    server: DynamicDefenseCENIStatusHTTPServer

    def do_GET(self) -> None:
        route = urlparse(self.path).path
        try:
            if route == "/":
                safe_log_action(
                    self.server.log_file,
                    "request",
                    "success",
                    "Received status server request.",
                    inputs={"method": "GET", "path": route},
                )
                self.handle_index()
            elif route.startswith("/api/"):
                status_code, payload = handle_api_request(route, self.server.input_path, self.server.log_file)
                self.send_json(payload, status_code)
            else:
                safe_log_action(
                    self.server.log_file,
                    "request",
                    "success",
                    "Received status server request.",
                    inputs={"method": "GET", "path": route},
                )
                self.send_json({"error": "not found", "path": route}, HTTPStatus.NOT_FOUND)
        except Exception as exc:
            safe_log_action(
                self.server.log_file,
                "error",
                "failure",
                "Status server request failed.",
                inputs={"method": "GET", "path": route},
                details={"error": str(exc)},
            )
            self.send_json({"error": "request failed", "message": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def handle_index(self) -> None:
        try:
            content = STATIC_HTML.read_text(encoding="utf-8")
        except Exception as exc:
            safe_log_action(
                self.server.log_file,
                "error",
                "failure",
                "Failed to read static status page.",
                inputs={"path": str(STATIC_HTML)},
                details={"error": str(exc)},
            )
            self.send_json({"error": "static status page unavailable"}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        encoded = content.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def handle_status(self) -> None:
        try:
            payload = load_status_payload(self.server.input_path)
        except FileNotFoundError as exc:
            safe_log_action(
                self.server.log_file,
                "error",
                "failure",
                "Status input file is missing.",
                inputs={"input": self.server.input_path},
                details={"error": str(exc)},
            )
            self.send_json(
                {
                    "status": "error",
                    "service": SERVICE_NAME,
                    "error": "input_not_found",
                    "message": str(exc),
                    "input": self.server.input_path,
                },
                HTTPStatus.NOT_FOUND,
            )
            return
        except Exception as exc:
            safe_log_action(
                self.server.log_file,
                "error",
                "failure",
                "Failed to read status input.",
                inputs={"input": self.server.input_path},
                details={"error": str(exc)},
            )
            self.send_json(
                {
                    "status": "error",
                    "service": SERVICE_NAME,
                    "error": "input_invalid",
                    "message": str(exc),
                    "input": self.server.input_path,
                },
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return

        self.send_json(extract_status(payload))

    def handle_logs(self) -> None:
        self.send_json(read_recent_logs(self.server.log_file, limit=50))

    def send_json(self, payload: Any, status_code: int = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: Any) -> None:
        return


def create_server(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    input_path: str | Path = DEFAULT_INPUT,
    log_file: str | Path = DEFAULT_LOG_FILE,
) -> DynamicDefenseCENIStatusHTTPServer:
    server = DynamicDefenseCENIStatusHTTPServer((host, port), input_path=input_path, log_file=log_file)
    safe_log_action(
        log_file,
        "startup",
        "success",
        "dynamic_defense_ceni status server starting.",
        inputs={
            "host": host,
            "port": server.server_address[1],
            "input": str(input_path),
            "log_file": str(log_file),
        },
    )
    return server


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    server = create_server(args.host, args.port, args.input, args.log_file)
    print(f"{SERVICE_NAME} listening on http://{server.server_address[0]}:{server.server_address[1]}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 130
    finally:
        safe_log_action(
            args.log_file,
            "exit",
            "success",
            "dynamic_defense_ceni status server exiting.",
            inputs={"host": args.host, "port": server.server_address[1]},
        )
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
