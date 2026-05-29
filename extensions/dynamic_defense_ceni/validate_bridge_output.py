#!/usr/bin/env python3
"""Validate dynamic_defense_ceni bridge output."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from action_logger import log_action


DEFAULT_LOG_FILE = "logs/dynamic_defense_ceni_actions.jsonl"
SOURCE = "dynamic_defense_ceni"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate dynamic_defense_ceni bridge output.")
    parser.add_argument("--input", required=True, help="Path to generated dynamic_defense.json.")
    parser.add_argument("--log-file", default=DEFAULT_LOG_FILE, help="JSONL action log path.")
    return parser.parse_args(argv)


def load_payload(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("payload must be a JSON object")
    return data


def validate_payload(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    if not payload.get("status"):
        errors.append("status is required")

    risk_score = payload.get("risk_score")
    if not isinstance(risk_score, (int, float)) or isinstance(risk_score, bool):
        errors.append("risk_score must be a number")
    elif risk_score < 0 or risk_score > 100:
        errors.append("risk_score must be between 0 and 100")

    if not isinstance(payload.get("affected_links"), list):
        errors.append("affected_links must be a list")
    if not isinstance(payload.get("affected_nodes"), list):
        errors.append("affected_nodes must be a list")
    if not isinstance(payload.get("alerts"), list):
        errors.append("alerts must be a list")
    if payload.get("source") != SOURCE:
        errors.append("source must be dynamic_defense_ceni")
    if not payload.get("version"):
        errors.append("version is required")

    return errors


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = Path(args.input)

    try:
        payload = load_payload(input_path)
        errors = validate_payload(payload)
    except Exception as exc:
        errors = [str(exc)]
        payload = {}

    passed = not errors
    result = "PASS" if passed else "FAIL"
    print(result)
    for error in errors:
        print(error)

    log_action(
        args.log_file,
        "validate",
        "success" if passed else "failure",
        "Bridge output validation passed." if passed else "Bridge output validation failed.",
        inputs={"input": str(input_path)},
        outputs={"result": result},
        details={"errors": errors, "source": payload.get("source"), "version": payload.get("version")},
    )

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
