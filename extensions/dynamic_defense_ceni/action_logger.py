#!/usr/bin/env python3
"""Append-only JSONL action logging for the CENI bridge."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


LOG_FIELDS = (
    "timestamp",
    "action",
    "status",
    "message",
    "inputs",
    "outputs",
    "details",
)


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp with a trailing Z."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _as_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return value if value is not None else {}


def log_action(
    log_file: str | Path,
    action: str,
    status: str,
    message: str = "",
    inputs: Mapping[str, Any] | None = None,
    outputs: Mapping[str, Any] | None = None,
    details: Mapping[str, Any] | None = None,
) -> None:
    """Append one UTF-8 JSONL action entry, creating the parent directory."""

    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": utc_timestamp(),
        "action": action,
        "status": status,
        "message": message,
        "inputs": dict(_as_mapping(inputs)),
        "outputs": dict(_as_mapping(outputs)),
        "details": dict(_as_mapping(details)),
    }

    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, default=str))
        handle.write("\n")
