from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


RESULTS_ROOT = Path("outputs") / "results"
KNOWN_STAGE_SUFFIXES = ("_flows", "_predictions", "_node_summary", "_repair_order")


def sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return cleaned or "result"


def derive_input_name(input_path: Path) -> str:
    raw_name = input_path.stem if input_path.is_file() else input_path.name
    while True:
        stripped = raw_name
        for suffix in KNOWN_STAGE_SUFFIXES:
            if stripped.endswith(suffix):
                stripped = stripped[: -len(suffix)]
                break
        if stripped == raw_name:
            break
        raw_name = stripped
    return sanitize_name(raw_name)


def make_result_dir(input_path: Path, result_dir: Path | None = None) -> Path:
    if result_dir is not None:
        result_dir.mkdir(parents=True, exist_ok=True)
        return result_dir

    auto_dir = RESULTS_ROOT / f"{datetime.now():%Y%m%d_%H%M%S}_{derive_input_name(input_path)}"
    auto_dir.mkdir(parents=True, exist_ok=False)
    return auto_dir


def make_stage_dir(input_path: Path, stage: str, result_dir: Path | None = None) -> Path:
    stage_dir = make_result_dir(input_path, result_dir=result_dir) / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    return stage_dir


def default_stage_filename(input_path: Path, suffix: str) -> str:
    return f"{derive_input_name(input_path)}_{suffix}.csv"


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
