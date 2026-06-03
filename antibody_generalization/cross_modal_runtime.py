from __future__ import annotations

import threading
import time
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import Any

from .runtime import StepPrerequisiteError

RUN_LOCK = threading.Lock()
RUN_STATE: dict[str, Any] = {
    "active_step": None,
    "status": "idle",
    "progress": 0,
    "started_at": None,
    "finished_at": None,
    "logs": [],
    "artifact": None,
    "completed_steps": [],
    "step_runs": {},
    "error": None,
}


def utc_now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def _empty_state() -> dict[str, Any]:
    return {
        "active_step": None,
        "status": "idle",
        "progress": 0,
        "started_at": None,
        "finished_at": None,
        "logs": [],
        "artifact": None,
        "completed_steps": [],
        "step_runs": {},
        "error": None,
    }


def reset_cross_modal_run() -> dict[str, Any]:
    with RUN_LOCK:
        RUN_STATE.update(_empty_state())
        return deepcopy(RUN_STATE)


def get_cross_modal_run_state() -> dict[str, Any]:
    with RUN_LOCK:
        return deepcopy(RUN_STATE)


def _artifact_for_step(step: dict[str, Any]) -> dict[str, Any]:
    return dict(step.get("artifact") or {"summary": step.get("primary", step.get("summary", "完成"))})


def _missing_previous_steps(step_key: str, steps: list[dict[str, Any]], completed_steps: list[str]) -> list[str]:
    ordered_keys = [item["key"] for item in steps]
    step_index = ordered_keys.index(step_key)
    completed = set(completed_steps)
    return [key for key in ordered_keys[:step_index] if key not in completed]


def _run_step(step: dict[str, Any]) -> None:
    step_key = step["key"]
    duration = max(float(step.get("duration", 1.5)), 0.8)
    process_logs = sorted(step.get("process_logs", []), key=lambda item: item[0])
    emitted = set()
    started = time.time()
    with RUN_LOCK:
        started_at = utc_now_iso()
        step_runs = dict(RUN_STATE.get("step_runs", {}))
        step_runs[step_key] = {
            "status": "running",
            "progress": 0,
            "started_at": started_at,
            "finished_at": None,
            "logs": [],
            "artifact": None,
            "error": None,
        }
        RUN_STATE.update(
            {
                "active_step": step_key,
                "status": "running",
                "progress": 0,
                "started_at": started_at,
                "finished_at": None,
                "logs": [],
                "artifact": None,
                "step_runs": step_runs,
                "error": None,
            }
        )

    try:
        while True:
            elapsed = time.time() - started
            progress = min(int((elapsed / duration) * 100), 100)
            new_logs = []
            for second, message in process_logs:
                if elapsed >= float(second) and second not in emitted:
                    emitted.add(second)
                    new_logs.append({"time": utc_now_iso(), "message": message})
            with RUN_LOCK:
                RUN_STATE["progress"] = progress
                RUN_STATE["logs"].extend(new_logs)
                RUN_STATE["step_runs"][step_key]["progress"] = progress
                RUN_STATE["step_runs"][step_key]["logs"].extend(new_logs)
            if progress >= 100:
                break
            time.sleep(0.25)

        with RUN_LOCK:
            completed = list(RUN_STATE.get("completed_steps", []))
            if step_key not in completed:
                completed.append(step_key)
            finished_at = utc_now_iso()
            artifact = _artifact_for_step(step)
            RUN_STATE["step_runs"][step_key].update(
                {
                    "status": "completed",
                    "progress": 100,
                    "finished_at": finished_at,
                    "artifact": artifact,
                    "error": None,
                }
            )
            RUN_STATE.update(
                {
                    "status": "completed",
                    "progress": 100,
                    "finished_at": finished_at,
                    "artifact": artifact,
                    "completed_steps": completed,
                }
            )
    except Exception as exc:  # noqa: BLE001
        with RUN_LOCK:
            finished_at = utc_now_iso()
            if step_key in RUN_STATE.get("step_runs", {}):
                RUN_STATE["step_runs"][step_key].update(
                    {"status": "error", "error": str(exc), "finished_at": finished_at}
                )
            RUN_STATE.update({"status": "error", "error": str(exc), "finished_at": finished_at})


def start_cross_modal_step(step_key: str, steps: list[dict[str, Any]]) -> dict[str, Any]:
    step = next((item for item in steps if item["key"] == step_key), None)
    if not step:
        raise KeyError(f"unknown cross-modal demo step: {step_key}")
    with RUN_LOCK:
        if RUN_STATE["status"] == "running":
            return deepcopy(RUN_STATE)
        missing_steps = _missing_previous_steps(step_key, steps, RUN_STATE.get("completed_steps", []))
        if missing_steps:
            raise StepPrerequisiteError(step_key, missing_steps)
    thread = threading.Thread(target=_run_step, args=(step,), daemon=True)
    thread.start()
    time.sleep(0.05)
    return get_cross_modal_run_state()
