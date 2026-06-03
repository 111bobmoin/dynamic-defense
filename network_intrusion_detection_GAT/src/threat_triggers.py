from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ThreatTriggerEvent:
    input_name: str | None
    scenario_ids: tuple[str, ...]
    triggered_at: str | None


def _as_string_list(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    if not isinstance(values, list):
        raise ValueError(f"Expected a list, got: {type(values)!r}")
    return tuple(str(value).strip() for value in values if str(value).strip())


def _normalize_input_name(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def load_threat_triggers(path: Path) -> list[ThreatTriggerEvent]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    events = payload.get("events")
    if not isinstance(events, list):
        raise ValueError(f"Invalid threat trigger file, missing 'events' list: {path}")

    loaded: list[ThreatTriggerEvent] = []
    for index, raw in enumerate(events, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"Trigger event #{index} is not an object.")
        scenario_ids = _as_string_list(raw.get("scenario_ids"))
        if not scenario_ids:
            raise ValueError(f"Trigger event #{index} must include at least one scenario id.")
        loaded.append(
            ThreatTriggerEvent(
                input_name=_normalize_input_name(raw.get("input_name")),
                scenario_ids=scenario_ids,
                triggered_at=_normalize_input_name(raw.get("triggered_at")),
            )
        )
    return loaded


def resolve_triggered_scenario_ids(
    events: list[ThreatTriggerEvent],
    *,
    input_names: list[str],
) -> tuple[list[str], list[ThreatTriggerEvent]]:
    normalized_names = {str(name).strip() for name in input_names if str(name).strip()}
    matched_events: list[ThreatTriggerEvent] = []
    ordered_ids: list[str] = []
    seen_ids: set[str] = set()

    for event in events:
        if event.input_name is not None and event.input_name not in normalized_names:
            continue
        matched_events.append(event)
        for scenario_id in event.scenario_ids:
            if scenario_id in seen_ids:
                continue
            seen_ids.add(scenario_id)
            ordered_ids.append(scenario_id)

    return ordered_ids, matched_events
