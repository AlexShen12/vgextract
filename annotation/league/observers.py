"""Stable intake entry point and legacy timestamp helpers for old artifacts."""
from __future__ import annotations

from pathlib import Path
from .core import Store, digest, read_json


def _clock_seconds(value):
    if isinstance(value, dict):
        value = value.get("displayed_text", value.get("seconds"))
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    if isinstance(value, str):
        parts = value.split(":")
        if 2 <= len(parts) <= 3 and all(part.isdigit() for part in parts):
            result = 0
            for part in parts:
                result = result*60+int(part)
            return result
    return None


def transition_reason(previous, row):
    old, new = previous["value"], row["value"]
    if digest(old) == digest(new):
        return None
    field = row["field"]
    if field in {"cs", "displayed_gold"}:
        return None  # Routine displayed-counter updates are already sampled.
    if field in {"game_clock", "objective_timer", "death_timer"}:
        a, b = _clock_seconds(old), _clock_seconds(new)
        if a is not None and b is not None:
            direction = 1 if field == "game_clock" else -1
            if abs((b-a)-direction*(row["time"]-previous["time"])) <= 2:
                return None
            return "clock_discontinuity" if field == "game_clock" else "timer_transition"
    if field == "position" and isinstance(old, dict) and isinstance(new, dict):
        if old.get("region") == new.get("region"):
            return None
    if isinstance(old, dict) and isinstance(new, dict) and old.get("state") == new.get("state") == "cooling_down":
        if isinstance(old.get("seconds"), (int, float)) and isinstance(new.get("seconds"), (int, float)):
            if 0 < new["seconds"] <= old["seconds"]:
                return None
    return "state_transition"


def _spent(task_dir: Path) -> dict:
    # Include unsuccessful lookup rounds and interrupted tasks in the cost ledger.
    metrics = [read_json(p)["metrics"] for p in task_dir.rglob("completion.json")]
    return {**{key: sum(m.get(key, 0) for m in metrics)
               for key in ("visual_tokens", "input_tokens", "output_tokens", "wall_seconds")},
            "peak_gpu_memory_bytes": max((m.get("peak_gpu_memory_bytes", 0) for m in metrics), default=0)}


def annotate(store: Store, config, runner=None, resolver=None, *, retry_failed=False, sample_times=None, pilot=False) -> dict:
    """Production intake uses restricted VL transcription and calibrated numeric fields."""
    from .intake import annotate as calibrated_intake
    return calibrated_intake(store, config, retry_failed=retry_failed, sample_times=sample_times, pilot=pilot)
