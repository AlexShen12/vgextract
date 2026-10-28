"""Versioned contracts, configuration and crash-safe run storage (CPU only)."""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import signal
import tempfile
import fcntl
from contextlib import contextmanager
from functools import wraps
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = 2
DEFAULT_RUN = "data/league_runs/midbeast-OLLZBe5-fg4-v2"
STATUSES = {"unreviewed", "human-verified", "rejected"}
VISIBILITY = {"visible", "unreadable", "not_shown", "occluded", "unknown"}
CAPABILITIES = ("episodic_retrieval", "persistent_state", "cumulative_counting",
                "event_linking", "trajectory_memory", "recurring_behavior")
EVENTS = {"attack", "cast", "channel", "movement", "purchase", "death", "recall",
          "state_transition", "encounter", "skirmish", "teamfight", "reset",
          "objective_contest", "objective_takedown", "tower_destruction"}
COMPOSITE_CRITERIA = {
    "encounter": {"opposing_entities_visible"},
    "skirmish": {"opposing_entities_visible", "damage_or_cast_visible"},
    "teamfight": {"multiple_allies_and_enemies_visible", "damage_or_cast_visible"},
    "objective_contest": {"objective_visible", "opposing_entities_visible", "objective_interaction_visible"},
    "objective_takedown": {"objective_death_or_notice_visible"},
    "tower_destruction": {"structure_destruction_or_notice_visible"},
    "reset": {"fountain_arrival_visible"},
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def uid(kind: str, *values: Any) -> str:
    return kind + "_" + digest(values)[:20]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".writing-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc.msg} "
                         f"(line {exc.lineno}, column {exc.colno})") from exc


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".writing-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps({"schema_version": SCHEMA, **row}, ensure_ascii=False,
                                        allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


@dataclass
class Route:
    name: str
    box: list[int]
    model: str
    fields: list[str]
    max_pixels: int = 262144

    def validate(self, width: int, height: int) -> None:
        if (len(self.box) != 4 or any(type(v) is not int for v in self.box)
                or not 0 <= self.box[0] < self.box[2] <= width
                or not 0 <= self.box[1] < self.box[3] <= height):
            raise ValueError(f"Invalid source-pixel box: {self.name}: {self.box}")
        if self.model not in {"instruct", "thinking"} or self.max_pixels < 1024:
            raise ValueError(f"Invalid reader/pixel budget: {self.name}")


@dataclass
class Config:
    patch: dict
    routes: list[Route]
    width: int = 1920
    height: int = 1080
    fps: float = 1.0
    window_seconds: float = 8.0
    overlap_seconds: float = 2.0
    review_fps: list[float] = field(default_factory=lambda: [4.0, 16.0])
    review_budget_fraction: float = 0.5
    quiet_audit_fraction: float = 0.05
    seed: int = 17
    max_questions: int = 60
    max_context: int = 32768
    instruct_output_tokens: int = 2048
    thinking_output_tokens: int = 8192
    solver_max_frames: int = 128
    solver_max_pixels: int = 2073600
    knowledge_queries: int = 3
    hud_frames_per_request: int = 2
    masks: list[list[int]] = field(default_factory=list)
    layout_profile: str = "domisum-1080p-v1"
    asset_cache: str = "data/league_assets"
    ocr_python: str | None = None
    mechanics: bool = False
    evaluation_enabled: bool = True
    intake_min_coverage: float = 0.7
    reader_workers: int = 4
    source_video: str = "data/youtube/lol_midbeast/OLLZBe5-fg4.mp4"
    intake_engine: str = "hybrid-v1"
    vl_python: str | None = None
    models: dict = field(default_factory=lambda: {
        "instruct": {"id": "Qwen/Qwen3-VL-8B-Instruct", "revision": "main"},
        "thinking": {"id": "Qwen/Qwen3-VL-8B-Thinking", "revision": "main"}})

    @classmethod
    def load(cls, path: Path) -> "Config":
        data = read_json(path)
        data["routes"] = [Route(**route) for route in data["routes"]]
        result = cls(**data)
        result.validate()
        return result

    def validate(self) -> None:
        from .knowledge import Patch
        if self.patch:
            Patch(**self.patch)
        elif self.mechanics:
            raise ValueError("Mechanics requires an explicitly established patch")
        if not 0 < self.overlap_seconds < self.window_seconds or self.fps <= 0:
            raise ValueError("Require positive FPS and 0 < overlap < window")
        if len(self.review_fps) != 2 or not self.fps < self.review_fps[0] < self.review_fps[1]:
            raise ValueError("Require exactly two increasing review FPS levels")
        if not 0 <= self.quiet_audit_fraction <= 1 or not 0 <= self.review_budget_fraction <= 2:
            raise ValueError("Invalid review/audit budget")
        if self.max_context <= max(self.instruct_output_tokens, self.thinking_output_tokens) + 1024:
            raise ValueError("Context must reserve space for visual inputs")
        if not 0 <= self.knowledge_queries <= 3 or self.max_questions < 1 or self.solver_max_frames < 2 or self.hud_frames_per_request < 1:
            raise ValueError("Invalid question/lookup/frame limits")
        if len({r.name for r in self.routes}) != len(self.routes):
            raise ValueError("Duplicate routes")
        for route in self.routes:
            route.validate(self.width, self.height)
        for mask in self.masks:
            Route("mask", mask, "instruct", []).validate(self.width, self.height)
        if self.evaluation_enabled and "thinking" not in self.models:
            raise ValueError("A Thinking checkpoint is required for optional solver evaluation")
        if not 0 < self.intake_min_coverage <= 1:
            raise ValueError("intake_min_coverage must be in (0, 1]")
        if type(self.reader_workers) is not int or not 1<=self.reader_workers<=8:
            raise ValueError("reader_workers must be an integer in [1, 8]")

    def to_dict(self) -> dict:
        return asdict(self)


def validate_observation(row: dict) -> dict:
    required = {"id", "task_id", "frame_id", "time", "route", "entity", "field", "value", "visibility"}
    if not required <= row.keys() or not finite(row["time"]) or row["time"] < 0:
        raise ValueError("Invalid observation fields/timestamp")
    if row["visibility"] not in VISIBILITY:
        raise ValueError("Invalid visibility")
    if not finite(row.get("confidence", 0)) or not 0 <= row.get("confidence", 0) <= 1:
        raise ValueError("Invalid confidence")
    for key in ("id", "task_id", "frame_id", "route", "entity", "field"):
        if not isinstance(row[key], str) or not row[key] or len(row[key]) > 256:
            raise ValueError(f"Invalid observation {key}")
    if row["visibility"] != "visible" and row["value"] is not None:
        raise ValueError("Nonvisible values must be null")
    if row["visibility"] == "visible" and row["value"] is None:
        raise ValueError("A null field is unknown/not shown, not a confirmed visible value")
    if row.get("schema_version", 1) >= 2:
        if not isinstance(row.get("slot_id"), str) or not row["slot_id"]:
            raise ValueError("Version-2 observations require a stable slot_id")
        if row.get("acceptance") not in {"accepted", "unknown", "absent"}:
            raise ValueError("Invalid intake acceptance")
        if row["acceptance"] == "accepted" and (row["visibility"] != "visible" or row["value"] is None):
            raise ValueError("Accepted readings must have a visible value")
    json.dumps(row, allow_nan=False)
    return row


def validate_question(row: dict) -> dict:
    if row.get("capability") not in CAPABILITIES or row.get("review_status") not in STATUSES:
        raise ValueError("Invalid question classification")
    if row.get("knowledge_category") not in {"visual_only", "mechanics_augmented"}:
        raise ValueError("Invalid knowledge category")
    start, end = row["interval"]
    if not finite(start) or not finite(end) or not 0 <= start <= end:
        raise ValueError("Invalid question interval")
    if not row.get("evidence_ids") or not isinstance(row.get("prompt"), str):
        raise ValueError("Questions require evidence and a prompt")
    answer_type, answer = row.get("answer_type"), row.get("answer")
    valid = {"number": finite(answer), "entity": isinstance(answer, str), "string": isinstance(answer, str),
             "sequence": isinstance(answer, list), "object": isinstance(answer, dict)}
    if answer_type not in valid or not valid[answer_type]:
        raise ValueError("Answer does not match its declared type")
    times = row.get("evidence_times", [])
    if not times or any(not finite(t) or not start <= t <= end for t in times):
        raise ValueError("Question interval must include its supporting evidence times")
    for interval in row.get("evidence_intervals", []):
        if len(interval) != 2 or not all(finite(t) for t in interval) or not start <= interval[0] <= interval[1] <= end:
            raise ValueError("Evidence interval outside question interval")
    json.dumps(row, allow_nan=False)
    return row


class StopRequested(RuntimeError):
    pass


class Store:
    """Single writer per run; atomic task commits are the resume boundary."""
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.stopping = False
        self._correction_snapshot = None

    def path(self, name: str) -> Path:
        return self.root / name

    def install_signals(self) -> None:
        def stop(*_):
            self.stopping = True
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)

    def check_stop(self) -> None:
        if self.stopping:
            raise StopRequested("Shutdown requested; committed tasks can be resumed")

    def rows(self, name: str) -> list[dict]:
        return read_rows(self.path(name + ".jsonl"))

    def write(self, name: str, rows: list[dict]) -> None:
        manifest = self.path("manifest.json")
        schema = read_json(manifest).get("schema_version", 1) if manifest.exists() else SCHEMA
        write_rows(self.path(name + ".jsonl"), [{"schema_version": schema, **row} for row in rows])

    def corrections(self) -> list[dict]:
        if self._correction_snapshot is not None:
            return self._correction_snapshot
        return [read_json(p) for p in sorted(self.path("corrections").glob("*.json"))]

    @contextmanager
    def review_snapshot(self):
        """Freeze a revision for one derived stage without blocking subsequent human edits."""
        if self._correction_snapshot is not None:
            yield
            return
        lock_path = self.path("corrections/.lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_SH)
            self._correction_snapshot = self.corrections()
        try:
            yield
        finally:
            self._correction_snapshot = None

    def effective(self, rows: list[dict], target: str) -> list[dict]:
        indexed = {r["id"]: dict(r) for r in rows}
        for correction in self.corrections():
            if correction["target"] == target and correction["record_id"] in indexed:
                indexed[correction["record_id"]].update(correction["changes"])
                indexed[correction["record_id"]]["review_status"] = correction["review_status"]
        return list(indexed.values())

    def fingerprint(self, config: Config) -> str:
        manifest = read_json(self.path("manifest.json"))
        if manifest.get("schema_version", 1) >= 2:
            from .intake_profile import load_profile
            return digest({"source": manifest["source"]["sha256"], "interval": manifest["interval"],
                "fps": config.fps, "masks": config.masks, "profile": load_profile(config.layout_profile),
                "engine": config.intake_engine,
                "readers": read_json(self.path("intake_runtime.json")) if self.path("intake_runtime.json").exists() else None,
                "implementation": implementation_hash(["intake.py", "hybrid_intake.py", "readers.py", "vl_reader.py", "numeric_reader.py", "playback.py", "intake_profile.py", "local_models.py"]),
                "schema": SCHEMA})
        observer_config = config.to_dict()
        for key in ("max_questions", "solver_max_frames", "solver_max_pixels"):
            observer_config.pop(key)
        return digest({"config": observer_config, "source": manifest["source"]["sha256"],
                       "interval": manifest["interval"], "model_revisions": {
                           role: {k: v for k, v in pin.items() if k != "snapshot"}
                           for role, pin in manifest.get("model_revisions", {}).items()},
                       "inference_runtime": {k: v for k, v in manifest.get("inference_runtime", {}).items() if k != "device"},
                       "schema": SCHEMA, "implementation": implementation_hash([
                           "core.py", "media.py", "knowledge.py", "inference.py", "observers.py", "prompts.py", "timeline.py"])})

    def stage_key(self, stage: str, config: Config, dependencies: list[str]) -> str:
        extra = {}
        if stage == "normalize":
            extra = {"implementation": implementation_hash(["normalization.py"])}
        elif stage == "generate-qa":
            extra = {"max_questions": config.max_questions, "mechanics": config.mechanics,
                     "mechanics_implementation":implementation_hash(["knowledge.py", "references.py"]) if config.mechanics else None,
                     "implementation": implementation_hash(["questions.py"])}
        elif stage == "fuse":
            extra = {"implementation": implementation_hash(["timeline.py", "normalization.py"])}
        elif stage in {"evaluate", "report"}:
            manifest = read_json(self.path("manifest.json"))
            extra = {"max_frames": config.solver_max_frames, "max_pixels": config.solver_max_pixels,
                     "output_tokens": config.thinking_output_tokens, "max_context": config.max_context,
                     "seed":config.seed,"knowledge_queries":config.knowledge_queries,
                     "evaluation_enabled":config.evaluation_enabled,
                     "model": config.models.get("thinking"),
                     "model_pin": manifest.get("model_revisions", {}).get("thinking"),
                     "runtime": manifest.get("inference_runtime"),
                     "implementation": implementation_hash(["evaluation.py", "inference.py", "prompts.py"])}
        corrections=self.corrections()
        if stage=="normalize":
            corrections=[c for c in corrections if c["target"]=="observations"]
        elif stage in {"fuse","generate-qa"}:
            corrections=[c for c in corrections if c["target"]!="questions"]
        return digest({"stage": stage, "run": self.fingerprint(config), "stage_configuration": extra,
                       "dependencies": {n: digest(self.rows(n)) for n in dependencies},
                       "corrections": corrections})

    def commit_stage(self, stage: str, key: str, **metadata) -> None:
        atomic_json(self.path(f"stages/{stage}.json"), {"key": key, "completed_at": now(), **metadata})


def implementation_hash(names=None) -> str:
    root = Path(__file__).parent
    return digest({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(root.glob("*.py")) if names is None or p.name in names})


def _asset_signature(cache: str) -> str | None:
    path = Path(cache)/"manifest.json"
    return digest(read_json(path)) if path.exists() else None


def runtime_info() -> dict:
    from importlib.metadata import distributions
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": {d.metadata["Name"]: d.version for d in distributions() if d.metadata["Name"]}}


def consistent_review(function):
    @wraps(function)
    def wrapped(store, *args, **kwargs):
        with store.review_snapshot():
            return function(store, *args, **kwargs)
    return wrapped
