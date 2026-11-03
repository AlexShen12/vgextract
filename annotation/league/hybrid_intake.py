"""Reduced production intake: one-pass numeric OCR, bars, and bounded VL text."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import logging
import math
import os
from pathlib import Path
import time

from .core import Store, atomic_json, digest, implementation_hash, read_json, uid
from .intake import (
    decoded_frames,
    evidence_crop,
    observation,
    subject,
    verify_readings,
    tesseract_signature,
    require_v2,
    collect,
)
from .intake_profile import load_profile, layout_matches
from .local_models import LocalVLRunner, worker_signature, LOCK
from .playback import PlaybackTracker
from .vl_reader import VLReader
from .readers import unknown
from .numeric_reader import NumericReader

DEFERRED = {
    "icons": "portrait/item/spell matching deferred",
    "ability_ranks": "deferred",
    "ability_cooldowns": "deferred",
    "minimap": "tracking not validated",
    "gameplay": "event experiments are not production evidence",
    "audio": "ASR experiment only",
}


def enabled_fields(profile):
    result = []
    for field in profile["fields"]:
        if field.get("enabled", True) and field["reader"] != "icon":
            backend = field.get(
                "backend",
                (
                    "vl"
                    if field["reader"] == "text" or field["parser"] == "health"
                    else "bar" if field["reader"] == "bar" else "tesseract"
                ),
            )
            if backend not in {"vl", "bar", "tesseract"}:
                raise ValueError(
                    "Production intake supports only VL, Tesseract and direct bars"
                )
            if backend == "tesseract" and field["parser"] not in {
                "clock",
                "timer",
                "gold",
                "integer",
                "level",
                "kda",
            }:
                continue
            result.append({**field, "backend": backend})
    if not result:
        raise ValueError("No enabled supported fields in this profile")
    return result


def check_engine(store):
    manifest = require_v2(store)
    previous = store.path("intake_runtime.json")
    if (
        manifest.get("intake_engine", "hybrid-v1") != "hybrid-v1"
        or (previous.exists() and read_json(previous).get("engine") != "hybrid-v1")
        or (not previous.exists() and any(store.path("tasks").glob("*.json")))
    ):
        raise ValueError(
            "Earlier extraction cannot be resumed with hybrid intake; prepare --run-dir data/league_runs/midbeast-OLLZBe5-fg4-v2"
        )


def runtime(store, config):
    check_engine(store)
    profile = load_profile(config.layout_profile)
    fields = enabled_fields(profile)
    python = config.vl_python or os.environ.get("LEAGUE_VL_PYTHON")
    signature = {
        "engine": "hybrid-v1",
        "model_pins": read_json(LOCK)["models"]["intake"],
        "tesseract": (
            tesseract_signature()
            if any(f["backend"] == "tesseract" for f in fields)
            else None
        ),
    }
    prepared = None
    if any(f["backend"] == "vl" for f in fields):
        if not python:
            raise ValueError("Set LEAGUE_VL_PYTHON to the isolated vLLM environment")
        prepared = worker_signature(python, config.asset_cache)
        signature["vl"] = prepared[1]
    from importlib.metadata import version

    signature["driver_packages"] = {
        name: version(name) for name in ("numpy", "Pillow", "av")
    }
    atomic_json(store.path("intake_runtime.json"), signature)
    return signature, prepared


def release_path(store, config, profile, suffix=""):
    manifest = require_v2(store)
    return (
        Path(config.asset_cache)
        / "releases"
        / (
            digest({"source": manifest["source"]["sha256"], "profile": profile})
            + suffix
            + ".json"
        )
    )


def preflight(store, config):
    from .intake import preview

    signature, prepared = runtime(store, config)
    result = {
        "runtime": signature,
        "layout": preview(store, config),
        "deferred": DEFERRED,
    }
    if prepared is not None:
        worker = LocalVLRunner(
            store,
            config.vl_python or os.environ["LEAGUE_VL_PYTHON"],
            config.asset_cache,
            signature=prepared,
        )
        try:
            worker.start()
            result["gpu_memory_peak_bytes"] = worker.peak_memory_bytes
        finally:
            worker.close()
    return result


def _panel_state(image, profile, field, layout_valid):
    if not layout_valid:
        return "unknown"
    panel = profile.get("panels", {}).get(field["route"])
    if not panel:
        return "present"
    for status in ("present", "absent"):
        anchors = panel.get(status + "_anchors", [])
        if (
            anchors
            and layout_matches(
                image, {**profile, "anchors": anchors, "required_anchors": len(anchors)}
            )[0]
        ):
            return status
    if panel.get("not_present_means_absent"):
        anchors = panel.get("present_anchors", [])
        if anchors:
            _, scores = layout_matches(
                image, {**profile, "anchors": anchors, "required_anchors": len(anchors)}
            )
            if not any(s["matched"] for s in scores):
                return "absent"
    return "unknown"


def pilot_windows(start, end):
    if end - start < 300:
        raise ValueError("A five-minute pilot needs at least five minutes of source")
    available = end - start - 300
    return [
        (start + i * (60 + available / 4), start + i * (60 + available / 4) + 60)
        for i in range(5)
    ]


def _require_release(store, config, profile, signature, key):
    from .intake_benchmark import release_policy

    release = release_path(store, config, profile)
    required = {
        "runtime": signature,
        "release_policy": release_policy(),
        "extraction_key": key,
    }
    record = read_json(release) if release.exists() else {}
    if (
        not profile.get("thresholds_frozen")
        or not record.get("passed")
        or any(record.get(k) != v for k, v in required.items())
    ):
        raise ValueError(
            "Full intake requires a passing frozen held-out intake-benchmark for this source/profile/runtime"
        )
    pilot_file = release_path(store, config, profile, "-pilot")
    pilot_record = read_json(pilot_file) if pilot_file.exists() else {}
    if not pilot_record.get("passed") or pilot_record.get("extraction_key") != key:
        raise ValueError("Full intake requires a passing five-minute runtime pilot")


def _sampling_targets(manifest, config, sample_times, pilot):
    start, end = manifest["interval"]
    selected = (
        sorted(set(float(t) for t in sample_times))
        if sample_times is not None
        else None
    )
    windows = pilot_windows(start, end) if pilot else []
    targets = [
        start + i / config.fps
        for i in range(math.ceil((end - start) * config.fps))
        if (
            selected is None
            or any(abs(start + i / config.fps - t) < 1e-6 for t in selected)
        )
        and (not pilot or any(a <= start + i / config.fps < b for a, b in windows))
    ]
    if selected is not None and (not selected or len(targets) != len(selected)):
        raise ValueError(
            "Benchmark times must be on the sampling grid inside the run interval"
        )
    return targets, selected


def _resume_tasks(store, key, targets):
    tasks = {}
    for path in store.path("tasks").glob("*.json"):
        task = read_json(path)
        if task.get("run_key") == key and any(
            abs(task["start"] - t) < 0.05 for t in targets
        ):
            tasks[task["id"]] = task
    return tasks


class _IntakeReader:
    """Per-frame evidence and reporting; the orchestrator owns worker lifetime."""

    def __init__(self, store, config, manifest, profile, signature, key, worker, pool):
        self.store, self.config, self.manifest = store, config, manifest
        self.profile, self.signature, self.key = profile, signature, key
        self.worker, self.pool = worker, pool
        self.fields = enabled_fields(profile)
        self.tracker = PlaybackTracker(manifest["source"]["sha256"], config.fps)
        self.numeric = NumericReader(profile["thresholds"]["number"])
        self.vl = VLReader(worker, store, config.seed) if worker is not None else None

    def cache_path(self, crop_hash, field):
        backend = field["backend"]
        runtime_key = {"vl": "vl", "tesseract": "tesseract", "bar": "driver_packages"}[backend]
        reader = "vl_reader.py" if backend == "vl" else "numeric_reader.py"
        cache_key = digest(
            {
                "crop": crop_hash,
                "settings": {k: v for k, v in field.items() if k not in {"box", "slot_id", "route"}},
                "runtime": self.signature[runtime_key],
                "implementation": implementation_hash([reader, "readers.py"]),
                "threshold": self.profile["thresholds"]["number"] if backend == "tesseract" else None,
                "seed": self.config.seed if backend == "vl" else None,
            }
        )
        return self.store.path("reader-cache/" + cache_key + "/result.json")

    def read_vl_groups(self, vl_pending):
        readings, cache_hits = [], 0
        # Group text separately from HP so changing HP cannot invalidate name requests.
        groups = defaultdict(list)
        for entry in vl_pending:
            groups[entry[0]["parser"]].append(entry)
        for parser, entries in groups.items():
            # Longleaf development probes showed grouped names could all
            # become not_shown despite legible single-crop transcription.
            group_size = 1 if parser == "name" else 4
            for offset in range(0, len(entries), group_size):
                group = entries[offset : offset + group_size]
                group_key = digest(
                    [str(entry[3].relative_to(self.store.root)) for entry in group]
                )
                directory = self.store.path("vl-requests/" + group_key)
                cached_group = directory / "result.json"
                if cached_group.exists():
                    results = read_json(cached_group)
                    cache_hits += len(group)
                else:
                    if self.vl is None:
                        raise ValueError(
                            "VL fields require the isolated local Qwen worker"
                        )
                    results = self.vl.read_group(
                        [(f, self.store.path(path)) for f, _, path, _ in group],
                        directory,
                    )
                    atomic_json(cached_group, results)
                for field, frame, path, unused in group:
                    reading = {
                        **results[field["id"]],
                        "cache_path": str(cached_group.relative_to(self.store.root)),
                    }
                    readings.append((field, reading, frame))
        return readings, cache_hits

    def read_fields(self, image, task, source_id):
        timestamp, layout_valid = task["start"], task["layout_valid"]
        readings, vl_pending, futures, cache_hits = [], [], [], 0
        for field in self.fields:
            self.store.check_stop()
            crop = image.crop(field["box"])
            crop_hash, path = evidence_crop(self.store, crop)
            frame = {
                "id": uid("frame", source_id, field["box"], crop_hash),
                "source_id": source_id,
                "time": timestamp,
                "route": field["route"],
                "box": field["box"],
                "path": path,
            }
            task["frames"].append(frame)
            state = _panel_state(image, self.profile, field, layout_valid)
            if state != "present":
                reading = (
                    unknown("unsupported or occluded layout/panel")
                    if state == "unknown"
                    else {
                        "acceptance": "absent",
                        "reason": "calibrated absent-panel anchors",
                        "raw_text": None,
                        "value": None,
                    }
                )
                readings.append((field, reading, frame))
                continue
            cached = self.cache_path(crop_hash, field)
            if field["backend"] == "vl":
                vl_pending.append((field, frame, path, cached))
            elif cached.exists():
                readings.append(
                    (
                        field,
                        {
                            **read_json(cached),
                            "cache_path": str(cached.relative_to(self.store.root)),
                        },
                        frame,
                    )
                )
                cache_hits += 1
            else:

                def read_one(crop=crop, field=field, cached=cached):
                    begin = time.monotonic()
                    result = self.numeric.read(crop, field, cached.parent)
                    result["wall_seconds"] = time.monotonic() - begin
                    atomic_json(cached, result)
                    return result

                futures.append((field, frame, cached, self.pool.submit(read_one)))
        vl_readings, vl_hits = self.read_vl_groups(vl_pending)
        readings.extend(vl_readings)
        cache_hits += vl_hits
        for field, frame, cached, future in futures:
            readings.append(
                (
                    field,
                    {
                        **future.result(),
                        "cache_path": str(cached.relative_to(self.store.root)),
                    },
                    frame,
                )
            )
        return readings, cache_hits

    def read(self, sample_time, timestamp, image):
        task_id = uid("task", self.key, timestamp)
        end = self.manifest["interval"][1]
        tick = time.monotonic()
        token_start = dict(getattr(self.worker, "tokens", {}))
        if self.config.masks:
            from PIL import ImageDraw

            draw = ImageDraw.Draw(image)
            for box in self.config.masks:
                draw.rectangle(box, fill="black")
        layout_valid, anchors = layout_matches(image, self.profile)
        source_id = uid("source", self.manifest["source"]["sha256"], timestamp)
        task = {
            "schema_version": 2,
            "id": task_id,
            "run_key": self.key,
            "runtime_id": digest(self.signature),
            "profile_hash": digest(self.profile),
            "route": "intake",
            "start": timestamp,
            "sample_time": sample_time,
            "end": min(timestamp + 1 / self.config.fps, end),
            "fps": self.config.fps,
            "pass": 0,
            "frames": [],
            "observations": [],
            "issues": [],
            "layout_valid": layout_valid,
        }
        atomic_json(
            self.store.path("intake-runtimes/" + digest(self.signature) + ".json"), self.signature
        )
        atomic_json(
            self.store.path("layout_profiles/" + digest(self.profile) + ".json"), self.profile
        )
        if not layout_valid:
            _, diagnostic = evidence_crop(self.store, image)
            atomic_json(
                self.store.path(f"layout-diagnostics/{task_id}.json"),
                {"time": timestamp, "path": diagnostic, "anchors": anchors},
            )
        readings, cache_hits = self.read_fields(image, task, source_id)
        verify_readings(readings)
        clock = next(
            (
                r["value"].get("seconds")
                for f, r, _ in readings
                if f["field"] == "game_clock"
                and r["acceptance"] == "accepted"
                and isinstance(r.get("value"), dict)
            ),
            None,
        )
        playback = self.tracker.update(timestamp, clock, image, layout_valid)
        identities = {
            f["slot_id"]: r["value"]
            for f, r, _ in readings
            if f["field"] in {"identity", "champion_name", "objective_name"}
            and r["acceptance"] == "accepted"
        }
        for field, reading, frame in readings:
            row = observation(
                task,
                frame,
                field,
                reading,
                subject(field, identities),
                frame["path"],
            )
            row.update(
                playback,
                evidence_modality="visual",
                backend=field["backend"],
                sample_time=sample_time,
            )
            task["observations"].append(row)
            if row["acceptance"] == "unknown":
                task["issues"].append(
                    {
                        "id": uid("issue", task_id, field["id"]),
                        "time": timestamp,
                        "route": field["route"],
                        "slot_id": field["slot_id"],
                        "field_id": field["id"],
                        "kind": "unreadable" if layout_valid else "layout_mismatch",
                        "detail": reading["reason"],
                        "frame_id": frame["id"],
                    }
                )
            if field["field"] == "kda" and row["acceptance"] == "accepted":
                for name, value in row["value"].items():
                    task["observations"].append(
                        {
                            **row,
                            "id": uid("obs", task_id, field["id"], name),
                            "field_id": field["id"] + "#" + name,
                            "field": name,
                            "value": value,
                            "parsed_value": value,
                            "derived_from": [row["id"]],
                        }
                    )
        task["coverage"] = {
            "accepted": sum(
                r["acceptance"] in {"accepted", "absent"} for _, r, _ in readings
            ),
            "expected": len(self.fields),
        }
        task["status"] = "ok" if layout_valid else "unsupported_layout"
        task["playback_tracker"] = self.tracker.state.copy()
        task["metrics"] = {
            "wall_seconds": time.monotonic() - tick,
            "reader_cache_hits": cache_hits,
            "new_reader_calls": len(self.fields) - cache_hits,
            "visual_tokens": getattr(self.worker, "tokens", {}).get("prompt_tokens", 0)
            - token_start.get("prompt_tokens", 0),
            "output_tokens": getattr(self.worker, "tokens", {}).get(
                "completion_tokens", 0
            )
            - token_start.get("completion_tokens", 0),
            "peak_gpu_memory_bytes": getattr(self.worker, "peak_memory_bytes", 0),
        }
        atomic_json(self.store.path("tasks/" + task_id + ".json"), task)
        logging.info(
            "hybrid intake time=%.3f accepted=%d/%d status=%s",
            timestamp,
            task["coverage"]["accepted"],
            len(self.fields),
            task["status"],
        )
        return task

    def report(self, tasks, targets, selected, pilot, parent, started):
        start, end = self.manifest["interval"]
        result = collect(
            self.store, sorted(tasks.values(), key=lambda t: t["start"]), self.config
        )
        result.update(
            sampling_mode=(
                "pilot" if pilot else "benchmark" if selected is not None else "full"
            ),
            stage_wall_seconds=time.monotonic() - started,
            expected_frames=len(targets),
            operational_passed=bool(targets) and len(tasks) == len(targets),
            gpu_memory_peak_bytes=max(
                [getattr(self.worker, "peak_memory_bytes", 0)]
                + [t["metrics"].get("peak_gpu_memory_bytes", 0) for t in tasks.values()]
            ),
        )
        if pilot:
            sampled = len(tasks) / self.config.fps
            result.update(
                sampled_seconds=sampled,
                projected_intake_seconds=max(
                    result["wall_seconds"], result["stage_wall_seconds"]
                )
                / max(sampled, 1)
                * (end - start),
                timing_basis="fresh pilot reader cache; resumed tasks retain original per-frame costs",
                gpu=self.signature.get("vl", {}).get("runtime", {}).get("gpu"),
            )
            memory_limit = (
                self.signature.get("vl", {}).get("runtime", {}).get("total_memory_bytes", 1)
                * 0.9
            )
            result["target_gpu_met"] = "L40S" in (result.get("gpu") or "").upper()
            result["runtime_passed"] = (
                result["target_gpu_met"]
                and sampled >= 300
                and result["projected_intake_seconds"] <= 14400
                and result["gpu_memory_peak_bytes"] < memory_limit
            )
            atomic_json(
                release_path(self.store, self.config, self.profile, "-pilot"),
                {
                    **result,
                    "extraction_key": self.key,
                    "passed": result["runtime_passed"]
                    and result["operational_passed"]
                    and result["quality_benchmark_passed"],
                },
            )
            atomic_json(parent.path("pilot_report.json"), result)
        atomic_json(self.store.path("intake_report.json"), result)
        return result


def annotate(
    store,
    config,
    runner=None,
    resolver=None,
    *,
    retry_failed=False,
    sample_times=None,
    pilot=False,
):
    started = time.monotonic()
    manifest = require_v2(store)
    check_engine(store)
    profile = load_profile(config.layout_profile)
    if profile.get("calibrated") is False:
        raise ValueError("Midbeast profile requires native-frame calibration in review")
    if (profile.get("calibration_source_sha256")
            and profile["calibration_source_sha256"] != manifest["source"]["sha256"]):
        raise ValueError("This calibrated profile belongs to another source; create a new profile revision")
    enabled_fields(profile)
    signature, prepared = runtime(store, config)
    key = store.fingerprint(config)
    parent = store
    if pilot:
        # Same-key retries retain their original costs; changed readers get a cold pilot.
        store = Store(parent.path("pilots/" + key))
        store.install_signals()
        for name, data in (("manifest.json", manifest), ("config.json", config.to_dict()),
                           ("intake_runtime.json", signature)):
            atomic_json(store.path(name), data)
    elif sample_times is None:
        _require_release(store, config, profile, signature, key)
    targets, selected = _sampling_targets(manifest, config, sample_times, pilot)
    tasks = _resume_tasks(store, key, targets)
    worker = runner
    if worker is None and prepared is not None:
        worker = LocalVLRunner(store, config.vl_python or os.environ["LEAGUE_VL_PYTHON"],
                               config.asset_cache, signature=prepared)
    pool = ThreadPoolExecutor(max_workers=config.reader_workers)
    reported = False
    try:
        reader = _IntakeReader(store, config, manifest, profile, signature, key, worker, pool)
        for sample_time, (timestamp, image) in zip(
                targets, decoded_frames(store, manifest, config.fps, targets), strict=True):
            store.check_stop()
            task_id = uid("task", key, timestamp)
            if task_id in tasks and (not retry_failed or tasks[task_id]["status"] == "ok"):
                reader.tracker.state = tasks[task_id]["playback_tracker"]
                continue
            task = reader.read(sample_time, timestamp, image)
            tasks[task_id] = task
        result = reader.report(tasks, targets, selected, pilot, parent, started)
        reported = True
        if not result["operational_passed"]:
            raise ValueError("Processing incomplete; committed evidence retained")
        stage = "intake-pilot" if pilot else "intake-benchmark-extract" if selected is not None else "annotate"
        store.commit_stage(stage, key, **result)
        if pilot and not result["runtime_passed"]:
            raise ValueError("Pilot failed the L40S target, memory or four-hour intake budget")
        return result
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        if worker is not None:
            worker.close()
        if not reported:
            collect(store, sorted(tasks.values(), key=lambda t: t["start"]), config)
