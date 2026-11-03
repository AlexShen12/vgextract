"""Independent crop/event/audio experiments. Never writes production evidence."""

from __future__ import annotations

import base64
import copy
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

from .core import (
    Store,
    StopRequested,
    atomic_json,
    digest,
    finite,
    read_json,
    implementation_hash,
)
from .hybrid_intake import pilot_windows
from .intake_profile import load_profile
from .local_models import LocalVLRunner, isolated_environment, LOCK
from .media import file_hash

EVENT_KINDS = (
    "death",
    "recall",
    "teamfight",
    "objective_takedown",
    "tower_destruction",
)


def initialize_labels(store, config, directory):
    """Save neutral source evidence and empty labels, never model-derived truth."""
    from .intake import sampled_frames

    manifest = read_json(store.path("manifest.json"))
    directory.mkdir(parents=True, exist_ok=True)
    source = manifest["source"]
    for name in ("crops", "events", "audio", "tracking"):
        if (directory / (name + ".json")).exists():
            raise ValueError("Refusing to overwrite existing benchmark labels")
    windows = pilot_windows(*manifest["interval"])
    starts = sorted(
        {math.floor(a) + offset for a, _ in windows for offset in (0, 15, 30, 45)}
    )
    frame_times = sorted({t + offset for t in starts for offset in (0, 4)})
    crops = {"schema_version": 2, "source_sha256": source["sha256"], "frames": []}
    for timepoint, image in sampled_frames(
        source["path"],
        frame_times[0],
        manifest["interval"][1],
        1,
        store.check_stop,
        frame_times,
    ):
        timestamp = round(timepoint)
        path = directory / "frames" / f"{timestamp:06d}.png"
        path.parent.mkdir(exist_ok=True)
        image.save(path)
        index = frame_times.index(timestamp)
        crops["frames"].append(
            {
                "time": timestamp,
                "source_pts": timepoint,
                "split": "development" if index < 20 else "held_out",
                "verified": False,
                "playback_segment_id": None,
                "case_tags": [],
                "labels": [],
                "evidence": str(path),
                "note": "Independently label source pixels; do not copy reader predictions.",
            }
        )
    events = {
        "schema_version": 2,
        "source_sha256": source["sha256"],
        "clips": [
            {
                "id": f"clip-{index:02d}",
                "start": start,
                "end": start + 8,
                "split": "development" if index < 10 else "held_out",
                "verified": False,
                "case_tags": [],
                "events": [],
            }
            for index, start in enumerate(starts)
        ],
    }
    audio = {
        "schema_version": 2,
        "source_sha256": source["sha256"],
        "segments": [
            {
                "id": f"audio-{index}",
                "start": math.floor(a),
                "end": math.floor(a) + 60,
                "verified": False,
                "transcript": None,
                "words": [],
                "names": [],
                "no_speech": False,
            }
            for index, (a, _) in enumerate(windows)
        ],
    }
    tracking = {
        "schema_version": 2,
        "source_sha256": source["sha256"],
        "clips": [
            {**copy.deepcopy(events["clips"][i]), "prompts": [], "reference_masks": []}
            for i in (0, 4, 8, 12, 16)
        ],
    }
    for name, value in (
        ("crops", crops),
        ("events", events),
        ("audio", audio),
        ("tracking", tracking),
    ):
        path = directory / (name + ".json")
        if path.exists():
            raise ValueError(f"Refusing to overwrite existing benchmark labels: {path}")
        atomic_json(path, value)
    return {
        "experimental": True,
        "status": "unverified_labels_created",
        "labels": str(directory),
        "frames": len(crops["frames"]),
        "clips": len(events["clips"]),
        "audio_seconds": 300,
        "next": "Verify independent source labels and playback-segment split before benchmark acceptance",
    }


def load_protocol(store, path, kind, draft=False):
    data = read_json(path)
    source = read_json(store.path("manifest.json"))["source"]
    if data.get("schema_version") != 2 or data.get("source_sha256") != source["sha256"]:
        raise ValueError("Experiment labels must match this source and schema")
    rows = data["segments" if kind == "audio" else "clips"]
    expected = 20 if kind == "events" else 5
    if len(rows) != expected or len({r["id"] for r in rows}) != expected:
        raise ValueError(f"{kind} requires {expected} distinct labeled samples")
    if not draft and not all(r.get("verified") is True for r in rows):
        raise ValueError(
            "Independently verified labels required; --draft is diagnostic only"
        )
    spans = sorted((r["start"], r["end"]) for r in rows)
    lower, upper = read_json(store.path("manifest.json"))["interval"]
    if any(
        not finite(a) or not finite(b) or not lower <= a < b <= upper for a, b in spans
    ):
        raise ValueError("Experiment sample outside source interval")
    if any(b > c for (a, b), (c, d) in zip(spans, spans[1:])):
        raise ValueError("Experiment samples must not overlap")
    if kind == "events":
        if any(abs(b - a - 8) > 1e-6 for a, b in spans) or any(
            sum(r["split"] == split for r in rows) != 10
            for split in ("development", "held_out")
        ):
            raise ValueError(
                "Events require ten development and ten held-out eight-second clips"
            )
        required = {
            "death",
            "recall",
            "fight",
            "objective_or_structure",
            "quiet_negative",
            "occlusion",
            "replayed_footage",
        }
        if not draft and not required <= {
            t for r in rows for t in r.get("case_tags", [])
        }:
            raise ValueError("Event labels omit required difficult/negative cases")
    if kind == "audio" and abs(sum(b - a for a, b in spans) - 300) > 0.05:
        raise ValueError("Audio pilot must contain five minutes")
    if kind == "tracking" and any(
        not 1 <= len(r.get("prompts", [])) <= 3 for r in rows
    ):
        raise ValueError(
            "Each tracking clip needs one to three independently placed object prompts"
        )
    return data


def prepare_clip(store, manifest, clip, directory, masks=()):
    from .intake import sampled_frames

    directory.mkdir(parents=True, exist_ok=True)
    prior = directory / "input-audit.json"
    if prior.exists():
        audit = read_json(prior)
        if (
            Path(audit["video"]).is_file()
            and file_hash(Path(audit["video"])) == audit["video_sha256"]
            and len(list(Path(audit["native_frames_dir"]).glob("*.png"))) == 16
        ):
            return audit
    frames_dir = directory / "frames"
    frames_dir.mkdir(exist_ok=True)
    native_dir = directory / "native"
    native_dir.mkdir(exist_ok=True)
    paths, source_times = [], []
    for index, (timestamp, image) in enumerate(
        sampled_frames(
            manifest["source"]["path"], clip["start"], clip["end"], 2, store.check_stop
        )
    ):
        image.save(native_dir / f"{index:05d}.png")
        if masks:
            from PIL import ImageDraw

            draw = ImageDraw.Draw(image)
            for box in masks:
                draw.rectangle(box, fill="black")
        path = frames_dir / f"{index:05d}.jpg"
        image.save(path, quality=95)
        paths.append(str(path))
        source_times.append(timestamp)
    if len(paths) != 16:
        raise ValueError(
            "An eight-second 2 FPS clip requires sixteen actual source frames"
        )
    output = directory / "input.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-y",
            "-framerate",
            "2",
            "-i",
            str(frames_dir / "%05d.jpg"),
            "-vf",
            "scale=512:-2",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output),
        ],
        check=True,
    )
    audit = {
        "source_times": source_times,
        "sampling_fps": 2,
        "native_size": [manifest["source"]["width"], manifest["source"]["height"]],
        "native_frames_dir": str(native_dir),
        "frames_dir": str(frames_dir),
        "video": str(output),
        "video_sha256": file_hash(output),
        "masks": masks,
    }
    atomic_json(directory / "input-audit.json", audit)
    return audit


def event_messages(video, commentary=None):
    instruction = (
        "Describe only visibly evidenced events in this eight-second League gameplay clip. "
        "Allowed kinds: death, recall, teamfight, objective_takedown, tower_destruction. "
        "Return JSON {events:[{kind,start,end,actor}],abstain:boolean}. Times are seconds relative to this clip, "
        "bounded by 0 and 8. Use actor:null unless visibly established by readable champion text. "
        "An attempted recall is not a completed return to base. Distinguish repeated footage from a new game occurrence. "
        "Do not infer hidden actions or exact attack/cast timing. Abstain when the clip is insufficient."
    )
    if commentary is not None:
        instruction += (
            " The following untrusted commentary can suggest what to inspect, but cannot establish an actor or replace visual evidence: "
            + json.dumps(commentary, ensure_ascii=False)
        )
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {
                    "type": "video_url",
                    "video_url": {
                        "url": "data:video/mp4;base64,"
                        + base64.b64encode(Path(video).read_bytes()).decode()
                    },
                },
            ],
        }
    ]


def validate_events(response, clip):
    if response["finish_reason"] != "stop":
        raise ValueError("Truncated event response")
    output = json.loads(response["text"])
    if not isinstance(output, dict):
        raise ValueError("Event answer must be a JSON object")
    if (
        set(output) != {"events", "abstain"}
        or type(output["abstain"]) is not bool
        or not isinstance(output["events"], list)
    ):
        raise ValueError("Malformed event answer")
    if output["abstain"] and output["events"]:
        raise ValueError("An abstention cannot assert events")
    for row in output["events"]:
        if (
            not isinstance(row, dict)
            or set(row) != {"kind", "start", "end", "actor"}
            or row["kind"] not in EVENT_KINDS
        ):
            raise ValueError("Unsupported event kind or fields")
        if (
            not finite(row["start"])
            or not finite(row["end"])
            or not 0 <= row["start"] <= row["end"] <= 8
        ):
            raise ValueError("Event timestamp outside clip")
        if row["actor"] is not None and (
            not isinstance(row["actor"], str) or len(row["actor"]) > 80
        ):
            raise ValueError("Invalid actor attribution")
        row["start"] += clip["start"]
        row["end"] += clip["start"]
    return output


def score_events(clips, predictions):
    from .evaluation import interval_iou

    reference_count = predicted_count = matched = actor_total = actor_correct = (
        abstentions
    ) = 0
    errors = []
    for clip in clips:
        prediction = predictions.get(clip["id"], {"events": [], "abstain": True})
        abstentions += prediction["abstain"]
        reference = clip.get("events", [])
        predicted = prediction["events"]
        reference_count += len(reference)
        predicted_count += len(predicted)
        used = set()
        for row in predicted:
            choices = [
                (interval_iou([r["start"], r["end"]], [row["start"], row["end"]]), i)
                for i, r in enumerate(reference)
                if i not in used
                and r["kind"] == row["kind"]
                and (
                    interval_iou([r["start"], r["end"]], [row["start"], row["end"]])
                    >= 0.3
                    or abs(r["start"] - row["start"]) <= 1
                )
            ]
            if choices:
                _, index = max(choices)
                used.add(index)
                matched += 1
                truth = reference[index]
                errors.append(abs(truth["start"] - row["start"]))
                if truth.get("actor"):
                    actor_total += 1
                    actor_correct += row.get("actor") == truth["actor"]
    return {
        "precision": matched / predicted_count if predicted_count else 0,
        "recall": matched / reference_count if reference_count else 0,
        "matched": matched,
        "reference_events": reference_count,
        "predicted_events": predicted_count,
        "actor_accuracy": actor_correct / actor_total if actor_total else None,
        "mean_onset_error_seconds": sum(errors) / len(errors) if errors else None,
        "abstentions": abstentions,
    }


def word_error_rate(expected, actual):
    reference, predicted = re.findall(r"\w+", expected.casefold()), re.findall(
        r"\w+", actual.casefold()
    )
    previous = list(range(len(predicted) + 1))
    for index, word in enumerate(reference, 1):
        current = [index]
        for other, candidate in enumerate(predicted, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[other] + 1,
                    previous[other - 1] + (word != candidate),
                )
            )
        previous = current
    return {
        "errors": previous[-1],
        "reference_words": len(reference),
        "wer": previous[-1] / len(reference) if reference else None,
        "false_words_on_silence": len(predicted) if not reference else 0,
    }


def score_tracking(clips, predictions, label_directory):
    import numpy as np
    from PIL import Image

    predicted = {
        (c["clip_id"], frame["frame_index"], row["object_id"]): row["mask_path"]
        for c in predictions
        for frame in c["frames"]
        for row in frame["masks"]
    }
    scores, missing = [], 0
    for clip in clips:
        if not clip.get("reference_masks"):
            raise ValueError(
                "Verified tracking needs independently labeled reference masks"
            )
        for label in clip["reference_masks"]:
            key = (clip["id"], label["frame_index"], label["object_id"])
            reference_path = (Path(label_directory) / label["path"]).resolve()
            truth = np.asarray(Image.open(reference_path).convert("L")) > 127
            if key not in predicted:
                missing += 1
                scores.append(0.0)
                continue
            mask = np.asarray(Image.open(predicted[key]).convert("L")) > 127
            if mask.shape != truth.shape:
                raise ValueError("Tracking masks must retain native source dimensions")
            union = int((mask | truth).sum())
            scores.append(float((mask & truth).sum()) / union if union else 1.0)
    return {
        "mean_mask_iou": sum(scores) / len(scores),
        "labeled_masks": len(scores),
        "missing_tracks": missing,
        "coordinate_system": "screen_pixels",
        "event_claims": False,
    }


def audio_metrics(reference, chunks):
    actual = " ".join(c.get("text") or "" for c in chunks)
    words = [w for c in chunks for w in c.get("words", [])]
    normalized = actual.casefold()
    names = reference.get("names", [])
    errors = []
    used = set()
    for target in reference.get("words", []):
        candidates = [
            (abs(word["start"] - target["start"]), i)
            for i, word in enumerate(words)
            if i not in used
            and word["text"].casefold().strip() == target["text"].casefold().strip()
        ]
        if candidates:
            error, index = min(candidates)
            used.add(index)
            errors.append(error)
    return {
        **word_error_rate(reference["transcript"], actual),
        "name_recall": (
            sum(name.casefold() in normalized for name in names) / len(names)
            if names
            else None
        ),
        "alignment_matched_words": len(errors),
        "alignment_reference_words": len(reference.get("words", [])),
        "mean_word_onset_error_seconds": sum(errors) / len(errors) if errors else None,
    }


def run_worker(store, python, job, directory):
    directory.mkdir(parents=True, exist_ok=True)
    job = {**job, "output": str(directory / "result.json")}
    from .local_models import checkpoint

    roles = ("asr", "aligner") if job["kind"] == "audio" else ("tracking",)
    pins = {role: checkpoint(job["cache"], role)[1] for role in roles}
    probe = subprocess.run(
        [
            python,
            "-c",
            "import importlib.metadata,json; print(json.dumps(sorted((d.metadata['Name'],d.version) for d in importlib.metadata.distributions())))",
        ],
        capture_output=True,
        text=True,
        check=True,
        env=isolated_environment(job["cache"]),
    )
    key = digest(
        {
            "job": job,
            "pins": pins,
            "packages": json.loads(probe.stdout),
            "code": implementation_hash(["experiment_worker.py"]),
        }
    )
    cached = directory / "worker-cache.json"
    if (
        cached.exists()
        and read_json(cached).get("key") == key
        and Path(job["output"]).exists()
    ):
        return read_json(Path(job["output"]))
    atomic_json(directory / "job.json", job)
    with (directory / "worker.log").open("w") as log:
        process = subprocess.Popen(
            [
                python,
                "-B",
                "-m",
                "annotation.league.experiment_worker",
                "--job",
                str(directory / "job.json"),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=isolated_environment(job["cache"]),
            start_new_session=True,
        )
        started = time.monotonic()
        try:
            while process.poll() is None:
                store.check_stop()
                if time.monotonic() - started > 1800:
                    raise RuntimeError("Experiment worker exceeded thirty minutes")
                time.sleep(0.25)
            if process.returncode:
                raise RuntimeError(
                    f"Experiment worker failed ({process.returncode}); inspect {directory / 'worker.log'}"
                )
        finally:
            if process.poll() is None:
                import signal

                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
    atomic_json(cached, {"key": key})
    return read_json(Path(job["output"]))


def audio_experiment(store, config, protocol, directory):
    manifest = read_json(store.path("manifest.json"))
    directory.mkdir(parents=True, exist_ok=True)
    segments = []
    for row in protocol["segments"]:
        path = directory / (row["id"] + ".wav")
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-nostdin",
                "-y",
                "-ss",
                str(row["start"]),
                "-i",
                manifest["source"]["path"],
                "-t",
                str(row["end"] - row["start"]),
                "-vn",
                "-ar",
                "16000",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                str(path),
            ],
            check=True,
        )
        segments.append(
            {
                "id": row["id"],
                "path": str(path),
                "start": row["start"],
                "end": row["end"],
            }
        )
    python = os.environ.get("LEAGUE_AUDIO_PYTHON")
    if not python:
        raise ValueError("Set LEAGUE_AUDIO_PYTHON to the isolated audio environment")
    result = run_worker(
        store,
        python,
        {"kind": "audio", "cache": config.asset_cache, "segments": segments},
        directory,
    )
    metrics = []
    for reference in protocol["segments"]:
        if reference.get("verified") and reference.get("transcript") is not None:
            chunks = [c for c in result["chunks"] if c["segment_id"] == reference["id"]]
            metrics.append(
                {"segment_id": reference["id"], **audio_metrics(reference, chunks)}
            )
    result["metrics"] = metrics
    result["latency_basis"] = (
        "Original worker timings; compatible cached transcriptions may be reused"
    )
    atomic_json(directory / "report.json", result)
    return result


def compare_events(store, config, protocol, audits, root, role, audio_result, draft):
    worker = LocalVLRunner(
        store,
        config.vl_python or os.environ.get("LEAGUE_VL_PYTHON", ""),
        config.asset_cache,
        role,
    )
    predictions = {
        mode: {}
        for mode in ("visual_only", "audio_assisted")
        if mode == "visual_only" or audio_result is not None
    }
    try:
        for clip in protocol["clips"]:
            for mode in predictions:
                commentary = (
                    None
                    if mode == "visual_only"
                    else [
                        c["text"]
                        for c in audio_result["chunks"]
                        if c.get("text")
                        and c["source_start"] < clip["end"]
                        and c["source_end"] > clip["start"]
                    ]
                )
                request_key = digest(
                    {
                        "video": audits[clip["id"]]["video_sha256"],
                        "model": worker.identity,
                        "commentary": commentary,
                        "prompt": implementation_hash(["experiments.py"]),
                        "max_tokens": 1536,
                    }
                )
                path = store.path("experiment-cache/events/" + request_key + ".json")
                if path.exists():
                    response = read_json(path)
                else:
                    response = worker.complete(
                        event_messages(audits[clip["id"]]["video"], commentary),
                        max_tokens=1536,
                    )
                    atomic_json(path, response)
                atomic_json(root / role / mode / (clip["id"] + ".json"), response)
                try:
                    predictions[mode][clip["id"]] = validate_events(response, clip)
                except (ValueError, KeyError, TypeError) as exc:
                    predictions[mode][clip["id"]] = {
                        "events": [],
                        "abstain": True,
                        "error": str(exc),
                    }
    finally:
        worker.close()
    result = {
        mode: {
            "predictions": rows,
            "metrics": (
                {
                    split: score_events(
                        [c for c in protocol["clips"] if c["split"] == split],
                        rows,
                    )
                    for split in ("development", "held_out")
                }
                if not draft
                else None
            ),
        }
        for mode, rows in predictions.items()
    }
    result["peak_memory_bytes"] = max(
        [worker.peak_memory_bytes]
        + [
            read_json(p).get("peak_memory_bytes", 0)
            for p in (root / role).glob("*/*.json")
        ]
    )
    result["runtime"] = worker.identity
    result["latency_seconds"] = sum(
        read_json(p).get("wall_seconds", 0) for p in (root / role).glob("*/*.json")
    )
    return result


def _record_experiment(report, name, operation, *args, failure_status="failed"):
    """Keep isolated failures in the report while allowing cancellation to propagate."""
    try:
        result = operation(*args)
    except StopRequested:
        raise
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        report["results"][name] = {"status": failure_status, "reason": str(exc), "experimental": True}
        return None
    report["results"][name] = result
    return result


def _crop_experiment(store, config, manifest, directory, draft):
    from .intake import annotate
    from .intake_benchmark import benchmark, load_labels

    protocol = load_labels(
        directory / "crops.json",
        manifest["source"]["sha256"],
        load_profile(config.layout_profile),
        draft=draft,
    )
    isolated = Store(store.path("experiment-cache/crops"))
    isolated.check_stop = store.check_stop
    atomic_json(isolated.path("manifest.json"), manifest)
    atomic_json(isolated.path("config.json"), config.to_dict())
    annotate(
        isolated,
        config,
        sample_times=[r["time"] for r in protocol["frames"]],
    )
    # Diagnostic benchmark cannot create a production release from an experiment.
    result = benchmark(
        isolated, config, directory / "crops.json", draft=True
    )
    result["artifact_directory"] = str(isolated.root)
    return result


def _audio_experiment(store, config, manifest, directory, draft):
    protocol = load_protocol(
        store, directory / "audio.json", "audio", draft
    )
    audio_key = digest(
        {
            "source": manifest["source"]["sha256"],
            "intervals": [
                (r["id"], r["start"], r["end"])
                for r in protocol["segments"]
            ],
        }
    )
    return audio_experiment(
        store,
        config,
        protocol,
        store.path("experiment-cache/audio/" + audio_key),
    )


def _tracking_experiment(store, config, manifest, directory, audits, draft):
    tracking_protocol = load_protocol(
        store, directory / "tracking.json", "tracking", draft
    )
    clips = [
        {
            **clip,
            **audits[clip["id"]],
            "width": manifest["source"]["width"],
            "height": manifest["source"]["height"],
        }
        for clip in tracking_protocol["clips"]
    ]
    python = os.environ.get("LEAGUE_TRACKING_PYTHON")
    if not python:
        raise ValueError(
            "Set LEAGUE_TRACKING_PYTHON for the separate SAM 3.1 subexperiment"
        )
    tracking = run_worker(
        store,
        python,
        {
            "kind": "tracking",
            "cache": config.asset_cache,
            "clips": [
                {
                    k: v
                    for k, v in clip.items()
                    if k
                    not in {
                        "events",
                        "verified",
                        "reference_masks",
                        "case_tags",
                        "split",
                    }
                }
                for clip in clips
            ],
        },
        store.path(
            "experiment-cache/tracking/"
            + digest(
                {
                    "source": manifest["source"]["sha256"],
                    "prompts": [clip["prompts"] for clip in clips],
                    "inputs": [clip["video_sha256"] for clip in clips],
                }
            )
        ),
    )
    tracking["metrics"] = (
        score_tracking(
            tracking_protocol["clips"], tracking["clips"], directory
        )
        if not draft
        else None
    )
    return tracking


def _event_experiments(store, config, manifest, directory, root, report, audio_result, draft):
    protocol = load_protocol(store, directory / "events.json", "events", draft)
    audits = {
        clip["id"]: prepare_clip(
            store,
            manifest,
            clip,
            store.path(
                "experiment-cache/clips/"
                + digest(
                    {
                        "source": manifest["source"]["sha256"],
                        "interval": [clip["start"], clip["end"]],
                        "masks": config.masks,
                        "decoder": implementation_hash(
                            ["intake.py", "media.py"]
                        ),
                        "recipe": "native-png-masked-jpeg95-mp4-512w-2fps-v1",
                    }
                )
            ),
            config.masks,
        )
        for clip in protocol["clips"]
    }
    for role in ("intake", "event_alternative"):
        _record_experiment(report, role, compare_events,
                           store, config, protocol, audits, root, role, audio_result, draft)
        atomic_json(root / "report.json", report)
    _record_experiment(report, "tracking", _tracking_experiment,
                       store, config, manifest, directory, audits, draft, failure_status="blocked")


def experiment(store, config, suite, *, labels=None, initialize=False, draft=False):
    manifest = read_json(store.path("manifest.json"))
    if file_hash(Path(manifest["source"]["path"])) != manifest["source"]["sha256"]:
        raise ValueError("Source changed; prepare a new run")
    directory = Path(labels).resolve() if labels else store.path("benchmarks")
    if initialize:
        return initialize_labels(store, config, directory)
    key = digest(
        {
            "source": manifest["source"]["sha256"],
            "suite": suite,
            "models": read_json(LOCK),
            "implementation": implementation_hash(
                ["experiments.py", "experiment_worker.py"]
            ),
            "labels": {p.name: read_json(p) for p in directory.glob("*.json")},
        }
    )
    root = store.path("experiments/" + key)
    root.mkdir(parents=True, exist_ok=True)
    report = {
        "experimental": True,
        "suite": suite,
        "source_sha256": manifest["source"]["sha256"],
        "draft_labels": draft,
        "production_evidence_modified": False,
        "marengo": "Hosted retrieval model; no public offline L40S weights established",
        "results": {},
    }
    try:
        if suite in {"crops", "all"}:
            _record_experiment(report, "crops", _crop_experiment,
                               store, config, manifest, directory, draft)
        audio_result = None
        if suite in {"audio", "all"}:
            audio_result = _record_experiment(report, "audio", _audio_experiment,
                                             store, config, manifest, directory, draft)
        if suite in {"events", "all"}:
            _event_experiments(store, config, manifest, directory, root, report, audio_result, draft)
        report["status"] = (
            "partial_experimental"
            if any(
                r.get("status") in {"failed", "blocked"}
                for r in report["results"].values()
            )
            else "completed_experimental"
        )
    except BaseException as exc:
        report["status"] = "interrupted" if isinstance(exc, StopRequested) else "failed"
        report["error"] = str(exc)
        raise

    finally:
        atomic_json(root / "report.json", report)
    return {**report, "report_path": str(root / "report.json")}
