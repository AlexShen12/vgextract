"""Schema-2 intake utilities and entry points for the reduced production readers."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import math
import os
import re
from pathlib import Path
import shutil
import subprocess

from .core import (
    atomic_json,
    digest,
    read_json,
    uid,
    validate_observation,
)
from .intake_profile import UNSUPPORTED, layout_matches, load_profile


def tesseract_signature():
    executable = os.environ.get("LEAGUE_TESSERACT", "tesseract")
    version = subprocess.check_output(
        [executable, "--version"], text=True
    ).splitlines()[0]
    languages = subprocess.check_output(
        [executable, "--list-langs"], text=True, stderr=subprocess.STDOUT
    )
    match = re.search(r'List of available languages in "([^"]+)"', languages)
    explicit = os.environ.get("LEAGUE_TESSDATA")
    prefix = Path(shutil.which(executable) or executable).resolve().parent.parent
    candidates = (
        [Path(explicit)]
        if explicit
        else ([Path(match[1])] if match else [])
        + (
            [Path(os.environ["TESSDATA_PREFIX"])]
            if os.environ.get("TESSDATA_PREFIX")
            else []
        )
        + [
            prefix / path
            for path in (
                "share/tessdata",
                "share/tesseract/tessdata",
                "share/tesseract-ocr/5/tessdata",
                "share/tesseract-ocr/4.00/tessdata",
            )
        ]
    )
    trained = next(
        (
            directory / "eng.traineddata"
            for directory in candidates
            if (directory / "eng.traineddata").is_file()
        ),
        None,
    )
    if trained is None or not trained.is_file():
        raise ValueError(
            "Cannot verify Tesseract English recognition data; install tesseract with setup_longleaf.sh"
        )
    # Tesseract 4 omits the directory in --list-langs. Force the verified file's
    # directory for all recognition calls so provenance describes the data used.
    os.environ["LEAGUE_TESSDATA"] = str(trained.parent)
    return {
        "version": version,
        "language": "eng",
        "traineddata_sha256": hashlib.sha256(trained.read_bytes()).hexdigest(),
    }


def require_v2(store):
    manifest = read_json(store.path("manifest.json"))
    if manifest.get("schema_version", 1) != 2:
        raise ValueError(
            "Version-1 runs are read-only. Start a new run with --run-dir data/league_runs/midbeast-OLLZBe5-fg4-v2"
        )
    return manifest


def sampled_frames(video, start, end, fps, check_stop=lambda: None, targets=None):
    """One forward decode, first native frame at/after each sample target."""
    import av

    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        if start:
            container.seek(int(start / stream.time_base), stream=stream, backward=True)
        target = start
        requested = iter(targets) if targets is not None else None
        if requested is not None:
            target = next(requested, None)
            if target is None:
                return
        for frame in container.decode(stream):
            check_stop()
            if frame.pts is None:
                raise ValueError("Decoded frame lacks source PTS")
            timestamp = float(frame.pts * frame.time_base)
            if timestamp >= end:
                break
            if timestamp + 1e-8 < target:
                continue
            if timestamp - target > max(0.1, 2 / float(stream.average_rate or 1)):
                raise ValueError(f"Source PTS gap at sample {target}: {timestamp}")
            yield timestamp, frame.to_image().convert("RGB")
            if requested is not None:
                target = next(requested, None)
                if target is None:
                    break
            else:
                target = (
                    start + (math.floor((timestamp - start) * fps + 1e-6) + 1) / fps
                )


def image_hash(image):
    return hashlib.sha256(
        str(image.size).encode() + image.convert("RGB").tobytes()
    ).hexdigest()


def decoded_frames(store, manifest, fps, targets=None):
    """Independent native-frame cache; OCR/profile changes never invalidate decode."""
    from PIL import Image
    from importlib.metadata import version

    start, end = manifest["interval"]
    key = digest(
        {
            "source": manifest["source"]["sha256"],
            "interval": [start, end],
            "fps": fps,
            "decoder": "first-frame-at-target-v1",
            "av": version("av"),
        }
    )
    directory = store.path("decoded/" + key)
    directory.mkdir(parents=True, exist_ok=True)
    decoder = None
    targets = None if targets is None else sorted(set(targets))
    try:
        for index in range(math.ceil((end - start) * fps)):
            store.check_stop()
            metadata = directory / f"{index:08d}.json"
            image_path = metadata.with_suffix(".png")
            target = start + index / fps
            if targets is not None and not any(abs(t - target) < 1e-6 for t in targets):
                continue
            if metadata.exists() and image_path.exists():
                if decoder is not None:
                    decoder.close()
                    decoder = None
                record = read_json(metadata)
                with Image.open(image_path) as source:
                    image = source.convert("RGB")
                if image_hash(image) != record["pixel_hash"]:
                    raise ValueError(f"Decoded frame cache hash mismatch: {image_path}")
                if abs(record["time"] - target) > max(
                    0.1, 2 / manifest["source"]["fps"]
                ):
                    raise ValueError("Cached source timestamp differs from sample grid")
                yield record["time"], image
                continue
            if decoder is None:
                remaining = (
                    None
                    if targets is None
                    else [t for t in targets if t >= target - 1e-6]
                )
                decoder = sampled_frames(
                    manifest["source"]["path"],
                    target,
                    end,
                    fps,
                    store.check_stop,
                    remaining,
                )
            timestamp, image = next(decoder)
            if abs(timestamp - target) > max(0.1, 2 / manifest["source"]["fps"]):
                raise ValueError("Decoded frame cache grid differs from source PTS")
            temporary = image_path.with_suffix(".tmp")
            image.save(temporary, format="PNG")
            temporary.replace(image_path)
            atomic_json(metadata, {"time": timestamp, "pixel_hash": image_hash(image)})
            yield timestamp, image
    finally:
        if decoder is not None:
            decoder.close()


def preflight(store, config):
    from .hybrid_intake import preflight as hybrid_preflight
    return hybrid_preflight(store, config)


def evidence_crop(store, image):
    key = image_hash(image)
    relative = f"field-inputs/{key}.png"
    path = store.path(relative)
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        temporary = path.with_suffix(".tmp")
        image.save(temporary, format="PNG")
        temporary.replace(path)
    return key, relative


def preview(store, config):
    from PIL import Image, ImageDraw
    from .numeric_reader import numeric_preprocess
    from .core import Route
    from .media import Media

    require_v2(store)
    profile = load_profile(config.layout_profile)
    media = Media(store, config)
    times = read_json(store.path("layout_validation.json"))["preview_times"]
    full = Route("layout", [0, 0, config.width, config.height], "instruct", [])
    frames = []
    matches = []
    for timestamp in times:
        frame = media.frame(timestamp, full)
        with Image.open(store.path(frame["path"])) as image:
            matched, scores = layout_matches(image, profile)
            matches.append({"time": timestamp, "matched": matched, "anchors": scores})
            overlay = image.copy()
            draw = ImageDraw.Draw(overlay)
            inputs = []
            processed = []
            for field in profile["fields"]:
                draw.rectangle(field["box"], outline="#00ffff", width=1)
                crop = image.crop(field["box"])
                crop_hash, path = evidence_crop(store, crop)
                crop_id = uid("frame", frame["source_id"], field["box"], crop_hash)
                inputs.append(
                    {
                        "id": crop_id,
                        "field_id": field["id"],
                        "path": path,
                        "box": field["box"],
                        "route": field["id"],
                        "time": frame["time"],
                    }
                )
                views = []
                if field["reader"] in {"number", "text"}:
                    views = [crop] if field.get("backend") == "vl" or field["reader"] == "text" else [numeric_preprocess(crop)]
                elif field["reader"] == "icon":
                    for inset in (0.08, 0.18):
                        w, h = crop.size
                        views.append(
                            crop.crop(
                                (
                                    int(w * inset),
                                    int(h * inset),
                                    int(w * (1 - inset)),
                                    int(h * (1 - inset)),
                                )
                            ).resize((24, 24))
                        )
                for view in views:
                    _, processed_path = evidence_crop(store, view)
                    processed.append(
                        {
                            "path": processed_path,
                            "size": list(view.size),
                            "source_frame": crop_id,
                        }
                    )
            overlay_path = (
                f"field-inputs/layout-{uid('preview',timestamp,digest(profile))}.png"
            )
            overlay.save(store.path(overlay_path))
            frames.append(
                {
                    **frame,
                    "overlay_path": overlay_path,
                    "inputs": inputs,
                    "processed_inputs": processed,
                }
            )
    store.write("intake_previews", frames)
    result = {
        "schema_version": 2,
        "profile_hash": digest(profile),
        "profile": profile["id"],
        "preprocessing_key": store.fingerprint(config),
        "intake_profile_validated": all(m["matched"] for m in matches),
        "samples": matches,
        "unsupported": profile.get("deferred", UNSUPPORTED),
        "semantic_accuracy_validated": False,
    }
    atomic_json(
        store.path("layout_validation.json"),
        {**read_json(store.path("layout_validation.json")), **result},
    )
    if not result["intake_profile_validated"]:
        raise ValueError(
            "unsupported_layout: inspect intake previews and calibrate in review"
        )
    return result


def subject(field, identities):
    if field.get("entity"):
        return field["entity"]
    if field["slot_id"].startswith("objective:"):
        label = identities.get(field["slot_id"])
        return "objective:" + label.casefold().replace(" ", "_") if label else "unknown"
    return identities.get(field["slot_id"], "unknown")


def verify_readings(readings):
    """Reject contradictory HP digits without deriving digits from bar pixels."""
    bars = {
        field["slot_id"]: reading
        for field, reading, _ in readings
        if field["field"] == "health_fraction" and reading["acceptance"] == "accepted"
    }
    for field, reading, _ in readings:
        if field["field"] != "health" or reading["acceptance"] != "accepted":
            continue
        value = reading.get("value")
        bar = bars.get(field["slot_id"])
        if not isinstance(value, dict) or not bar:
            continue
        difference = abs(value["current"] / value["maximum"] - bar["value"])
        reading["verification"] = {
            "status": "consistent_with_bar" if difference <= 0.06 else "conflict",
            "method": "independent_numeric_health_and_bar_proportion",
            "difference": difference,
            "tolerance": 0.06,
        }
        if difference > 0.06:
            reading.update(
                acceptance="unknown",
                reason="numeric health conflicts with independently measured bar",
            )


def observation(task, frame, field, reading, identity, path):
    acceptance = reading["acceptance"]
    row = {
        "schema_version": 2,
        "id": uid("obs", task["id"], field["id"]),
        "task_id": task["id"],
        "frame_id": frame["id"],
        "source_id": frame["source_id"],
        "time": frame["time"],
        "route": field["route"],
        "box": field["box"],
        "slot_id": field["slot_id"],
        "entity": identity,
        "resolved_identity": None if identity == "unknown" else identity,
        "binding": field["slot_id"],
        "field": field["field"],
        "field_id": field["id"],
        "value": reading.get("value") if acceptance == "accepted" else None,
        "raw_text": reading.get("raw_text"),
        "parsed_value": reading.get("value"),
        "visibility": (
            "visible"
            if acceptance == "accepted"
            else "not_shown" if acceptance == "absent" else "unknown"
        ),
        "acceptance": acceptance,
        "acceptance_reason": reading["reason"],
        "reader": {**reading.get("reader", {}), "runtime_id": task.get("runtime_id")},
        "profile_hash": task.get("profile_hash"),
        "verification": reading.get(
            "verification", {"status": "not_independently_checked"}
        ),
        "confidence": 0.0,
        "reader_artifact": reading.get("cache_path"),
        "evidence_path": path,
        "review_status": "unreviewed",
        "identity_evidence": "visual reference" if identity != "unknown" else None,
        "pass": 0,
        "sampling_fps": task["fps"],
    }
    return validate_observation(row)


def collect(store, tasks, config):
    from .intake_benchmark import release_policy

    profile = load_profile(config.layout_profile)
    unsupported = profile.get("deferred", UNSUPPORTED)
    observations = [o for task in tasks for o in task["observations"]]
    native = [o for o in observations if not o.get("derived_from")]
    store.write("observations", observations)
    store.write("intake_records", native)
    store.write("event_candidates", [])
    store.write("issues", [i for task in tasks for i in task["issues"]])
    store.write(
        "coverage",
        [
            {
                k: t[k]
                for k in (
                    "id",
                    "route",
                    "start",
                    "end",
                    "fps",
                    "pass",
                    "status",
                    "coverage",
                    "issues",
                )
            }
            for t in tasks
        ]
        + [
            {"route": route, "status": "unsupported", "reason": reason}
            for route, reason in unsupported.items()
        ],
    )
    groups = defaultdict(Counter)
    for row in native:
        groups[row["route"]][row["acceptance"]] += 1
    coverage = {
        route: {
            **counts,
            "coverage": (counts["accepted"] + counts["absent"]) / sum(counts.values()),
        }
        for route, counts in groups.items()
    }
    result = {
        "schema_version": 2,
        "frames": len(tasks),
        "fields": len(native),
        "normalized_observations": len(observations),
        "routes": coverage,
        "unsupported": unsupported,
        "accepted": sum(o["acceptance"] == "accepted" for o in native),
        "unknown": sum(o["acceptance"] == "unknown" for o in native),
        "quality_benchmark_passed": False,
        "thresholds_frozen": load_profile(config.layout_profile).get(
            "thresholds_frozen", False
        ),
        "meaning": "Operational field coverage; semantic release accuracy requires held-out labels",
        "wall_seconds": sum(t["metrics"]["wall_seconds"] for t in tasks),
    }
    result["operational_passed"] = bool(tasks)
    result["deferred"] = load_profile(config.layout_profile).get("deferred", {})
    release = (
        Path(config.asset_cache)
        / "releases"
        / (
            digest(
                {
                    "source": require_v2(store)["source"]["sha256"],
                    "profile": load_profile(config.layout_profile),
                }
            )
            + ".json"
        )
    )
    if release.exists():
        record = read_json(release)
        result["quality_benchmark_passed"] = bool(
            result["thresholds_frozen"]
            and record.get("passed")
            and record.get("release_policy") == release_policy()
            and record.get("extraction_key") == store.fingerprint(config)
            and record.get("runtime") == read_json(store.path("intake_runtime.json"))
        )
    atomic_json(store.path("intake_report.json"), result)
    return result


def annotate(store, config, runner=None, resolver=None, *, retry_failed=False, sample_times=None, pilot=False):
    from .hybrid_intake import annotate as hybrid_annotate
    return hybrid_annotate(store, config, runner, resolver, retry_failed=retry_failed,
                           sample_times=sample_times, pilot=pilot)
