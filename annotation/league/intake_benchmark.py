"""Held-out acceptance gates. Missing predictions always reduce coverage."""

from __future__ import annotations

from collections import defaultdict
import copy
from itertools import groupby
from pathlib import Path

from .core import atomic_json, digest, finite, read_json, implementation_hash
from .intake_profile import load_profile

GATES = {
    "number": (0.99, 0.90),
    "text": (0.95, 0.70),
    "icon": (0.98, 0.70),
    "bar": (0.95, 0.90),
    "presence": (0.99, 0.95),
}
REQUIRED_CASES = {
    "changing_selections",
    "dead_champions",
    "overlays",
    "korean_names",
    "ambiguous_icons",
    "cooldown_rank_distinctions",
    "absent_panels_or_notices",
}


def release_policy():
    return digest(
        {
            "gates": GATES,
            "cases": sorted(REQUIRED_CASES),
            "implementation": implementation_hash(["intake_benchmark.py"]),
        }
    )


def label_coverage(data, fields, capabilities=None):
    """An easy subset cannot certify untested readers or difficult cases."""
    missing = {}
    for split in ("development", "held_out"):
        frames = [f for f in data["frames"] if f["split"] == split]
        labels = [label for frame in frames for label in frame["labels"]]
        visible = [
            fields[label["field_id"]]
            for label in labels
            if label["availability"] == "visible"
        ]
        required = {
            "number:" + kind
            for kind in (
                "integer",
                "level",
                "rank",
                "clock",
                "gold",
                "health",
                "kda",
                "cooldown",
                "timer",
            )
        }
        required |= {
            "icon:" + kind for kind in ("champion", "item", "spell", "passive")
        }
        required |= {"bar", "english_text", "korean_text", "objective_text", "absence"}
        if capabilities is not None:
            required = set(capabilities["field_groups"])
        present = {f["reader"] + ":" + f.get("parser", "") for f in visible}
        present |= {
            "icon:" + f.get("kind", "") for f in visible if f["reader"] == "icon"
        }
        if any(f["reader"] == "bar" for f in visible):
            present.add("bar")
        if any(f.get("parser") == "objective" for f in visible):
            present.add("objective_text")
        for label in labels:
            text = label.get("raw_text", "")
            if (
                fields[label["field_id"]]["reader"] == "text"
                and label["availability"] == "visible"
            ):
                if any("\uac00" <= c <= "\ud7a3" for c in text):
                    present.add("korean_text")
                if any(c.isascii() and c.isalpha() for c in text):
                    present.add("english_text")
        if any(label["availability"] == "absent" for label in labels):
            present.add("absence")
        missing[split] = sorted(required - present)
    cases = {tag for frame in data["frames"] for tag in frame.get("case_tags", [])}
    required_cases = (
        set(capabilities["cases"]) if capabilities is not None else REQUIRED_CASES
    )
    return {
        "missing_field_groups": missing,
        "missing_cases": sorted(required_cases - cases),
        "passed": not any(missing.values()) and required_cases <= cases,
    }


def load_labels(path, source, profile, *, draft=False):
    data = read_json(path)
    if data.get("source_sha256") != source or data.get("schema_version") != 2:
        raise ValueError("Benchmark source/version mismatch")
    frames = data.get("frames", [])
    if len(frames) != 40 or len({f["time"] for f in frames}) != 40:
        raise ValueError("Benchmark requires 40 distinct manually verified frames")
    if profile.get("capabilities"):
        from .hybrid_intake import enabled_fields

        known = {f["id"]: f for f in enabled_fields(profile)}
    else:
        known = {f["id"]: f for f in profile["fields"]}
    for split in ("development", "held_out"):
        group = [f for f in frames if f.get("split") == split]
        if len(group) != 20 or not all(
            (draft or f.get("verified") is True) and f.get("labels") for f in group
        ):
            raise ValueError(
                f"Benchmark requires 20 manually verified, labeled {split} frames"
            )
    for frame in frames:
        if not finite(frame["time"]) or frame["time"] < 0:
            raise ValueError("Invalid benchmark timestamp")
        seen = set()
        for label in frame["labels"]:
            field = label["field_id"]
            if (
                field not in known
                or field in seen
                or label.get("availability") not in {"visible", "absent", "unreadable"}
            ):
                raise ValueError("Invalid or duplicate benchmark field label")
            if label["availability"] == "visible" and "value" not in label:
                raise ValueError(
                    "Visible benchmark fields require an independently verified value"
                )
            if (
                label["availability"] == "visible"
                and known[field]["reader"] in {"number", "text"}
                and "raw_text" not in label
            ):
                raise ValueError("Text labels require exact visible raw_text")
            seen.add(field)
    if (
        min(
            abs(a["time"] - b["time"])
            for a in frames
            for b in frames
            if a["split"] != b["split"]
        )
        < 8
    ):
        raise ValueError(
            "Development and held-out samples must occupy separate eight-second windows"
        )
    if profile.get("calibration_frames") and not draft:
        if any(
            abs(frame["time"] - t) < 8
            for frame in frames
            if frame["split"] == "held_out"
            for t in profile["calibration_frames"]
        ):
            raise ValueError(
                "Held-out frames overlap inspected calibration/development windows"
            )
    if profile.get("capabilities") and not draft:
        segments = {
            split: {f.get("playback_segment_id") for f in frames if f["split"] == split}
            for split in ("development", "held_out")
        }
        if (
            any(not s for group in segments.values() for s in group)
            or segments["development"] & segments["held_out"]
        ):
            raise ValueError(
                "Verified Midbeast splits require distinct, independently checked playback segments"
            )
    return data


def correct(prediction, label, kind):
    if label["availability"] == "absent":
        return prediction is not None and prediction.get("acceptance") == "absent"
    if (
        label["availability"] != "visible"
        or prediction is None
        or prediction.get("acceptance") != "accepted"
    ):
        return False
    if kind == "bar":
        return (
            finite(prediction.get("value"))
            and abs(prediction["value"] - label["value"]) <= 0.05
        )
    if prediction.get("value") != label["value"]:
        return False
    return kind not in {"number", "text"} or prediction.get("raw_text") == label.get(
        "raw_text"
    )


def select_development_gate(candidates, precision, *, icons=False):
    """Maximize development acceptance subject to precision, including icon margin.

    Each candidate is (score, margin, correct). Tied icon identities cannot be
    certified by selecting a zero margin. Re-extraction still has to establish
    actual coverage and the bounded reader's conflict behavior.
    """
    candidates = [(s, m, hit) for s, m, hit in candidates if 0 <= s <= 1]
    margins = (
        sorted({margin for _, margin, _ in candidates if 0 < margin <= 1})
        if icons
        else [0]
    )
    options = []
    for margin_threshold in margins:
        eligible = sorted(
            (score, hit)
            for score, margin, hit in candidates
            if not icons or margin >= margin_threshold
        )
        accepted = correct_count = 0
        for threshold, group in groupby(reversed(eligible), key=lambda c: c[0]):
            outcomes = [hit for _, hit in group]
            accepted += len(outcomes)
            correct_count += sum(outcomes)
            if correct_count / accepted >= precision:
                options.append((accepted, threshold, margin_threshold))
    if not options:
        raise ValueError("Development candidates cannot meet the precision gate")
    _, threshold, margin = max(options)
    return threshold, margin


def parsed_value(value):
    if isinstance(value, dict):
        return {k: parsed_value(v) for k, v in value.items() if k != "displayed_text"}
    return value


def score(frames, predictions, fields, families=None):
    metrics = {
        name: {
            "eligible": 0,
            "accepted": 0,
            "covered": 0,
            "correct": 0,
            "false_visible": 0,
            "parsed_correct": 0,
        }
        for name in (families if families is not None else GATES)
    }
    slices = defaultdict(lambda: {"eligible": 0, "accepted": 0, "correct": 0})
    for frame in frames:
        for label in frame["labels"]:
            kind = fields[label["field_id"]]["reader"]
            if kind not in metrics:
                raise ValueError("Benchmark label targets a deferred field family")
            prediction = predictions.get((frame["time"], label["field_id"]))
            visible = (
                prediction is not None and prediction.get("acceptance") == "accepted"
            )
            presence = prediction is not None and prediction.get("acceptance") in {
                "accepted",
                "absent",
            }
            pm = metrics["presence"]
            if label["availability"] != "unreadable":
                pm["eligible"] += 1
                pm["accepted"] += presence
                pm["covered"] += presence
                pm["correct"] += bool(
                    presence and ((label["availability"] == "visible") == visible)
                )
            if label["availability"] != "visible":
                metrics[kind]["accepted"] += visible
                metrics[kind]["false_visible"] += visible
                continue
            m = metrics[kind]
            m["eligible"] += 1
            m["accepted"] += visible
            m["covered"] += visible
            hit = correct(prediction, label, kind)
            m["correct"] += hit
            if kind == "number" and visible:
                m["parsed_correct"] += parsed_value(
                    prediction.get("value")
                ) == parsed_value(label["value"])
            sm = slices[fields[label["field_id"]]["field"]]
            sm["eligible"] += 1
            sm["accepted"] += visible
            sm["correct"] += hit
    for name, m in metrics.items():
        precision, coverage = GATES[name]
        m["precision"] = m["correct"] / m["accepted"] if m["accepted"] else 0
        m["coverage"] = m["covered"] / m["eligible"] if m["eligible"] else 0
        if name == "number":
            m["parsed_precision"] = (
                m["parsed_correct"] / m["accepted"] if m["accepted"] else 0
            )
            m["exact_transcription_precision"] = m["precision"]
        # Both denominators are mandatory; a missing family cannot pass vacuously.
        m["passed"] = (
            m["eligible"] > 0
            and m["precision"] >= precision
            and m["coverage"] >= coverage
        )
    return {
        "families": metrics,
        "field_slices": dict(slices),
        "passed": all(m["passed"] for m in metrics.values()),
    }


def benchmark(store, config, labels, *, freeze=False, draft=False):
    manifest = read_json(store.path("manifest.json"))
    profile = load_profile(config.layout_profile)
    if draft and freeze:
        raise ValueError("Draft labels cannot freeze production thresholds")
    data = load_labels(labels, manifest["source"]["sha256"], profile, draft=draft)
    predictions = {
        (r.get("sample_time", r["time"]), r["field_id"]): r
        for r in store.rows("intake_records")
    }
    fields = {f["id"]: f for f in profile["fields"]}
    capabilities = profile.get("capabilities")
    families = capabilities["families"] if capabilities else None
    coverage = label_coverage(data, fields, capabilities)
    development = [f for f in data["frames"] if f["split"] == "development"]
    held_out = [f for f in data["frames"] if f["split"] == "held_out"]
    current_key = store.fingerprint(config)
    extraction = store.path("stages/intake-benchmark-extract.json")
    if not extraction.exists() or read_json(extraction).get("key") != current_key:
        raise ValueError(
            "Benchmark extraction is stale; re-extract frames with this profile/runtime"
        )
    if freeze:
        if not coverage["passed"]:
            raise ValueError(
                "Benchmark labels do not cover every promised reader/case: "
                + str(coverage)
            )
        frozen = copy.deepcopy(profile)
        if capabilities:
            development_result = score(development, predictions, fields, families)
            if not development_result["passed"]:
                raise ValueError(
                    "Development quality gates fail; acceptance rules were not frozen"
                )
        # Choose confidence and icon-margin thresholds from development only.
        for kind in ("number", "text", "icon"):
            if capabilities:
                # VL acceptance uses typed output and independent checks, not model
                # self-confidence. Freeze the measured development rules unchanged.
                continue
            candidates = []
            for frame in development:
                for label in frame["labels"]:
                    if fields[label["field_id"]]["reader"] != kind:
                        continue
                    pred = predictions.get((frame["time"], label["field_id"]))
                    if not pred or not pred.get("reader_artifact"):
                        continue
                    cached = read_json(store.path(pred["reader_artifact"]))
                    if kind == "icon":
                        candidate = (cached.get("candidates") or [{}])[0]
                        value = candidate.get("id")
                        confidence = candidate.get("score", 0)
                        margin = cached.get("reader", {}).get("margin", 0)
                        raw = None
                    else:
                        attempts = [
                            a for a in cached.get("attempts", []) if "value" in a
                        ]
                        best = max(
                            attempts, key=lambda a: a.get("score", 0), default={}
                        )
                        value, confidence, raw = (
                            best.get("value"),
                            best.get("score", 0),
                            best.get("raw_text"),
                        )
                        margin = 0
                    hit = (
                        label["availability"] == "visible"
                        and value == label.get("value")
                        and (kind == "icon" or raw == label.get("raw_text"))
                    )
                    candidates.append((confidence, margin, hit))
            try:
                threshold, margin = select_development_gate(
                    candidates, GATES[kind][0], icons=kind == "icon"
                )
            except ValueError as exc:
                raise ValueError(
                    f"Development set cannot meet {kind} precision; thresholds were not frozen"
                ) from exc
            frozen["thresholds"][kind] = threshold
            if kind == "icon":
                frozen["thresholds"]["icon_margin"] = margin
        frozen["thresholds_frozen"] = True
        frozen["development_labels_hash"] = digest(development)
        frozen["parent"] = digest(profile)
        frozen["revision"] = profile.get("revision", 1) + 1
        path = store.path("layout_profiles/" + digest(frozen) + ".json")
        atomic_json(path, frozen)
        config.layout_profile = str(path)
        atomic_json(store.path("config.json"), config.to_dict())
        return {
            "passed": False,
            "status": "thresholds_frozen_reextract_required",
            "profile": str(path),
            "next": "Re-extract the same benchmark frames with the frozen profile, then run intake-benchmark without --freeze",
        }
    result = {
        "schema_version": 2,
        "labels_hash": digest(data),
        "profile_hash": digest(profile),
        "development": score(development, predictions, fields, families),
        "held_out": score(held_out, predictions, fields, families),
        "enabled_capabilities": capabilities,
        "deferred": profile.get("deferred", {}),
        "runtime": read_json(store.path("intake_runtime.json")),
        "source_sha256": manifest["source"]["sha256"],
        "extraction_key": current_key,
        "release_policy": release_policy(),
        "label_coverage": coverage,
        "thresholds_frozen": profile.get("thresholds_frozen", False),
        "draft_labels": draft,
        "passed": False,
    }
    result["passed"] = bool(
        not draft
        and coverage["passed"]
        and result["thresholds_frozen"]
        and result["held_out"]["passed"]
        and profile.get("development_labels_hash") == digest(development)
    )
    atomic_json(store.path("intake_benchmark.json"), result)
    path = (
        Path(config.asset_cache)
        / "releases"
        / (
            digest({"source": manifest["source"]["sha256"], "profile": profile})
            + ".json"
        )
    )
    if not draft:
        atomic_json(path, result)
    return result
