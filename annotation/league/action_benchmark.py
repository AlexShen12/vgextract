"""Shared independent action-label initialization, validation and scoring helpers."""
from __future__ import annotations

from .action_state import load_action_profile
from .core import atomic_json, finite, read_json
from .intake_profile import load_profile


def initialize_actions(store, config, directory, action_profile=None):
    manifest = read_json(store.path("manifest.json"))
    target = directory / "actions.json"
    if target.exists():
        raise ValueError("Refusing to overwrite existing independent action labels")
    lower, upper = manifest["interval"]
    if upper-lower < 200:
        raise ValueError("Twenty disjoint action clips need a source interval of at least 200 seconds")
    profile = load_action_profile(action_profile)
    calibration = load_profile(profile["anchor_profile"]).get("calibration_frames", [])
    clips = []
    for i in range(20):
        start = lower + (upper-lower-12)*i/20
        if i >= 10:
            while any(start-8 <= t <= start+16 for t in calibration):
                start += 1
        clips.append({"id": f"action-clip-{i:02d}", "start": round(start, 3), "end": round(start+8, 3),
                      "split": "development" if i < 10 else "held_out", "verified": False,
                      "repeat_group": None, "case_tags": [], "identities": [], "bindings": [],
                      "transitions": [], "actions": []})
    atomic_json(target, {"schema_version": 3, "source_sha256": manifest["source"]["sha256"],
                         "independent_of_predictions": False, "labeler": None,
                         "note": "Initial sampling grid. Independently inspect source, adjust case coverage, and label before verification. Empty lists are not negative labels.",
                         "clips": clips})
    return {"experimental": True, "status": "unverified_action_labels_created", "labels": str(target),
            "clips": 20, "quality_metrics": None}


def validate_labels(store, data, profile, draft=False):
    source = read_json(store.path("manifest.json"))
    if data.get("schema_version") != 3 or data.get("source_sha256") != source["source"]["sha256"]:
        raise ValueError("Action labels source/schema mismatch")
    clips = data.get("clips", [])
    if len(clips) != 20 or len({c["id"] for c in clips}) != 20:
        raise ValueError("Action pilot requires twenty distinct clips")
    if any(sum(c["split"] == s for c in clips) != 10 for s in ("development", "held_out")):
        raise ValueError("Action pilot requires ten development and ten held-out clips")
    spans = sorted((c["start"], c["end"]) for c in clips)
    lower, upper = source["interval"]
    if any(not finite(a) or not finite(b) or not lower <= a < b <= upper or abs(b-a-8) > 1e-6 for a, b in spans):
        raise ValueError("Action clips require eight seconds inside the source")
    if any(b > c for (_, b), (c, _) in zip(spans, spans[1:])):
        raise ValueError("Action clips overlap")
    dev = [c for c in clips if c["split"] == "development"]
    held = [c for c in clips if c["split"] == "held_out"]
    if any(max(a["start"], b["start"])-min(a["end"], b["end"]) < 8 for a in dev for b in held):
        raise ValueError("Cross-split action windows require an eight-second guard")
    calibration = load_profile(profile["anchor_profile"]).get("calibration_frames", [])
    if any(c["start"]-8 <= t <= c["end"]+8 for c in held for t in calibration):
        raise ValueError("Held-out actions overlap inspected calibration neighborhoods")
    groups = {}
    for clip in clips:
        group = clip.get("repeat_group")
        if group and groups.setdefault(group, clip["split"]) != clip["split"]:
            raise ValueError("Repeated footage must stay in one split")
        if not draft and clip.get("verified") is not True:
            raise ValueError("Independently verified action labels required")
        for category in ("identities", "bindings", "transitions", "actions"):
            if not isinstance(clip.get(category), list):
                raise ValueError("Action label categories must be lists")
            for row in clip[category]:
                if category in ("identities", "bindings"):
                    if not finite(row.get("time")) or not clip["start"] <= row["time"] < clip["end"]:
                        raise ValueError("Identity/binding label time outside clip")
                elif (not finite(row.get("start")) or not finite(row.get("end"))
                      or not clip["start"] <= row["start"] <= row["end"] <= clip["end"]):
                    raise ValueError("Action label time outside clip")
    return clips


def detection_metrics(reference, predictions, keys, tolerance=.25):
    used, pairs, duplicates = set(), [], 0
    for p in sorted(predictions, key=lambda r: (r["start"], str(r.get("id", "")))):
        choices = [(abs(p["start"]-r["start"]), i) for i, r in enumerate(reference) if i not in used
                   and all(p.get(k) == r.get(k) for k in keys)
                   and (p["start"] <= r["end"]+tolerance and p["end"] >= r["start"]-tolerance)]
        if choices:
            _, i = min(choices)
            used.add(i)
            pairs.append((reference[i], p))
        elif any(all(p.get(k) == reference[i].get(k) for k in keys)
                 and p["start"] <= reference[i]["end"]+tolerance
                 and p["end"] >= reference[i]["start"]-tolerance for i in used):
            duplicates += 1
    return {"precision": len(pairs)/len(predictions) if predictions else None,
            "recall": len(pairs)/len(reference) if reference else None, "matched": len(pairs),
            "reference_count": len(reference), "prediction_count": len(predictions),
            "onset_error_seconds": sum(abs(p["start"]-r["start"]) for r, p in pairs)/len(pairs) if pairs else None,
            "actor_accuracy": sum(p.get("champion") == r.get("champion") for r, p in pairs if r.get("champion"))/sum(bool(r.get("champion")) for r, p in pairs) if any(r.get("champion") for r, p in pairs) else None,
            "name_accuracy": sum(p.get("name") == r.get("name") for r, p in pairs if r.get("name"))/sum(bool(r.get("name")) for r, p in pairs) if any(r.get("name") for r, p in pairs) else None,
            "duplicate_predictions": duplicates}


def identity_metrics(reference, predictions, target):
    resolved = correct = 0
    for r in reference:
        choices = [p for p in predictions if p["owner_id"] == r["owner_id"] and abs(p["time"]-r["time"]) <= .125]
        values = {p.get(target) for p in choices if p.get(target)}
        if len(values) == 1:
            resolved += 1
            correct += next(iter(values)) == r.get(target)
    return {"accuracy": correct/resolved if resolved else None, "coverage": resolved/len(reference) if reference else None,
            "reference_count": len(reference), "resolved": resolved, "correct": correct}
