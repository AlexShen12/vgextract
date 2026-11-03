"""Independent reference labels, frozen choices and non-vacuous caption acceptance."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from .action_icons import AbilityIdentity
from .action_v3 import load_profile
from .caption_pilot import COMPONENTS, caption_pilot
from .caption_segments import POLICY
from .caption_scene import iou
from .captions import EVENT_CRITERIA
from .core import atomic_json, digest, finite, implementation_hash, read_json, read_rows
from .intake_profile import load_profile as anchor_profile
from .local_models import LOCK


KINDS = set(EVENT_CRITERIA) | {"cs_increased", "displayed_state"}
CASES = {"counter_only", "cast_miss", "ambiguous_impact", "competing_targets", "rapid_attacks", "occlusion",
         "ownership_change", "clock_ticks", "pause", "rewind", "repeat", "chunk_boundary", "last_hit", "ability_hit"}
MINIMUM_CLAIMS = 20
MINIMUM_PER_KIND = 5
ANNOTATION_CATEGORIES = ("claims", "bindings", "phases", "playback", "repeats")


def initialize_labels(store, directory, profile):
    target = Path(directory)/"captions.json"
    if target.exists():
        raise ValueError("Refusing to overwrite independent caption labels")
    manifest = read_json(store.path("manifest.json"))
    lower, upper = manifest["interval"]
    calibration = anchor_profile(profile["anchor_profile"]).get("calibration_frames", [])
    clips, cursor = [], lower
    for i in range(20):
        start = max(cursor, lower+(upper-lower-24)*i/20)
        if i >= 10:
            while any(start-8 <= t <= start+32 for t in calibration):
                start += 1
        if start+24 > upper:
            raise ValueError("Source interval is too short for twenty guarded caption neighborhoods")
        clips.append({"id": f"caption-neighborhood-{i:02d}", "start": start, "end": start+24,
            "split": "development" if i < 10 else "held_out", "repeat_group": None, "verified": False,
            "labeler": None, "independent_of_predictions": False, "inspected": False, "explicit_negative": False,
            "negative_annotations": {category: False for category in ANNOTATION_CATEGORIES},
            "case_tags": [], "claims": [], "bindings": [], "phases": [], "playback": [], "repeats": []})
        cursor = start+32
    data = {"schema_version": 1, "source_sha256": manifest["source"]["sha256"], "labeler": None,
            "independent_of_predictions": False, "clips": clips}
    atomic_json(target, data)
    return {"status": "unverified_caption_labels_created", "labels": str(target), "quality_metrics": None}


def validate_labels(store, data, profile, *, required_splits=()):
    manifest = read_json(store.path("manifest.json"))
    if data.get("schema_version") != 1 or data.get("source_sha256") != manifest["source"]["sha256"]:
        raise ValueError("Caption label source/schema mismatch")
    clips = data.get("clips", [])
    if len(clips) != 20 or len({c["id"] for c in clips}) != 20 or any(sum(c["split"] == s for c in clips) != 10 for s in ("development", "held_out")):
        raise ValueError("Caption benchmark requires ten development and ten held-out neighborhoods")
    lower, upper = manifest["interval"]
    calibration = anchor_profile(profile["anchor_profile"]).get("calibration_frames", [])
    groups = {}
    for c in clips:
        if not all(finite(c.get(k)) for k in ("start", "end")) or not lower <= c["start"] < c["end"] <= upper or not 8 <= c["end"]-c["start"] <= 60:
            raise ValueError("Caption neighborhoods must be 8–60 source seconds")
        if c.get("repeat_group") is not None and not isinstance(c["repeat_group"], str):
            raise ValueError("Repeat groups must be strings or null")
        if c.get("repeat_group") and groups.setdefault(c["repeat_group"], c["split"]) != c["split"]:
            raise ValueError("Repeated footage must stay in one split")
        if c["split"] == "held_out" and any(c["start"]-8 <= t <= c["end"]+8 for t in calibration):
            raise ValueError("Held-out captions overlap calibration neighborhoods")
        if any(not isinstance(c.get(k), list) for k in ("claims", "bindings", "phases", "playback", "repeats", "case_tags")):
            raise ValueError("Caption annotation categories must be lists")
        if any(type(c.get(k)) is not bool for k in ("verified", "inspected", "explicit_negative", "independent_of_predictions")):
            raise ValueError("Caption verification and inspection flags must be booleans")
        if any(not isinstance(t, str) or not t or len(t) > 100 for t in c["case_tags"]):
            raise ValueError("Caption case tags must be bounded strings")
        if c["split"] in required_splits and (c.get("verified") is not True or c.get("independent_of_predictions") is not True
                or not c.get("labeler") or c.get("inspected") is not True or c.get("contaminated")
                or (not c["claims"] and c.get("explicit_negative") is not True)):
            raise ValueError("Independent inspected labels and explicit negatives are required")
        negatives = c.get("negative_annotations", {})
        if not isinstance(negatives, dict) or any(k not in ANNOTATION_CATEGORIES or type(v) is not bool for k, v in negatives.items()):
            raise ValueError("Negative annotations must identify inspected categories explicitly")
        if c["split"] in required_splits and any(not c[k] and negatives.get(k) is not True for k in ANNOTATION_CATEGORIES):
            raise ValueError("Empty annotation arrays require explicit category negatives")
        ids = set()
        for claim in c["claims"]:
            if (claim.get("kind") not in KINDS or not isinstance(claim.get("id"), str) or not claim["id"] or claim["id"] in ids
                    or not isinstance(claim.get("values", {}), dict)):
                raise ValueError("Unknown or duplicate reference claim")
            ids.add(claim["id"])
            if not all(finite(claim.get(k)) for k in ("start", "end")) or not c["start"] <= claim["start"] <= claim["end"] <= c["end"]:
                raise ValueError("Reference claim outside neighborhood")
            if claim["kind"] == "cs_increased":
                values = claim.get("values", {})
                if any(type(values.get(k)) is not int for k in ("before", "after", "delta")) or not values["after"]-values["before"] == values["delta"] > 0:
                    raise ValueError("CS references require exact before/after/delta values")
            if claim["kind"] == "displayed_state" and (not isinstance(claim.get("values", {}).get("field"), str)
                    or "value" not in claim["values"]):
                raise ValueError("Displayed-state references require field/value pairs")
        for binding in c["bindings"]:
            box = binding.get("box")
            if (binding.get("claim_id") not in ids or binding.get("role") not in {"actor", "target"}
                    or not finite(binding.get("time")) or not c["start"] <= binding["time"] <= c["end"]
                    or not isinstance(box, list) or len(box) != 4 or not all(finite(v) for v in box)
                    or not 0 <= box[0] < box[2] <= 1920 or not 0 <= box[1] < box[3] <= 1080):
                raise ValueError("Invalid independent binding region")
        for phase in c["phases"]:
            if (phase.get("phase") not in {"setup", "action", "result"} or not all(finite(phase.get(k)) for k in ("start", "end"))
                    or not c["start"] <= phase["start"] <= phase["end"] <= c["end"]):
                raise ValueError("Invalid independent phase bounds")
        for playback in c["playback"]:
            if (playback.get("kind") not in {"pause", "cut", "rewind", "ownership_change", "clock_ticks"}
                    or not all(finite(playback.get(k)) for k in ("start", "end"))
                    or not c["start"] <= playback["start"] <= playback["end"] <= c["end"]):
                raise ValueError("Invalid independent playback interval")
        for repeat in c["repeats"]:
            if not all(finite(repeat.get(k)) for k in ("start", "end", "earlier_start")) or not c["start"] <= repeat["start"] < repeat["end"] <= c["end"] or not lower <= repeat["earlier_start"] < repeat["start"]:
                raise ValueError("Invalid independent replay correspondence")
    ordered = sorted(clips, key=lambda c: c["start"])
    if any(a["end"] > b["start"] for a, b in zip(ordered, ordered[1:])):
        raise ValueError("Caption neighborhoods overlap")
    if any(max(a["start"], b["start"])-min(a["end"], b["end"]) < 8 for a in clips for b in clips if a["split"] != b["split"]):
        raise ValueError("Caption splits need an eight-second guard")
    return clips


def assessments(root):
    root = Path(root)
    generation = read_json(root/"caption_coverage.json")["key"]
    rows = {}
    for path in sorted((root/"assessments").glob("*.json")):
        row = read_json(path)
        if row["generation_key"] == generation:
            rows[row["caption_id"]] = row
    return rows


def score_clip(clip, root):
    root = Path(root)
    captions, clips = read_rows(root/"captions.jsonl"), read_rows(root/"caption_clips.jsonl")
    reviewed = assessments(root)
    audit = read_json(root/"input-audit.json")
    known = {"frames": {f["id"] for f in audit["frames"]}}
    for table in ("states", "events", "normalized_observations"):
        known[table] = {r["id"] for r in read_rows(root/(table+".jsonl"))}
    by_kind, used = {}, set()
    for kind in sorted(KINDS):
        predicted = [c for c in captions if c["claims"][0]["kind"] == kind]
        references = [r for r in clip["claims"] if r["kind"] == kind]
        matched = duplicates = evidence_valid = binding_references = binding_correct = 0
        onset_error = 0.0
        for caption in predicted:
            claim = caption["claims"][0]
            compatible = [r for r in references if caption["start"] <= r["end"]+.25 and caption["end"] >= r["start"]-.25
                          and all(claim["values"].get(k) == v for k, v in r.get("values", {}).items())]
            candidates = [r for r in compatible if r["id"] not in used]
            if candidates:
                reference = min(candidates, key=lambda r: abs(r["start"]-caption["start"]))
                used.add(reference["id"])
                matched += 1
                onset_error += abs(reference["start"]-caption["start"])
                frame_times = {f["id"]: f["time"] for f in audit["frames"]}
                for binding in [b for b in clip["bindings"] if b["claim_id"] == reference["id"]]:
                    binding_references += 1
                    regions = [r for b in caption["actor_bindings"] if b["role"] == binding["role"] for r in b.get("regions", [])]
                    nearby = [r for r in regions if abs(frame_times.get(r["frame_id"], -100)-binding["time"]) <= .1]
                    def correct_region(region):
                        box, truth = region["box"], binding["box"]
                        if binding["role"] == "target":
                            return iou(box, truth) >= .5
                        center = ((box[0]+box[2])/2, (box[1]+box[3])/2)
                        return truth[0] <= center[0] <= truth[2] and truth[1] <= center[1] <= truth[3] and iou(box, truth) >= .1
                    binding_correct += bool(nearby and any(correct_region(r) for r in nearby))
            elif compatible:
                duplicates += 1
            evidence_valid += all(r["id"] in known.get(r["table"], set()) for r in caption["evidence_refs"])
        judged = [reviewed[c["id"]] for c in predicted if c["id"] in reviewed]
        by_kind[kind] = {"reference_count": len(references), "prediction_count": len(predicted), "matched": matched,
            "precision": matched/len(predicted) if predicted else None, "recall": matched/len(references) if references else None,
            "duplicates": duplicates, "evidence_valid": evidence_valid, "reviewed": len(judged),
            "supported": sum(r["supported"] for r in judged), "actor_correct": sum(r["actor_correct"] for r in judged),
            "target_correct": sum(r["target_correct"] for r in judged), "reviewed_evidence_valid": sum(r["evidence_valid"] for r in judged),
            "binding_reference_count": binding_references, "binding_correct": binding_correct, "onset_error_total": onset_error}
    retained = sum(any(c["start"] <= p["start"] and c["end"] >= p["end"] for c in clips) for p in clip["phases"])
    repeat_predictions = [c for c in clips if c.get("earlier_occurrence_id")]
    clip_index = {c["id"]: c for c in clips}
    repeat_correct = sum(any(abs(c["start"]-r["start"]) <= 1 and abs(clip_index[c["earlier_occurrence_id"]]["start"]-r["earlier_start"]) <= 1 for r in clip["repeats"])
                         for c in repeat_predictions if c["earlier_occurrence_id"] in clip_index)
    playback_correct = 0
    for reference in clip["playback"]:
        nearby = [f for f in audit["frames"] if reference["start"]-.25 <= f["time"] <= reference["end"]+.25]
        if reference["kind"] == "pause":
            correct = any(f.get("playback", {}).get("playback_state") == "paused" for f in nearby)
        elif reference["kind"] == "clock_ticks":
            correct = bool(nearby) and all(f.get("playback", {}).get("playback_state") == "playing" for f in nearby)
        elif reference["kind"] == "ownership_change":
            correct = len({f["owner_epochs"]["selected"] for f in nearby}) > 1
        else:
            reasons = {"game_clock_rewind"} if reference["kind"] == "rewind" else {"visual_cut", "game_clock_discontinuity", "layout_transition"}
            correct = any(reasons & set(f.get("playback", {}).get("discontinuity_reasons", [])) for f in nearby)
        playback_correct += correct
    return {"by_kind": by_kind, "phase_reference_count": len(clip["phases"]), "phases_retained": retained,
            "playback_reference_count": len(clip["playback"]), "playback_correct": playback_correct,
            "repeat_reference_count": len(clip["repeats"]), "repeat_prediction_count": len(repeat_predictions), "repeat_correct": repeat_correct,
            "missing_cases": sorted(CASES-set(clip["case_tags"]))}


def aggregate(results, clips):
    by_kind = {}
    for kind in sorted(KINDS):
        sums = Counter()
        for result in results:
            sums.update({k: v for k, v in result["by_kind"][kind].items() if isinstance(v, int)})
            sums["onset_error_total"] += result["by_kind"][kind]["onset_error_total"]
        n, r, reviewed = sums["prediction_count"], sums["reference_count"], sums["reviewed"]
        by_kind[kind] = {**sums, "precision": sums["matched"]/n if n else None, "recall": sums["matched"]/r if r else None,
            "supported_fraction": sums["supported"]/reviewed if reviewed else None,
            "onset_error_seconds": sums["onset_error_total"]/sums["matched"] if sums["matched"] else None,
            "region_binding_accuracy": sums["binding_correct"]/sums["binding_reference_count"] if sums["binding_reference_count"] else None,
            "status": "tested" if min(n, r, reviewed) >= MINIMUM_PER_KIND else "untested_insufficient_cases"}
    totals = Counter()
    for value in by_kind.values():
        totals.update({k: v for k, v in value.items() if isinstance(v, int)})
    n, judged = totals["prediction_count"], totals["reviewed"]
    missing = sorted(CASES-{t for c in clips for t in c["case_tags"]})
    support = totals["supported"]/judged if judged else None
    required = {"cs_increased", "attack", "cast", "targeted_attack", "minion_last_hit", "ability_hit"}
    passed = bool(n >= MINIMUM_CLAIMS and judged == n and support >= .95
                  and totals["actor_correct"]/judged >= .95 and totals["target_correct"]/judged >= .95
                  and totals["evidence_valid"] == n and totals["reviewed_evidence_valid"] == n
                  and not missing and all(by_kind[k]["status"] == "tested" and by_kind[k]["supported_fraction"] >= .95
                                          and by_kind[k]["precision"] >= .95 for k in required))
    return {"passed": passed, "supported_fraction": support, "reviewed": judged, "prediction_count": n,
        "minimum_claims": MINIMUM_CLAIMS, "minimum_per_kind": MINIMUM_PER_KIND, "by_kind": by_kind, "missing_cases": missing,
        "actor_binding_accuracy": totals["actor_correct"]/judged if judged else None,
        "target_binding_accuracy": totals["target_correct"]/judged if judged else None,
        "evidence_valid_fraction": totals["evidence_valid"]/n if n else None,
        "phase_reference_count": sum(r["phase_reference_count"] for r in results), "phases_retained": sum(r["phases_retained"] for r in results),
        "playback_reference_count": sum(r["playback_reference_count"] for r in results), "playback_correct": sum(r["playback_correct"] for r in results),
        "repeat_reference_count": sum(r["repeat_reference_count"] for r in results),
        "repeat_prediction_count": sum(r["repeat_prediction_count"] for r in results), "repeat_correct": sum(r["repeat_correct"] for r in results),
        "acceptance_scope": "selected-player source/profile pilot only"}


def experiment_captions(store, config, *, labels=None, initialize=False, draft=False, freeze=False,
                        infer=False, action_profile=None, icon_references=None, runner=None):
    directory = Path(labels) if labels else store.path("benchmarks/captions")
    profile = load_profile(action_profile)
    if initialize:
        return initialize_labels(store, directory, profile)
    target = directory/"captions.json"
    data = read_json(target)
    clips = validate_labels(store, data, profile, required_splits=("development",) if freeze else () if draft else ("development", "held_out"))
    reference = Path(icon_references) if icon_references else Path(config.asset_cache)/"manifest.json"
    signature = {"source": data["source_sha256"], "profile": profile, "masks": config.masks, "policy": POLICY,
        "model": read_json(LOCK)["models"]["intake"], "infer": infer,
        "icons": AbilityIdentity(reference).signature if reference.exists() else None,
        "implementation": implementation_hash(COMPONENTS+["caption_benchmark.py", "caption_review.py"]),
        "development": [c for c in clips if c["split"] == "development"]}
    frozen = directory/"development-freeze.json"
    if freeze:
        if frozen.exists() and read_json(frozen)["signature"] != digest(signature):
            raise ValueError("Caption development choices already frozen; use a new benchmark revision")
        atomic_json(frozen, {"signature": digest(signature), "choices": signature})
        return {"status": "development_frozen", "freeze": str(frozen)}
    if not draft and (not frozen.exists() or read_json(frozen)["signature"] != digest(signature)):
        raise ValueError("Freeze unchanged development choices before held-out caption scoring")
    path = store.path("experiments/captions/benchmarks/"+digest([data, signature, draft])[:20]+".json")
    report = {"schema_version": 1, "experimental": True, "draft": draft, "status": "running_experimental",
              "clips": {}, "quality_metrics": None, "production_evidence_modified": False, "report": str(path)}
    try:
        for clip in clips:
            if draft and clip["split"] != "development":
                continue
            if digest(read_json(target)) != digest(data):
                raise ValueError("Caption references changed during evaluation; restart with preserved partial results")
            result = caption_pilot(store, config, clip["start"], clip["end"], infer=infer, runner=runner,
                                   action_profile=action_profile, icon_references=icon_references)
            if list((Path(result["root"])/"reviews").glob("*.json")):
                raise ValueError("Corrected pilot predictions cannot be used for benchmark acceptance")
            report["clips"][clip["id"]] = {"split": clip["split"], "root": result["root"], "resources": result["resource_metrics"],
                                           "metrics": None if draft else score_clip(clip, result["root"])}
            atomic_json(path, report)
        if digest(read_json(target)) != digest(data):
            raise ValueError("Caption references changed during evaluation")
        if not draft:
            held = [r["metrics"] for r in report["clips"].values() if r["split"] == "held_out"]
            report["quality_metrics"] = aggregate(held, [c for c in clips if c["split"] == "held_out"])
        report["status"] = "completed_experimental" if draft or report["quality_metrics"]["passed"] else "validation_pending_or_failed"
    except BaseException as exc:
        report.update(status="failed_experimental", error=str(exc))
        raise
    finally:
        atomic_json(path, report)
    return report
