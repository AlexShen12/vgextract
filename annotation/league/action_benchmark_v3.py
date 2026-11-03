"""Independent v3 benchmark: CPU core by default, explicit optional scene inference."""
import copy
from collections import Counter
import os
from pathlib import Path

from .action_benchmark import detection_metrics, identity_metrics, initialize_actions, validate_labels
from .action_icons import AbilityIdentity
from .action_v3 import action_window, load_profile
from .action_state import POLICY
from .core import atomic_json, digest, implementation_hash, read_json, read_rows
from .local_models import LOCK, LocalVLRunner
from .intake_profile import check_box

CASES = {"rapid_sequence", "quiet_negative", "multiple_champions", "occlusion", "ownership_change", "item_transition",
         "player_bar", "ambiguous_bar", "minion_last_hit", "jungle_credit", "passive_gold", "purchase",
         "recall_start", "recall_complete", "recall_interrupted", "recall_censored"}


def validate(store, data, profile, draft):
    if data.get("schema_version") != 3:
        raise ValueError("Actions v3 requires a separate version-3 label manifest")
    clips = validate_labels(store, data, profile, draft=draft)
    if not draft and (data.get("independent_of_predictions") is not True or not data.get("labeler")):
        raise ValueError("Independent labeler/provenance required; model predictions are not reference labels")
    for clip in clips:
        if not draft and (clip.get("independent_of_predictions") is not True or not clip.get("labeler")):
            raise ValueError("Each verified clip needs its independent labeler/provenance")
        for row in clip["bindings"]:
            if type(row.get("binding_visible")) is not bool:
                raise ValueError("Binding references must explicitly label visibility, including negative frames")
            if row["binding_visible"] and (not row.get("bar_box") or not row.get("actor_region")):
                raise ValueError("Positive binding references need independent bar and actor regions")
            if row["binding_visible"]:
                check_box(row["bar_box"], profile["size"])
                check_box(row["actor_region"], profile["size"])
        for row in clip["actions"]:
            if row.get("kind") not in {"creep_credit","minion_last_hit","jungle_last_hit","recall","ability_cast","basic_attack","movement","item_use","gold_change"}:
                raise ValueError("Unknown independent action kind")
            if row["kind"] == "recall" and row.get("terminal_status") not in {"completed","interrupted","ending_unknown"}:
                raise ValueError("Recall references need an explicit terminal status")
    return clips


def overlap(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0]))*max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union else 0


def score_clip(clip, report):
    directory = Path(report["results"]["core"]["directory"])
    body, identities = read_rows(directory/"body_readings.jsonl"), read_rows(directory/"identities.jsonl")
    actions = read_rows(directory/"actions.jsonl")
    transitions = read_rows(directory/"transitions.jsonl")
    established = [t for t in transitions if t.get("activation_supported") is True]
    resolved = correct = false_binds = positive = resolved_positive = 0
    for reference in clip["bindings"]:
        matches = [r for r in body if r["owner_id"] == reference["owner_id"] and abs(r["time"]-reference["time"]) <= .034]
        candidate = min(matches, key=lambda r: abs(r["time"]-reference["time"])) if matches else None
        positive += reference["binding_visible"]
        if candidate and candidate["binding_visible"]:
            resolved += 1
            if not reference["binding_visible"]:
                false_binds += 1
            else:
                resolved_positive += 1
                a, b = candidate["actor_region"], reference["actor_region"]
                center = ((a[0]+a[2])/2, (a[1]+a[3])/2)
                correct += overlap(candidate["bar_box"], reference["bar_box"]) >= .5 and b[0] <= center[0] <= b[2] and b[1] <= center[1] <= b[3]
    def metrics(predictions):
        predictions = [p for p in predictions if p["evidence_class"] != "unresolved"]
        result = {}
        for kind in ("creep_credit", "minion_last_hit", "recall", "ability_cast"):
            references = [r for r in clip["actions"] if r["kind"] == kind or kind == "creep_credit" and r["kind"] == "minion_last_hit"]
            detected = [r for r in predictions if r["kind"] == kind or kind == "creep_credit" and r["kind"] == "minion_last_hit"]
            result[kind] = detection_metrics(references, detected, ("owner_id",))
            if kind == "ability_cast":
                matched = []
                used = set()
                for reference in references:
                    choices = [(abs(p["start"]-reference["start"]),i,p) for i,p in enumerate(detected)
                               if i not in used and p.get("owner_id") == reference.get("owner_id")
                               and p["start"] <= reference["end"]+.25 and p["end"] >= reference["start"]-.25]
                    if choices:
                        _,i,p = min(choices,key=lambda choice:choice[:2]);used.add(i)
                        if reference.get("slot_id"):
                            matched.append(p.get("slot_id") == reference["slot_id"])
                result[kind]["slot_accuracy"] = sum(matched)/len(matched) if matched else None
                result[kind]["slot_reference_matches"] = len(matched)
            if kind == "creep_credit":
                complete_counts = all(r.get("count_delta") is not None for r in references)
                result[kind]["aggregate_count_error"] = sum(r.get("count_delta") or 0 for r in detected)-sum(r["count_delta"] for r in references) if complete_counts else None
            if kind == "recall":
                pairs = [(r, [p for p in detected if p["start"] <= r["end"]+.25 and p["end"] >= r["start"]-.25]) for r in references]
                unique = [(r, p[0]) for r, p in pairs if len(p) == 1]
                result[kind]["terminal_status_accuracy"] = sum(r["terminal_status"] == p["terminal_status"] for r, p in unique)/len(unique) if unique else None
                result[kind]["end_error_seconds"] = sum(abs(r["end"]-p["end"]) for r, p in unique)/len(unique) if unique else None
        return result
    ordered = sorted([r for r in clip["actions"] if r.get("sequence_id")],key=lambda r:r["start"])
    order_pairs = order_correct = 0
    for first,second in zip(ordered,ordered[1:]):
        if first["sequence_id"] != second["sequence_id"] or first["end"] >= second["start"]:
            continue
        order_pairs += 1
        matches = [[p for p in actions if p.get("owner_id") == r.get("owner_id") and p.get("slot_id") == r.get("slot_id")
                    and p["evidence_class"] != "unresolved" and abs(p["start"]-r["start"]) <= .5] for r in (first,second)]
        order_correct += len(matches[0]) == len(matches[1]) == 1 and matches[0][0]["end"] < matches[1][0]["start"]
    return {"binding": {"accuracy": correct/resolved if resolved else None, "coverage": resolved_positive/positive if positive else None,
                        "bound_frame_fraction": resolved/len(clip["bindings"]) if clip["bindings"] else None,
                        "reference_frames": len(clip["bindings"]), "positive_reference_frames": positive, "false_binds": false_binds},
            "ability_icon_identity": identity_metrics(clip["identities"], identities, "hud_champion"),
            "transitions": detection_metrics(clip["transitions"], established, ("owner_id","slot_id")),
            "transition_candidates": detection_metrics(clip["transitions"], transitions, ("owner_id","slot_id")),
            "unresolved_transition_candidates": len(transitions)-len(established),
            "core_only": metrics(read_rows(directory/"core_actions.jsonl")), "with_scene": metrics(actions),
            "corroborated_only": metrics([a for a in actions if a["evidence_class"] == "corroborated"]),
            "scene_only": metrics([a for a in actions if a["evidence_class"] == "scene-only"]),
            "rapid_order_accuracy": order_correct/order_pairs if order_pairs else None, "rapid_order_pairs": order_pairs,
            "unresolved": sum(a["evidence_class"] == "unresolved" for a in actions),
            "evidence_classes": dict(Counter(a["evidence_class"] for a in actions)),
            "resampling_intervals": len(read_rows(directory/"resampling.jsonl"))}


def experiment_actions(store, config, *, labels=None, initialize=False, draft=False, freeze=False,
                       action_profile=None, assistance=None, runner=None, icon_references=None, infer=False):
    if assistance:
        raise ValueError("--assistance is obsolete for actions v3; use --icon-references")
    directory = Path(labels) if labels else store.path("benchmarks/actions-v3")
    target = directory/"actions.json"
    profile = load_profile(action_profile)
    if initialize:
        if target.exists():
            raise ValueError("Refusing to overwrite existing independent action labels")
        prior = store.path("benchmarks/actions-v2/actions.json")
        if prior.exists():
            data = copy.deepcopy(read_json(prior))
            data.update(schema_version=3, forked_from=str(prior), independent_of_predictions=False, labeler=None)
            for clip in data["clips"]:
                clip.update(verified=False, independent_of_predictions=False, labeler=None,
                            identities=[], bindings=[], transitions=[], actions=[], case_tags=[])
            atomic_json(target, data)
        else:
            initialize_actions(store, config, directory, action_profile or Path(__file__).with_name("profiles")/"midbeast-actions-1080p-v2.json")
        return {"status": "unverified_action_labels_created", "labels": str(target), "clips": 20, "quality_metrics": None}
    data = read_json(target)
    clips = validate(store, data, profile, draft=True)
    reference = Path(icon_references) if icon_references else Path(config.asset_cache)/"manifest.json"
    signature = {"profile": profile, "policy": POLICY, "source": data["source_sha256"], "infer": infer,
        "implementation": implementation_hash(["action_v3.py", "action_body.py", "action_events.py", "action_icons.py", "action_benchmark_v3.py", "action_benchmark.py", "action_pipeline.py", "action_state.py", "numeric_reader.py", "readers.py", "media.py", "core.py", "local_models.py"]),
        "icon_references": AbilityIdentity(reference).signature if reference.exists() else None,
        "model": read_json(LOCK)["models"]["intake"] if infer else None,
        "development": [c for c in clips if c["split"] == "development"]}
    frozen = directory/"development-freeze.json"
    if freeze:
        if data.get("independent_of_predictions") is not True or not data.get("labeler") or not all(c["verified"] for c in clips if c["split"] == "development"):
            raise ValueError("Independent verified development labels required before freezing")
        if any(c.get("independent_of_predictions") is not True or not c.get("labeler") for c in clips if c["split"] == "development"):
            raise ValueError("Independent development clip provenance required before freezing")
        if frozen.exists() and read_json(frozen)["signature"] != digest(signature):
            raise ValueError("Development choices already frozen; use a new benchmark revision")
        atomic_json(frozen, {"signature": digest(signature), "choices": signature})
        return {"status": "development_frozen", "freeze": str(frozen)}
    if not draft:
        validate(store, data, profile, draft=False)
        if not frozen.exists() or read_json(frozen)["signature"] != digest(signature):
            raise ValueError("Freeze unchanged development choices before held-out scoring")
    path = store.path("experiments/actions-v3/benchmarks/"+digest([data, signature, draft])[:20]+".json")
    report = {"schema_version": 3, "experimental": True, "status": "running_experimental", "draft": draft,
              "clips": {}, "quality_metrics": None, "missing_cases": sorted(CASES-{t for c in clips for t in c["case_tags"]}),
              "production_evidence_modified": False}
    owned = runner is None
    try:
        if infer and owned:
            runner = LocalVLRunner(store, config.vl_python or os.environ.get("LEAGUE_VL_PYTHON", ""), config.asset_cache, image_limit=8)
        for clip in clips:
            if digest(read_json(target)) != digest(data):
                raise ValueError("Reference labels changed during evaluation; preserve partial results and restart")
            if draft and clip["split"] != "development":
                continue
            result = action_window(store, config, clip["start"], clip["end"], detect=True, infer=infer, runner=runner,
                                   action_profile=action_profile, icon_references=icon_references)
            report["clips"][clip["id"]] = {"split": clip["split"], "report": result["report"],
                "resources": result["resource_metrics"], "metrics": score_clip(clip, result) if not draft else None}
            atomic_json(path, report)
        report["status"] = "completed_experimental"
        if not draft:
            report["quality_metrics"] = {c: r["metrics"] for c, r in report["clips"].items()}
    except BaseException as exc:
        report.update(status="failed_experimental", error=str(exc))
        raise
    finally:
        if infer and owned and runner:
            runner.close()
        atomic_json(path, report)
    return report
