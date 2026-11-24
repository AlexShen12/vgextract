"""Source reference scoring for the first v3 scene pass."""
from collections import Counter
from pathlib import Path
from statistics import mean

from .action_v3 import KINDS
from .core import read_json, read_rows

POINT_TOLERANCE = .25
INTERVAL_IOU = .5
ROLES = {"selected", "other", "unknown"}
SLOTS = {"Q", "W", "E", "R"}


def validate_scene_references(clips):
    for clip in clips:
        if not isinstance(clip.get("inspection_note"), str) or not clip["inspection_note"].strip():
            raise ValueError(f"Scene clip {clip['id']} needs a source inspection note, including quiet clips")
        for row in clip["actions"]:
            role = row.get("actor_role")
            if row["kind"] not in KINDS or role not in ROLES or row.get("owner_id") != ("selected" if role == "selected" else None):
                raise ValueError(f"Invalid scene kind or actor role in {clip['id']}")
            if not isinstance(row.get("evidence_note"), str) or not row["evidence_note"].strip():
                raise ValueError(f"Scene events in {clip['id']} need visible reference evidence")
            if type(row.get("boundary_resolvable")) is not bool or any(
                type(row.get(flag, False)) is not bool for flag in ("left_censored", "right_censored")
            ):
                raise ValueError(f"Invalid scene boundary flags in {clip['id']}")
            if row.get("slot_id") is not None and (not isinstance(row["slot_id"], str) or not row["slot_id"]):
                raise ValueError(f"Invalid scene slot in {clip['id']}")


def temporal_fit(reference, proposal):
    if reference["kind"] != proposal["kind"]:
        return None
    a, b = reference["start"], reference["end"]
    c, d = proposal["start"], proposal["end"]
    if a == b:
        error = max(abs(a-c), abs(a-d))
        return 1-error/POINT_TOLERANCE if error <= POINT_TOLERANCE + 1e-9 else None
    iou = max(0, min(b, d)-max(a, c))/(max(b, d)-min(a, c))
    return iou if iou >= INTERVAL_IOU else None


def match_scene_events(references, proposals):
    """Maximize the number of one-to-one matches using deterministic time preferences."""
    edges = {}
    for p, proposal in enumerate(proposals):
        candidates = [(fit, abs(reference["start"]-proposal["start"]), r)
                      for r, reference in enumerate(references)
                      if (fit := temporal_fit(reference, proposal)) is not None]
        edges[p] = [r for _, _, r in sorted(candidates, key=lambda x: (-x[0], x[1], x[2]))]
    assigned = {}

    def augment(p, seen):
        for r in edges[p]:
            if r in seen:
                continue
            seen.add(r)
            if r not in assigned or augment(assigned[r], seen):
                assigned[r] = p
                return True
        return False

    for p in sorted(edges, key=lambda i: (len(edges[i]), proposals[i]["start"], i)):
        augment(p, set())
    matched = {p: r for r, p in assigned.items()}
    return matched, {p for p in edges if edges[p] and p not in matched}


def prediction_role(row):
    if row.get("owner_id") == "selected":
        return "selected"
    return "other" if (row.get("actor_track") or "").startswith("other-") else "unknown"


def scene_ledger(clip, report):
    references = clip["actions"]
    proposals = read_rows(Path(report["results"]["scene_baseline"]["proposals"]))
    frames = {row["id"]: row["time"] for row in read_json(Path(report["input_audit"]))["frames"]}
    pairs, duplicates = match_scene_events(references, proposals)
    return {"clip_id": clip["id"], "references": references,
            "unmatched_reference_indexes": sorted(set(range(len(references)))-set(pairs.values())),
            "proposals": [{**row, "actor_role": prediction_role(row), "matched_reference_index": pairs.get(i),
                           "duplicate": i in duplicates, "visible_support_review": "pending",
                           "cited_evidence": [{**e, "source_time": frames[e["frame_id"]]} for e in row["evidence"]]}
                          for i, row in enumerate(proposals)],
            "request_statuses": dict(Counter(row["status"] for row in report["requests"]))}


def fraction(n, d):
    return n/d if d else None


def summarize_scene(ledgers):
    rows = list(ledgers)
    references = [r for clip in rows for r in clip["references"]]
    proposals = [p for clip in rows for p in clip["proposals"]]
    matches = [(clip["references"][p["matched_reference_index"]], p) for clip in rows for p in clip["proposals"]
               if p["matched_reference_index"] is not None]
    requests = Counter()
    for clip in rows:
        requests.update(clip["request_statuses"])
    categories = {}
    for kind in KINDS:
        actual = sum(r["kind"] == kind for r in references)
        predicted = [p for p in proposals if p["kind"] == kind]
        pairs = [(r, p) for r, p in matches if r["kind"] == kind]
        onsets = [abs(r["start"]-p["start"]) for r, p in pairs if r["boundary_resolvable"] and not r.get("left_censored")]
        endings = [abs(r["end"]-p["end"]) for r, p in pairs if r["boundary_resolvable"] and not r.get("right_censored")]
        categories[kind] = {"tested": bool(actual), "references": actual, "predictions": len(predicted),
                            "matched": len(pairs), "false_positives": len(predicted)-len(pairs),
                            "misses": actual-len(pairs), "duplicates": sum(p["duplicate"] for p in predicted),
                            "precision": fraction(len(pairs), len(predicted)), "recall": fraction(len(pairs), actual),
                            "f1": fraction(2*len(pairs), actual+len(predicted)) if actual else None,
                            "onset_mae_seconds": mean(onsets) if onsets else None, "onset_matches": len(onsets),
                            "end_mae_seconds": mean(endings) if endings else None, "end_matches": len(endings)}

    def attribution(eligible, expected, resolved):
        actual = sum(eligible(r) for r in references)
        pairs = [(r, p) for r, p in matches if eligible(r)]
        attempted = [(r, p) for r, p in pairs if resolved(p)]
        correct = sum(expected(r) == expected(p) for r, p in attempted)
        return {"references": actual, "event_matches": len(pairs), "resolved": len(attempted), "correct": correct,
                "incorrect": len(attempted)-correct, "unresolved": len(pairs)-len(attempted),
                "accuracy": fraction(correct, len(attempted)), "coverage": fraction(len(attempted), len(pairs)),
                "end_to_end_recall": fraction(correct, actual)}

    known_actor = lambda r: r["actor_role"] != "unknown"
    known_slot = lambda r: r["kind"] == "ability_cast" and r["actor_role"] == "selected" and r.get("slot_id") in SLOTS
    return {"categories": categories, "actor_role": attribution(known_actor, lambda r: r["actor_role"], known_actor),
            "ability_slot": attribution(known_slot, lambda r: r.get("slot_id"), known_slot),
            "request_statuses": dict(requests), "visible_support_review": dict(Counter(p["visible_support_review"] for p in proposals))}
