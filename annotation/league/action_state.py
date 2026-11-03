"""Champion-independent temporal contracts for the League action experiment."""
from __future__ import annotations

from pathlib import Path

from .core import digest, finite, read_json, uid
from .intake_profile import check_box, load_profile

VERSION = 2
STATES = ("ready", "cooling_down", "charge", "recast", "unavailable", "unknown")
CLASSES = ("corroborated", "hud-supported", "scene-only", "unresolved")
POLICY = {"hud_fps": 16, "scene_fps": 4, "review_fps": 16,
          "before": 1, "after": 2, "burst_seconds": .5, "link_seconds": 1,
          "native_margin": .25, "quiet_audit_fraction": .05, "seed": 17}


def load_action_profile(path=None):
    path = Path(path) if path else Path(__file__).with_name("profiles") / "midbeast-actions-1080p-v1.json"
    profile = read_json(path)
    if profile.get("schema_version") != VERSION or profile.get("size") != [1920, 1080]:
        raise ValueError("Unsupported action profile schema/geometry")
    anchor = load_profile(profile["anchor_profile"])
    if anchor["size"] != profile["size"]:
        raise ValueError("Action and anchor geometry differ")
    for key in ("scene_box", "hud_box", "clock_box"):
        check_box(profile[key], profile["size"])
    hud = profile["hud_box"]
    if hud[2]-hud[0] > 960 or hud[3]-hud[1] > 300:
        raise ValueError("Action HUD view must fit the native 960 by 300 pixel budget")
    def in_hud(box):
        if not (hud[0] <= box[0] < box[2] <= hud[2] and hud[1] <= box[1] < box[3] <= hud[3]):
            raise ValueError("HUD identity and slots must be inside the supplied native HUD view")
    owners = profile.get("owners", [])
    if not owners or len({r["id"] for r in owners}) != len(owners):
        raise ValueError("Action profile needs distinct HUD owners")
    for owner in owners:
        check_box(owner["identity_box"], profile["size"])
        in_hud(owner["identity_box"])
        if len({s["id"] for s in owner["slots"]}) != len(owner["slots"]):
            raise ValueError("Duplicate action slot")
        for slot in owner["slots"]:
            if slot["kind"] not in ("ability", "summoner", "item"):
                raise ValueError("Invalid action slot kind")
            check_box(slot["box"], profile["size"])
            in_hud(slot["box"])
            check_box(slot["countdown_box"], profile["size"])
            if not (slot["box"][0] <= slot["countdown_box"][0] < slot["countdown_box"][2] <= slot["box"][2]
                    and slot["box"][1] <= slot["countdown_box"][1] < slot["countdown_box"][3] <= slot["box"][3]):
                raise ValueError("Countdown crop must be inside its slot")
    if any(not finite(v) or not 0 <= v <= 1 for v in profile["thresholds"].values()):
        raise ValueError("Invalid action reader thresholds")
    return profile


def transitions(readings):
    """Unknown gaps do not establish new activations. Countdown ticks never count."""
    previous, result = {}, []
    for row in sorted(readings, key=lambda r: (r["time"], r["owner_id"], r["slot_id"])):
        key = (row["segment_id"], row["owner_epoch"], row["owner_id"], row["slot_id"], row.get("item_epoch", 0))
        old = previous.get(key)
        if old and row["time"] - old["time"] > 2 / POLICY["hud_fps"] + 1e-6:
            old = None  # Missing observations cannot establish an activation bracket.
        reason = None
        if old and row["state"] != "unknown" and old["state"] != "unknown":
            if old["state"] in ("ready", "recast") and row["state"] == "cooling_down":
                reason = "cooldown_started"
            elif (finite(old.get("charges")) and finite(row.get("charges"))
                  and row["charges"] < old["charges"] and row.get("charge_consumption_visible")):
                reason = "charge_consumed"
            elif row.get("recast_visible") and not old.get("recast_visible"):
                reason = "recast_activation"
        elif old and old["state"] == "unknown" and row["state"] == "cooling_down":
            reason = "cooldown_visibility_changed"
        if reason:
            result.append({"id": uid("action-transition", key, old["id"], row["id"], reason),
                           "owner_id": row["owner_id"], "owner_epoch": row["owner_epoch"],
                           "segment_id": row["segment_id"], "slot_id": row["slot_id"],
                           "slot_kind": row["slot_kind"], "reason": reason,
                           "activation_supported": reason != "cooldown_visibility_changed",
                           "start": old["time"], "end": row["time"],
                           "evidence_ids": [old["id"], row["id"]],
                           "frame_ids": [old["frame_id"], row["frame_id"]],
                           "item_effect": row.get("item_effect", "unknown"), "name": row.get("name")})
        previous[key] = row
    return result


def deduplicate_scene(rows):
    """Merge only shared-frame proposals of the same actor, slot and action."""
    result = []
    for row in sorted(rows, key=lambda r: (r["start"], r["id"])):
        match = next((r for r in result if all(r.get(k) == row.get(k) for k in
                      ("mode", "segment_id", "owner_epoch", "actor_track", "kind", "slot_id", "name"))
                      and r.get("actor_track")
                      and set(r.get("action_frame_ids", r["frame_ids"])) & set(row.get("action_frame_ids", row["frame_ids"]))), None)
        if match:
            match["start"], match["end"] = min(match["start"], row["start"]), max(match["end"], row["end"])
            match["frame_ids"] = sorted(set(match["frame_ids"] + row["frame_ids"]))
            match["action_frame_ids"] = sorted(set(match.get("action_frame_ids", match["frame_ids"]) + row.get("action_frame_ids", row["frame_ids"])))
            match["evidence_ids"] = sorted(set(match["evidence_ids"] + row["evidence_ids"]))
        else:
            result.append(dict(row))
    return result


def link_actions(hud, scene, bindings, mode="unassisted", policy=None):
    policy = policy or POLICY
    scene = deduplicate_scene([s for s in scene if s["mode"] == mode])
    used, result = set(), []
    for transition in hud:
        available = [b for b in bindings if b["mode"] == mode and all(b[k] == transition[k] for k in
                     ("owner_id", "owner_epoch", "segment_id"))
                     and b["start"] <= transition["start"] and transition["end"] <= b["end"]]
        available.sort(key=lambda b: not bool(b.get("actor_track")))
        binding = available[0] if available else None
        candidates = [s for s in scene if binding and binding.get("actor_track") and s["id"] not in used
                      and s["segment_id"] == transition["segment_id"]
                      and s["owner_epoch"] == transition["owner_epoch"]
                      and s.get("actor_track") == binding["actor_track"]
                      and s.get("slot_id") == transition["slot_id"]
                      and s["kind"] == ("item_use" if transition["slot_kind"] == "item" else "ability_cast")
                      and s["start"] <= transition["end"] + policy["link_seconds"]
                      and s["end"] >= transition["start"] - policy["link_seconds"]]
        # One scene proposal compatible with multiple activations is unresolved too.
        unique = len(candidates) == 1 and sum(
            t["slot_id"] == transition["slot_id"] and t["owner_epoch"] == transition["owner_epoch"]
            and t["segment_id"] == transition["segment_id"]
            and candidates[0]["start"] <= t["end"] + policy["link_seconds"]
            and candidates[0]["end"] >= t["start"] - policy["link_seconds"] for t in hud) == 1
        matched = candidates[0] if unique else None
        scene_item_identity = (matched and matched.get("name") and any(e.get("cue") == "item_activation" for e in matched.get("evidence", [])))
        item_unknown = transition["slot_kind"] == "item" and not (transition["item_effect"] == "active"
            or transition["item_effect"] == "unknown" and scene_item_identity)
        name_conflict = bool(matched and matched.get("name") and transition.get("name")
                             and matched["name"].casefold() != transition["name"].casefold())
        support = "unresolved" if not transition.get("activation_supported", True) or not binding or item_unknown or name_conflict or len(candidates) > 1 or (candidates and not unique) else (
            "corroborated" if matched else "hud-supported")
        if matched and support == "corroborated":
            used.add(matched["id"])
        result.append({**transition, "id": uid("linked-action", transition["id"], mode),
                       "transition_id": transition["id"], "mode": mode,
                       "evidence_class": support, "champion": binding["champion"] if binding else None,
                       "actor_track": binding["actor_track"] if binding else None,
                       "actor_description": binding["actor_description"] if binding else "HUD owner unresolved",
                       "kind": ("item_transition" if item_unknown else "item_use") if transition["slot_kind"] == "item" else "ability_cast",
                       "name": transition.get("name") or (matched.get("name") if matched else None),
                       "scene_ids": [matched["id"]] if matched else [],
                       "competing_scene_ids": [s["id"] for s in candidates] if not unique or name_conflict else [],
                       "binding_ids": [b["id"] for b in available],
                       "evidence_ids": sorted(set(transition["evidence_ids"]
                            + [e for b in available for e in b["evidence_ids"]]
                            + (matched["evidence_ids"] if matched else []))),
                       "frame_ids": sorted(set(transition["frame_ids"] + (matched["frame_ids"] if matched else []))),
                       "experimental": True, "review_status": "unreviewed", "precision": "sample_bounded",
                       "outcome": None})
    for row in scene:
        if row["id"] not in used:
            result.append({**row, "evidence_class": "scene-only", "champion": row.get("actor"),
                           "experimental": True, "review_status": "unreviewed", "outcome": None})
    return sorted(result, key=lambda r: (r["start"], r["id"]))


def escalation_windows(hud, linked, frames, policy=None):
    policy = policy or POLICY
    bounds = {}
    for frame in frames:
        segment = bounds.setdefault(frame["segment_id"], [frame["time"], frame["time"]])
        segment[1] = frame["time"]
    result = []
    unresolved = {a.get("transition_id") for a in linked if a["evidence_class"] == "unresolved"}
    for row in hud:
        reasons = []
        if row["id"] in unresolved:
            reasons.append("unresolved_binding_or_scene")
        if any(other["id"] != row["id"] and other["segment_id"] == row["segment_id"]
               and abs(other["end"] - row["end"]) <= policy["burst_seconds"] for other in hud):
            reasons.append("rapid_sequence")
        if reasons:
            lower, upper = bounds[row["segment_id"]]
            result.append({"transition_id": row["id"], "segment_id": row["segment_id"],
                           "start": max(lower, row["start"] - policy["before"]),
                           "end": min(upper + 1 / policy["hud_fps"], row["end"] + policy["after"]),
                           "native_start": max(lower, row["start"] - policy["native_margin"]),
                           "native_end": min(upper + 1 / policy["hud_fps"], row["end"] + policy["native_margin"]),
                           "reasons": reasons})
    # Sample stable one-second bins by hash, independently of model outputs.
    for segment, (lower, upper) in bounds.items():
        for second in range(int(lower), int(upper) + 1):
            if any(t["segment_id"] == segment and second - 1 <= t["end"] < second + 2 for t in hud):
                continue
            if int(digest([policy["seed"], segment, second])[:8], 16) / 2**32 < policy["quiet_audit_fraction"]:
                result.append({"transition_id": None, "segment_id": segment,
                               "start": max(lower, second), "end": min(upper + 1 / policy["hud_fps"], second + 1),
                               "reasons": ["quiet_audit"]})
    return result
