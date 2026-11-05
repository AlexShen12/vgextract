"""Experimental association of saved scene casts with Q/W/E/R HUD candidates."""
import re
from collections import defaultdict

from .action_state import POLICY
from .core import uid

SLOTS = frozenset("QWER")
QUIET_SECONDS = .25
CONFIRM_SECONDS = .35
MAX_OCR_DELAY = 2 / POLICY["hud_fps"]
LINK_SECONDS = 1 / POLICY["scene_fps"] + 1 / POLICY["hud_fps"]
SCENE_OFFSETS = (-.375, -.25, -.125, 0, .125, .25, .375)


def cooldown_seconds(row, minimum_confidence):
    if row["state"] == "cooling_down":
        return row["seconds"]
    # The existing OCR records this text, but the decimal-only parser drops it.
    raw = row["raw"]
    if raw["score"] >= minimum_confidence:
        match = re.fullmatch(r"([1-9]\d?):([0-5]\d)", raw["raw_text"].strip())
        if match:
            return int(match[1])*60 + int(match[2])
    return None


def cooldown_candidates(readings, thresholds):
    """Find a visual change followed by a persistent countdown, without inferring readiness."""
    groups = defaultdict(list)
    for row in readings:
        if row["slot_kind"] == "ability" and row["slot_id"] in SLOTS:
            groups[(row["owner_id"], row["owner_epoch"], row["segment_id"], row["slot_id"])].append(row)
    candidates = []
    for key, group in groups.items():
        group.sort(key=lambda row: row["time"])
        seconds = [cooldown_seconds(row, thresholds["ocr_confidence"]) for row in group]
        for index, row in enumerate(group[1:], 1):
            if row["pixel_change"] < thresholds["pixel_change"]:
                continue
            now = row["time"]
            before = [i for i in range(index) if now - QUIET_SECONDS - 1e-6 <= group[i]["time"]]
            after = [i for i in range(index, len(group)) if group[i]["time"] <= now + CONFIRM_SECONDS]
            if len(before) < 3 or any(seconds[i] is not None for i in before):
                continue
            window = before + after
            if any(group[i].get("identity_changed") for i in window) or any(
                group[j]["time"] - group[i]["time"] > MAX_OCR_DELAY + 1e-6 for i, j in zip(window, window[1:])
            ):
                continue
            observed = [i for i in after if seconds[i] is not None]
            if len(observed) < 3 or group[observed[0]]["time"] - now > MAX_OCR_DELAY + 1e-6:
                continue
            values = [seconds[i] for i in observed]
            if values[0] - values[-1] > 1 or any(a < b for a, b in zip(values, values[1:])):
                continue
            previous = group[index-1]
            candidates.append({"id": uid("hud-replay-candidate", key, previous["id"], row["id"]),
                "owner_id": row["owner_id"], "owner_epoch": row["owner_epoch"], "segment_id": row["segment_id"],
                "slot_id": row["slot_id"], "start": previous["time"], "end": now,
                "onset_frame_ids": [previous["frame_id"], row["frame_id"]],
                "confirmed_readings": [{"frame_id": group[i]["frame_id"], "time": group[i]["time"],
                                        "seconds": seconds[i]} for i in observed],
                "reason": "persistent_cooldown_after_icon_change"})
    return sorted(candidates, key=lambda row: (row["start"], row["slot_id"]))


def grounded_scene_frames(candidate, frames, body, bindings):
    """Select up to seven frames around a uniquely bound HUD candidate."""
    context = ("owner_id", "owner_epoch", "segment_id")
    tracks = {binding["actor_track"] for binding in bindings if binding.get("actor_track") and
              all(binding[k] == candidate[k] for k in context) and
              binding["start"] <= candidate["start"] <= candidate["end"] <= binding["end"]}
    if len(tracks) != 1:
        return None, []
    track = next(iter(tracks))
    visible = {row["frame_id"] for row in body if row.get("binding_visible") and row["actor_track"] == track}
    epochs = next(frame["owner_epochs"] for frame in frames if frame["id"] == candidate["onset_frame_ids"][-1])
    usable = [frame for frame in frames if frame["layout_valid"] and frame["id"] in visible and
              frame["segment_id"] == candidate["segment_id"] and frame["owner_epochs"] == epochs]
    selected = {}
    for offset in SCENE_OFFSETS:
        time = candidate["end"] + offset
        frame = min(usable, key=lambda frame: abs(frame["time"] - time), default=None)
        if frame and abs(frame["time"] - time) <= 1 / POLICY["hud_fps"] + .001:
            selected[frame["id"]] = frame
    window = sorted(selected.values(), key=lambda frame: frame["time"])
    if not window or not window[0]["time"] < candidate["end"] <= window[-1]["time"]:
        return track, []
    return track, window


def associate_scene_casts(scene, candidates, bindings):
    """Preserve the scene slot and show a HUD suggestion only for a unique pairing."""
    scene = [row for row in scene if row["kind"] == "ability_cast"]
    edges = defaultdict(list)
    reverse = defaultdict(list)
    for index, proposal in enumerate(scene):
        if not proposal.get("owner_id") or not proposal.get("actor_track"):
            continue
        for candidate in candidates:
            context = ("owner_id", "owner_epoch", "segment_id")
            if any(proposal[k] != candidate[k] for k in context):
                continue
            if not proposal["start"] - LINK_SECONDS <= candidate["end"] <= proposal["end"] + LINK_SECONDS:
                continue
            if not any(binding["actor_track"] == proposal["actor_track"] and
                       all(binding[k] == candidate[k] for k in context) and
                       binding["start"] <= candidate["start"] <= candidate["end"] <= binding["end"]
                       for binding in bindings):
                continue
            edges[index].append(candidate)
            reverse[candidate["id"]].append(index)

    result = []
    for index, proposal in enumerate(scene):
        nearby = edges[index]
        candidate = nearby[0] if len(nearby) == 1 and len(reverse[nearby[0]["id"]]) == 1 else None
        scene_slot = proposal.get("slot_id")
        hud_slot = candidate["slot_id"] if candidate else None
        if candidate:
            status = "agrees" if scene_slot == hud_slot else "slot_conflict" if scene_slot else "slot_suggested"
        else:
            status = "ambiguous" if nearby else "scene_only"
        result.append({"scene_id": proposal["id"], "scene_start": proposal["start"], "scene_end": proposal["end"],
                       "scene_slot": scene_slot, "hud_slot_suggestion": hud_slot, "status": status,
                       "linked_candidate": candidate, "eligible_candidate_ids": [row["id"] for row in nearby]})
    return result
