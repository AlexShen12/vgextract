"""Source-time caption grouping. Context padding never becomes a visual claim."""
from __future__ import annotations

from collections import defaultdict

from .captions import combine_refs, ref
from .core import uid


POLICY = {
    "hud_fps": 16, "scene_fps": 4, "review_fps": 16,
    "setup_seconds": 2, "result_seconds": 3, "merge_gap_seconds": 2,
    "max_clip_seconds": 30, "native_margin": .25, "output_tokens": 4096,
    "playback": {"pause_seconds": 2, "motion_threshold": .002, "cut_threshold": .3},
    "repeat": {"mean_error": .015, "anchor_count": 3, "minimum_span": 1,
               "time_tolerance": .08, "minimum_motion": .004},
}


def context_key(frame, owner="selected"):
    return (frame["segment_id"], owner, frame["owner_epochs"][owner])


def contexts(frames, interval, policy=POLICY):
    """Partition once across all decode/inference packets; unknowns are barriers."""
    groups = []
    for frame in sorted({f["id"]: f for f in frames}.values(), key=lambda f: f["time"]):
        if not frame.get("layout_valid") or frame.get("playback", {}).get("playback_state") in {"unknown", "paused"}:
            if groups:
                groups[-1]["end"] = min(groups[-1]["end"], frame["time"])
            continue
        key = context_key(frame)
        if (not groups or groups[-1]["context"] != key
                or frame["time"]-groups[-1]["frames"][-1]["time"] > 1/policy["hud_fps"]+1/frame.get("source_fps", 1000000)+.002):
            if groups:
                groups[-1]["end"] = min(groups[-1]["end"], frame["time"])
            groups.append({"context": key, "start": frame["time"], "frames": []})
        groups[-1]["frames"].append(frame)
        groups[-1]["end"] = min(interval[1], frame["time"]+1/policy["hud_fps"])
    return groups


def deduplicate_events(events):
    """Only shared action cues can identify duplicate observations of an action."""
    result = []
    action_cues = {"attack_visible", "cast_visible", "ability_cast_visible", "actor_action_linked_to_death",
                   "movement_visible", "recall_channel_visible"}
    def action_frames(row):
        return {f for cue in row["visual_evidence"] if cue["criterion"] in action_cues for f in cue["frame_ids"]}
    for event in sorted(events, key=lambda e: (e["start"], e["id"])):
        match = next((r for r in result if all(r.get(k) == event.get(k) for k in
                     ("kind", "actor", "target", "ability", "playback_segment_id", "owner_epoch"))
                     and r["actor_binding"]["binding"] == event["actor_binding"]["binding"]
                     and r.get("target_binding", {}).get("binding") == event.get("target_binding", {}).get("binding")
                     and action_frames(r) & action_frames(event)), None)
        if match is None:
            result.append({**event, "proposal_ids": [event["id"]]})
            continue
        # Keep the strongest complete chain intact; union only provenance. This
        # cannot manufacture a dense chain by combining incompatible packets.
        ids = sorted(set(match["proposal_ids"] + [event["id"]]))
        if (event.get("sampling_fps", 0), len(event["evidence_ids"])) > (match.get("sampling_fps", 0), len(match["evidence_ids"])):
            match.clear()
            match.update(event)
        match["proposal_ids"] = ids
    conflicts = set()
    for i, a in enumerate(result):
        for b in result[i+1:]:
            if (a["kind"] == b["kind"] and a["kind"] in {"targeted_attack", "ability_hit", "minion_last_hit"}
                    and a["actor_binding"]["binding"] == b["actor_binding"]["binding"]
                    and a.get("playback_segment_id") == b.get("playback_segment_id")
                    and a.get("owner_epoch") == b.get("owner_epoch")
                    and a.get("target") != b.get("target") and action_frames(a) & action_frames(b)):
                conflicts.update((a["id"], b["id"]))
    return [r for r in result if r["id"] not in conflicts], [
        {"record_id": r["id"], "reason": "competing_target_attribution"} for r in result if r["id"] in conflicts]


def segment_captions(captions, frames, interval, source_sha256, policy=POLICY):
    clips = []
    by_context = defaultdict(list)
    for caption in captions:
        by_context[(caption.get("playback_segment_id"), caption.get("owner_id"), caption.get("owner_epoch"))].append(caption)
    for context in contexts(frames, interval, policy):
        selected = sorted((c for c in by_context[context["context"]]
                           if context["start"] <= c["start"] <= c["end"] < context["end"]),
                          key=lambda c: (c["start"], c["end"], c["id"]))
        episodes = []
        for caption in selected:
            if (not episodes or caption["start"]-max(c["end"] for c in episodes[-1]) > policy["merge_gap_seconds"]):
                episodes.append([])
            episodes[-1].append(caption)
        for episode in episodes:
            scene = [c for c in episode if c["claims"][0]["kind"] not in {"displayed_state", "cs_increased"}]
            anchors = scene or episode
            lower = max(context["start"], min(episode[0]["start"], anchors[0]["start"]-policy["setup_seconds"]))
            upper = min(context["end"], max(max(c["end"] for c in episode), max(c["end"] for c in anchors)+policy["result_seconds"]))
            chain = []
            while lower < upper:
                end = min(upper, lower+policy["max_clip_seconds"])
                if end < upper:
                    quiet = sorted({c["end"] for c in episode if lower < c["end"] <= end
                                    and not any(x["start"] < c["end"] < x["end"] for x in episode)})
                    if quiet and quiet[-1] > lower+policy["max_clip_seconds"]/2:
                        end = quiet[-1]
                included = [c for c in episode if c["start"] < end and (c["end"] > lower or c["start"] == c["end"] == lower)]
                # Context-only tails do not need their own clip.
                if included:
                    action = [c for c in included if c["claims"][0]["kind"] not in {"displayed_state", "cs_increased"}] or included
                    first, last = min(c["start"] for c in action), max(c["end"] for c in action)
                    phases = {
                        "setup": [ref("frames", f["id"]) for f in context["frames"] if lower <= f["time"] < min(first, end)],
                        "action": combine_refs(*[c["evidence_refs"] for c in action]),
                        "result": [ref("frames", f["id"]) for f in context["frames"] if max(last, lower) < f["time"] < end],
                    }
                    phases["action"] = [r for r in phases["action"] if r["table"] != "frames" or any(
                        f["id"] == r["id"] and lower <= f["time"] < end for f in context["frames"])]
                    clip = {"schema_version": 1, "id": uid("caption-clip", source_sha256, context["context"], lower, end, [c["id"] for c in included]),
                            "source_sha256": source_sha256, "time_basis": "source_video_seconds", "start": lower, "end": end,
                            "playback_segment_id": context["context"][0], "owner_id": context["context"][1], "owner_epoch": context["context"][2],
                            "caption_ids": [c["id"] for c in included], "text": " ".join(c["text"] for c in included),
                            "phases": phases, "setup_status": "observed_context" if phases["setup"] else "unknown",
                            "result_status": "observed_context" if phases["result"] else "unknown",
                            "left_censored": first-lower < policy["setup_seconds"] or first < lower,
                            "right_censored": end-last < policy["result_seconds"] or last > end,
                            "previous_clip_id": None, "next_clip_id": None, "repeat_group_id": None, "earlier_occurrence_id": None,
                            "experimental": True, "review_status": "unreviewed"}
                    chain.append(clip)
                lower = end
            for i, clip in enumerate(chain):
                clip["previous_clip_id"] = chain[i-1]["id"] if i else None
                clip["next_clip_id"] = chain[i+1]["id"] if i+1 < len(chain) else None
            clips += chain
    return sorted(clips, key=lambda c: (c["start"], c["id"]))


def link_repeats(clips, frames, policy=POLICY):
    """Conservative moving-sequence correspondence; clocks alone never link."""
    import numpy as np
    samples = sorted({f["id"]: f for f in frames if f.get("scene_sample") and f.get("playback", {}).get("scene_signature")}.values(), key=lambda f: f["time"])
    vectors = {f["id"]: np.frombuffer(bytes.fromhex(f["playback"]["scene_signature"]), dtype=np.uint8).astype(float)/255 for f in samples}
    cfg = policy["repeat"]
    def anchors(clip):
        choices = [f for f in samples if clip["start"] <= f["time"] < clip["end"]]
        if len(choices) < cfg["anchor_count"]:
            return []
        return [choices[round(i*(len(choices)-1)/(cfg["anchor_count"]-1))] for i in range(cfg["anchor_count"])]
    for i, clip in enumerate(clips):
        a = anchors(clip)
        if not a or a[-1]["time"]-a[0]["time"] < cfg["minimum_span"]:
            continue
        # Static pauses and static scenes cannot establish replay identity.
        if max(float(abs(vectors[f["id"]]-vectors[a[0]["id"]]).mean()) for f in a[1:]) < cfg["minimum_motion"]:
            continue
        matches = []
        for earlier in clips[:i]:
            if earlier["end"] > clip["start"] or earlier["playback_segment_id"] == clip["playback_segment_id"]:
                continue
            b = anchors(earlier)
            if not b or any(abs((x["time"]-a[0]["time"])-(y["time"]-b[0]["time"])) > cfg["time_tolerance"] for x, y in zip(a, b)):
                continue
            if all(float(abs(vectors[x["id"]]-vectors[y["id"]]).mean()) <= cfg["mean_error"] for x, y in zip(a, b)):
                matches.append(earlier)
        groups = {m.get("repeat_group_id") or m["id"] for m in matches}
        if len(groups) == 1:
            earlier = matches[0]
            group = earlier.get("repeat_group_id") or uid("caption-repeat", earlier["id"])
            earlier["repeat_group_id"] = clip["repeat_group_id"] = group
            clip["earlier_occurrence_id"] = earlier["id"]
            clip["repeat_evidence_refs"] = [ref("frames", f["id"]) for f in a+anchors(earlier)]
    return clips
