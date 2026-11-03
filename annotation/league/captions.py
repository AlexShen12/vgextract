"""Render checkable visual claims; wording never promotes a counter into an action."""
from __future__ import annotations

from collections import Counter
import json

from .core import atomic_json, consistent_review, finite, read_json, uid


DEPENDENCIES = ["states", "events", "normalized_observations", "issues"]
EVENT_CRITERIA = {
    "movement": {"movement_visible"},
    "attack": {"attack_visible"},
    "targeted_attack": {"attack_visible", "targeting_visible"},
    "cast": {"cast_visible"},
    "recall": {"recall_channel_visible"},
    "minion_last_hit": {"minion_death_visible", "actor_action_linked_to_death"},
    "ability_hit": {"ability_cast_visible", "target_impact_visible", "cast_linked_to_impact"},
}
NUMERIC_FIELDS = {"cs": "CS", "kills": "kill counter", "deaths": "death counter",
                  "assists": "assist counter", "level": "level"}
TEXT_FIELDS = {"player_name": "player name", "game_clock": "game clock",
               "notice": "notice", "objective_notice": "objective notice", "central_notice": "central notice"}


def ref(table, key):
    return {"table": table, "id": key}


def refs(*groups):
    return sorted({(r["table"], r["id"]) for group in groups for r in group})


def combine_refs(*groups):
    return [ref(table, key) for table, key in refs(*groups)]


def usable(row, pilot=False):
    return ((not row.get("experimental") or (pilot and row.get("caption_pilot") is True)) and row.get("review_status") != "rejected"
            and row.get("evidence_modality", "visual") == "visual")


def event_grounding(event, frame_times, *, frame_context=None, body=None):
    """Validate explicit actor/target bindings and criterion-to-frame links.

    Criterion names alone, high confidence, HUD cooldowns, CS and gold changes
    cannot establish an impact or last hit. All links must resolve in the clip.
    """
    kind = event["kind"]
    if kind not in EVENT_CRITERIA:
        raise ValueError("unsupported_event_kind")
    if event.get("target") and kind not in {"targeted_attack", "minion_last_hit", "ability_hit"}:
        raise ValueError("unqualified_target_attribution")
    start, end = event["start"], event["end"]
    if not finite(start) or not finite(end) or not 0 <= start <= end:
        raise ValueError("invalid_event_interval")
    evidence = event.get("evidence_ids", [])
    if not isinstance(evidence, list) or not evidence or any(not isinstance(f, str) for f in evidence):
        raise ValueError("missing_event_frames")
    frames = set(evidence)
    if any(f not in frame_times or not start <= frame_times[f] <= end for f in frames):
        raise ValueError("unknown_or_outside_event_frame")

    def binding(role):
        value = event.get(role + "_binding")
        entity = event.get(role)
        if (not isinstance(value, dict) or not isinstance(entity, str) or entity in {"", "unknown"}
                or value.get("entity") != entity or not isinstance(value.get("binding"), str)
                or value["binding"] in {"", "unknown"}):
            raise ValueError("unresolved_" + role + "_binding")
        ids = value.get("frame_ids")
        if (not isinstance(ids, list) or not ids or any(not isinstance(f, str) for f in ids)
                or not set(ids) <= frames):
            raise ValueError("missing_" + role + "_binding_evidence")
        return {"id": uid("caption-binding", role, value), "role": role,
                "entity": entity, "binding": value["binding"], "scope": "scene_entity",
                "evidence_refs": [ref("frames", f) for f in sorted(set(ids))]}

    bindings = [binding("actor")]
    if kind in {"minion_last_hit", "ability_hit", "targeted_attack"} or event.get("target"):
        bindings.append(binding("target"))
    visual = event.get("visual_evidence")
    if not isinstance(visual, list):
        raise ValueError("missing_visual_criteria")
    criteria = {}
    for cue in visual:
        if not isinstance(cue, dict) or not isinstance(cue.get("criterion"), str):
            raise ValueError("invalid_visual_criterion")
        ids = cue.get("frame_ids")
        if (not isinstance(ids, list) or not ids or any(not isinstance(f, str) for f in ids)
                or not set(ids) <= frames):
            raise ValueError("missing_criterion_evidence")
        criteria.setdefault(cue["criterion"], set()).update(ids)
    if not EVENT_CRITERIA[kind] <= criteria.keys():
        raise ValueError("missing_visual_criteria")
    if kind in {"attack", "targeted_attack", "cast", "minion_last_hit", "ability_hit"}:
        fps = event.get("sampling_fps", 0)
        cue_frames = {f for name in EVENT_CRITERIA[kind] for f in criteria[name]}
        cue_times = sorted({frame_times[f] for f in cue_frames})
        # First-frame-at-target decoding quantizes times to native PTS. Pilot
        # audits record source FPS so this tolerance cannot disguise sparse input.
        tolerance = .002
        if frame_context is not None:
            source_rates = [frame_context.get(f, {}).get("source_fps", 0) for f in cue_frames]
            if (finite(fps) and source_rates and all(finite(rate) and rate >= 16 for rate in source_rates)
                    and fps < min(source_rates)-.01):
                tolerance += 1/min(source_rates)
        if (not finite(fps) or fps < 16 or len(cue_times) < 2
                or any(b-a > 1/fps + tolerance for a, b in zip(cue_times, cue_times[1:]))):
            raise ValueError("insufficient_temporal_evidence")
    if kind == "minion_last_hit" and not event["target"].startswith("minion:"):
        raise ValueError("unresolved_minion_target")
    if kind in {"minion_last_hit", "ability_hit"}:
        action, outcome = (("actor_action_linked_to_death", "minion_death_visible") if kind == "minion_last_hit"
                           else ("ability_cast_visible", "target_impact_visible"))
        if min(frame_times[f] for f in criteria[action]) > max(frame_times[f] for f in criteria[outcome]):
            raise ValueError("outcome_precedes_action")
    if kind == "ability_hit" and (not isinstance(event.get("ability"), str)
                                  or not event["ability"].strip() or len(event["ability"]) > 120):
        raise ValueError("unresolved_ability")
    if frame_context is not None:
        # Pilot events require a single supported scene/owner context, including
        # every intervening sampled frame, not just matching endpoint labels.
        context = [f for f in frame_context.values() if start <= f["time"] <= end]
        if not context or any(f not in frame_context for f in frames):
            raise ValueError("missing_playback_context")
        owner = event.get("owner_id")
        if any(not f.get("layout_valid") or f["segment_id"] != event.get("playback_segment_id")
               or f["owner_epochs"].get(owner) != event.get("owner_epoch")
               or f.get("playback", {}).get("playback_state") in {"paused", "unknown"} for f in context):
            raise ValueError("unsupported_playback_or_owner_continuity")
        ordered = sorted(context, key=lambda f: f["time"])
        if any(b["time"]-a["time"] > 1/16+1/a.get("source_fps", 1000000)+.002 for a, b in zip(ordered, ordered[1:])):
            raise ValueError("unsupported_playback_gap")
        for f in frames:
            bound = (body or {}).get(f, {})
            if (not bound.get("binding_visible") or bound.get("review_status") == "rejected" or bound.get("actor_track") != event["actor_binding"]["binding"]
                    or bound.get("owner_id") != owner):
                raise ValueError("missing_actor_binding_at_cue")
        if kind == "movement" and len({frame_times[f] for f in criteria["movement_visible"]}) < 2:
            raise ValueError("insufficient_movement_evidence")
        known_names = {body[f].get("champion") for f in frames}-{None}
        if not event["actor"].startswith("player:") and known_names != {event["actor"]}:
            raise ValueError("missing_actor_identity_evidence")
        if len(bindings) > 1:
            samples = event["target_binding"].get("regions", [])
            regions = {r["frame_id"]: r.get("box") for r in samples}
            needed = set().union(*(criteria[name] for name in EVENT_CRITERIA[kind]))
            if not needed <= regions.keys():
                raise ValueError("missing_target_binding_at_cue")
            for f in needed:
                box = regions[f]
                width, height = frame_context[f].get("size", [1920, 1080])
                if (not isinstance(box, list) or len(box) != 4 or not all(finite(x) for x in box)
                        or not 0 <= box[0] < box[2] <= width or not 0 <= box[1] < box[3] <= height):
                    raise ValueError("invalid_target_region")
            bindings[1]["regions"] = samples
        bindings[0]["regions"] = [{"frame_id": f, "box": body[f]["actor_region"]} for f in sorted(frames)]
    return bindings, [{"criterion": name, "evidence_refs": [ref("frames", f) for f in sorted(criteria[name])]}
                      for name in sorted(EVENT_CRITERIA[kind])]


def _display(field, value):
    if field in NUMERIC_FIELDS and finite(value) and value >= 0 and int(value) == value:
        return NUMERIC_FIELDS[field], str(int(value))
    if field in TEXT_FIELDS and isinstance(value, str) and value.strip():
        return TEXT_FIELDS[field], json.dumps(value, ensure_ascii=False)
    if field == "kda" and isinstance(value, dict) and all(
            finite(value.get(k)) and value[k] >= 0 and int(value[k]) == value[k] for k in ("kills", "deaths", "assists")):
        return "K/D/A", "/".join(str(int(value[k])) for k in ("kills", "deaths", "assists"))
    if field in {"health", "displayed_gold", "game_clock", "objective_timer"}:
        # Keep the displayed precision, especially abbreviated gold and clocks.
        if isinstance(value, dict) and isinstance(value.get("displayed_text"), str):
            return field.replace("_", " "), json.dumps(value["displayed_text"], ensure_ascii=False)
        if finite(value) and value >= 0:
            return field.replace("_", " "), str(value)
    if field in {"health_bar", "health_fraction"} and finite(value) and 0 <= value <= 1:
        return "health bar proportion", str(value)
    return None


def construct_captions(states, events, observations, *, interval, source_sha256, frame_times=None, issues=(),
                       pilot=False, frame_context=None, body=None):
    """Return timestamped captions and omissions, without model inference.

    A slot binding may describe a HUD reading without inventing a champion.
    Counter deltas additionally require a resolved identity and verified segment.
    Point readings are never described as continuously true between samples.
    """
    indexed = {o["id"]: o for o in observations}
    times = dict(frame_times or {})
    times.update({o["frame_id"]: o["time"] for o in observations})
    captions, omissions, previous = [], [], {}

    def omit(row, reason):
        omissions.append({"record_id": row["id"], "reason": reason})

    def allowed(row, start, end):
        if not usable(row, pilot):
            return "ineligible_evidence"
        if not all(finite(t) for t in (start, end)) or not interval[0] <= start <= end <= interval[1]:
            return "outside_source_interval"
        for issue in issues:
            if issue.get("kind") not in {"editorial_overlay", "layout_mismatch", "identity_conflict"}:
                continue
            a, b = issue.get("start", issue.get("time")), issue.get("end", issue.get("time"))
            if finite(a) and finite(b) and start <= b and a <= end:
                return "evidence_issue"
        return None

    def emit(kind, text, start, end, bindings, evidence, values, criteria=()):
        evidence = combine_refs(evidence, *[b["evidence_refs"] for b in bindings])
        claim = {"id": uid("caption-claim", kind, start, end, bindings, evidence, values),
                 "kind": kind, "text": text, "interval": [start, end], "values": values,
                 "actor_binding_id": bindings[0]["id"],
                 "target_binding_id": bindings[1]["id"] if len(bindings) > 1 else None,
                 "evidence_refs": evidence, "criteria": list(criteria),
                 "support": "accepted_evidence", "review_status": "unreviewed"}
        captions.append({"schema_version": 1, "id": uid("caption", claim),
                         "source_sha256": source_sha256, "time_basis": "source_video_seconds",
                         "start": start, "end": end, "text": text, "claims": [claim],
                         "actor_bindings": bindings, "evidence_refs": evidence,
                         "evidence_modality": "visual", "review_status": "unreviewed",
                         "generator": "deterministic-evidence-v1"})

    for state in sorted(states, key=lambda r: (r["time"], r["id"])):
        field, value = state["field"], state.get("value")
        subject = state.get("slot_id") or state.get("entity")
        key = (subject, field)  # a segment change invalidates the previous comparison
        old = previous.pop(key, None)
        reason = allowed(state, state["time"], state["time"])
        if reason or state.get("status") != "confirmed":
            omit(state, reason or "unconfirmed_state")
            continue
        shown = _display(field, value)
        if shown is None:
            omit(state, "unsupported_state_field_or_value")
            continue
        evidence = [ref("states", state["id"])]
        supporting = [indexed[i] for i in state.get("evidence_ids", []) if i in indexed]
        supporting = [o for o in supporting if usable(o, pilot) and o.get("visibility") == "visible"
                      and o.get("value") == value and o.get("entity") == state.get("entity")
                      and o.get("time") == state["time"] and o.get("field") == field
                      and o.get("slot_id") == state.get("slot_id")
                      and (o.get("review_status") == "human-verified" or
                           (o.get("acceptance") == "accepted" if o.get("schema_version", 1) >= 2
                            else o.get("confidence", 0) >= .75))]
        if not supporting:
            omit(state, "missing_observation_evidence")
            continue
        evidence += [ref("normalized_observations", o["id"]) for o in supporting]
        frames = sorted({o["frame_id"] for o in supporting})
        evidence += [ref("frames", f) for f in frames]
        entity = state.get("entity")
        entity = entity if entity and entity != "unknown" else None
        slot = state.get("slot_id")
        if not entity and not slot:
            omit(state, "unresolved_actor_binding")
            continue
        label = state.get("display_label") or entity or f"HUD slot {slot}"
        binding = {"id": uid("caption-binding", entity, slot, evidence), "role": "actor",
                   "entity": entity, "binding": slot or entity,
                   "scope": "resolved_entity" if entity else "hud_slot",
                   "evidence_refs": evidence}
        old_state = old[0] if old else None
        # Two confirmed readings can measure a displayed counter change, never
        # the number, source or animation of individual last hits.
        comparable = (old and entity and old_state.get("entity") == entity
                      and state.get("playback_segment_id") is not None
                      and state.get("playback_segment_id") == old_state.get("playback_segment_id")
                      and state.get("continuity_verified") is True and old_state.get("continuity_verified") is True
                      and state["time"] > old_state["time"]
                      and not allowed(state, old_state["time"], state["time"]))
        if field == "cs" and comparable and value > old_state["value"]:
            comparison_evidence = combine_refs(old[1], evidence)
            comparison_binding = {**binding, "id": uid("caption-binding", entity, slot, comparison_evidence),
                                  "evidence_refs": comparison_evidence}
            emit("cs_increased", f"{label}'s displayed CS increased from {old_state['value']} to {value} "
                 f"(+{value-old_state['value']}) between the two observations.",
                 old_state["time"], state["time"], [comparison_binding], comparison_evidence,
                 {"field": "cs", "before": old_state["value"], "after": value, "delta": value-old_state["value"]})
        elif not (comparable and old_state["value"] == value):
            emit("displayed_state", f"{label}: displayed {shown[0]} is {shown[1]}.",
                 state["time"], state["time"], [binding], evidence, {"field": field, "value": value})
        else:
            omit(state, "unchanged_reading")
        previous[key] = (state, evidence)

    for event in sorted(events, key=lambda r: (r["start"], r["id"])):
        reason = allowed(event, event["start"], event["end"])
        confidence = event.get("confidence", 0)
        if not reason and not (pilot and event.get("acceptance") == "grounded") and event.get("review_status") != "human-verified" and (
                not finite(confidence) or confidence < .75):
            reason = "unaccepted_event"
        if reason:
            omit(event, reason)
            continue
        try:
            bindings, criteria = event_grounding(event, times, frame_context=frame_context, body=body)
        except ValueError as exc:
            omit(event, str(exc))
            continue
        actor, target, kind = event.get("actor_label", event["actor"]), event.get("target_label", event.get("target")), event["kind"]
        text = {"movement": f"{actor} is observed moving.",
                "attack": f"{actor} is observed attacking.",
                "targeted_attack": f"{actor} is observed directing an attack at {target}.",
                "cast": f"{actor} is observed casting an ability.",
                "recall": f"{actor} is observed channeling recall.",
                "minion_last_hit": f"{actor} is observed landing the last hit on {target}.",
                "ability_hit": f"{actor}'s {event.get('ability')} is observed hitting {target}."}[kind]
        evidence = [ref("events", event["id"])] + [ref("frames", f) for f in event["evidence_ids"]]
        emit(kind, text, event["start"], event["end"], bindings, evidence,
             {"actor": event["actor"], "target": event.get("target"), "ability": event.get("ability")}, criteria)
        if pilot:
            captions[-1].update(playback_segment_id=event.get("playback_segment_id"), owner_id=event.get("owner_id"),
                                owner_epoch=event.get("owner_epoch"), actor_track=event["actor_binding"]["binding"])

    unique = {c["id"]: c for c in captions}
    captions = sorted(unique.values(), key=lambda c: (c["start"], c["end"], c["id"]))
    if pilot:
        for caption in captions:
            caption.update(experimental=True, caption_pilot=True)
    return captions, omissions


@consistent_review
def generate_captions(store, config):
    from .review import frame_index

    manifest = read_json(store.path("manifest.json"))
    captions, omissions = construct_captions(
        store.rows("states"), store.effective(store.rows("events"), "events"),
        store.rows("normalized_observations"), interval=manifest["interval"],
        source_sha256=manifest["source"]["sha256"], issues=store.rows("issues"),
        frame_times={key: f["time"] for key, f in frame_index(store).items()})
    store.check_stop()
    store.write("captions", captions)
    store.write("caption_omissions", omissions)
    summary = {"generated": len(captions), "claims": sum(len(c["claims"]) for c in captions),
               "counts": dict(Counter(claim["kind"] for c in captions for claim in c["claims"])),
               "omitted": dict(Counter(o["reason"] for o in omissions)), "vlm_output_tokens": 0,
               "note": "Accepted visual evidence only; independent caption quality acceptance remains pending"}
    atomic_json(store.path("caption_coverage.json"), summary)
    store.commit_stage("generate-captions", store.stage_key("generate-captions", config, DEPENDENCIES), **summary)
    return summary
