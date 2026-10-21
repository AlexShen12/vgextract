"""Conservative evidence fusion. Unseen intervals never become confirmed game state."""
from __future__ import annotations

from collections import defaultdict
import re
from .core import (COMPOSITE_CRITERIA, EVENTS, Store, digest, finite, uid,
                   validate_observation, consistent_review)


def normalize_output(output: dict, task: dict, route, identities=None) -> tuple[list, list, list]:
    if not isinstance(output, dict):
        raise ValueError("Observer answer must be an object")
    if any(not isinstance(output.get(key, []), list) for key in ("observations", "events", "issues")):
        raise ValueError("Observer observations/events/issues must be lists")
    frames = {f["id"]: f for f in task["frames"]}
    observations, events, issues = [], [], []
    expanded = []
    for item in output.get("observations", []):
        if isinstance(item, dict) and isinstance(item.get("values"), dict):
            if not isinstance(item.get("field_visibility", {}), dict):
                issues.append({"route": route.name, "kind": "invalid_output", "detail": "field_visibility must be an object"})
                continue
            for name, value in item["values"].items():
                expanded.append({k: v for k, v in {**item, "field": name, "value": value,
                    "visibility": item.get("field_visibility", {}).get(name, item.get("visibility", "visible"))}.items()
                                 if k not in {"values", "field_visibility"}})
        else:
            expanded.append(item)
    for item in expanded:
        try:
            frame = frames[item["frame_id"]]
            if item["field"] not in route.fields:
                raise ValueError("Field not allowed for this fixed route")
            row = {"id": uid("obs", task["id"], item), "task_id": task["id"],
                   "frame_id": frame["id"], "source_id": frame["source_id"],
                   "time": frame["time"], "route": route.name, "box": route.box,
                   "entity": item.get("entity") or "unknown", "binding": item.get("binding"),
                   "field": item["field"], "value": item.get("value"),
                   "visibility": item.get("visibility", "unknown"),
                   "confidence": item.get("confidence", 0),
                   "identity_evidence": item.get("identity_evidence"),
                   "review_status": "unreviewed", "model": task["model"],
                   "pass": task["pass"], "sampling_fps": task["fps"]}
            if row["field"] in {"displayed_gold", "game_clock"} and row["visibility"] == "visible":
                value = row["value"]
                if not isinstance(value, str) and not (isinstance(value, dict) and isinstance(value.get("displayed_text"), str)):
                    raise ValueError("Gold/clock observations must retain the exact displayed text")
            if identities is not None:
                from .knowledge import identity_key
                raw = row["entity"]
                canonical = identities.get(identity_key(raw))
                if canonical:
                    row["entity"] = canonical
                elif raw != "unknown" and raw != "match" and not re.match(r"^(team|structure|ward|objective|player):", raw):
                    row.update(entity="unknown", value=None, visibility="unknown")
                    issues.append({"time": row["time"], "route": route.name,
                                   "kind": "identity_conflict", "detail": "Entity is not a patch-roster identity or typed world object"})
            if route.name in {"left_sidebar", "right_sidebar"}:
                side = route.name.split("_")[0]
                if not re.fullmatch(side + r":row[1-5]", row.get("binding") or ""):
                    row.update(entity="unknown", value=None, visibility="unknown")
                    issues.append({"time": row["time"], "route": route.name, "kind": "identity_conflict",
                                   "detail": "Sidebar value lacks a calibrated side/row binding"})
            if route.name == "selected_hud" and row["field"] not in {"identity", "selection"}:
                if not row["identity_evidence"] or row["entity"] == "unknown":
                    row.update(value=None, visibility="unknown", entity="unknown")
                    issues.append({"time": row["time"], "route": route.name,
                                   "kind": "identity_conflict", "detail": "Selected field has no grounded identity"})
            observations.append(validate_observation(row))
            if row["visibility"] == "unreadable":
                issues.append({"time": row["time"], "route": route.name,
                               "kind": "unreadable", "detail": row["field"]})
            elif row["visibility"] == "visible" and row["confidence"] < .75:
                issues.append({"time": row["time"], "route": route.name,
                               "kind": "low_confidence", "detail": row["field"]})
        except (KeyError, TypeError, ValueError) as exc:
            issues.append({"route": route.name, "kind": "invalid_output", "detail": str(exc)})
    for item in output.get("events", []):
        try:
            start, end = frames[item["start_frame"]], frames[item["end_frame"]]
            kind = item["kind"]
            if kind not in EVENTS or start["time"] > end["time"]:
                raise ValueError("Invalid event kind or bounds")
            if not COMPOSITE_CRITERIA.get(kind, set()) <= set(item.get("criteria", [])):
                raise ValueError("Composite event lacks required visual criteria")
            confidence = item.get("confidence", 0)
            if not finite(confidence) or not 0 <= confidence <= 1:
                raise ValueError("Invalid event confidence")
            # Event feed can attribute notice-backed events, not world-only casts/movement.
            if route.model == "instruct" and route.name not in {"event_feed", "central_notice"}:
                raise ValueError("This HUD route cannot independently label world events")
            if route.model == "instruct" and kind not in {"death", "objective_takedown", "tower_destruction"}:
                raise ValueError("HUD notice does not establish this event")
            row = {"id": uid("event", task["id"], item), "kind": kind,
                   "actor": item.get("actor"), "target": item.get("target"), "object": item.get("object"),
                   "location": item.get("location"), "start": start["time"], "end": end["time"],
                   "evidence_ids": sorted({start["id"], end["id"]}), "routes": [route.name],
                   "task_ids": [task["id"]], "criteria": item.get("criteria", []),
                   "detail": item.get("detail", ""), "confidence": confidence,
                   "sampling_fps": task["fps"], "review_status": "unreviewed"}
            if identities is not None:
                from .knowledge import identity_key
                for key in ("actor", "target"):
                    if row[key]:
                        row[key] = identities.get(identity_key(str(row[key])), "unknown")
            events.append(row)
            if confidence < .75:
                issues.append({"time": start["time"], "route": route.name,
                               "kind": "low_confidence", "detail": kind})
        except (KeyError, TypeError, ValueError) as exc:
            issues.append({"route": route.name, "kind": "invalid_output", "detail": str(exc)})
    for issue in output.get("issues", []):
        if isinstance(issue, dict):
            frame = frames.get(issue.get("frame_id"), task["frames"][0])
            issues.append({"time": frame["time"], "route": route.name,
                           "kind": issue.get("kind", "unknown"), "detail": str(issue.get("detail", ""))})
    return observations, events, issues


def fuse_observations(observations: list[dict]) -> tuple[list, list]:
    grouped = defaultdict(list)
    for row in observations:
        if row.get("review_status") != "rejected":
            grouped[(row["entity"], row["field"], round(row["time"], 3))].append(row)
    states, conflicts = [], []
    previous = {}
    for (entity, field, time), group in sorted(grouped.items(), key=lambda pair: pair[0][2]):
        visible = [r for r in group if r["visibility"] == "visible"
                   and (r["confidence"] >= .75 or r.get("review_status") == "human-verified")
                   and entity != "unknown"]
        verified = [r for r in visible if r.get("review_status") == "human-verified"]
        candidates = verified or visible
        values = {digest(r["value"]) for r in candidates}
        status = "conflict" if len(values) > 1 else "confirmed" if values else "unobserved"
        key = (entity, field)
        state = {"id": uid("state", entity, field, time, sorted(r["id"] for r in group)),
                 "entity": entity, "field": field, "time": time,
                 "value": candidates[0]["value"] if status == "confirmed" else None,
                 "status": status, "evidence_ids": [r["id"] for r in group],
                 "frame_ids": sorted({r["frame_id"] for r in group}),
                 "last_confirmed_time": previous.get(key),
                 "review_status": "human-verified" if verified and status == "confirmed" else "unreviewed"}
        if status == "confirmed":
            state["last_confirmed_time"] = time
            previous[key] = time
        if status == "conflict":
            conflicts.append({"id": uid("conflict", state["id"]), "time": time,
                              "entity": entity, "field": field, "evidence_ids": state["evidence_ids"],
                              "candidates": [r["value"] for r in candidates]})
        states.append(state)
    return states, conflicts


def fuse_events(events: list[dict]) -> list[dict]:
    fused = []
    for event in sorted(events, key=lambda e: (e["start"], e["id"])):
        if event.get("review_status") == "rejected":
            continue
        match = None
        for candidate in reversed(fused):
            if event["start"] - candidate["end"] > 3:
                continue
            same = all(candidate.get(k) == event.get(k) for k in ("kind", "actor", "target", "object"))
            shared = bool(set(candidate["evidence_ids"]) & set(event["evidence_ids"]))
            notice = event["kind"] in {"death", "objective_takedown", "tower_destruction"}
            overlapping = event["start"] <= candidate["end"] and candidate["start"] <= event["end"]
            # Point events only merge at a common frame; notice display persistence gets 2s.
            close_notice = notice and event["actor"] and abs(event["start"] - candidate["start"]) <= 2
            if same and (shared or overlapping or close_notice):
                match = candidate
                break
        if match:
            if match.get("location") and event.get("location") and match["location"] != event["location"]:
                match["location"] = None
                match.setdefault("ambiguity_flags", []).append("conflicting_location")
            elif not match.get("location") and "conflicting_location" not in match.get("ambiguity_flags", []):
                match["location"] = event.get("location")
            for key in ("evidence_ids", "routes", "task_ids", "criteria"):
                match[key] = sorted(set(match[key]) | set(event[key]))
            if event["kind"] in {"death", "objective_takedown", "tower_destruction"}:
                # A persistent notice must not lengthen the underlying point event.
                if event["end"] - event["start"] < match["end"] - match["start"]:
                    match["start"], match["end"] = event["start"], event["end"]
            else:
                # Overlapping clips can expose different portions of one duration event.
                match["start"] = min(match["start"], event["start"])
                match["end"] = max(match["end"], event["end"])
            match["confidence"] = max(match["confidence"], event["confidence"])
            match["sampling_fps"] = max(match["sampling_fps"], event["sampling_fps"])
        else:
            fused.append({**event, **{k: list(event[k]) for k in ("evidence_ids", "routes", "task_ids", "criteria")}})
    for event in fused:
        event["id"] = uid("fused_event", event["kind"], event.get("actor"), event.get("target"),
                          event.get("object"), sorted(event["evidence_ids"]))
        event["precision"] = "sample_bounded"  # FPS does not establish exact semantic onset.
    return fused


def objective_episodes(events: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for event in events:
        if event.get("object") and event["kind"] in {"objective_contest", "objective_takedown", "tower_destruction"}:
            groups[event["object"]].append(event)
        elif event["kind"] in {"recall", "reset"} and event.get("actor"):
            groups["base_reset:" + event["actor"]].append(event)
        elif event.get("location") and event["kind"] in {"encounter", "skirmish", "teamfight"}:
            groups["lane:" + event["location"]].append(event)
    episodes = []
    for objective, group in groups.items():
        current = []
        for event in sorted(group, key=lambda e: e["start"]):
            if current and event["start"] - current[-1]["end"] > 90:
                episodes.append(_episode(objective, current))
                current = []
            current.append(event)
        if current:
            episodes.append(_episode(objective, current))
    return episodes


def _episode(objective, events):
    return {"id": uid("episode", objective, [e["id"] for e in events]), "objective": objective,
            "start": min(e["start"] for e in events), "end": max(e["end"] for e in events),
            "event_ids": [e["id"] for e in events],
            "evidence_ids": sorted({i for e in events for i in e["evidence_ids"]}),
            "interpretation": "Temporal association of visible events; no causal or intent claim"}


@consistent_review
def fuse(store: Store, config) -> dict:
    observations = store.effective(store.rows("observations"), "observations")
    states, conflicts = fuse_observations(observations)
    events = fuse_events(store.rows("event_candidates"))
    events = store.effective(events, "events")
    store.write("states", states)
    store.write("conflicts", conflicts)
    store.write("events", events)
    store.write("episodes", objective_episodes([e for e in events if e.get("review_status") != "rejected"]))
    store.commit_stage("fuse", store.stage_key("fuse", config, ["observations", "event_candidates"]),
                       states=len(states), events=len(events), conflicts=len(conflicts))
    return {"states": len(states), "events": len(events), "conflicts": len(conflicts)}
