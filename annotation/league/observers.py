"""Staged fixed routing and bounded automatic rereads."""
from __future__ import annotations

import random
import logging
from bisect import bisect_left
from dataclasses import replace
from pathlib import Path

from .core import Store, atomic_json, digest, now, read_json, uid
from .inference import InputBudgetError, infer_with_knowledge
from .media import Media, windows
from .prompts import observer_prompt
from .timeline import fuse_observations, normalize_output


def _clock_seconds(value):
    if isinstance(value, dict):
        value = value.get("displayed_text", value.get("seconds"))
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    if isinstance(value, str):
        parts = value.split(":")
        if 2 <= len(parts) <= 3 and all(part.isdigit() for part in parts):
            result = 0
            for part in parts:
                result = result*60+int(part)
            return result
    return None


def transition_reason(previous, row):
    old, new = previous["value"], row["value"]
    if digest(old) == digest(new):
        return None
    field = row["field"]
    if field in {"cs", "displayed_gold"}:
        return None  # Routine displayed-counter updates are already sampled.
    if field in {"game_clock", "objective_timer", "death_timer"}:
        a, b = _clock_seconds(old), _clock_seconds(new)
        if a is not None and b is not None:
            direction = 1 if field == "game_clock" else -1
            if abs((b-a)-direction*(row["time"]-previous["time"])) <= 2:
                return None
            return "clock_discontinuity" if field == "game_clock" else "timer_transition"
    if field == "position" and isinstance(old, dict) and isinstance(new, dict):
        if old.get("region") == new.get("region"):
            return None
    if isinstance(old, dict) and isinstance(new, dict) and old.get("state") == new.get("state") == "cooling_down":
        if isinstance(old.get("seconds"), (int, float)) and isinstance(new.get("seconds"), (int, float)):
            if 0 < new["seconds"] <= old["seconds"]:
                return None
    return "state_transition"


def task_key(run_key, route, start, end, pass_number, fps):
    return uid("task", run_key, route.name, start, end, pass_number, fps)


def task_results(store: Store, run_key: str) -> list[dict]:
    return [item for path in sorted(store.path("tasks").glob("*.json"))
            if (item := read_json(path)).get("run_key") == run_key]


def collect(store: Store, results: list[dict]) -> None:
    for field, artifact in (("observations", "observations"), ("events", "event_candidates"), ("issues", "issues")):
        store.write(artifact, [r for task in results for r in task.get(field, [])])
    store.write("coverage", [{"id": t["id"], "route": t["route"], "start": t["start"], "end": t["end"],
                              "fps": t["fps"], "pass": t["pass"], "status": t["status"],
                              "frame_ids": [f["id"] for f in t.get("frames", [])],
                              "times": [f["time"] for f in t.get("frames", [])],
                              "issues": t.get("issues", [])} for t in results])


def _spent(task_dir: Path) -> dict:
    # Include unsuccessful lookup rounds and interrupted tasks in the cost ledger.
    metrics = [read_json(p)["metrics"] for p in task_dir.rglob("completion.json")]
    return {**{key: sum(m.get(key, 0) for m in metrics)
               for key in ("visual_tokens", "input_tokens", "output_tokens", "wall_seconds")},
            "peak_gpu_memory_bytes": max((m.get("peak_gpu_memory_bytes", 0) for m in metrics), default=0)}


def annotate(store: Store, config, runner, resolver, *, retry_failed=False) -> dict:
    media = Media(store, config)
    run_key = store.fingerprint(config)
    start, end = media.manifest["interval"]
    intervals = list(windows(start, end, config.window_seconds, config.overlap_seconds))
    results = {r["id"]: r for r in task_results(store, run_key)}
    from .knowledge import identity_key
    roster = resolver.roster()["champions"]
    identities = {identity_key(str(c[field])): c["internal_id"]
                  for c in roster for field in ("internal_id", "numeric_id", "name")}

    def execute(route, begin, finish, pass_number, fps, limit=None):
        store.check_stop()
        task_id = task_key(run_key, route, begin, finish, pass_number, fps)
        if task_id in results and (results[task_id]["status"] == "ok" or not retry_failed):
            return results[task_id]
        task = {"id": task_id, "run_key": run_key, "route": route.name, "start": begin,
                "end": finish, "pass": pass_number, "fps": fps, "model": config.models[route.model],
                "device": media.manifest.get("inference_runtime", {}).get("device"),
                "runtime": media.manifest.get("inference_runtime", {}),
                "created_at": now(), "observations": [], "events": [], "issues": []}
        directory = store.path("inference/" + task_id)
        try:
            task["frames"] = media.frames(begin, finish, fps, route)
            chunk_size = config.hud_frames_per_request if route.model == "instruct" else len(task["frames"])
            for index, offset in enumerate(range(0, len(task["frames"]), chunk_size)):
                store.check_stop()
                frames = task["frames"][offset:offset+chunk_size]
                segment_path = directory / f"segment-{index:04d}.json"
                if segment_path.exists():
                    segment = read_json(segment_path)
                else:
                    messages = [{"role": "user", "content": observer_prompt(route, frames, fps)}]
                    segment_dir = directory / f"segment-{index:04d}"
                    attempt = len(list(segment_dir.glob("attempt-*")))
                    remaining = None if limit is None else limit - _spent(directory).get("visual_tokens", 0)
                    reader = replace(route, max_pixels=route.max_pixels * 2 ** pass_number)
                    answer = infer_with_knowledge(runner, resolver, messages, frames, reader,
                        segment_dir / f"attempt-{attempt:04d}", max_queries=config.knowledge_queries, fps=fps,
                        visual_limit=remaining)
                    if answer["status"] != "answered":
                        raise ValueError("Static lookup budget exhausted before a visual answer")
                    observations, events, issues = normalize_output(answer["answer"], {**task, "frames": frames}, route, identities)
                    segment = {"observations": observations, "events": events, "issues": issues,
                               "knowledge_queries": answer["knowledge_queries"]}
                    atomic_json(segment_path, segment)
                for key in ("observations", "events", "issues", "knowledge_queries"):
                    task.setdefault(key, []).extend(segment[key])
            task.update(status="ok", metrics=_spent(directory))
            if any(i["kind"] in {"invalid_output", "unreadable", "low_confidence", "identity_conflict", "layout_mismatch"} for i in task["issues"]):
                task["status"] = "partial"
        except (ValueError, InputBudgetError) as exc:
            task.update(status="unresolved", error=str(exc), metrics=_spent(directory))
            task["issues"].append({"kind": "unresolved", "route": route.name, "time": begin, "detail": str(exc)})
        atomic_json(store.path(f"tasks/{task_id}.json"), task)
        results[task_id] = task
        logging.getLogger(__name__).info("route=%s window=%.3f:%.3f pass=%s status=%s tokens=%s",
            route.name, begin, finish, pass_number, task["status"], task.get("metrics", {}).get("visual_tokens", 0))
        return task

    # Technical calibration precedes the full run and does not require human review.
    pilot_intervals = sorted(set(min(intervals, key=lambda v: abs(v[0]-target))
                                 for target in (24, 780, 1620, 300)))
    pilot = []
    for role in ("instruct", "thinking"):
        for route in (r for r in config.routes if r.model == role):
            for begin, finish in pilot_intervals:
                pilot.append(execute(route, begin, finish, 0, config.fps))
    collect(store, list(results.values()))
    incompatible = []
    for route in config.routes:
        samples = [t for t in pilot if t["route"] == route.name]
        if all(t["status"] == "unresolved" for t in samples) or all(
                any(i["kind"] == "layout_mismatch" for i in t.get("issues", [])) for t in samples):
            incompatible.append(route.name)
    atomic_json(store.path("pilot_report.json"), {"intervals": pilot_intervals, "tasks": len(pilot),
        "incompatible_routes": incompatible, "human_review_required": False,
        "partial_or_unresolved": sum(t["status"] != "ok" for t in pilot),
        "meaning": "Technical/layout calibration only; semantic accuracy requires reviewed references"})
    if incompatible:
        raise ValueError(f"Pilot failed on fixed routes {incompatible}; inspect outputs, adjust configuration, then resume")
    # Keep one resident model and finish all initial tasks for a role before switching.
    for role in ("instruct", "thinking"):
        for route in (r for r in config.routes if r.model == role):
            for begin, finish in intervals:
                execute(route, begin, finish, 0, config.fps)
    collect(store, list(results.values()))
    first_cost = sum(t.get("metrics", {}).get("visual_tokens", 0) for t in results.values() if t["pass"] == 0)
    allowance = int(first_cost * config.review_budget_fraction)
    role_allowances = {role: int(sum(t.get("metrics", {}).get("visual_tokens", 0) for t in results.values()
                                   if t["pass"] == 0 and next(r.model for r in config.routes if r.name == t["route"]) == role)
                               * config.review_budget_fraction) for role in ("instruct", "thinking")}
    requests = make_review_requests(store, config, intervals)
    store.write("review_requests", requests)
    route_map = {r.name: r for r in config.routes}
    for round_index, fps in enumerate(config.review_fps, 1):
        current_conflicts = fuse_observations([o for t in results.values() for o in t.get("observations", [])])[1] if round_index == 2 else []
        for role in ("instruct", "thinking"):
            for request in requests:
                route = route_map[request["route"]]
                if route.model != role:
                    continue
                # Second round is reserved for unresolved/disagreeing first reviews.
                if round_index == 2:
                    prior_id = task_key(run_key, route, *request["interval"], 1, config.review_fps[0])
                    prior = results.get(prior_id, {})
                    disagreement = any(request["interval"][0] <= c["time"] <= request["interval"][1] for c in current_conflicts)
                    unreadable = any(o["visibility"] in {"unreadable", "unknown"} for o in prior.get("observations", []))
                    if prior.get("status") == "ok" and not prior.get("issues") and not disagreement and not unreadable:
                        continue
                used = sum(t.get("metrics", {}).get("visual_tokens", 0) for t in results.values() if t["pass"] > 0)
                role_used = sum(t.get("metrics", {}).get("visual_tokens", 0) for t in results.values()
                                if t["pass"] > 0 and route_map[t["route"]].model == role)
                remaining = min(allowance-used, role_allowances[role]-role_used)
                key = task_key(run_key, route, *request["interval"], round_index, fps)
                if remaining <= 0 and key not in results:
                    request.setdefault("outcomes", []).append({"round": round_index, "status": "budget_exhausted"})
                    continue
                result = execute(route, *request["interval"], round_index, fps, max(0, remaining))
                request.setdefault("outcomes", []).append({"round": round_index, "task_id": result["id"],
                                                          "status": result["status"]})
    collect(store, list(results.values()))
    store.write("review_requests", requests)
    store.write("human_review_queue", [r for r in requests if any(
        o["status"] != "ok" for o in r.get("outcomes", []))])
    audited_routes = {t["route"] for t in results.values() if t["status"] in {"ok", "partial"}}
    validation = read_json(store.path("layout_validation.json"))
    validation["model_preprocessing_validated"] = len(audited_routes) == len(config.routes)
    validation["automated_layout_issues"] = [i for t in results.values() for i in t.get("issues", [])
                                           if i["kind"] == "layout_mismatch"]
    atomic_json(store.path("layout_validation.json"), validation)
    summary = {"tasks": len(results), "unresolved": sum(t["status"] != "ok" for t in results.values()),
               "first_pass_visual_tokens": first_cost, "review_token_allowance": allowance,
               "review_visual_tokens": sum(t.get("metrics", {}).get("visual_tokens", 0)
                                           for t in results.values() if t["pass"] > 0)}
    store.commit_stage("annotate", run_key, **summary)
    return summary


def make_review_requests(store: Store, config, intervals):
    observations, events, issues = store.rows("observations"), store.rows("event_candidates"), store.rows("issues")
    triggers = []
    previous = {}
    world_routes = [r.name for r in config.routes if r.model == "thinking"]
    for row in sorted(observations, key=lambda o: o["time"]):
        key = row["entity"], row["field"], row["route"]
        if row["visibility"] in {"unreadable", "unknown"}:
            triggers.append((row["time"], row["route"], "unreadable"))
        elif key in previous:
            reason = transition_reason(previous[key], row)
            if reason:
                triggers.append((row["time"], row["route"], reason))
                if row["field"] in {"health", "alive", "kills", "deaths", "inventory", "Q", "W", "E", "R", "summoner_1", "summoner_2"}:
                    triggers.extend((row["time"], name, "hud_event_context") for name in world_routes)
        elif row["field"] == "notice" and row["value"]:
            triggers.append((row["time"], row["route"], "event_notice"))
        previous[key] = row
    for event in events:
        triggers.extend((event["start"], route, "event") for route in event["routes"])
    _, conflicts = fuse_observations(observations)
    by_id = {o["id"]: o for o in observations}
    for conflict in conflicts:
        triggers.extend((conflict["time"], by_id[i]["route"], "cross_route_disagreement")
                        for i in conflict["evidence_ids"])
    for issue in issues:
        triggers.append((issue.get("time", intervals[0][0]), issue["route"], issue["kind"]))
    # Quantize into shared discovery windows: overlapping events never create duplicate jobs.
    dedup = {}
    lower, upper = intervals[0][0], intervals[-1][1]
    centers = [(a+b)/2 for a, b in intervals]
    for time, route, reason in triggers:
        index = bisect_left(centers, time)
        nearby = intervals[max(0, index-1):min(len(intervals), index+1)]
        begin, finish = min(nearby, key=lambda v: abs((v[0]+v[1])/2-time))
        interval = (max(lower, begin-2), min(upper, finish+2))
        key = route, interval
        dedup.setdefault(key, {"route": route, "interval": list(interval), "reasons": []})["reasons"].append(reason)
    rng = random.Random(config.seed)
    for interval in intervals:
        if rng.random() < config.quiet_audit_fraction:
            for route in config.routes:
                key = route.name, interval
                dedup.setdefault(key, {"route": route.name, "interval": list(interval), "reasons": []})["reasons"].append("quiet_audit")
    ranked = [{"id": uid("review", r["route"], r["interval"]), **r,
               "reasons": sorted(set(r["reasons"]))} for r in dedup.values()]
    critical = {"cross_route_disagreement", "identity_conflict", "layout_mismatch", "unreadable", "low_confidence", "invalid_output"}
    return sorted(ranked, key=lambda r: (0 if critical & set(r["reasons"]) else 1 if "event" in r["reasons"] else 2,
                                         r["interval"][0], r["route"]))
