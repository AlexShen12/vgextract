"""Resumable caption pilot; never writes production observations/events/questions."""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import fcntl
import logging
import os
from pathlib import Path
import resource
import sys
import time

from .action_body import bind_frames, reconcile_tracks
from .action_events import counter_events, read_counters
from .action_icons import AbilityIdentity
from .action_pipeline import native_review, prepare_inputs
from .action_v3 import load_profile
from .caption_scene import CaptionSceneReader, accepted_events, visible_precursor
from .caption_segments import POLICY, context_key, deduplicate_events, link_repeats, segment_captions
from .captions import construct_captions
from .core import atomic_json, digest, implementation_hash, read_json, read_rows, uid, write_rows
from .local_models import LOCK, LocalVLRunner
from .timeline import fuse_observations


COMPONENTS = ["caption_pilot.py", "caption_scene.py", "caption_segments.py", "captions.py", "playback.py",
              "action_pipeline.py", "action_body.py", "action_icons.py", "action_events.py", "action_v3.py",
              "action_state.py", "timeline.py", "core.py", "readers.py", "numeric_reader.py", "local_models.py", "media.py"]


@contextmanager
def pilot_lock(root):
    with (Path(root)/".pilot.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Caption pilot is running; retry review after it finishes") from exc
        yield


def effective(root, table):
    rows = {r["id"]: r for r in read_rows(Path(root)/(table+".jsonl"))}
    for path in sorted((Path(root)/"reviews").glob("*.json")):
        correction = read_json(path)
        if correction["table"] == table and correction["record_id"] in rows:
            rows[correction["record_id"]].update(correction["changes"], review_status=correction["review_status"])
    return list(rows.values())


def counter_observations(counters, frames, *, body=()):
    """Require neighboring stable readings, retaining every unknown as a barrier."""
    known = {f["id"]: f for f in frames}
    bodies = {r["frame_id"]: r for r in body if r["owner_id"] == "selected"}
    groups = {}
    for row in counters:
        if row["field"] == "cs" and row["owner_id"] == "selected":
            groups.setdefault((row["segment_id"], row["owner_epoch"]), []).append(row)
    observations = []
    for rows in groups.values():
        rows.sort(key=lambda r: r["time"])
        for i, row in enumerate(rows):
            frame = known[row["frame_id"]]
            usable = (row["value"] is not None and row.get("layout_valid") and not row.get("identity_changed")
                      and frame.get("playback", {}).get("playback_state") == "playing")
            neighbors = [rows[j] for j in (i-1, i+1) if 0 <= j < len(rows)
                         and abs(rows[j]["time"]-row["time"]) <= .095 and rows[j]["value"] == row["value"]
                         and rows[j].get("layout_valid") and not rows[j].get("identity_changed")]
            usable = bool(usable and neighbors)
            context = uid("owner", row["segment_id"], row["owner_epoch"])
            bound = bodies.get(row["frame_id"], {})
            body_known = bound.get("binding_visible") and bound.get("review_status") != "rejected"
            identity_scope = "hud_icons" if row.get("champion") else "body_binding" if body_known else "hud_slot"
            observations.append({"schema_version": 2, "id": uid("caption-counter", row["id"]),
                "frame_id": row["frame_id"], "time": row["time"], "field": "cs",
                "entity": "unknown" if identity_scope == "hud_slot" else "player:"+context,
                "identity_scope": identity_scope,
                "slot_id": "selected", "binding": context, "value": row["value"] if usable else None,
                "visibility": "visible" if usable else "unknown", "acceptance": "accepted" if usable else "unknown",
                "playback_segment_id": uid("caption-owner-segment", context),
                "continuity_verified": frame.get("playback", {}).get("continuity_verified", False),
                "evidence_modality": "visual", "experimental": True, "caption_pilot": True,
                "derived_from": [row["id"]]+[n["id"] for n in neighbors]+([bound["id"]] if identity_scope == "body_binding" else []), "review_status": "unreviewed"})
    return observations


def regenerate(root):
    """Called under the pilot lock; upstream corrections are a derived-stage key."""
    root = Path(root)
    audit = read_json(root/"input-audit.json")
    policy = audit["policy"]
    frames = audit["frames"]
    body = effective(root, "body_readings")
    events, omissions = accepted_events(effective(root, "scene_proposals"), frames, body)
    originals = events
    events, duplicates = deduplicate_events(events)
    conflicts = {r["record_id"] for r in duplicates}
    weak = [visible_precursor(e, {f["id"]: f["time"] for f in frames}) for e in originals if e["id"] in conflicts]
    if weak:
        retained, _ = accepted_events([e for e in weak if e], frames, body)
        events, _ = deduplicate_events(events+retained)
    omissions += duplicates
    # Duplicate proposal IDs stay in the lineage, while emitted events are atomic.
    write_rows(root/"events.jsonl", events)
    observations = effective(root, "normalized_observations")
    bodies = {r["frame_id"]: r for r in body}
    for observation in observations:
        bound = bodies.get(observation["frame_id"], {})
        if observation.get("identity_scope") == "body_binding" and (not bound.get("binding_visible") or bound.get("review_status") == "rejected"):
            observation.update(entity="unknown", continuity_verified=False)
    states, issues = fuse_observations(observations, caption_pilot=True)
    for state in states:
        state.update(experimental=True, caption_pilot=True,
                     display_label="HUD slot selected" if state["entity"] == "unknown" else "The selected champion")
    context = {f["id"]: f for f in frames}
    captions, renderer_omissions = construct_captions(states, events, observations,
        interval=audit["interval"], source_sha256=audit["source_sha256"], frame_times={k: f["time"] for k, f in context.items()},
        issues=issues, pilot=True, frame_context=context,
        body={r["frame_id"]: r for r in body if r["owner_id"] == "selected"})
    for caption in captions:
        if "owner_epoch" not in caption:
            refs = [r["id"] for r in caption["evidence_refs"] if r["table"] == "frames"]
            first = min((context[f] for f in refs), key=lambda f: f["time"])
            caption.update(playback_segment_id=first["segment_id"], owner_id="selected", owner_epoch=first["owner_epochs"]["selected"])
    # Repeated point descriptions do not acquire an interval. In particular the
    # first, not-yet-comparable clock sample need not duplicate the next reading.
    distinct, displayed = [], {}
    for caption in captions:
        claim = caption["claims"][0]
        key = (caption["playback_segment_id"], caption["owner_epoch"], caption["actor_bindings"][0]["entity"], claim["values"].get("field"))
        if claim["kind"] == "displayed_state":
            old = displayed.get(key)
            if (old and old["claims"][0]["values"] == claim["values"]
                    and caption["start"]-old["end"] <= policy["merge_gap_seconds"]):
                renderer_omissions.append({"record_id": next(r["id"] for r in caption["evidence_refs"] if r["table"] == "states"),
                                          "table": "states", "reason": "duplicate_display_reading"})
                continue
            displayed[key] = caption
        elif claim["kind"] == "cs_increased":
            displayed.pop(key, None)
        distinct.append(caption)
    captions = distinct
    clips = segment_captions(captions, frames, audit["interval"], audit["source_sha256"], policy)
    for clip in clips:
        clip["pilot_id"] = root.name
    # Source-time occurrences may live in separate pilot neighborhoods. Compare
    # compatible prior artifacts in memory; never rewrite their claims or clocks.
    prior_clips, prior_frames = [], []
    for path in sorted(root.parent.glob("*/input-audit.json")):
        if path.parent == root or not (path.parent/"caption_clips.jsonl").exists():
            continue
        previous = read_json(path)
        if (previous["source_sha256"] == audit["source_sha256"] and previous["profile"] == audit["profile"]
                and previous.get("masks") == audit.get("masks") and previous["policy"] == policy):
            prior_frames += previous["frames"]
            prior_clips += [{**c, "pilot_id": path.parent.name} for c in read_rows(path.parent/"caption_clips.jsonl") if c["end"] <= audit["interval"][0]]
    combined = link_repeats(sorted(prior_clips+clips, key=lambda c: (c["start"], c["id"])), prior_frames+frames, policy)
    clip_index = {c["id"]: c for c in combined}
    clips = [c for c in combined if c["pilot_id"] == root.name]
    for clip in clips:
        earlier = clip_index.get(clip.get("earlier_occurrence_id"))
        clip["earlier_pilot_id"] = earlier["pilot_id"] if earlier else None
    for name, rows in (("states", states), ("captions", captions), ("caption_clips", clips),
                       ("caption_omissions", omissions+renderer_omissions)):
        write_rows(root/(name+".jsonl"), rows)
    summary = {"schema_version": 1, "experimental": True, "generated": len(captions), "clips": len(clips),
        "counts": dict(Counter(c["claims"][0]["kind"] for c in captions)),
        "omitted": dict(Counter(o["reason"] for o in omissions+renderer_omissions)),
        "quality_metrics": None, "acceptance_status": "independent_validation_pending", "vlm_rendering_tokens": 0,
        "key": digest({"audit": audit, "events": events, "observations": observations, "body": body, "clips": clips,
                       "corrections": [read_json(p) for p in sorted((root/"reviews").glob("*.json"))],
                       "implementation": implementation_hash(COMPONENTS)})}
    atomic_json(root/"caption_coverage.json", summary)
    if (root/"report.json").exists():
        report = read_json(root/"report.json")
        report["coverage"] = summary
        atomic_json(root/"report.json", report)
    return summary


def dense_windows(proposals, frames, policy=POLICY):
    """Coalesce candidate context while honoring every owner/playback boundary."""
    windows = []
    for row in sorted(proposals, key=lambda r: r["start"]):
        key = (row["playback_segment_id"], row["owner_id"], row["owner_epoch"])
        available = [f for f in frames if context_key(f) == key]
        if not available:
            continue
        margin = 1 if row.get("kind") == "creep_credit" else .25
        a = max(available[0]["time"], row["start"]-margin)
        b = min(available[-1]["time"]+1/policy["hud_fps"], row["end"]+margin)
        old = next((w for w in windows if w["context"] == key and a <= w["end"] and b >= w["start"]), None)
        if old:
            old.update(start=min(a, old["start"]), end=max(b, old["end"]))
        else:
            windows.append({"context": key, "segment_id": key[0], "start": a, "end": b})
    return windows


def _cpu_detection(store, config, root, audit, icon):
    from PIL import Image
    frames, profile = [f for f in audit["frames"] if not f.get("native_review")], audit["profile"]
    path = root/"cpu-complete.json"
    if path.exists() and all((root/(name+".jsonl")).exists() for name in
                             ("identities", "body_readings", "bindings", "counter_readings", "normalized_observations")):
        return read_rows(root/"identities.jsonl"), read_rows(root/"body_readings.jsonl")
    identities = icon.read_frames(frames, profile, "core") if icon else []
    indexed = {r["frame_id"]: r for r in identities}
    # Icon recognition deliberately skips unsupported layouts. Keep explicit
    # unknown rows so those frames remain continuity barriers, not stage errors.
    identities = [indexed.get(f["id"], {"id": uid("unknown-kit", f["id"]), "frame_id": f["id"],
        "time": f["time"], "owner_id": "selected", "owner_epoch": f["owner_epochs"]["selected"],
        "segment_id": f["segment_id"], "hud_champion": None,
        "reason": "unsupported_layout" if not f["layout_valid"] else "icon_identity_unavailable"}) for f in frames]
    # A supported kit change is also a selection boundary, including future rows.
    by_frame, last_original, last_name, epoch = {r["frame_id"]: r for r in identities}, None, None, -1
    for f in frames:
        original = (f["segment_id"], f["owner_epochs"]["selected"])
        name = by_frame[f["id"]].get("hud_champion")
        if original != last_original or (last_name and name and last_name != name):
            epoch += 1
            last_name = None
        if name:
            last_name = name
        last_original = original
        f["owner_epochs"]["selected"] = epoch
        by_frame[f["id"]]["owner_epoch"] = epoch
    atomic_json(root/"input-audit.json", audit)
    body, bindings = bind_frames(frames, identities, profile, store.check_stop)
    for name, rows in (("identities", identities), ("body_readings", body), ("bindings", bindings)):
        write_rows(root/(name+".jsonl"), rows)
    counters = []
    for f in frames:
        store.check_stop()
        with Image.open(f["native_path"]) as image:
            counters += read_counters(image, f, profile, root, by_frame[f["id"]])
    write_rows(root/"counter_readings.jsonl", counters)
    write_rows(root/"normalized_observations.jsonl", counter_observations(counters, frames, body=body))
    atomic_json(path, {"frames": len(frames)})
    return identities, body


def caption_pilot(store, config, start, end, *, infer=False, action_profile=None, icon_references=None, runner=None):
    started = time.monotonic()
    profile = load_profile(action_profile)
    if [o["id"] for o in profile["owners"]] != ["selected"]:
        raise ValueError("Caption pilot requires a single selected HUD owner")
    reference = Path(icon_references) if icon_references else Path(config.asset_cache)/"manifest.json"
    if icon_references and not reference.exists():
        raise ValueError("Ability icon references do not exist")
    icon = AbilityIdentity(reference) if reference.exists() else None
    audit = prepare_inputs(store, config, start, end, profile, POLICY, namespace="captions", max_duration=None,
        cache_identity={"implementation": implementation_hash(COMPONENTS), "model": read_json(LOCK)["models"]["intake"],
                        "icons": icon.signature if icon else None})
    root = Path(audit["root"])
    with pilot_lock(root):
        return _run(store, config, audit, root, icon, infer, runner, started)


def _run(store, config, audit, root, icon, infer, runner, started):
    report = {"schema_version": 1, "id": root.name, "root": str(root), "report": str(root/"report.json"),
              "interval": audit["interval"], "status": "running_experimental", "experimental": True,
              "production_evidence_modified": False, "quality_metrics": None, "infer": infer, "policy": POLICY,
              "icon_reference_version": icon.version if icon else None, "requests": []}
    atomic_json(root/"report.json", report)
    owned, reader = runner is None, None
    proposals = read_rows(root/"scene_proposals.jsonl")
    try:
        identities, body = _cpu_detection(store, config, root, audit, icon)
        body = effective(root, "body_readings")
        logging.info("Caption pilot CPU detection complete: %d frames, %d visible bindings", len(audit["frames"]), sum(r["binding_visible"] for r in body))
        report["coverage"] = regenerate(root)
        if infer and runner is None:
            runner = LocalVLRunner(store, config.vl_python or os.environ.get("LEAGUE_VL_PYTHON", ""), config.asset_cache, image_limit=8)
        reader = CaptionSceneReader(store, root, audit["profile"], POLICY, body, identities, runner if infer else None)
        def inspect(frames, stage):
            try:
                return reader.read(frames, stage)
            finally:
                # Parsed packets survive a later GPU failure or cancellation.
                proposals[:] = list({p["id"]: p for p in reader.proposals}.values())
                write_rows(root/"scene_proposals.jsonl", proposals)
        base = [f for f in audit["frames"] if not f.get("native_review")]
        coarse = inspect([f for f in base if f["scene_sample"] and f["layout_valid"]], "4fps")
        credits = [{**c, "playback_segment_id": c["segment_id"]} for c in counter_events(read_rows(root/"counter_readings.jsonl")) if c["kind"] == "creep_credit"]
        windows = dense_windows(coarse+credits, base)
        resampling = []
        for window in windows:
            dense = [f for f in base if window["start"] <= f["time"] < window["end"] and context_key(f) == window["context"]]
            parsed = inspect(dense, "16fps")
            resampling.append({**window, "stage": "16fps", "frame_ids": [f["id"] for f in dense]})
            confirmed, unresolved = accepted_events(parsed, audit["frames"], body)
            # Every incomplete dense impact chain remains eligible for a native bracket.
            unresolved_ids = {o["record_id"] for o in unresolved} | {pid for o in unresolved for pid in o.get("source_proposal_ids", [])}
            candidates = [p for p in parsed if p["id"] in unresolved_ids]
            if reader.runtime is not None:
                # Generic movement/attacks do not resolve a possible last-hit
                # bracket. Counter income remains a separate aggregate claim.
                candidates += [c for c in credits if window["start"] <= c["end"] < window["end"]
                               and (c["playback_segment_id"], c["owner_id"], c["owner_epoch"]) == window["context"]
                               and not any(e["kind"] == "minion_last_hit" and e["start"] <= c["end"]+.5
                                           and e["end"] >= c["start"]-.5 for e in confirmed)]
            brackets = []
            for candidate in sorted(candidates, key=lambda p: p["start"]):
                a = max(window["start"], candidate["start"]-POLICY["native_margin"])
                b = min(window["end"], candidate["end"]+POLICY["native_margin"])
                if brackets and a <= brackets[-1]["native_end"]:
                    brackets[-1]["native_end"] = max(b, brackets[-1]["native_end"])
                else:
                    brackets.append({**window, "native_start": a, "native_end": b})
            for native_window in brackets:
                native = native_review(store, audit, native_window, config)
                native_ids = icon.read_frames(native, audit["profile"], "core") if icon else []
                native_body, bindings = bind_frames(native, native_ids, audit["profile"], store.check_stop)
                native_body, bindings = reconcile_tracks(native_body, bindings, body)
                raw_body = read_rows(root/"body_readings.jsonl")
                write_rows(root/"body_readings.jsonl", list({r["frame_id"]: r for r in raw_body+native_body}.values()))
                body = effective(root, "body_readings")
                identities = list({r["frame_id"]: r for r in identities+native_ids}.values())
                write_rows(root/"identities.jsonl", identities)
                reader.body.update({r["frame_id"]: r for r in body})
                reader.identities.update({r["frame_id"]: r for r in native_ids})
                inspect(native, "native")
                resampling.append({**native_window, "stage": "native", "frame_ids": [f["id"] for f in native]})
        write_rows(root/"resampling.jsonl", resampling)
        report.update(status="completed_experimental", scene_uncertainty=reader.uncertainty)
    except BaseException as exc:
        report.update(status="interrupted_experimental" if store.stopping else "failed_experimental", error=str(exc))
        raise
    finally:
        if (root/"normalized_observations.jsonl").exists():
            report["coverage"] = regenerate(root)
        requests = reader.requests if reader else []
        report["requests"] = requests
        report["runtime"] = runner.identity if infer and runner is not None else reader.runtime if reader else None
        report["scene_uncertainty"] = reader.uncertainty if reader else []
        report["resource_metrics"] = {"wall_seconds_this_run": time.monotonic()-started, "requests": len(requests),
            "generated_tokens_this_run": sum(r["usage"].get("completion_tokens", 0) for r in requests if not r["cache_reused"]),
            "prompt_tokens": sum(r["usage"].get("prompt_tokens", 0) for r in requests),
            "prompt_tokens_this_run": sum(r["usage"].get("prompt_tokens", 0) for r in requests if not r["cache_reused"]),
            "cache_reused": sum(r["cache_reused"] for r in requests),
            "truncated_requests": sum(r.get("finish_reason") == "length" for r in requests),
            "invalid_responses": sum(r.get("validation_error") is not None for r in requests),
            "peak_gpu_memory_this_run_bytes": getattr(runner, "peak_memory_bytes", 0) if infer else 0,
            "peak_host_memory_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == "darwin" else 1024)}
        report["resource_metrics"]["host_memory_scope"] = "driver_process_only"
        if owned and infer and runner is not None:
            runner.close()
        atomic_json(root/"report.json", report)
    return report
