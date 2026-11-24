"""Run a bounded champion visibility pilot and derive reviewed timeline events."""

import argparse
from collections import Counter, defaultdict
import fcntl
import json
import math
from pathlib import Path
import time

from .body_frame_matching import (UnresolvedMainTrack, match_bodies, record_assignments,
                                  render_matcher, team_pools, verify_revisions)
from .body_operational import identity_inputs, track_revision, track_samples
from .core import atomic_json, digest, read_json, read_rows, uid, write_rows
from .media import file_hash


FPS = 4
# Allow native frame rounding while treating a skipped sample as a gap.
MAX_SAMPLE_GAP = 1.5 / FPS


def sample_frames(clip):
    buckets = {}
    for frame in clip["frames"]:
        bucket = math.floor((frame["time"] - clip["start"]) * FPS + 1e-6)
        buckets.setdefault(bucket, frame)
    return list(buckets.values())


def presence_key(row):
    return row["clip_id"], row["roster_id"], row["team"], row["champion"]


def candidate_changes(observations):
    previous = {}
    for row in observations:
        key = presence_key(row)
        prior = previous.get(key)
        previous[key] = row
        if (prior and prior["status"] != row["status"] and prior["frame_sha256"] != row["frame_sha256"]
                and 0 < row["time"] - prior["time"] <= MAX_SAMPLE_GAP):
            yield {"champion": row["champion"], "team": row["team"], "clip_id": row["clip_id"],
                   "start": prior["time"], "end": row["time"], "before": prior["status"], "after": row["status"],
                   "presence_observation_ids": [prior["id"], row["id"]], "experimental": True}


def scan(root, intake_run, cache, output, *, clip_ids, main_champion, slot_skim=None, render_gallery=None,
         friendly_side=None, main_tracks=None, check_inputs=False):
    start = time.monotonic()
    root, cache, output = (Path(path).resolve() for path in (root, cache, output))
    prepared, campaign, intake, source, _, observations, profiles, rosters = identity_inputs(
        root, intake_run, cache, slot_skim)
    clips = sorted((clip for clip in prepared["clips"] if clip["id"] in clip_ids and clip["source_sha256"] == source),
                   key=lambda clip: clip["start"])
    if not clips or {clip["id"] for clip in clips} != set(clip_ids):
        raise ValueError("Choose prepared clips from the scoreboard source")
    if any(left["end"] > right["start"] for left, right in zip(clips, clips[1:])):
        raise ValueError("Choose nonoverlapping clips so each frame can vote once")
    matcher, skins = render_matcher(root, cache, prepared, campaign, rosters, render_gallery)
    encoder_lock = read_json(matcher.lock)
    if digest(encoder_lock) != matcher.lock_hash:
        raise ValueError("Body encoder lock changed after preparation")
    if not Path(encoder_lock["code_root"]).is_dir() or not Path(encoder_lock["weights"]).is_file():
        raise ValueError("Stage the pinned DINOv2 checkout and weights")

    frame_plan = {}
    for clip in clips:
        if not clip["jobs"] or not all((root / "tracks" / (job + "-complete.json")).is_file() for job in clip["jobs"]):
            raise ValueError(f"Clip {clip['id']} needs completed body tracking")
        for frame in sample_frames(clip):
            roster = rosters.at(frame["time"])
            if not roster or roster["status"] != "confirmed":
                raise ValueError(f"Clip {clip['id']} needs confirmed scoreboard coverage at {frame['time']}")
            side, pools = team_pools(roster, main_champion, friendly_side)
            missing = {name for pool in pools.values() for name in pool if name not in matcher.entries}
            if missing:
                raise ValueError(f"Audited render coverage is missing for {', '.join(sorted(missing))}")
            frame_plan[frame["id"]] = (clip, frame, roster, side, pools)
    if not frame_plan:
        raise ValueError("Selected clips have no prepared frames")
    roster_members = {name for _, _, _, _, pools in frame_plan.values() for pool in pools.values() for name in pool}
    for name in roster_members:
        for renders in matcher.entries[name].values():
            for entry in renders:
                if not all(Path(entry[field]).is_file() for field in ("path", "mask_path")):
                    raise ValueError(f"Stage the render and mask for {entry['id']}")

    main_review = read_json(main_tracks) if main_tracks else {"source_sha256": source, "frames": {}}
    overrides = main_review["frames"]
    if main_review["source_sha256"] != source or not overrides.keys() <= frame_plan.keys():
        raise ValueError("Main track decisions must use selected frames from this source")
    for identifier, decision in overrides.items():
        if decision["frame_sha256"] != frame_plan[identifier][1]["sha256"]:
            raise ValueError("Main track decision refers to different frame pixels")

    tracking = track_revision(root)
    report = {"experimental": True, "source_sha256": source, "fps": FPS, "main_champion": main_champion,
              "clips": [{"id": clip["id"], "start": clip["start"], "end": clip["end"]} for clip in clips],
              "sampled_frames": len(frame_plan), "track_revision": tracking,
              "prepared_sha256": digest(prepared), "gallery_sha256": digest(matcher.entries),
              "main_tracks_sha256": digest(main_review), "skin_pool": skins}
    if check_inputs:
        return report
    if (output / "report.json").exists():
        raise ValueError("Choose a fresh output directory for the pilot")
    frames, presence = [], []
    counts = Counter()
    assignments = defaultdict(lambda: {"friendly": {}, "enemy": {}})
    for clip in clips:
        plan = [frame_plan[frame["id"]] for frame in sample_frames(clip)]
        samples = defaultdict(list)
        for row in track_samples(root, {**prepared, "clips": [clip]}, source, frame_ids={frame["id"] for _, frame, *_ in plan}):
            samples[row["frame_id"]].append(row)
        for _, frame, roster, side, pools in plan:
            rows = samples[frame["id"]]
            if not rows and file_hash(Path(frame["path"])) != frame["sha256"]:
                raise ValueError("Prepared frame pixels changed")
            scope = roster["id"], side
            assigned = assignments[scope]
            decision = overrides.get(frame["id"])
            try:
                predictions, excluded = match_bodies(
                    rows, pools, matcher, assigned, main_champion, main_track=decision["track_id"] if decision else None,
                    main_offscreen=bool(decision and decision["track_id"] is None))
                main_status = "excluded" if excluded else "offscreen_reviewed" if decision else "no_friendly_track"
            except UnresolvedMainTrack:
                if decision:
                    raise
                predictions, excluded = match_bodies(
                    [row for row in rows if row["team"] == "enemy"], pools, matcher, assigned, main_champion)
                main_status = "unresolved"
            counts["main_unresolved_frames"] += main_status == "unresolved"
            counts["frames_with_tracks"] += bool(rows)
            counts["matched_bodies"] += sum(bool(row["champion"]) for row in predictions)
            counts["render_comparisons"] += sum(row["render_comparisons"] for row in predictions)
            counts["reused_assignments"] += sum(row["used_assigned_skin"] for row in predictions)
            counts["new_assignments"] += record_assignments(assigned, predictions)
            base = {"frame_id": frame["id"], "frame_sha256": frame["sha256"], "time": frame["time"],
                    "source_sha256": source, "clip_id": clip["id"], "roster_id": roster["id"],
                    "playback_segment_id": roster["segment_id"] if slot_skim is None else None}
            frames.append({**base, "frame_path": frame["path"], "main_status": main_status,
                           "excluded_main_track": excluded, "predictions": predictions})
            visible = {(row["team"], row["champion"]): row for row in predictions if row["champion"]}
            for team, pool in pools.items():
                for champion in pool:
                    row = visible.get((team, champion))
                    presence.append({**base, "id": uid("champion-presence", base, team, champion),
                                     "champion": champion, "team": team, "status": "candidate_visible" if row else "unknown",
                                     "match_id": row["id"] if row else None, "experimental": True})

    verify_revisions(root, tracking, intake, observations, profiles, rosters, slot_skim)
    changes = list(candidate_changes(presence))
    counts["candidate_visibility_changes"] = len(changes)
    outputs = {"frame_matches.jsonl": frames, "presence_observations.jsonl": presence,
               "presence_change_candidates.jsonl": changes}
    report.update(counts=dict(counts), encodes=dict(matcher.encoded), seconds=time.monotonic() - start,
                  encoder_signature=matcher.encoder.signature if "encoder" in matcher.__dict__ else None,
                  assignments=[{"roster_id": roster, "friendly_side": side, "teams": teams}
                               for (roster, side), teams in assignments.items()])
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (output / "report.json").exists():
            raise ValueError("A concurrent pilot has already published to this output directory")
        for name, rows in outputs.items():
            write_rows(output / name, rows)
        report["files"] = {name: file_hash(output / name) for name in outputs}
        atomic_json(output / "report.json", report)
    return {"report": str(output / "report.json"), **report["counts"]}


def reviewed_transitions(observations, review, source):
    """Bracket visibility changes with consecutive, explicitly reviewed source frames."""
    if review["source_sha256"] != source or not review["reviewer"].strip():
        raise ValueError("Visibility review needs a reviewer and matching source")
    known = {row["id"]: row for row in observations}
    decisions = {}
    for item in review["observations"]:
        identifier = item["observation_id"]
        row = known.get(identifier)
        if (not row or row["source_sha256"] != source or identifier in decisions
                or item["frame_sha256"] != row["frame_sha256"] or item["status"] not in {"visible", "not_visible", "unknown"}):
            raise ValueError("Visibility decisions must cite distinct, unchanged source observations")
        decisions[identifier] = item["status"]

    events, previous = [], {}
    for row in sorted(observations, key=lambda item: item["time"]):
        key = presence_key(row)
        status = decisions.get(row["id"], "unknown")
        prior = previous.pop(key, None)
        if status == "unknown":
            continue
        if prior and prior[0]["frame_sha256"] == row["frame_sha256"] and prior[1] != status:
            continue
        previous[key] = row, status
        if not prior:
            continue
        earlier, earlier_status = prior
        if earlier_status == status or not 0 < row["time"] - earlier["time"] <= MAX_SAMPLE_GAP:
            continue
        kind = "champion_appeared" if status == "visible" else "champion_disappeared"
        evidence = [earlier["id"], row["id"]]
        events.append({"id": uid("champion-visibility", source, kind, evidence), "kind": kind, "actor": row["champion"],
                       "team": row["team"], "source_sha256": source, "body_clip_id": row["clip_id"],
                       "playback_segment_id": row.get("playback_segment_id"),
                       "roster_id": row["roster_id"], "start": earlier["time"], "end": row["time"],
                       "evidence_ids": [earlier["frame_id"], row["frame_id"]], "presence_observation_ids": evidence,
                       "before": earlier_status, "after": status, "routes": ["body_presence"], "task_ids": [],
                       "criteria": ["reviewed_visibility_change"], "sampling_fps": FPS,
                       "precision": "sample_bounded", "evidence_modality": "visual", "experimental": True,
                       "review_status": "human-verified", "reviewer": review["reviewer"], "confidence": 1.0})
    return events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pilot = commands.add_parser("scan")
    pilot.add_argument("--root", type=Path, required=True)
    pilot.add_argument("--cache", type=Path, default=Path("data/league_assets"))
    pilot.add_argument("--output", type=Path, required=True)
    scoreboard = pilot.add_mutually_exclusive_group(required=True)
    scoreboard.add_argument("--intake-run", type=Path)
    scoreboard.add_argument("--scoreboard-skim", type=Path)
    pilot.add_argument("--clip-id", action="append", required=True)
    pilot.add_argument("--main-champion", required=True)
    pilot.add_argument("--friendly-side", choices=("left", "right"))
    pilot.add_argument("--main-tracks", type=Path)
    pilot.add_argument("--render-gallery", type=Path)
    pilot.add_argument("--check-inputs", action="store_true")
    derive = commands.add_parser("transitions")
    derive.add_argument("--pilot", type=Path, required=True)
    derive.add_argument("--review", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "scan":
        result = scan(args.root, args.intake_run or args.scoreboard_skim, args.cache, args.output,
                      clip_ids=args.clip_id, main_champion=args.main_champion, slot_skim=args.scoreboard_skim,
                      render_gallery=args.render_gallery, friendly_side=args.friendly_side,
                      main_tracks=args.main_tracks, check_inputs=args.check_inputs)
    else:
        report = read_json(args.pilot / "report.json")
        observations = args.pilot / "presence_observations.jsonl"
        if file_hash(observations) != report["files"][observations.name]:
            raise ValueError("Pilot visibility observations changed after the scan")
        review = read_json(args.review)
        events = reviewed_transitions(read_rows(observations), review, report["source_sha256"])
        output = args.pilot / "presence-transitions" / (digest(review)[:20] + ".jsonl")
        write_rows(output, events)
        result = {"path": str(output), "events": len(events), "experimental": True}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
