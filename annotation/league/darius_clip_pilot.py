"""CPU-only cut and gameplay clip comparison on the pinned Darius recording."""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from html import escape
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
from urllib.parse import quote

from .caption_segments import POLICY, contexts, episode_windows
from .core import atomic_json, finite, read_json, read_rows, uid, write_rows
from .media import file_hash
from .normalization import accepted


SOURCE_SHA = "f4f36e1a274617ecbe224eaf7fc3f8375f757acb7a517bcc0ee7780fcd83f240"
REVISION = "43d5214f15ef4d032b4e"
RUN = Path("data/league_runs/midbeast-pass1-final-v1")
BENCHMARKS = Path("data/league_runs/midbeast-OLLZBe5-fg4-v2/benchmarks")
WINDOWS = ((0, 60), (418, 478), (836, 896), (1254, 1314), (1672, 1732))
MATCH_TOLERANCE = .25
DETECTOR_VERSION = "0.7.3"
DETECTOR_SETTINGS = {"adaptive_threshold": 3.0, "min_content_val": 15.0,
                     "window_width": 2, "min_scene_len": 15}
FRAME_COUNT = 103858
HARD = {"edit", "rewind"}
CONTINUOUS = {"camera_switch", "overlay_or_motion"}
CONTEXT_REASONS = {"layout_transition", "game_clock_discontinuity", "game_clock_rewind"}
PHASES = ("setup", "action", "outcome")


def scan_cuts(source, expected_frames):
    from scenedetect import AdaptiveDetector, SceneManager
    from scenedetect.backends.pyav import VideoStreamAv

    manager = SceneManager()
    manager.add_detector(AdaptiveDetector(**DETECTOR_SETTINGS))
    with VideoStreamAv(str(source), threading_mode="AUTO") as video:
        frames = manager.detect_scenes(video=video, show_progress=False)
    if frames != expected_frames:
        raise ValueError(f"Incomplete source scan ({frames} of {expected_frames} frames)")
    # The first scene starts at the source origin; subsequent starts are cuts.
    return [start.seconds for start, _ in manager.get_scene_list()[1:]]


def split_intervals(windows, exclusions):
    for left, right in sorted(exclusions):
        windows = [(a, b) for start, end in windows
                   for a, b in ((start, min(end, left)), (max(start, right), end)) if a < b]
    return windows


def contains(intervals, time):
    return any(start <= time < end for start, end in intervals)


def compare_cuts(detected, playback, clocks, windows, review_windows):
    visual = [row for row in playback if row.get("visual_motion") is not None]
    baseline = [row for row in visual if "visual_cut" in row.get("discontinuity_reasons", ())]
    coverage = []
    for row in visual:
        start, end = row["start"], row["end"]
        if coverage and start - coverage[-1][1] <= .019:
            coverage[-1][1] = max(coverage[-1][1], end)
        else:
            coverage.append([start, end])
    # Closest pairs win, with stable timestamp ordering for equal distances.
    pairs = sorted((abs(time - row["start"]), time, row["start"], row["id"])
                   for time in detected for row in baseline
                   if abs(time - row["start"]) <= MATCH_TOLERANCE
                   and any(a <= time < b and a <= row["start"] < b for a, b in windows))
    matched, used = {}, set()
    for _, time, _, row_id in pairs:
        if time not in matched and row_id not in used:
            matched[time] = row_id
            used.add(row_id)
    by_id = {row["id"]: row for row in baseline}
    candidates = [(time, by_id.get(matched.get(time))) for time in detected]
    candidates.extend((None, row) for row in baseline if row["id"] not in used)
    context = [row for row in playback if CONTEXT_REASONS.intersection(row.get("discontinuity_reasons", ()))]
    times = [t if t is not None else row["start"] for t, row in candidates]
    candidates.extend((None, row) for row in context
                      if row not in baseline and all(abs(row["start"] - t) > MATCH_TOLERANCE for t in times))
    clock_times = [row["time"] for row in clocks]
    rows = []
    for detected_time, tracker in candidates:
        time = detected_time if detected_time is not None else tracker["start"]
        tracker_reasons = tracker.get("discontinuity_reasons", ()) if tracker else ()
        tracker_time = tracker["start"] if "visual_cut" in tracker_reasons else None
        if detected_time is not None:
            comparison = "matched" if tracker_time is not None else (
                "detector_only" if contains(coverage, time) else "outside_tracker_coverage")
        else:
            comparison = "tracker_only" if tracker_time is not None else "context_only"
        index = bisect_left(clock_times, time)
        neighbors = [row for row in clocks[max(0, index-1):index+1] if abs(row["time"] - time) <= 2]
        nearby = [{"id": row["id"], "time": row["start"], "reasons": row["discontinuity_reasons"],
                   "evidence_ids": row.get("evidence_ids", [])}
                  for row in context if abs(row["start"] - time) <= MATCH_TOLERANCE]
        rows.append({"id": uid("darius-cut", SOURCE_SHA, time, detected_time, tracker_time),
                     "time": time, "detector_time": detected_time, "tracker_time": tracker_time,
                     "tracker_id": tracker["id"] if tracker else None, "comparison": comparison,
                     "review_eligible": contains(review_windows, time), "clock_readings": neighbors,
                     "nearby_context_changes": nearby})
    return sorted(rows, key=lambda row: (row["time"], row["id"]))


def review_windows(benchmarks):
    exclusions = []
    for name in ("captions/captions.json", "caption-events/v1/events.json", "scene-understanding/v2-pilot/reservation.json"):
        document = read_json(benchmarks / name)
        if (document.get("source_sha256") or document["source"]["sha256"]) != SOURCE_SHA:
            raise ValueError(f"Unexpected held-out source in {name}")
        guard = max(8, document.get("guard_seconds", 8))
        exclusions.extend((r["start"] - guard, r["end"] + guard)
                          for r in document["clips"] if r["split"] == "held_out")
    return split_intervals(WINDOWS, exclusions)


def read_clocks(path):
    import json

    samples = defaultdict(list)
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if '"game_clock"' not in line:
                continue
            row = json.loads(line)
            value = row.get("value")
            if row["field"] == "game_clock" and accepted(row) and isinstance(value, dict) and finite(value.get("seconds")):
                samples[row["time"]].append((value["seconds"], row["id"]))
    return [{"time": time, "seconds": rows[0][0], "evidence_ids": [r[1] for r in rows]}
            for time, rows in sorted(samples.items()) if all(r[0] == rows[0][0] for r in rows)]


def collect_seeds(events, commentary, lead_ids, windows):
    seeds = []
    for row in events:
        kind = row["kind"]
        if kind == "kill_announcement" and "Darius" in (row.get("actor"), row.get("target")):
            start = end = row["start"]
        elif kind in {"observed_channel", "hud_slot_activation"} and row.get("owner_id") == "selected":
            start, end = row["start"], row["end"]
        else:
            continue
        if contains(windows, start):
            seeds.append({"id": row["id"], "kind": kind, "start": start, "end": end,
                          "evidence_ids": row.get("frame_ids", []) + [row["id"]]})
    for row in commentary:
        if row["id"] in lead_ids and contains(windows, row["start"]):
            seeds.append({"id": row["id"], "kind": "commentary_lead", "start": row["start"],
                          "end": row["start"], "evidence_ids": [row["id"]]})
    if lead_ids - {row["id"] for row in seeds if row["kind"] == "commentary_lead"}:
        raise ValueError("Commentary leads must identify eligible source commentary")
    return sorted(seeds, key=lambda row: (row["start"], row["end"], row["id"]))


def clip_contexts(selection, windows, cuts, fps):
    """Use the caption pilot's continuity barriers and the reviewed edit times."""
    continuous = {r["tracker_time"] for r in cuts if r["classification"] in CONTINUOUS}
    edits = [r["time"] for r in cuts if r["classification"] in HARD]
    edits.sort()
    # An epoch with a single resolved champion can still contain unreadable HUD frames.
    champions = defaultdict(set)
    for row in selection:
        if row.get("champion"):
            champions[row["owner_epoch"]].add(row["champion"])
    for start, end in windows:
        frames = []
        previous = None
        segment = None
        for row in selection:
            if not start <= row["start"] < end:
                continue
            playing = row["playback_state"] == "playing" and champions[row["owner_epoch"]] == {"Darius"}
            can_bridge = (previous and playing and previous["playback_state"] == "playing"
                          and row["owner_epoch"] == previous["owner_epoch"]
                          and row["start"] in continuous
                          and row.get("discontinuity_reasons") == ["visual_cut"])
            if not previous or (row["playback_segment_id"] != previous["playback_segment_id"] and not can_bridge):
                segment = row["playback_segment_id"]
            # Include the cut's index in the context, then clamp the old context at its exact PTS.
            edit_index = bisect_right(edits, row["start"])
            frames.append({"id": row["id"], "time": row["start"], "source_fps": fps,
                           "segment_id": (segment, edit_index), "owner_epochs": {"selected": row["owner_epoch"]},
                           "layout_valid": playing, "playback": row})
            previous = row
        for group in contexts(frames, (start, end)):
            edit_index = group["context"][0][1]
            if edit_index < len(edits):
                group["end"] = min(group["end"], edits[edit_index])
            if group["start"] < group["end"]:
                yield group


def group_clips(seeds, groups, references):
    clips = {}
    for group in groups:
        # The seed onset remains useful when the far edge of a cue is censored.
        eligible = [{**r, "end": min(r["end"], group["end"])} for r in seeds
                    if group["start"] <= r["start"] < group["end"]]
        for chain in episode_windows(eligible, group, lambda r: (r["start"], r["end"]), lambda _: True):
            for start, end, included in chain:
                seed_ids = [r["id"] for r in included]
                relevant = [ref for ref in references if set(seed_ids).intersection(ref["seed_ids"])]
                phases = [span for ref in relevant for span in ref["phases"].values() if span is not None]
                original = [start, end]
                if phases:
                    left = max(group["start"], min(start, min(span[0] for span in phases)))
                    right = min(group["end"], max(end, max(span[1] for span in phases)))
                    if right - left <= POLICY["max_clip_seconds"]:
                        start, end = left, right
                key = (start, end, group["start"], group["end"])
                clip = clips.setdefault(key, {"start": start, "end": end, "initial_ranges": [],
                                              "context_range": [group["start"], group["end"]],
                                              "seed_ids": [], "evidence_ids": [], "episode_ids": set()})
                clip["initial_ranges"].append(original)
                clip["seed_ids"].extend(seed_ids)
                clip["evidence_ids"].extend(e for r in included for e in r["evidence_ids"])
                clip["episode_ids"].update(ref["id"] for ref in relevant)
    # Refining separate seeds can produce nested ranges for the same reviewed episode.
    # Keep the containing range and every seed that led to it.
    selected = []
    for clip in sorted(clips.values(), key=lambda r: (r["start"], -r["end"])):
        containing = next((r for r in selected if clip["end"] <= r["end"] and r["context_range"] == clip["context_range"]
                           and r["episode_ids"] & clip["episode_ids"]), None)
        if containing:
            for key in ("seed_ids", "evidence_ids", "initial_ranges"):
                containing[key].extend(clip[key])
            containing["episode_ids"].update(clip["episode_ids"])
        else:
            selected.append(clip)
    result = []
    for clip in selected:
        seed_ids = sorted(set(clip["seed_ids"]))
        result.append({**clip, "seed_ids": seed_ids, "evidence_ids": sorted(set(clip["evidence_ids"])),
                       "episode_ids": sorted(clip["episode_ids"]),
                       "source_sha256": SOURCE_SHA, "time_basis": "source_video_seconds",
                       "id": uid("darius-clip", SOURCE_SHA, clip["start"], clip["end"], seed_ids)})
    return sorted(result, key=lambda r: (r["start"], r["end"], r["id"]))


def assess_clips(clips, references):
    summaries = []
    for ref in references:
        candidates = [clip for clip in clips if set(ref["seed_ids"]).intersection(clip["seed_ids"])]
        visible = [name for name in PHASES if ref["phases"][name] is not None]

        def covers(clip, name):
            start, end = ref["phases"][name]
            return clip["start"] <= start and end <= clip["end"]

        summaries.append({"id": ref["id"], "clip_ids": [r["id"] for r in candidates],
                          "phases": {name: ("unresolved" if name not in visible else "retained"
                                            if any(covers(r, name) for r in candidates) else "missing") for name in PHASES},
                          "complete": (any(all(covers(r, name) for name in PHASES) for r in candidates)
                                       if len(visible) == len(PHASES) else None)})
    return summaries


def review_references(review, seeds, windows):
    ids = {r["id"] for r in seeds}
    references = review.get("episodes", [])
    for ref in references:
        spans = [s for s in ref["phases"].values() if s is not None]
        if (not ref["seed_ids"] or not set(ref["seed_ids"]) <= ids or set(ref["phases"]) != set(PHASES)
                or any(len(s) != 2 or not all(finite(t) for t in s) or s[0] >= s[1] for s in spans)
                or (spans and not any(a <= min(s[0] for s in spans) < max(s[1] for s in spans) <= b
                                      for a, b in windows))):
            raise ValueError(f"Invalid source review episode {ref['id']}")
    if len({r["id"] for r in references}) != len(references):
        raise ValueError("Duplicate source review episode ID")
    return references


def retention(assessments):
    return {phase: dict(Counter(a["phases"][phase] for a in assessments)) for phase in PHASES} | {
        "complete_episodes": {"retained": sum(a["complete"] is True for a in assessments),
                              "observable": sum(a["complete"] is not None for a in assessments)}}


def write_review_html(path, source, report, cuts, clips, seeds, references, windows):
    def play(start, end, label=None):
        text = f"{start:.2f}–{end:.2f} s"
        return f'<button data-start="{start}" data-end="{end}" title="{text}">{escape(label or text)}</button>'

    def table(headers, rows):
        head = "".join(f"<th scope='col'>{escape(text)}</th>" for text in headers)
        body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>" for cells in rows)
        return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

    clip_rows = {r["id"]: r for arm in clips.values() for r in arm}
    assessment = {arm: {r["id"]: r for r in values} for arm, values in report["episode_assessments"].items()}
    episodes = []
    for ref in references:
        witnesses = " ".join(play(*span, name.capitalize()) for name, span in ref["phases"].items() if span)
        cells = [escape(ref["id"]), f'{escape(ref["note"])}<br>{witnesses}']
        for arm in clips:
            row = assessment[arm][ref["id"]]
            players = " ".join(play(clip_rows[id]["start"], clip_rows[id]["end"]) for id in row["clip_ids"])
            statuses = ", ".join(f"{p} {status}" for p, status in row["phases"].items())
            cells.append(f"{players}<br>{escape(statuses)}")
        episodes.append(cells)
    cut_rows = []
    for cut in cuts:
        time = cut["time"]
        window = next(((a, b) for a, b in windows if a <= time < b), None)
        if window:
            cut_rows.append((play(max(window[0], time - .4), min(window[1], time + .4)),
                             escape(cut["comparison"]), escape(cut["classification"]),
                             escape(cut.get("note", ""))))
    kinds = {r["id"]: r["kind"].replace("_", " ") for r in seeds}
    all_clips = [(escape(arm), play(r["start"], r["end"]), escape(", ".join(kinds[s] for s in r["seed_ids"])))
                 for arm, rows in clips.items() for r in rows]
    counts = report["cut_comparison"]
    classes = report["reviewed_classifications"]
    summary = (f"{report['detected_cuts']} PySceneDetect candidates across the upload. "
               f"{counts.get('detector_only', 0)} detector-only and {counts.get('matched', 0)} matched within tracker coverage. "
               f"{classes.get('edit', 0)} edits and {classes.get('rewind', 0)} rewinds confirmed in the reviewed windows.")
    source_url = quote(os.path.relpath(source, path.parent))
    path.write_text(f'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Darius clip pilot</title>
<style>
body {{background:#f4f6f7;color:#14242b;font:15px/1.5 system-ui;margin:24px auto;padding:0 16px;max-width:1100px}}
video {{background:#14242b;width:min(100%,780px);max-height:44vh;display:block}}
table {{border-collapse:collapse;width:100%;text-align:left;margin:12px 0 24px}}
th,td {{padding:8px;border-bottom:1px solid #c7d2d8;vertical-align:top}}
button,summary {{cursor:pointer}} button {{padding:5px 9px;white-space:nowrap}}
.player {{position:sticky;top:0;background:#f4f6f7;padding:8px 0}}
</style>
<h1>Darius clip pilot</h1>
<p>{escape(summary)}</p>
<p><a href="report.json">Metrics</a> · <a href="cuts.jsonl">Cut comparisons</a> ·
<a href="baseline-clips.jsonl">Baseline clips</a> · <a href="reviewed-clips.jsonl">Reviewed clips</a> ·
<a href="seeds.jsonl">Candidate seeds</a></p>
<div class="player"><video controls preload="metadata" src="{escape(source_url, quote=True)}"></video>
<div id="status" role="status">Choose a source range to play.</div></div>
<h2>Gameplay episode review</h2>
{table(('Episode', 'Source observation', 'Baseline boundaries', 'Reviewed boundaries'), episodes)}
<details><summary>All candidate clips</summary>{table(('Version', 'Source range', 'Seed evidence'), all_clips)}</details>
<details><summary>Cut review</summary>{table(('Source range', 'Comparison', 'Classification', 'Observation'), cut_rows)}</details>
<script>
const video = document.querySelector('video');
const status = document.querySelector('#status');
let end = Infinity;
let stopId = null;
const stopAtEnd = (_, frame) => {{
  if ((frame?.mediaTime ?? video.currentTime) >= end) {{ video.pause(); return; }}
  if (video.requestVideoFrameCallback && !video.paused) stopId = video.requestVideoFrameCallback(stopAtEnd);
}};
video.addEventListener('timeupdate', () => {{ if (video.currentTime >= end) video.pause(); }});
video.addEventListener('play', () => {{
  if (video.requestVideoFrameCallback) {{
    if (stopId !== null) video.cancelVideoFrameCallback(stopId);
    stopId = video.requestVideoFrameCallback(stopAtEnd);
  }}
}});
for (const button of document.querySelectorAll('button[data-start]')) {{
  button.addEventListener('click', () => {{
    video.pause();
    end = Number(button.dataset.end);
    video.currentTime = Number(button.dataset.start);
    status.textContent = 'Source range ' + button.textContent;
    video.play().catch(error => {{ status.textContent = error.message; }});
  }});
}}
</script></html>''', encoding="utf-8")


def run(run_dir, benchmarks, output, review_path=None):
    manifest = read_json(run_dir / "manifest.json")
    source = (run_dir / manifest["source"]["path"]).resolve()
    if manifest["source"]["sha256"] != SOURCE_SHA or file_hash(source) != SOURCE_SHA:
        raise ValueError("This pilot requires the pinned Darius source MP4")
    snapshot = run_dir / "revisions" / REVISION
    if ((output.is_relative_to(run_dir) and not output.is_relative_to(run_dir / "experiments"))
            or source.is_relative_to(output)):
        raise ValueError("Choose a separate experiment output directory")
    checksums = read_json(snapshot / "checksums.json")["files"]
    tables = ("playback_timeline", "selection_timeline", "events", "commentary_timeline", "observations")
    paths = {table: snapshot / (table + ".jsonl") for table in tables}
    if any(file_hash(path) != checksums[path.name] for path in paths.values()):
        raise ValueError("Input tables differ from the pinned Pass 1 revision")
    windows = review_windows(benchmarks)
    recipe = {"source_sha256": SOURCE_SHA, "version": DETECTOR_VERSION, "settings": DETECTOR_SETTINGS, "frames": FRAME_COUNT}
    cache = output / "detector-scan.json"
    if cache.exists():
        scan = read_json(cache)
        if any(scan.get(key) != value for key, value in recipe.items()):
            raise ValueError("Scan cache uses a different source or detector recipe")
    else:
        try:
            installed_version = version("scenedetect-headless")
        except PackageNotFoundError:
            installed_version = None
        if installed_version != DETECTOR_VERSION:
            raise ValueError("Install requirements-darius-clips.txt for the pinned detector")
        scan = {**recipe, "cuts": scan_cuts(source, FRAME_COUNT)}
        atomic_json(cache, scan)
    cut_times = scan["cuts"]
    if (any(not finite(t) or not 0 < t < manifest["source"]["duration"] for t in cut_times)
            or cut_times != sorted(set(cut_times))):
        raise ValueError("Cached cut timestamps are invalid")

    cuts = compare_cuts(cut_times, read_rows(paths["playback_timeline"]), read_clocks(paths["observations"]), WINDOWS, windows)
    review = read_json(review_path) if review_path is not None else {}
    if review and (review.get("source_sha256") != SOURCE_SHA or review.get("revision") != REVISION):
        raise ValueError("Review must use the pinned source and Pass 1 revision")
    cut_reviews = {r["id"]: r for r in review.get("cuts", [])}
    eligible = {r["id"] for r in cuts if r["review_eligible"]}
    if (len(cut_reviews) != len(review.get("cuts", [])) or set(cut_reviews) - eligible
            or any(r["classification"] not in HARD | CONTINUOUS | {"unresolved"} for r in cut_reviews.values())):
        raise ValueError("Cut labels must be unique, valid and in the guarded review windows")
    for cut in cuts:
        annotation = cut_reviews.get(cut["id"], {})
        cut.update(classification=annotation.get("classification", "unreviewed"), note=annotation.get("note", ""))

    seeds = collect_seeds(read_rows(paths["events"]), read_rows(paths["commentary_timeline"]),
                          set(review.get("commentary_leads", [])), windows)
    refs = review_references(review, seeds, windows)
    selection = read_rows(paths["selection_timeline"])
    fps = manifest["source"]["fps"]
    groups = {arm: list(clip_contexts(selection, windows, boundaries, fps))
              for arm, boundaries in (("baseline", []), ("reviewed", cuts))}
    clips = {arm: group_clips(seeds, contexts, refs) for arm, contexts in groups.items()}
    assessments = {arm: assess_clips(rows, refs) for arm, rows in clips.items()}
    initial = {arm: assess_clips(group_clips(seeds, contexts, []), refs) for arm, contexts in groups.items()}
    confirmed = [r for r in cuts if r["classification"] in HARD]
    phase_ranges = [(min(s[0] for s in r["phases"].values()), max(s[1] for s in r["phases"].values()))
                    for r in refs if all(r["phases"].values())]
    report = {"schema_version": 1, "source_sha256": SOURCE_SHA, "input_revision": REVISION,
              "detector": recipe, "detected_cuts": len(cut_times), "match_tolerance": MATCH_TOLERANCE,
              "tracker_windows": WINDOWS, "review_windows": windows, "reviewer": review.get("reviewer"),
              "review_method": review.get("method"), "prior_review_exposure": review.get("prior_review_exposure"),
              "cut_comparison": dict(Counter(r["comparison"] for r in cuts)),
              "reviewed_classifications": dict(Counter(r["classification"] for r in cuts if r["review_eligible"])),
              "reviewed_detector_classifications": dict(Counter(r["classification"] for r in cuts
                                                                  if r["review_eligible"] and r["detector_time"] is not None)),
              "confirmed_discontinuities": [{"id": r["id"], "time": r["time"], "classification": r["classification"],
                                            "comparison": r["comparison"]} for r in confirmed],
              "camera_switches_inside_complete_episodes": [r["id"] for r in cuts if r["classification"] == "camera_switch"
                                                            and any(a < r["time"] < b for a, b in phase_ranges)],
              "seed_counts": dict(Counter(r["kind"] for r in seeds)),
              "clip_counts": {arm: len(rows) for arm, rows in clips.items()},
              "omitted_seeds": {arm: sorted({r["id"] for r in seeds} - {s for r in rows for s in r["seed_ids"]})
                                for arm, rows in clips.items()},
              "retention": {arm: retention(assessment) for arm, assessment in assessments.items()},
              "initial_retention": {arm: retention(assessment) for arm, assessment in initial.items()},
              "episode_assessments": assessments}
    write_rows(output / "cuts.jsonl", cuts)
    write_rows(output / "seeds.jsonl", seeds)
    for arm, rows in clips.items():
        write_rows(output / f"{arm}-clips.jsonl", rows)
    write_review_html(output / "review.html", source, report, cuts, clips, seeds, refs, windows)
    atomic_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=RUN)
    parser.add_argument("--benchmarks", type=Path, default=BENCHMARKS)
    parser.add_argument("--output", type=Path, help="Defaults to experiments/darius-clips/r1 under the Pass 1 run")
    parser.add_argument("--review", type=Path, help="Source review JSON with cuts, commentary_leads and episodes")
    args = parser.parse_args()
    output = (args.output or args.run_dir / "experiments/darius-clips/r1").resolve()
    report = run(args.run_dir.resolve(), args.benchmarks, output, args.review)
    print(f"{report['detected_cuts']} cut candidates. Clip counts {report['clip_counts']}. Review {output / 'review.html'}")


if __name__ == "__main__":
    main()
