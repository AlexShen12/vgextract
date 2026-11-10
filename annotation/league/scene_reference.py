"""Source-only development references for scene-understanding stages 2 and 3.

This module prepares evidence and validates annotations. It performs no recognition
inference, changes no historical benchmark, and cannot certify human verification.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re

from .core import atomic_json, finite, read_json, uid

NEGATIVE_CASES = (
    "initial_cooldown", "cost_countdown_ambiguity", "camera_motion",
    "competing_caster", "interrupted_recall", "quiet",
)
SEQUENCE_CASES = (
    "isolated_ability", "rapid_attacks", "empowered_attack",
    "observable_miss", "obscured_outcome", "recall_lifecycle",
)
KINDS = {"movement", "basic_attack", "ability_cast", "item_use", "recall",
         "minion_last_hit", "base_arrival", "hud_activation"}


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def interval(row, lower, upper, prefix=""):
    start, end = row.get(prefix+"start"), row.get(prefix+"end")
    if not (finite(start) and finite(end) and lower <= start <= end <= upper):
        raise ValueError("Invalid source interval")
    return start, end


def validate_manifest(manifest):
    if manifest.get("schema_version") != 1 or manifest.get("split") != "development":
        raise ValueError("This revision prepares development material only")
    if manifest.get("stage") not in (2, 3):
        raise ValueError("Only stages 2 and 3 are supported")
    source = manifest["source"]
    if not re.fullmatch(r"[0-9a-f]{64}", source["sha256"]):
        raise ValueError("Source SHA-256 is required")
    if (source.get("width"), source.get("height")) != (1920, 1080):
        raise ValueError("Source review geometry requires native 1920 by 1080 input")
    guard = manifest.get("guard_seconds", 8)
    if not finite(guard) or guard < 8:
        raise ValueError("Preserve at least eight seconds of held-out guard")
    seen = set()
    for clip in manifest["clips"]:
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", clip["id"]) or clip["id"] in seen:
            raise ValueError("Invalid or duplicate clip ID")
        seen.add(clip["id"])
        start, end = interval(clip, 0, float("inf"))
        if start == end or clip.get("split") != "development":
            raise ValueError("Development clips must have positive duration")
        if clip.get("focus_interval") is not None:
            focus = clip["focus_interval"]
            if not isinstance(focus, list) or len(focus) != 2:
                raise ValueError("Focus interval must have two source boundaries")
            interval({"start": focus[0], "end": focus[1]}, start, end)
        for excluded in manifest.get("held_out_exclusions", []):
            a, b = interval(excluded, 0, float("inf"))
            if start < b+guard and a-guard < end:
                raise ValueError(f"Clip {clip['id']} crosses held-out guard {excluded['id']}")


def record_exposure(path, source_sha256, clip_id, start, end, reason):
    """Preparation is conservatively exposed even if a candidate is later rejected."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = read_json(path) if path.exists() else {"schema_version": 1, "entries": []}
        entry = {"id": uid("scene-exposure", source_sha256, clip_id, start, end, reason),
                 "source_sha256": source_sha256, "clip_id": clip_id, "start": start,
                 "end": end, "split": "development", "reason": reason}
        if not any(row["id"] == entry["id"] for row in data["entries"]):
            data["entries"].append(entry)
        atomic_json(path, data)


def sample_source(source, start, end, fps):
    import av
    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "2")))
        container.seek(int(start / stream.time_base), stream=stream, backward=True)
        next_time = start
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            time = float(frame.pts * frame.time_base)
            if time < start:
                continue
            if time >= end:
                break
            if fps is not None and time+1e-8 < next_time:
                continue
            yield time, frame.to_image().convert("RGB")
            if fps is not None:
                import math
                next_time = start+(math.floor((time-start)*fps+1e-6)+1)/fps


def materialize(source, source_sha256, start, end, fps, directory):
    from PIL import Image, ImageDraw
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows, sheets, cards = [], [], []
    for number, (time, image) in enumerate(sample_source(source, start, end, fps)):
        frame_id = uid("scene-source-frame", source_sha256, time)
        native = directory/f"{number:05d}-source.jpg"
        image.save(native, quality=95, subsampling=0)
        card = Image.new("RGB", (960, 700), (20, 20, 20))
        card.paste(image.resize((960, 540), Image.Resampling.LANCZOS), (0, 25))
        # Preserve native-coordinate full source separately; this strip is enlarged.
        card.paste(image.crop((789, 978, 983, 1032)).resize((582, 162), Image.Resampling.NEAREST), (0, 538))
        ImageDraw.Draw(card).text((4, 5), f"sample {number:03d} | source PTS {time:.6f}s", fill="white")
        # The enlarged strip covers only the scaled bottom HUD, never gameplay.
        cards.append(card)
        rows.append({"id": frame_id, "index": number, "time": time,
                     "source_path": str(native.resolve())})
        if len(cards) == 4:
            sheet = Image.new("RGB", (1920, 1400), (20, 20, 20))
            for offset, item in enumerate(cards):
                sheet.paste(item, ((offset % 2)*960, (offset // 2)*700))
            path = directory/f"sheet-{len(sheets):03d}.jpg"
            sheet.save(path, quality=94, subsampling=0)
            sheets.append({"path": str(path.resolve()), "indexes": list(range(number-3, number+1))})
            cards = []
    if cards:
        sheet = Image.new("RGB", (1920, 1400), (20, 20, 20))
        for offset, item in enumerate(cards):
            sheet.paste(item, ((offset % 2)*960, (offset // 2)*700))
        path = directory/f"sheet-{len(sheets):03d}.jpg"
        sheet.save(path, quality=94, subsampling=0)
        sheets.append({"path": str(path.resolve()), "indexes": list(range(len(rows)-len(cards), len(rows)))})
    if not rows:
        raise ValueError("No source frames decoded")
    return {"start": start, "end": end, "sampling_fps": fps or "native",
            "source_sha256": source_sha256, "frames": rows, "sheets": sheets}


def prepare(manifest_path, root, fps=4):
    manifest_path, root = Path(manifest_path), Path(root)
    manifest = read_json(manifest_path)
    validate_manifest(manifest)
    if not finite(fps) or not 0 < fps <= 60:
        raise ValueError("Invalid review sampling FPS")
    if file_hash(manifest["source"]["path"]) != manifest["source"]["sha256"]:
        raise ValueError("Source bytes changed")
    root.mkdir(parents=True, exist_ok=True)
    inputs = []
    for clip in manifest["clips"]:
        path = root/clip["id"]/"source-inputs.json"
        record_exposure(manifest_path.parent/"exposure-ledger.json", manifest["source"]["sha256"],
                        clip["id"], clip["start"], clip["end"], "prepared_source_review")
        signature = {"source_sha256": manifest["source"]["sha256"], "start": clip["start"],
                     "end": clip["end"], "sampling_fps": fps}
        if path.exists():
            data = read_json(path)
            if any(data.get(key) != value for key,value in signature.items()):
                raise ValueError("Changed preparation requires a new output directory")
        else:
            data = {"id": clip["id"], **materialize(manifest["source"]["path"],
                manifest["source"]["sha256"], clip["start"], clip["end"], fps, path.parent/"inputs")}
            atomic_json(path, data)
        inputs.append({"id": clip["id"], "source_inputs": str(path.resolve())})
        print(clip["id"], len(data["frames"]), "source frames", flush=True)
    atomic_json(root/"source-review-index.json", {"source": manifest["source"], "clips": inputs,
        "method": "Source-only review. Overview sampling does not certify unseen frames or quiet intervals."})
    return inputs


def validate_label(label, clip, source_sha256):
    if label.get("clip_id") != clip["id"] or label.get("source_sha256") != source_sha256:
        raise ValueError("Label provenance differs from source/clip")
    if label.get("review_status") not in {"source_reviewed_model_draft", "human_verified", "rejected"}:
        raise ValueError("Explicit review status required")
    review = label["review"]
    if not review.get("reviewer") or review.get("prediction_outputs_seen") is not False:
        raise ValueError("Source reviewer and prediction-exposure statement required")
    if label["review_status"] == "human_verified" and review.get("human_reviewed") is not True:
        raise ValueError("A model review is not human verification")
    if review.get("method") is None or not review.get("inspected_sheets"):
        raise ValueError("Record the actual inspection method and evidence")
    if not isinstance(label.get("exhaustive"), bool):
        raise ValueError("Declare annotation completeness")
    seen = set()
    for event in label.get("events", []):
        if event["id"] in seen or event.get("kind") not in KINDS:
            raise ValueError("Invalid or duplicate event")
        seen.add(event["id"])
        start, end = interval(event, clip["start"], clip["end"])
        if event.get("actor") not in {"selected", "other", "unknown"}:
            raise ValueError("Actor must retain unknown when unresolved")
        if event.get("slot") not in {None, "Q", "W", "E", "R"}:
            raise ValueError("Invalid slot")
        if event.get("onset_bracket") is not None:
            bracket = event["onset_bracket"]
            if not isinstance(bracket, list) or len(bracket) != 2:
                raise ValueError("Onset must be an uncertainty bracket")
            interval({"start": bracket[0], "end": bracket[1]}, start, end)
        if event.get("end_bracket") is not None:
            bracket = event["end_bracket"]
            if not isinstance(bracket, list) or len(bracket) != 2:
                raise ValueError("End must be an uncertainty bracket")
            interval({"start": bracket[0], "end": bracket[1]}, start, end)
        if not event.get("evidence") or not isinstance(event.get("phases"), list):
            raise ValueError("Source evidence and phase list required")
        for evidence in event["evidence"]:
            if not finite(evidence.get("time")) or not clip["start"] <= evidence["time"] < clip["end"] or not evidence.get("cue"):
                raise ValueError("Evidence must cite an in-clip source time and cue")
        for phase in event["phases"]:
            interval(phase, start, end)
        for field in ("target", "outcome", "ability_name"):
            if event.get(field) is not None and not event.get(field+"_evidence"):
                raise ValueError(f"{field} assertion requires separate evidence")
        if event["kind"] == "hud_activation" and (event.get("target") is not None or event.get("outcome") is not None):
            raise ValueError("HUD activation alone cannot establish target/outcome")
        if event["kind"] == "recall" and event.get("terminal_status") not in {"completed", "interrupted", "ending_unknown"}:
            raise ValueError("Recall needs an explicit observed terminal status")
    for unknown in label.get("unknown_intervals", []):
        interval(unknown, clip["start"], clip["end"])
        if not unknown.get("reason"):
            raise ValueError("Unknown interval requires a reason")
    for case in label.get("cases", []):
        if case.get("tag") not in set(NEGATIVE_CASES+SEQUENCE_CASES):
            raise ValueError("Unknown case tag")
        if case.get("status") not in {"supported", "not_present", "unresolved"} or not case.get("reason"):
            raise ValueError("Case judgment must include its scope and reason")
        if case["status"] == "supported":
            start, _ = interval(case, clip["start"], clip["end"])
            if not case.get("evidence"):
                raise ValueError("Supported case needs visible source evidence")
            if not case.get("scope") or not case.get("forbidden_claim"):
                raise ValueError("Supported negative/sequence case needs scope and expected restriction")
            if case["tag"] == "initial_cooldown" and start > clip["start"]+.125:
                raise ValueError("Initial cooldown must be observed at clip entry")
            if case["tag"] == "quiet":
                if not label["exhaustive"] or review.get("full_native_review") is not True or not case.get("scope"):
                    raise ValueError("Quiet requires complete native review and explicit scope")
            if case["tag"] == "interrupted_recall" and not any(e["kind"] == "recall" and e.get("terminal_status") == "interrupted" for e in label["events"]):
                raise ValueError("Interrupted-recall negative requires a positive observed attempt")
            events = [event for event in label["events"] if event["start"] <= case["end"] and case["start"] <= event["end"]]
            physical_casts = [event for event in events if event["kind"] == "ability_cast"]
            attacks = [event for event in events if event["kind"] == "basic_attack"]
            activations = [event for event in events if event["kind"] == "hud_activation"]
            if case["tag"] == "rapid_attacks" and len({event["start"] for event in attacks}) < 2:
                raise ValueError("Rapid attacks require separately evidenced physical onsets")
            if case["tag"] == "isolated_ability" and not any(
                {"preparation", "recovery"} <= {phase["phase"] for phase in event["phases"]}
                and {"cast", "release"} & {phase["phase"] for phase in event["phases"]} for event in physical_casts
            ):
                raise ValueError("Complete isolated ability requires preparation, release and recovery evidence")
            if case["tag"] == "empowered_attack" and not any(
                activation["id"] in attack.get("linked_event_ids", []) or attack["id"] in activation.get("linked_event_ids", [])
                for activation in activations for attack in attacks
            ):
                raise ValueError("Empowered attack requires separately linked activation and physical attack")
            def outcome_kind(event):
                outcome = event.get("outcome")
                return outcome.get("kind") if isinstance(outcome, dict) else outcome
            if case["tag"] == "observable_miss" and not any(outcome_kind(event) in {"miss", "no_hit", "cast_miss", "observed_miss"} for event in physical_casts):
                raise ValueError("A missing hit observation is not a demonstrated miss")
            if case["tag"] == "obscured_outcome" and not any(event.get("outcome") is None and event.get("uncertainty") for event in physical_casts):
                raise ValueError("Obscured result needs an observed physical cast and unresolved outcome")
            if case["tag"] == "recall_lifecycle" and not any(event["kind"] == "recall" for event in events):
                raise ValueError("Recall lifecycle requires a source-observed channel")
    return True


def derive_frames(manifest_path, labels_dir, output):
    """Map labels onto native PTS; do not upgrade sparse source-review coverage."""
    import av
    manifest = read_json(manifest_path)
    validate_manifest(manifest)
    if file_hash(manifest["source"]["path"]) != manifest["source"]["sha256"]:
        raise ValueError("Source bytes changed")
    totals = []
    for clip in manifest["clips"]:
        label_path = Path(labels_dir)/(clip["id"]+".json")
        if not label_path.exists():
            raise ValueError(f"Missing source review for {clip['id']}")
        label = read_json(label_path)
        validate_label(label, clip, manifest["source"]["sha256"])
        label_digest = file_hash(label_path)
        directory = Path(output)/clip["id"]
        directory.mkdir(parents=True, exist_ok=True)
        count = with_events = 0
        # Decode PTS without converting full frames to RGB a second time.
        with av.open(manifest["source"]["path"]) as container, (directory/"frames.jsonl").open("w") as stream:
            video = container.streams.video[0]
            video.codec_context.thread_count = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "2")))
            container.seek(int(clip["start"]/video.time_base), stream=video, backward=True)
            for frame in container.decode(video):
                if frame.pts is None:
                    continue
                time = float(frame.pts*video.time_base)
                if time < clip["start"]:
                    continue
                if time >= clip["end"]:
                    break
                unknown = [item["reason"] for item in label.get("unknown_intervals", []) if item["start"] <= time <= item["end"]]
                observations = [{"event_id": event["id"], "kind": event["kind"],
                    "actor": event["actor"], "slot": event.get("slot"),
                    "ability_name": event.get("ability_name"), "target": event.get("target"),
                    "outcome": event.get("outcome"),
                    "timing_semantics": "onset_uncertainty_bracket" if event["kind"] == "hud_activation" else "visible_occurrence_interval",
                    "phases": [phase["phase"] for phase in event["phases"] if phase["start"] <= time <= phase["end"]]}
                    for event in label["events"] if event["start"] <= time <= event["end"]
                    and (not unknown or event["kind"] in {"hud_activation", "recall"})]
                stream.write(json.dumps({"frame_id": uid("scene-source-frame", manifest["source"]["sha256"], time),
                    "index": count, "source_pts_seconds": time, "observations": observations,
                    "unknown_reasons": unknown, "review_status": label["review_status"],
                    "exhaustive": label["exhaustive"], "label_sha256": label_digest,
                    "derivation": "Intervals mapped onto native source PTS; empty observations do not establish inactivity."})+"\n")
                count += 1
                with_events += bool(observations)
        totals.append({"clip_id": clip["id"], "native_frames": count, "frames_with_events": with_events})
    atomic_json(Path(output)/"frame-summary.json", {"clips": totals,
        "native_frames": sum(row["native_frames"] for row in totals), "derived": True})
    return totals


def summarize(manifest_path, labels_dir):
    manifest = read_json(manifest_path)
    validate_manifest(manifest)
    totals = {tag: {"supported_clips": [], "unresolved_clips": []} for tag in NEGATIVE_CASES+SEQUENCE_CASES}
    missing, rows = [], []
    for clip in manifest["clips"]:
        path = Path(labels_dir)/(clip["id"]+".json")
        if not path.exists():
            missing.append(clip["id"])
            continue
        label = read_json(path)
        validate_label(label, clip, manifest["source"]["sha256"])
        for case in label["cases"]:
            key = "supported_clips" if case["status"] == "supported" else "unresolved_clips" if case["status"] == "unresolved" else None
            if key and clip["id"] not in totals[case["tag"]][key]:
                totals[case["tag"]][key].append(clip["id"])
        rows.append({"id": clip["id"], "start": clip["start"], "end": clip["end"],
                     "review_status": label["review_status"], "events": len(label["events"]),
                     "exhaustive": label["exhaustive"], "label_sha256": file_hash(path)})
    return {"stage": manifest["stage"], "source_sha256": manifest["source"]["sha256"],
            "clips": rows, "missing_labels": missing, "coverage": totals,
            "limitations": "Model source reviews do not establish human verification, exhaustive action coverage or recognition accuracy."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--manifest", required=True, type=Path)
    prepare_parser.add_argument("--output", required=True, type=Path)
    prepare_parser.add_argument("--fps", type=float, default=4)
    detail = commands.add_parser("detail")
    detail.add_argument("--manifest", required=True, type=Path)
    detail.add_argument("--clip", required=True)
    detail.add_argument("--start", required=True, type=float)
    detail.add_argument("--end", required=True, type=float)
    detail.add_argument("--fps", type=float, default=None, help="Omit for native frames")
    detail.add_argument("--output", required=True, type=Path)
    report = commands.add_parser("report")
    report.add_argument("--manifest", required=True, type=Path)
    report.add_argument("--labels", required=True, type=Path)
    report.add_argument("--output", required=True, type=Path)
    derive = commands.add_parser("derive")
    derive.add_argument("--manifest", required=True, type=Path)
    derive.add_argument("--labels", required=True, type=Path)
    derive.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.manifest, args.output, args.fps)
    elif args.command == "detail":
        manifest = read_json(args.manifest)
        validate_manifest(manifest)
        clip = next(c for c in manifest["clips"] if c["id"] == args.clip)
        interval({"start": args.start, "end": args.end}, clip["start"], clip["end"])
        if args.fps is not None and (not finite(args.fps) or not 0 < args.fps <= 60):
            raise ValueError("Invalid detail sampling FPS")
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError("Detail output must be new; preserve previous evidence")
        record_exposure(args.manifest.parent/"exposure-ledger.json", manifest["source"]["sha256"],
                        clip["id"], args.start, args.end, "native_or_dense_source_review")
        data = materialize(manifest["source"]["path"], manifest["source"]["sha256"],
                           args.start, args.end, args.fps, args.output)
        atomic_json(args.output/"source-inputs.json", data)
        print(json.dumps({"frames": len(data["frames"]), "sheets": len(data["sheets"])}))
    elif args.command == "report":
        result = summarize(args.manifest, args.labels)
        atomic_json(args.output, result)
        print(json.dumps(result, indent=2))
    else:
        print(json.dumps(derive_frames(args.manifest, args.labels, args.output), indent=2))


if __name__ == "__main__":
    main()
