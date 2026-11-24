"""Prepare and run an experimental simultaneous multi-champion SAM pilot."""

import argparse
from itertools import permutations
import json
from pathlib import Path
import time

from .core import atomic_json, digest, read_json
from .media import file_hash


FPS = 16
TEXT_PROMPT = "video game character"


def box_iou(first, second):
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    area_first = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    area_second = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = area_first + area_second - intersection
    return intersection / union if union else 0.0


def _binding_observations(binding):
    return {row["frame_index"]: row["body_box"] for row in binding["observations"]}


def select_pair(bindings):
    friendly = [row for row in bindings if row["team"] == "friendly"]
    enemy = [row for row in bindings if row["team"] == "enemy"]
    ranked = []
    for ally in friendly:
        for foe in enemy:
            ally_obs, foe_obs = _binding_observations(ally), _binding_observations(foe)
            confirmed = max(ally["confirmation_frame_index"], foe["confirmation_frame_index"])
            common = sorted(index for index in ally_obs.keys() & foe_obs.keys() if index >= confirmed)
            if not common:
                continue
            start = common[0]
            end = min(ally["end_frame_index"], foe["end_frame_index"])
            length = end - start
            ranked.append((length, min(len(ally_obs), len(foe_obs)), ally, foe, start, end,
                           ally_obs[start], foe_obs[start]))
    if not ranked:
        raise ValueError("No stable friendly/enemy champion-bar pair overlaps in this clip")
    ranked.sort(key=lambda row: (-row[0], -row[1], row[2]["bar_box"], row[3]["bar_box"]))
    best = ranked[0]
    ambiguous = len(ranked) > 1 and ranked[1][:2] == best[:2]
    return best, ambiguous, [
        {"common_frames": row[0], "support_frames": row[1],
         "friendly_bar_box": row[2]["bar_box"], "enemy_bar_box": row[3]["bar_box"],
         "seed_frame_index": row[4], "end_frame_index": row[5]}
        for row in ranked
    ]


def prepare(args):
    from PIL import Image, ImageDraw
    from .action_v3 import load_profile
    from .body_campaign import stable_healthbar_bindings
    from .scene_reference import sample_source

    source = Path(args.video).resolve()
    if not source.is_file() or not 0 <= args.start < args.end:
        raise ValueError("Source video and a valid [start, end) interval are required")
    profile = load_profile(args.profile)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_hash = file_hash(source)
    import shutil
    package_source = output / "source" / "annotation"
    package_source.parent.mkdir(parents=True)
    shutil.copytree(Path(__file__).resolve().parents[1], package_source,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    submit_script = Path(__file__).with_name("slurm_sam_multichampion.sl")
    shutil.copy2(submit_script, output / "slurm_sam_multichampion.sl")
    source_files = sorted(path for path in (output / "source").rglob("*") if path.is_file())
    (output / "source" / "SOURCE_SHA256SUMS").write_text("".join(
        f'{file_hash(path)}  {path.relative_to(output / "source").as_posix()}\n'
        for path in source_files))
    frames_dir, sam_dir = output / "frames", output / "inputs-all"
    frames_dir.mkdir()
    sam_dir.mkdir()
    sampled = []
    for index, (pts, image) in enumerate(sample_source(source, args.start, args.end, args.fps)):
        frame_path = frames_dir / f"{index:05d}.png"
        sam_path = sam_dir / f"{index:05d}.jpg"
        image.save(frame_path, format="PNG")
        image.save(sam_path, quality=95, subsampling=0)
        sampled.append({"id": digest([source_hash, pts])[:20], "time": pts,
                        "path": frame_path.relative_to(output).as_posix(),
                        "sha256": file_hash(frame_path),
                        "sam_path": sam_path.relative_to(output).as_posix(),
                        "sam_sha256": file_hash(sam_path), "segment_id": "source-window"})
    if len(sampled) < 4:
        raise ValueError("The selected window needs at least four sampled frames")
    detector_frames = [{**row, "path": str(output / row["path"])} for row in sampled]
    bindings = stable_healthbar_bindings(detector_frames, profile, require_level=True)
    pair, ambiguous, alternatives = select_pair(bindings)
    _, _, ally, foe, seed, end, ally_box, foe_box = pair
    usable_end = min(end, len(sampled))
    if usable_end - seed < 8:
        raise ValueError("The stable two-champion overlap is shorter than eight frames")
    kept = []
    final_sam_dir = output / "inputs"
    final_sam_dir.mkdir()
    for index, source_frame in enumerate(sampled[seed:usable_end]):
        selected_frame = dict(source_frame)
        target_path = final_sam_dir / f"{index:05d}.jpg"
        shutil.copy2(output / source_frame["sam_path"], target_path)
        selected_frame["sam_path"] = target_path.relative_to(output).as_posix()
        selected_frame["sam_sha256"] = file_hash(target_path)
        kept.append(selected_frame)
    shutil.rmtree(sam_dir)
    targets = [
        {"key": "friendly-1", "team": "friendly", "champion": None,
         "box_xyxy": ally_box, "bar_box": ally["bar_box"],
         "binding_frames": ally["evidence_frame_ids"]},
        {"key": "enemy-1", "team": "enemy", "champion": None,
         "box_xyxy": foe_box, "bar_box": foe["bar_box"],
         "binding_frames": foe["evidence_frame_ids"]},
    ]
    selected = {"schema_version": 1, "experimental": True,
        "source_path": str(source), "source_sha256": source_hash,
        "profile": profile, "profile_hash": digest(profile), "fps": args.fps,
        "requested_window": [args.start, args.end], "seed_source_frame_index": seed,
        "seed_time": sampled[seed]["time"], "end_source_frame_index": usable_end,
        "match_group": f"sam-multichampion-{source_hash[:16]}",
        "frames": kept, "targets": targets,
        "candidate_bindings": [{"team": row["team"], "bar_box": row["bar_box"],
                                 "box_xyxy": row["actor_region"],
                                 "confirmation_frame_index": row["confirmation_frame_index"],
                                 "end_frame_index": row["end_frame_index"],
                                 "evidence_frame_ids": row["evidence_frame_ids"]}
                                for row in bindings],
        "pair_alternatives": alternatives,
        "selection_warning": "Multiple equally supported bar pairs; review/edit targets before running" if ambiguous else None,
        "matching_note": "Champion labels are unknown until DINOv2 matching or human review."}
    atomic_json(output / "experiment.json", selected)
    checksums = [f'{file_hash(output / "experiment.json")}  experiment.json',
                 f'{file_hash(output / "slurm_sam_multichampion.sl")}  slurm_sam_multichampion.sl']
    for frame in kept:
        checksums.extend([f'{frame["sha256"]}  {frame["path"]}',
                          f'{frame["sam_sha256"]}  {frame["sam_path"]}'])
    (output / "INPUT_SHA256SUMS").write_text("\n".join(checksums) + "\n")
    first = Image.open(output / kept[0]["path"]).convert("RGB")
    preview = first.copy()
    draw = ImageDraw.Draw(preview)
    for color, target in zip(("cyan", "red"), targets):
        draw.rectangle(target["box_xyxy"], outline=color, width=5)
        draw.rectangle(target["bar_box"], outline="white", width=3)
        draw.text((target["box_xyxy"][0], max(0, target["box_xyxy"][1] - 20)),
                  f'{target["key"]} ({target["team"]})', fill="white", stroke_width=2, stroke_fill="black")
    preview.save(output / "seed-preview.jpg", quality=92)
    return {"output": str(output), "sampled_frames": len(sampled), "tracked_frames": len(kept),
            "seed_time": selected["seed_time"], "targets": targets,
            "selection_warning": selected["selection_warning"]}


def _mask_rows(outputs):
    import numpy as np

    result = []
    for mask, object_id in zip(outputs["out_binary_masks"], outputs["out_obj_ids"], strict=True):
        if hasattr(mask, "detach"):
            mask = mask.detach().cpu().numpy()
        mask = np.asarray(mask).squeeze().astype(bool)
        ys, xs = np.where(mask)
        if len(xs):
            result.append({"object_id": int(object_id), "mask": mask,
                           "box_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                           "pixels": int(mask.sum())})
    return result


def _assign_seed_ids(targets, masks, minimum_iou=0.20):
    if not masks:
        return {target["key"]: None for target in targets}, {target["key"]: 0.0 for target in targets}
    ids = [row["object_id"] for row in masks]
    possibilities = []
    for choice in permutations(ids, min(len(ids), len(targets))):
        assignment = dict(zip([row["key"] for row in targets], choice))
        scores = {target["key"]: next((box_iou(target["box_xyxy"], row["box_xyxy"])
                    for row in masks if row["object_id"] == assignment.get(target["key"])), 0.0)
                  for target in targets}
        possibilities.append((sum(scores.values()), assignment, scores))
    _, assignment, scores = max(possibilities, key=lambda row: row[0])
    for target in targets:
        if scores[target["key"]] < minimum_iou:
            assignment[target["key"]] = None
    return assignment, scores


def _save_frame_target(output, experiment_root, arm, target, frame, object_id, masks, profile, preview=False):
    from PIL import Image

    from .body_identity import healthbar_proposals
    from .body_operational import body_mask

    root = Path(output)
    image_rel = frame["path"]
    with Image.open(experiment_root / image_rel) as im:
        image = im.convert("RGB")
    mask_row = next((row for row in masks if row["object_id"] == object_id), None) if object_id is not None else None
    record = {"arm": arm, "target_key": target["key"], "team": target["team"],
              "champion": target.get("champion"), "frame_index": frame["frame_index"],
              "time": frame["time"], "object_id": object_id, "visible": bool(mask_row),
              "source_frame": image_rel, "source_frame_sha256": frame["sha256"]}
    if not mask_row:
        return record
    raw = Image.fromarray(mask_row["mask"].astype("uint8") * 255)
    if raw.size != image.size:
        raise ValueError("SAM mask geometry differs from the native source frame")
    mask_dir = root / arm / "masks" / target["key"]
    mask_dir.mkdir(parents=True, exist_ok=True)
    mask_path = mask_dir / f'{frame["frame_index"]:05d}.png'
    raw.save(mask_path)
    bars = healthbar_proposals(image, profile)
    clean, removed = body_mask(image, mask_path, profile, bars)
    bbox = clean.getbbox()
    record.update({"mask_path": mask_path.relative_to(root).as_posix(),
                   "mask_sha256": file_hash(mask_path), "mask_box_xyxy": mask_row["box_xyxy"],
                   "mask_pixels": mask_row["pixels"], "ui_removed_fraction": removed})
    if bbox:
        crop_dir = root / arm / "crops" / target["key"]
        crop_dir.mkdir(parents=True, exist_ok=True)
        body_path = crop_dir / f'{frame["frame_index"]:05d}-image.png'
        crop_mask_path = crop_dir / f'{frame["frame_index"]:05d}-mask.png'
        image.crop(bbox).save(body_path)
        clean.crop(bbox).save(crop_mask_path)
        record.update({"image_path": body_path.relative_to(root).as_posix(),
                       "crop_mask_path": crop_mask_path.relative_to(root).as_posix(),
                       "image_sha256": file_hash(body_path),
                       "crop_mask_sha256": file_hash(crop_mask_path),
                       "body_box_xyxy": list(bbox)})
    if preview:
        from PIL import ImageDraw
        overlay = image.copy()
        color = (0, 225, 255) if target["team"] == "friendly" else (255, 80, 80)
        alpha = Image.fromarray(mask_row["mask"].astype("uint8") * 96)
        overlay.paste(color, (0, 0, overlay.width, overlay.height), alpha)
        draw = ImageDraw.Draw(overlay)
        draw.rectangle(target["box_xyxy"], outline="white", width=3)
        draw.text((target["box_xyxy"][0], target["box_xyxy"][1]), target["key"], fill="white",
                  stroke_width=2, stroke_fill="black")
        overlay.save(root / arm / f'frame-{frame["frame_index"]:05d}-{target["key"]}.jpg', quality=90)
    return record


def _run_session(predictor, experiment, targets, arm, output, *, with_text=False, stop=lambda: None):
    from PIL import Image

    root = Path(output)
    frames = experiment["frames"]
    experiment_root = Path(experiment["_root"])
    session = predictor.handle_request({"type": "start_session", "resource_path": str(experiment_root / "inputs")})["session_id"]
    try:
        first = frames[0]
        with Image.open(experiment_root / first["path"]) as image:
            width, height = image.size
        boxes = [[target["box_xyxy"][0] / width, target["box_xyxy"][1] / height,
                  (target["box_xyxy"][2] - target["box_xyxy"][0]) / width,
                  (target["box_xyxy"][3] - target["box_xyxy"][1]) / height]
                 for target in targets]
        seed_request = {"type": "add_prompt", "session_id": session, "frame_index": 0,
                        "bounding_boxes": boxes, "bounding_box_labels": [1] * len(boxes)}
        if with_text:
            seed_request["text"] = TEXT_PROMPT
        seed = predictor.handle_request(seed_request)
        seed_masks = _mask_rows(seed["outputs"])
        mapping, seed_ious = _assign_seed_ids(targets, seed_masks)
        records, frame_reports, error = [], [], None
        expected_indices = set(range(len(frames)))
        try:
            for response in predictor.handle_stream_request({
                "type": "propagate_in_video", "session_id": session,
                "start_frame_index": 0, "propagation_direction": "forward"}):
                stop()
                index = int(response["frame_index"])
                if not 0 <= index < len(frames):
                    raise ValueError("SAM returned a frame outside the prepared clip")
                output_masks = _mask_rows(response["outputs"])
                frame = {**frames[index], "frame_index": index}
                for target in targets:
                    object_id = mapping[target["key"]]
                    record = _save_frame_target(root, experiment_root, arm, target, frame, object_id, output_masks,
                                                experiment["profile"],
                                                preview=index in {0, len(frames) // 2, len(frames) - 1})
                    records.append(record)
                frame_reports.append({"frame_index": index,
                                      "object_ids": {target["key"]: mapping[target["key"]] for target in targets},
                                      "all_masks": [{k: v for k, v in row.items() if k != "mask"}
                                                    for row in output_masks]})
        except RuntimeError as exc:
            if str(exc) != "No points are provided; please add points first":
                raise
            error = str(exc)
        returned = {row["frame_index"] for row in frame_reports}
        complete = returned == expected_indices
        return {"status": "propagation_failed" if error else "completed" if complete else "incomplete",
                "error": error,
                "seed_objects": [{k: v for k, v in row.items() if k != "mask"} for row in seed_masks],
                "seed_target_iou": seed_ious, "target_object_ids": mapping,
                "frames_expected": len(frames), "frames_returned": len(frame_reports),
                "frame_indices_complete": complete,
                "frames": frame_reports, "samples": records}
    finally:
        predictor.handle_request({"type": "close_session", "session_id": session})


def run(args):
    import torch
    from sam3.model_builder import build_sam3_multiplex_video_predictor
    from .local_models import checkpoint
    from .sam_compat import adapt_session_init, load_verified_state

    experiment_path = Path(args.experiment).resolve()
    experiment = read_json(experiment_path)
    experiment["_root"] = str(experiment_path.parent)
    if experiment.get("selection_warning") and not args.allow_ambiguous:
        raise ValueError("Prepared bar pair is ambiguous; review targets, then rerun with --allow-ambiguous")
    if len(experiment["targets"]) < 2:
        raise ValueError("The multi-champion pilot needs two targets")
    root = experiment_path.parent
    for frame in experiment["frames"]:
        for key, hash_key in (("path", "sha256"), ("sam_path", "sam_sha256")):
            path = root / frame[key]
            if file_hash(path) != frame[hash_key]:
                raise ValueError(f"Prepared {key} changed: {path}")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    cache = Path(args.cache).resolve()
    model_root, pin = checkpoint(cache, "tracking")
    weights = model_root / "sam3.1_multiplex.pt"
    predictor = build_sam3_multiplex_video_predictor(checkpoint_path=str(weights), max_num_objects=16,
        use_fa3=False, use_rope_real=False, compile=False, warm_up=False)
    state = torch.load(str(weights), map_location="cpu", weights_only=True, mmap=True)
    model_load = load_verified_state(predictor.model, state)
    del state
    model_load.update(adapt_session_init(predictor.model))
    started = time.monotonic()
    arms = {}
    for arm, joint, with_text in (("independent-boxes", False, False),
                                  ("joint-boxes", True, False),
                                  ("joint-boxes-text", True, True)):
        if joint:
            result = _run_session(predictor, experiment, experiment["targets"], arm, output,
                                  with_text=with_text)
            arms[arm] = result
        else:
            # Separate sessions form the single-target baseline.
            arm_results = []
            for target in experiment["targets"]:
                result = _run_session(predictor, experiment, [target], arm, output)
                arm_results.append(result)
            arms[arm] = {"status": "completed" if all(r["status"] == "completed" for r in arm_results)
                         else "propagation_failed", "target_runs": arm_results,
                         "samples": [sample for row in arm_results for sample in row["samples"]]}
        atomic_json(output / "report.json", {"job_id": __import__("os").environ.get("SLURM_JOB_ID"),
            "experimental": True, "model": pin, "model_load": model_load,
            "source_sha256": experiment["source_sha256"], "seed_time": experiment["seed_time"],
            "targets": experiment["targets"], "arms": arms, "complete": False})
    all_samples = [sample for result in arms.values()
                   for sample in (result.get("samples", []) if isinstance(result, dict) else [])]
    atomic_json(output / "dino-samples.json", all_samples)
    report = {"job_id": __import__("os").environ.get("SLURM_JOB_ID"), "experimental": True,
              "model": pin, "model_load": model_load, "source_sha256": experiment["source_sha256"],
              "seed_time": experiment["seed_time"], "targets": experiment["targets"], "arms": arms,
              "dino_samples": "dino-samples.json", "dino_sample_count": len(all_samples),
              "wall_seconds": time.monotonic() - started,
              "complete": all(result.get("status") == "completed" for result in arms.values()),
              "mask_note": "Native binary SAM masks and bar-cleaned DINOv2 image/mask crops are saved per target and frame."}
    atomic_json(output / "report.json", report)
    return report


def match_dino(args):
    import numpy as np
    from PIL import Image
    from .body_identity import BodyGallery, FrozenEncoder, unit

    root = Path(args.results).resolve()
    samples = read_json(root / "dino-samples.json")
    raw_gallery = read_json(Path(args.gallery))
    if isinstance(raw_gallery, dict):
        raw_gallery = raw_gallery.get("gallery", raw_gallery.get("entries", []))
    experiment_path = Path(args.experiment).resolve()
    experiment = read_json(experiment_path)
    references = [row for row in raw_gallery if row.get("source_sha256") != experiment["source_sha256"]]
    if not references:
        raise ValueError("No reference gallery remains after excluding the pilot source video")
    policy = read_json(Path(args.policy))
    encoder = FrozenEncoder(args.encoder_lock, args.device)
    gallery_vectors = []
    for row in references:
        image_path, mask_path = Path(row["path"]), Path(row["mask_path"])
        if file_hash(image_path) != row["sha256"] or file_hash(mask_path) != row["mask_sha256"]:
            raise ValueError("DINOv2 gallery image or mask hash changed")
        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            gallery_vectors.append(encoder.encode(image.convert("RGB"), mask.convert("L")))
    gallery = BodyGallery(references, gallery_vectors, encoder_signature=encoder.signature,
                          excluded_groups={experiment.get("match_group")})
    by_track = {}
    per_sample = []
    for row in samples:
        if not row.get("visible") or not row.get("image_path") or not row.get("crop_mask_path"):
            continue
        body_path, crop_mask_path = root / row["image_path"], root / row["crop_mask_path"]
        if (file_hash(body_path) != row["image_sha256"]
                or file_hash(crop_mask_path) != row["crop_mask_sha256"]):
            raise ValueError("DINOv2 body crop or foreground mask changed")
        with Image.open(body_path) as image, Image.open(crop_mask_path) as mask:
            vector = encoder.encode(image.convert("RGB"), mask.convert("L"))
        prediction = gallery.match(vector, policy)
        per_sample.append({"arm": row["arm"], "target_key": row["target_key"], "team": row["team"],
                           "time": row["time"], "object_id": row["object_id"], **prediction})
        by_track.setdefault((row["arm"], row["target_key"], row["team"]), []).append(vector)
    pooled = []
    for (arm, target_key, team), vectors in sorted(by_track.items()):
        prediction = gallery.match(unit(np.mean(vectors, axis=0)), policy)
        pooled.append({"arm": arm, "target_key": target_key, "team": team,
                       "visible_frames_encoded": len(vectors), **prediction})
    report = {"experimental": True, "encoder_signature": encoder.signature,
              "source_sha256_excluded_from_gallery": experiment["source_sha256"],
              "gallery_references": len(references), "score_is_probability": False,
              "per_frame": per_sample, "pooled_tracks": pooled,
              "note": "DINOv2 similarity is appearance evidence; only the frozen policy may accept a name."}
    atomic_json(root / "dino-match.json", report)
    return {"path": str(root / "dino-match.json"), "samples": len(per_sample),
            "pooled_tracks": pooled}


def compare_skins(args):
    from collections import Counter

    import numpy as np

    from .body_frame_matching import SkinMatcher
    from .body_operational import verified_crop

    root = args.results.resolve()
    report_path = root / "report.json"
    report = read_json(report_path)
    if not report.get("complete") or args.arm not in report["arms"]:
        raise ValueError("Choose a completed SAM arm from the results report")
    samples = [row for row in report["arms"][args.arm]["samples"]
               if row["target_key"] == args.target and row.get("visible")
               and row.get("image_path") and row.get("crop_mask_path")]
    if not samples or len({row["frame_index"] for row in samples}) != len(samples):
        raise ValueError("Choose a SAM target with distinct visible, masked body crops")
    samples.sort(key=lambda row: row["frame_index"])
    queries = [(root / row["image_path"], root / row["crop_mask_path"], row["image_sha256"], row["crop_mask_sha256"])
               for row in samples]
    frames = []
    for row, (_, mask_path, _, mask_sha) in zip(samples, queries, strict=True):
        mask = verified_crop(mask_path, mask_sha).convert("L").point(lambda value: 255 if value > 127 else 0)
        box = mask.getbbox()
        if box is None:
            raise ValueError(f"Empty body mask for frame {row['frame_index']}")
        foreground_pixels = mask.histogram()[255]
        frames.append({**{key: row[key] for key in ("frame_index", "time", "image_sha256", "crop_mask_sha256")},
                       "foreground_pixels": foreground_pixels,
                       "foreground_fraction": foreground_pixels / ((box[2] - box[0]) * (box[3] - box[1]))})
    gallery = [row for row in read_json(args.gallery)
               if row.get("source_sha256") != report["source_sha256"]
               and (not args.champion or row["champion"] in args.champion)]
    comparison = {"experimental": True, "scores_are_probabilities": False, "source_sha256": report["source_sha256"],
                  "sam_report_sha256": file_hash(report_path), "gallery_sha256": file_hash(args.gallery),
                  "arm": args.arm, "target": args.target, "runs": {},
                  "aggregation": "Maximum view cosine per skin per frame, then mean across visible frames",
                  "frames": frames}
    selected = None
    if args.top_foreground_frames is not None:
        by_area = sorted(range(len(frames)), key=lambda index: -frames[index]["foreground_pixels"])
        selected = sorted(by_area[:args.top_foreground_frames])
        comparison["top_foreground_frame_indices"] = [frames[index]["frame_index"] for index in selected]
    for name, lock in (("dinov2", args.encoder_lock), ("siglip", args.siglip_lock)):
        pin = read_json(lock)
        matcher = SkinMatcher(gallery, args.cache, lock, digest(pin), None, encoder_name=name)
        entries, labels, starts = [], [], []
        for champion, skins in sorted(matcher.entries.items()):
            for skin, views in sorted(skins.items()):
                labels.append({"champion": champion, "skin_id": skin, "skin_name": views[0].get("skin_name"),
                               "views": len(views)})
                starts.append(len(entries))
                entries.extend(views)
        if not entries or set(args.champion or ()) - matcher.entries.keys():
            raise ValueError("Each requested champion needs audited 3D skin renders in the gallery")
        scores = np.maximum.reduceat(matcher.score_views(queries, entries), starts, axis=1)
        comparison["gallery_views"] = len(entries)
        comparison["champions"] = sorted(matcher.entries)
        run = {"encoder_signature": matcher.encoder.signature, "encoder_lock_sha256": digest(pin),
               "new_embeddings": dict(matcher.encoded)}
        rankings = {"ranked_skins": scores}
        if selected is not None:
            rankings["ranked_skins_top_foreground"] = scores[selected]
        for key, frame_scores in rankings.items():
            wins = Counter(frame_scores.argmax(axis=1).tolist())
            ranked = [{**label, "mean_best_view_cosine": float(values.mean()),
                       "winning_frames": wins[index], "per_frame_cosine": values.tolist()}
                      for index, (label, values) in enumerate(zip(labels, frame_scores.T, strict=True))]
            run[key] = sorted(ranked, key=lambda row: -row["mean_best_view_cosine"])
        comparison["runs"][name] = run
    atomic_json(args.output, comparison)
    receipt = {"path": str(args.output), "frames": len(samples), "gallery_views": comparison["gallery_views"],
               "top_skin": {name: run["ranked_skins"][0]["skin_id"] for name, run in comparison["runs"].items()}}
    if selected is not None:
        receipt["top_skin_by_foreground"] = {
            name: run["ranked_skins_top_foreground"][0]["skin_id"] for name, run in comparison["runs"].items()}
    return receipt


def main(argv=None):
    from .sam_appearances import DEFAULT_DINO_LOCK, DEFAULT_SIGLIP_LOCK

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare", help="Sample full frames and bind one stable target per team")
    prepare_parser.add_argument("--video", type=Path, required=True)
    prepare_parser.add_argument("--profile", type=Path, required=True)
    prepare_parser.add_argument("--start", type=float, default=39.9)
    prepare_parser.add_argument("--end", type=float, default=41.5)
    prepare_parser.add_argument("--fps", type=int, default=FPS)
    prepare_parser.add_argument("--output", type=Path, required=True)
    run_parser = commands.add_parser("run", help="Run independent and simultaneous box-seeded SAM arms")
    run_parser.add_argument("--experiment", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--cache", type=Path, required=True)
    run_parser.add_argument("--allow-ambiguous", action="store_true")
    match_parser = commands.add_parser("match-dino", help="Embed saved body crops and rank an approved DINOv2 gallery")
    match_parser.add_argument("--experiment", type=Path, required=True)
    match_parser.add_argument("--results", type=Path, required=True)
    match_parser.add_argument("--encoder-lock", type=Path, required=True)
    match_parser.add_argument("--gallery", type=Path, required=True)
    match_parser.add_argument("--policy", type=Path, required=True)
    match_parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    compare = commands.add_parser("compare-skins", help="Compare DINOv2 and SigLIP on the same SAM crops and 3D skin renders")
    compare.add_argument("--results", type=Path, required=True)
    compare.add_argument("--arm", required=True)
    compare.add_argument("--target", required=True)
    compare.add_argument("--gallery", type=Path, required=True, help="Audited 3D render gallery JSON")
    compare.add_argument("--champion", action="append", help="Restrict the gallery to this candidate champion (repeatable)")
    compare.add_argument("--top-foreground-frames", type=int, metavar="N",
                         help="Also rank skins on up to N frames with the most foreground mask pixels")
    compare.add_argument("--encoder-lock", type=Path, required=True, help="DINOv2 encoder lock")
    compare.add_argument("--siglip-lock", type=Path, required=True, help="SigLIP model lock")
    compare.add_argument("--cache", type=Path, default=Path("data/league_assets"))
    compare.add_argument("--output", type=Path, required=True)
    prepare_appearances = commands.add_parser("prepare-appearances", help="Scan scoreboard-backed gameplay for anonymous SAM seeds")
    prepare_appearances.add_argument("--root", type=Path, required=True)
    prepare_appearances.add_argument("--video", type=Path, required=True)
    prepare_appearances.add_argument("--scoreboard-skim", type=Path, required=True)
    prepare_appearances.add_argument("--profile", type=Path, required=True)
    prepare_appearances.add_argument("--player", required=True)
    prepare_appearances.add_argument("--player-offscreen", type=Path)
    prepare_appearances.add_argument("--proposals-from", type=Path,
                                     help="Reuse raw health-bar proposals and lossless frames from an earlier scan in this root")
    prepare_appearances.add_argument("--start", type=float, required=True)
    prepare_appearances.add_argument("--end", type=float, required=True)
    prepare_appearances.add_argument("--siglip-lock", type=Path, default=DEFAULT_SIGLIP_LOCK)
    prepare_appearances.add_argument("--patch", nargs=2, metavar=("DATA_DRAGON", "COMMUNITY_DRAGON"),
                                     help="Use these pinned reference versions for a separate match")
    mask = commands.add_parser("mask-appearances", help="Run resumable bidirectional SAM propagation")
    mask.add_argument("--root", type=Path, required=True)
    mask.add_argument("--cache", type=Path, required=True)
    mask.add_argument("--smoke", action="store_true")
    mask.add_argument("--require-smoke", action="store_true")
    mask.add_argument("--shard", type=int, default=0)
    mask.add_argument("--shards", type=int, default=1)
    match = commands.add_parser("match-appearances", help="Rank up to five SAM masks per appearance and confirm skins")
    match.add_argument("--root", type=Path, required=True)
    match.add_argument("--gallery", type=Path, required=True)
    match.add_argument("--scoreboard-skim", type=Path, help="Relocated copy of the pinned scoreboard evidence")
    match.add_argument("--cache", type=Path, default=Path("data/league_assets"))
    match.add_argument("--encoder", choices=("siglip", "dinov2"), default="siglip")
    match.add_argument("--siglip-lock", type=Path, default=DEFAULT_SIGLIP_LOCK)
    match.add_argument("--encoder-lock", type=Path, default=DEFAULT_DINO_LOCK)
    match.add_argument("--require-complete", action="store_true", help="Fail the command if any SAM appearances are pending or failed")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args)
    elif args.command == "run":
        result = run(args)
    elif args.command == "compare-skins":
        if args.top_foreground_frames is not None and args.top_foreground_frames < 1:
            compare.error("--top-foreground-frames must be positive")
        result = compare_skins(args)
    elif args.command in {"prepare-appearances", "mask-appearances", "match-appearances"}:
        from . import sam_appearances

        result = getattr(sam_appearances, args.command.replace("-", "_"))(args)
    else:
        result = match_dino(args)
    print(json.dumps(result, indent=2))
    if args.command == "mask-appearances" and (result["statuses"].get("failed") or
                                               (args.smoke and not result.get("smoke_passed"))):
        raise SystemExit(1)
    if args.command == "match-appearances" and args.require_complete and not result["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
