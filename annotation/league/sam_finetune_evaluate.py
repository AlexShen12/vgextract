"""Paired held-out image and fixed-tracker comparisons; no online corrections."""
from __future__ import annotations

import gc
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw

from .core import atomic_json, read_json
from .media import file_hash
from .sam_finetune import checked_image, instance_mask, rle_decode, seed_tracker, verify_labels
from .sam_finetune_metrics import associate_bar, iou, boundary_f
from .sam_finetune_train import load_model, model_lock, score_split


def overlay(image, masks, dest):
    canvas = image.convert("RGBA")
    for mask, color in zip(masks, [(255,70,40), (30,220,255), (170,240,60)]):
        tint = np.zeros((image.height, image.width, 4), dtype=np.uint8)
        tint[mask] = [*color, 100]
        canvas = Image.alpha_composite(canvas, Image.fromarray(tint))
    dest.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(dest)


def track_clip(predictor, root, clip, seed, mask, arm):
    import torch
    from .sam_multichampion import _mask_rows
    folder = root / "evaluation" / "tracking" / arm / clip["id"]
    folder.mkdir(parents=True, exist_ok=True)
    frames = clip["frames"]
    for frame in frames:
        if file_hash(root / frame["path"]) != frame["path_sha256"]:
            raise ValueError("Tracking source frame changed")
    video_dir = (root / frames[0]["path"]).parent
    start = time.monotonic()
    session = predictor.handle_request({"type": "start_session", "resource_path": str(video_dir),
                                        "offload_video_to_cpu": True})["session_id"]
    outputs, error = {}, None
    try:
        response = seed_tracker(predictor, session, seed["frame_index"], 1, mask,
                                seed["crop_box"], seed["source_size"])

        def save(response):
            index = int(response["frame_index"])
            if not 0 <= index < len(frames):
                raise ValueError("Unexpected propagated frame index")
            rows = _mask_rows(response["outputs"])
            predicted = next((r["mask"] for r in rows if r["object_id"] == 1), None)
            record = {"frame_index": index, "visible": predicted is not None and bool(predicted.any())}
            if record["visible"]:
                if predicted.shape != (seed["source_size"][1], seed["source_size"][0]):
                    raise ValueError("Tracker changed source geometry")
                dest = folder / f"{index:06}.png"
                Image.fromarray(predicted.astype(np.uint8)*255).save(dest)
                record.update(mask_path=str(dest.relative_to(root)), mask_sha256=file_hash(dest))
            outputs[index] = record
        save(response)
        for direction in ("forward", "backward"):
            for response in predictor.handle_stream_request({"type": "propagate_in_video", "session_id": session,
                "start_frame_index": seed["frame_index"], "propagation_direction": direction}):
                save(response)
        torch.cuda.synchronize()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        predictor.handle_request({"type": "close_session", "session_id": session})
    result = {"status": "completed" if len(outputs) == len(frames) and error is None else "incomplete",
              "expected_frames": len(frames), "returned_frames": len(outputs), "error": error,
              "tracking_seconds": time.monotonic()-start,
              "frames": [outputs[i] for i in sorted(outputs)]}
    atomic_json(folder / "track.json", result)
    return result


def score_track(root, clip, target_id, result, image_rows, reviews, arm):
    frames = {r["frame_index"]: r for r in result["frames"]}
    scored = []
    for row in image_rows:
        if row.get("clip_id") != clip["id"]:
            continue
        review = reviews["images"][row["id"]]
        if review["status"] != "accepted":
            continue
        target = next((i for i in review["instances"] if i["track_id"] == target_id), None)
        if target is None:
            continue
        truth = instance_mask(root, row, target)
        output = frames.get(row["frame_index"])
        prediction = np.zeros_like(truth)
        if output and output["visible"]:
            path = root / output["mask_path"]
            if file_hash(path) != output["mask_sha256"]:
                raise ValueError("Tracking mask changed")
            with Image.open(path) as image:
                prediction = np.asarray(image.crop(row["crop_box"])) > 0
        overlap = iou(truth, prediction)
        others = [instance_mask(root, row, i) for i in review["instances"] if i["track_id"] != target_id]
        other_overlap = max((iou(m, prediction) for m in others), default=0.)
        scored.append({"sample_id": row["id"], "iou": overlap, "boundary_f": boundary_f(truth, prediction),
                       "missing": not prediction.any(), "drift": overlap < .5,
                       "identity_switch": other_overlap > max(.2, overlap),
                       "contamination": float((prediction & ~truth).sum()) / max(1,int(prediction.sum()))})
        with checked_image(root, row) as image:
            overlay(image, [truth, prediction], root / "evaluation/overlays" / arm / (row["id"] + ".png"))
    return {"reference_frames": len(scored), "mean_iou": float(np.mean([r["iou"] for r in scored])) if scored else None,
            "missing_masks": sum(r["missing"] for r in scored), "drift_frames": sum(r["drift"] for r in scored),
            "identity_switch_frames": sum(r["identity_switch"] for r in scored), "rows": scored}


def evaluate(args):
    import torch
    verify_labels(args.root)
    completion = read_json(args.root / "training/completed.json")
    if completion["status"] != "completed":
        raise ValueError("Training has not completed")
    root = args.root
    output = root / "evaluation"
    report = {"schema": 1, "status": "running", "labels_sha256": file_hash(root / "labels.lock.json"),
              "base": model_lock(root), "best_checkpoint_sha256": file_hash(root / "training/best.pt"),
              "correction_policy": "none", "reviewer_kind": "agent", "image": {}, "tracking": {},
              "model_load_seconds": {}}
    torch.cuda.reset_peak_memory_stats()
    for arm in ("baseline", "finetuned"):
        start = time.monotonic()
        model = load_model(args, root / "training/best.pt" if arm == "finetuned" else None)
        torch.cuda.synchronize()
        report["model_load_seconds"][arm] = time.monotonic()-start
        report["image"][arm] = score_split(model, root, "test", output / "predictions" / arm)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        atomic_json(output / "scorecard.json", report)
    from .local_models import checkpoint
    from .sam_compat import load_multiplex_predictor
    folder, pin = checkpoint(args.tracking_cache, "tracking")
    report["tracker"] = pin
    start = time.monotonic()
    predictor, load = load_multiplex_predictor(folder / "sam3.1_multiplex.pt")
    torch.cuda.synchronize()
    report["model_load_seconds"]["tracker"] = time.monotonic()-start
    report["tracker_load"] = load
    dataset = read_json(root / "dataset.json")
    reviews = read_json(root / "reviews.json")
    for clip in dataset["clips"]:
        if clip["split"] != "test":
            continue
        refs = sorted((r for r in dataset["images"] if r.get("clip_id") == clip["id"]), key=lambda r:r["frame_index"])
        target_id = "target"
        seeds = [(r, next((i for i in reviews["images"][r["id"]]["instances"] if i["track_id"] == target_id), None))
                 for r in refs if reviews["images"][r["id"]]["status"] == "accepted"]
        seeds = [(r,i) for r,i in seeds if i is not None]
        if not seeds:
            raise ValueError("Each held-out clip needs a reviewed target")
        seed, instance = seeds[0]
        if "bar_box" not in instance:
            raise ValueError("Seed target requires a reviewed crop-coordinate health-bar box")
        report["tracking"][clip["id"]] = {}
        for arm in ("baseline", "finetuned", "reviewed"):
            if arm == "reviewed":
                mask, seeding_seconds = instance_mask(root, seed, instance), 0.
            else:
                saved = read_json(output / "predictions" / arm / (seed["id"] + ".json"))
                masks = [rle_decode(p["mask"]) for p in saved["predictions"]]
                chosen = associate_bar(masks, instance["bar_box"])
                mask = masks[chosen] if chosen is not None else None
                seeding_seconds = saved["seconds"]
            if mask is None:
                result = {"status": "seed_rejected", "expected_frames": len(clip["frames"]), "returned_frames": 0,
                          "tracking_seconds": 0., "frames": []}
            else:
                result = track_clip(predictor, root, clip, seed, mask, arm)
            result["metrics"] = score_track(root, clip, target_id, result, refs, reviews, arm)
            result["seeding_seconds"] = seeding_seconds
            result["total_seconds"] = seeding_seconds + result["tracking_seconds"]
            report["tracking"][clip["id"]][arm] = {k:v for k,v in result.items() if k != "frames"}
            atomic_json(output / "scorecard.json", report)
    report["peak_gpu_bytes"] = torch.cuda.max_memory_allocated()
    report["status"] = "completed" if all(r["status"] in ("completed", "seed_rejected")
        for arms in report["tracking"].values() for r in arms.values()) else "incomplete"
    atomic_json(output / "scorecard.json", report)
    lines = ["# League SAM 3 fine-tuning pilot", "", "Exploratory evaluation against agent-reviewed labels.", "",
             "| Segmenter | Mean instance IoU | Recall @ .5 | Negative FP crops |", "|---|---:|---:|---:|"]
    for arm, row in report["image"].items():
        lines.append(f"| {arm} | {row['mean_instance_iou']:.3f} | {row['target_recall_at_50']:.3f} | {row['negative_false_positive_images']}/{row['negative_images']} |")
    lines += ["", "Tracking details, drift, identity switches, missing masks, frame completion, runtime and memory are in scorecard.json.",
              "One held-out match cannot establish broad generalization. The production checkpoint is unchanged.", ""]
    (output / "report.md").write_text("\n".join(lines))
    return report
