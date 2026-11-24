"""Trial box prompts on short gameplay crops using the pinned SAM model."""

import argparse
import json
import os
from pathlib import Path
import time


def box_iou(first, second):
    left, top = (max(first[i], second[i]) for i in (0, 1))
    right, bottom = (min(first[i], second[i]) for i in (2, 3))
    intersection = max(0, right - left) * max(0, bottom - top)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / (first_area + second_area - intersection)


def inspect_frame(outputs, source, preview_path=None, prompt_box=None):
    import numpy as np
    from PIL import Image, ImageDraw

    objects = []
    preview = Image.open(source).convert("RGB") if preview_path else None
    colors = ((0, 235, 255), (255, 105, 100), (125, 245, 90), (255, 185, 55))
    for mask, object_id in zip(outputs["out_binary_masks"], outputs["out_obj_ids"], strict=True):
        mask = np.asarray(mask, dtype=bool).squeeze()
        rows, columns = np.where(mask)
        if not len(rows):
            continue
        if preview and mask.shape != (preview.height, preview.width):
            raise ValueError("SAM mask geometry does not match the input crop")
        box = [int(columns.min()), int(rows.min()), int(columns.max()) + 1, int(rows.max()) + 1]
        object_id = int(object_id)
        objects.append({"id": object_id, "box_xyxy": box, "pixels": len(rows)})
        if preview:
            color = colors[object_id % len(colors)]
            alpha = Image.fromarray(mask.astype("uint8") * 92)
            preview.paste(color, (0, 0, preview.width, preview.height), alpha)
            draw = ImageDraw.Draw(preview)
            draw.rectangle(box, outline=color, width=2)
            draw.text((box[0] + 2, box[1] + 2), str(object_id), fill="white", stroke_width=2, stroke_fill="black")
    if preview:
        if prompt_box:
            ImageDraw.Draw(preview).rectangle(prompt_box, outline="white", width=2)
        preview.save(preview_path)
    return objects


def trial(predictor, case, mode, inputs, output):
    from PIL import Image

    directory = inputs / case["clip"]
    frames = sorted(directory.glob("*.jpg"))
    with Image.open(frames[0]) as first:
        width, height = first.size
    box = case["box_xyxy"]
    left, top, right, bottom = box
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError(f"Prompt box outside the input crop for {case['id']}")
    normalized = [left / width, top / height, (right - left) / width, (bottom - top) / height]
    session = predictor.handle_request({"type": "start_session", "resource_path": str(directory)})["session_id"]
    try:
        request = {
            "type": "add_prompt", "session_id": session, "frame_index": 0,
            "bounding_boxes": [normalized], "bounding_box_labels": [1],
        }
        if mode == "box-and-text":
            request["text"] = case["text"]
        seed = predictor.handle_request(request)
        seed_objects = inspect_frame(seed["outputs"], frames[0], output / "prompt.jpg", box)
        for obj in seed_objects:
            obj["prompt_box_iou"] = box_iou(box, obj["box_xyxy"])
        best = max(seed_objects, key=lambda obj: obj["prompt_box_iou"], default=None)
        target = best["id"] if best and best["prompt_box_iou"] >= .2 else None
        results = []
        error = None
        try:
            for response in predictor.handle_stream_request({
                "type": "propagate_in_video", "session_id": session, "start_frame_index": 0,
                "propagation_direction": "forward",
            }):
                index = response["frame_index"]
                preview_path = output / f"frame-{index:02d}.jpg" if index in (0, 4, 8, 16, len(frames) - 1) else None
                objects = inspect_frame(response["outputs"], frames[index], preview_path, box if index == 0 else None)
                results.append({"frame_index": index, "objects": objects})
        except RuntimeError as exc:
            if str(exc) != "No points are provided; please add points first":
                raise
            error = str(exc)
        if error is None and [row["frame_index"] for row in results] != list(range(len(frames))):
            raise ValueError("SAM did not return the expected frame sequence")
        return {
            "status": "propagation_failed" if error else "completed", "error": error,
            "frames_expected": len(frames),
            "seed_objects": seed_objects, "target_id": target, "frames": results,
            "target_frames_after_seed": [
                row["frame_index"] for row in results[1:] if any(obj["id"] == target for obj in row["objects"])
            ],
        }
    finally:
        predictor.handle_request({"type": "close_session", "session_id": session})


def main():
    import torch
    from sam3.model_builder import build_sam3_multiplex_video_predictor

    from annotation.league.core import atomic_json
    from annotation.league.local_models import checkpoint
    from annotation.league.sam_compat import adapt_session_init, load_verified_state

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.manifest.read_text())
    torch.manual_seed(17)
    started = time.monotonic()
    root, pin = checkpoint(args.cache, "tracking")
    weights = root / "sam3.1_multiplex.pt"
    predictor = build_sam3_multiplex_video_predictor(
        checkpoint_path=str(weights), max_num_objects=16, use_fa3=False,
        use_rope_real=False, compile=False, warm_up=False,
    )
    state = torch.load(str(weights), map_location="cpu", weights_only=True, mmap=True)
    model_load = load_verified_state(predictor.model, state)
    del state
    model_load.update(adapt_session_init(predictor.model))
    report = {
        "job_id": os.environ.get("SLURM_JOB_ID"), "gpu": torch.cuda.get_device_name(0),
        "checkpoint": pin, "model_load": model_load, "manifest": manifest, "cases": {},
    }
    for case in manifest["cases"]:
        report["cases"][case["id"]] = {}
        for mode in ("box", "box-and-text"):
            output = args.output / case["id"] / mode
            output.mkdir(parents=True)
            result = trial(predictor, case, mode, args.inputs, output)
            report["cases"][case["id"]][mode] = result
            atomic_json(args.output / "report.json", report)
            print(case["id"], mode, result["status"], "seed objects", len(result["seed_objects"]),
                  "target frames after seed", len(result["target_frames_after_seed"]), flush=True)
    report["wall_seconds"] = time.monotonic() - started
    report["peak_gpu_bytes"] = torch.cuda.max_memory_allocated()
    atomic_json(args.output / "report.json", report)


if __name__ == "__main__":
    main()
