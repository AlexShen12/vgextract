"""Auditable League SAM 3 dataset and experiment commands (CPU-safe imports).

Nothing in this module promotes proposals to reviewed labels automatically.
The production model lock and existing appearance receipts are never modified.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

from .core import atomic_json, digest, read_json
from .media import file_hash, probe

CODE_REVISION = "f1647e8d0c1d280991f4d1b7d974c2704537fe66"
PROMPT = "League of Legends champion"
DEFAULT_ROOT = Path("data/league_recognition_v1/sam3-finetune-v1")
DEFAULT_SOURCES = Path(__file__).with_name("sam_finetune_sources.json")
SPLITS = ("train", "validation", "test")
FPS = 16


def safe_path(root, relative):
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("Artifact path escapes experiment root")
    return path


def checked_image(root, row, key="path"):
    path = safe_path(root, row[key])
    if file_hash(path) != row[key + "_sha256"]:
        raise ValueError(f"Changed image: {path}")
    return Image.open(path)


def validate_sources(sources, *, reviewed=False):
    seen_ids, seen_matches, seen_hashes = set(), set(), set()
    for row in sources:
        if row["split"] not in SPLITS or row["provider"] not in ("midbeast", "domisum"):
            raise ValueError("Invalid source split or provider")
        if row["video_id"] in seen_ids or row["match_id"] in seen_matches:
            raise ValueError("Duplicate video or underlying match")
        seen_ids.add(row["video_id"])
        seen_matches.add(row["match_id"])
        if row.get("source_sha256"):
            if row["source_sha256"] in seen_hashes:
                raise ValueError("Identical source content across matches")
            seen_hashes.add(row["source_sha256"])
        if reviewed and (row.get("gameplay_review") != "accepted" or not row.get("match_evidence")):
            raise ValueError("Source needs gameplay and underlying-match review")
    if len(sources) != 6 or Counter(r["provider"] for r in sources) != {"midbeast": 3, "domisum": 3}:
        raise ValueError("Pilot requires three Midbeast and three Domisum matches")
    if Counter(r["split"] for r in sources) != {"train": 4, "validation": 1, "test": 1}:
        raise ValueError("Pilot split must be four/one/one whole matches")


def collect(args):
    from ingestion.youtube.download_videos import main as download
    sources = read_json(args.sources)["sources"]
    validate_sources(sources)
    args.root.mkdir(parents=True, exist_ok=True)
    failures = []
    for row in sources:
        directory = args.video_root / ("lol_midbeast" if row["provider"] == "midbeast"
                                       else "lol_domisum_" + row["champion"].lower())
        video = directory / (row["video_id"] + ".mp4")
        if not video.exists():
            status = download(["--source", "https://www.youtube.com/watch?v=" + row["video_id"],
                               "--output-dir", str(directory), "--max-height", "1080"])
            if status:
                failures.append(row["video_id"])
                # Preserve the downloader's stop-on-bot-check behavior across sources.
                break
        info = probe(video)
        if info["height"] > 1080 or info["duration"] < 60:
            raise ValueError("Downloaded source is outside pilot geometry/duration")
        row.update(video_path=str(video.resolve()), source_sha256=file_hash(video), **info)
    atomic_json(args.root / "sources.json", {"schema": 1, "sources": sources, "failures": failures})
    return {"downloaded": sum("source_sha256" in r for r in sources), "failures": failures}


def native_points(points, crop_box, *, inverse=False):
    result = np.asarray(points, dtype=float).reshape(-1, 2).copy()
    offset = np.asarray(crop_box[:2], dtype=float)
    return (result - offset if inverse else result + offset).tolist()


def validate_box(box, size):
    if len(box) != 4 or any(type(v) is not int for v in box):
        raise ValueError("Crop box requires integer native coordinates")
    x0, y0, x1, y1 = box
    if not 0 <= x0 < x1 <= size[0] or not 0 <= y0 < y1 <= size[1]:
        raise ValueError("Crop outside source geometry")


def prepare(args):
    from .scene_reference import sample_source
    sources_doc = read_json(args.root / "sources.json")
    sources = sources_doc["sources"]
    selection = read_json(args.selection)
    # Reviewed source identity is recorded with the selected intervals, not inferred from titles.
    for row in sources:
        chosen = selection["matches"][row["video_id"]]
        row.update({k: chosen[k] for k in ("match_id", "gameplay_review", "match_evidence")})
    validate_sources(sources, reviewed=True)
    recipe = {"sources": sources, "selection": selection, "fps": FPS, "version": 1}
    path = args.root / "dataset.json"
    previous, prior_reviews = None, None
    if path.exists():
        old = read_json(path)
        if old["recipe_sha256"] != digest(recipe):
            if not args.revise_unfrozen or (args.root / "labels.lock.json").exists():
                raise ValueError("Dataset recipe changed; choose a new experiment root or explicitly revise unfrozen inputs")
            previous = old
            reviews_path = args.root / "reviews.json"
            prior_reviews = read_json(reviews_path) if reviews_path.exists() else None
            if prior_reviews and prior_reviews["dataset_sha256"] != file_hash(path):
                raise ValueError("Prior reviews do not match the dataset being revised")
            archive = args.root / "revisions" / (file_hash(path) + "-" + digest(prior_reviews)[:12])
            archive.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, archive / "dataset.json")
            if prior_reviews:
                shutil.copy2(reviews_path, archive / "reviews.json")
        else:
            return {"status": "reused", "images": len(old["images"])}
    images, clips = [], []
    for source in sources:
        video = Path(source["video_path"])
        if file_hash(video) != source["source_sha256"]:
            raise ValueError("Source footage changed")
        chosen = selection["matches"][source["video_id"]]
        if len(chosen["clips"]) != 4 or len(chosen["negatives"]) != 4:
            raise ValueError("Each match needs four clips and four negative crops")
        if {c["behavior"] for c in chosen["clips"]} != {"walking", "casting", "minions", "interaction"}:
            raise ValueError("Missing behavior stratum")
        for number, clip in enumerate(chosen["clips"]):
            start = float(clip["start"])
            if not math.isfinite(start) or not 0 <= start <= source["duration"] - 5:
                raise ValueError("Invalid five-second interval")
            validate_box(clip["crop_box"], (source["width"], source["height"]))
            clip_id = source["video_id"] + f"-{number}"
            folder = args.root / "clips" / clip_id
            folder.mkdir(parents=True, exist_ok=True)
            rows = []
            for index, (pts, image) in enumerate(sample_source(video, start, start + 5, FPS)):
                frame = folder / f"{index:06}.jpg"
                image.save(frame, quality=95)
                rows.append({"index": index, "source_time": pts, "path": str(frame.relative_to(args.root)),
                             "path_sha256": file_hash(frame)})
            if not 78 <= len(rows) <= 81:
                raise ValueError("Incomplete five-second source clip")
            refs = sorted({round(i * (len(rows) - 1) / 6) for i in range(7)})
            for index in refs:
                record = rows[index]
                with checked_image(args.root, record) as full:
                    crop = full.crop(clip["crop_box"])
                image_id = clip_id + f"-{index:06}"
                dest = args.root / "images" / (image_id + ".png")
                dest.parent.mkdir(parents=True, exist_ok=True)
                crop.save(dest)
                images.append({"id": image_id, "clip_id": clip_id, "frame_index": index,
                               "source_time": record["source_time"], "video_id": source["video_id"],
                               "match_id": source["match_id"], "split": source["split"],
                               "source_sha256": source["source_sha256"], "crop_box": clip["crop_box"],
                               "source_size": [source["width"], source["height"]], "negative": False,
                               "path": str(dest.relative_to(args.root)), "path_sha256": file_hash(dest)})
            clips.append({"id": clip_id, "split": source["split"], "video_id": source["video_id"],
                          "behavior": clip["behavior"], "start": start, "end": start + 5,
                          "frames": rows, "reference_indices": refs})
        for number, negative in enumerate(chosen["negatives"]):
            validate_box(negative["crop_box"], (source["width"], source["height"]))
            t = float(negative["time"])
            if not math.isfinite(t) or not 0 <= t < source["duration"]:
                raise ValueError("Invalid negative timestamp")
            pts, image = next(sample_source(video, t, min(t + 1, source["duration"]), 1))
            image_id = source["video_id"] + f"-negative-{number}"
            dest = args.root / "images" / (image_id + ".png")
            image.crop(negative["crop_box"]).save(dest)
            images.append({"id": image_id, "source_time": pts, "video_id": source["video_id"],
                           "match_id": source["match_id"], "split": source["split"], "negative": True,
                           "source_sha256": source["source_sha256"], "source_size": [source["width"], source["height"]],
                           "crop_box": negative["crop_box"], "path": str(dest.relative_to(args.root)),
                           "path_sha256": file_hash(dest)})
    manifest = {"schema": 1, "recipe_sha256": digest(recipe), "recipe": recipe, "images": images, "clips": clips}
    atomic_json(path, manifest)
    reviews = {"schema": 1, "dataset_sha256": file_hash(path), "reviewer": "Codex", "reviewer_kind": "agent",
               "images": {r["id"]: {"status": "unreviewed", "image_sha256": r["path_sha256"],
                                     "all_champions_labeled": False, "instances": [], "reason": ""} for r in images}}
    atomic_json(args.root / "reviews.template.json", reviews)
    if prior_reviews:
        old_rows = {r["id"]:r for r in previous["images"]}
        for row in images:
            if old_rows.get(row["id"]) == row:
                reviews["images"][row["id"]] = prior_reviews["images"][row["id"]]
        reviews["prior_dataset_sha256"] = prior_reviews["dataset_sha256"]
        atomic_json(args.root / "reviews.json", reviews)
    return {"images": len(images), "clips": len(clips), "status": "needs_review"}


def polygon_mask(size, polygons):
    image = Image.new("L", tuple(size))
    draw = ImageDraw.Draw(image)
    for polygon in polygons:
        coords = np.asarray(polygon, dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) < 3 or not np.isfinite(coords).all():
            raise ValueError("Invalid mask polygon")
        if (coords < 0).any() or (coords[:, 0] >= size[0]).any() or (coords[:, 1] >= size[1]).any():
            raise ValueError("Polygon outside crop")
        draw.polygon([tuple(p) for p in coords], fill=255)
    return np.asarray(image) > 0


def instance_mask(root, image_row, instance):
    size = (image_row["crop_box"][2] - image_row["crop_box"][0],
            image_row["crop_box"][3] - image_row["crop_box"][1])
    if "mask_path" in instance:
        mask_row = {"path": instance["mask_path"], "path_sha256": instance["mask_sha256"]}
        with checked_image(root, mask_row) as image:
            if image.size != size:
                raise ValueError("Mask differs from crop geometry")
            values = np.asarray(image.convert("L"))
            if not np.isin(values, [0, 255]).all():
                raise ValueError("Reviewed mask must be binary")
            mask = values > 0
    else:
        mask = polygon_mask(size, instance.get("polygons", []))
    if not mask.any():
        raise ValueError("Empty visible instance mask")
    return mask


def rle_encode(mask):
    """Uncompressed COCO RLE, using column-major order, with no GPU dependency."""
    flat = np.asarray(mask, dtype=bool).ravel(order="F")
    counts, last, run = [], False, 0
    for value in flat:
        if bool(value) == last:
            run += 1
        else:
            counts.append(run)
            run, last = 1, bool(value)
    counts.append(run)
    return {"size": list(mask.shape), "counts": counts}


def rle_decode(rle):
    h, w = rle["size"]
    counts = rle["counts"]
    if not isinstance(counts, list):
        from pycocotools import mask as mask_util
        return mask_util.decode(rle).astype(bool)
    if any(type(c) is not int or c < 0 for c in counts) or sum(counts) != h * w:
        raise ValueError("Invalid RLE counts")
    return np.repeat(np.arange(len(counts)) % 2, counts).reshape((h, w), order="F").astype(bool)


def mask_box(mask):
    y, x = np.where(mask)
    return [int(x.min()), int(y.min()), int(x.max() + 1), int(y.max() + 1)]


def export(args):
    if (args.root / "labels.lock.json").exists():
        return verify_labels(args.root)
    dataset = read_json(args.root / "dataset.json")
    reviews_path = args.root / "reviews.json"
    reviews = read_json(reviews_path)
    if reviews["dataset_sha256"] != file_hash(args.root / "dataset.json"):
        raise ValueError("Reviews belong to another dataset")
    if reviews.get("reviewer_kind") != "agent" or not reviews.get("reviewer"):
        raise ValueError("Pilot requires explicit agent review provenance")
    if set(reviews["images"]) != {r["id"] for r in dataset["images"]}:
        raise ValueError("Review coverage differs from dataset")
    docs = {s: {"images": [], "annotations": [], "categories": [{"id": 1, "name": PROMPT}]} for s in SPLITS}
    excluded, match_splits = [], {}
    for number, row in enumerate(dataset["images"], 1):
        split = row["split"]
        for identity in (row["match_id"], row["video_id"], row["source_sha256"]):
            if identity in match_splits and match_splits[identity] != split:
                raise ValueError("Source or underlying match leaks across splits")
            match_splits[identity] = split
        review = reviews["images"][row["id"]]
        if review["image_sha256"] != row["path_sha256"]:
            raise ValueError("Review image hash differs")
        if review["status"] == "unreviewed":
            raise ValueError("Unreviewed images cannot enter an export")
        if review["status"] in ("uncertain", "rejected"):
            if not review.get("reason"):
                raise ValueError("Excluded image requires a reason")
            excluded.append(row["id"])
            continue
        if review["status"] != "accepted" or not review.get("all_champions_labeled"):
            raise ValueError("Accepted crop needs an exhaustive champion review")
        if bool(row["negative"]) != (len(review["instances"]) == 0):
            raise ValueError("Positive/negative annotation contradiction")
        with checked_image(args.root, row) as image:
            width, height = image.size
        dest = args.root / "coco" / split / Path(row["path"]).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(safe_path(args.root, row["path"]), dest)
        docs[split]["images"].append({"id": number, "file_name": dest.name, "width": width, "height": height,
                                      "sample_id": row["id"], "source_sha256": row["source_sha256"]})
        tracks = set()
        for instance in review["instances"]:
            if not instance.get("track_id") or instance["track_id"] in tracks:
                raise ValueError("Each instance needs a unique clip-local track ID")
            tracks.add(instance["track_id"])
            if "bar_box" in instance:
                validate_box(instance["bar_box"], (width, height))
            mask = instance_mask(args.root, row, instance)
            x0, y0, x1, y1 = mask_box(mask)
            docs[split]["annotations"].append({"id": number * 1000 + len(tracks), "image_id": number,
                "category_id": 1, "iscrowd": 0, "bbox": [x0, y0, x1-x0, y1-y0], "area": int(mask.sum()),
                "segmentation": rle_encode(mask), "track_id": instance["track_id"]})
    hashes = {}
    for split, doc in docs.items():
        if not doc["images"] or not doc["annotations"]:
            raise ValueError("Every split needs visible champion masks")
        if not any(not any(a["image_id"] == i["id"] for a in doc["annotations"]) for i in doc["images"]):
            raise ValueError("Every split needs reviewed negatives")
        path = args.root / "coco" / split / "_annotations.coco.json"
        atomic_json(path, doc)
        hashes[split] = file_hash(path)
    lock = {"schema": 1, "dataset_sha256": file_hash(args.root / "dataset.json"),
            "reviews_sha256": file_hash(reviews_path), "coco_sha256": hashes, "excluded": excluded,
            "reviewer": reviews["reviewer"], "reviewer_kind": reviews["reviewer_kind"],
            "counts": {s: {"images": len(d["images"]), "instances": len(d["annotations"])} for s,d in docs.items()}}
    lock_path = args.root / "labels.lock.json"
    if lock_path.exists() and read_json(lock_path) != lock:
        raise ValueError("Frozen labels changed; choose a new experiment root")
    atomic_json(lock_path, lock)
    return lock


def verify_labels(root):
    lock = read_json(root / "labels.lock.json")
    for name, expected in (("dataset.json", lock["dataset_sha256"]), ("reviews.json", lock["reviews_sha256"])):
        if file_hash(root / name) != expected:
            raise ValueError("Frozen dataset/review changed")
    for split, sha in lock["coco_sha256"].items():
        if file_hash(root / "coco" / split / "_annotations.coco.json") != sha:
            raise ValueError("Frozen COCO annotations changed")
    dataset = read_json(root / "dataset.json")
    for row in dataset["images"]:
        with checked_image(root, row):
            pass
        exported = root / "coco" / row["split"] / Path(row["path"]).name
        if row["id"] not in lock["excluded"] and file_hash(exported) != row["path_sha256"]:
            raise ValueError("Exported COCO image changed")
    return lock


def foreground_points(mask, count=3):
    from scipy.ndimage import distance_transform_edt
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not mask.any():
        raise ValueError("Cannot sample an empty mask")
    # Padding makes boundary distance correct even when foreground touches an image edge.
    distances = distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
    safe = distances >= max(1., float(distances.max()) * .4)
    selected = []
    score = distances.copy()
    yy, xx = np.indices(mask.shape)
    for _ in range(count):
        if not np.any(score > 0):
            break
        y, x = np.unravel_index(np.argmax(score), mask.shape)
        selected.append([int(x), int(y)])
        score[(xx-x)**2 + (yy-y)**2 < max(4., distances[y,x])**2] = 0
        score[~safe] = 0
    return selected


def seed_tracker(predictor, session_id, frame_index, object_id, mask, crop_box, source_size):
    validate_box(crop_box, source_size)
    if np.asarray(mask).shape != (crop_box[3]-crop_box[1], crop_box[2]-crop_box[0]):
        raise ValueError("Seed mask differs from crop geometry")
    points = native_points(foreground_points(mask), crop_box)
    width, height = source_size
    return predictor.handle_request({"type": "add_prompt", "session_id": session_id,
        "frame_index": frame_index, "obj_id": object_id,
        "points": [[x/width, y/height] for x,y in points], "point_labels": [1] * len(points),
        "rel_coordinates": True})


def review_panels(args):
    """Render proposals beside untouched crops; this never changes review status."""
    dataset = read_json(args.root / "dataset.json")
    count = 0
    for row in dataset["images"]:
        proposal = safe_path(args.root,args.proposal_dir) / row["id"] / "proposal.json"
        if not proposal.exists():
            continue
        draft = read_json(proposal)
        if draft["image_sha256"] != row["path_sha256"]:
            raise ValueError("Proposal belongs to another source image")
        with checked_image(args.root, row) as source:
            source = source.convert("RGB")
        layer = source.convert("RGBA")
        palette = [(255,60,40), (40,220,255), (150,255,50), (250,70,255), (255,220,40)]
        for index, entry in enumerate(draft["masks"]):
            path = safe_path(args.root, entry["mask_path"])
            if file_hash(path) != entry["mask_sha256"]:
                raise ValueError("Proposal mask changed")
            with Image.open(path) as image:
                mask = np.asarray(image.convert("L")) > 0
            rgba = np.zeros((source.height, source.width, 4), dtype=np.uint8)
            rgba[mask] = [*palette[index % len(palette)], 100]
            layer = Image.alpha_composite(layer, Image.fromarray(rgba))
            if mask.any():
                x0,y0,x1,y1 = mask_box(mask)
                draw = ImageDraw.Draw(layer)
                draw.rectangle([x0,y0,x1,y1], outline=palette[index % len(palette)], width=2)
                draw.text((x0,max(0,y0-14)), str(index), fill="white", stroke_width=1, stroke_fill="black")
        panel = Image.new("RGB", (source.width*2, source.height+30))
        panel.paste(source,(0,30)); panel.paste(layer.convert("RGB"),(source.width,30))
        ImageDraw.Draw(panel).text((8,8), row["id"] + "   source | UNREVIEWED proposals", fill="white")
        dest = args.root / "review-panels" / (row["id"] + ".jpg")
        dest.parent.mkdir(parents=True, exist_ok=True)
        panel.save(dest,quality=95)
        count += 1
    return {"panels":count, "review_status_changed":False}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("collect", "prepare", "export", "review-panels", "cache-model", "propose", "smoke", "train", "evaluate"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--root", type=Path, default=DEFAULT_ROOT)
        if name == "collect":
            cmd.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
            cmd.add_argument("--video-root", type=Path, default=Path("data/youtube"))
        if name == "prepare":
            cmd.add_argument("--selection", type=Path, required=True)
            cmd.add_argument("--revise-unfrozen", action="store_true", help="Archive metadata and reset changed-image reviews before labels are frozen")
        if name == "cache-model":
            cmd.add_argument("--revision", required=True)
        if name in ("propose", "smoke", "train", "evaluate"):
            cmd.add_argument("--sam-source", type=Path, required=True)
        if name in ("smoke", "train"):
            cmd.add_argument("--epochs", type=int, default=20)
        if name == "propose":
            cmd.add_argument("--limit", type=int)
        if name == "review-panels":
            cmd.add_argument("--proposal-dir", default="proposals")
        if name == "evaluate":
            cmd.add_argument("--tracking-cache", type=Path, required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    args.root = args.root.resolve()
    if args.command == "review-panels":
        result = review_panels(args)
    elif args.command in ("collect", "prepare", "export"):
        result = globals()[args.command](args)
    else:
        from . import sam_finetune_train
        result = getattr(sam_finetune_train, args.command.replace("-", "_"))(args)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 1 if result.get("failures") else 0


if __name__ == "__main__":
    raise SystemExit(main())
