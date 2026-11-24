"""Paired SAM video tracks and deterministic, mask-based seed selection."""

from collections import Counter
from contextlib import nullcontext
from functools import partial
import os
from pathlib import Path
from statistics import mean
import tempfile
import time

import numpy as np
from PIL import Image

from .body_campaign import stable_proposal_bindings
from .body_operational import verified_crop
from .core import Store, atomic_json, digest, read_json, uid
from .media import file_hash
from .sam_appearances import (FPS, materialize_clean_crops, prompt_appearance, propagate_appearance,
                              receipt_valid, stage_lock, stitch_fragments)
from .sam_multichampion import _mask_rows, box_iou

ARMS = ("sam31_box_text", "sam31_points", "sam21_points")
SAM21 = {
    "code_revision": "2b90b9f5ceec907a1c18123530e92e794ad901a4",
    "config": "configs/sam2.1/sam2.1_hiera_l.yaml",
    "checkpoint": "sam2.1_hiera_large.pt",
    "sha256": "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318",
}


def recover_observations(prepared):
    frames, profile = prepared["frames"], prepared["recipe"]["profile"]
    barriers = {i for i, frame in enumerate(frames) if frame["scoreboard_visible"]}
    bindings = stable_proposal_bindings(
        frames, profile, (None if i in barriers else frame["proposals"] for i, frame in enumerate(frames)),
    )
    source = prepared["recipe"]["source_sha256"]
    return {uid("appearance", source, team, [(o["frame_index"], o["bar_box"]) for o in rows]): rows
            for team, rows in stitch_fragments(bindings, frames, profile["body_binder"]["maximum_step_pixels"],
                                               barriers)}


def target_observation(observations, index, *, interpolate=True):
    key = str(index)
    if key in observations:
        return observations[key], "observed"
    if interpolate and str(index - 1) in observations and str(index + 1) in observations:
        left, right = observations[str(index - 1)], observations[str(index + 1)]
        return {"frame_index": index, **{key: [round((a + b) / 2) for a, b in zip(left[key], right[key])]
                                        for key in ("bar_box", "body_box")}}, "interpolated"
    return None, "unavailable"


def torso_point(observation):
    left, top, right, bottom = observation["body_box"]
    x0, _, x1, bar_bottom = observation["bar_box"]
    return [min(right - 1, max(left, (x0 + x1) // 2)),
            min(bottom - 1, max(top, bar_bottom + round((bottom - top) * .30)))]


def neighboring_points(observation, frame):
    x, y = torso_point(observation)
    left, top, right, bottom = observation["body_box"]
    w, h = right - left, bottom - top
    nearby = set()
    for other in frame["raw_proposals"]:
        if box_iou(observation["bar_box"], other["bar_box"]) >= .5:
            continue
        a, b = torso_point({"bar_box": other["bar_box"], "body_box": other["actor_region"]})
        distance = ((a - x) / w) ** 2 + ((b - y) / h) ** 2
        if .04 <= distance <= 1.5:
            nearby.add((distance, a, b))
    return [[a, b] for _, a, b in sorted(nearby)[:4]]


def seed_candidates(appearance, observations):
    seed = appearance["seed"]
    original = seed["frame_index"]
    available = {index: observations[str(index)] for index in appearance["seed_frames"] if index != original}
    seeds = [seed]
    first, last = appearance["first"], appearance["last"]
    for target in (first + .25 * (last - first), first + .75 * (last - first)):
        if available:
            index = min(available, key=lambda i: (abs(i - target), i))
            seeds.append(available.pop(index))
    return seeds


def point_seed(predictor, prompt, target, *, points, size, sam21):
    width, height = size
    coords = [[x / width, y / height] for x, y in points]
    labels = [1] + [0] * (len(coords) - 1)
    if sam21:
        request = {**prompt, "points": coords, "point_labels": labels, "obj_id": 1}
        obj = 1
    else:
        obj, iou, masks = prompt_appearance(predictor, prompt, target)
        if obj is None:
            return obj, iou, masks
        request = {"type": "add_prompt", "session_id": prompt["session_id"], "frame_index": prompt["frame_index"],
                   "points": coords, "point_labels": labels, "obj_id": obj, "rel_coordinates": True}
    masks = _mask_rows(predictor.handle_request(request)["outputs"])
    mask = next((row for row in masks if row["object_id"] == obj), None)
    return (obj, box_iou(target["box_xyxy"], mask["box_xyxy"]), masks) if mask else (None, 0., masks)


class SAM21Video:
    """Expose one SAM 2.1 object through the existing native-mask propagation loop."""

    def __init__(self, predictor):
        self.predictor = predictor
        self.state = None

    @staticmethod
    def outputs(index, ids, logits):
        return {"frame_index": index, "outputs": {"out_obj_ids": ids, "out_binary_masks": logits > 0}}

    def handle_request(self, request):
        kind = request["type"]
        if kind == "start_session":
            self.state = self.predictor.init_state(video_path=request["resource_path"], offload_video_to_cpu=True)
            return {"session_id": "sam21"}
        if kind == "close_session":
            self.state = None
            return {}
        if kind != "add_prompt":
            raise ValueError(f"Unsupported SAM 2.1 request {kind}")
        width, height = self.state["video_width"], self.state["video_height"]
        left, top, w, h = request["bounding_boxes"][0]
        box = [left * width, top * height, (left + w) * width, (top + h) * height]
        points = [[x * width, y * height] for x, y in request["points"]]
        return self.outputs(*self.predictor.add_new_points_or_box(
            self.state, frame_idx=request["frame_index"], obj_id=request["obj_id"],
            points=points, labels=request["point_labels"], box=box,
        ))

    def handle_stream_request(self, request):
        for result in self.predictor.propagate_in_video(
            self.state, start_frame_idx=request["start_frame_index"],
            reverse=request["propagation_direction"] == "backward",
        ):
            yield self.outputs(*result)


def load_predictor(arm, cache, sam21_weights, specification):
    import torch

    if arm == "sam21_points":
        from sam2.build_sam import build_sam2_video_predictor

        if sam21_weights is None or file_hash(sam21_weights) != SAM21["sha256"]:
            raise ValueError("Stage the checksum-pinned SAM 2.1 Hiera Large checkpoint")
        predictor = SAM21Video(build_sam2_video_predictor(
            SAM21["config"], str(sam21_weights), device="cuda", apply_postprocessing=False,
        ))
        return predictor, SAM21, lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    from .local_models import checkpoint
    from .sam_compat import load_multiplex_predictor

    root, pin = checkpoint(cache, "tracking")
    if pin["specification"] != specification:
        raise ValueError("SAM 3.1 checkpoint differs from the prepared match")
    predictor, load = load_multiplex_predictor(root / "sam3.1_multiplex.pt")
    return predictor, {**pin, "model_load": load}, nullcontext


def trial_metrics(result, directory, observations, frames):
    masks = {row["frame_index"]: row for row in result["samples"] if row["visible"]}
    covered, contamination, aligned = [], [], {}
    for index in range(result["first"], result["last"] + 1):
        observation, _ = target_observation(observations, index)
        if observation is None:
            continue
        row = masks.get(index)
        if row is None:
            covered.append(0.)
            continue
        mask = verified_crop(directory / row["mask_path"], row["mask_sha256"]).convert("L")
        x, y = torso_point(observation)
        covered.append(float(np.mean(np.asarray(mask.crop((x - 5, y - 5, x + 6, y + 6))) > 127)))
        aligned[index] = np.asarray(mask.crop((x - 160, y - 160, x + 160, y + 160))) > 127
        neighbors = neighboring_points(observation, frames[index])
        contamination.extend(float(np.mean(np.asarray(mask.crop((a - 5, b - 5, a + 6, b + 6))) > 127))
                             for a, b in neighbors)
    stability = []
    for index, left in aligned.items():
        if index + 1 in aligned:
            right = aligned[index + 1]
            union = np.count_nonzero(left | right)
            stability.append(np.count_nonzero(left & right) / union if union else 0.)
    coverage = mean(covered) if covered else 0.
    overlap = mean(contamination) if contamination else 0.
    consistency = mean(stability) if stability else 0.
    return {"target_coverage": coverage, "neighbor_overlap": overlap, "stability": consistency,
            "score": coverage + consistency - overlap, "observed_frames": len(covered),
            "neighbor_points": len(contamination), "stability_pairs": len(stability)}


def track_one(predictor, arm, source, output, match, appearance):
    frames, profile = match["frames"], match["profile"]
    observations = appearance["observations"]
    original = appearance["seed"]

    def run(seed, first, last, directory):
        request = prompt_appearance
        if arm != ARMS[0]:
            request = partial(point_seed, points=[torso_point(seed), *neighboring_points(seed, frames[seed["frame_index"]])],
                              size=profile["size"], sam21=arm == "sam21_points")
        result = propagate_appearance(predictor, source, directory, frames,
                                      {**appearance, "first": first, "last": last, "seed": seed},
                                      profile, FPS, seed_request=request, allow_incomplete=True)
        return {**result, "first": first, "last": last}

    trials, completed = [], []
    selected = original
    if arm != ARMS[0]:
        for seed in seed_candidates(appearance, observations):
            index = seed["frame_index"]
            first = max(appearance["first"], index - FPS // 2)
            last = min(appearance["last"], index + FPS // 2)
            with tempfile.TemporaryDirectory(prefix="seed-", dir=output) as directory:
                directory = Path(directory)
                result = run(seed, first, last, directory)
                metrics = trial_metrics(result, directory, observations, frames)
            trials.append({"frame_index": index, "first": first, "last": last,
                           "status": result["status"], **metrics})
            if result["status"] == "completed":
                completed.append((-metrics["score"], abs(index - original["frame_index"]), index, seed))
        if not completed:
            status = "propagation_incomplete" if any(t["status"] == "propagation_incomplete" for t in trials) else "seed_rejected"
            return {"status": status, "seed_trials": trials, "selected_seed": None, "samples": [], "crops": []}
        selected = min(completed)[-1]

    full = run(selected, appearance["first"], appearance["last"], output)
    crops = materialize_clean_crops(source, output, full["samples"], frames, profile)
    return {**full, "crops": crops, "selected_seed": selected["frame_index"], "seed_trials": trials}


def track_recipe(study_sha, match, arm):
    return {"study_sha256": study_sha, "source_sha256": match["source_sha256"], "arm": arm,
            "fps": FPS, "model": SAM21 if arm == "sam21_points" else match["sam"]}


def track_receipt(root, study_sha, match, arm, appearance):
    parent = root / "tracks" / match["id"] / arm / appearance["id"]
    path = parent / "receipt.json"
    if not path.is_file():
        return None, None
    receipt = read_json(path)
    artifacts = parent / receipt["artifacts"]
    recipe = track_recipe(study_sha, match, arm)
    if (receipt["recipe"] != recipe or receipt["appearance_id"] != appearance["id"]
            or not artifacts.resolve().is_relative_to(parent.resolve())
            or not receipt_valid(receipt, artifacts, digest(recipe))
            or not receipt_valid({**receipt, "samples": receipt["crops"]}, artifacts, digest(recipe))):
        raise ValueError(f"Track receipt or masks changed for {match['id']}/{arm}/{appearance['id']}")
    return receipt, artifacts


def track(root, study, study_sha, arm, cache, sam21_weights=None, *, smoke=False, loader=load_predictor):
    stop = Store(root)
    stop.install_signals()
    statuses = Counter()
    predictor = None
    for match in study["matches"]:
        recipe = track_recipe(study_sha, match, arm)
        source = root / "inputs" / match["id"]
        rows = match["appearances"][:1] if smoke else match["appearances"]
        for appearance in rows:
            stop.check_stop()
            parent = root / "tracks" / match["id"] / arm / appearance["id"]
            with stage_lock(parent / ".lock"):
                try:
                    previous, _ = track_receipt(root, study_sha, match, arm, appearance)
                except ValueError:
                    # Preserve the invalid receipt for diagnosis until a fresh result commits.
                    previous = None
                if previous:
                    statuses[previous["status"]] += 1
                    continue
                if predictor is None:
                    predictor, pin, runtime = loader(arm, cache, sam21_weights, match["sam"])
                started = time.monotonic()
                with tempfile.TemporaryDirectory(prefix=".tracking-", dir=parent) as work:
                    work = Path(work)
                    artifacts = work / "artifacts"
                    artifacts.mkdir()
                    with runtime():
                        result = track_one(predictor, arm, source, artifacts, match, appearance)
                    artifact_name = "artifacts-" + work.name.removeprefix(".tracking-")
                    artifacts.rename(parent / artifact_name)
                    atomic_json(parent / "receipt.json", {
                        **result, "recipe": recipe, "manifest_sha256": digest(recipe), "model": pin,
                        "appearance_id": appearance["id"], "artifacts": artifact_name,
                        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "wall_seconds": time.monotonic() - started,
                    })
                statuses[result["status"]] += 1
    return {"arm": arm, "smoke": smoke, "statuses": dict(statuses)}
