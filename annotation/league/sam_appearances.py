"""Seed anonymous champion appearances and match their SAM crops to 3D skins."""

from collections import Counter, defaultdict
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import shutil
import tempfile
import time

from .body_siglip import DEFAULT_SIGLIP_LOCK
from .core import atomic_json, digest, finite, read_json, uid
from .media import file_hash
from .sam_multichampion import box_iou

FPS = 16
PATCH = ["16.22.1", "16.22"]
DEFAULT_DINO_LOCK = Path("data/league_assets/body-encoder/dinov2-vits14/encoder-lock.json")
MASK_PROMPT = "box-and-champion-text-v1"


@contextmanager
def stage_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def stitch_fragments(bindings, frames, step, barriers):
    """Link nearby endpoints only when neither endpoint has another viable link."""
    ordered = sorted(bindings, key=lambda row: row["seed_frame_index"])
    outgoing, incoming = defaultdict(list), defaultdict(list)
    for i, left in enumerate(ordered):
        tail = left["observations"][-1]
        a = tail["frame_index"]
        for j in range(i + 1, len(ordered)):
            right = ordered[j]
            head = right["observations"][0]
            b = head["frame_index"]
            if frames[b]["time"] - frames[a]["time"] > .5:
                break
            if b <= a or right["team"] != left["team"] or any(k in barriers for k in range(a + 1, b)):
                continue
            if head.get("level") is not None and tail.get("level") is not None and head["level"] - tail["level"] not in (0, 1):
                continue
            distance = sum(abs(p - q) for p, q in zip(tail["bar_box"][:2], head["bar_box"][:2]))
            if distance <= step:
                outgoing[i].append(j)
                incoming[j].append(i)
    links = {i: choices[0] for i, choices in outgoing.items()
             if len(choices) == len(incoming[choices[0]]) == 1}
    for i, row in enumerate(ordered):
        if i in links.values():
            continue
        observations = list(row["observations"])
        j = i
        while j in links:
            j = links[j]
            observations.extend(ordered[j]["observations"])
        yield row["team"], observations


def pick_seed(team, observations, frames, offscreen):
    first, last = observations[0]["frame_index"], observations[-1]["frame_index"]
    if team == "friendly" and any(
        (own := frames[row["frame_index"]]["player_bar"]) is not None
        and box_iou(row["bar_box"], own) >= .5 for row in observations
    ):
        return None, "player_excluded"
    midpoint = (frames[first]["time"] + frames[last]["time"]) / 2
    candidates = []
    for row in observations:
        frame = frames[row["frame_index"]]
        if team == "friendly" and frame["player_bar"] is None and frame["id"] not in offscreen:
            continue
        overlap = max((box_iou(row["body_box"], other["actor_region"]) for other in frame["proposals"]
                       if box_iou(row["bar_box"], other["bar_box"]) < .5), default=0)
        candidates.append((overlap, abs(frame["time"] - midpoint), frame["time"], row))
    if not candidates:
        return None, "player_binding_unresolved"
    return min(candidates, key=lambda entry: entry[:3])[-1], "prepared"


def save_seed_preview(root, appearance, frame):
    from PIL import Image, ImageDraw

    with Image.open(root / frame["path"]) as image:
        image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    color = "cyan" if appearance["team"] == "friendly" else "red"
    draw.rectangle(appearance["seed"]["body_box"], outline=color, width=5)
    draw.text((16, 16), f'{appearance["team"]}  {frame["time"]:.3f}s', fill="white", stroke_width=2,
              stroke_fill="black")
    image.thumbnail((960, 540))
    image.save(root / appearance["seed_preview"], quality=90)


def prepare_appearances(args):
    from .action_body import candidates
    from .action_v3 import load_profile
    from .body_campaign import stable_healthbar_bindings, stable_proposal_bindings
    from .body_level import LEVEL_READER_VERSION, level_proposals
    from .body_slot_context import MatchRosterIndex
    from .local_models import LOCK
    from .scene_reference import sample_source

    root, source = args.root.resolve(), args.video.resolve()
    if not finite(args.start) or not finite(args.end) or not 0 <= args.start < args.end:
        raise ValueError("Choose a valid source interval")
    with stage_lock(root / ".prepare.lock"):
        output = root / "appearances.json"
        source_sha = file_hash(source)
        roster_file = read_json(args.scoreboard_skim / "match-roster.json")
        index = MatchRosterIndex(args.scoreboard_skim, source_sha, set(roster_file["champions"]))
        roster = index.roster
        membership = {(side, champion) for side, champion in roster["membership"]}
        sides = {side for side, champion in membership if champion == args.player}
        if len(sides) != 1 or any(sum(side == team for side, _ in membership) != 5 for team in ("left", "right")):
            raise ValueError("A complete scoreboard and unique player side are required")
        friendly = sides.pop()
        if (roster["status"] not in {"confirmed", "experimental"}
                or not roster["interval"][0] <= args.start < args.end <= roster["interval"][1]):
            raise ValueError("Scoreboard does not cover the requested source interval")
        offscreen = read_json(args.player_offscreen) if args.player_offscreen else {"source_sha256": source_sha, "frames": []}
        if offscreen["source_sha256"] != source_sha:
            raise ValueError("Player-offscreen review refers to another video")
        profile = load_profile(args.profile)
        previous = args.proposals_from.resolve() if args.proposals_from else None
        recipe = {"schema": 1, "source_sha256": source_sha, "window": [args.start, args.end], "fps": FPS,
                  "scoreboard": index.revision, "roster_id": roster["id"], "roster_status": roster["status"],
                  "profile": profile, "player": args.player, "friendly_side": friendly,
                  "patch": list(getattr(args, "patch", None) or PATCH),
                  "player_offscreen_review": offscreen, "sam": read_json(LOCK)["models"]["tracking"],
                  "siglip_lock_hash": digest(read_json(args.siglip_lock)), "level_reader": LEVEL_READER_VERSION,
                  "prior_proposals_sha256": file_hash(previous) if previous else None}
        staging = root / "prepare-recipe.json"
        if staging.exists() and read_json(staging) != recipe:
            raise ValueError("Preparation inputs changed; choose a new run directory")
        if output.exists():
            return read_json(output)["counts"]
        atomic_json(staging, recipe)
        for name in ("frames", "sam-inputs", "seed-previews"):
            (root / name).mkdir(exist_ok=True)
        frames = []
        if previous is not None:
            raw = read_json(previous)
            for key in ("source_sha256", "window", "fps", "profile"):
                if raw["recipe"][key] != recipe[key]:
                    raise ValueError(f"Prior health-bar proposals disagree with the new run on {key}")
            frames = raw["frames"]
        for n, (pts, image) in enumerate(() if previous else sample_source(source, args.start, args.end, FPS)):
            native, sam = f"frames/{n:06}.png", f"sam-inputs/{n:06}.jpg"
            for relative, options in ((native, {"format": "PNG", "compress_level": 1}),
                                      (sam, {"format": "JPEG", "quality": 95, "subsampling": 0})):
                path = root / relative
                if not path.exists():
                    temporary = path.with_suffix(".tmp")
                    image.save(temporary, **options)
                    temporary.replace(path)
            frames.append({"id": uid("sam-frame", source_sha, pts), "time": pts, "path": native, "sam_path": sam,
                           "sha256": file_hash(root / native), "sam_sha256": file_hash(root / sam)})
        if not frames:
            raise ValueError("No source frames in the requested interval")
        print(f"Prepared {len(frames)} source frames", flush=True)

        barriers = set()

        def observe(n, image, proposals):
            frame = frames[n]
            frame.update(raw_proposals=proposals or [], scoreboard_visible=proposals is None)
            if proposals is None:
                barriers.add(n)
                frame.update(proposals=[], player_bar=None)
                return None
            if previous is None:
                own, _ = candidates(image, profile, require_coherent_fill=True)
                frame["player_bar"] = own[0]["bar_box"] if len(own) == 1 and own[0]["hud_health_fraction"] is not None else None
            frame["proposals"] = level_proposals(image, profile, proposals)
            return frame["proposals"]

        if previous is not None:
            from .body_operational import verified_crop

            def reuse_detections():
                for n, frame in enumerate(frames):
                    if n % 400 == 0:
                        print(f"Checking champion levels at frame {n} of {len(frames)}", flush=True)
                    if frame.get("scoreboard_visible"):
                        yield observe(n, None, None)
                        continue
                    proposals = frame.get("raw_proposals", frame["proposals"])
                    image = verified_crop(root / frame["path"], frame["sha256"]) if proposals else None
                    yield observe(n, image, proposals)

            bindings = stable_proposal_bindings(frames, profile, reuse_detections())
        else:
            bindings = stable_healthbar_bindings([{**f, "path": str(root / f["path"])} for f in frames], profile, observe)
        offscreen_ids = set(offscreen["frames"])
        if offscreen_ids - {frame["id"] for frame in frames}:
            raise ValueError("Player-offscreen review contains a frame outside this run")
        appearances = []
        for team, observations in stitch_fragments(bindings, frames, profile["body_binder"]["maximum_step_pixels"], barriers):
            first, last = observations[0]["frame_index"], observations[-1]["frame_index"]
            identifier = uid("appearance", source_sha, team, [(o["frame_index"], o["bar_box"]) for o in observations])
            seed, status = pick_seed(team, observations, frames, offscreen_ids)
            row = {"id": identifier, "team": team, "first": first, "last": last, "start": frames[first]["time"],
                   "end": frames[last]["time"], "status": status, "seed": seed}
            if seed is not None:
                row["seed_preview"] = f"seed-previews/{identifier}.jpg"
                save_seed_preview(root, row, frames[seed["frame_index"]])
            appearances.append(row)
        pools = {team: sorted(champion for side, champion in membership if (side == friendly) == (team == "friendly")
                             and champion != args.player) for team in ("friendly", "enemy")}
        counts = {"frames": len(frames), "appearances": len(appearances), "statuses": dict(Counter(
            row["status"] for row in appearances)), "prepared_by_team": dict(Counter(
                row["team"] for row in appearances if row["status"] == "prepared"))}
        atomic_json(output, {"recipe": recipe, "frames": frames, "appearances": appearances, "team_pools": pools,
                             "membership": sorted(membership), "counts": counts})
        return counts


def sampled_indices(appearance, fps):
    first, last = appearance["first"], appearance["last"]
    seed = appearance["seed"]["frame_index"]
    return sorted(set(range(first, last + 1, FPS // fps)) | {last, seed})


def prompt_appearance(predictor, prompt, target):
    from .sam_multichampion import _assign_seed_ids, _mask_rows

    masks = _mask_rows(predictor.handle_request(prompt)["outputs"])
    mapping, ious = _assign_seed_ids([target], masks)
    return mapping[target["key"]], ious[target["key"]], masks


def propagate_appearance(predictor, root, output, frames, appearance, profile, fps, *,
                         seed_request=prompt_appearance, allow_incomplete=False):
    from PIL import Image

    from .sam_multichampion import TEXT_PROMPT, _mask_rows

    indices = sampled_indices(appearance, fps)
    seed = appearance["seed"]
    seed_index = indices.index(seed["frame_index"])
    target = {"key": appearance["id"], "team": appearance["team"], "box_xyxy": seed["body_box"]}
    width, height = profile["size"]
    left, top, right, bottom = seed["body_box"]
    with tempfile.TemporaryDirectory(prefix="sam-video-", dir=output) as temp:
        inputs = Path(temp)
        for n, index in enumerate(indices):
            snapshot = root / frames[index]["sam_path"]
            if file_hash(snapshot) != frames[index]["sam_sha256"]:
                raise ValueError(f"SAM input changed at frame {index}")
            (inputs / f"{n:06}.jpg").symlink_to(snapshot)
        session = predictor.handle_request({"type": "start_session", "resource_path": str(inputs),
                                            "offload_video_to_cpu": True})["session_id"]
        try:
            prompt = {"type": "add_prompt", "session_id": session, "frame_index": seed_index,
                      "text": TEXT_PROMPT,
                      "bounding_boxes": [[left / width, top / height, (right - left) / width, (bottom - top) / height]],
                      "bounding_box_labels": [1]}
            obj, seed_iou, seed_masks = seed_request(predictor, prompt, target)
            if obj is None:
                return {"status": "seed_rejected", "seed_iou": seed_iou,
                        "seed_objects": [{k: v for k, v in mask.items() if k != "mask"} for mask in seed_masks],
                        "samples": []}

            records = {}

            def save(index, masks):
                frame_index = indices[index]
                frame = frames[frame_index]
                if any(mask["mask"].shape != (height, width) for mask in masks):
                    raise ValueError("SAM returned a mask outside native source geometry")
                selected = next((mask for mask in masks if mask["object_id"] == obj), None)
                row = {"frame_index": frame_index, "time": frame["time"], "object_id": obj,
                       "visible": selected is not None, "source_frame_sha256": frame["sha256"]}
                if selected is not None:
                    relative = Path("propagated") / f"{frame_index:06}.png"
                    path = output / relative
                    path.parent.mkdir(exist_ok=True)
                    Image.fromarray(selected["mask"].astype("uint8") * 255).save(path)
                    row.update(mask_path=str(relative), mask_sha256=file_hash(path), mask_pixels=selected["pixels"])
                records[index] = row

            save(seed_index, seed_masks)
            lost_directions = []
            for direction, possible in (("forward", range(seed_index, len(indices))),
                                        ("backward", range(seed_index + 1))):
                try:
                    for response in predictor.handle_stream_request({"type": "propagate_in_video", "session_id": session,
                                                                     "start_frame_index": seed_index,
                                                                     "propagation_direction": direction}):
                        index = int(response["frame_index"])
                        if index not in possible or (index in records and index != seed_index):
                            raise ValueError("SAM returned an unexpected propagation frame")
                        if index != seed_index:
                            save(index, _mask_rows(response["outputs"]))
                except RuntimeError as exc:
                    if str(exc) != "No points are provided; please add points first":
                        raise
                    lost_directions.append(direction)
                else:
                    if allow_incomplete and any(index not in records for index in possible):
                        lost_directions.append(direction)
            complete = len(records) == len(indices)
            if not complete and not lost_directions:
                raise ValueError(f"SAM covered {len(records)} of {len(indices)} sampled frames")
            return {"status": "completed" if complete else "propagation_incomplete",
                    "seed_iou": seed_iou, "frames_expected": len(indices),
                    "lost_directions": lost_directions, "samples": [records[n] for n in sorted(records)]}
        finally:
            predictor.handle_request({"type": "close_session", "session_id": session})


def receipt_valid(receipt, directory, manifest_hash, sam_pin=None):
    if (receipt.get("manifest_sha256") != manifest_hash
            or (sam_pin is not None and receipt.get("sam") != sam_pin)
            or receipt["status"] not in {"completed", "seed_rejected", "propagation_incomplete"}):
        return False
    for row in receipt["samples"]:
        for key, sha in (("mask_path", "mask_sha256"), ("image_path", "image_sha256"),
                         ("crop_mask_path", "crop_mask_sha256")):
            if key in row:
                path = directory / row[key]
                if not path.is_file() or file_hash(path) != row[sha]:
                    return False
    return True


def mask_appearances(args):
    from .body_level import LEVEL_READER_VERSION
    from .local_models import checkpoint

    root = args.root.resolve()
    manifest_path = root / "appearances.json"
    manifest_hash = file_hash(manifest_path)
    prepared = read_json(manifest_path)
    profile = prepared["recipe"]["profile"]
    rows = [row for row in prepared["appearances"] if row["status"] == "prepared"]
    expected_teams = {row["team"] for row in rows}
    if (prepared["recipe"].get("level_reader") != LEVEL_READER_VERSION
            or any(row["seed"].get("level") not in range(1, 19) for row in rows)):
        raise ValueError("Prepare appearances with the current champion-level detector before running SAM")
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        raise ValueError("Choose a zero-based shard within the shard count")
    if args.smoke:
        longest = sorted(rows, key=lambda row: (row["first"] - row["last"], row["id"]))
        rows = [row for team in sorted(expected_teams) for row in [a for a in longest if a["team"] == team][:3]]
    else:
        # Greedy frame-count balancing keeps every appearance wholly on one worker.
        shards = [[] for _ in range(args.shards)]
        loads = [0] * args.shards
        for row in sorted(rows, key=lambda row: (row["first"] - row["last"], row["id"])):
            i = min(range(args.shards), key=loads.__getitem__)
            shards[i].append(row)
            loads[i] += row["last"] - row["first"] + 1
        rows = shards[args.shard]

    output = root / "masks"
    output.mkdir(exist_ok=True)
    if args.require_smoke and not any(
        row.get("manifest_sha256") == manifest_hash and row.get("mask_prompt") == MASK_PROMPT
        and row.get("smoke_passed") is True
        for row in map(read_json, output.glob("worker-*-smoke.json"))
    ):
        raise ValueError("The prepared appearance manifest needs a successful SAM smoke result")
    started = time.monotonic()
    if not rows:
        return {"appearances": 0, "statuses": {}, "wall_seconds": 0}
    model_root, pin = checkpoint(args.cache, "tracking")
    if pin["specification"] != prepared["recipe"]["sam"]:
        raise ValueError("Tracking checkpoint differs from the prepared pin")

    import gc
    import torch
    from .sam_compat import load_multiplex_predictor

    predictor = None
    statuses = Counter()
    completed_teams = set()
    job_id = os.environ.get("SLURM_JOB_ID")
    for row in rows:
        identifier = row["id"]
        target = output / identifier
        with stage_lock(output / (identifier + ".lock")):
            path = target / "receipt.json"
            if path.exists():
                receipt = read_json(path)
                if receipt.get("mask_prompt") == MASK_PROMPT and receipt_valid(receipt, target, manifest_hash, pin):
                    statuses[receipt["status"]] += 1
                    if receipt["status"] == "completed":
                        completed_teams.add(row["team"])
                    continue
            if predictor is None:
                predictor, model_load = load_multiplex_predictor(model_root / "sam3.1_multiplex.pt")
            tic = time.monotonic()
            target.mkdir(exist_ok=True)
            outcome = None
            best_partial = None
            for fps in (16, 8, 4):
                with tempfile.TemporaryDirectory(prefix=f".{identifier}-", dir=output) as temp:
                    work = Path(temp)
                    try:
                        outcome = propagate_appearance(predictor, root, work, prepared["frames"], row, profile, fps)
                    except RuntimeError as exc:
                        if "out of memory" not in str(exc).lower():
                            outcome = {"status": "failed", "error": str(exc), "samples": []}
                        else:
                            print(f"{identifier}: memory exhausted at {fps} FPS", flush=True)
                    except ValueError as exc:
                        outcome = {"status": "failed", "error": str(exc), "samples": []}
                    if outcome is not None:
                        if outcome["status"] == "seed_rejected" and best_partial is not None:
                            outcome = None
                            continue
                        incomplete = outcome["status"] == "propagation_incomplete"
                        if incomplete:
                            coverage = len(outcome["samples"]) / outcome["frames_expected"]
                            if best_partial is not None and coverage <= best_partial[0]:
                                outcome = None
                                continue
                            best_partial = coverage, fps, outcome
                        # Move artifacts before publishing their receipt.
                        if (target / "propagated").exists():
                            shutil.rmtree(target / "propagated")
                        if (work / "propagated").exists():
                            (work / "propagated").rename(target / "propagated")
                        if incomplete:
                            print(f"{identifier}: propagation incomplete at {fps} FPS", flush=True)
                            outcome = None
                            continue
                        break
                gc.collect()
                torch.cuda.empty_cache()
            if outcome is None and best_partial is not None:
                _, fps, outcome = best_partial
            elif outcome is None:
                outcome = {"status": "failed", "error": "SAM exhausted memory at 4 FPS", "samples": []}
            receipt = {**outcome, "appearance_id": identifier, "manifest_sha256": manifest_hash, "sam": pin,
                       "model_load": model_load, "fps": fps, "job_id": job_id, "mask_prompt": MASK_PROMPT,
                       "wall_seconds": time.monotonic() - tic}
            atomic_json(path, receipt)
            statuses[outcome["status"]] += 1
            if outcome["status"] == "completed":
                completed_teams.add(row["team"])
            print(f"{identifier}: {outcome['status']} at {fps} FPS ({len(outcome['samples'])} frames)", flush=True)
    summary = {"appearances": len(rows), "statuses": dict(statuses), "job_id": job_id,
               "mask_prompt": MASK_PROMPT,
               "wall_seconds": time.monotonic() - started,
               "peak_gpu_bytes": torch.cuda.max_memory_allocated(), "manifest_sha256": manifest_hash}
    if args.smoke:
        summary["smoke_passed"] = completed_teams == expected_teams and not statuses["failed"]
    atomic_json(output / f"worker-{job_id or 'local'}-{'smoke' if args.smoke else args.shard}.json", summary)
    return summary


def choose_masks(samples, directory):
    from .body_operational import verified_crop

    eligible = []
    for row in samples:
        if not row.get("crop_mask_path"):
            continue
        mask = verified_crop(directory / row["crop_mask_path"], row["crop_mask_sha256"])
        mask = mask.convert("L").point(lambda value: 255 if value > 127 else 0)
        box = mask.getbbox()
        if box is None:
            continue
        pixels = mask.histogram()[255]
        fill = pixels / ((box[2] - box[0]) * (box[3] - box[1]))
        eligible.append({**row, "foreground_pixels": pixels, "foreground_fraction": fill,
                         "mask_quality": fill * pixels ** .5})
    by_quality = sorted(eligible, key=lambda row: (-row["mask_quality"], row["frame_index"]))
    spaced, remaining = [], []
    for row in by_quality:
        if len(spaced) == 5:
            break
        if all(abs(row["time"] - prior["time"]) >= .25 for prior in spaced):
            spaced.append(row)
        else:
            remaining.append(row)
    return sorted((spaced + remaining)[:5], key=lambda row: row["frame_index"])


def materialize_clean_crops(root, directory, samples, frames, profile):
    from .body_operational import body_mask, verified_crop

    result = []
    output = directory / "crops"
    output.mkdir(exist_ok=True)
    for row in samples:
        if not row["visible"]:
            continue
        index = row["frame_index"]
        frame = frames[index]
        if frame["sha256"] != row["source_frame_sha256"]:
            raise ValueError(f"Mask source differs from native frame {index}")
        image = verified_crop(root / frame["path"], frame["sha256"])
        clean, removed = body_mask(image, directory / row["mask_path"], profile, frame["proposals"])
        box = clean.getbbox()
        if not box:
            continue
        crop = output / f"{index:06}-image.png"
        mask = output / f"{index:06}-mask.png"
        image.crop(box).save(crop)
        clean.crop(box).save(mask)
        result.append({**row, "image_path": str(crop.relative_to(directory)), "image_sha256": file_hash(crop),
                       "crop_mask_path": str(mask.relative_to(directory)), "crop_mask_sha256": file_hash(mask),
                       "ui_removed_fraction": removed})
    return result


def rank_masks(matcher, queries, pool):
    import numpy as np

    groups = [(champion, skin, views) for champion in pool
              for skin, views in sorted(matcher.entries.get(champion, {}).items())]
    if not queries or not groups:
        return [], None
    starts = np.cumsum([0, *(len(views) for _, _, views in groups[:-1])])
    entries = [view for _, _, views in groups for view in views]
    scores = np.maximum.reduceat(matcher.score_views(queries, entries), starts, axis=1)
    maxima = scores == scores.max(axis=1, keepdims=True)
    unique = maxima & (maxima.sum(axis=1, keepdims=True) == 1)
    ranking = []
    for i, (champion, skin, views) in enumerate(groups):
        ranking.append({"champion": champion, "skin_id": skin, "skin_name": views[0].get("skin_name"),
                        "score": float(scores[:, i].mean()), "mask_wins": int(unique[:, i].sum()),
                        "mask_scores": scores[:, i].tolist()})
    ranking.sort(key=lambda row: (-row["score"], row["champion"], row["skin_id"]))
    top = ranking[0]
    unique_winner = len(ranking) == 1 or top["score"] > ranking[1]["score"]
    vote = top if len(queries) >= 3 and unique_winner and 2 * top["mask_wins"] > len(queries) else None
    return ranking, vote


class AppearanceLocks:
    def __init__(self):
        self.assigned = {"friendly": {}, "enemy": {}}
        self.pending = defaultdict(list)
        self.events = []

    def observe(self, appearance, candidate):
        if candidate is None:
            return
        team, identifier = appearance["team"], appearance["id"]
        name, skin = candidate["champion"], candidate["skin_id"]
        assigned = self.assigned[team]
        if name in assigned:
            current = assigned[name]
            if current["skin_id"] == skin:
                return
            del assigned[name]
            for key in list(self.pending):
                if key[:2] == (team, name):
                    del self.pending[key]
            self.events.append({"event": "suspended", "team": team, "champion": name,
                                "previous": current, "appearance_id": identifier, "challenger_skin_id": skin})
        votes = self.pending[team, name, skin]
        if any(row["id"] == identifier for row in votes):
            return
        prior = next((row for row in votes if appearance["start"] - row["end"] >= 1), None)
        votes.append(appearance)
        if prior is not None:
            assigned[name] = {"skin_id": skin, "supporting_appearances": [prior["id"], identifier]}
            self.events.append({"event": "locked", "team": team, "champion": name, **assigned[name]})


def write_mask_preview(path, selected, directory):
    from PIL import Image, ImageDraw, ImageOps

    sheet = Image.new("RGB", (160 * len(selected), 180), (35, 35, 35))
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(selected):
        with Image.open(directory / row["image_path"]) as source, Image.open(directory / row["crop_mask_path"]) as mask:
            crop = Image.composite(source.convert("RGB"), Image.new("RGB", source.size, (127,) * 3), mask.convert("L"))
        sheet.paste(ImageOps.contain(crop, (160, 152)), (160 * i, 0))
        draw.text((160 * i + 5, 155), f'{row["time"]:.2f}s', fill="white")
    path.parent.mkdir(exist_ok=True)
    sheet.save(path, quality=90)


def match_appearances(args):
    from .body_frame_matching import SkinMatcher, skin_lock_scope
    from .body_operational import skin_pool_report
    from .body_slot_context import skim_revision_matches

    root = args.root.resolve()
    manifest = root / "appearances.json"
    manifest_hash = file_hash(manifest)
    prep = read_json(manifest)
    recipe, pools = prep["recipe"], prep["team_pools"]
    revision = dict(recipe["scoreboard"])
    if args.scoreboard_skim is not None:
        revision["root"] = str(args.scoreboard_skim.resolve())
    if not skim_revision_matches(revision):
        raise ValueError("Scoreboard evidence changed after appearance preparation")
    lock = args.siglip_lock if args.encoder == "siglip" else args.encoder_lock
    lock_hash = digest(read_json(lock))
    if args.encoder == "siglip" and recipe["siglip_lock_hash"] != lock_hash:
        raise ValueError("SigLIP pin changed after appearance preparation")
    gallery_path = args.gallery.resolve()
    gallery = read_json(gallery_path)
    gallery_file_hash = file_hash(gallery_path)
    candidates = set().union(*pools.values())
    gallery = [row for row in gallery if row["champion"] in candidates]
    class Roster:
        rosters = {"pilot": {"champions": sorted(candidates)}}

    patch = recipe["patch"]
    coverage = skin_pool_report(args.cache, {"gallery": gallery, "reference_version": patch[0],
                                             "communitydragon_version": patch[1]}, Roster)
    if coverage["status"] != "pinned":
        raise ValueError("Stage the pinned skin catalog before matching appearances")
    gallery = [row for row in gallery if str(row.get("skin_id")) in
               coverage["champions"][row["champion"]]["eligible_skin_ids"]]
    matcher = SkinMatcher(gallery, args.cache, lock, lock_hash, patch, encoder_name=args.encoder)
    missing = {team: [name for name in pool if not matcher.entries.get(name)] for team, pool in pools.items()}
    scope = skin_lock_scope(matcher, pools, recipe["source_sha256"], recipe["roster_id"])
    locks, outcomes, workers = AppearanceLocks(), [], {}
    for appearance in sorted(prep["appearances"], key=lambda row: (row["end"], row["start"], row["id"])):
        identifier, team = appearance["id"], appearance["team"]
        outcome = {**appearance, "full_ranking": [], "ranking": []}
        if appearance["status"] != "prepared":
            outcomes.append(outcome)
            continue
        directory = root / "masks" / identifier
        path = directory / "receipt.json"
        if not path.is_file():
            outcome["status"] = "mask_pending"
            outcomes.append(outcome)
            continue
        receipt = read_json(path)
        if (receipt.get("manifest_sha256") != manifest_hash or receipt.get("mask_prompt") != MASK_PROMPT
                or receipt.get("sam", {}).get("specification") != recipe["sam"]):
            raise ValueError(f"SAM result differs from the prepared run: {identifier}")
        workers[receipt.get("job_id") or "local"] = workers.get(receipt.get("job_id") or "local", 0) + receipt["wall_seconds"]
        outcome.update(status=receipt["status"], mask_fps=receipt["fps"], sam_receipt=str(path.relative_to(root)))
        if receipt["status"] == "failed":
            outcome["mask_error"] = receipt.get("error")
        elif not receipt_valid(receipt, directory, manifest_hash):
            raise ValueError(f"SAM artifacts changed: {identifier}")
        if receipt["status"] not in {"completed", "propagation_incomplete"}:
            outcomes.append(outcome)
            continue
        with stage_lock(directory / ".crops.lock"):
            crop_receipt_path = directory / "crop-receipt.json"
            crop_key = digest([receipt, recipe["profile"]])
            cached = read_json(crop_receipt_path) if crop_receipt_path.is_file() else None
            if cached and receipt_valid(cached, directory, crop_key):
                crops = cached["samples"]
            else:
                crops = materialize_clean_crops(root, directory, receipt["samples"], prep["frames"], recipe["profile"])
                atomic_json(crop_receipt_path, {"manifest_sha256": crop_key, "status": "completed", "samples": crops})
            selected = choose_masks(crops, directory)
        queries = [(directory / row["image_path"], directory / row["crop_mask_path"],
                    row["image_sha256"], row["crop_mask_sha256"]) for row in selected]
        full, vote = rank_masks(matcher, queries, pools[team])
        if receipt["status"] != "completed" or missing[team]:
            vote = None
        locks.observe(appearance, vote)
        assigned = locks.assigned[team]
        restricted = [row for row in full if row["champion"] not in assigned
                      or row["skin_id"] == assigned[row["champion"]]["skin_id"]]
        outcome.update(status=("ranked" if receipt["status"] == "completed" else "ranked_partial") if full else
                              "missing_renders" if selected else "no_usable_masks",
                       selected_masks=selected, full_ranking=full, ranking=restricted,
                       vote=vote)
        if selected:
            preview = Path("mask-previews") / (identifier + ".jpg")
            write_mask_preview(root / preview, selected, directory)
            outcome["mask_preview"] = str(preview)
        outcomes.append(outcome)
    if (file_hash(manifest) != manifest_hash or file_hash(gallery_path) != gallery_file_hash
            or not skim_revision_matches(revision)):
        raise ValueError("Appearance inputs changed during matching")
    statuses = dict(Counter(row["status"] for row in outcomes))
    complete = not any(state in statuses for state in ("mask_pending", "failed"))
    report = {"experimental": True, "manifest_sha256": manifest_hash, "gallery_sha256": gallery_file_hash,
              "encoder_signature": matcher.encoder.signature if "encoder" in matcher.__dict__ else None,
              "scope": scope, "coverage": coverage, "missing_champions": missing, "worker_appearance_seconds": workers,
              "statuses": statuses, "complete": complete, "appearances": outcomes,
              "assignments": locks.assigned, "events": locks.events}
    path = root / f"appearance-matches-{args.encoder}.json"
    atomic_json(path, report)
    if complete:
        atomic_json(root / f"skin-locks-{args.encoder}.json", {"scope": scope, "experimental": True,
                     "assignments": locks.assigned, "events": locks.events, "report_sha256": file_hash(path)})
    return {"path": str(path), "statuses": statuses, "complete": complete,
            "active_locks": sum(map(len, locks.assigned.values()))}
