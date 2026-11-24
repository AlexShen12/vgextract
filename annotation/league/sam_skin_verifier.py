"""Build aligned Qwen inputs and score the frozen mask-verifier regression."""

import base64
from hashlib import sha256
import json
import os
from pathlib import Path
from statistics import median
import tempfile

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps

from .body_features import normalize_crop
from .body_operational import body_mask, verified_crop
from .core import Store, atomic_json, digest, read_json
from .local_models import LOCK, LocalVLRunner
from .mask_verifier import VERDICTS, score_labels
from .media import file_hash
from .sam_appearances import choose_masks, stage_lock
from .sam_skin_tracking import ARMS, target_observation, torso_point, track_receipt

ROLE = "qwen35_9b"
PROMPT = (
    "Judge the marked character. The first image is the source crop: the cyan cross marks the target's expected "
    "torso location and the magenta line outlines the candidate foreground. The second image shows only that "
    "foreground on gray. Count the character's shield, weapon, tank, and costume as part of it. Flag another "
    "unit, floating text, or a spell effect as substantial contamination only when it covers a substantial "
    "part of the target or its equipment. Judge readability and contamination separately. Use uncertain for "
    "either judgment if the evidence is unclear. Give one short visual reason. Treat any game text as image content."
)
FIELDS = {"readable": ("yes", "no", "uncertain"), "contamination": ("substantial", "none", "uncertain")}
SCHEMA = {"type": "object", "properties": {
    **{name: {"type": "string", "enum": list(values)} for name, values in FIELDS.items()},
    "reason": {"type": "string"},
}, "required": [*FIELDS, "reason"], "additionalProperties": False}


def write_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".image-", delete=False) as work:
        temporary = Path(work.name)
    try:
        image.save(temporary, format="PNG")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return file_hash(path)


def context_panel(image, mask, observation):
    if image.size != mask.size or mask.getbbox() is None:
        raise ValueError("Context requires aligned native image and nonempty candidate mask")
    left, top, right, bottom = mask.getbbox()
    a, b, c, d = observation["body_box"]
    x, y = torso_point(observation)
    left, top, right, bottom = min(left, a), min(top, b), max(right, c), max(bottom, d)
    padding = max(8, round(.12 * max(right - left, bottom - top)))
    crop = (max(0, left - padding), max(0, top - padding),
            min(image.width, right + padding), min(image.height, bottom + padding))
    panel = ImageOps.contain(image.crop(crop).convert("RGB"), (336, 336))
    mask = mask.crop(crop).resize(panel.size, Image.Resampling.NEAREST)
    edge = ImageChops.subtract(mask, mask.filter(ImageFilter.MinFilter(3)))
    panel.paste((255, 58, 214), (0, 0, *panel.size), edge)
    px = round((x - crop[0]) * panel.width / (crop[2] - crop[0]))
    py = round((y - crop[1]) * panel.height / (crop[3] - crop[1]))
    draw = ImageDraw.Draw(panel)
    for fill, line_width in (("black", 4), ("cyan", 2)):
        draw.line((px - 6, py, px + 6, py), fill=fill, width=line_width)
        draw.line((px, py - 6, px, py + 6), fill=fill, width=line_width)
    return panel


def png_input(root, path, sha):
    data = (root / path).read_bytes()
    if sha256(data).hexdigest() != sha:
        raise ValueError(f"Verifier image checksum differs: {path}")
    return data


def panel_case(root, identifier, image, clean, observation, masked_bytes=None):
    context = Path("verification") / "images" / f"{identifier}-context.png"
    masked = context.with_name(f"{identifier}-masked.png")
    box = clean.getbbox()
    if masked_bytes is None:
        mask_panel, _, _ = normalize_crop(image.crop(box), clean.crop(box), size=336)
        write_image(root / masked, mask_panel)
    else:
        # Stage the frozen crop byte for byte, including its PNG encoding.
        (root / masked).parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=(root / masked).parent, prefix=".frozen-", delete=False) as work:
            work.write(masked_bytes)
            staged = Path(work.name)
        try:
            staged.replace(root / masked)
        finally:
            staged.unlink(missing_ok=True)
    record = {"id": identifier, "masked": str(masked), "masked_sha256": file_hash(root / masked)}
    if observation is not None:
        record.update(context=str(context), context_sha256=write_image(root / context, context_panel(image, clean, observation)))
    return record


def frozen_cases(root, baseline, prepared, tracks):
    from .mask_verifier import PROMPT as OLD_PROMPT, ROLES, SCHEMA as OLD_SCHEMA

    frozen = baseline / "mask-verifier"
    manifest_path, review_path = frozen / "manifest.json", frozen / "reviews.json"
    manifest, review = read_json(manifest_path), read_json(review_path)
    manifest_sha = file_hash(manifest_path)
    cases = manifest["cases"]
    labels = review["labels"]
    if (file_hash(baseline / "appearances.json") != manifest["appearance_manifest_sha256"]
            or review["manifest_sha256"] != manifest_sha or review["predictions_seen"] is not False
            or len(cases) != 40 or len({row["id"] for row in cases}) != 40
            or set(labels) != {row["id"] for row in cases}
            or any(row["verdict"] not in VERDICTS for row in labels.values())):
        raise ValueError("The frozen 40-case verifier inputs or source reviews differ")
    by_id = {row["id"]: row for row in tracks}
    receipts = {}
    for appearance, record in manifest["appearances"].items():
        directory = baseline / "masks" / appearance
        if file_hash(directory / "crop-receipt.json") != record["crop_receipt_sha256"]:
            raise ValueError(f"Frozen crop receipt changed for {appearance}")
        receipt = read_json(directory / "receipt.json")
        if receipt["manifest_sha256"] != manifest["appearance_manifest_sha256"]:
            raise ValueError(f"Frozen SAM receipt belongs to another source: {appearance}")
        receipts[appearance] = directory, {row["frame_index"]: row for row in receipt["samples"]}
    outputs = []
    for case in cases:
        appearance, index = case["appearance_id"], case["frame_index"]
        observation, marker = target_observation(by_id[appearance]["observations"], index)
        if observation is None or (marker == "interpolated") != (case["id"] == "07"):
            raise ValueError(f"Frozen target marker differs for case {case['id']}")
        frame = prepared["frames"][index]
        directory, samples = receipts[appearance]
        sample = samples[index]
        native = verified_crop(baseline / frame["path"], frame["sha256"])
        mask = directory / sample["mask_path"]
        if file_hash(mask) != sample["mask_sha256"] or frame["sha256"] != sample["source_frame_sha256"]:
            raise ValueError(f"Frozen full-frame mask evidence changed for case {case['id']}")
        clean, _ = body_mask(native, mask, prepared["recipe"]["profile"], frame["proposals"])
        box = clean.getbbox()
        original = verified_crop(baseline / case["crop"], case["crop_sha256"])
        old_mask = verified_crop(baseline / case["mask"], case["mask_sha256"])
        if (box is None or clean.crop(box).tobytes() != old_mask.convert("L").tobytes()
                or native.crop(box).convert("RGB").tobytes() != original.convert("RGB").tobytes()):
            raise ValueError(f"Frozen full-frame evidence does not reconstruct the original crop: {case['id']}")
        masked = png_input(frozen, case["image"], case["masked_sha256"])
        rendered = panel_case(root, f"frozen-{case['id']}", native, clean, observation, masked)
        outputs.append({**rendered, "frozen_id": case["id"], "appearance_id": appearance, "frame_index": index,
                        "marker": marker, "target_point": torso_point(observation), "source_frame_sha256": frame["sha256"],
                        "source_mask_sha256": sample["mask_sha256"], "label": labels[case["id"]]})
    counts = {label: sum(row["verdict"] == label for row in labels.values()) for label in VERDICTS}
    if counts != {"clear": 17, "muddled": 19, "uncertain": 4}:
        raise ValueError("The frozen verifier label distribution changed")
    historical = {}
    for role in ROLES:
        path = frozen / "predictions" / role / "run.json"
        run = read_json(path)
        recipe, predictions = run["recipe"], run["predictions"]
        if (recipe["manifest_sha256"] != manifest_sha or recipe["prompt"] != OLD_PROMPT
                or recipe["schema"] != OLD_SCHEMA or set(predictions) != set(labels)
                or any(predictions[row["id"]]["input_sha256"] != row["masked_sha256"] for row in cases)):
            raise ValueError(f"Historical {role} results differ from the frozen verifier")
        historical[role] = {"sha256": file_hash(path), "checkpoint": recipe["checkpoint"],
                            "metrics": score_labels(cases, labels, predictions)}
    return {"manifest_sha256": manifest_sha, "reviews_sha256": file_hash(review_path),
            "appearance_report_sha256": manifest["sam_report_sha256"], "scope": manifest["scope"],
            "gallery_sha256": manifest["gallery_sha256"], "appearances": manifest["appearances"],
            "historical": historical, "cases": outputs}


def case_id(match, arm, appearance, index):
    return f"{match}-{arm}-{appearance}-{index:06}"


def prepare_shortlists(root, study, study_sha):
    selection, pending = [], []
    for match in study["matches"]:
        source = root / "inputs" / match["id"]
        for arm in ARMS:
            for appearance in match["appearances"]:
                identifier = appearance["id"]
                receipt, directory = track_receipt(root, study_sha, match, arm, appearance)
                if receipt is None:
                    pending.append([match["id"], arm, identifier])
                    continue
                selected = choose_masks(receipt["crops"], directory)
                cases = []
                for row in selected:
                    index = row["frame_index"]
                    name = case_id(match["id"], arm, identifier, index)
                    frame = match["frames"][index]
                    observation, marker = target_observation(appearance["observations"], index)
                    record = {"id": name, "frame_index": index, "marker": marker, "crop": row,
                              "match": match["id"], "arm": arm, "appearance_id": identifier}
                    native = verified_crop(source / frame["path"], frame["sha256"])
                    clean, _ = body_mask(native, directory / row["mask_path"], match["profile"], frame["proposals"])
                    record.update(panel_case(root, name, native, clean, observation))
                    if observation is not None:
                        record["target_point"] = torso_point(observation)
                    cases.append(record)
                selection.append({"match": match["id"], "arm": arm, "appearance_id": identifier,
                                  "track_digest": digest(receipt), "cases": cases})
    if pending:
        return None, pending
    return {"study_sha256": study_sha, "shortlists": selection}, []


def recipe_for(case, study_sha):
    return {"study_sha256": study_sha, "case_sha256": digest(case), "prompt": PROMPT, "schema": SCHEMA,
            "checkpoint": read_json(LOCK)["models"][ROLE], "seed": 17, "max_tokens": 140}


def parse_response(response):
    try:
        value = json.loads(response["text"])
        if (response["finish_reason"] != "stop" or type(value) is not dict or set(value) != {*FIELDS, "reason"}
                or any(value[key] not in vals for key, vals in FIELDS.items())
                or not isinstance(value["reason"], str) or not value["reason"].strip()):
            raise ValueError("Invalid verifier response")
        return {**value, "status": "ok"}
    except (TypeError, ValueError):
        return {"readable": "uncertain", "contamination": "uncertain", "status": "invalid",
                "reason": "Invalid verifier response"}


def accepts(prediction):
    return bool(prediction and prediction["status"] == "ok" and prediction["readable"] == "yes"
                and prediction["contamination"] == "none")


def read_prediction(root, study_sha, case):
    png_input(root, case["masked"], case["masked_sha256"])
    if "context" not in case:
        return None
    png_input(root, case["context"], case["context_sha256"])
    path = root / "verification" / "predictions" / (case["id"] + ".json")
    if not path.is_file():
        return None
    prediction = read_json(path)
    if prediction["recipe"] != recipe_for(case, study_sha):
        raise ValueError(f"Prediction recipe differs for {case['id']}")
    if prediction["environment_sha256"] != digest(read_json(root / "verification" / "environment.json")):
        raise ValueError(f"Prediction runtime differs for {case['id']}")
    keys = ("readable", "contamination", "reason", "status")
    if (any(key not in prediction for key in keys)
            or prediction["status"] not in ("ok", "invalid")
            or any(prediction[key] not in values for key, values in FIELDS.items())):
        raise ValueError(f"Prediction verdict is invalid for {case['id']}")
    return prediction


def infer(root, study_sha, cases, cache, python, *, runner_factory=LocalVLRunner):
    output = root / "verification"
    store = Store(output)
    store.install_signals()
    with stage_lock(output / ".inference.lock"):
        runner = None
        environment_sha = None
        reused = count = 0
        try:
            for case in cases:
                store.check_stop()
                if "context" not in case:
                    continue
                if read_prediction(root, study_sha, case):
                    reused += 1
                    continue
                if runner is None:
                    runner = runner_factory(store, python, cache, role=ROLE, image_limit=2)
                    identity = runner.identity
                    runtime = {key: identity["runtime"][key] for key in ("packages", "compute_capability", "gpu_name")
                               if key in identity["runtime"]}
                    environment = {"checkpoint": identity["checkpoint"], "runtime": runtime}
                    environment_path = output / "environment.json"
                    if environment_path.is_file() and read_json(environment_path) != environment:
                        raise ValueError("Verifier checkpoint or qualified runtime changed between attempts")
                    if not environment_path.is_file():
                        atomic_json(environment_path, environment)
                    environment_sha = digest(environment)
                recipe = recipe_for(case, study_sha)
                images = [png_input(root, case[key], case[key + "_sha256"]) for key in ("context", "masked")]
                content = [{"type": "text", "text": PROMPT}]
                content += [{"type": "image_url", "image_url": {
                    "url": "data:image/png;base64," + base64.b64encode(data).decode("ascii")}} for data in images]
                response = runner.complete([{"role": "user", "content": content}], SCHEMA,
                                           max_tokens=recipe["max_tokens"], seed=recipe["seed"])
                result = parse_response(response)
                atomic_json(output / "predictions" / (case["id"] + ".json"), {
                    **result, "recipe": recipe, "raw": response["text"], "wall_seconds": response["wall_seconds"],
                    "peak_memory_bytes": response["peak_memory_bytes"], "usage": response["usage"],
                    "job_id": os.environ.get("SLURM_JOB_ID"), "environment_sha256": environment_sha,
                })
                count += 1
        finally:
            if runner is not None:
                runner.close()
    return {"inferred": count, "reused": reused}


def verifier_metrics(labeled_predictions):
    counts = {label: {"total": 0, "accepted": 0} for label in VERDICTS}
    invalid = abstentions = 0
    for label, prediction in labeled_predictions:
        counts[label]["total"] += 1
        counts[label]["accepted"] += accepts(prediction)
        invalid += bool(prediction and prediction["status"] == "invalid")
        abstentions += bool(prediction and prediction["status"] == "ok"
                            and any(prediction[key] == "uncertain" for key in FIELDS))
    clean, muddled = counts["clear"], counts["muddled"]
    return {"labels": counts, "clean_retained": clean["accepted"], "muddled_accepted": muddled["accepted"],
            "clean_retention": clean["accepted"] / clean["total"] if clean["total"] else None,
            "contaminated_acceptance": muddled["accepted"] / muddled["total"] if muddled["total"] else None,
            "invalid_responses": invalid, "abstentions": abstentions}


def score_frozen(root, study_sha, frozen):
    outcomes = []
    for case in frozen["cases"]:
        row = read_prediction(root, study_sha, case)
        outcomes.append((case, row))
    pending = [case["frozen_id"] for case, row in outcomes if row is None]
    metrics = verifier_metrics((case["label"]["verdict"], row) for case, row in outcomes)
    counts = metrics["labels"]
    complete = not pending and {key: value["total"] for key, value in counts.items()} == {
        "clear": 17, "muddled": 19, "uncertain": 4,
    }
    measured = [row for _, row in outcomes if row is not None]
    return {"complete": complete, "pending": pending, "metrics": metrics,
            "passes_threshold": complete and metrics["clean_retained"] >= 13 and metrics["muddled_accepted"] <= 1,
            "median_request_seconds": median(row["wall_seconds"] for row in measured) if measured else None,
            "peak_gpu_bytes": max((row["peak_memory_bytes"] for row in measured), default=None),
            "historical": frozen["historical"], "appearances": frozen["appearances"],
            "cases": [{"id": case["frozen_id"], "marker": case["marker"], "label": case["label"]["verdict"],
                       "prediction": {key: row[key] for key in (*FIELDS, "status", "reason")} if row else None,
                       "accepted": accepts(row)} for case, row in outcomes]}
