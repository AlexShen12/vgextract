"""Compare frozen Qwen mask-quality predictions without modifying skin matching."""

import argparse
import base64
from collections import Counter
import json
import os
from pathlib import Path
from statistics import median
import sys

from .body_features import normalize_crop
from .body_operational import verified_crop
from .core import Store, atomic_json, digest, read_json
from .media import file_hash
from .sam_appearances import stage_lock

APPEARANCES = ("appearance_db6654019460a6101930", "appearance_43efe796f7017448bbe4")
ROLES = ("qwen35_9b", "intake")
VERDICTS = ("clear", "muddled", "uncertain")
SCHEMA = {"type": "object", "properties": {
    "verdict": {"type": "string", "enum": list(VERDICTS)},
    "reason": {"type": "string"},
}, "required": ["verdict", "reason"], "additionalProperties": False}
PROMPT = (
    "Judge the visible foreground in this masked gameplay crop. The flat gray is background. "
    "Return clear if it depicts one recognizable game character with a readable body silhouette and no material "
    "obstruction. Return muddled if other bodies, minions, numbers or UI, effects, or missing parts obscure the "
    "character or break up its silhouette. Small stray details can be ignored if the character is still clearly "
    "readable. Return uncertain if you cannot tell. Give the verdict and one short visual reason. "
    "Read any game text as image content. Output JSON."
)


def evenly_spaced(rows, count):
    if len(rows) < count:
        raise ValueError(f"Need {count} additional crops, found {len(rows)}")
    return [rows[round(i * (len(rows) - 1) / (count - 1))] for i in range(count)]


def prepare(run, output):
    from PIL import Image, ImageDraw

    report_path = run / "appearance-matches-siglip.json"
    report = read_json(report_path)
    matches = {row["id"]: row for row in report["appearances"]}
    prepared_path = output / "manifest.json"
    if prepared_path.exists():
        ready = read_json(prepared_path)
        for row in ready["cases"]:
            if file_hash(output / row["image"]) != row["masked_sha256"]:
                raise ValueError("Frozen verifier image changed")
        return {"cases": len(ready["cases"]), "path": str(prepared_path)}
    image_dir = output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    cases, appearances = [], {}
    for appearance in APPEARANCES:
        row = matches[appearance]
        if row["status"] != "ranked" or len(row["selected_masks"]) != 5:
            raise ValueError("The verifier needs the two original five-mask appearance rankings")
        relative = Path("masks") / appearance
        directory = run / relative
        receipt_path = directory / "crop-receipt.json"
        receipt = read_json(receipt_path)
        samples = {sample["frame_index"]: sample for sample in receipt["samples"]}
        baseline = [sample["frame_index"] for sample in row["selected_masks"]]
        for sample in row["selected_masks"]:
            original = samples[sample["frame_index"]]
            if any(original[field] != sample[field] for field in ("image_sha256", "crop_mask_sha256")):
                raise ValueError("The baseline ranking references different crop pixels")
        extras = evenly_spaced([samples[n] for n in sorted(samples) if n not in baseline], 15)
        selected = sorted([samples[n] for n in baseline] + extras, key=lambda sample: sample["frame_index"])
        appearances[appearance] = {"team": row["team"], "start": row["start"], "end": row["end"],
                                   "baseline_frames": baseline, "baseline_ranking": row["full_ranking"],
                                   "crop_receipt_sha256": file_hash(receipt_path)}
        for sample in selected:
            identifier = f'{len(cases) + 1:02}'
            image_path = relative / sample["image_path"]
            mask_path = relative / sample["crop_mask_path"]
            image = verified_crop(run / image_path, sample["image_sha256"])
            mask = verified_crop(run / mask_path, sample["crop_mask_sha256"])
            rendered, _, _ = normalize_crop(image, mask, size=336)
            rendered_path = Path("images") / f"{identifier}.png"
            rendered.save(output / rendered_path)
            cases.append({"id": identifier, "appearance_id": appearance, "frame_index": sample["frame_index"],
                          "time": sample["time"], "image": str(rendered_path),
                          "masked_sha256": file_hash(output / rendered_path), "crop": str(image_path),
                          "mask": str(mask_path), "crop_sha256": sample["image_sha256"],
                          "mask_sha256": sample["crop_mask_sha256"]})
    # Review sheets use the exact input image and anonymous ID.
    for offset in range(0, len(cases), 10):
        sheet = Image.new("RGB", (1200, 516), (35, 35, 35))
        draw = ImageDraw.Draw(sheet)
        for i, row in enumerate(cases[offset:offset + 10]):
            x, y = (i % 5) * 240, (i // 5) * 258
            with Image.open(output / row["image"]) as image:
                sheet.paste(image.resize((240, 240), Image.Resampling.LANCZOS), (x, y))
            draw.text((x + 8, y + 242), row["id"], fill="white")
        sheet.save(output / f"review-{offset // 10 + 1}.png")
    atomic_json(prepared_path, {"schema": 1, "sam_report_sha256": file_hash(report_path),
                                "appearance_manifest_sha256": report["manifest_sha256"],
                                "gallery_sha256": report["gallery_sha256"], "scope": report["scope"],
                                "appearances": appearances, "cases": cases})
    return {"cases": len(cases), "path": str(prepared_path)}


def run_model(root, role, cache, python):
    from .local_models import LOCK, LocalVLRunner

    manifest = read_json(root / "manifest.json")
    recipe = {"manifest_sha256": file_hash(root / "manifest.json"), "prompt": PROMPT, "schema": SCHEMA,
              "max_tokens": 100, "seed": 17, "checkpoint": read_json(LOCK)["models"][role]}
    out = root / "predictions" / role
    runner = None
    reused = 0
    with stage_lock(out / ".lock"):
        path = out / "run.json"
        record = read_json(path) if path.exists() else {"recipe": recipe, "predictions": {}}
        if record["recipe"] != recipe:
            raise ValueError("Verifier recipe differs from the saved run")
        store = Store(out)
        store.install_signals()
        try:
            for item in manifest["cases"]:
                image = root / item["image"]
                if file_hash(image) != item["masked_sha256"]:
                    raise ValueError(f"Frozen verifier image changed: {item['id']}")
                if item["id"] in record["predictions"]:
                    reused += 1
                    continue
                if runner is None:
                    runner = LocalVLRunner(store, python, cache, role=role, image_limit=1)
                    # GPU UUID can differ between jobs on the same qualified runtime.
                    runtime = runner.identity["runtime"]
                    environment = {key: runtime[key] for key in ("packages", "compute_capability", "gpu_name") if key in runtime}
                    if record.get("environment", environment) != environment:
                        raise ValueError("Verifier runtime changed between attempts")
                    record["environment"] = environment
                message = {"role": "user", "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {
                        "url": "data:image/png;base64," + base64.b64encode(image.read_bytes()).decode("ascii")}},
                ]}
                response = runner.complete([message], SCHEMA, max_tokens=recipe["max_tokens"], seed=recipe["seed"])
                try:
                    value = json.loads(response["text"])
                    if (response["finish_reason"] != "stop" or type(value) is not dict
                            or set(value) != {"verdict", "reason"} or value["verdict"] not in VERDICTS
                            or not isinstance(value["reason"], str) or not value["reason"].strip()):
                        raise ValueError("Invalid verifier response")
                    status = "ok"
                except (ValueError, TypeError):
                    value = {"verdict": "uncertain", "reason": "Invalid model response"}
                    status = "invalid"
                record["predictions"][item["id"]] = {**value, "status": status, "input_sha256": item["masked_sha256"],
                    "wall_seconds": response["wall_seconds"], "peak_memory_bytes": response["peak_memory_bytes"],
                    "usage": response["usage"], "raw": response["text"]}
                record.setdefault("job_ids", [])
                if (job := os.environ.get("SLURM_JOB_ID")) and job not in record["job_ids"]:
                    record["job_ids"].append(job)
                atomic_json(path, record)
                print(f"{role} {len(record['predictions'])}/{len(manifest['cases'])}", flush=True)
        finally:
            if runner is not None:
                runner.close()
    return {"role": role, "predictions": len(record["predictions"]), "reused": reused, "path": str(path)}


def score_labels(cases, labels, predictions):
    table = {label: Counter() for label in VERDICTS}
    for case in cases:
        identifier = case["id"]
        table[labels[identifier]["verdict"]][predictions[identifier]["verdict"]] += 1
    clean = sum(table["clear"].values())
    muddled = sum(table["muddled"].values())
    keep = table["clear"]["clear"]
    falsely_accepted = table["muddled"]["clear"]
    retention = keep / clean if clean else None
    return {"review_labels": {key: sum(value.values()) for key, value in table.items()},
            "prediction_counts": dict(Counter(predictions[row["id"]]["verdict"] for row in cases)),
            "confusion": {key: {v: tally[v] for v in VERDICTS} for key, tally in table.items()},
            "false_acceptances": falsely_accepted, "muddled_crops": muddled,
            "clean_retained": keep, "clean_crops": clean, "clean_retention": retention,
            "uncertain_predictions": sum(tally["uncertain"] for tally in table.values()),
            "invalid_responses": sum(predictions[row["id"]]["status"] == "invalid" for row in cases),
            "passes_threshold": bool(clean and muddled and falsely_accepted <= 1 and retention >= .75)}


def evaluate(root, run, cache, siglip_lock):
    from .body_frame_matching import SkinMatcher, skin_lock_scope
    from .local_models import LOCK
    from .sam_appearances import rank_masks

    manifest_path = root / "manifest.json"
    manifest = read_json(manifest_path)
    manifest_sha = file_hash(manifest_path)
    reviews_path = root / "reviews.json"
    reviews = read_json(reviews_path)
    labels = reviews["labels"]
    cases = manifest["cases"]
    identifiers = {case["id"] for case in cases}
    if (reviews["manifest_sha256"] != manifest_sha or reviews["predictions_seen"] is not False
            or labels.keys() != identifiers or any(row["verdict"] not in VERDICTS for row in labels.values())):
        raise ValueError("Need frozen independent visual reviews for all verifier cases")
    for case in cases:
        for path, pin in ((root / case["image"], case["masked_sha256"]),
                          (run / case["crop"], case["crop_sha256"]),
                          (run / case["mask"], case["mask_sha256"])):
            if file_hash(path) != pin:
                raise ValueError(f"Verifier evidence changed for crop {case['id']}")
    gallery_path = run / "gallery.json"
    scope = manifest["scope"]
    lock_hash = digest(read_json(siglip_lock))
    if lock_hash != scope["encoder_lock_hash"]:
        raise ValueError("Replay requires the original SigLIP checkpoint")
    matcher = SkinMatcher(read_json(gallery_path), cache, siglip_lock, lock_hash,
                          scope["patch"], encoder_name="siglip")
    if skin_lock_scope(matcher, scope["team_pools"], scope["source_sha256"], scope["roster_id"]) != scope:
        raise ValueError("Replay requires the original SigLIP render views")
    pins = read_json(LOCK)["models"]
    results, pending = {}, []
    for role in ROLES:
        path = root / "predictions" / role / "run.json"
        if not path.exists():
            pending.append(role)
            continue
        record = read_json(path)
        preds = record["predictions"]
        if (record["recipe"]["manifest_sha256"] != manifest_sha or record["recipe"]["checkpoint"] != pins[role]
                or record["recipe"]["prompt"] != PROMPT or record["recipe"]["schema"] != SCHEMA
                or set(preds) - identifiers):
            raise ValueError(f"Predictions differ from frozen experiment for {role}")
        if any(preds[row["id"]]["input_sha256"] != row["masked_sha256"] for row in cases if row["id"] in preds):
            raise ValueError(f"Prediction input hash changed for {role}")
        if set(preds) != identifiers:
            pending.append(role)
            continue
        metrics = score_labels(cases, labels, preds)
        comparisons = {}
        for appearance_id, appearance in manifest["appearances"].items():
            subset = [case for case in cases if case["appearance_id"] == appearance_id]
            kept = [case for case in subset if preds[case["id"]]["verdict"] == "clear"]
            queries = [(run / row["crop"], run / row["mask"], row["crop_sha256"], row["mask_sha256"]) for row in kept]
            ranking, _ = rank_masks(matcher, queries, manifest["scope"]["team_pools"][appearance["team"]])
            new_rank = {row["skin_id"]: i for i, row in enumerate(ranking, 1)}
            changes = [{"skin_id": row["skin_id"], "skin_name": row["skin_name"], "baseline_rank": i,
                        "gated_rank": new_rank.get(row["skin_id"])}
                       for i, row in enumerate(appearance["baseline_ranking"], 1)]
            comparisons[appearance_id] = {
                "start": appearance["start"], "end": appearance["end"], "masks_tested": len(subset),
                "retained": len(kept), "retained_ids": [case["id"] for case in kept],
                "baseline_frames": appearance["baseline_frames"],
                "baseline_top_skin": appearance["baseline_ranking"][0]["skin_name"],
                "gated_top_skin": ranking[0]["skin_name"] if ranking else None, "gated_ranking": ranking,
                "rank_changes": changes, "mask_metrics": score_labels(subset, labels, preds),
            }
        results[role] = {"model": pins[role], "job_ids": record.get("job_ids", []), "metrics": metrics,
                         "median_request_seconds": median(row["wall_seconds"] for row in preds.values()),
                         "peak_gpu_bytes": max(row["peak_memory_bytes"] for row in preds.values()),
                         "appearances": comparisons}
    passing = [role for role in results if results[role]["metrics"]["passes_threshold"]]
    winner = min(passing, key=lambda role: results[role]["median_request_seconds"]) if passing and not pending else None
    path = root / "comparison.json"
    comparison = {"experimental": True, "manifest_sha256": manifest_sha, "reviews_sha256": file_hash(reviews_path),
                  "baseline": "Original five masks per appearance", "replay": "All accepted masks from the frozen sample",
                  "threshold": {"max_false_acceptances": 1, "min_clean_retention": .75},
                  "complete": not pending, "pending": pending, "comparison": results, "selected_verifier": winner}
    atomic_json(path, comparison)
    return {"path": str(path), "pending": pending, "selected_verifier": winner,
            "metrics": {role: result["metrics"] for role, result in results.items()}}


def main(argv=None):
    from .body_siglip import DEFAULT_SIGLIP_LOCK

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    draft = commands.add_parser("prepare")
    draft.add_argument("--run", required=True, type=Path)
    draft.add_argument("--root", required=True, type=Path)
    infer = commands.add_parser("infer")
    infer.add_argument("--root", required=True, type=Path)
    infer.add_argument("--role", required=True, choices=ROLES)
    infer.add_argument("--cache", required=True, type=Path)
    infer.add_argument("--vl-python", default=sys.executable)
    score = commands.add_parser("evaluate")
    score.add_argument("--root", required=True, type=Path)
    score.add_argument("--run", required=True, type=Path)
    score.add_argument("--cache", type=Path, default=Path("data/league_assets"))
    score.add_argument("--siglip-lock", type=Path, default=DEFAULT_SIGLIP_LOCK)
    score.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.run.resolve(), args.root.resolve())
    elif args.command == "infer":
        result = run_model(args.root.resolve(), args.role, args.cache.resolve(), args.vl_python)
    else:
        result = evaluate(args.root.resolve(), args.run.resolve(), args.cache.resolve(), args.siglip_lock.resolve())
    print(json.dumps(result, indent=2))
    if args.command == "evaluate" and args.require_complete and result["pending"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
