"""Selected-owner SAM pilot using existing HUD evidence and CPU body binding."""
from __future__ import annotations

import argparse
import copy
import logging
from pathlib import Path
import time

from .action_body import bind_frames
from .action_pipeline import prepare_inputs
from .action_v3 import load_profile
from .caption_segments import POLICY, contexts
from .core import Config, Store, atomic_json, digest, implementation_hash, read_json, uid, write_rows
from .hybrid_intake import _panel_state
from .intake import image_hash, layout_matches
from .intake_profile import load_profile as load_layout
from .inventory_reader import InventoryReader
from .inventory_references import load_inventory_references, reference_root
from .local_models import LOCK, checkpoint
from .media import file_hash
from .normalization import accepted, effective_intake

VERSION = 1
SOURCE_CLIPS = (0, 1, 2, 3, 6)
COMPONENTS = ["sam_tracking.py", "sam_compat.py", "experiment_worker.py", "action_pipeline.py", "action_body.py",
              "inventory_reader.py", "inventory_references.py", "intake_profile.py", "normalization.py",
              "caption_segments.py", "playback.py", "media.py", "intake.py", "hybrid_intake.py"]


def validate_labels(data, source, labels_dir):
    if data.get("source_sha256") != source["sha256"] or data.get("source_only") is not True:
        raise ValueError("SAM labels need independent source-only provenance and matching source")
    clips = data.get("clips", [])
    expected = {f"event-development-{i:02d}" for i in SOURCE_CLIPS}
    if len(clips) != 5 or {c["id"] for c in clips} != expected:
        raise ValueError("SAM pilot requires the five fixed development neighborhoods")
    for clip in clips:
        if clip["split"] != "development" or abs(clip["end"] - clip["start"] - 12) > 1e-6:
            raise ValueError("SAM pilot uses twelve-second development clips")
        seed = clip.get("seed")
        if seed and (not clip["start"] <= seed["time"] < min(clip["end"], clip["start"] + 4)
                     or not seed.get("source_only") or not seed.get("labeler")
                     or len(seed["point"]) != 2 or not 0 <= seed["point"][0] < source["width"]
                     or not 0 <= seed["point"][1] < source["height"]):
            raise ValueError("Invalid independent body seed")
        for ref in clip["references"]:
            if ref["visibility"] not in {"visible", "occluded", "absent", "unknown"} or not ref.get("labeler"):
                raise ValueError("SAM reference visibility and labeler must be explicit")
            if not clip["start"] <= ref["time"] < clip["end"]:
                raise ValueError("SAM reference lies outside source clip")
            if ref["visibility"] == "visible":
                path = (labels_dir / ref["mask_path"]).resolve()
                if not path.is_relative_to(labels_dir.resolve()) or not path.is_file():
                    raise ValueError("Missing source-only native mask")
                from PIL import Image
                with Image.open(path) as image:
                    if image.size != (source["width"], source["height"]):
                        raise ValueError("Reference mask geometry differs from source")
    return clips


def hud_revision(store):
    portraits = [r for r in store.rows("observations") if r.get("field") == "inventory.champion" and r.get("route") == "selected_hud"]
    return digest({"portraits": portraits, "corrections": store.corrections()})


def portrait_rows(store, config, frames, evidence_store=None):
    """Use the default HUD portrait reader and its existing cache without a VL worker."""
    from PIL import Image
    layout = load_layout(config.layout_profile)
    field = next(f for f in layout["fields"] if f["route"] == "selected_hud" and f["field"] == "inventory.champion")
    root = reference_root(config.asset_cache, layout["inventory"]["reference_version"])
    reader = InventoryReader(store, root, load_inventory_references(root, layout["inventory"]["reference_version"]), layout["inventory"])
    evidence_store = evidence_store or store
    corrections = evidence_store.corrections()
    corrected = effective_intake(evidence_store.effective(evidence_store.rows("observations"), "observations"),
                                {c["record_id"] for c in corrections if c["target"] == "observations"})
    rows = []
    for frame in frames:
        store.check_stop()
        with Image.open(frame["native_path"]) as image:
            valid, _ = layout_matches(image, layout)
            state = _panel_state(image, layout, field, valid)
            crop = image.crop(field["box"])
            masked = any(min(field["box"][2], b[2]) > max(field["box"][0], b[0]) and
                         min(field["box"][3], b[3]) > max(field["box"][1], b[1]) for b in config.masks)
            reading = reader.read(crop, field, image_hash(crop))[0] if state == "present" and not masked else {"acceptance": "unknown", "value": None}
        prior = [r for r in corrected if r.get("field") == "inventory.champion" and r.get("route") == "selected_hud"
                 and r.get("box") == field["box"] and abs(r["time"] - frame["time"]) < .001
                 and r.get("inventory_catalog_id") == reader.catalog_id]
        if len(prior) == 1:
            reading = {**reading, "acceptance": "accepted" if accepted(prior[0]) else "unknown",
                       "value": prior[0].get("value") if accepted(prior[0]) else None,
                       "prior_observation_id": prior[0]["id"], "review_status": prior[0].get("review_status", "unreviewed")}
        row = {"id": uid("sam-hud-portrait", frame["id"], reader.signature), "frame_id": frame["id"],
               "time": frame["time"], "route": "selected_hud", "slot_id": "selected", "owner_id": "selected",
               "field": "inventory.champion", "backend": "inventory", "panel_state": state,
               "visibility": "visible" if reading["acceptance"] == "accepted" else "unknown",
               "review_status": "unreviewed", **reading, "inventory_catalog_id": reader.catalog_id,
               "box": field["box"], "native_path": frame["native_path"], "reader_signature": reader.signature}
        row["hud_champion"] = row.get("value") if accepted(row) else None
        rows.append(row)
    return rows


def choose_seed(clip, frames, bodies, arm):
    seed = clip.get("seed")
    if not seed:
        return None, "no_independent_seed"
    frame = min(frames, key=lambda f: abs(f["time"] - seed["time"]))
    if abs(frame["time"] - seed["time"]) > .04:
        return None, "seed_frame_missing"
    if arm == "verified":
        return {"frame_id": frame["id"], "time": frame["time"], "point": seed["point"], "source": "source_only_label",
                "labeler": seed["labeler"]}, None
    body = next((b for b in bodies if b["frame_id"] == frame["id"]), None)
    if not body or not body.get("binding_visible") or body.get("review_status") == "rejected":
        return None, "automatic_body_seed_unavailable"
    x, y, r, b = body["actor_region"]
    return {"frame_id": frame["id"], "time": frame["time"], "point": [(x + r) / 2, (y + b) / 2],
            "source": "existing_body_binder", "body_reading_id": body["id"], "actor_track": body["actor_track"]}, None


def check_seed_mask(result, selected):
    import numpy as np
    from PIL import Image
    prompt = selected["prompts"][0]
    frame = next((f for f in result["frames"] if f["frame_index"] == prompt["frame_index"]), None)
    row = next((m for m in (frame or {}).get("masks", []) if m["object_id"] == 1), None)
    if row is None:
        raise ValueError("SAM preflight produced no seed mask")
    mask = np.asarray(Image.open(row["mask_path"]).convert("L")) > 127
    if mask.shape != (selected["height"], selected["width"]):
        raise ValueError("SAM preflight mask geometry is invalid")
    x, y = (int(v) for v in prompt["point"])
    if not mask[y, x] or not 0 < mask.mean() < .1:
        raise ValueError("SAM preflight mask must contain its seed and remain localized")
    return {"seed_mask_area_fraction": float(mask.mean()), "contains_seed": True}


def prepare(store, config, labels_dir, evidence_store=None):
    from PIL import Image, ImageDraw
    source = read_json(store.path("manifest.json"))["source"]
    evidence_store = evidence_store or store
    data = read_json(labels_dir / "labels.json")
    clips = validate_labels(data, source, labels_dir)
    layout = load_layout(config.layout_profile)
    catalog_root = reference_root(config.asset_cache, layout["inventory"]["reference_version"])
    catalog = load_inventory_references(catalog_root, layout["inventory"]["reference_version"])
    labels_hash = digest({"labels": data, "masks": {r["mask_path"]: file_hash(labels_dir / r["mask_path"])
                         for c in clips for r in c["references"] if r["visibility"] == "visible"}})
    recipe = {"version": VERSION, "source": source["sha256"], "labels": labels_hash,
              "profile": load_profile(), "layout": layout, "catalog": digest(catalog), "policy": POLICY,
              "model": read_json(LOCK)["models"]["tracking"], "implementation": implementation_hash(COMPONENTS),
              "hud_revision": hud_revision(evidence_store)}
    root = store.path("experiments/sam-tracking/" + digest(recipe)[:20])
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "recipe.json", recipe)
    atomic_json(root / "labels.json", data)
    from shutil import copy2
    snapshot = root / "source-labels"
    snapshot.mkdir(exist_ok=True)
    atomic_json(snapshot / "labels.json", data)
    for clip in clips:
        for ref in clip["references"]:
            if ref["visibility"] == "visible":
                target = snapshot / ref["mask_path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                copy2(labels_dir / ref["mask_path"], target)
    prepared = []
    for clip in clips:
        store.check_stop()
        logging.info("SAM prepare %s: decoding source %.3f–%.3f", clip["id"], clip["start"], clip["end"])
        audit = prepare_inputs(store, config, clip["start"], clip["end"], recipe["profile"], POLICY,
                               namespace="sam-tracking", cache_identity={"sam_frame_protocol": VERSION})
        frames = audit["frames"]
        logging.info("SAM prepare %s: %d frames, reading existing HUD portrait adapter", clip["id"], len(frames))
        portraits = portrait_rows(store, config, frames, evidence_store)
        logging.info("SAM prepare %s: binding selected body", clip["id"])
        body, bindings = bind_frames(frames, portraits, recipe["profile"], store.check_stop)
        croot = root / clip["id"]
        croot.mkdir(exist_ok=True)
        write_rows(croot / "portraits.jsonl", portraits)
        write_rows(croot / "body_readings.jsonl", body)
        write_rows(croot / "bindings.jsonl", bindings)
        row = {**clip, "frames": frames, "arms": {}, "portrait_path": str(croot / "portraits.jsonl")}
        for arm in ("verified", "automatic"):
            seed, reason = choose_seed(clip, frames, body, arm)
            selected = next((g for g in contexts(frames, [clip["start"], clip["end"]]) if seed and
                             any(f["id"] == seed["frame_id"] for f in g["frames"])), None)
            if not selected:
                row["arms"][arm] = {"seed": seed, "status": reason or "seed_continuity_unsupported"}
                continue
            named = next((p["hud_champion"] for p in portraits if p["frame_id"] == seed["frame_id"]), None)
            names = {p["hud_champion"] for p in portraits if p["hud_champion"] and any(f["id"] == p["frame_id"] for f in selected["frames"])}
            seed["champion"] = named if names == {named} else None
            seed["portrait_evidence_ids"] = [p["id"] for p in portraits if p["hud_champion"] == seed["champion"] and p["frame_id"] == seed["frame_id"]]
            directory = croot / "input-frames" / digest(selected["context"])[:12]
            directory.mkdir(parents=True, exist_ok=True)
            native = []
            for i, frame in enumerate(selected["frames"]):
                path = directory / f"{i:06d}.jpg"
                if not path.exists():
                    with Image.open(frame["native_path"]) as image:
                        screen = Image.new("RGB", image.size)
                        sx, sy, ex, ey = recipe["profile"]["scene_box"]
                        screen.paste(image.crop((sx, sy, ex, ey)), (sx, sy))
                        draw = ImageDraw.Draw(screen)
                        for box in config.masks:
                            draw.rectangle(box, fill="black")
                        screen.save(path, quality=95)
                native.append({"id": frame["id"], "time": frame["time"], "path": str(path), "sha256": file_hash(path)})
            row["arms"][arm] = {"status": "prepared", "seed": seed, "frames": native,
                "prompts": [{"object_id": 1, "point": seed["point"], "frame_index": next(i for i, f in enumerate(native) if f["id"] == seed["frame_id"])}],
                "frames_dir": str(directory), "width": source["width"], "height": source["height"],
                "context": selected["context"], "transform": "identity_source_native_canvas"}
        prepared.append(row)
        logging.info("SAM prepared %s: %s", clip["id"], {a: r["status"] for a, r in row["arms"].items()})
    report = {"schema_version": VERSION, "source": source, "root": str(root), "recipe": recipe,
              "labels_dir": str(snapshot.resolve()), "asset_cache": str(config.asset_cache),
              "evidence_run": str(evidence_store.root.resolve()),
              "clips": prepared, "experimental": True,
              "status": "prepared", "quality_metrics": None}
    atomic_json(root / "prepared.json", report)
    return report


def run(store, prepared_path, arm, preflight=False):
    import importlib.metadata
    import resource
    import torch
    from .experiment_worker import tracking_job
    prepared = read_json(prepared_path)
    if prepared["recipe"]["implementation"] != implementation_hash(COMPONENTS):
        raise ValueError("SAM implementation changed since source preparation")
    if hud_revision(Store(Path(prepared["evidence_run"]))) != prepared["recipe"]["hud_revision"]:
        raise ValueError("HUD evidence or corrections changed since source preparation")
    root = Path(prepared["root"]) / ("preflight" if preflight else arm)
    root.mkdir(parents=True, exist_ok=True)
    source = prepared["source"]
    if file_hash(Path(source["path"])) != source["sha256"]:
        raise ValueError("Source video changed")
    runtime = {"packages": {n: importlib.metadata.version(n) for n in ("torch", "torchvision", "sam3", "numpy")},
               "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)}
    import sys
    runtime["python"] = sys.version
    environment = Path(sys.prefix) / "installed.txt"
    runtime["environment_manifest_sha256"] = file_hash(environment) if environment.exists() else None
    import sam3.model_builder
    runtime["sam_builder_sha256"] = file_hash(Path(sam3.model_builder.__file__))
    _, pin = checkpoint(prepared["asset_cache"], "tracking")
    signature = digest({"recipe": prepared["recipe"], "arm": arm, "preflight": preflight, "runtime": runtime, "pin": pin})
    path = root / "report.json"
    report = read_json(path) if path.exists() else {"key": signature, "arm": arm, "preflight": preflight,
              "root": str(root), "prepared": str(prepared_path), "clips": {}, "runtime": runtime, "checkpoint": pin}
    if report["key"] != signature:
        raise ValueError("SAM resume runtime/configuration mismatch")
    if not preflight:
        qualification = read_json(Path(prepared["root"]) / "preflight/report.json")
        if qualification["status"] != "completed" or qualification["runtime"] != runtime:
            raise ValueError("Successful matching SAM preflight is required")
    started = time.monotonic()
    todo = []
    clips = prepared["clips"][:1] if preflight else prepared["clips"]
    for clip in clips:
        selected = copy.deepcopy(clip["arms"][arm])
        if selected["status"] != "prepared":
            report["clips"][clip["id"]] = {"status": "unseeded", "reason": selected["status"]}
            continue
        if preflight:
            count = min(len(selected["frames"]), selected["prompts"][0]["frame_index"] + 32)
            selected["frames"] = selected["frames"][:count]
            from shutil import copy2
            short = root / "preflight-inputs"
            short.mkdir(exist_ok=True)
            for i, f in enumerate(selected["frames"]):
                copy2(f["path"], short / f"{i:06d}.jpg")
            selected["frames_dir"] = str(short)
        if any(file_hash(Path(f["path"])) != f["sha256"] for f in selected["frames"]):
            raise ValueError("SAM source frame changed since preparation")
        done = root / (clip["id"] + "-complete.json")
        if done.exists():
            previous = read_json(done)
            valid = all(Path(m["mask_path"]).is_file() and file_hash(Path(m["mask_path"])) == m.get("sha256")
                        for f in previous["frames"] for m in f["masks"])
            if valid:
                report["clips"][clip["id"]] = {"status": "completed", "result": str(done)}
                continue
        todo.append({**selected, "id": clip["id"]})
    report.update(status="running", runtime=runtime)
    import os
    report["slurm_job_id"] = os.environ.get("SLURM_JOB_ID")
    atomic_json(path, report)
    try:
        if todo:
            torch.manual_seed(17)
            worker = tracking_job({"protocol": "selected-sam-v1", "clips": todo, "cache": prepared["asset_cache"],
                          "output": str(root / "worker.json")}, stop=store.check_stop)
            report["model_load"] = worker["model_load"]
        for clip in todo:
            report["clips"][clip["id"]] = {"status": "completed", "result": str(root / (clip["id"] + "-complete.json"))}
        if preflight and not any(c["status"] == "completed" for c in report["clips"].values()):
            raise ValueError("Preflight supplied no supported seed")
        if preflight:
            selected = todo[0] if todo else {**prepared["clips"][0]["arms"][arm], "id": prepared["clips"][0]["id"]}
            report["seed_check"] = check_seed_mask(read_json(root / (selected["id"] + "-complete.json")), selected)
        report.update(status="completed")
    except BaseException as exc:
        report.update(status="interrupted" if store.stopping else "failed", error=str(exc))
        raise
    finally:
        report["resources"] = {"wall_seconds_this_run": time.monotonic() - started,
            "torch_peak_memory_bytes": torch.cuda.max_memory_allocated(),
            "peak_host_memory_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
        atomic_json(path, report)
    return report


def collect(prepared_path):
    import numpy as np
    from PIL import Image
    prepared = read_json(prepared_path)
    root = Path(prepared["root"])
    summary = {"experimental": True, "scope": "selected-player development segmentation only", "arms": {}}
    for arm in ("verified", "automatic"):
        report = root / arm / "report.json"
        if not report.exists() or read_json(report)["status"] != "completed":
            summary["arms"][arm] = {"status": "pending", "metrics": None}
            continue
        output = {"clips": [], "seeded": 0, "clips_total": len(prepared["clips"]), "visible_references": 0,
                  "predicted_visible": 0, "negative_references": 0, "false_visible": 0, "unknown_references": 0, "ious": []}
        for clip in prepared["clips"]:
            info = clip["arms"][arm]
            done = root / arm / (clip["id"] + "-complete.json")
            output["seeded"] += info["status"] == "prepared"
            predictions = {f["frame_index"]: f["masks"] for f in read_json(done)["frames"]} if done.exists() else {}
            for ref in clip["references"]:
                if ref["visibility"] == "unknown":
                    output["unknown_references"] += 1
                    continue
                frames = info.get("frames", [])
                index = min(range(len(frames)), key=lambda i: abs(frames[i]["time"] - ref["time"])) if frames else None
                masks = predictions.get(index, []) if index is not None and abs(frames[index]["time"] - ref["time"]) <= .04 else []
                mask = np.zeros((prepared["source"]["height"], prepared["source"]["width"]), dtype=bool)
                for row in masks:
                    if row["object_id"] == 1:
                        if file_hash(Path(row["mask_path"])) != row.get("sha256"):
                            raise ValueError("SAM mask changed after prediction")
                        mask = np.asarray(Image.open(row["mask_path"]).convert("L")) > 127
                if ref["visibility"] == "visible":
                    truth = np.asarray(Image.open(Path(prepared["labels_dir"]) / ref["mask_path"]).convert("L")) > 127
                    output["visible_references"] += 1
                    output["predicted_visible"] += bool(mask.any())
                    output["ious"].append(float((mask & truth).sum()) / max(1, int((mask | truth).sum())))
                else:
                    output["negative_references"] += 1
                    output["false_visible"] += bool(mask.any())
            output["clips"].append({"id": clip["id"], "seed": info.get("seed"), "complete": done.exists()})
        output["mean_mask_iou"] = sum(output["ious"]) / len(output["ious"]) if output["ious"] else None
        output["identity_switches"] = None
        output["identity_switch_status"] = "untested_with_sparse_single_actor_masks"
        output["status"] = "diagnostic_only" if output["visible_references"] else "untested"
        summary["arms"][arm] = output
    atomic_json(root / "summary.json", summary)
    return summary


def bundle(prepared_path, output):
    """Use the existing caption bundle importer for metadata and native masks."""
    import io
    import json
    import tarfile
    prepared_path = Path(prepared_path)
    prepared = read_json(prepared_path)
    root = Path(prepared["root"])
    run_root = root.parents[2]
    files = {"run/manifest.json": run_root / "manifest.json"}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix in {".json", ".jsonl", ".png", ".log", ".out", ".err"}:
            files["run/" + p.relative_to(run_root).as_posix()] = p
    index = {"schema_version": 1, "source_run": str(run_root), "source": prepared["source"],
             "execution_report": str(root.relative_to(run_root) / prepared_path.name),
             "files": {name: file_hash(path) for name, path in files.items()}, "images": "native_masks_only"}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        payload = json.dumps(index).encode()
        info = tarfile.TarInfo("bundle-index.json")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
        for name, path in sorted(files.items()):
            archive.add(path, arcname=name, recursive=False)
    return {"bundle": str(output), "sha256": file_hash(output), "files": len(files)}


def main(argv=None):
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "preflight", "run", "collect", "review", "bundle"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--arm", choices=("verified", "automatic"), default="verified")
    parser.add_argument("--port", type=int, default=8768)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    store = Store(args.run_dir)
    store.install_signals()
    if args.mode == "prepare":
        import os
        config = Config.load(Path(__file__).with_name("midbeast.json"))
        config.asset_cache = os.environ.get("LEAGUE_ASSET_CACHE", config.asset_cache)
        evidence_store = Store(args.source_run) if args.source_run else store
        if args.source_run:
            original = read_json(evidence_store.path("manifest.json"))
            if store.path("manifest.json").exists() and read_json(store.path("manifest.json"))["source"] != original["source"]:
                raise ValueError("Isolated SAM run belongs to a different source")
            if not store.path("manifest.json").exists():
                atomic_json(store.path("manifest.json"), original)
        result = prepare(store, config, args.labels, evidence_store)
    elif args.mode == "collect":
        result = collect(args.prepared)
    elif args.mode == "review":
        from .sam_tracking_review import serve
        return serve(args.prepared, args.port)
    elif args.mode == "bundle":
        result = bundle(args.prepared, args.output)
    else:
        result = run(store, args.prepared, args.arm, args.mode == "preflight")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
