"""Portable staging and CPU validation of development scene reference drafts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

from .core import atomic_json, read_json
from .scene_reference import (derive_frames, file_hash, materialize, record_exposure,
                              summarize, validate_manifest)


def relocate(value, replacements):
    """Rebase usable artifact paths while preserving the historical originals."""
    if isinstance(value, str):
        for old, new in sorted(replacements.items(), key=lambda item: -len(item[0])):
            if value == old or value.startswith(old+"/"):
                return new+value[len(old):]
        return value
    if isinstance(value, list):
        return [relocate(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: item if key.startswith("original_") else relocate(item, replacements)
                for key, item in value.items()}
    return value


def inside(root, relative):
    target = (root/relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("Reference artifact escapes the run directory")
    return target


def stage(references, run_dir, source):
    references, run_dir, source = Path(references), Path(run_dir).resolve(), Path(source).resolve()
    metadata = read_json(references/"transfer-manifest.json")
    if file_hash(source) != metadata["source_sha256"]:
        raise ValueError("Longleaf source differs from the reviewed recording")
    benchmark = run_dir/"benchmarks/scene-understanding/v1"
    benchmark.mkdir(parents=True, exist_ok=True)
    replacements = {metadata["original_run_root"]: str(run_dir),
                    metadata["original_relative_run_root"]: str(run_dir)}
    for old, relative in metadata.get("extra_asset_relocations", {}).items():
        replacements[old] = str(inside(run_dir, relative))
    for item in metadata["files"]:
        original = references/item["path"]
        if not original.resolve().is_relative_to(references.resolve()) or file_hash(original) != item["sha256"]:
            raise ValueError("Reference transfer snapshot was modified")
        target = inside(benchmark, item["path"])
        if original.suffix == ".json":
            data = relocate(read_json(original), replacements)
            if original.name in {"negative-candidates.json", "sequence-candidates.json", "negative-cases.json"}:
                data["source"]["path"] = str(source)
                validate_manifest(data)
            # Never silently overwrite a newer remote annotation revision.
            if target.exists() and read_json(target) != data:
                raise ValueError(f"Remote reference changed; preserve a new revision: {target}")
            atomic_json(target, data)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            content = original.read_text()
            for old, new in sorted(replacements.items(), key=lambda item: -len(item[0])):
                content = content.replace(old, new)
            if target.exists() and target.read_text() != content:
                raise ValueError("Remote reference prose changed; preserve a new revision")
            target.write_text(content)
    for item in metadata.get("assets", []):
        original = inside(references, item["path"])
        target = inside(run_dir, item["destination"])
        if file_hash(original) != item["sha256"]:
            raise ValueError("Transferred evidence asset changed")
        if target.exists() and file_hash(target) != item["sha256"]:
            raise ValueError("Remote evidence asset changed; preserve a new revision")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(original.read_bytes())
    atomic_json(run_dir/"reference-transfer.json", {
        "source": str(source), "source_sha256": metadata["source_sha256"],
        "transfer_manifest_sha256": file_hash(references/"transfer-manifest.json"),
        "path_relocations": replacements,
        "method": "GitHub reference snapshot rebased onto the identical existing Longleaf source. Original review attestations retained; this is not a new independent review."})
    return benchmark, metadata


def restore_review_inputs(metadata, run_dir, source, benchmark):
    results = []
    for request in metadata["review_input_requests"]:
        directory = inside(run_dir, request["directory"])
        index_path = directory/"source-inputs.json"
        fps = request["sampling_fps"]
        fps = None if fps == "native" else fps
        record_exposure(benchmark/"restored-exposure-ledger.json", metadata["source_sha256"],
                        request["clip_id"], request["start"], request["end"], "regenerated_source_review")
        if index_path.exists():
            data = read_json(index_path)
            if (data["source_sha256"] != metadata["source_sha256"]
                or data["start"] != request["start"] or data["end"] != request["end"]
                or data["sampling_fps"] != (fps or "native")):
                raise ValueError("Changed review input needs a new output directory")
            if not all(Path(frame["source_path"]).is_file() for frame in data["frames"]):
                raise ValueError("Partial review inputs require explicit repair in a new revision")
        else:
            # Base views store their images in inputs/; detail exports may store
            # them either there or alongside source-inputs.json.
            image_dir = inside(run_dir, request["image_directory"])
            data = {"id": request["clip_id"], **materialize(source, metadata["source_sha256"],
                request["start"], request["end"], fps, image_dir)}
            atomic_json(index_path, data)
        results.append({"clip_id": request["clip_id"], "directory": str(directory),
                        "frames": len(data["frames"]), "sampling_fps": data["sampling_fps"]})
        print(request["clip_id"], len(data["frames"]), "restored source frames", flush=True)
    return results


def audit_labels(manifest_path, labels_dir):
    manifest = read_json(manifest_path)
    assets, missing = set(), []
    for clip in manifest["clips"]:
        label = read_json(Path(labels_dir)/(clip["id"]+".json"))
        for field in ("inspected_sheets", "inspected_native_frames"):
            for value in label["review"].get(field, []):
                path = Path(value)
                assets.add(str(path))
                if not path.is_file():
                    missing.append({"clip_id": clip["id"], "path": str(path)})
    return {"referenced_assets": len(assets), "missing_assets": missing,
            "scope": "File availability audit; does not verify the truth of reviewer attestations."}


def run(references, run_dir, source):
    started = time.monotonic()
    run_dir = Path(run_dir).resolve()
    benchmark, metadata = stage(references, run_dir, source)
    restored = restore_review_inputs(metadata, run_dir, source, benchmark)
    stages = []
    for number, stem in ((2, "negative"), (3, "sequence")):
        manifest = benchmark/(stem+"-candidates.json")
        labels = benchmark/(stem+"-labels")
        result = summarize(manifest, labels)
        assets = audit_labels(manifest, labels)
        if result["missing_labels"] or assets["missing_assets"]:
            raise ValueError(f"Stage {number} reference inputs are incomplete: {assets['missing_assets']}")
        frames = derive_frames(manifest, labels, run_dir/f"experiments/scene-understanding/v1/derived-stage-{number}")
        result.update(asset_audit=assets, derived_frames=sum(row["native_frames"] for row in frames))
        atomic_json(benchmark/f"stage-{number}-longleaf-report.json", result)
        stages.append({"stage": number, "reviewed_clips": len(result["clips"]),
                       "source_review_status": "model_draft", "derived_frames": result["derived_frames"],
                       "supported_case_clips": {key: len(value["supported_clips"]) for key,value in result["coverage"].items()}})
    commit = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=True).stdout.strip()
    report = {"status": "completed_development_reference_preparation", "git_commit": commit,
              "source_sha256": metadata["source_sha256"], "restored_inputs": restored, "stages": stages,
              "elapsed_seconds": time.monotonic()-started, "recognition_inference": False,
              "human_verified": False, "held_out_evaluation": False,
              "limitations": "Source model drafts and documented coverage gaps remain; successful preparation is not an accuracy result."}
    atomic_json(run_dir/"campaign-report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.references, args.run_dir, args.source), indent=2))


if __name__ == "__main__":
    main()
