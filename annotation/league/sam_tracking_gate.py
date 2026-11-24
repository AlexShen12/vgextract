"""CPU qualification of the frozen SAM preflight before comparison allocations."""
from pathlib import Path

from .core import atomic_json, read_json
from .media import file_hash

GPU_LIMIT = 36 * 1024**3
HOST_LIMIT = 80 * 1024**3
MINIMUM_SEED_IOU = .2
COMPARISON_LIMIT_SECONDS = 6300


def qualify(prepared_path):
    import numpy as np
    from PIL import Image

    prepared = read_json(Path(prepared_path))
    root = Path(prepared["root"])
    report = read_json(root / "preflight/report.json")
    errors = []
    if report.get("status") != "completed" or not report.get("seed_check", {}).get("contains_seed"):
        errors.append("preflight_incomplete_or_seed_unverified")
    if report.get("runtime", {}).get("gpu") != "NVIDIA L40S":
        errors.append("unexpected_gpu_class")
    resources = report.get("resources", {})
    gpu = resources.get("torch_peak_memory_bytes", 0)
    host = resources.get("peak_host_memory_bytes", 0)
    if not 0 < gpu < GPU_LIMIT or not 0 < host < HOST_LIMIT:
        errors.append("preflight_memory_headroom_insufficient")
    clip = prepared["clips"][0]
    selected = clip["arms"]["verified"]
    predicted = read_json(root / "preflight" / (clip["id"] + "-complete.json"))
    prompt = selected["prompts"][0]
    frame = next(f for f in predicted["frames"] if f["frame_index"] == prompt["frame_index"])
    mask_row = next(m for m in frame["masks"] if m["object_id"] == 1)
    if file_hash(Path(mask_row["mask_path"])) != mask_row["sha256"]:
        raise ValueError("Preflight mask changed after prediction")
    reference = next((r for r in clip["references"] if r["visibility"] == "visible"
                      and abs(r["time"] - selected["seed"]["time"]) < .04), None)
    if reference is None:
        raise ValueError("Independent visible seed mask is required before comparisons")
    mask = np.asarray(Image.open(mask_row["mask_path"]).convert("L")) > 127
    truth = np.asarray(Image.open(Path(prepared["labels_dir"]) / reference["mask_path"]).convert("L")) > 127
    if mask.shape != truth.shape:
        raise ValueError("Independent mask and SAM geometry differ")
    overlap = float((mask & truth).sum()) / max(1, int((mask | truth).sum()))
    if overlap < MINIMUM_SEED_IOU:
        errors.append("seed_mask_fails_independent_body_overlap")
    projections = {}
    for arm in ("verified", "automatic"):
        count = sum(len(c["arms"][arm]["frames"]) - c["arms"][arm]["prompts"][0]["frame_index"]
                    for c in prepared["clips"] if c["arms"][arm]["status"] == "prepared")
        projections[arm] = resources.get("wall_seconds_this_run", 0) * count / max(1, len(predicted["frames"]))
        if not 0 < projections[arm] <= .8 * COMPARISON_LIMIT_SECONDS:
            errors.append(arm + "_runtime_projection_exceeds_allocation_headroom")
    result = {"status": "passed" if not errors else "failed", "errors": errors,
              "seed_mask_iou": overlap, "minimum_seed_iou": MINIMUM_SEED_IOU,
              "projected_seconds": projections, "preflight_resources": resources,
              "qualification_implementation_sha256": file_hash(Path(__file__)),
              "scope": "Development allocation qualification; recognition acceptance remains untested"}
    atomic_json(root / "gate.json", result)
    if errors:
        raise ValueError("SAM comparison gate failed: " + ", ".join(errors))
    return result


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(qualify(args.prepared), indent=2))
