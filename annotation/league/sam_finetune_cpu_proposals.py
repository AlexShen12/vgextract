"""Annotation-only SAM 3 proposals on CPU when the GPU queue is unavailable.

This uses the official Transformers weights at the same model revision. Training
and paired evaluation still use the pinned Meta implementation on the L40S.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import time

import numpy as np
from PIL import Image

from .core import atomic_json, digest, read_json
from .media import file_hash
from .sam_finetune import DEFAULT_ROOT, PROMPT, checked_image


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("cache", "run"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--tag")
    parser.add_argument("--boxes", type=Path, help="Reviewed crop-coordinate body boxes keyed by sample ID")
    parser.add_argument("--sample-prefix")
    parser.add_argument("--threshold", type=float, default=.5)
    args = parser.parse_args(argv)
    if not 0 < args.threshold < 1:
        parser.error("threshold must be between zero and one")
    root = args.root.resolve()
    base = read_json(root / "weights/model-lock.json")
    folder = root / "weights/sam3-transformers"
    lock_path = root / "weights/proposal-model-lock.json"
    if args.stage == "cache":
        from huggingface_hub import snapshot_download
        snapshot_download(base["id"], revision=base["revision"], local_dir=str(folder),
            allow_patterns=["model.safetensors", "*.json", "merges.txt", "vocab.json"])
        lock = {"id":base["id"], "revision":base["revision"],
                "files":{p.name:file_hash(p) for p in folder.iterdir() if p.is_file()}}
        if lock_path.exists() and read_json(lock_path) != lock:
            raise ValueError("Portable proposal initialization changed")
        atomic_json(lock_path, lock)
        print("Pinned annotation-only CPU weights cached",flush=True)
        return
    import torch
    import transformers
    from transformers import Sam3Model, Sam3Processor
    lock = read_json(lock_path)
    if lock["revision"] != base["revision"]:
        raise ValueError("Proposal weights differ from the training model revision")
    for name, sha in lock["files"].items():
        if file_hash(folder/name) != sha:
            raise ValueError("Portable proposal checkpoint changed")
    torch.set_num_threads(int(__import__('os').environ.get("SLURM_CPUS_PER_TASK", "8")))
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("The Mac GPU is unavailable")
    dtype = torch.bfloat16 if args.device == "cpu" else torch.float32
    if args.tag and not re.fullmatch(r"[a-zA-Z0-9_-]+",args.tag):
        raise ValueError("Proposal tag must be a simple directory suffix")
    boxes = read_json(args.boxes) if args.boxes else None
    recipe = {"engine":"transformers-"+args.device, "transformers":transformers.__version__, "torch":str(torch.__version__),
              "model":lock, "implementation_sha256":file_hash(Path(__file__)), "prompt":args.prompt,
              "boxes_sha256":file_hash(args.boxes) if args.boxes else None,
              "dtype":str(dtype), "threshold":args.threshold}
    recipe_sha = digest(recipe)
    model = Sam3Model.from_pretrained(str(folder),local_files_only=True, dtype=dtype,
                                      attn_implementation="sdpa").to(args.device).eval()
    processor = Sam3Processor.from_pretrained(str(folder),local_files_only=True)
    rows = read_json(root/"dataset.json")["images"]
    if boxes is not None:
        rows = [r for r in rows if r["id"] in boxes]
    if args.sample_prefix:
        rows = [r for r in rows if r["id"].startswith(args.sample_prefix)]
    if args.limit:
        rows = rows[:args.limit]
    for number,row in enumerate(rows):
        dest = root/("proposals-"+(args.tag or args.device))/row["id"]
        receipt = dest/"proposal.json"
        if receipt.exists():
            old = read_json(receipt)
            if old["recipe_sha256"] != recipe_sha or old["image_sha256"] != row["path_sha256"]:
                raise ValueError("CPU proposal recipe changed")
            if all(file_hash(root/p["mask_path"]) == p["mask_sha256"] for p in old["masks"]):
                continue
            raise ValueError("CPU proposal mask changed")
        start=time.monotonic()
        with checked_image(root,row) as image:
            kwargs={"text":args.prompt} if boxes is None else {
                "input_boxes":[boxes[row["id"]]], "input_boxes_labels":[[1]*len(boxes[row["id"]])]}
            inputs = processor(images=image.convert("RGB"),return_tensors="pt",**kwargs).to(args.device)
        import contextlib
        autocast = torch.autocast("cpu", dtype=torch.bfloat16) if args.device == "cpu" else contextlib.nullcontext()
        with torch.inference_mode(), autocast:
            outputs=model(**inputs)
        result=processor.post_process_instance_segmentation(outputs,threshold=args.threshold,mask_threshold=.5,
                      target_sizes=inputs["original_sizes"].cpu().tolist())[0]
        dest.mkdir(parents=True,exist_ok=True)
        masks=[]
        for index,(mask,score) in enumerate(zip(result["masks"],result["scores"])):
            path=dest/f"{index}.png"
            Image.fromarray(mask.cpu().numpy().astype(np.uint8)*255).save(path)
            masks.append({"mask_path":str(path.relative_to(root)),"mask_sha256":file_hash(path),"score":float(score)})
        elapsed=time.monotonic()-start
        atomic_json(receipt,{"image_sha256":row["path_sha256"],"recipe_sha256":recipe_sha,
                            "recipe":recipe,"masks":masks,"seconds":elapsed,"status":"unreviewed"})
        print(f"{number+1}/{len(rows)} {row['id']} {len(masks)} proposals {elapsed:.2f}s",flush=True)


if __name__ == "__main__":
    main()
