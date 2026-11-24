"""SAM 3 image fine-tuning using pinned upstream datasets, matcher and losses.

GPU imports are deferred so dataset operations remain usable on a CPU laptop.
The small explicit loop makes the smoke checks and validation selection auditable.
"""
from __future__ import annotations

from functools import partial
import hashlib
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
import time

import numpy as np
from PIL import Image

from .core import atomic_json, digest, read_json
from .media import file_hash
from .sam_finetune import (CODE_REVISION, PROMPT, checked_image, instance_mask, rle_encode,
                           verify_labels)
from .sam_finetune_metrics import aggregate, instance_metrics

CONFIG_PATH = Path(__file__).with_name("sam_finetune_config.json")


def verify_source(path):
    revision = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if revision != CODE_REVISION:
        raise ValueError("SAM source revision differs from experiment pin")
    subprocess.run(["git", "-C", str(path), "diff", "--quiet", "HEAD", "--"], check=True)
    sys.path.insert(0, str(path.resolve()))


def cache_model(args):
    from huggingface_hub import snapshot_download
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        raise ValueError("Use an immutable forty-character Hugging Face revision")
    folder = args.root / "weights" / "sam3"
    snapshot_download("facebook/sam3", revision=args.revision, local_dir=str(folder),
                      allow_patterns=["sam3.pt", "config.json"])
    path = folder / "sam3.pt"
    lock = {"id": "facebook/sam3", "revision": args.revision, "code_revision": CODE_REVISION,
            "path": str(path.relative_to(args.root)), "sha256": file_hash(path)}
    target = args.root / "weights" / "model-lock.json"
    if target.exists() and read_json(target) != lock:
        raise ValueError("Initialization changed; use a new experiment root")
    atomic_json(target, lock)
    return lock


def tensor_digest(named):
    import torch
    hasher = hashlib.sha256()
    for name, value in named:
        hasher.update(name.encode())
        data = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        hasher.update(data.tobytes())
    return hasher.hexdigest()


def freeze_backbones(model):
    for name, value in model.named_parameters():
        value.requires_grad_(not name.startswith("backbone."))
    model.backbone.eval()


def model_lock(root):
    lock = read_json(root / "weights" / "model-lock.json")
    if lock["id"] != "facebook/sam3" or lock["code_revision"] != CODE_REVISION:
        raise ValueError("Unexpected initialization model")
    if file_hash(root / lock["path"]) != lock["sha256"]:
        raise ValueError("Initialization checkpoint changed")
    return lock


def apply_head_checkpoint(model, data, *, base_sha256, labels_sha256):
    import torch
    if data["base_sha256"] != base_sha256 or data["code_revision"] != CODE_REVISION:
        raise ValueError("Fine-tuned checkpoint belongs to another initialization")
    if data["labels_sha256"] != labels_sha256:
        raise ValueError("Fine-tuned checkpoint belongs to another dataset")
    names = {n for n,p in model.named_parameters() if p.requires_grad}
    if set(data["heads"]) != names:
        raise ValueError("Fine-tuned checkpoint does not cover every trainable tensor")
    parameters = dict(model.named_parameters())
    # Validate the complete payload before mutating any parameters.
    if any(parameters[n].shape != data["heads"][n].shape for n in names):
        raise ValueError("Fine-tuned checkpoint tensor shape differs")
    with torch.no_grad():
        for name in names:
            parameters[name].copy_(data["heads"][name])


def load_model(args, checkpoint=None):
    import torch
    verify_source(args.sam_source)
    from sam3.model_builder import build_sam3_image_model
    lock = model_lock(args.root)
    model = build_sam3_image_model(checkpoint_path=str(args.root / lock["path"]),
                                  load_from_HF=False, enable_segmentation=True,
                                  eval_mode=False, device="cuda", compile=False)
    raw = torch.load(args.root / lock["path"], map_location="cpu", weights_only=True, mmap=True)
    raw = raw.get("model", raw)
    state = {key[len("detector."):]: value for key,value in raw.items() if key.startswith("detector.")}
    expected = model.state_dict()
    # Account for historical upstream image-model checkpoint names explicitly.
    if not state:
        state = {key[len("sam3_model."):]: value for key,value in raw.items() if key.startswith("sam3_model.")}
    if not state:
        state = raw
    missing = set(expected) - set(state)
    extra = set(state) - set(expected)
    derived = lambda key: key.endswith((".attn.freqs_cis", ".attn.freqs_cis_real", ".attn.freqs_cis_imag"))
    if any(not derived(k) for k in missing | extra):
        raise ValueError(f"Incomplete SAM image initialization: missing={sorted(missing)[:5]}, extra={sorted(extra)[:5]}")
    for key in set(expected) & set(state):
        if expected[key].shape != state[key].shape:
            raise ValueError(f"Initialization tensor shape differs: {key}")
    model.load_state_dict(state, strict=False)
    del raw, state
    freeze_backbones(model)
    if checkpoint is not None:
        data = torch.load(checkpoint, map_location="cpu", weights_only=True)
        apply_head_checkpoint(model, data, base_sha256=lock["sha256"],
                              labels_sha256=file_hash(args.root / "labels.lock.json"))
    return model


def save_checkpoint(path, model, root, recipe, epoch, validation):
    import torch
    payload = {"heads": {n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad},
               "base_sha256": model_lock(root)["sha256"], "code_revision": CODE_REVISION,
               "labels_sha256": file_hash(root / "labels.lock.json"), "recipe_sha256": digest(recipe),
               "epoch": epoch, "validation": validation}
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        name = Path(tmp.name)
    try:
        torch.save(payload, name)
        os.replace(name, path)
    finally:
        name.unlink(missing_ok=True)


def build_loader(root, split):
    from hydra.utils import instantiate
    from torch.utils.data import DataLoader
    from sam3.train.data.sam3_image_dataset import Sam3ImageDataset
    from sam3.train.data.coco_json_loaders import COCO_FROM_JSON
    from sam3.train.data.collator import collate_fn_api
    transforms = [{"_target_": "sam3.train.transforms.basic_for_api.ComposeAPI", "transforms": [
        {"_target_": "sam3.train.transforms.segmentation.DecodeRle"},
        {"_target_": "sam3.train.transforms.basic_for_api.RandomResizeAPI", "sizes": [1008],
         "max_size": 1008, "square": True, "consistent_transform": False},
        {"_target_": "sam3.train.transforms.basic_for_api.ToTensorAPI"},
        {"_target_": "sam3.train.transforms.basic_for_api.NormalizeAPI", "mean": [.5,.5,.5], "std": [.5,.5,.5]}
    ]}]
    folder = root / "coco" / split
    dataset = Sam3ImageDataset(img_folder=str(folder), ann_file=str(folder / "_annotations.coco.json"),
        transforms=instantiate(transforms), max_ann_per_img=200, multiplier=1, training=split == "train",
        load_segmentation=True, use_caching=False,
        coco_json_loader=partial(COCO_FROM_JSON, include_negatives=True, category_chunk_size=1))
    return DataLoader(dataset, batch_size=1, shuffle=split == "train", num_workers=0,
                      collate_fn=partial(collate_fn_api, dict_key="all", with_seg_masks=True))


def build_loss(model):
    from sam3.train.matcher import BinaryHungarianMatcherV2, BinaryOneToManyMatcher
    from sam3.train.loss.loss_fns import Boxes, IABCEMdetr, Masks
    from sam3.train.loss.sam3_loss import Sam3LossWrapper
    matcher = BinaryHungarianMatcherV2(focal=True, cost_class=2., cost_bbox=5., cost_giou=2.,
                                      alpha=.25, gamma=2., stable=False)
    model.matcher = matcher
    return Sam3LossWrapper(normalization="local", matcher=matcher,
        o2m_matcher=BinaryOneToManyMatcher(alpha=.3, threshold=.4, topk=4), o2m_weight=2.,
        use_o2m_matcher_on_o2m_aux=False, scale_by_find_batch_size=True,
        loss_fns_find=[Boxes(weight_dict={"loss_bbox":5., "loss_giou":2.}),
            IABCEMdetr(weak_loss=False, weight_dict={"loss_ce":20., "presence_loss":20.},
                      pos_weight=10., alpha=.25, gamma=2., use_presence=True, pos_focal=False,
                      pad_n_queries=200, pad_scale_pos=1.),
            Masks(weight_dict={"loss_mask":200., "loss_dice":10.}, focal_alpha=.25,
                  focal_gamma=2., compute_aux=False)])


def predict(model, image):
    import torch
    from sam3.model.sam3_image_processor import Sam3Processor
    model.eval()
    processor = Sam3Processor(model, confidence_threshold=.5)
    torch.cuda.synchronize()
    start = time.monotonic()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = processor.set_image(image.convert("RGB"))
        output = processor.set_text_prompt(state=state, prompt=PROMPT)
    torch.cuda.synchronize()
    masks = output["masks"].detach().cpu().numpy().reshape(-1, image.height, image.width)
    scores = output["scores"].float().detach().cpu().tolist()
    return [{"mask": mask.astype(bool), "score": float(score)} for mask,score in zip(masks,scores)], time.monotonic()-start


def score_split(model, root, split, output=None):
    doc = read_json(root / "coco" / split / "_annotations.coco.json")
    rows = []
    for image in doc["images"]:
        with Image.open(root / "coco" / split / image["file_name"]) as crop:
            predictions, elapsed = predict(model, crop)
        from .sam_finetune import rle_decode
        truth = [rle_decode(a["segmentation"]) for a in doc["annotations"] if a["image_id"] == image["id"]]
        rows.append({"sample_id": image["sample_id"], "seconds": elapsed,
                     **instance_metrics(truth, [p["mask"] for p in predictions])})
        if output is not None:
            atomic_json(output / (image["sample_id"] + ".json"), {"sample_id": image["sample_id"],
                "seconds": elapsed, "predictions": [{"score": p["score"], "mask": rle_encode(p["mask"])} for p in predictions]})
    return {**aggregate(rows), "rows": rows, "inference_seconds": sum(r["seconds"] for r in rows)}


def propose(args):
    """Draft masks only; proposals are never training labels until visual review."""
    model = load_model(args)
    initialization = model_lock(args.root)["sha256"]
    implementation = file_hash(Path(__file__))
    rows = read_json(args.root / "dataset.json")["images"]
    if args.limit:
        rows = rows[:args.limit]
    for row in rows:
        folder = args.root / "proposals" / row["id"]
        receipt = folder / "proposal.json"
        if receipt.exists():
            prior = read_json(receipt)
            if (prior["image_sha256"] != row["path_sha256"] or prior["base_sha256"] != initialization
                    or prior.get("implementation_sha256") != implementation):
                raise ValueError("Proposal recipe changed; choose a new proposal root")
            if all(file_hash(args.root / p["mask_path"]) == p["mask_sha256"] for p in prior["masks"]):
                continue
            raise ValueError("Saved proposal mask changed")
        with checked_image(args.root, row) as image:
            masks, elapsed = predict(model, image)
        folder.mkdir(parents=True, exist_ok=True)
        for index, pred in enumerate(masks):
            Image.fromarray(pred["mask"].astype(np.uint8)*255).save(folder / f"{index}.png")
        atomic_json(folder / "proposal.json", {"image_sha256": row["path_sha256"],
                    "base_sha256": initialization, "implementation_sha256": implementation, "seconds": elapsed,
                    "masks": [{"mask_path": str((folder / f"{i}.png").relative_to(args.root)),
                               "mask_sha256": file_hash(folder / f"{i}.png"), "score": p["score"]}
                              for i,p in enumerate(masks)], "status": "unreviewed"})
    return {"status": "proposals_only", "images": len(rows)}


def train_run(args, *, smoke_run):
    import torch
    verify_labels(args.root)
    config = read_json(CONFIG_PATH)
    if not 1 <= args.epochs <= config["max_epochs"]:
        raise ValueError("Pilot epoch limit is twenty")
    recipe = {"config": config, "initialization": model_lock(args.root),
              "runtime": {"torch":str(torch.__version__), "cuda":torch.version.cuda,
                          "gpu":torch.cuda.get_device_name(), "python":sys.version},
              "labels_sha256": file_hash(args.root / "labels.lock.json"),
              "implementation_sha256": {p.name: file_hash(p) for p in
                  [Path(__file__), Path(__file__).with_name("sam_finetune.py"),
                   Path(__file__).with_name("sam_finetune_metrics.py")]}, "epochs": args.epochs}
    recipe_sha = digest(recipe)
    if not smoke_run:
        receipt = read_json(args.root / "training" / "smoke.json")
        if receipt.get("status") != "passed" or receipt["recipe_sha256"] != recipe_sha:
            raise ValueError("A matching successful smoke is required before training")
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.cuda.reset_peak_memory_stats()
    model = load_model(args)
    frozen_before = tensor_digest((n,p) for n,p in model.named_parameters() if not p.requires_grad)
    trained_before = tensor_digest((n,p) for n,p in model.named_parameters() if p.requires_grad)
    criterion = build_loss(model)
    loader = build_loader(args.root, "train")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=config["learning_rate"], weight_decay=config["weight_decay"])
    from sam3.model.utils.misc import copy_data_to_device
    history, best, start = [], -1., time.monotonic()
    directory = args.root / "training"
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(directory / "recipe.json", recipe)
    for epoch in range(1 if smoke_run else args.epochs):
        model.train()
        model.backbone.eval()
        optimizer.zero_grad(set_to_none=True)
        losses, positives, negatives = [], 0, 0
        # Smoke exercises both positive and negative inputs, with at least one optimizer step.
        batches = None
        if smoke_run:
            coco = read_json(args.root / "coco/train/_annotations.coco.json")
            positive_ids = {a["image_id"] for a in coco["annotations"]}
            ids = [i for i,r in enumerate(coco["images"]) if r["id"] in positive_ids][:3]
            ids += [i for i,r in enumerate(coco["images"]) if r["id"] not in positive_ids][:1]
            from sam3.train.data.collator import collate_fn_api
            batches = [collate_fn_api([loader.dataset[i]], "all", with_seg_masks=True) for i in ids]
        total = len(batches) if smoke_run else len(loader)
        for step, batch in enumerate(batches if smoke_run else loader):
            batch = copy_data_to_device(batch["all"], torch.device("cuda"), non_blocking=True)
            has_target = int(batch.find_targets[0].num_boxes.sum().item()) > 0
            positives += int(has_target)
            negatives += int(not has_target)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                stages = model(batch)
                result = criterion(stages, [model.back_convert(t) for t in batch.find_targets])
                value = result["core_loss"]
            if not torch.isfinite(value) or not all(torch.isfinite(v).all() for v in result.values()):
                raise ValueError("Non-finite training loss")
            if has_target and not all(k in result for k in ("loss_mask", "loss_dice")):
                raise ValueError("Mask supervision is missing")
            losses.append({k:float(v.detach()) for k,v in result.items()})
            # Normalize the final partial accumulation group by its actual size.
            group_start = (step // 4) * 4
            group_size = min(4, total-group_start)
            (value / group_size).backward()
            if (step+1) % 4 == 0 or step+1 == total:
                norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], .1)
                if not torch.isfinite(norm):
                    raise ValueError("Non-finite training gradient")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if smoke_run and step+1 == total:
                break
        if smoke_run:
            if not positives or not negatives:
                raise ValueError("Smoke did not exercise positive and negative samples")
            validation = {"smoke": True}
        else:
            validation = score_split(model, args.root, "validation")
        history.append({"epoch": epoch+1, "mean_loss": float(np.mean([r["core_loss"] for r in losses])),
                        "positives": positives, "negatives": negatives, "validation": validation})
        target = directory / ("smoke.pt" if smoke_run else "last.pt")
        save_checkpoint(target, model, args.root, recipe, epoch+1, validation)
        if not smoke_run and validation["mean_instance_iou"] > best:
            best = validation["mean_instance_iou"]
            save_checkpoint(directory / "best.pt", model, args.root, recipe, epoch+1, validation)
        atomic_json(directory / ("smoke-history.json" if smoke_run else "history.json"), history)
    frozen_after = tensor_digest((n,p) for n,p in model.named_parameters() if not p.requires_grad)
    trained_after = tensor_digest((n,p) for n,p in model.named_parameters() if p.requires_grad)
    if frozen_after != frozen_before or trained_after == trained_before:
        raise ValueError("Frozen/trainable parameter invariant failed")
    expected_heads = trained_after
    del model, optimizer, criterion, loader
    torch.cuda.empty_cache()
    reloaded = load_model(args, target)
    if tensor_digest((n,p) for n,p in reloaded.named_parameters() if p.requires_grad) != expected_heads:
        raise ValueError("Checkpoint reload differs from saved trained weights")
    receipt = {"status": "passed" if smoke_run else "completed", "recipe_sha256": recipe_sha,
               "frozen_unchanged": True, "trainable_changed": True, "reload_verified": True,
               "peak_gpu_bytes": torch.cuda.max_memory_allocated(), "seconds": time.monotonic()-start,
               "checkpoint_sha256": file_hash(target), "epochs": len(history)}
    atomic_json(directory / ("smoke.json" if smoke_run else "completed.json"), receipt)
    return receipt


def smoke(args):
    return train_run(args, smoke_run=True)


def train(args):
    return train_run(args, smoke_run=False)


def evaluate(args):
    from .sam_finetune_evaluate import evaluate as run
    return run(args)
