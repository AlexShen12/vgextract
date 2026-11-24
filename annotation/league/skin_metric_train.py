"""Train MobileNetV3 Small for synthetic skin and chroma retrieval."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import io
import json
import math
import os
from pathlib import Path
import random
import signal
import tempfile

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image, ImageEnhance

from .core import atomic_json, implementation_hash, read_json
from .media import file_hash

SIZE = 224
DIMENSIONS = 128
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
PREPROCESS = {"size": SIZE, "foreground_extent": 0.82, "background": [70, 72, 70],
              "mean": list(MEAN), "std": list(STD)}
SEED = 1622
WEIGHTS = MobileNet_V3_Small_Weights.IMAGENET1K_V1


def save_torch(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as out:
        temporary = Path(out.name)
        try:
            torch.save(data, out)
            out.flush()
            os.fsync(out.fileno())
            out.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def load_torch(path, device="cpu"):
    return torch.load(path, map_location=device, weights_only=True)


def initialize(root):
    directory = root / "weights"
    path = directory / "mobilenet_v3_small_imagenet1k_v1.pt"
    lock = directory / "initialization.json"
    if path.exists() and lock.exists():
        if file_hash(path) != read_json(lock)["sha256"]:
            raise ValueError("Cached ImageNet initialization changed")
        return path
    model = mobilenet_v3_small(weights=WEIGHTS, progress=False)
    save_torch(path, model.state_dict())
    atomic_json(lock, {"url": WEIGHTS.url, "sha256": file_hash(path), "torch": str(torch.__version__)})
    return path


def initial_state(root):
    folder = root / "weights"
    path = folder / "mobilenet_v3_small_imagenet1k_v1.pt"
    sha = file_hash(path)
    if read_json(folder / "initialization.json")["sha256"] != sha:
        raise ValueError("ImageNet initialization checksum mismatch")
    return path, sha


def encoder(initial=None):
    net = mobilenet_v3_small(weights=None)
    if initial:
        net.load_state_dict(load_torch(initial))
    features = net.classifier[-1].in_features
    net.classifier[-1] = nn.Linear(features, DIMENSIONS, bias=False)
    return net


def preprocess(path, rng=None):
    with Image.open(path) as source:
        image = source.convert("RGBA")
    box = image.getchannel("A").getbbox()
    if box is None:
        raise ValueError(f"Empty crop: {path}")
    image = image.crop(box)
    extent = rng.uniform(.48, .92) if rng else .82
    factor = SIZE * extent / max(image.size)
    image = image.resize(tuple(max(1, round(axis * factor)) for axis in image.size), Image.Resampling.LANCZOS)
    if rng and rng.random() < .35:
        down = rng.uniform(.28, .62)
        small = tuple(max(1, round(axis * down)) for axis in image.size)
        image = image.resize(small, Image.Resampling.BILINEAR).resize(image.size, Image.Resampling.BILINEAR)
    color = tuple(rng.randint(36, 130) for _ in range(3)) if rng else tuple(PREPROCESS["background"])
    canvas = Image.new("RGB", (SIZE, SIZE), color)
    offset = tuple((SIZE - axis) // 2 for axis in image.size)
    if rng:
        offset = tuple(max(0, min(SIZE - axis, pos + rng.randint(-10, 10)))
                       for axis, pos in zip(image.size, offset, strict=True))
    canvas.paste(image, offset, image)
    if rng:
        canvas = ImageEnhance.Brightness(canvas).enhance(rng.uniform(.88, 1.12))
        if rng.random() < .5:
            buffer = io.BytesIO()
            canvas.save(buffer, "JPEG", quality=rng.randint(58, 95))
            with Image.open(buffer) as compressed:
                canvas = compressed.convert("RGB")
    tensor = pil_to_tensor(canvas).float().div_(255)
    return (tensor - torch.tensor(MEAN)[:, None, None]) / torch.tensor(STD)[:, None, None]


class RenderViews(Dataset):
    def __init__(self, root, identities, split, augmentation_epoch=None):
        self.root = root
        self.augmentation_epoch = augmentation_epoch
        self.counts = {"train": 16, "validation": 4, "test": 4}
        self.rows = []
        for index, identity in enumerate(identities):
            views = [view["path"] for view in identity["views"] if view["split"] == split]
            if len(views) != self.counts[split]:
                raise ValueError(f"Incomplete {split} views for {identity['id']}")
            self.rows.extend((index, view) for view in views)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        label, path = self.rows[index]
        rng = random.Random(f"{SEED}:{self.augmentation_epoch}:{index}") if self.augmentation_epoch else None
        return preprocess(self.root / path, rng), label


class ArcFace(nn.Module):
    def __init__(self, count):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(count, DIMENSIONS))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        margin = .4
        cosine = F.linear(embeddings.float(), F.normalize(self.weight.float())).clamp(-1 + 1e-6, 1 - 1e-6)
        target = cosine.gather(1, labels[:, None])
        adjusted = target * math.cos(margin) - (1 - target.square()).sqrt() * math.sin(margin)
        adjusted = torch.where(target > math.cos(math.pi - margin),
                               adjusted, target - math.sin(math.pi - margin) * margin)
        return F.cross_entropy(cosine.scatter(1, labels[:, None], adjusted) * 32, labels)


def triplet_loss(embeddings, labels):
    """Use the same batch-hard objective in both arms, varying only batch composition."""
    similarities = embeddings @ embeddings.T
    positive = labels[:, None] == labels[None, :]
    hardest_positive = similarities.masked_fill(~positive | torch.eye(
        len(labels), device=labels.device, dtype=torch.bool), 1).amin(dim=1)
    hardest_negative = similarities.masked_fill(positive, -1).amax(dim=1)
    return (hardest_negative - hardest_positive + .15).clamp_min(0).mean()


def class_groups(identities, epoch, width, neighbors=None):
    """Group each class exactly once per epoch in both arms."""
    rng = random.Random(SEED + epoch)
    order = list(range(len(identities)))
    rng.shuffle(order)
    families = defaultdict(list)
    champions = defaultdict(list)
    for index in order:
        identity = identities[index]
        families[identity["parent_skin_id"]].append(index)
        champions[identity["champion"]].append(index)
    unused = set(order)
    anchors = iter(order)
    while unused:
        group = []
        while len(group) < width and unused:
            anchor = next(index for index in anchors if index in unused)
            group.append(anchor)
            unused.remove(anchor)
            if neighbors is None:
                continue
            identity = identities[anchor]
            candidates = (
                families[identity["parent_skin_id"]],
                (n for n in champions[identity["champion"]]
                 if identities[n]["parent_skin_id"] != identity["parent_skin_id"]),
                (n for n in neighbors[anchor] if identities[n]["champion"] != identity["champion"]),
            )
            for category in candidates:
                if len(group) == width:
                    break
                match = next((n for n in category if n in unused), None)
                if match is not None:
                    group.append(match)
                    unused.remove(match)
        yield group


def training_batches(identities, epoch, width, neighbors):
    for group in class_groups(identities, epoch, width, neighbors):
        views = [random.Random(f"{SEED}:{epoch}:{index}").sample(range(16), 16) for index in group]
        for start in range(0, 16, 4):
            yield [16 * index + view for index, order in zip(group, views, strict=True)
                   for view in order[start:start + 4]]


@torch.inference_mode()
def encode(model, dataset, device, workers):
    model.eval()
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=workers,
                        pin_memory=device.type == "cuda")
    return torch.cat([F.normalize(model(images.to(device, non_blocking=True)).float())
                      for images, _ in loader])


def mine_neighbors(gallery):
    centers = F.normalize(gallery.mean(dim=1))
    neighbors = []
    for start in range(0, len(centers), 256):
        scores = centers[start:start + 256] @ centers.T
        index = torch.arange(len(scores), device=scores.device)
        scores[index, index + start] = -torch.inf
        neighbors.extend(scores.argsort(dim=1, descending=True, stable=True)[:, :min(64, len(centers) - 1)].tolist())
    return neighbors


def top_classes(queries, gallery):
    classes, views, features = gallery.shape
    matrix = gallery.reshape(classes * views, features).T
    for block in queries.split(128):
        similarity = (block @ matrix).reshape(-1, classes, views).amax(dim=2)
        yield similarity.argsort(dim=1, descending=True, stable=True)[:, :min(classes, 5)].tolist()


def score_rankings(ranks, labels, identities):
    results = defaultdict(list)
    confusions = Counter()
    for rank, label in zip(ranks, labels, strict=True):
        prediction = rank[0]
        outcome = (prediction == label, label in rank)
        results["all"].append((label, outcome))
        results[identities[label]["kind"]].append((label, outcome))
        source, target = identities[label], identities[prediction]
        if prediction != label and source["parent_skin_id"] == target["parent_skin_id"]:
            confusions[source["id"], target["id"], source["parent_skin_id"]] += 1
    metrics = {}
    for group, rows in sorted(results.items()):
        by_class = defaultdict(list)
        for label, outcome in rows:
            by_class[label].append(outcome)
        report = {"queries": len(rows), "identities": len(by_class)}
        for index, key in enumerate(("top1", "top5")):
            report[key] = sum(outcome[index] for _, outcome in rows) / len(rows)
            report[f"macro_{key}"] = sum(sum(outcome[index] for outcome in values) / len(values)
                                         for values in by_class.values()) / len(by_class)
        metrics[group] = report
    return {"metrics": metrics,
            "within_parent_confusions": [{"expected": source, "predicted": target,
                                          "parent_skin_id": parent, "count": count}
                                         for (source, target, parent), count in sorted(confusions.items())]}


def scoring(model, gallery, dataset, identities, device, workers):
    ranks = [row for group in top_classes(encode(model, dataset, device, workers), gallery) for row in group]
    labels = [label for label, _ in dataset.rows]
    return score_rankings(ranks, labels, identities), ranks


def save_model(path, model, lock, **data):
    save_torch(path, {"config": lock, "encoder": model.state_dict(), **data})


def continuation_config(source, config, output):
    if source.resolve() == output.resolve():
        raise ValueError("Continuation must write to a separate run directory")
    original = read_json(source / "config.json")
    ignored = {"epochs", "implementation_sha256"}
    if any(original.get(key) != config.get(key) for key in (original.keys() | config.keys()) - ignored):
        raise ValueError("Continuation must keep the original training configuration")
    if original["epochs"] >= config["epochs"]:
        raise ValueError("Continuation needs a later final epoch")
    return {
        "source_run": str(source.resolve()),
        "epoch": original["epochs"],
        "last_sha256": file_hash(source / "last.pt"),
        "best_sha256": file_hash(source / "best.pt"),
    }


def train(root, dataset_path, strategy, device, epochs, workers, batch_classes, scratch=False,
          *, output=None, continue_from=None):
    dataset = read_json(dataset_path)
    pilot = dataset_path.parent == root / "pilot"
    review, review_sha = {}, None
    if pilot and not scratch:
        from .skin_metric_dataset import pilot_review

        review = pilot_review(root)
        review_sha = file_hash(root / "pilot" / "review.json")
    rejected = set(review.get("rejected", []))
    active = sorted((row for row in dataset["identities"]
                     if row["status"] == "renderable" and row["id"] not in rejected),
                    key=lambda row: row["id"])
    if len(active) < 2:
        raise ValueError("Training needs at least two distinct identities")
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    device = torch.device(device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    path, sha = (None, None) if scratch else initial_state(root)
    config = {"dataset_sha256": file_hash(dataset_path), "initial_sha256": sha, "strategy": strategy,
              "pilot_review_sha256": review_sha,
              "epochs": epochs, "batch_classes": batch_classes, "seed": SEED,
              "arcface": {"scale": 32, "margin": .4}, "triplet": {"weight": .2, "margin": .15},
              "embedding_dimensions": DIMENSIONS, "preprocess": PREPROCESS,
              "implementation_sha256": implementation_hash(["skin_metric_train.py"]),
              "torch": str(torch.__version__)}
    output = output or root / ("pilot/runs" if scratch or pilot else "runs") / strategy
    if continue_from:
        config["continuation"] = continuation_config(continue_from, config, output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "last.pt"
    config_path = output / "config.json"
    if config_path.exists() and read_json(config_path) != config:
        raise ValueError(f"Training configuration changed in {output}")
    atomic_json(config_path, config)
    model = encoder(path).to(device)
    classifier = ArcFace(len(active)).to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.parameters(), "lr": .0003},
        {"params": classifier.parameters(), "lr": .001},
    ], weight_decay=.0001)
    eval_sets = {split: RenderViews(root, active, split) for split in ("train", "validation", "test")}
    best, start, history = 0, 1, []
    if checkpoint.exists() or continue_from:
        resuming_new_run = not checkpoint.exists()
        if resuming_new_run:
            source_config = read_json(continue_from / "config.json")
            state = load_torch(continue_from / "last.pt", device)
            selected = load_torch(continue_from / "best.pt", "cpu")
            if (state["config"] != source_config or selected["config"] != source_config
                    or state["epoch"] != config["continuation"]["epoch"]
                    or len(state["history"]) != state["epoch"]
                    or selected["validation"]["metrics"]["all"]["macro_top1"] != state["best"]):
                raise ValueError("Source run does not have a complete checkpoint and validation selection")
            # Keep the original best eligible in case further training lowers validation accuracy.
            save_torch(output / "best.pt", {**selected, "config": config})
            save_torch(checkpoint, {**state, "config": config})
        else:
            state = load_torch(checkpoint, device)
            if state["config"] != config:
                raise ValueError(f"Training configuration changed in {output}")
        model.load_state_dict(state["encoder"])
        classifier.load_state_dict(state["arcface"])
        optimizer.load_state_dict(state["optimizer"])
        best, start, history = state["best"], state["epoch"] + 1, state["history"]
    elif (output / "best.pt").exists():
        # An interruption can land between best.pt and the first last.pt.
        print("Replaying the initial epoch", flush=True)

    stopping = False

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    gallery = None
    for epoch in range(start, epochs + 1):
        if stopping:
            raise SystemExit("Stopped before the next epoch. Resume from last.pt.")
        torch.manual_seed(SEED + epoch)
        neighbors = None
        if strategy == "hard" and epoch > 1:
            if gallery is None:
                gallery = encode(model, eval_sets["train"], device, workers).reshape(-1, 16, DIMENSIONS)
            neighbors = mine_neighbors(gallery)
        gallery = None
        dataset_train = RenderViews(root, active, "train", epoch)
        loader = DataLoader(dataset_train, batch_sampler=list(training_batches(active, epoch, batch_classes, neighbors)),
                            num_workers=workers, pin_memory=device.type == "cuda")
        model.train()
        total, count = torch.zeros((), device=device), 0
        for images, labels in loader:
            if stopping:
                raise SystemExit("Stopped during an epoch. Resume from the prior completed epoch.")
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
                features = model(images)
            vectors = F.normalize(features.float())
            loss = classifier(vectors, labels) + .2 * triplet_loss(vectors, labels)
            loss.backward()
            optimizer.step()
            total += loss.detach() * len(labels)
            count += len(labels)
        gallery = encode(model, eval_sets["train"], device, workers).reshape(-1, 16, DIMENSIONS)
        validation, _ = scoring(model, gallery, eval_sets["validation"], active, device, workers)
        score = validation["metrics"]["all"]["macro_top1"]
        result = {"epoch": epoch, "loss": float(total / count), "views": count, "validation": validation}
        print(f"epoch={epoch} loss={result['loss']:.4f} validation_macro_top1={score:.4f}", flush=True)
        if not history or score > best:
            best = score
            save_model(output / "best.pt", model, config, epoch=epoch, validation=validation)
        history.append(result)
        save_model(checkpoint, model, config, epoch=epoch, best=best, history=history,
                   arcface=classifier.state_dict(), optimizer=optimizer.state_dict())
        atomic_json(output / "history.json", history)

    if stopping:
        raise SystemExit("Checkpoint saved. Resume to finish scoring.")
    export_run(output, config, eval_sets, active, device, workers)


def export_run(output, config, eval_sets, active, device, workers):
    state = load_torch(output / "best.pt", device)
    model = encoder().to(device)
    model.load_state_dict(state["encoder"])
    gallery = encode(model, eval_sets["train"], device, workers).reshape(-1, 16, DIMENSIONS)
    validation, _ = scoring(model, gallery, eval_sets["validation"], active, device, workers)
    if validation != state["validation"]:
        raise ValueError("Reloaded best checkpoint did not reproduce validation metrics")
    test, ranks = scoring(model, gallery, eval_sets["test"], active, device, workers)
    labels = [{key: row[key] for key in ("id", "kind", "name", "parent_skin_id", "champion")} for row in active]
    save_model(output / "inference.pt", model, config, labels=labels)
    save_torch(output / "gallery.pt", {"vectors": gallery.cpu(), "ids": [row["id"] for row in labels],
                                      "dataset_sha256": config["dataset_sha256"],
                                      "encoder_sha256": file_hash(output / "inference.pt")})
    _, loaded_gallery, exported = load_export(output, device)
    reloaded_test, reloaded_ranks = scoring(exported, loaded_gallery, eval_sets["test"], active, device, workers)
    if reloaded_test != test or reloaded_ranks != ranks:
        raise ValueError("Reloaded inference artifact did not reproduce test retrieval")
    atomic_json(output / "metrics.json", {"config": config, "domain": "synthetic_bind_pose",
                                         "selected_epoch": state["epoch"], "validation": validation, "test": test,
                                         "reload_verified": True})


def load_export(output, device):
    state = load_torch(output / "inference.pt", device)
    gallery = load_torch(output / "gallery.pt", device)
    labels = state["labels"]
    if (gallery["ids"] != [row["id"] for row in labels]
            or gallery["dataset_sha256"] != state["config"]["dataset_sha256"]
            or gallery["encoder_sha256"] != file_hash(output / "inference.pt")
            or gallery["vectors"].shape != (len(labels), 16, DIMENSIONS)):
        raise ValueError("Gallery and encoder export disagree")
    model = encoder().to(device)
    model.load_state_dict(state["encoder"])
    model.eval()
    return labels, gallery["vectors"], model


@torch.inference_mode()
def predict(output, crops, device):
    labels, gallery, model = load_export(output, device)
    for path in crops:
        query = F.normalize(model(preprocess(path)[None].to(device)).float())
        rank = next(top_classes(query, gallery))[0]
        print(json.dumps({"crop": str(path), "ranked": [labels[index] for index in rank]}))


def compare(root):
    reports = {strategy: read_json(root / "runs" / strategy / "metrics.json") for strategy in ("random", "hard")}
    first, second = reports.values()
    keys = first["config"].keys() | second["config"].keys()
    if any(first["config"].get(key) != second["config"].get(key) for key in keys - {"strategy"}):
        raise ValueError("Runs do not share the same experiment configuration")
    left, right = (report["test"]["metrics"]["all"] for report in reports.values())
    atomic_json(root / "comparison.json", {"runs": reports, "test_macro_top1_delta": right["macro_top1"] - left["macro_top1"],
                                           "test_macro_top5_delta": right["macro_top5"] - left["macro_top5"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    sub.add_parser("compare")
    fitting = sub.add_parser("train")
    fitting.add_argument("--dataset", type=Path)
    fitting.add_argument("--strategy", choices=("random", "hard"), required=True)
    fitting.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    fitting.add_argument("--epochs", type=int, default=8)
    fitting.add_argument("--workers", type=int, default=6)
    fitting.add_argument("--batch-classes", type=int, default=16)
    fitting.add_argument("--output", type=Path, help="Separate directory for checkpoints and exports")
    fitting.add_argument("--continue-from", type=Path, help="Completed source run with last.pt and best.pt")
    fitting.add_argument("--scratch-pilot", action="store_true", help="CPU smoke training without pretrained weights")
    pred = sub.add_parser("predict")
    pred.add_argument("--run", type=Path, required=True)
    pred.add_argument("--crop", type=Path, action="append", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "init":
        print(initialize(root))
    elif args.command == "compare":
        compare(root)
    elif args.command == "predict":
        predict(args.run, args.crop, torch.device("cpu"))
    else:
        if args.epochs < 1 or args.batch_classes < 2 or args.workers < 0:
            parser.error("Invalid epochs, batch size, or worker count")
        dataset = (args.dataset or root / "dataset.json").resolve()
        if args.scratch_pilot and (args.device != "cpu" or dataset == root / "dataset.json"):
            parser.error("Scratch training requires a separate pilot dataset and CPU")
        if args.continue_from and (args.output is None or args.scratch_pilot):
            parser.error("Continuing a completed run requires a separate --output")
        train(root, dataset, args.strategy, args.device, args.epochs, args.workers,
              args.batch_classes, args.scratch_pilot, output=args.output, continue_from=args.continue_from)


if __name__ == "__main__":
    main()
