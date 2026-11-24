"""Compare reference-view aggregation rules on a frozen skin metric export."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from .core import atomic_json, read_json
from .media import file_hash
from .skin_metric_train import RenderViews, encode, load_export, preprocess, score_rankings

VIEW_COUNTS = (1, 2, 4, 8)
SHORTLIST = 20


def candidate_scores(similarities, view_counts):
    # Ties use original gallery order, including within the shortlist.
    shortlist = similarities.amax(dim=2).argsort(dim=1, descending=True, stable=True)
    shortlist = shortlist[:, :SHORTLIST].sort(dim=1).values
    views = similarities.gather(1, shortlist[..., None].expand(-1, -1, similarities.shape[2]))
    ordered_views = views.topk(max(view_counts), dim=2).values
    scores = {count: ordered_views[:, :, :count].mean(dim=2) for count in view_counts}
    return shortlist, scores


def evaluate(queries, gallery, targets, identities, view_counts):
    classes, views, features = gallery.shape
    matrix = gallery.reshape(classes * views, features).T
    ranks = {count: [] for count in view_counts}
    for block in queries.split(128):
        shortlist, scores = candidate_scores((block @ matrix).reshape(-1, classes, views), view_counts)
        for count, values in scores.items():
            order = values.argsort(dim=1, descending=True, stable=True)[:, :5]
            ranks[count].extend(shortlist.gather(1, order).tolist())
    return {count: score_rankings(ranks[count], targets, identities) for count in view_counts}


def gameplay(track_dir, model, gallery, identities, selected, device):
    track = read_json(track_dir / "track.json")
    crops = [track_dir / row["crop"] for row in track["frames"]]
    if len(crops) < 2:
        raise ValueError("A gameplay track needs at least two crops")
    pixels = torch.stack([preprocess(path) for path in crops]).to(device)
    queries = F.normalize(model(pixels).float())
    rows = {}
    for scope, indices in (
        ("all", list(range(len(identities)))),
        ("known_champion", [i for i, row in enumerate(identities) if row["champion"] == track["champion"]]),
    ):
        if not indices:
            raise ValueError(f"No gallery entries for {track['champion']}")
        references = gallery[indices]
        classes, views, features = references.shape
        similarities = (queries @ references.reshape(-1, features).T).reshape(-1, classes, views)

        def rank(scores, candidates):
            order = scores.argsort(descending=True, stable=True)[:5].tolist()
            return [{"identity": identities[indices[candidates[i]]], "similarity": float(scores[i])}
                    for i in order]

        counts = tuple(sorted({1, selected}))
        shortlist, scores = candidate_scores(similarities, counts)
        per_frame = {count: [rank(values, candidates.tolist()) for values, candidates
                             in zip(scores[count], shortlist, strict=True)] for count in counts}
        # Use a shared shortlist so every candidate is scored on every frame.
        shared = shortlist.flatten().unique(sorted=True)
        ordered = similarities[:, shared].topk(max(counts), dim=2).values
        track_ranks = {count: rank(ordered[:, :, :count].mean(dim=(0, 2)), shared.tolist()) for count in counts}
        rows[scope] = {"per_frame": per_frame, "track_mean": track_ranks}
    return {"track": track, "crop_sha256": [file_hash(path) for path in crops],
            "exact_skin_verified": False, "predictions": rows}


@torch.inference_mode()
def run(root, source, output, track_dir, workers):
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    identities, gallery, model = load_export(source, device)
    baseline = read_json(source / "metrics.json")
    dataset_path = root / "dataset.json"
    if file_hash(dataset_path) != baseline["config"]["dataset_sha256"]:
        raise ValueError("Dataset does not match the frozen source run")
    dataset = {row["id"]: row for row in read_json(dataset_path)["identities"]}
    active = [dataset[row["id"]] for row in identities]

    def score(split, view_counts):
        views = RenderViews(root, active, split)
        targets = [label for label, _ in views.rows]
        return evaluate(encode(model, views, device, workers), gallery, targets, identities, view_counts)

    validation = score("validation", VIEW_COUNTS)
    if validation[1] != baseline["validation"]:
        raise ValueError("One-view ranking did not reproduce the source validation metrics")
    selected = max(VIEW_COUNTS, key=lambda count: validation[count]["metrics"]["all"]["macro_top1"])
    print(f"Selected {selected} reference views on validation", flush=True)
    best_multi = max(VIEW_COUNTS[1:], key=lambda count: validation[count]["metrics"]["all"]["macro_top1"])
    test = score("test", tuple(sorted({1, selected, best_multi})))
    if test[1] != baseline["test"]:
        raise ValueError("One-view ranking did not reproduce the source test metrics")
    report = {
        "source_run": str(source),
        "source_epoch": baseline["selected_epoch"],
        "encoder_sha256": file_hash(source / "inference.pt"),
        "gallery_sha256": file_hash(source / "gallery.pt"),
        "dataset_sha256": baseline["config"]["dataset_sha256"],
        "shortlist_size": SHORTLIST,
        "selected_views": selected,
        "best_multi_view_count": best_multi,
        "validation": validation,
        "test": test,
        "test_macro_top1_delta": (test[selected]["metrics"]["all"]["macro_top1"]
                                  - test[1]["metrics"]["all"]["macro_top1"]),
        "gameplay": gameplay(track_dir, model, gallery, identities, best_multi, device),
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "metrics.json", report)
    print(f"Test macro top-1 delta: {report['test_macro_top1_delta']:+.4f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--track", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        parser.error("Reranking must write to a separate output directory")
    run(args.root, args.source, args.output, args.track, args.workers)
