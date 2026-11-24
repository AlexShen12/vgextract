"""Instance geometry metrics shared by validation and the held-out scorecard."""
from __future__ import annotations

import numpy as np


def iou(a, b):
    if a.shape != b.shape:
        raise ValueError("Mask geometry mismatch")
    return float((a & b).sum()) / max(1, int((a | b).sum()))


def boundary_f(a, b, tolerance=2):
    from scipy.ndimage import binary_erosion, binary_dilation
    ba, bb = a ^ binary_erosion(a), b ^ binary_erosion(b)
    if not ba.any() or not bb.any():
        return float(not ba.any() and not bb.any())
    precision = (ba & binary_dilation(bb, iterations=tolerance)).sum() / ba.sum()
    recall = (bb & binary_dilation(ba, iterations=tolerance)).sum() / bb.sum()
    return float(2 * precision * recall / max(1e-12, precision + recall))


def instance_metrics(truth, predictions):
    from scipy.optimize import linear_sum_assignment
    matrix = np.array([[iou(t, p) for p in predictions] for t in truth], dtype=float).reshape(len(truth), len(predictions))
    matches = dict(zip(*linear_sum_assignment(-matrix))) if truth and predictions else {}
    rows = []
    for index, target in enumerate(truth):
        pred = predictions[matches[index]] if index in matches else np.zeros_like(target)
        overlap = iou(target, pred)
        rows.append({"iou": overlap, "boundary_f": boundary_f(target, pred),
                     "recalled_at_50": overlap >= .5,
                     "contamination": float((pred & ~target).sum()) / max(1, int(pred.sum())),
                     "missing": not pred.any()})
    return {"instances": rows, "predicted_instances": len(predictions),
            "false_positive_instances": len(predictions) - sum(r["recalled_at_50"] for r in rows),
            "negative": len(truth) == 0, "negative_false_positive": not truth and bool(predictions)}


def aggregate(rows):
    instances = [i for row in rows for i in row["instances"]]
    negatives = [r for r in rows if r["negative"]]
    return {"images": len(rows), "instances": len(instances),
            "mean_instance_iou": float(np.mean([r["iou"] for r in instances])) if instances else None,
            "boundary_f": float(np.mean([r["boundary_f"] for r in instances])) if instances else None,
            "target_recall_at_50": float(np.mean([r["recalled_at_50"] for r in instances])) if instances else None,
            "contamination": float(np.mean([r["contamination"] for r in instances])) if instances else None,
            "missing_instances": sum(r["missing"] for r in instances),
            "negative_images": len(negatives),
            "negative_false_positive_images": sum(r["negative_false_positive"] for r in negatives),
            "false_positive_instances": sum(r["false_positive_instances"] for r in rows)}


def associate_bar(masks, bar_box):
    """Choose a body under a reviewed bar without a guessed torso point.

    Boxes and masks use crop pixels. Abstain when geometry is ambiguous.
    This policy is identical for both segmenters and never reads a body label.
    """
    from .sam_finetune import mask_box
    bx0, _, bx1, by1 = bar_box
    width = bx1 - bx0
    if width <= 0:
        raise ValueError("Invalid target health bar")
    cx = (bx0 + bx1) / 2
    candidates = []
    for index, mask in enumerate(masks):
        if not mask.any():
            continue
        x0, y0, x1, y1 = mask_box(mask)
        if y1 <= by1 or y0 < by1 - width or y0 > by1 + 3*width:
            continue
        dx = max(x0-cx, cx-x1, 0) / width
        center_dx = abs((x0+x1)/2-cx) / width
        score = dx*dx + .2*center_dx + max(0, y0-by1)/(3*width)
        if score <= 1.5:
            candidates.append((score, index))
    candidates.sort()
    if not candidates or (len(candidates) > 1 and candidates[1][0]-candidates[0][0] < .15):
        return None
    return candidates[0][1]
