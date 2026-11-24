"""Calibrate body feature policies on paired development clips, then validate the winner."""

from collections import defaultdict
from itertools import product
from pathlib import Path
import resource
import sys
import time

import numpy as np

from .body_identity import ARMS, TrackIdentity
from .body_names import select_frames
from .body_operational import LazyMatcher, track_samples
from .core import atomic_json, digest, implementation_hash, read_json, write_rows
from .media import file_hash
from .recognition_metrics import body_score, source_aligned

COMPONENTS = ["body_appearance.py", "body_features.py", "body_identity.py", "body_operational.py", "body_names.py",
              "recognition_metrics.py"]


def policy_inputs(prepared, campaign, gallery_kind):
    return {"prepared_hash": digest(prepared), "gallery_hash": digest(prepared["gallery"]),
            "labels_hash": digest([row for row in campaign["labels"] if row["split"] in {"development", "validation"}]),
            "gallery_kind": gallery_kind, "implementation": implementation_hash(COMPONENTS)}


def validate_policy(root, policy, prepared, campaign, gallery_kind):
    fields = {key: value for key, value in policy.items() if key != "development_freeze"}
    expected = policy_inputs(prepared, campaign, gallery_kind)
    if policy.get("inputs") != expected or policy.get("development_freeze") != digest(fields):
        raise ValueError("Appearance policy does not match the prepared inputs or implementation")
    receipt = read_json(Path(root) / policy["validation_receipt"])
    if (digest(receipt) != policy["validation_hash"]
            or not all(receipt.get(team, {}).get("passed") for team in ("friendly", "enemy"))):
        raise ValueError("Appearance policy requires a passing validation receipt")
    return {**policy, "matching": {**policy["matching"], "development_freeze": policy["development_freeze"]}}


def raw_predictions(matcher, samples, *, context=False):
    rows = []
    roster = {"champions": sorted({entry["champion"] for entry in matcher.prepared["gallery"]})}
    for row in samples:
        candidate_roster = {"champions": row["accepted_roster"]} if context and row.get("accepted_roster") else roster
        prediction = matcher(row, candidate_roster) if row["visible"] else {"champion": None}
        hud = row.get("accepted_hud_champion") if context and row.get("hud_body_binding_accepted") else None
        if hud and prediction.get("champion") != hud:
            prediction = {**prediction, "champion": None, "score": None, "reason": "HUD identity conflict"}
        rows.append((row, prediction))
    return rows


def appearance_predictions(raw, matching):
    tracker = TrackIdentity()
    for row, prediction in raw:
        local = prediction.get("used_local_features")
        threshold = matching["score"] if local or not matching.get("patch_weight") else matching["global_score"]
        margin = matching["margin"] if local or not matching.get("patch_weight") else matching["global_margin"]
        candidate = prediction.get("champion")
        if prediction.get("score") is None or prediction["score"] < threshold or prediction["margin"] < margin:
            candidate = None
        result = tracker.update({**row, "time": row["vote_time"]}, {"champion": candidate})
        yield {**row, **result, "time": row["time"]}


def score(raw, labels, matching):
    predictions = appearance_predictions(raw, matching)
    return body_score(labels, [{**row, "arm": "trial"} for _, row in source_aligned(labels, predictions)],
                      arms=["trial"])["trial"]


def compare(root, policy_path, split, only_arm=None):
    from .body_campaign import verify_campaign
    from .body_operational import track_revision

    root = Path(root).resolve()
    campaign = verify_campaign(read_json(root / "campaign.json"))
    prepared = read_json(root / "prepared.json")
    if prepared["recipe"]["campaign_hash"] != digest(campaign) or prepared["gallery"] != campaign["gallery"]:
        raise ValueError("Body campaign changed after preparation")
    policy = validate_policy(root, read_json(policy_path), prepared, campaign, "combined")
    policy_hash = digest(policy)
    if split == "held_out":
        validation_path = root / "validation-score.json"
        validation = read_json(validation_path) if validation_path.is_file() else {}
        if validation.get("policy_hash") != policy_hash or not validation.get("qualified_arms"):
            raise ValueError("At least one arm must pass validation for this appearance policy")
    output = root / "comparisons" / split
    tracking = track_revision(root)
    clips = [clip for clip in prepared["clips"] if clip["split"] == split]
    samples = select_frames((row for source in sorted({clip["source_sha256"] for clip in clips})
                             for row in track_samples(root, {**prepared, "clips": clips}, source)),
                            cadence=policy.get("cadence", "quality"))
    reports = []
    encoder = None
    for gallery, context in ARMS:
        arm = f"{gallery}-{'context' if context else 'appearance'}"
        if only_arm and only_arm != arm:
            continue
        started = time.monotonic()
        matcher = LazyMatcher(root, prepared, root / "encoder-lock.json", policy, gallery, "cpu")
        if encoder is not None:
            matcher.encoder = encoder
        raw = raw_predictions(matcher, samples, context=context)
        encoder = matcher.__dict__.get("encoder", encoder)
        path = output / arm / "predictions.jsonl"
        # Keep enough source geometry for independent alignment, without names or input HUD fields.
        fields = ("id", "source_sha256", "frame_id", "body_box", "track_id", "team", "segment_id", "owner_epoch",
                  "track_epoch", "time", "vote_time", "champion", "skin_id", "identity_votes")
        write_rows(path, [{**{key: row[key] for key in fields if key in row}, "arm": arm}
                          for row in appearance_predictions(raw, policy["matching"])])
        report = {"arm": arm, "policy_hash": policy_hash, "tracking": tracking, "predictions_sha256": file_hash(path),
                  "seconds": time.monotonic() - started, "encoder_calls": matcher.encoder_calls,
                  "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)}
        atomic_json(path.with_name("report.json"), report)
        reports.append(report)
    if not reports or track_revision(root) != tracking:
        raise ValueError("Unknown comparison arm or tracking changed during comparison")
    return {"split": split, "reports": reports}


def rank(metrics):
    precision = min(metrics[team]["precision"] or 0 for team in ("friendly", "enemy"))
    coverage = min(metrics[team]["coverage"] or 0 for team in ("friendly", "enemy"))
    return precision >= .98, coverage if precision >= .98 else precision, coverage


def calibrate(raw, labels, matching):
    grid = list(product((.55, .65, .75, .8, .85, .9, .95), (.03, .06, .1, .15, .2)))

    def search(base, routes):
        best = None
        for threshold, separation in grid:
            current = {**base}
            for prefix in routes:
                current[prefix + "score"] = threshold
                current[prefix + "margin"] = separation
            metrics = score(raw, labels, current)
            ordering = (*rank(metrics), threshold, separation)
            if best is None or ordering > best[0]:
                best = ordering, current, metrics
        return best[1:]

    policy, metrics = search(matching, ("", "global_"))
    routes = {bool(prediction.get("used_local_features")) for _, prediction in raw if prediction.get("score") is not None}
    if matching.get("patch_weight") and routes == {False, True}:
        # Re-evaluate complete vote sequences with one route fixed. A track can use both routes.
        policy, _ = search(policy, ("",))
        policy, metrics = search(policy, ("global_",))
    return policy, metrics


def fit(root, cache, *, gallery_kind="combined", device="cpu"):
    from .body_campaign import verify_campaign

    root = Path(root).resolve()
    if (root / "appearance-policy.json").exists():
        raise ValueError("Appearance policy already frozen. Use a new campaign revision.")
    campaign = verify_campaign(read_json(root / "campaign.json"), cache)
    prepared = read_json(root / "prepared.json")
    if digest(campaign) != prepared["recipe"]["campaign_hash"] or campaign["gallery"] != prepared["gallery"]:
        raise ValueError("Body campaign changed after preparation")
    labels = {split: [row for row in campaign["labels"] if row["split"] == split] for split in ("development", "validation")}
    if any(not rows for rows in labels.values()) or not prepared["gallery"]:
        raise ValueError("Fit requires development and validation labels and an admitted gallery")
    selected = defaultdict(list)
    source_hashes = {clip["source_sha256"] for clip in prepared["clips"] if clip["split"] in labels}
    for source in sorted(source_hashes):
        subset = {**prepared, "clips": [clip for clip in prepared["clips"] if clip["split"] in labels]}
        for row in track_samples(root, subset, source):
            selected[row["split"]].append(row)
    for split in labels:
        if not any(source_aligned(labels[split], selected[split])):
            raise ValueError(f"{split} labels do not cite prepared body observations")

    inputs = policy_inputs(prepared, campaign, gallery_kind)
    reports = []
    encoder = None
    initial_matching = {"development_freeze": "development-search", "score": 0, "margin": 0,
                        "global_score": 0, "global_margin": 0, "weights": [1, 0, 0]}

    def trial(label, features, matching, cadence="quality"):
        nonlocal encoder
        for report in reports:
            if (report["features"], report["matching_settings"], report["cadence"]) == (features, matching, cadence):
                return report
        start = time.monotonic()
        matcher = LazyMatcher(root, prepared, root / "encoder-lock.json",
                              {"features": features, "matching": matching}, gallery_kind, device)
        if encoder is not None:
            matcher.encoder = encoder
        raw = raw_predictions(matcher, select_frames(selected["development"], cadence=cadence))
        encoder = matcher.__dict__.get("encoder", encoder)
        frozen, metrics = calibrate(raw, labels["development"], matching)
        result = {"trial": label, "features": features, "matching_settings": matching,
                  "matching": frozen, "cadence": cadence, "development": metrics,
                  "seconds": time.monotonic() - start}
        reports.append(result)
        atomic_json(root / "appearance-trials.json", {"inputs": inputs, "trials": reports})
        return result

    def winner(candidates):
        def ordering(item):
            features, matching = item["features"], item["matching_settings"]
            complexity = (bool(features.get("render_variant")) + bool(matching.get("patch_weight"))
                          + sum(bool(weight) for weight in matching.get("weights", []))
                          + (matching.get("agreement", 1) > 1) + bool(matching.get("template_quota")))
            return (*rank(item["development"]), -item["seconds"], -complexity)

        return max(candidates, key=ordering)

    geometry = [trial("stretched baseline", {"crop": "legacy", "weights": [1, 0, 0]}, initial_matching)]
    geometry.extend(trial(f"padding {padding}", {"padding": padding, "weights": [1, 0, 0]}, initial_matching)
                    for padding in (0, .1, .2))
    best = winner(geometry)
    if gallery_kind != "gameplay":
        heights = [row["body_box"][3] - row["body_box"][1] for row in selected["development"] if row.get("body_box")]
        effects = [{"body_height": max(8, round(float(np.median(heights))))}] if heights else []
        effects += [{"blur": .6}, {"brightness": .85}, {"jpeg_quality": 70}]
        candidates = [best] + [trial("render quality", {**best["features"], "render_variant": effect}, initial_matching)
                               for effect in effects]
        best = winner(candidates)

    candidates = [best]
    for layer in (-1, -4):
        for weight in (.25, .5):
            candidates.append(trial("foreground patches", {**best["features"], "patch_layer": layer},
                                    {**best["matching_settings"], "patch_weight": weight}))
    best = winner(candidates)

    candidates = [best]
    for rgb in range(5):
        for gray in range(5 - rgb):
            weights = [rgb / 4, gray / 4, (4 - rgb - gray) / 4]
            candidates.append(trial("appearance fusion", {**best["features"], "weights": weights},
                                    {**best["matching_settings"], "weights": weights}))
    best = winner(candidates)
    candidates = [best]
    for support, quota in ((3, None), (1, 64), (3, 64)):
        settings = {**best["matching_settings"], "agreement": support, "template_quota": quota}
        candidates.append(trial("reference agreement", best["features"], settings))
    best = winner(candidates)

    best = winner([best, trial("fixed cadence", best["features"], best["matching_settings"], "fixed")])
    validation_matcher = LazyMatcher(root, prepared, root / "encoder-lock.json",
                                     {"features": best["features"], "matching": best["matching_settings"]}, gallery_kind, device)
    if encoder is not None:
        validation_matcher.encoder = encoder
    validation_frames = select_frames(selected["validation"], cadence=best["cadence"])
    validation = score(raw_predictions(validation_matcher, validation_frames), labels["validation"], best["matching"])
    receipt = "appearance-validation.json"
    atomic_json(root / receipt, validation)
    passed = all(best["development"][team]["passed"] and validation[team]["passed"] for team in ("friendly", "enemy"))
    if passed:
        policy = {"inputs": inputs, "features": best["features"], "matching": best["matching"], "cadence": best["cadence"],
                  "validation_receipt": receipt, "validation_hash": digest(validation)}
        policy["development_freeze"] = digest(policy)
        atomic_json(root / "appearance-policy.json", policy)
    return {"policy_frozen": passed, "selected_trial": best, "validation": validation, "trials": len(reports)}
