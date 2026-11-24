"""Evaluate the operational name-reuse path against independent source labels."""

from pathlib import Path

from .body_appearance import validate_policy
from .body_campaign import coverage
from .body_identity import ARMS
from .body_roster import name_key
from .core import atomic_json, digest, read_json, read_rows
from .media import file_hash
from .recognition_metrics import body_score, interval, source_aligned


def human_verified(row):
    return row.get("reviewer_kind") == "human" and row.get("verified") is True and row.get("source_only") is True


def enough_labels(counts):
    return not (counts["pilot_champion_gap"] or counts["segment_gap"] or counts["negative_gap"]
                or any(counts["champion_gaps"].values()) or any(counts["held_out_champion_gaps"].values())
                or any(counts["held_out_team_gaps"].values())
                or any(split["positive_gap"] or split["negative_gap"] for split in counts["splits"].values()))


def audit_tracking(root):
    from .body_operational import overlap, track_revision, track_samples

    root = Path(root).resolve()
    campaign = read_json(root / "campaign.json")
    prepared = read_json(root / "prepared.json")
    if digest(campaign) != prepared["recipe"]["campaign_hash"]:
        raise ValueError("Body campaign changed after preparation")
    clips = [clip for clip in prepared["clips"] if clip["split"] == "development"]
    labels = [row for row in campaign["labels"] if row["split"] == "development"
              and row.get("visibility") == "readable" and row.get("body_box") and not row.get("negative_kind")]
    if not labels:
        raise ValueError("Tracking audit requires independent development body boxes")
    jobs = {job["id"]: job for job in prepared["jobs"]}
    proposals = []
    for clip in clips:
        for identifier in clip["jobs"]:
            for index, proposal in enumerate(jobs[identifier]["proposals"]):
                for observation in proposal.get("observations", ()):
                    frame = clip["frames"][observation["frame_index"]]
                    track = f"{identifier}-{index + 1}"
                    proposals.append({**observation, "id": f"{track}:{frame['id']}", "track_id": track,
                                      "team": proposal["team"], "time": frame["time"], "source_sha256": clip["source_sha256"]})
    tracking = track_revision(root)
    masks = [row for source in sorted({clip["source_sha256"] for clip in clips})
             for row in track_samples(root, {**prepared, "clips": clips}, source) if row.get("body_box")]
    report = {"prepared_hash": digest(prepared), "labels_hash": digest(labels), "tracking": tracking,
              "eligible": len(labels), "completed_tracking_jobs": sum(
                  (root / "tracks" / f"{identifier}-complete.json").exists()
                  for clip in clips for identifier in clip["jobs"])}
    def reference_coverage(label_box, proposal_box):
        return overlap(label_box, proposal_box) / ((label_box[2] - label_box[0]) * (label_box[3] - label_box[1]))

    for stage, predictions in (("proposal", proposals), ("mask", masks)):
        matches = list(source_aligned(labels, predictions, box_overlap=reference_coverage if stage == "proposal" else None))
        tracks = {}
        for label, row in matches:
            tracks.setdefault(label["segment_label_id"], set()).add(row["track_id"])
        report[stage] = {"matched": len(matches), "missed": len(labels) - len(matches),
                         "match_criterion": "covers at least half the labeled body box" if stage == "proposal" else "box IoU >= 0.5",
                         "mean_box_iou": sum(row["localization_iou"] for _, row in matches) / len(matches) if matches else None,
                         "wrong_team": sum(label["team"] != row["team"] for label, row in matches),
                         "tracks_per_labeled_segment": {key: len(value) for key, value in tracks.items()}}
    if track_revision(root) != tracking:
        raise ValueError("Tracking changed during audit")
    atomic_json(root / "tracking-development-audit.json", report)
    return report


def score_comparisons(root, split, policy_path):
    from .body_operational import track_revision

    root = Path(root).resolve()
    receipt = root / f"{split}-score.json"
    if receipt.exists():
        raise ValueError("This evaluation already has a receipt")
    campaign = read_json(root / "campaign.json")
    prepared = read_json(root / "prepared.json")
    if digest(campaign) != prepared["recipe"]["campaign_hash"]:
        raise ValueError("Body campaign changed after preparation")
    policy = validate_policy(root, read_json(policy_path), prepared, campaign, "combined")
    policy_hash = digest(policy)
    labels = [row for row in campaign["labels"] if row["split"] == split]
    if not labels:
        raise ValueError("Comparison scoring needs independent source labels")
    counts = coverage(campaign)
    human = bool(labels) and all(human_verified(row) and row.get("body_box") for row in labels)
    validation = None
    if split == "held_out":
        validation = read_json(root / "validation-score.json")
        if not human or not enough_labels(counts) or validation.get("policy_hash") != policy_hash:
            raise ValueError("Held-out scoring needs independent source labels and a matching validation receipt")
    predictions, reports, arms = [], [], []
    tracking = track_revision(root)
    for gallery, context in ARMS:
        arm = f"{gallery}-{'context' if context else 'appearance'}"
        path = root / "comparisons" / split / arm / "predictions.jsonl"
        report = read_json(path.with_name("report.json"))
        if (report["arm"] != arm or report["policy_hash"] != policy_hash or report["tracking"] != tracking
                or report["predictions_sha256"] != file_hash(path)):
            raise ValueError(f"Comparison inputs changed for {arm}")
        predictions.extend(row for _, row in source_aligned(labels, read_rows(path)))
        arms.append(arm)
        reports.append(report)
    metrics = body_score(labels, predictions, arms=arms)
    qualified = [arm for arm in arms if all(metrics[arm][team]["passed"] for team in ("friendly", "enemy"))
                 and (validation is None or arm in validation["qualified_arms"])]
    result = {"policy_hash": policy_hash, "labels_hash": digest(labels), "metrics": metrics,
              "qualified_arms": qualified, "human_verified": human, "coverage": counts, "runs": reports,
              "release_passed": bool(validation and qualified)}
    atomic_json(receipt, result)
    return result


def score_identities(root, result_paths, split, gallery_kind, policy_path=None):
    from .body_operational import track_revision
    from .body_slot_context import skim_revision_matches

    root = Path(root).resolve()
    receipt = root / f"identity-{gallery_kind}-{split}-score.json"
    if receipt.exists():
        raise ValueError("This evaluation already has a receipt")
    campaign = read_json(root / "campaign.json")
    prepared = read_json(root / "prepared.json")
    if digest(campaign) != prepared["recipe"]["campaign_hash"]:
        raise ValueError("Body campaign changed after preparation")
    policy = validate_policy(root, read_json(policy_path or root / "appearance-policy.json"), prepared, campaign, gallery_kind)
    policy_hash = digest(policy)
    labels = [row for row in campaign["labels"] if row["split"] == split]
    if not labels:
        raise ValueError("Body evaluation requires independent source labels")
    if split == "held_out" and (not all(human_verified(row) and row.get("body_box") for row in labels)
                                or not enough_labels(coverage(campaign))):
        raise ValueError("Held-out scoring needs independent source boxes and complete label coverage")
    validation_path = root / f"identity-{gallery_kind}-validation-score.json"
    if split == "held_out":
        validation = read_json(validation_path) if validation_path.is_file() else {}
        if (validation.get("policy_hash") != policy_hash
                or not all(validation.get("metrics", {}).get(team, {}).get("passed") for team in ("friendly", "enemy"))):
            raise ValueError("Held-out identity scoring needs passing operational validation for this policy")
    results = []
    tracking = track_revision(root)
    for path in result_paths:
        path = Path(path).resolve()
        result = read_json(path)
        recipe = result["recipe"]
        if (not path.is_relative_to(root / "identities") or path.stem != digest(recipe)[:20]
                or recipe["prepared"] != digest(prepared) or recipe["policy"] != policy_hash
                or recipe["gallery_kind"] != gallery_kind or recipe["tracks"] != tracking
                or (recipe.get("slot_skim") and not skim_revision_matches(recipe["slot_skim"]))):
            raise ValueError("Identity result was produced with different prepared inputs or policy")
        results.append((file_hash(path), result))
    sources = {result["recipe"]["source_sha256"] for _, result in results}
    if len(sources) != len(results):
        raise ValueError("Provide one identity revision per source")
    if split == "held_out" and any(roster.get("status") == "experimental"
                                   or roster.get("confirmation_method") in {"unreviewed_cross_opening", "single_match_membership_skim"}
                                   for _, result in results for roster in result["rosters"]):
        raise ValueError("Unreviewed cross-opening identities are experimental")
    predictions = []
    wrong_names = 0
    first_readable, first_named = {}, {}
    for label in labels:
        if label.get("visibility") == "readable" and label.get("champion") and not label.get("negative_kind"):
            segment = label["segment_label_id"]
            if "time" in label:
                first_readable[segment] = min(first_readable.get(segment, label["time"]), label["time"])
    expected_rosters = [label for label in campaign.get("roster_labels", [])
                        if label.get("split") == split and human_verified(label)]
    confirmed_rosters = {}
    unbound_names = set()
    observed_sources = {row["source_sha256"] for row in labels if row.get("source_sha256")}
    if observed_sources and not sources <= observed_sources:
        raise ValueError("Identity results must belong to the scored split")
    for _, result in results:
        source = result["recipe"]["source_sha256"]
        rows = [row for row in result["predictions"] if row["split"] == split and row["source_sha256"] == source]
        unbound_names.update((row["source_sha256"], row.get("scope"), name_key(row.get("player_name")))
                             for row in rows if row.get("visual_attempt") is not None)
        for reference, row in source_aligned(labels, rows):
            wrong_name = bool(row.get("champion") and row.get("binding_id") and (
                reference.get("champion") != row["champion"] or (reference.get("player_name")
                and name_key(reference["player_name"]) != name_key(row.get("player_name")))))
            wrong_names += int(wrong_name)
            predictions.append({**row, "arm": gallery_kind})
            segment = reference["segment_label_id"]
            available = row.get("identity_available_at")
            if (segment in first_readable and available is not None and not wrong_name
                    and row["champion"] == reference.get("champion") and row["team"] == reference["team"]):
                first_named[segment] = min(first_named.get(segment, available), available)
        for roster in result["rosters"]:
            if roster["status"] == "confirmed":
                key = (roster["source_sha256"], roster["segment_id"])
                confirmed_rosters.setdefault(key, []).append(roster)
    roster_checks = []
    for label in expected_rosters:
        matches = confirmed_rosters.get((label["source_sha256"], label.get("segment_id")), [])
        if "interval" in label:
            # Cross-opening truth uses source time bounds, independent of the generated scope ID.
            matches = [roster for (source, _), group in confirmed_rosters.items() if source == label["source_sha256"]
                       for roster in group if roster.get("interval") == label["interval"]]
        expected = sorted(tuple(pair) for pair in label["membership"])
        exact = len(matches) == 1 and expected == sorted(tuple(pair) for pair in matches[0]["membership"])
        roster_checks.append({"label_id": label.get("id"), "roster_ids": [r["id"] for r in matches], "exact": exact})
    counts = coverage(campaign)
    metrics = body_score(labels, predictions, arms=[gallery_kind])[gallery_kind]
    human = all(human_verified(row) for row in labels)
    qualified = (split == "held_out" and human and enough_labels(counts) and observed_sources <= sources
                 and bool(roster_checks) and all(metrics.get(team, {}).get("passed") for team in ("friendly", "enemy")))
    attempts = sum(result["counts"].get("visual_attempts", 0) for _, result in results)
    backfilled = sum(bool(row.get("backfilled") and row.get("champion")) for row in predictions)
    named = sum(bool(row.get("champion")) for row in predictions)
    rosters_correct = sum(row["exact"] for row in roster_checks)
    report = {
        "split": split, "arm": gallery_kind, "human_verified": human, "release_passed": qualified,
        "result_hashes": [sha for sha, _ in results], "policy_hash": policy_hash, "labels_hash": digest(labels),
        "metrics": metrics, "coverage": counts, "false_name_transfers": wrong_names,
        "roster_checks": roster_checks, "rosters_correct": rosters_correct,
        "roster_accuracy": rosters_correct / len(roster_checks) if roster_checks else None,
        "roster_accuracy_95ci": interval(rosters_correct, len(roster_checks)),
        "visual_attempts_per_unbound_name": attempts / len(unbound_names) if unbound_names else None,
        "backfill_coverage": backfilled / named if named else None,
        "naming_delay_seconds": {key: max(0, first_named[key] - first_readable[key]) for key in first_named},
        "peak_rss_bytes": max((result.get("peak_rss_bytes", 0) for _, result in results), default=0) or None,
        "operation_counts": {key: sum(result["counts"].get(key, 0) for _, result in results)
                             for key in ("visual_attempts", "registry_hits", "skipped_visual_calls", "backfilled", "encoder_calls")},
        "seconds": sum(result["seconds"] for _, result in results),
        "uncertainty_note": "Sample intervals include correlated observations. Track segment counts are reported separately.",
    }
    atomic_json(receipt, report)
    return report
