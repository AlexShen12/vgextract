"""Replay matched SigLIP votes and score independently reviewed skin-study evidence."""

from collections import Counter
from hashlib import sha256
import json
from statistics import mean

import numpy as np
from PIL import Image, ImageDraw

from .body_frame_matching import SkinMatcher, skin_lock_scope
from .body_operational import verified_crop
from .core import atomic_json, digest, read_json
from .mask_verifier import VERDICTS
from .media import file_hash
from .sam_appearances import AppearanceLocks, choose_masks, rank_masks, stage_lock
from .sam_skin_tracking import ARMS, track_receipt
from .sam_skin_verifier import accepts, case_id, read_prediction, score_frozen, verifier_metrics

MIDBEAST = "midbeast"
SECOND = "second"
XAYAH = ("appearance_3e3ca6b831b741fc4fe0", "appearance_f8a9f3d44edd68f4e79c")
SINGED = ("appearance_ae865eba35d0aa4ab30b", "appearance_2795d7200f2a54003f0e")


def by_time(rows):
    return sorted(rows, key=lambda row: (row["end"], row["start"], row["id"]))


def replay_votes(rows):
    ledger = AppearanceLocks()
    for row in by_time(rows):
        ledger.observe(row, row.get("vote"))
    return {"assignments": ledger.assigned, "events": ledger.events}


def check_keys(value, expected, description):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"Review needs exactly the frozen {description}")


def draft_reviews(root, study, study_sha, manifest, shortlist_sha):
    template = root / "verification" / "reviews.template.json"
    rows = {}
    for match in study["matches"]:
        rows[match["id"]] = {"source_sha256": match["source_sha256"], "gallery_sha256": match["gallery_sha256"],
                             "appearances": {}}
        for appearance in match["appearances"]:
            rows[match["id"]]["appearances"][appearance["id"]] = {
                "champion": None, "skin_id": None, "reason": "",
                "frames": {str(index): {"status": None, "polygons": []} for index in appearance["review_frames"]},
            }
    cases = {case["id"]: {"verdict": None, "reason": ""}
             for row in manifest["shortlists"] for case in row["cases"]}
    atomic_json(template, {"study_sha256": study_sha, "shortlists_sha256": shortlist_sha,
                           "reviewer": "", "predictions_seen": None, "matches": rows, "crops": cases})
    return {"complete": True, "review_template": str(template), "shortlists": len(manifest["shortlists"]),
            "crop_cases": len(cases)}


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def validate_frame_review(frame, size):
    status, polygons = frame["status"], frame["polygons"]
    width, height = size
    if (status not in ("visible", "absent", "ambiguous") or type(polygons) is not list
            or bool(polygons) != (status == "visible")):
        raise ValueError("Visible source reviews need native target polygons. Absent and ambiguous frames use []")
    if any(type(poly) is not list or len(poly) < 3 or any(type(point) is not list or len(point) != 2
           or any(type(n) is not int for n in point) or not 0 <= point[0] < width or not 0 <= point[1] < height
           for point in poly) for poly in polygons):
        raise ValueError("Source target polygons must be within native frame coordinates")
    if any(not sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(poly, poly[1:] + poly[:1]))
           for poly in polygons):
        raise ValueError("Source target polygons must enclose positive area")


def load_reviews(root, study, study_sha, manifest, shortlist_sha):
    path = root / "verification" / "reviews.json"
    if not path.is_file():
        raise ValueError("Use verify --prepare-only and freeze independent labels in verification/reviews.json")
    raw = path.read_bytes()
    reviews = json.loads(raw)
    if (reviews["study_sha256"] != study_sha or reviews["shortlists_sha256"] != shortlist_sha
            or reviews["predictions_seen"] is not False or not nonempty(reviews["reviewer"])):
        raise ValueError("Reviews must be independently frozen against this study and shortlist")
    check_keys(reviews["matches"], (row["id"] for row in study["matches"]), "matches")
    check_keys(reviews["crops"], (case["id"] for row in manifest["shortlists"] for case in row["cases"]), "crop IDs")
    for crop in reviews["crops"].values():
        if crop["verdict"] not in VERDICTS or not nonempty(crop["reason"]):
            raise ValueError("Every crop needs a clear, muddled, or uncertain review and a reason")
    for match in study["matches"]:
        review = reviews["matches"][match["id"]]
        if (review["source_sha256"] != match["source_sha256"] or review["gallery_sha256"] != match["gallery_sha256"]):
            raise ValueError(f"Source or gallery review belongs to another match: {match['id']}")
        check_keys(review["appearances"], (row["id"] for row in match["appearances"]), "appearance IDs")
        for appearance in match["appearances"]:
            label = review["appearances"][appearance["id"]]
            champion, skin = label["champion"], label["skin_id"]
            if (not nonempty(label["reason"]) or (champion is not None and champion not in match["team_pools"][appearance["team"]])
                    or (skin is not None and (champion is None or not isinstance(skin, str) or not skin.isdecimal()))):
                raise ValueError(f"Invalid identity review for {appearance['id']}")
            check_keys(label["frames"], map(str, appearance["review_frames"]), "source frame indices")
            for frame in label["frames"].values():
                validate_frame_review(frame, match["profile"]["size"])
    return reviews, sha256(raw).hexdigest()


def freeze_reviews(root, reviews_sha, *, create=False):
    directory = root / "verification"
    path = directory / "review-receipt.json"
    receipt = {"reviews_sha256": reviews_sha}
    with stage_lock(directory / ".reviews.lock"):
        if path.is_file():
            if read_json(path) != receipt:
                raise ValueError("Independent reviews changed after the verifier was started")
        elif create:
            atomic_json(path, receipt)
        else:
            return False
    return True


def load_matcher(root, match, cache, lock, lock_hash):
    directory = root / "inputs" / match["id"]
    gallery = directory / "gallery.json"
    if file_hash(gallery) != match["gallery_sha256"]:
        raise ValueError(f"Staged gallery checksum differs for {match['id']}")
    entries = read_json(gallery)
    for entry in entries:
        for key, hash_key in (("path", "sha256"), ("mask_path", "mask_sha256")):
            path = (directory / entry[key]).resolve()
            if not path.is_relative_to(directory.resolve()) or file_hash(path) != entry[hash_key]:
                raise ValueError(f"Gallery render changed: {entry['id']}")
            entry[key] = str(path)
    matcher = SkinMatcher(entries, cache, lock, lock_hash, match["patch"], encoder_name="siglip")
    if skin_lock_scope(matcher, match["team_pools"], match["source_sha256"], match["scope"]["roster_id"]) != match["scope"]:
        raise ValueError(f"Skin gallery or encoder scope differs for {match['id']}")
    return matcher


def measure_frame(mask, frame_review, size):
    if frame_review["status"] == "ambiguous":
        return {"target_coverage": None, "contamination": None}
    reference = Image.new("1", size)
    draw = ImageDraw.Draw(reference)
    for poly in frame_review["polygons"]:
        draw.polygon([tuple(point) for point in poly], fill=1)
    target = np.asarray(reference)
    foreground = np.asarray(mask.convert("L")) > 127 if mask is not None else np.zeros_like(target)
    found = int(np.count_nonzero(foreground))
    expected = int(np.count_nonzero(target))
    overlap = int(np.count_nonzero(foreground & target))
    return {"target_coverage": overlap / expected if expected else None,
            "contamination": (found - overlap) / found if found else None}


def score_geometry(source, match, directory, receipt, review):
    samples = {row["frame_index"]: row for row in receipt["samples"]}
    outcomes = []
    for key, label in review["frames"].items():
        index = int(key)
        frame = match["frames"][index]
        if file_hash(source / frame["path"]) != frame["sha256"]:
            raise ValueError(f"Source frame changed during review: {match['id']}/{index}")
        sample = samples.get(index)
        mask = None
        if sample and sample["visible"]:
            mask = verified_crop(directory / sample["mask_path"], sample["mask_sha256"])
        outcomes.append({"frame_index": index, "review": label["status"], "mask_present": mask is not None,
                         **measure_frame(mask, label, match["profile"]["size"])})
    return outcomes


def geometry_summary(rows):
    counts = Counter(row["review"] for row in rows)
    metrics = {}
    for key in ("target_coverage", "contamination"):
        values = [row[key] for row in rows if row[key] is not None]
        metrics[key] = {"mean": mean(values) if values else None, "scored_frames": len(values)}
    return {"reviewed_frames": dict(counts), "masks_present": sum(row["mask_present"] for row in rows), **metrics}


def classify_locks(replay, reviews):
    locks = []
    for event in replay["events"]:
        if event["event"] != "locked":
            continue
        truth = [reviews.get(identifier, {}) for identifier in event["supporting_appearances"]]
        wrong = any(any(label.get(key) is not None and label[key] != event[key]
                        for key in ("champion", "skin_id")) for label in truth)
        unknown = any(any(label.get(key) is None for key in ("champion", "skin_id")) for label in truth)
        locks.append({**event, "review": "false" if wrong else "unscorable" if unknown else "correct"})
    return {"locks": locks, "counts": {key: sum(row["review"] == key for row in locks)
                                      for key in ("correct", "false", "unscorable")}}


def regression(replay):
    false_events = [event for event in replay["events"] if event["event"] == "locked"
                    and event["team"] == "enemy" and event["champion"] == "Xayah"
                    and (event["skin_id"] == "498038" or set(XAYAH).intersection(event["supporting_appearances"]))]
    singed = replay["assignments"]["enemy"].get("Singed")
    return {"original_xayah_supports": any(event["skin_id"] == "498038" and event["supporting_appearances"] == list(XAYAH)
                                           for event in false_events),
            "xayah_lock_events": len(false_events),
            "singed_preserved": bool(singed and singed["skin_id"] == "27006"),
            "original_singed_supports": bool(singed and singed["supporting_appearances"] == list(SINGED))}


def score_arm(root, study_sha, match, arm, matcher, shortlists, reviews, predictions):
    review = reviews["matches"][match["id"]]["appearances"]
    source = root / "inputs" / match["id"]
    outcomes, pending, labeled = [], [], []
    statuses = Counter()
    for appearance in match["appearances"]:
        identifier = appearance["id"]
        receipt, directory = track_receipt(root, study_sha, match, arm, appearance)
        selection = shortlists.get((match["id"], arm, identifier))
        if receipt is None or selection is None:
            pending.append([match["id"], arm, identifier])
            continue
        if selection["track_digest"] != digest(receipt):
            raise ValueError(f"The frozen shortlist differs from the saved track for {identifier}")
        cases = selection["cases"]
        statuses[receipt["status"]] += 1
        if ([case["crop"] for case in cases] != choose_masks(receipt["crops"], directory)
                or any(case["id"] != case_id(match["id"], arm, identifier, case["frame_index"]) for case in cases)):
            raise ValueError(f"The shortlist crop differs from the saved mask for {identifier}")
        kept = [case for case in cases if accepts(predictions[case["id"]])]
        pending.extend(case["id"] for case in cases if "context" in case and predictions[case["id"]] is None)
        labels = [(reviews["crops"][case["id"]]["verdict"], predictions[case["id"]]) for case in cases]
        labeled.extend(labels)
        record = {**{key: appearance[key] for key in ("id", "team", "start", "end")},
                  "status": receipt["status"], "selected_seed": receipt["selected_seed"],
                  "crop_ids": [case["id"] for case in cases], "retained_ids": [case["id"] for case in kept],
                  "reviewed_geometry": score_geometry(source, match, directory, receipt, review[identifier]),
                  "verifier": verifier_metrics(labels)}
        for mode, selected in (("ungated", cases), ("gated", kept)):
            queries = [(directory / row["image_path"], directory / row["crop_mask_path"],
                        row["image_sha256"], row["crop_mask_sha256"]) for row in (case["crop"] for case in selected)]
            ranking, vote = rank_masks(matcher, queries, match["team_pools"][appearance["team"]])
            record[mode] = {"top": ranking[0] if ranking else None,
                            "vote": vote if receipt["status"] == "completed" else None}
        outcomes.append(record)
    result = {"complete": not pending, "pending": pending, "track_statuses": dict(statuses),
              "unavailable_markers": sum("context" not in case for row in outcomes for case in
                                         shortlists[(match["id"], arm, row["id"])]["cases"]),
              "geometry": geometry_summary([row for outcome in outcomes for row in outcome["reviewed_geometry"]]),
              "verifier": verifier_metrics(labeled), "appearances": outcomes}
    for mode in ("ungated", "gated"):
        decision = replay_votes([{**row, "vote": row[mode]["vote"]} for row in outcomes])
        result[mode] = {**decision, "review": classify_locks(decision, review)}
        if match["id"] == MIDBEAST:
            result[mode]["regression"] = regression(decision)
    return result


def score(args, study, study_sha):
    root = args.root
    frozen = score_frozen(root, study_sha, study["frozen_verifier"])
    previous = study["historical_replay"]
    report = {"experimental": True, "study_sha256": study_sha, "frozen_verifier": frozen,
              "historical_replay": {**previous, "regression": regression(previous),
                                    "frozen_appearance_report_sha256": study["frozen_verifier"]["appearance_report_sha256"]}}
    if args.frozen_only:
        output = root / "frozen-score.json"
        atomic_json(output, {**report, "complete": frozen["complete"]})
        return {"path": str(output), "complete": frozen["complete"], "passes_threshold": frozen["passes_threshold"]}
    output = root / "scorecard.json"
    path = root / "verification" / "shortlists.json"
    if not path.is_file():
        atomic_json(output, {**report, "complete": False, "pending": ["verified shortlists"]})
        return {"path": str(output), "complete": False, "pending": ["verified shortlists"]}
    manifest = read_json(path)
    if manifest["study_sha256"] != study_sha:
        raise ValueError("Cannot score shortlists from another study")
    if not (root / "verification" / "reviews.json").is_file():
        atomic_json(output, {**report, "complete": False, "pending": ["independent reviews"]})
        return {"path": str(output), "complete": False, "pending": ["independent reviews"]}
    shortlist_sha = file_hash(path)
    reviews, reviews_sha = load_reviews(root, study, study_sha, manifest, shortlist_sha)
    if not freeze_reviews(root, reviews_sha):
        atomic_json(output, {**report, "complete": False, "pending": ["frozen review receipt"]})
        return {"path": str(output), "complete": False, "pending": ["frozen review receipt"]}
    lock_hash = digest(read_json(args.siglip_lock))
    if lock_hash != study["siglip_lock_hash"]:
        raise ValueError("The SigLIP checkpoint differs from the prepared study")
    shortlists = {(row["match"], row["arm"], row["appearance_id"]): row for row in manifest["shortlists"]}
    expected = {(match["id"], arm, row["id"]) for match in study["matches"]
                for arm in ARMS for row in match["appearances"]}
    if set(shortlists) != expected or len(shortlists) != len(manifest["shortlists"]):
        raise ValueError("Shortlists must cover each frozen appearance and arm exactly once")
    predictions = {case["id"]: read_prediction(root, study_sha, case)
                   for row in manifest["shortlists"] for case in row["cases"]}
    if len(predictions) != sum(len(row["cases"]) for row in manifest["shortlists"]):
        raise ValueError("Shortlist crop IDs must be unique")
    matches = {}
    for match in study["matches"]:
        matcher = load_matcher(root, match, args.cache, args.siglip_lock, lock_hash)
        matches[match["id"]] = {"scope": match["scope"], "gallery_coverage": match["gallery_coverage"],
                               "arms": {arm: score_arm(root, study_sha, match, arm, matcher, shortlists, reviews,
                                                       predictions) for arm in ARMS}}
    complete = frozen["complete"] and all(arm["complete"] for match in matches.values() for arm in match["arms"].values())
    midbeast = matches[MIDBEAST]["arms"]
    baseline = midbeast[ARMS[0]]["ungated"]["regression"]
    candidates = []
    for name, arm in midbeast.items():
        for mode in ("ungated", "gated"):
            control = arm[mode]["regression"]
            passed = (arm["complete"] and not arm["track_statuses"].get("propagation_incomplete")
                      and baseline["original_xayah_supports"] and baseline["singed_preserved"]
                      and control["xayah_lock_events"] == 0 and control["singed_preserved"])
            control["passes_regression"] = bool(passed)
            if mode == "gated" and passed:
                candidates.append(name)
    cross_match = matches.get(SECOND, {}).get("arms", {})
    ready = bool(complete and frozen["passes_threshold"] and any(
        name in cross_match and not cross_match[name]["track_statuses"].get("propagation_incomplete")
        and all(cross_match[name]["verifier"]["labels"][label]["total"] for label in ("clear", "muddled"))
        for name in candidates
    ))
    atomic_json(output, {**report, "complete": complete, "shortlists_sha256": shortlist_sha,
                         "reviews_sha256": reviews_sha, "matches": matches, "second_match_present": SECOND in matches,
                         "gated_arms_passing_midbeast_regression": candidates, "qwen_seed_ranking_ready": ready})
    return {"path": str(output), "complete": complete, "second_match_present": SECOND in matches,
            "gated_arms_passing_midbeast_regression": candidates, "qwen_seed_ranking_ready": ready}
