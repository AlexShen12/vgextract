"""Exact-ID precision, coverage and uncertainty with explicit empty denominators."""
import math
from collections import defaultdict


def source_aligned(labels, predictions, *, box_overlap=None):
    """Assign independently labeled bodies to predictions by native frame and box."""
    from .sam_npc_boxes import box_iou

    measure = box_overlap or box_iou

    def frame(row):
        return row.get("source_sha256"), round(row["time"], 6)

    labels_by_frame = defaultdict(list)
    legacy = {}
    for row in labels:
        if row.get("body_box") is not None:
            labels_by_frame[frame(row)].append(row)
        else:
            legacy[row["id"]] = row
    edges = []
    for prediction in predictions:
        reference = legacy.get(prediction["id"])
        prediction_source = prediction.get("source_sha256")
        if reference is not None and (not prediction_source or reference.get("source_sha256") in {None, prediction_source}):
            edges.append((-1.0, reference["id"], prediction["id"], reference, prediction))
            continue
        box = prediction.get("body_box")
        if box is None:
            continue
        for reference in labels_by_frame[frame(prediction)]:
            overlap = measure(reference["body_box"], box)
            if overlap >= .5:
                edges.append((-overlap, reference["id"], prediction["id"], reference, prediction))
    assigned_labels, assigned_predictions = set(), set()
    for neg_overlap, label_id, prediction_id, label, prediction in sorted(edges, key=lambda edge: edge[:3]):
        if label_id in assigned_labels or prediction_id in assigned_predictions:
            continue
        assigned_labels.add(label_id)
        assigned_predictions.add(prediction_id)
        overlap_iou = -neg_overlap
        if box_overlap and label.get("body_box"):
            overlap_iou = box_iou(label["body_box"], prediction["body_box"])
        yield label, {**prediction, "id": label_id, "localization_iou": overlap_iou}


def interval(successes, total):
    if not total:
        return None
    z=1.959963984540054; p=successes/total
    center=(p+z*z/(2*total))/(1+z*z/total)
    radius=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/(1+z*z/total)
    return [max(0,center-radius), min(1,center+radius)]


def identity_metrics(labels, predictions, *, field="champion", minimum_precision=.98, minimum_coverage=.7):
    if len({p["id"] for p in predictions}) != len(predictions):
        raise ValueError("Duplicate prediction IDs")
    known={r["id"]:r for r in labels}; predicted={r["id"]:r for r in predictions}
    if set(predicted)-set(known):
        raise ValueError("Predictions cite unlabelled samples")
    eligible=correct=accepted=false_unknown=covered=0
    for identifier, label in known.items():
        expected=label.get(field)
        value=predicted.get(identifier,{}).get(field)
        if field=="champion":
            from .body_identity import canonical_champion
            expected=canonical_champion(expected);value=canonical_champion(value)
        readable=label.get("visibility")=="readable" and expected is not None and not label.get("negative_kind")
        eligible+=int(readable); accepted+=int(value is not None)
        correct+=int(readable and value is not None and value==expected)
        covered+=int(readable and value is not None and value==expected)
        false_unknown+=int(not readable and value is not None)
    precision=correct/accepted if accepted else None; coverage=covered/eligible if eligible else None
    return {"correct":correct,"accepted":accepted,"eligible":eligible,"covered":covered,
            "precision":precision,"coverage":coverage,"false_matches_on_unknown":false_unknown,
            "precision_95ci":interval(correct,accepted),"coverage_95ci":interval(covered,eligible),
            "passed":precision is not None and coverage is not None and precision>=minimum_precision and coverage>=minimum_coverage}


def body_score(labels, predictions, *, arms=()):
    result={}
    known={l["id"]:l for l in labels}
    if len(known)!=len(labels) or any(p["id"] not in known for p in predictions):
        raise ValueError("Body predictions require unique source labels")
    for arm in sorted(set(arms) | {p["arm"] for p in predictions}):
        result[arm]={}
        for team in ("friendly","enemy"):
            references=[l for l in labels if l["team"]==team]
            rows=[{**p,"champion":p.get("champion") if p["team"]==team else "__wrong_team__"}
                  for p in predictions if p["arm"]==arm and known[p["id"]]["team"]==team]
            result[arm][team]=identity_metrics(references,rows)
            result[arm][team]["skin"]=identity_metrics(references,rows,field="skin_id")
            localized = {row["id"] for row in rows}
            result[arm][team]["localization_misses"] = sum(
                label["id"] not in localized for label in references
                if label.get("visibility") == "readable" and label.get("champion") and not label.get("negative_kind"))
        rows=sorted((p for p in predictions if p["arm"]==arm),key=lambda p:p["time"])
        previous={}; switches=0
        for row in rows:
            key=(row["track_id"],row["segment_id"],row["owner_epoch"])
            if row.get("champion"):
                switches+=int(key in previous and previous[key]!=row["champion"])
                previous[key]=row["champion"]
        result[arm]["identity_switches"]=switches
    return result
