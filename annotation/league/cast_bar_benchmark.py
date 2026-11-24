"""Frozen development policy and occurrence-level cast-bar evaluation."""
import argparse
from collections import Counter
import json
from pathlib import Path

from .core import atomic_json, digest, finite, now, read_json, read_rows
from .recognition_dataset import CASES, coverage, validate, validate_label
from .recognition_metrics import interval


def overlap(a,b):
    if a["start"]==a["end"] or b["start"]==b["end"]:
        return float(abs(a["start"]-b["start"])<=.25)
    intersection=max(0,min(a["end"],b["end"])-max(a["start"],b["start"]))
    if b.get("left_censored") or b.get("right_censored"):
        return intersection/max(b["end"]-b["start"],1e-9)
    return intersection/max(max(a["end"],b["end"])-min(a["start"],b["start"]),1e-9)


def score_occurrences(labels,predictions,negative_minutes=0,negative_clip_ids=()):
    if len({p["id"] for p in predictions})!=len(predictions): raise ValueError("Duplicate channel predictions")
    pairs=[]
    for i,p in enumerate(predictions):
        for j,l in enumerate(labels):
            if p["clip_id"]==l["clip_id"] and p["owner_id"]==l["owner_id"]:
                value=overlap(p,l)
                if value>=.5:pairs.append((value,i,j))
    used_p=set();used_l=set();matches=[]
    for _,i,j in sorted(pairs,reverse=True):
        if i not in used_p and j not in used_l:
            used_p.add(i);used_l.add(j);matches.append((predictions[i],labels[j]))
    readable=[l for l in labels if l["visibility"]=="readable" and l.get("ability_id")]
    accepted=[p for p in predictions if p.get("ability_id")]
    correct=sum(p.get("ability_id") is not None and p.get("ability_id")==l.get("ability_id") and l["visibility"]=="readable" for p,l in matches)
    classified=sum(p["channel_kind"]==l["channel_kind"] and l["channel_kind"] in {"recall","ability"} for p,l in matches)
    class_total=sum(l["channel_kind"] in {"recall","ability"} for l in labels)
    precision=correct/len(accepted) if accepted else None;cover=correct/len(readable) if readable else None
    classification=classified/class_total if class_total else None
    false_events=len(predictions)-len(matches)
    return {"exact_id_correct":correct,"accepted_named":len(accepted),"readable_named":len(readable),
        "exact_id_precision":precision,"readable_named_coverage":cover,"kind_accuracy":classification,
        "precision_95ci":interval(correct,len(accepted)),"coverage_95ci":interval(correct,len(readable)),
        "matched_occurrences":len(matches),"unmatched_predictions":false_events,"missed_occurrences":len(labels)-len(matches),
        "duplicate_candidates":sum(i not in used_p and any(overlap(p,l)>=.5 and p["clip_id"]==l["clip_id"] for l in labels) for i,p in enumerate(predictions)),
        "false_events_per_negative_minute":sum(p["clip_id"] in negative_clip_ids for p in predictions)/negative_minutes if negative_minutes else None,
        "onset_absolute_errors":[abs(p["start"]-l["start"]) for p,l in matches if not l.get("left_censored")],
        "offset_absolute_errors":[abs(p["end"]-l["end"]) for p,l in matches if not l.get("right_censored")],
        "thresholds_passed":precision is not None and cover is not None and classification is not None
            and precision>=.98 and cover>=.8 and classification>=.95}


def validate_predictions(predictions,clips):
    known={c["id"]:c for c in clips}
    if len({p["id"] for p in predictions})!=len(predictions):raise ValueError("Duplicate occurrence IDs")
    for prediction in predictions:
        clip=known.get(prediction["clip_id"])
        if (clip is None or not finite(prediction["start"]) or not finite(prediction["end"])
                or not clip["start"]<=prediction["start"]<=prediction["end"]<=clip["end"]
                or prediction["owner_id"]!="selected" or prediction["channel_kind"] not in {"ability","recall","unknown"}):
            raise ValueError("Prediction crosses source clip/scope boundaries")


def reader_contract(record,layouts):
    readers=record.get("reader_signatures")
    if readers is not None:
        if not isinstance(readers,dict) or not set(layouts)<=set(readers) or not all(isinstance(v,str) and v for v in readers.values()):
            raise ValueError("Each calibrated layout requires a frozen reader signature")
        signature=digest(readers)
        if record.get("reader_signature",signature)!=signature:raise ValueError("Aggregate reader signature mismatch")
        return signature
    if len(set(layouts))>1:raise ValueError("Multiple layouts require reader_signatures keyed by layout ID")
    if not isinstance(record.get("reader_signature"),str) or not record["reader_signature"]:raise ValueError("Missing reader signature")
    return record["reader_signature"]


def evaluate(root,predictions_path,split):
    root=Path(root); data=read_json(root/"dataset.json");sources=validate(data)
    freeze=read_json(root/"reader-freeze.json")
    envelope=read_json(Path(predictions_path))
    layouts={s["layout_id"] for s in sources.values()}
    if reader_contract(envelope,layouts)!=freeze["reader_signature"] or envelope["catalog_id"]!=freeze["catalog_id"]:
        raise ValueError("Predictions differ from the frozen reader/catalog")
    development={c["id"]:read_json(root/"labels"/(c["id"]+".json")) for c in data["clips"] if sources[c["source_id"]]["split"]=="development"}
    if digest(development)!=freeze["development_labels_hash"]:raise ValueError("Development labels changed after freezing")
    selected=[c for c in data["clips"] if sources[c["source_id"]]["split"]==split]
    references=[];minutes=0;human=True;negatives=[]
    for clip in selected:
        label=read_json(root/"labels"/(clip["id"]+".json"));validate_label(label,clip)
        human &= label["verified"] and label["reviewer_kind"]=="human" and label["source_only"] and label["exhaustive"]
        references.extend({**e,"clip_id":clip["id"]} for e in label["events"])
        if not label["events"] and label["exhaustive"]:
            minutes+=(clip["end"]-clip["start"])/60;negatives.append(clip["id"])
    selected_ids={c["id"] for c in selected}
    predictions=envelope["occurrences"]
    validate_predictions(predictions,selected)
    if any(p["clip_id"] not in selected_ids for p in predictions):raise ValueError("Predictions cross evaluation splits")
    result=score_occurrences(references,predictions,minutes,negatives)
    cases=Counter(tag for c in selected for tag in set(c["case_tags"]))
    result.update(split=split,human_verified=bool(human and selected),case_counts=dict(cases),
        dataset_coverage=coverage(root),reader_freeze=digest(freeze),
        release_passed=bool(split=="held_out" and selected and human and result["thresholds_passed"]
                            and len(references)>=50 and all(cases[c]>=5 for c in CASES) and coverage(root)["release_ready"]))
    receipt=root/(split+"-evaluation.json")
    if receipt.exists():raise ValueError("Evaluation already recorded; preserve it and create a new revision")
    atomic_json(receipt,result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("command",choices=("freeze-reader","score"))
    p.add_argument("--root",type=Path,required=True);p.add_argument("--input",type=Path,required=True)
    p.add_argument("--split",choices=("validation","held_out"),default="held_out");args=p.parse_args()
    if args.command=="freeze-reader":
        data=read_json(args.root/"dataset.json");sources=validate(data);policy=read_json(args.input)
        if not {"catalog_id","policy","development_predictions"}<=set(policy):raise ValueError("Incomplete development freeze")
        signature=reader_contract(policy,{s["layout_id"] for s in sources.values()})
        labels={c["id"]:read_json(args.root/"labels"/(c["id"]+".json")) for c in data["clips"] if sources[c["source_id"]]["split"]=="development"}
        if not labels:raise ValueError("No development labels")
        if any(p["clip_id"] not in labels for p in policy["development_predictions"]):raise ValueError("Freeze predictions cross development split")
        validate_predictions(policy["development_predictions"],[c for c in data["clips"] if c["id"] in labels])
        references=[{**e,"clip_id":clip_id} for clip_id,label in labels.items() for e in label["events"]]
        report=score_occurrences(references,policy["development_predictions"])
        if not report["thresholds_passed"]:raise ValueError("Development thresholds must pass before freezing")
        result={**policy,"reader_signature":signature,"development_report":report,"development_labels_hash":digest(labels),"created_at":now()}
        if (args.root/"reader-freeze.json").exists():raise ValueError("Reader already frozen")
        atomic_json(args.root/"reader-freeze.json",result)
    else:result=evaluate(args.root,args.input,args.split)
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
