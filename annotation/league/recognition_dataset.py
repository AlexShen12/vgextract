"""Source-only cast-bar labeling campaign; predictions never populate references."""
import argparse
from collections import Counter
import fcntl
import json
from pathlib import Path
import re

from .core import atomic_json, digest, finite, now, read_json
from .media import file_hash

CASES = ("named_cast", "sustained_channel", "charged_ability", "recall_complete", "recall_interrupted",
         "partial_visibility", "clipped_boundary", "repeated_cast", "absent_bar", "cost_countdown",
         "resource_bar", "notice", "scoreboard_overlay", "apprehend_recall")
TARGET = {"positive_occurrences": 300, "negative_clips": 150, "champions": 12, "matches": 6,
          "layouts": 3, "held_out_positives": 50, "held_out_per_case": 5}


def record_exposure(root,source,clip):
    """A portable envelope also understood by existing source-interval exclusions."""
    path=Path(root)/"exposure"/(source["id"]+".json")
    with (Path(root)/".exposure.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        data=read_json(path) if path.exists() else {"source_sha256":source["sha256"],"clips":[]}
        entry={"id":clip["id"],"start":clip["start"],"end":clip["end"],"reason":"source_reference_preparation"}
        if entry not in data["clips"]:data["clips"].append(entry)
        atomic_json(path,data)


def initialize(root):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    path = root/"dataset.json"
    if path.exists():
        return read_json(path)
    data = {"schema_version": 1, "created_at": now(), "status": "collecting", "target": TARGET,
            "split_matches": {"development": 3, "validation": 1, "held_out": 2}, "sources": [], "clips": []}
    atomic_json(path, data)
    return data


def validate(data):
    sources = {s["id"]: s for s in data["sources"]}
    if len(sources) != len(data["sources"]):
        raise ValueError("Duplicate source ID")
    groups = {}
    for source in sources.values():
        if source["split"] not in {"development", "validation", "held_out"} or not source.get("match_group"):
            raise ValueError("Source requires preassigned match group and split")
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", source["id"]) or not re.fullmatch(r"[a-f0-9]{64}", source["sha256"]):
            raise ValueError("Invalid source ID/hash")
        for key in ("match_group", "sha256"):
            group = (key, source[key])
            if group in groups and groups[group] != source["split"]:
                raise ValueError("Underlying match or duplicate video crosses splits")
            groups[group] = source["split"]
        if source.get("previously_exposed") and source["split"] != "development":
            raise ValueError("Exposed footage is development-only")
    identifiers = set()
    for clip in data["clips"]:
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", clip["id"]) or clip["id"] in identifiers:
            raise ValueError("Invalid/duplicate clip ID")
        identifiers.add(clip["id"])
        if clip["source_id"] not in sources or not finite(clip["start"]) or not finite(clip["end"]) or not 0 <= clip["start"] < clip["end"]:
            raise ValueError("Invalid clip source/interval")
        if clip["end"]-clip["start"] > 30:
            raise ValueError("Bar labeling clips must be at most thirty seconds")
        if set(clip.get("case_tags", []))-set(CASES):
            raise ValueError("Unknown bar case tag")
    return sources


def validate_label(label, clip):
    if label.get("reviewer_kind") not in {"model", "human"} or not isinstance(label.get("reviewer"), str) or not label["reviewer"].strip():
        raise ValueError("Explicit model/human reviewer required")
    if label.get("source_only") is not True or type(label.get("verified")) is not bool:
        raise ValueError("Source-only review and explicit verification required")
    if label["verified"] and (label["reviewer_kind"] != "human" or label.get("human_attestation") is not True):
        raise ValueError("Only an explicit human source review may verify labels")
    if type(label.get("exhaustive")) is not bool:
        raise ValueError("Label must state whether the clip was exhaustively reviewed")
    seen = set()
    for event in label["events"]:
        if event["id"] in seen or not isinstance(event["id"], str):
            raise ValueError("Duplicate/invalid bar event ID")
        seen.add(event["id"])
        if not all(finite(event[k]) for k in ("start", "end")) or not clip["start"] <= event["start"] <= event["end"] <= clip["end"]:
            raise ValueError("Bar interval outside clip")
        if event["channel_kind"] not in {"ability", "recall", "unknown"} or event["visibility"] not in {"readable", "unreadable", "unknown"}:
            raise ValueError("Invalid bar kind/visibility")
        if event["owner_id"] != "selected" or not isinstance(event["raw_text"], str) or len(event["raw_text"]) > 500:
            raise ValueError("English selected-player bars only")
        for field in ("text_box", "progress_box"):
            box = event[field]
            if len(box) != 4 or not all(type(x) is int for x in box) or not 0 <= box[0] < box[2] or not 0 <= box[1] < box[3]:
                raise ValueError("Bar label requires native-coordinate regions")
        if event.get("ability_id") is not None:
            if event["channel_kind"] != "ability" or not re.fullmatch(r"[A-Za-z]+:[QWER]", event["ability_id"]):
                raise ValueError("Invalid ability ID")
            if event.get("champion") and event["ability_id"].split(":")[0] != event["champion"]:
                raise ValueError("Ability contradicts champion label")
        if event.get("terminal_status") not in {"completed", "interrupted", "ending_unknown"}:
            raise ValueError("Explicit channel ending status required")
        if any(type(event.get(k)) is not bool for k in ("left_censored", "right_censored")):
            raise ValueError("Censored boundaries must be explicit")
    if label["verified"]:
        events=label["events"];tags=set(clip.get("case_tags",[]))
        names=Counter((e.get("ability_id") or e["raw_text"].strip().casefold()) for e in events if e["channel_kind"]=="ability" and e["raw_text"].strip())
        support={"named_cast":any(e["channel_kind"]=="ability" and e["visibility"]=="readable" and e["raw_text"].strip() for e in events),
                 "recall_complete":any(e["channel_kind"]=="recall" and e["terminal_status"]=="completed" and not e["right_censored"] for e in events),
                 "recall_interrupted":any(e["channel_kind"]=="recall" and e["terminal_status"]=="interrupted" for e in events),
                 "clipped_boundary":any(e["left_censored"] or e["right_censored"] for e in events),
                 "repeated_cast":any(n>=2 for n in names.values()),
                 "charged_ability":any(e.get("bar_type")=="charge" for e in events),
                 "sustained_channel":any((e.get("bar_type")=="channel" or e["channel_kind"]=="recall") and e["end"]-e["start"]>=1 for e in events),
                 "partial_visibility":any(e["visibility"]!="readable" or e.get("partially_visible") is True for e in events),
                 "absent_bar":not events and label["exhaustive"],
                 "apprehend_recall":any(e["channel_kind"]=="ability" and e["raw_text"].strip().casefold()=="apprehend" for e in events)}
        if any(not support[tag] for tag in tags&support.keys()):
            raise ValueError("Verified case tags are not supported by the source event labels")
        documented=set()
        for evidence in label.get("case_evidence",[]):
            box=evidence.get("box",[])
            if (not finite(evidence.get("start")) or not finite(evidence.get("end"))
                    or not clip["start"]<=evidence["start"]<=evidence["end"]<=clip["end"]
                    or len(box)!=4 or not all(type(v) is int for v in box) or not 0<=box[0]<box[2] or not 0<=box[1]<box[3]
                    or not isinstance(evidence.get("source_note"),str) or not evidence["source_note"].strip()):
                raise ValueError("Case evidence needs a source interval, native region and observation note")
            documented.add(evidence["case_tag"])
        if tags-support.keys()-documented:
            raise ValueError("Verified negative-case tags require source case_evidence")


def save_label(root, clip_id, label):
    root = Path(root)
    with (root/".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = read_json(root/"dataset.json"); sources=validate(data)
        clip = next(c for c in data["clips"] if c["id"] == clip_id)
        if (root/"freeze.json").exists():
            raise ValueError("Frozen labels require a new dataset revision")
        if (root/"reader-freeze.json").exists() and sources[clip["source_id"]]["split"]=="development":
            raise ValueError("Development labels are frozen with the reader; create a new revision")
        validate_label(label, clip)
        record_exposure(root,sources[clip["source_id"]],clip)
        path = root/"labels"/(clip_id+".json")
        if path.exists():
            old = read_json(path)
            atomic_json(root/"revisions"/(clip_id+"-"+digest(old)+".json"), old)
        atomic_json(path, {**label, "clip_id": clip_id, "updated_at": now()})
    return label


def coverage(root):
    root = Path(root); data = read_json(root/"dataset.json"); sources = validate(data)
    positives = negatives = held_positives = 0
    champions, layouts, matches, held_cases = set(), set(), set(), Counter()
    missing, unverified, pending_held_out = [], [], []
    split_groups = {split: set() for split in data["split_matches"]}
    for source in sources.values():
        split_groups[source["split"]].add(source["match_group"])
    for clip in data["clips"]:
        path = root/"labels"/(clip["id"]+".json")
        if not path.exists():
            missing.append(clip["id"]); continue
        label = read_json(path); validate_label(label, clip)
        source = sources[clip["source_id"]]
        if not label["verified"] or not label["exhaustive"]:
            unverified.append(clip["id"])
            if source["split"] == "held_out": pending_held_out.append(clip["id"])
        positives += len(label["events"])
        negatives += int(not label["events"] and label["exhaustive"])
        champions.update(e["champion"] for e in label["events"] if e.get("champion"))
        matches.add(source["match_group"]); layouts.add(source["layout_id"])
        if source["split"] == "held_out" and label["verified"] and label["exhaustive"]:
            held_positives += len(label["events"]); held_cases.update(set(clip["case_tags"]))
    counts = {"positive_occurrences": positives, "negative_clips": negatives, "champions": len(champions),
              "matches": len(matches), "layouts": len(layouts), "held_out_positives": held_positives}
    gaps = {key: max(0, value-counts[key]) for key, value in TARGET.items() if key in counts}
    cases = {case: max(0, 5-held_cases[case]) for case in CASES}
    split_gaps = {s: max(0, n-len(split_groups[s])) for s,n in data["split_matches"].items()}
    return {"counts": counts, "gaps": gaps, "held_out_case_gaps": cases, "split_match_gaps": split_gaps,
            "missing_labels": missing, "unverified_clips": unverified, "pending_held_out": pending_held_out,
            "release_ready": not any(gaps.values()) and not any(cases.values()) and not any(split_gaps.values()) and not missing and not pending_held_out}


def add_source(root, source, clips=()):
    root = Path(root)
    with (root/".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (root/"freeze.json").exists():
            raise ValueError("Dataset is frozen")
        if file_hash(Path(source["path"])) != source["sha256"]:
            raise ValueError("Source hash mismatch")
        data = read_json(root/"dataset.json")
        data["sources"].append(source); data["clips"].extend(clips)
        validate(data); atomic_json(root/"dataset.json", data)
    return data


def draft_label(root, clip_id, runner):
    """Source-only model proposals. Sparse sampled spans remain explicitly unverified."""
    import base64
    import io
    from .scene_reference import sample_source
    root=Path(root);data=read_json(root/"dataset.json");sources=validate(data)
    clip=next(c for c in data["clips"] if c["id"]==clip_id);source=sources[clip["source_id"]]
    if file_hash(Path(source["path"]))!=source["sha256"]:raise ValueError("Draft source hash mismatch")
    record_exposure(root,source,clip)
    fields={"frame_index":{"type":"integer"},"raw_text":{"type":"string","maxLength":300},
            "visibility":{"enum":["readable","unreadable","unknown","absent"]},
            "channel_kind":{"enum":["recall","ability","unknown"]}}
    schema={"type":"array","items":{"type":"object","additionalProperties":False,"properties":fields,"required":list(fields)},"maxItems":8}
    frames=list(sample_source(source["path"],clip["start"],clip["end"],8));observations=[]
    raw_root=root/"draft-requests"/clip_id
    import hashlib
    atomic_json(raw_root/"input-audit.json",{"source_sha256":source["sha256"],"clip_id":clip_id,"sampling_fps":8,
        "frames":[{"frame_index":i,"source_pts":pts,"rgb_sha256":hashlib.sha256(image.tobytes()).hexdigest()} for i,(pts,image) in enumerate(frames)]})
    for first in range(0,len(frames),8):
        batch=frames[first:first+8]
        content=[{"type":"text","text":"Read only the selected-player cast/recall bar in each source frame. Copy visible English text verbatim. A cooldown number or resource bar is not a cast bar. Do not infer names, champions, slots, targets or outcomes. Return one reading per supplied frame_index. Regions: "+json.dumps({k:clip[k] for k in ("text_box","progress_box")})}]
        for offset,(pts,image) in enumerate(batch,first):
            buf=io.BytesIO();image.save(buf,format="PNG")
            content.extend([{"type":"text","text":f"frame_index={offset}, source_pts={pts:.9f}"},
                {"type":"image_url","image_url":{"url":"data:image/png;base64,"+base64.b64encode(buf.getvalue()).decode()}}])
        response=runner.complete([{"role":"user","content":content}],schema=schema,max_tokens=2048,timeout=600)
        atomic_json(raw_root/f"{first:06d}.json",response)
        rows=json.loads(response["text"])
        if not isinstance(rows,list) or {r["frame_index"] for r in rows}!=set(range(first,first+len(batch))) or len(rows)!=len(batch):
            raise ValueError("Draft model omitted or invented source frames")
        for row in rows:
            if row["visibility"] not in fields["visibility"]["enum"] or row["channel_kind"] not in fields["channel_kind"]["enum"]:
                raise ValueError("Invalid draft bar reading")
        observations.extend(rows)
    events=[]
    for row in sorted(observations,key=lambda r:r["frame_index"]):
        if row["visibility"]=="absent":continue
        t=frames[row["frame_index"]][0]
        previous=events[-1] if events else None
        if previous and previous["raw_text"]==row["raw_text"] and previous["channel_kind"]==row["channel_kind"] and t-previous["end"]<=.13:
            previous["end"]=t;continue
        events.append({"id":f"draft-{row['frame_index']}","start":t,"end":t,"raw_text":row["raw_text"],
            "visibility":row["visibility"],"channel_kind":row["channel_kind"],"owner_id":"selected",
            "champion":None,"ability_id":None,"terminal_status":"ending_unknown","left_censored":t==frames[0][0],
            "right_censored":False,"text_box":clip["text_box"],"progress_box":clip["progress_box"]})
    if events:events[-1]["right_censored"]=events[-1]["end"]==frames[-1][0]
    label={"reviewer":"local source-only model","reviewer_kind":"model","source_only":True,"verified":False,
           "human_attestation":False,"exhaustive":False,"events":events,"sampling_fps":8,
           "boundary_status":"sampled draft; native-frame human review pending","runtime":runner.identity}
    save_label(root,clip_id,label);return label


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("init", "coverage", "add-source", "import-label", "freeze", "draft"))
    p.add_argument("--root", type=Path, required=True); p.add_argument("--input", type=Path); p.add_argument("--clip")
    p.add_argument("--vl-python");p.add_argument("--cache",type=Path,default=Path("data/league_assets"))
    args = p.parse_args()
    if args.command == "init": result = initialize(args.root)
    elif args.command == "coverage": result = coverage(args.root)
    elif args.command == "add-source":
        record = read_json(args.input); result = add_source(args.root, record["source"], record.get("clips", []))
    elif args.command == "import-label": result = save_label(args.root, args.clip, read_json(args.input))
    elif args.command == "draft":
        from .core import Store
        from .local_models import LocalVLRunner
        runner=LocalVLRunner(Store(args.root/"draft-runtime"),args.vl_python,args.cache,image_limit=8)
        try:result=draft_label(args.root,args.clip,runner)
        finally:runner.close()
    else:
        result = coverage(args.root)
        if not result["release_ready"]:
            raise ValueError("Dataset has collection or human review gaps")
        labels = {p.name: read_json(p) for p in sorted((args.root/"labels").glob("*.json"))}
        result = {"dataset_hash": digest(read_json(args.root/"dataset.json")), "labels_hash": digest(labels), "created_at": now()}
        atomic_json(args.root/"freeze.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
