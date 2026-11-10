"""Reserve a temporal holdout using exposure metadata, without opening footage."""
from pathlib import Path
import argparse
import json

from .core import atomic_json, digest, finite, read_json
from .intake_profile import load_profile
from .scene_reference import file_hash, materialize


def merged_intervals(rows, lower, upper, guard=8):
    if guard < 8:
        raise ValueError("Keep the existing eight-second split guard")
    result = []
    for a, b in sorted((max(lower, r[0]-guard), min(upper, r[1]+guard)) for r in rows):
        if not all(finite(x) for x in (a,b)) or a > b:
            raise ValueError("Invalid exposure interval")
        if result and a <= result[-1][1]:
            result[-1][1] = max(result[-1][1], b)
        else:
            result.append([a,b])
    return result


def exposure_records(root):
    root = Path(root)
    source = read_json(root/"manifest.json")["source"]
    sha = source["sha256"]
    records, snapshots = [], []
    paths = set(root.joinpath("benchmarks").rglob("*.json"))
    for name in ("input-audit.json", "study-snapshot.json", "source-inputs.json", "manifest.json"):
        paths.update(root.joinpath("experiments").rglob(name))
    # Imported inference metadata participates in the same source-identity check.
    for directory in (root.parent/"artifacts", root.parents[1]/"artifacts", root.parents[1]/"league_caption_pilot_validation"):
        if directory.exists():
            paths.update(directory.rglob("input-audit.json"))
    def add(a,b,path,reason):
        if finite(a) and finite(b) and 0 <= a <= b <= source["duration"]:
            records.append({"source_sha256": sha, "start": a, "end": b,
                            "provenance": str(path), "reason": reason})
    for path in sorted(paths):
        data = read_json(path)
        if not isinstance(data,dict):
            continue
        nested = data.get("source")
        recorded_sha = data.get("source_sha256") or (nested.get("sha256") if isinstance(nested,dict) else None)
        same = recorded_sha == sha
        # A run-owned input audit without a redundant hash still belongs to its run.
        if recorded_sha is None and path.name == "input-audit.json" and path.is_relative_to(root/"experiments"):
            same = True
        before = len(records)
        if same:
            for key in ("interval", "context_interval"):
                value = data.get(key)
                if isinstance(value,list) and len(value)==2:
                    add(*value,path,key)
            if path.name == "source-inputs.json":
                add(data.get("start"),data.get("end"),path,"prepared_source_review")
            for clip in data.get("clips",[]):
                if isinstance(clip,dict):
                    add(clip.get("start"),clip.get("end"),path,"prior_reference_or_experiment_clip")
            for value in data.get("excluded_intervals",[]):
                if isinstance(value,list) and len(value)==2:
                    add(*value,path,"historical_exclusion_snapshot")
        for row in data.get("entries",[]):
            if isinstance(row,dict) and row.get("source_sha256")==sha:
                add(row.get("start"),row.get("end"),path,"explicit_exposure_ledger")
        if len(records)>before:
            snapshots.append({"path":str(path),"sha256":file_hash(path)})
    for name in ("previews.jsonl","intake_previews.jsonl"):
        path = root/name
        if path.exists():
            for line in path.read_text().splitlines():
                row=json.loads(line)
                add(row.get("time"),row.get("time"),path,"prepared_preview")
            snapshots.append({"path":str(path),"sha256":file_hash(path)})
    profile=load_profile("midbeast-1080p-v1")
    for time in profile.get("calibration_frames",[]):
        add(time,time,"midbeast-1080p-v1","source_calibration")
    return {"source":source,"records":records,"metadata_snapshots":snapshots,
            "calibration_profile_sha256":digest(profile),"guard_seconds":8}


def reserve(snapshot, count=5):
    source=snapshot["source"]
    lower,upper=0,source["duration"]
    blocked=merged_intervals([[r["start"],r["end"]] for r in snapshot["records"]],lower,upper,8)
    gaps=[];cursor=lower
    for a,b in blocked+[[upper,upper]]:
        if a-cursor >= 8.5:
            gaps.append([cursor+.25,a-.25])
        cursor=max(cursor,b)
    if len(gaps)<count:
        raise ValueError(f"Only {len(gaps)} fresh windows remain; do not relax exposure guards")
    chosen=sorted([[a,min(b,a+24)] for a,b in sorted(gaps,key=lambda g: (-(g[1]-g[0]),g[0]))[:count]])
    clips=[{"id":f"holdout-{i:02d}","split":"held_out","start":a,"end":b,
            "reference_status":"unlabeled","repeat_group":None,
            "context_class":"contextual" if b-a>=12 else "short_negative_candidate"}
           for i,(a,b) in enumerate(chosen)]
    reservation={"schema_version":1,"source":source,"evaluation_scope":"within_recording_temporal_holdout",
                 "cross_match_generalization":False,"guard_seconds":8,
                 "exposure_sha256":digest(snapshot),"clips":clips,
                 "known_repeat_policy":"Known reference/experiment clips and their full contexts are excluded; new repeat/continuity ambiguities require independent source review."}
    reservation["reservation_sha256"]=digest(reservation)
    return reservation


def validate_reservation(reservation,snapshot):
    try:
        expected=reserve(snapshot,len(reservation["clips"]))
    except ValueError as exc:
        raise ValueError("Reservation exposure history changed; create a new split revision") from exc
    if reservation!=expected:
        raise ValueError("Reservation or exposure history changed; create a new split revision")
    return True


def prepare_references(reservation_path, source, output):
    reservation_path, output = Path(reservation_path), Path(output)
    reservation=read_json(reservation_path)
    validate_reservation(reservation,read_json(reservation_path.with_name("exposure-snapshot.json")))
    if file_hash(source)!=reservation["source"]["sha256"]:
        raise ValueError("Reference source identity changed")
    rows=[]
    for clip in reservation["clips"]:
        folder=output/clip["id"]
        path=folder/"source-inputs.json"
        if path.exists():
            data=read_json(path)
            if (data["start"],data["end"],data["source_sha256"])!=(clip["start"],clip["end"],reservation["source"]["sha256"]):
                raise ValueError("Reference input revision changed")
        else:
            data={"id":clip["id"],"split":"held_out",**materialize(source,reservation["source"]["sha256"],
                clip["start"],clip["end"],16,folder/"inputs")}
            atomic_json(path,data)
        rows.append({"id":clip["id"],"source_inputs":str(path.resolve()),"frames":len(data["frames"])})
        print(clip["id"],len(data["frames"]),"source-reference frames",flush=True)
    atomic_json(output/"reference-input-index.json",{"source":reservation["source"],"clips":rows,
        "reservation_sha256":reservation["reservation_sha256"],"recognition_inference":False,
        "purpose":"Blinded source-reference review only; these inputs are not a developer tuning set."})
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--prepare-reference",type=Path)
    parser.add_argument("--source",type=Path)
    args=parser.parse_args()
    if args.prepare_reference:
        if not args.source:
            raise ValueError("An exact source path is required")
        prepare_references(args.prepare_reference,args.source,args.output)
        return
    if not args.run_dir:
        raise ValueError("Source run is required for exposure collection")
    if args.output.exists():
        raise ValueError("Refusing to overwrite a holdout reservation")
    snapshot=exposure_records(args.run_dir)
    reservation=reserve(snapshot)
    atomic_json(args.output/"exposure-snapshot.json",snapshot)
    atomic_json(args.output/"reservation.json",reservation)
    print(json.dumps({"reservation_sha256":reservation["reservation_sha256"],"clips":reservation["clips"]},indent=2))


if __name__=="__main__":
    main()
