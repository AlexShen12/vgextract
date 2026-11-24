"""Asynchronous Longleaf caption jobs and portable, provenance-preserving results."""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
from pathlib import Path
import tarfile

from .caption_event_benchmark import aggregate, experiment, prediction_fingerprint, score, validate
from .caption_occurrences import EVENT_POLICY, RUBRIC, RUBRIC_VERSION
from .core import Config, Store, atomic_json, digest, now, read_json, read_rows
from .media import file_hash


def isolated_store(source_run, work_root, arm):
    original=read_json(Path(source_run)/"manifest.json")
    root=Path(work_root)/arm
    root.mkdir(parents=True,exist_ok=True)
    target=root/"manifest.json"
    if target.exists() and read_json(target)["source"]!=original["source"]:
        raise ValueError("Experiment run already belongs to a different source")
    if not target.exists():atomic_json(target,original)
    return Store(root)


def collect(report_path, labels=None):
    """Never invokes perception or rewrites the immutable execution report."""
    report_path=Path(report_path)
    original=read_json(report_path)
    data=read_json(Path(labels)/"events.json" if labels else report_path.parent/"reference-snapshot.json")
    snapshot=read_json(report_path.parent/"reference-snapshot.json")
    if digest(data)!=digest(snapshot):
        raise ValueError("References differ from executed source-only snapshot; use a new benchmark revision")
    # Find the isolated run by the standard experiments/caption-events layout.
    store=Store(report_path.parents[3])
    required=("development",) if original["draft"] else ("held_out",)
    validate(store,data,splits=required)
    expected={c["id"] for c in data["clips"] if c["split"] in required}
    if original.get("preflight") or set(original["clips"])!=expected:
        raise ValueError("Complete split predictions are required for benchmark collection")
    if digest([c for c in data["clips"] if c["split"]=="development"]) != digest(original["signature"]["development"]):
        raise ValueError("Development reference revision differs from executed configuration")
    reviewed=copy.deepcopy(original)
    metrics=[];clips=[]
    for c in data["clips"]:
        if c["id"] not in original["clips"]:continue
        record=original["clips"][c["id"]];root=Path(record["root"])
        if read_json(root/"caption_coverage.json")["key"] != record["generation_key"]:
            raise ValueError("Predictions changed after execution")
        audit=read_json(root/'input-audit.json')
        if audit.get('source_sha256')!=data['source_sha256'] or audit.get('interval')!=[c['start'],c['end']]:
            raise ValueError('Prediction source interval does not match reference neighborhood')
        if record.get('prediction_fingerprint')!=prediction_fingerprint(root):
            raise ValueError('Prediction artifacts changed after execution')
        if list((root/"reviews").glob("*.json")):
            raise ValueError("Corrected predictions cannot be benchmark acceptance")
        metric=score(c,root);reviewed["clips"][c["id"]]["metrics"]=metric
        metrics.append(metric);clips.append(c)
    if not metrics:raise ValueError("No saved predictions to collect")
    reviewed.update(quality_metrics=aggregate(metrics,clips),assessed_at=now(),execution_report=str(report_path),
                    scoring_implementation=digest(Path(__file__).with_name("caption_event_benchmark.py").read_text()))
    reviewed["status"]="validation_passed" if reviewed["quality_metrics"]["passed"] else "validation_pending_or_failed"
    destination=report_path.with_name("assessed-report.json");reviewed["report"]=str(destination)
    atomic_json(destination,reviewed)
    return reviewed


def bundle(report_path, output, include_images=False):
    report_path=Path(report_path).resolve();report=read_json(report_path)
    store_root=report_path.parents[3]
    files={}
    def add(path):
        if path.is_file():files[str(Path("run")/path.relative_to(store_root))]=path
    for p in (store_root/"manifest.json",report_path,report_path.parent/"reference-snapshot.json",report_path.parent/"reference-method.json",report_path.parent/"assessed-report.json"):
        add(p)
    snapshot=report_path.parent/"reference-snapshot.json"
    if snapshot.exists():files['run/benchmarks/caption-events/v1/events.json']=snapshot
    for p in (store_root/'workers').glob('*.log'):add(p)
    for item in report["clips"].values():
        root=Path(item["root"])
        for p in root.rglob("*"):
            if p.suffix in {".json",".jsonl"}:add(p)
        if include_images:
            ids={r["id"] for c in read_rows(root/"captions.jsonl") for r in c["evidence_refs"] if r["table"]=="frames"}
            ids|={f for o in read_rows(root/"scene_occurrences.jsonl") if o.get("record_type")!="unresolved_candidate" for frames in o["phase_evidence"].values() for f in frames}
            audit=read_json(root/"input-audit.json")
            for f in audit["frames"]:
                if f["id"] in ids:
                    add(Path(f["native_path"]))
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    source=Path(report.get("implementation_root",Path(__file__).parent))
    if not source.is_dir():raise ValueError("Executed implementation snapshot is unavailable; do not substitute current code")
    expected=report.get("signature",{}).get("implementation")
    if expected:
        from .caption_pilot import COMPONENTS
        names=COMPONENTS+["caption_event_benchmark.py","caption_review.py"]
        actual=digest({p.name:file_hash(p) for p in sorted(source.glob('*.py')) if p.name in names})
        if actual!=expected:raise ValueError("Executed implementation changed; bundle its original immutable snapshot")
    index={"schema_version":1,"source_run":str(store_root),"source":read_json(store_root/"manifest.json")["source"],
           "created_at":now(),"execution_report":str(report_path.relative_to(store_root)),
           "files":{name:file_hash(p) for name,p in sorted(files.items())},"images":"cited_native_only" if include_images else "metadata_only"}
    import io
    with tarfile.open(output,"w:gz") as arc:
        raw=(json.dumps(index,indent=2)+"\n").encode();info=tarfile.TarInfo("bundle-index.json");info.size=len(raw);arc.addfile(info,io.BytesIO(raw))
        for name,path in sorted(files.items()):arc.add(path,arcname=name,recursive=False)
        # Source snapshot makes later reproduction independent of a moving checkout.
        for p in sorted(source.rglob("*")):
            if p.is_file() and p.suffix in {".py",".json",".html",".sh",".sl"} and "__pycache__" not in p.parts:
                arc.add(p,arcname=str(Path("implementation/annotation/league")/p.relative_to(source)),recursive=False)
        for name in ("annotation/__init__.py", "ingestion/__init__.py", "ingestion/common/__init__.py", "ingestion/common/replay_state.py"):
            p=source.parents[1]/name
            if p.is_file():arc.add(p,arcname="implementation/"+name,recursive=False)
    return {"status":"results_bundled","bundle":str(output),"sha256":file_hash(output),"files":len(files),"video_exported":False}


def import_bundle(archive, run_dir, source=None):
    """Project absolute remote paths without regenerating or changing claim facts."""
    destination=Path(run_dir).resolve();destination.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive,"r:gz") as arc:
        index=json.load(arc.extractfile("bundle-index.json"))
        local_source=Path(source).resolve() if source else None
        manifest=destination/"manifest.json"
        if manifest.exists():local_source=Path(read_json(manifest)["source"]["path"])
        if local_source is None or file_hash(local_source)!=index["source"]["sha256"]:
            raise ValueError("A local copy of the exact source MP4 is required")
        replacements={index["source_run"]:str(destination),index["source"]["path"]:str(local_source)}
        def project(value):
            if isinstance(value,str):
                for old,new in replacements.items():value=value.replace(old,new)
                return value
            if isinstance(value,list):return [project(v) for v in value]
            if isinstance(value,dict):return {k:project(v) for k,v in value.items()}
            return value
        validated=[]
        for member in arc.getmembers():
            if not member.name.startswith("run/"):continue
            relative=Path(member.name).relative_to("run")
            if member.isdir():continue
            if not member.isfile() or ".." in relative.parts or relative.is_absolute():raise ValueError("Unsafe bundle member")
            expected=index["files"].get(member.name)
            payload=arc.extractfile(member).read()
            import hashlib
            if not expected or hashlib.sha256(payload).hexdigest()!=expected:raise ValueError("Bundle integrity mismatch")
            target=destination/relative
            if not target.resolve().is_relative_to(destination):raise ValueError("Bundle path escapes destination")
            if target.exists() and relative.name!="manifest.json":raise ValueError("Refusing to overwrite existing pilot evidence")
            validated.append((target,payload))
        for target,payload in validated:
            if target==manifest and target.exists():continue
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.suffix==".json":atomic_json(target,project(json.loads(payload)))
            elif target.suffix==".jsonl":target.write_text("".join(json.dumps(project(json.loads(line)))+"\n" for line in payload.decode().splitlines() if line))
            else:target.write_bytes(payload)
        atomic_json(destination/"caption-bundle-imports"/(file_hash(Path(archive))+".json"),
                    {"bundle_index":index,"archive_sha256":file_hash(Path(archive)),"claims_modified":False,"source_verified":True})
    return {"status":"results_imported","run_dir":str(destination),"source_verified":True}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("preflight","development","freeze","held-out","collect","bundle","import"))
    parser.add_argument("--arm",choices=("baseline","event-v2"),default="event-v2")
    parser.add_argument("--source-run",type=Path)
    parser.add_argument("--work-root",type=Path)
    parser.add_argument("--labels",type=Path)
    parser.add_argument("--report",type=Path)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--archive",type=Path)
    parser.add_argument("--run-dir",type=Path)
    parser.add_argument("--source",type=Path)
    parser.add_argument("--include-images",action="store_true",help="Bundle original cited PNGs as well as compact metadata")
    parser.add_argument("--icon-references",type=Path)
    args=parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
    if args.mode=="bundle":result=bundle(args.report,args.output,include_images=args.include_images)
    elif args.mode=="import":result=import_bundle(args.archive,args.run_dir,args.source)
    elif args.mode=="collect":result=collect(args.report,args.labels)
    else:
        if not all((args.source_run,args.work_root,args.labels)):parser.error("source-run, work-root and labels are required")
        store=isolated_store(args.source_run,args.work_root,args.arm);store.install_signals()
        config=Config.load(Path(__file__).with_name("midbeast.json"))
        config.asset_cache=os.environ.get("LEAGUE_ASSET_CACHE",config.asset_cache)
        config.vl_python=os.environ.get("LEAGUE_VL_PYTHON",config.vl_python)
        from .__main__ import batch_writer
        with batch_writer(store,None):
            runtime_recipe=None
            if args.mode=="development":
                references=read_json(args.labels/'events.json')
                validate(store,references,splits=('development',))
                from .caption_pilot import COMPONENTS
                from .core import implementation_hash
                from .local_models import LOCK
                from .action_v3 import load_profile
                from .intake_profile import load_profile as anchor_profile
                from .action_icons import AbilityIdentity
                implementation=implementation_hash(COMPONENTS+["caption_event_benchmark.py","caption_review.py"])
                profile=load_profile()
                reference=args.icon_references or Path(config.asset_cache)/'manifest.json'
                icons=AbilityIdentity(reference).signature if reference.exists() else None
                reports=[read_json(p) for p in sorted(store.path("experiments/caption-events").glob("*/report.json"),key=lambda p:p.stat().st_mtime,reverse=True)]
                qualified=[r for r in reports if r.get("preflight") is True and r.get("status")=="development_complete"
                    and r.get("signature",{}).get("implementation")==implementation
                    and r["signature"].get("model")==read_json(LOCK)["models"]["intake"]
                    and r["signature"].get("arm")==args.arm
                    and r["signature"].get("source")==references['source_sha256']
                    and r["signature"].get("policy")==EVENT_POLICY and r["signature"].get("rubric")==RUBRIC
                    and r["signature"].get("rubric_version")==RUBRIC_VERSION
                    and r["signature"].get("profile")==profile and r["signature"].get("masks")==config.masks
                    and r["signature"].get("anchors")==anchor_profile(profile['anchor_profile'])
                    and r["signature"].get("icons")==icons and r.get('runtime_recipe')
                    and all(c.get("resources",{}).get("requests",0)>0 and c["resources"].get("invalid_responses",0)==0
                            and c["resources"].get("truncated_requests",0)==0 for c in r.get("clips",{}).values()) and r.get("clips")]
                if not qualified:raise ValueError("Run a successful matching preflight and inspect its resources before development")
                runtime_recipe=qualified[0]['runtime_recipe']
            result=experiment(store,config,labels=args.labels,arm=args.arm,draft=args.mode in {"development","preflight"},
                freeze=args.mode=="freeze",preflight=args.mode=="preflight",infer=args.mode!="freeze",icon_references=args.icon_references,
                runtime_recipe=runtime_recipe)
    print(json.dumps(result,indent=2))
    return 0


if __name__=="__main__":raise SystemExit(main())
