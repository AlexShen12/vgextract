"""Source-index coverage audit that does not display sealed reference answers."""
import argparse
from pathlib import Path

from .core import atomic_json, read_json
from .scene_ablation_score import validate_reference
from .scene_reference import file_hash


def validate_coverage(label,clip,index,index_path,sha):
    validate_reference(label,clip,sha,require_complete=True)
    review=label["review"]
    if (index.get("id")!=clip["id"] or index.get("source_sha256")!=sha
        or [index.get("start"),index.get("end")]!=[clip["start"],clip["end"]]
        or index.get("sampling_fps")!=16):
        raise ValueError("Assigned source index differs from the complete reserved context")
    if (review.get("source_index")!=str(Path(index_path).resolve())
        or review.get("inspected_interval")!=[clip["start"],clip["end"]]
        or review.get("sampling_fps")!=16):
        raise ValueError("Source review used a different index, cadence or context")
    if (review.get("overview_sheets_reviewed")!=len(index["sheets"])
        or review.get("native_hud_samples_reviewed")!=len(index["frames"])):
        raise ValueError("Complete HUD inventory requires review of every assigned source sample")
    if label["scope"]["physical_actions"]=="complete_observable" and review.get("native_source_samples_reviewed")!=len(index["frames"]):
        raise ValueError("Complete physical inventory requires full source-view coverage")
    return True


def audit(run_dir,references,index_root):
    run_dir,references,index_root=map(Path,(run_dir,references,index_root))
    study=read_json(run_dir/"study.json");rows=[]
    for clip in study["clips"]:
        path=references/clip["split"]/(clip["id"]+".json")
        group="development-inputs" if clip["split"]=="development" else "inputs"
        index_path=index_root/group/clip["id"]/"source-inputs.json"
        validate_coverage(read_json(path),clip,read_json(index_path),index_path,study["source"]["sha256"])
        rows.append({"clip_id":clip["id"],"reference_path":str(path.resolve()),"reference_sha256":file_hash(path),
                     "index_path":str(index_path.resolve()),"index_sha256":file_hash(index_path),"coverage_status":"passed"})
    result={"status":"passed","answers_displayed":False,"clips":rows,
            "limitation":"Checks source identity and reviewer coverage attestations; does not prove label correctness or human verification."}
    atomic_json(run_dir/"source-reference-provenance.json",result)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir",type=Path,required=True)
    p.add_argument("--references",type=Path,required=True)
    p.add_argument("--source-inputs",type=Path,required=True)
    args=p.parse_args()
    result=audit(args.run_dir,args.references,args.source_inputs)
    print("Source-index coverage audit:",result["status"],len(result["clips"]),"clips; answers not displayed")
