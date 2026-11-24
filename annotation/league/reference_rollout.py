"""Paired catalog comparison and gated fresh-run profile promotion."""
import argparse
import json
from pathlib import Path

from .core import Store, atomic_json, digest, read_json
from .intake import image_hash
from .intake_profile import load_profile, layout_matches
from .inventory_reader import CARDINAL_SHIFTS, InventoryReader
from .inventory_references import reference_root, load_inventory_references
from .media import file_hash
from .recognition_metrics import identity_metrics


def compare(cache, cases_path, output, old_profile="midbeast-1080p-inventory-v1", new_profile="midbeast-1080p-inventory-v2"):
    from PIL import Image
    from .hybrid_intake import _panel_state
    data=read_json(Path(cases_path));output=Path(output);results={};signatures={}
    cases=data["cases"]
    if not cases or len({c["id"] for c in cases})!=len(cases):raise ValueError("Unique comparison cases required")
    for arm,profile_name in (("old",old_profile),("new",new_profile)):
        profile=load_profile(profile_name);root=reference_root(cache,profile["inventory"]["reference_version"])
        store=Store(output/arm);store.root.mkdir(parents=True,exist_ok=True)
        reader=InventoryReader(store,root,load_inventory_references(root,profile["inventory"]["reference_version"]),profile["inventory"])
        fields={f["id"]:f for f in profile["fields"] if f.get("backend")=="inventory"}
        labels={};predictions={}
        for case in cases:
            field=fields[case["field_id"]];path=Path(case["image"])
            if file_hash(path)!=case["sha256"]:raise ValueError("Comparison frame changed")
            with Image.open(path) as image:
                valid,_=layout_matches(image,profile)
                panel=_panel_state(image,profile,field,valid)
                crop=image.crop(field["box"])
                masks=data.get("masks",[])
                def clear(box):return not any(a<box[2] and c>box[0] and b<box[3] and d>box[1] for a,b,c,d in masks)
                result=reader.read(crop,field,image_hash(crop))[0] if valid and panel=="present" and clear(field["box"]) else {"acceptance":"unknown","value":None}
                def shifted():
                    for dx,dy in CARDINAL_SHIFTS:
                        a,b,c,d=field["box"];box=[a+dx,b+dy,c+dx,d+dy]
                        if 0<=box[0]<box[2]<=image.width and 0<=box[1]<box[3]<=image.height and clear(box):
                            shifted_crop=image.crop(box);yield (dx,dy),shifted_crop,image_hash(shifted_crop)
                if valid and panel=="present" and clear(field["box"]):result=reader.retry_alignment(result,field,shifted())
            key=field["route"]+"/"+field["kind"]
            labels.setdefault(key,[]).append({"id":case["id"],"champion":case.get("value"),
                "visibility":"readable" if case["availability"]=="visible" else "unknown"})
            predictions.setdefault(key,[]).append({"id":case["id"],"champion":result["value"] if result["acceptance"]=="accepted" else None})
        results[arm]={key:identity_metrics(rows,predictions[key]) for key,rows in labels.items()}
        signatures[arm]=reader.signature
    required={route+"/"+kind for route in ("selected_hud","scoreboard") for kind in ("champion","item")}
    human=all(c.get("verified") is True and c.get("reviewer_kind")=="human" and c.get("source_only") is True for c in cases)
    passed=human and set(results["new"])==required and all(r["passed"] for r in results["new"].values())
    report={"old_profile":old_profile,"new_profile":new_profile,"new_profile_hash":digest(load_profile(new_profile)),
            "reader_signatures":signatures,"metrics":results,
            "comparison_cases_hash":digest(data),"human_verified":human,"passed":passed,"gameplay_patch":None}
    atomic_json(output/"comparison.json",report);return report


def promote(config_path,comparison_path,intake_gate_path,pilot_gate_path):
    comparison=read_json(Path(comparison_path));gate=read_json(Path(intake_gate_path));pilot=read_json(Path(pilot_gate_path))
    required={route+"/"+kind for route in ("selected_hud","scoreboard") for kind in ("champion","item")}
    metrics=comparison.get("metrics",{}).get("new",{})
    if (not comparison.get("passed") or not comparison.get("human_verified") or set(metrics)!=required
            or not all(row.get("passed") for row in metrics.values()) or gate.get("passed") is not True
            or gate.get("draft_labels") is not False or gate.get("profile_hash")!=comparison["new_profile_hash"]
            or gate.get("held_out",{}).get("passed") is not True or not gate.get("runtime",{}).get("inventory")
            or pilot.get("passed") is not True or pilot.get("runtime_passed") is not True
            or pilot.get("expected_frames",0)<300 or pilot.get("extraction_key")!=gate.get("extraction_key")):
        raise ValueError("Verified paired comparison, held-out intake and five-minute runtime gates are required")
    profile=load_profile(comparison["new_profile"])
    if digest(profile)!=comparison["new_profile_hash"] or not profile.get("thresholds_frozen"):
        raise ValueError("Frozen profile changed since the paired comparison")
    config=read_json(Path(config_path))
    cache=config.get("asset_cache","data/league_assets")
    catalog_root=reference_root(cache,profile["inventory"]["reference_version"])
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        reader=InventoryReader(Store(Path(directory)),catalog_root,
            load_inventory_references(catalog_root,profile["inventory"]["reference_version"]),profile["inventory"])
        if reader.signature!=comparison["reader_signatures"]["new"] or gate["runtime"]["inventory"].get("catalog_id")!=reader.catalog_id:
            raise ValueError("Reference catalog/reader changed since validation")
        if gate["runtime"]["inventory"].get("skin_catalog_id")!=reader.skin_signature:
            raise ValueError("Skin reference catalog changed since validation")
    from .intake_profile import PROFILE_DIR
    name="midbeast-1080p-inventory-v2-"+digest(profile)[:12]
    atomic_json(PROFILE_DIR/(name+".json"),profile)
    config["layout_profile"]=name
    atomic_json(Path(config_path),config)
    return {"default_profile":profile["id"],"saved_run_configs_modified":False}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("command",choices=("compare","promote"))
    p.add_argument("--cache",type=Path,default=Path("data/league_assets"));p.add_argument("--cases",type=Path)
    p.add_argument("--output",type=Path);p.add_argument("--comparison",type=Path);p.add_argument("--intake-gate",type=Path)
    p.add_argument("--pilot-gate",type=Path)
    p.add_argument("--old-profile",default="midbeast-1080p-inventory-v1")
    p.add_argument("--new-profile",default="midbeast-1080p-inventory-v2")
    p.add_argument("--config",type=Path,default=Path(__file__).with_name("midbeast.json"));args=p.parse_args()
    result=compare(args.cache,args.cases,args.output,args.old_profile,args.new_profile) if args.command=="compare" else promote(args.config,args.comparison,args.intake_gate,args.pilot_gate)
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
