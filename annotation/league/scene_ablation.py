"""Matched-input scene/HUD development pilot; held-out inference requires a freeze."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

from .action_body import bind_frames
from .action_hud_replay import associate_scene_casts, cooldown_candidates, hud_activation_actions
from .action_pipeline import packets, prepare_inputs, scan_hud
from .action_state import POLICY, deduplicate_scene
from .action_v3 import SceneReader, cache_runtime, load_profile
from .core import Config, Store, atomic_json, digest, read_json, read_rows, uid, write_rows
from .local_models import LOCK, LocalVLRunner
from .scene_reference import file_hash
from .scene_split import validate_reservation

ARMS=("scene-only","hud-only","hud-grounded")
SOURCE_BOX=[0,40,1920,930]
EXTRA_MASKS=[[0,40,450,190],[1611,730,1920,1080]]
FILES=("scene_ablation.py","scene_split.py","scene_ablation_score.py","action_body.py",
       "action_pipeline.py","action_state.py","action_hud_replay.py","action_v3.py","local_models.py",
       "core.py","numeric_reader.py","readers.py","media.py","playback.py","intake.py",
       "intake_profile.py","action_events.py","action_icons.py","models.local.json",
       "profiles/midbeast-1080p-v1.json")


def recipe(study):
    directory=Path(__file__).parent
    return {"version":1,"source_sha256":study["source"]["sha256"],
        "reservation_sha256":study["reservation_sha256"],"scene_fps":4,"hud_fps":16,
        "clips":study["clips"],
        "world_box":SOURCE_BOX,"masks":study["masks"]+EXTRA_MASKS,"profile":study["profile"],
        "model":read_json(LOCK)["models"]["intake"],"max_tokens":4096,"seed":17,
        "scene_protocol":"Existing SceneReader; identical 4 FPS packets and anonymous actor metadata. Grounded packets additionally receive same-frame HUD and local HUD candidates.",
        "timing_policy":"Physical intervals scored separately from cooldown-onset activations; scene activation hypotheses use the first cited cast frame. Cooldown timing is an operational visual target, not button-press ground truth.",
        "implementation":{name:file_hash(directory/name) for name in FILES}}


def render_view(image, profile, masks, include_hud):
    from PIL import Image,ImageDraw
    masked=image.convert("RGB").copy();draw=ImageDraw.Draw(masked)
    for box in list(masks)+EXTRA_MASKS:
        draw.rectangle(box,fill="black")
    world=masked.crop(SOURCE_BOX).resize((960,445),Image.Resampling.LANCZOS)
    result=Image.new("RGB",(960,635),"black")
    result.paste(world,(0,20))
    if include_hud:
        result.paste(image.crop(profile["hud_box"]),(0,485))
    return result


def primary_actions(candidates, scene=(), bindings=()):
    links=associate_scene_casts(scene,candidates,bindings) if scene else []
    support=[]
    indexed={row["id"]:row for row in scene}
    for link in links:
        if link["status"] in {"agrees","slot_suggested"}:
            support.append({**indexed[link["scene_id"]],"hud_candidate_id":link["linked_candidate"]["id"]})
    actions=hud_activation_actions(candidates,[],support)
    return [{**row,"kind":"ability_activation","evidence_origin":"hud",
             "physical_action_established":bool(row["scene_ids"]),"target":None,"outcome":None}
            for row in actions],links


def validate_prepared(prepared,pinned,include_scene=True):
    if prepared["recipe_sha256"]!=digest(pinned) or file_hash(prepared["hud_candidates"])!=prepared["hud_sha256"]:
        raise ValueError("Prepared source evidence changed")
    for views in prepared["views"].values() if include_scene else ():
        if any(file_hash(f["input_path"])!=f["input_sha256"] for f in views):
            raise ValueError("Prepared scene image changed")


def init_study(root,metadata,source):
    root,metadata=Path(root),Path(metadata)
    reservation=read_json(metadata/"reservation.json")
    validate_reservation(reservation,read_json(metadata/"exposure-snapshot.json"))
    if file_hash(source)!=reservation["source"]["sha256"]:
        raise ValueError("Comparison source changed")
    config=Config.load(Path(__file__).with_name("midbeast.json"))
    profile=load_profile();profile["scene_box"]=SOURCE_BOX
    study={"schema_version":1,"source":{**reservation["source"],"path":str(Path(source).resolve())},
        "reservation_sha256":reservation["reservation_sha256"],"masks":config.masks,"profile":profile,
        "evaluation_scope":"within_recording_five_window_pilot",
        "clips":[{k:c[k] for k in ("id","start","end","split")} for c in
                 read_json(metadata/"development-candidates.json")["clips"]+reservation["clips"]]}
    path=root/"study.json"
    if path.exists() and read_json(path)!=study:
        raise ValueError("Study changed; use a new run revision")
    atomic_json(path,study)
    atomic_json(root/"manifest.json",{"source":study["source"],"interval":[0,study["source"]["duration"]]})
    atomic_json(root/"config.json",config.to_dict())
    return study


def prepare(root,metadata,source):
    from PIL import Image
    root=Path(root);study=init_study(root,metadata,source)
    store=Store(root);config=Config.load(root/"config.json")
    pinned=recipe(study);reports=[]
    for clip in study["clips"]:
        folder=root/"prepared"/clip["id"]
        path=folder/"prepared.json"
        if path.exists():
            prior=read_json(path)
            validate_prepared(prior,pinned)
            reports.append(prior);continue
        audit=prepare_inputs(store,config,clip["start"],clip["end"],study["profile"],
            {**POLICY,"scene_fps":4,"hud_fps":16},
            cache_identity={"pilot":digest(pinned),"clip_id":clip["id"]})
        body,bindings=bind_frames(audit["frames"],[],study["profile"],store.check_stop)
        hud_profile=copy.deepcopy(study["profile"])
        for owner in hud_profile["owners"]:
            owner["slots"]=[slot for slot in owner["slots"] if slot["kind"]=="ability" and slot["id"] in "QWER"]
        readings,_=scan_hud(store,{**audit,"profile":hud_profile},mode="core")
        frames={f["id"]:f for f in audit["frames"]}
        for row in readings:
            row["layout_valid"]=frames[row["frame_id"]]["layout_valid"]
        candidates=cooldown_candidates(readings,study["profile"]["thresholds"])
        folder.mkdir(parents=True,exist_ok=True)
        for name,rows in (("body",body),("bindings",bindings),("hud_readings",readings),("hud_candidates",candidates)):
            write_rows(folder/(name+".jsonl"),rows)
        views={arm:[] for arm in ("scene-only","hud-grounded")}
        for frame in audit["frames"]:
            if not frame["scene_sample"] or not frame["layout_valid"]:
                continue
            with Image.open(frame["native_path"]) as image:
                for arm in views:
                    target=folder/"views"/arm/(frame["id"]+".png")
                    target.parent.mkdir(parents=True,exist_ok=True)
                    render_view(image,study["profile"],study["masks"],arm=="hud-grounded").save(target)
                    views[arm].append({**frame,"input_path":str(target),"input_sha256":file_hash(target)})
        if [f["id"] for f in views["scene-only"]]!=[f["id"] for f in views["hud-grounded"]]:
            raise ValueError("Scene arms do not share exact timestamps")
        report={"clip":clip,"recipe_sha256":digest(pinned),"audit":str(Path(audit["root"])/"input-audit.json"),
            "body":str(folder/"body.jsonl"),"bindings":str(folder/"bindings.jsonl"),
            "hud_candidates":str(folder/"hud_candidates.jsonl"),"hud_sha256":file_hash(folder/"hud_candidates.jsonl"),
            "views":views,"scene_frames":len(views["scene-only"]),"hud_frames":len(audit["frames"])}
        atomic_json(path,report);reports.append(report)
        print(clip["id"],report["scene_frames"],"matched scene frames prepared",flush=True)
    atomic_json(root/"preparation-report.json",{"recipe":pinned,"clips":[r["clip"] for r in reports],"recognition_inference":False})
    return reports


def assert_frozen(root,study):
    path=Path(root)/"recipe-freeze.json"
    if not path.exists():
        raise ValueError("Freeze development recipe and independent references before held-out inference")
    frozen=read_json(path)
    if frozen["recipe_sha256"]!=digest(recipe(study)):
        raise ValueError("Frozen recipe changed")
    for item in frozen["reference_files"]+frozen.get("input_files",[]):
        if file_hash(item["path"])!=item["sha256"]:
            raise ValueError("Frozen source reference changed")
    return frozen


def validate_run(result,study,arm,split):
    expected={c["id"] for c in study["clips"] if c["split"]==split}
    actual=[c["clip_id"] for c in result["clips"]]
    if (result.get("status")!="completed" or result.get("recipe_sha256")!=digest(recipe(study))
        or result.get("arm")!=arm or result.get("scope")!=split
        or set(actual)!=expected or len(actual)!=len(expected)
        or any(c.get("status")!="completed" or c.get("split")!=split or c.get("arm")!=arm
               or c.get("recipe_sha256")!=result["recipe_sha256"] for c in result["clips"])):
        raise ValueError("Complete unchanged inference on every expected clip before scoring or freezing")


def run_arm(root,arm,split="development",preflight=False):
    root=Path(root);study=read_json(root/"study.json");store=Store(root);store.install_signals()
    if arm not in ARMS or split not in {"development","held_out"} or preflight and split!="development":
        raise ValueError("Invalid arm or split")
    frozen=assert_frozen(root,study) if split=="held_out" else None
    pinned=recipe(study);scope="preflight" if preflight else split
    if not preflight and split=="development" and arm!="hud-only":
        pre=read_json(root/"results/preflight"/arm/"report.json")
        if not pre.get("qualified") or pre["recipe_sha256"]!=digest(pinned):
            raise ValueError("Qualify the unchanged GPU preflight before development inference")
    clips=[c for c in study["clips"] if c["split"]==split]
    if preflight:
        clips=[c for c in clips if c["id"] in {"negative-00","negative-17"}]
    runner=None;reports=[]
    try:
        if arm!="hud-only":
            runner=LocalVLRunner(store,os.environ["LEAGUE_VL_PYTHON"],os.environ["LEAGUE_ASSET_CACHE"],image_limit=8,
                                 log_dir=root/"workers"/scope/arm)
            if frozen and digest(cache_runtime(runner.identity))!=frozen["runtime_sha256"]:
                raise ValueError("Held-out model/runtime differs from the frozen development runs")
        for clip in clips:
            prepared=read_json(root/"prepared"/clip["id"]/"prepared.json")
            validate_prepared(prepared,pinned,include_scene=arm!="hud-only")
            folder=root/"results"/scope/arm/clip["id"];folder.mkdir(parents=True,exist_ok=True)
            start,end=clip["start"],min(clip["end"],clip["start"]+4) if preflight else clip["end"]
            candidates=[r for r in read_rows(Path(prepared["hud_candidates"])) if start<=r["start"]<=r["end"]<end]
            body=read_rows(Path(prepared["body"])) if arm!="hud-only" else []
            bindings=read_rows(Path(prepared["bindings"])) if arm!="hud-only" else []
            scene=[];reader=None
            if runner:
                frames=[f for f in prepared["views"][arm] if start<=f["time"]<end]
                if not frames:
                    raise ValueError("No eligible scene input; resolve layout/decoding before inference")
                reader=SceneReader(store,folder,study["profile"],runner,body,POLICY)
                for batch in packets(frames):
                    grounding=None
                    if arm=="hud-grounded":
                        grounding={"primary_slot_activations":[{k:r[k] for k in ("id","slot_id","start","end")} for r in candidates
                            if batch[0]["time"]-.125<=r["end"]<=batch[-1]["time"]+.125],
                            "policy":"HUD activations are primary separate observations. Add physical context only for independent visible animation or cast text; never invent a hit or outcome."}
                    scene.extend(reader.read(batch,grounding=grounding)["actions"])
                scene=deduplicate_scene(scene)
            if arm=="scene-only":
                activations=[{**row,"id":uid("scene-activation",row["id"]),"kind":"ability_activation", "end":row["start"],
                    "evidence_origin":"scene_inferred","outcome":None,"target":None}
                    for row in scene if row["kind"]=="ability_cast" and row.get("owner_id")=="selected"]
                links=[]
            else:
                activations,links=primary_actions(candidates,scene if arm=="hud-grounded" else [],bindings)
            write_rows(folder/"events.jsonl",scene+activations)
            requests=reader.requests if reader else []
            report={"clip_id":clip["id"],"arm":arm,"split":split,"scope":scope,
                "start":start,"end":end,"recipe_sha256":digest(pinned),"events":str(folder/"events.jsonl"),
                "hud_sha256":prepared["hud_sha256"] if arm!="scene-only" else None,
                "requests":requests,"associations":links,"status":"completed",
                "runtime":cache_runtime(runner.identity) if runner else {"engine":"existing_hud_ocr_and_persistence"},
                "physical_capability":arm!="hud-only","source_labels_read":False,
                "peak_gpu_memory_bytes":runner.peak_memory_bytes if runner else 0}
            atomic_json(folder/"report.json",report);reports.append(report)
            print(arm,clip["id"],len(scene),"scene events",len(activations),"activation observations",flush=True)
    except BaseException as exc:
        atomic_json(root/"results"/scope/arm/"failure.json",{"error":str(exc),"completed_clips":[r["clip_id"] for r in reports]})
        raise
    finally:
        if runner:
            runner.close()
    result={"arm":arm,"scope":scope,"recipe_sha256":digest(pinned),"clips":reports,"status":"completed",
            "accepted_requests":sum(r.get("status")=="accepted" for report in reports for r in report["requests"]),
            "rejected_requests":sum(r.get("status")=="rejected" for report in reports for r in report["requests"])}
    result["qualified"]=arm=="hud-only" or result["accepted_requests"]>0
    atomic_json(root/"results"/scope/arm/"report.json",result)
    if preflight and not result["qualified"]:
        raise ValueError("Preflight produced no accepted scene responses; inspect rejected responses before continuing")
    return result


def freeze(root,reference_dir):
    from .scene_ablation_score import validate_reference
    root,reference_dir=Path(root),Path(reference_dir)
    study=read_json(root/"study.json");pinned=recipe(study);references=[]
    review=read_json(root/"development-review.json")
    if (not review.get("reviewer") or review.get("recipe_sha256")!=digest(pinned)
        or review.get("held_out_answers_used_for_tuning") is not False
        or review.get("unresolved_infrastructure_failures")!=0 or set(review.get("reviewed_arms",[]))!=set(ARMS)):
        raise ValueError("Review all development arms and infrastructure before recipe freeze")
    for clip in study["clips"]:
        path=reference_dir/clip["split"]/(clip["id"]+".json")
        validate_reference(read_json(path),clip,study["source"]["sha256"],require_complete=True)
        references.append({"path":str(path.resolve()),"sha256":file_hash(path)})
    runtimes=[]
    for arm in ARMS:
        result=read_json(root/"results/development"/arm/"report.json")
        validate_run(result,study,arm,"development")
        if arm!="hud-only":
            if not result["accepted_requests"]:
                raise ValueError("GPU development produced no valid scene responses")
            runtimes.extend(digest(r["runtime"]) for r in result["clips"])
    if len(set(runtimes))!=1:
        raise ValueError("Scene arms used different model/runtime configurations")
    inputs=[]
    for clip in study["clips"]:
        path=root/"prepared"/clip["id"]/"prepared.json";data=read_json(path)
        paths=[path,Path(data["body"]),Path(data["bindings"]),Path(data["hud_candidates"])]
        paths.extend(Path(f["input_path"]) for arm in ("scene-only","hud-grounded") for f in data["views"][arm])
        inputs.extend({"path":str(p.resolve()),"sha256":file_hash(p)} for p in paths)
    value={"recipe":pinned,"recipe_sha256":digest(pinned),"reference_files":references,"input_files":inputs,
           "development_review_sha256":file_hash(root/"development-review.json"),
           "runtime_sha256":runtimes[0],"reference_status":"independent_source_agent_drafts",
           "held_out_inference_started":False}
    target=root/"recipe-freeze.json"
    if target.exists() and read_json(target)!=value:
        raise ValueError("Refusing to overwrite a frozen evaluation recipe")
    atomic_json(target,value)
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["prepare","run","freeze","score"])
    parser.add_argument("--run-dir",type=Path,required=True)
    parser.add_argument("--metadata",type=Path)
    parser.add_argument("--source",type=Path)
    parser.add_argument("--arm",choices=ARMS)
    parser.add_argument("--split",choices=["development","held_out"],default="development")
    parser.add_argument("--preflight",action="store_true")
    parser.add_argument("--references",type=Path)
    args=parser.parse_args()
    if args.command=="prepare":
        prepare(args.run_dir,args.metadata,args.source)
    elif args.command=="run":
        run_arm(args.run_dir,args.arm,args.split,args.preflight)
    elif args.command=="freeze":
        freeze(args.run_dir,args.references)
    else:
        from .scene_ablation_score import score_campaign
        score_campaign(args.run_dir,args.references,args.split)


if __name__=="__main__":
    main()
