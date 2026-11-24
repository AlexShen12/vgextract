"""Isolated two-team body retrieval campaign. No submission and no production writes."""
import argparse
import json
from pathlib import Path
import resource
import time

from .core import Store, atomic_json, digest, now, read_json, read_rows, write_rows
from .media import file_hash

TARGET = {"track_segments": 300, "per_champion": 20, "negative_segments": 50, "pilot_champions": 10}


def initialize(root):
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    path=root/"campaign.json"
    if path.exists(): return read_json(path)
    data={"schema_version":1,"created_at":now(),"experimental":True,"targets":TARGET,"gpu_ceiling_seconds":28800,
          "pilot_champions":[],"clips":[],"gallery":[],"labels":[],"budget_ledger":[]}
    atomic_json(path,data); return data


def coverage(data):
    from .body_identity import canonical_champion

    labels = [row for row in data["labels"] if row.get("split") != "gallery"]
    positives, negative_rows = [], []
    pilot = set(data["pilot_champions"])
    for row in labels:
        destination = positives if canonical_champion(row.get("champion")) in pilot and not row.get("negative_kind") else negative_rows
        destination.append(row)
    segments = {row["segment_label_id"] for row in positives}
    by_champion = {name: len({row["segment_label_id"] for row in positives if row["champion"] == name})
                   for name in data["pilot_champions"]}
    negatives = len({row["segment_label_id"] for row in negative_rows})
    splits = {}
    for split, positive_target, negative_target in (("development", 150, 25), ("validation", 50, 10), ("held_out", 100, 15)):
        positive_count = len({row["segment_label_id"] for row in positives if row.get("split") == split})
        negative_count = len({row["segment_label_id"] for row in negative_rows if row.get("split") == split})
        splits[split] = {"positive": positive_count, "negative": negative_count,
                         "positive_gap": max(0, positive_target - positive_count),
                         "negative_gap": max(0, negative_target - negative_count)}
    held_out = [row for row in positives if row.get("split") == "held_out" and row.get("visibility") == "readable"]
    held_out_champion_gaps = {name: max(0, 10 - len({r["segment_label_id"] for r in held_out if r["champion"] == name}))
                              for name in data["pilot_champions"]}
    held_out_team_gaps = {team: max(0, 50 - len({r["segment_label_id"] for r in held_out if r.get("team") == team}))
                          for team in ("friendly", "enemy")}
    return {"track_segments":len(segments),"negative_segments":negatives,"segments_per_champion":by_champion,
            "pilot_champion_gap":max(0,10-len(data["pilot_champions"])),
            "segment_gap":max(0,300-len(segments)),"negative_gap":max(0,50-negatives),
            "champion_gaps":{name:max(0,20-count) for name,count in by_champion.items()},
            "splits": splits, "held_out_champion_gaps": held_out_champion_gaps, "held_out_team_gaps": held_out_team_gaps,
            "unverified_held_out":sum(l.get("split")=="held_out" and
                (l.get("reviewer_kind")!="human" or l.get("verified") is not True or l.get("source_only") is not True) for l in labels),
            "gallery_entries":len(data["gallery"]),"quality_measured":False}


def verify_campaign(data,cache=None):
    if cache is not None:
        from .inventory_references import load_inventory_references,reference_root
        version=data.get("reference_version","16.22.1")
        catalog=load_inventory_references(reference_root(cache,version),version)
        roster={a["id"] for a in catalog["assets"] if a["kind"]=="champion"}
        if not set(data["pilot_champions"])<=roster:raise ValueError("Pilot identities must use IDs in the pinned official roster")
    if len(set(data["pilot_champions"]))!=len(data["pilot_champions"]): raise ValueError("Duplicate pilot champions")
    groups={}; sources={}; ids=set(); source_hashes={}
    clips_by_source = {}
    for clip in data["clips"]:
        if clip["split"] not in {"gallery","development","validation","held_out"} or not clip.get("match_group"):
            raise ValueError("Body clip needs a frozen split/match group")
        if clip["id"] in ids or not clip["id"].replace("-", "").isalnum(): raise ValueError("Invalid body clip ID")
        ids.add(clip["id"])
        if clip["match_group"] in groups and groups[clip["match_group"]]!=clip["split"]:
            raise ValueError("Body match crosses gallery/evaluation splits")
        groups[clip["match_group"]]=clip["split"]
        if clip["source_sha256"] in sources and sources[clip["source_sha256"]] != clip["match_group"]:
            raise ValueError("Duplicate source belongs to different match groups")
        sources[clip["source_sha256"]] = clip["match_group"]
        if clip.get("previously_inspected") and clip["split"] != "development":
            raise ValueError("Previously inspected footage is development-only")
        if not 0<clip["end"]-clip["start"]<=12: raise ValueError("Body clips must be at most twelve seconds")
        if clip.get("continuity_source_reviewed") is not True:
            raise ValueError("Body windows require source-only playback/selection continuity review")
        if clip.get("accepted_roster") and not clip.get("roster_evidence"):
            raise ValueError("Contextual roster candidates require accepted HUD evidence")
        source_path = Path(clip["source_path"]).resolve()
        if source_path not in source_hashes:
            source_hashes[source_path] = file_hash(source_path)
        if source_hashes[source_path] != clip["source_sha256"]:
            raise ValueError("Body source changed")
        clips_by_source.setdefault(clip["source_sha256"], []).append(clip)
    label_ids = set()
    segment_metadata = {}
    for label in data["labels"]:
        if label["id"] in label_ids:
            raise ValueError("Duplicate body label ID")
        label_ids.add(label["id"])
        key = label["segment_label_id"]
        metadata = tuple(label.get(field) for field in ("source_sha256", "split", "team", "champion", "negative_kind"))
        if segment_metadata.setdefault(key, metadata) != metadata:
            raise ValueError("Body segment labels disagree on source, split, team or identity")
        if "body_box" in label:
            from .core import finite

            box = label["body_box"]
            time = label.get("time")
            if (not finite(time) or not isinstance(box, list) or len(box) != 4 or not all(finite(n) for n in box)
                    or not (0 <= box[0] < box[2] and 0 <= box[1] < box[3])
                    or label["team"] not in {"friendly", "enemy"}
                    or not any(clip["split"] == label["split"] and clip["start"] <= time < clip["end"]
                               for clip in clips_by_source.get(label.get("source_sha256"), ()))):
                raise ValueError("Source body label needs a native box and time within a clip of its split")
            if label.get("champion") and label["champion"] not in data["pilot_champions"] and not label.get("negative_kind"):
                raise ValueError("Out-of-gallery source labels must declare a negative kind")
    for entry in data["gallery"]:
        from .body_identity import reference_approved
        if not reference_approved(entry):raise ValueError("Unreviewed body reference; model drafts cannot claim human verification")
        if entry["champion"] not in data["pilot_champions"]:raise ValueError("Body gallery identity is outside the declared pilot roster")
        if entry["reference_kind"] not in {"gameplay","rendered"} or not entry.get("identity_evidence"):
            raise ValueError("Gallery identity needs independent evidence or a render audit")
        for key,hashkey in (("path","sha256"),("mask_path","mask_sha256")):
            if file_hash(Path(entry[key]))!=entry[hashkey]: raise ValueError("Gallery pixels changed")
        if entry.get("match_group") in {c["match_group"] for c in data["clips"] if c["split"]!="gallery"}:
            raise ValueError("Gallery overlaps evaluation match")
    return data


def proposal_matches(previous, current, maximum_step):
    """Match each bar to its unique nearest same-team neighbor in both directions."""
    def nearest(source, targets):
        matches = {}
        for index, candidate in enumerate(source):
            x, y = candidate["bar_box"][:2]
            distances = sorted((abs(x - target["bar_box"][0]) + abs(y - target["bar_box"][1]), i)
                               for i, target in enumerate(targets) if candidate["team"] == target["team"])
            if distances and distances[0][0] <= maximum_step and (
                len(distances) == 1 or distances[0][0] < distances[1][0]
            ):
                matches[index] = distances[0][1]
        return matches

    forward = nearest(current, previous)
    reverse = nearest(previous, current)
    return {index: prior for index, prior in forward.items() if reverse.get(prior) == index}


def stable_healthbar_bindings(frames, profile, observe=None, *, require_level=False):
    """Confirm independent champion proposals on consecutive frames of one scene."""
    from PIL import Image
    from .body_identity import healthbar_proposals
    from .body_level import level_proposals
    from .hybrid_intake import _panel_state
    from .intake_profile import load_profile as load_layout

    layout = load_layout(profile["anchor_profile"]) if profile.get("anchor_profile") else None
    scoreboard = layout if layout and layout.get("panels", {}).get("scoreboard") else None

    def detect():
        for index, frame in enumerate(frames):
            with Image.open(frame["path"]) as image:
                suppressed = scoreboard and _panel_state(image, scoreboard, {"route": "scoreboard"}, True) == "present"
                current = None if suppressed else healthbar_proposals(image, profile, require_coherent_fill=True)
                if observe is not None:
                    current = observe(index, image, current)
                if require_level and current:
                    current = level_proposals(image, profile, current)
                yield current

    return stable_proposal_bindings(frames, profile, detect())


def stable_proposal_bindings(frames, profile, proposals):
    config = profile["body_binder"]
    required = config["confirm_samples"]
    bindings = []
    previous = []
    for index, (frame, current) in enumerate(zip(frames, proposals, strict=True)):
        if index:
            prior_frame = frames[index - 1]
            contiguous = 0 < frame["time"] - prior_frame["time"] <= .095
            if not contiguous or frame.get("segment_id") != prior_frame.get("segment_id"):
                previous = []
        if current is None:
            previous = []
            continue
        matches = proposal_matches([row["last_proposal"] for row in previous], current, config["maximum_step_pixels"])
        following = []
        for i, proposal in enumerate(current):
            prior = previous[matches[i]] if i in matches else None
            level = proposal.get("level")
            previous_level = prior["last_proposal"].get("level") if prior else None
            if level is not None and previous_level is not None and level - previous_level not in (0, 1):
                prior = None
            first = prior["first_proposal"] if prior else proposal
            evidence = prior["evidence"] if prior else []
            binding = prior["binding"] if prior else None
            if binding is None:
                evidence = [*evidence, {"frame_index": index, "body_box": proposal["actor_region"],
                                        "bar_box": proposal["bar_box"], "level": level}]
                if len(evidence) == required:
                    binding = {**first, "seed_frame_index": evidence[0]["frame_index"], "confirmation_frame_index": index,
                               "evidence_frame_ids": [frames[row["frame_index"]]["id"] for row in evidence],
                               "observations": list(evidence[:-1])}
                    bindings.append(binding)
            if binding is not None:
                binding["end_frame_index"] = index + 1
                binding["observations"].append({"frame_index": index, "body_box": proposal["actor_region"],
                                                "bar_box": proposal["bar_box"], "level": level})
            following.append({"last_proposal": proposal, "first_proposal": first,
                              "evidence": evidence, "binding": binding})
        previous = following
    return bindings


def proposal_jobs(clip, frames, directory, profile):
    jobs = []
    for binding in stable_healthbar_bindings(frames, profile, require_level=True):
        start = binding["seed_frame_index"]
        end = binding["end_frame_index"]
        left, top, right, bottom = binding["actor_region"]
        prompt = {"object_id": 1, "frame_index": start, "box_xyxy": binding["actor_region"],
                  "point": [(left + right) / 2, (top + bottom) / 2]}
        jobs.append({"id": f"{clip['id']}-proposal-{len(jobs)}",
                     "frames_dir": str((directory / "sam-inputs").resolve()), "frames": frames,
                     "width": profile["size"][0], "height": profile["size"][1], "prompts": [prompt],
                     "proposals": [binding], "source_clip": clip["id"], "seed_frame_index": start,
                     "end_frame_index": end})
    return jobs


def prepare(root,cache=Path("data/league_assets")):
    from PIL import Image
    from .action_v3 import load_profile
    from .scene_reference import sample_source
    root=Path(root); data=verify_campaign(read_json(root/"campaign.json"),cache)
    encoder_lock=root/"encoder-lock.json"
    if data["clips"] and not encoder_lock.exists():raise ValueError("Stage the exact offline encoder lock before preparation")
    recipe={"campaign_hash":digest(data),"reader_version":file_hash(Path(__file__).with_name("body_identity.py")),
            "healthbar_detector_version":file_hash(Path(__file__).with_name("action_body.py")),
            "encoder_lock_hash":digest(read_json(encoder_lock)) if encoder_lock.exists() else None,
            "preparer_version":file_hash(Path(__file__)),
            "worker_version":file_hash(Path(__file__).with_name("experiment_worker.py")),
            "sam_adapter_version":file_hash(Path(__file__).with_name("sam_compat.py"))}
    jobs=[]; clips=[]
    for clip in data["clips"]:
        if clip["split"]=="gallery": continue
        profile=load_profile(clip.get("action_profile")); frames=[]
        directory=root/"prepared"/digest(recipe)[:20]/clip["id"]
        for index,(pts,image) in enumerate(sample_source(clip["source_path"],clip["start"],clip["end"],16)):
            directory.mkdir(parents=True,exist_ok=True)
            native=directory/f"{index:06d}.png"; native.parent.mkdir(parents=True,exist_ok=True); image.save(native)
            frame_dir=directory/"sam-inputs"; frame_dir.mkdir(exist_ok=True)
            sam_path=frame_dir/f"{index:06d}.jpg";image.save(sam_path,quality=95,subsampling=0)
            frames.append({"id":digest([clip["source_sha256"],pts])[:20],"path":str(native.resolve()),"time":pts,
                           "sha256":file_hash(native),"sam_path":str(sam_path.resolve()),"sam_sha256":file_hash(sam_path)})
        if not frames: raise ValueError("No body source frames")
        hud_contexts=[]
        if clip.get("config_path"):
            from .core import Config
            from .sam_tracking import portrait_rows
            from .action_body import bind_frames
            config=Config.load(Path(clip["config_path"]))
            context_frames=[{**f,"native_path":f["path"],"layout_valid":True,"segment_id":clip["id"],
                             "owner_epochs":{"selected":0}} for f in frames]
            portraits=portrait_rows(Store(root/"hud-evidence"),config,context_frames)
            bodies,_=bind_frames(context_frames,portraits,profile)
            names={r["frame_id"]:r for r in portraits}
            for index,body in enumerate(bodies):
                portrait=names[body["frame_id"]]
                hud_contexts.append({"frame_id":body["frame_id"],"time":body["time"],
                    "source_frame_sha256":frames[index]["sha256"],"champion":portrait["hud_champion"],
                    "binding_visible":body["binding_visible"],"actor_region":body.get("actor_region"),
                    "catalog_id":portrait["inventory_catalog_id"],"binding_id":body["id"]})
        clip_jobs = proposal_jobs(clip, frames, directory, profile)
        jobs.extend(clip_jobs)
        clips.append({**clip, "frames": frames, "jobs": [job["id"] for job in clip_jobs],
                      "unseeded": not clip_jobs, "hud_contexts": hud_contexts})
    data={"schema_version":1,"recipe":recipe,"clips":clips,"jobs":jobs,"gallery":data["gallery"],
            "pilot_champions":data["pilot_champions"],"seed_policy":"consecutive_healthbar_bindings_16fps",
            "late_entrants":"confirm a new binding on consecutive frames; misses remain in the denominator"}
    atomic_json(root/"prepared.json",data); return data


def track(root,cache,preflight=False):
    import torch
    from .experiment_worker import tracking_job
    root=Path(root); prepared=read_json(root/"prepared.json")
    data=verify_campaign(read_json(root/"campaign.json"),cache)
    if digest(data)!=prepared["recipe"]["campaign_hash"]: raise ValueError("Body campaign changed since preparation")
    if digest(read_json(root/"encoder-lock.json"))!=prepared["recipe"]["encoder_lock_hash"]:
        raise ValueError("Body encoder pin changed after preparation")
    checks={"preparer_version":Path(__file__),"reader_version":Path(__file__).with_name("body_identity.py"),
            "worker_version":Path(__file__).with_name("experiment_worker.py"),"sam_adapter_version":Path(__file__).with_name("sam_compat.py")}
    if "healthbar_detector_version" in prepared["recipe"]:
        checks["healthbar_detector_version"] = Path(__file__).with_name("action_body.py")
    if any(file_hash(path)!=prepared["recipe"][key] for key,path in checks.items()):raise ValueError("Body implementation changed since preparation")
    jobs=prepared["jobs"][:1] if preflight else prepared["jobs"]
    from .body_identity import reference_approved
    if not prepared["gallery"] or not all(reference_approved(e) for e in prepared["gallery"]):
        raise ValueError("Reviewed gameplay/render reference preparation is required before GPU work")
    if not jobs: raise ValueError("No automatic body proposals; localization coverage remains zero")
    output=root/("preflight" if preflight else "tracks"); output.mkdir(parents=True,exist_ok=True)
    source_frame_offset = 0
    if preflight:
        from shutil import copy2
        short=output/"inputs"; short.mkdir(exist_ok=True)
        first = jobs[0]
        source_frame_offset = first["prompts"][0].get("frame_index", 0)
        end = min(first.get("end_frame_index", len(first["frames"])), source_frame_offset + 32)
        selected = first["frames"][source_frame_offset:end]
        for i, frame in enumerate(selected):
            copy2(frame["sam_path"], short / f"{i:06d}.jpg")
        jobs = [{**first, "frames": selected, "frames_dir": str(short.resolve()),
                 "seed_frame_index": 0, "end_frame_index": len(selected),
                 "prompts": [{**prompt, "frame_index": 0} for prompt in first["prompts"]]}]
    for job in jobs:
        for frame in job["frames"]:
            if file_hash(Path(frame["path"]))!=frame["sha256"]: raise ValueError("Body frame changed")
            if file_hash(Path(frame["sam_path"]))!=frame["sam_sha256"]:raise ValueError("SAM input pixels changed")
    started=time.monotonic()
    if not preflight:
        gate=read_json(root/"qualification.json")
        if gate.get("passed") is not True or gate["prepared_hash"]!=digest(prepared):raise ValueError("Matching body preflight qualification required")
        if gate["preflight_hash"]!=digest(read_json(root/"preflight/report.json")):
            raise ValueError("Body preflight changed after qualification")
        if any(file_hash(Path(entry["path"]))!=entry["sha256"] for entry in gate["seed_inputs"]):
            raise ValueError("Qualified seed evidence changed")
    store=Store(root);store.install_signals()
    first_clip_started=None
    def stop():
        nonlocal first_clip_started
        store.check_stop()
        if first_clip_started is None:first_clip_started=time.monotonic()
    result=tracking_job({"protocol":"selected-sam-v1","clips":jobs,"cache":str(cache),"output":str(output/"worker.json")},stop=stop)
    import sys
    ended=time.monotonic()
    result.update(wall_seconds=time.monotonic()-started,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                  peak_host_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  peak_host_memory_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=="darwin" else 1024),
                  model_load_seconds=(first_clip_started or ended)-started,
                  inference_wall_seconds=ended-(first_clip_started or ended),
                  sampled_frames=sum(len(clip["frames"]) for clip in result["clips"]),
                  prepared_hash=digest(prepared),status="completed",source_frame_offset=source_frame_offset)
    atomic_json(output/"report.json",result); return result


def embed(root,encoder_lock,device="cuda"):
    from PIL import Image
    import numpy as np
    from .body_identity import FrozenEncoder
    root=Path(root); prepared=read_json(root/"prepared.json")
    if digest(read_json(Path(encoder_lock)))!=prepared["recipe"]["encoder_lock_hash"]:raise ValueError("Embedding encoder differs from the prepared pin")
    encoder=FrozenEncoder(encoder_lock,device)
    entries=prepared["gallery"]; gallery=[]
    from .body_identity import unit
    def encode(image,mask):
        import hashlib
        identity=digest([encoder.signature,hashlib.sha256(image.tobytes()).hexdigest(),
                         hashlib.sha256(mask.tobytes()).hexdigest(),image.size,mask.size,image.mode,mask.mode])
        path=root/"embedding-cache"/(identity+".json")
        if path.exists():return unit(read_json(path)["vector"]).tolist()
        vector=encoder.encode(image,mask).tolist();atomic_json(path,{"vector":vector,"encoder_signature":encoder.signature});return vector
    for entry in entries:
        if file_hash(Path(entry["path"]))!=entry["sha256"] or file_hash(Path(entry["mask_path"]))!=entry["mask_sha256"]:
            raise ValueError("Gallery reference pixels changed")
        with Image.open(entry["path"]) as image,Image.open(entry["mask_path"]) as mask:
            gallery.append(encode(image,mask))
    samples=[]; vectors=[]
    for clip in prepared["clips"]:
        for job_id in clip["jobs"]:
            job=next(j for j in prepared["jobs"] if j["id"]==job_id)
            result=read_json(root/"tracks"/(job_id+"-complete.json"))
            for frame in result["frames"]:
                index=frame["frame_index"]
                if index%4: continue
                source=clip["frames"][index]
                if file_hash(Path(source["path"]))!=source["sha256"]:raise ValueError("Body source frame changed after tracking")
                for record in frame["masks"]:
                    path=Path(record["mask_path"])
                    if file_hash(path)!=record["sha256"]: raise ValueError("SAM body mask changed")
                    with Image.open(source["path"]) as image,Image.open(path) as mask:
                        pixels=np.asarray(mask)>127
                        # Remove all UI and overhead text, even if a mask mistakenly includes it.
                        profile=__import__("annotation.league.action_v3",fromlist=["load_profile"]).load_profile(clip.get("action_profile"))
                        scene_box=profile["scene_box"]; keep=np.zeros_like(pixels); keep[scene_box[1]:scene_box[3],scene_box[0]:scene_box[2]]=True
                        proposal=job["proposals"][record["object_id"]-1]
                        for a,b,c,d in profile["body_binder"]["exclude_boxes"]:keep[b:d,a:c]=False
                        from .body_identity import healthbar_proposals
                        for bar in healthbar_proposals(image,profile):
                            a,b,c,d=bar["bar_box"]
                            keep[max(0,b-32):d+3,max(0,a-30):c+30]=False
                        pixels &= keep
                        if not pixels.any(): continue
                        ys,xs=np.where(pixels); box=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)]
                        body=image.crop(box); foreground=Image.fromarray(pixels.astype("uint8")*255).crop(box)
                        vector=encode(body,foreground)
                    track_id=job_id+"-"+str(record["object_id"])
                    sample={"id":digest([source["id"],track_id])[:20],"track_id":track_id,"time":source["time"],
                            "team":proposal["team"],"segment_id":clip["id"],"owner_epoch":0,"visible":True,
                            "source_sha256":clip["source_sha256"],"frame_id":source["id"],"body_box":box,
                            "match_group":clip["match_group"],"split":clip["split"],
                            "accepted_roster":clip.get("accepted_roster"),"mask_sha256":record["sha256"]}
                    contexts=[c for c in clip.get("hud_contexts",[]) if c["source_frame_sha256"]==source["sha256"]
                              and abs(c["time"]-source["time"])<1e-6 and c["binding_visible"] and c.get("champion") and c.get("actor_region")]
                    if len(contexts)==1:
                        context=contexts[0];a,b,c,d=context["actor_region"]
                        intersection=max(0,min(box[2],c)-max(box[0],a))*max(0,min(box[3],d)-max(box[1],b))
                        union=(box[2]-box[0])*(box[3]-box[1])+(c-a)*(d-b)-intersection
                        if union>0 and intersection/union>=.5:
                            sample.update(accepted_hud_champion=context["champion"],hud_body_binding_accepted=True,
                                          hud_catalog_id=context["catalog_id"],hud_binding_id=context["binding_id"])
                    samples.append(sample); vectors.append(vector)
                    if len(samples)%32==0:atomic_json(root/"embeddings-partial.json",{"samples":samples,"query_vectors":vectors,"prepared_hash":digest(prepared)})
    data={"encoder_signature":encoder.signature,"gallery":entries,"gallery_vectors":gallery,
          "samples":samples,"query_vectors":vectors,"evaluation_match_groups":sorted({c["match_group"] for c in prepared["clips"]}),
          "prepared_hash":digest(prepared)}
    atomic_json(root/"embeddings.json",data); return {"samples":len(samples),"gallery":len(gallery),"encoder_signature":encoder.signature}


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("command",choices=("init","coverage","prepare","audit-tracking","preflight","qualify","track","embed","freeze","run","score","identify","match-frame","fit-appearance","reconcile-budget"))
    p.add_argument("--root",type=Path,required=True); p.add_argument("--cache",type=Path,default=Path("data/league_assets"))
    p.add_argument("--encoder-lock",type=Path); p.add_argument("--policy",type=Path); p.add_argument("--arm"); p.add_argument("--device",choices=("cpu","cuda"))
    p.add_argument("--intake-run", type=Path)
    p.add_argument("--scoreboard-skim", "--slot-skim", dest="slot_skim", type=Path,
                   help="Use the champion pool from a completed single-match scoreboard skim")
    p.add_argument("--frame-time", type=float, help="Exact prepared frame timestamp in source seconds for match-frame")
    p.add_argument("--clip-id", help="Prepared clip to use if multiple clips contain the frame")
    p.add_argument("--main-champion", help="Main player's champion ID from the confirmed scoreboard")
    p.add_argument("--friendly-side", choices=("left", "right"), help="Friendly scoreboard side if the main champion appears twice")
    main = p.add_mutually_exclusive_group()
    main.add_argument("--main-track", help="Main player's visible track ID if no unique HUD binding is available")
    main.add_argument("--main-offscreen", action="store_true", help="The main player's body is absent from this frame")
    p.add_argument("--render-gallery", type=Path, help="Audited render gallery JSON for match-frame, overriding the prepared gallery")
    p.add_argument("--siglip-lock", type=Path, help="Use a local SigLIP model lock for match-frame")
    p.add_argument("--skin-encoder", choices=("siglip", "dinov2"), default="siglip")
    p.add_argument("--skin-assignments", type=Path, help="Read compatible locks from a completed appearance pass")
    p.add_argument("--gallery-kind", choices=("gameplay", "rendered", "combined"), default="combined")
    p.add_argument("--budget-root", type=Path)
    p.add_argument("--gpu-seconds", type=int, default=0)
    p.add_argument("--submission", type=Path, action="append", default=[])
    p.add_argument("--job-receipts", type=Path)
    p.add_argument("--identity-result", type=Path, action="append", default=[])
    p.add_argument("--split",choices=("validation","held_out"),default="validation")
    p.add_argument("--recovery",action="store_true")
    args=p.parse_args()
    if (args.siglip_lock is not None or args.skin_assignments is not None) and args.command != "match-frame":
        p.error("--siglip-lock and --skin-assignments require match-frame")
    if args.command == "reconcile-budget":
        from .recognition_budget import reconcile
        result = reconcile(args.root, [read_json(path) for path in args.submission],
                           read_json(args.job_receipts) if args.job_receipts else ())
    elif args.command == "match-frame":
        from .body_frame_matching import match_frame

        if args.frame_time is None or not args.main_champion or not (args.intake_run or args.slot_skim):
            p.error("match-frame requires --frame-time, --main-champion, and --intake-run or --scoreboard-skim")
        if args.device == "cuda":
            p.error("match-frame runs the frozen skin encoder on CPU")
        result = match_frame(args.root, args.intake_run or args.slot_skim, args.cache, frame_time=args.frame_time,
                             main_champion=args.main_champion, slot_skim=args.slot_skim, friendly_side=args.friendly_side,
                             main_track=args.main_track, main_offscreen=args.main_offscreen, clip_id=args.clip_id,
                             render_gallery=args.render_gallery, siglip_lock=args.siglip_lock,
                             encoder=args.skin_encoder, skin_assignments=args.skin_assignments)
    elif args.command in {"identify", "fit-appearance"}:
        from .body_stage import execute_identity
        if args.command == "identify" and not (args.intake_run or args.slot_skim):
            p.error("identify requires --intake-run or --scoreboard-skim")
        result = execute_identity(args.command, args.root, args.cache, intake_run=args.intake_run or args.slot_skim,
                                  policy_path=args.policy, gallery_kind=args.gallery_kind, device=args.device or "cpu",
                                  budget_root=args.budget_root, gpu_seconds=args.gpu_seconds, slot_skim=args.slot_skim,
                                  recovery=args.recovery)
    elif args.command=="init": result=initialize(args.root)
    elif args.command=="coverage": result=coverage(read_json(args.root/"campaign.json"))
    elif args.command=="prepare": result=prepare(args.root,args.cache)
    elif args.command == "audit-tracking":
        from .body_evaluation import audit_tracking

        result = audit_tracking(args.root)
    elif args.command in {"qualify","preflight","track","embed"}:
        from .body_stage import execute
        result=execute(args.command,args.root,args.cache,recovery=args.recovery,encoder_lock=args.encoder_lock,
                       device=args.device or "cuda",budget_root=args.budget_root)
    elif args.command=="freeze":
        from .body_policy import freeze
        result=freeze(args.root)
    elif args.command == "run" and args.policy and "features" in read_json(args.policy):
        from .body_appearance import compare

        if args.device == "cuda":
            p.error("Frozen appearance comparisons run on CPU")
        result = compare(args.root, args.policy, args.split, args.arm)
    elif args.command=="run":
        from .body_identity import run_arms
        from .body_policy import validate_freeze
        policy=read_json(args.policy or args.root/"policy.json");data=validate_freeze(args.root,policy)
        result=run_arms(args.root/"comparisons",data,data["gallery"],data["gallery_vectors"],data["query_vectors"],policy,only_arm=args.arm)
    elif args.command == "score" and args.identity_result:
        from .body_evaluation import score_identities
        result = score_identities(args.root, args.identity_result, args.split, args.gallery_kind, args.policy)
    elif args.command == "score" and args.policy and "features" in read_json(args.policy):
        from .body_evaluation import score_comparisons

        result = score_comparisons(args.root, args.split, args.policy)
    else:
        from .recognition_metrics import body_score, source_aligned
        from .body_policy import validate_freeze
        policy = read_json(args.policy or args.root / "policy.json")
        validate_freeze(args.root, policy)
        policy_hash = digest(policy)
        receipt=args.root/(args.split+"-score.json")
        if receipt.exists():raise ValueError("This evaluation is recorded; create a new campaign revision")
        data=read_json(args.root/"campaign.json"); labels=[l for l in data["labels"] if l["split"]==args.split]
        predictions = [row for path in (args.root / "comparisons").glob("*/predictions.jsonl")
                       for _, row in source_aligned(labels, read_rows(path))]
        metrics=body_score(labels,predictions);counts=coverage(data)
        from .body_evaluation import enough_labels, human_verified

        human = bool(labels) and all(human_verified(label) and label.get("body_box") for label in labels)
        qualified=[arm for arm,m in metrics.items() if all(m[team]["passed"] for team in ("friendly","enemy"))]
        enough = enough_labels(counts)
        if args.split == "held_out":
            validation = read_json(args.root / "validation-score.json")
            if not human or not enough or validation.get("policy_hash") != policy_hash:
                raise ValueError("Held-out scoring needs independent source labels and matching validation")
            qualified = [arm for arm in qualified if arm in validation["qualified_arms"]]
        result={"metrics":metrics,"coverage":counts,"human_verified":human,"qualified_arms":qualified,"policy_hash":policy_hash,
                "release_passed":bool(args.split=="held_out" and human and enough and qualified),"experimental":True,
                "uncertainty_note":"Wilson intervals describe sampled readings; adjacent samples are correlated. Segment coverage is reported separately."}
        atomic_json(receipt,result)
    print(json.dumps(result,indent=2))


if __name__=="__main__": main()
