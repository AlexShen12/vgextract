"""Opt-in CPU cast-bar intake for independently calibrated native HUD layouts."""
import argparse
import json
from pathlib import Path

from .cast_bars import AbilityCatalog, CastBarReader, rebuild
from .core import Store, atomic_json, digest, implementation_hash, read_json, read_rows, uid, write_rows
from .intake_profile import check_box, layout_matches
from .media import file_hash, probe
from .playback import PlaybackTracker
from .scene_reference import sample_source


def run(store,video,profile_path,catalog_path,start,end,identities=()):
    profile=read_json(Path(profile_path));source=probe(Path(video))
    source={**source,"path":str(Path(video).resolve()),"sha256":file_hash(Path(video))}
    if store.path("manifest.json").exists():
        manifest=read_json(store.path("manifest.json"))
        if manifest["source"]["sha256"]!=source["sha256"]:raise ValueError("A different source requires a new run directory")
        if not manifest.get("experimental_only") and not manifest["interval"][0]<=start<end<=manifest["interval"][1]:
            raise ValueError("Cast interval lies outside the prepared run")
    if profile.get("schema_version")!=1 or profile.get("size")!=[source["width"],source["height"]]:
        raise ValueError("Cast profile must match native video geometry")
    if not profile.get("calibration_source_sha256") or not profile.get("calibration_frames") or len(profile.get("anchors",[]))<2:
        raise ValueError("Cast layout requires source calibration and two independent UI anchors")
    for key in ("text_box","progress_box","scene_box","clock_box","identity_box"):
        check_box(profile[key],profile["size"])
    for anchor in profile["anchors"]:check_box(anchor["box"],profile["size"])
    if source["fps"]<16 or not 0<=start<end<=source["duration"]:raise ValueError("Invalid cast source interval/cadence")
    catalog=AbilityCatalog(catalog_path,profile.get("aliases",[]))
    implementation=implementation_hash(["cast_bars.py","cast_bar_intake.py","playback.py","readers.py","numeric_reader.py","intake_profile.py"])
    recipe={"source_sha256":source["sha256"],"interval":[start,end],"profile":profile,"catalog":catalog.signature,
            "identity_inputs":digest(list(identities)),"implementation":implementation}
    root=store.path("experiments/cast-bars/"+digest(recipe)[:20]);root.mkdir(parents=True,exist_ok=True)
    reader=CastBarReader(root,catalog,{**profile,"context_signature":implementation},profile.get("masks",[]))
    tracker=PlaybackTracker(source["sha256"],16,scene_box=profile["scene_box"],masks=profile.get("masks",[]))
    rows=[];epoch=0;last_name=None;last_portrait=None;clock=None
    for index,(pts,image) in enumerate(sample_source(video,start,end,16)):
        store.check_stop();valid,_=layout_matches(image,profile)
        if index%16==0:
            from .numeric_reader import numeric_preprocess
            from .readers import tesseract,parse_text
            clock_path=root/"clock.png";numeric_preprocess(image.crop(profile["clock_box"])).save(clock_path)
            raw=tesseract(clock_path,whitelist="0123456789:")
            try:clock=parse_text(raw["raw_text"],"clock")["seconds"] if raw.get("score",0)>=.85 else None
            except ValueError:clock=None
        playback=tracker.update(pts,clock,image,valid)
        import numpy as np
        portrait=np.asarray(image.crop(profile["identity_box"]).convert("RGB").resize((24,24)),dtype=float)
        if last_portrait is not None and np.abs(portrait-last_portrait).mean()/255>.12:
            epoch+=1;last_name=None;last_portrait=portrait
        if last_portrait is None:last_portrait=portrait
        candidates=[r for r in identities if r.get("source_sha256")==source["sha256"] and abs(r["time"]-pts)<1e-6
                    and r.get("owner_id","selected")=="selected" and r.get("acceptance")=="accepted"]
        champion=candidates[0].get("value") if len(candidates)==1 else None
        if champion and last_name and champion!=last_name:epoch+=1
        if champion:last_name=champion
        frame={"id":uid("cast-source-frame",source["sha256"],pts),"time":pts,"layout_valid":valid,
               "segment_id":playback["playback_segment_id"],"owner_epochs":{"selected":epoch}}
        rows.append(reader.read(image,frame,{"owner_id":"selected","hud_champion":champion}))
        if len(rows)%32==0:write_rows(root/"cast_bar_readings.jsonl",rows)
    write_rows(root/"cast_bar_readings.jsonl",rows);_,events=rebuild(root)
    report={"status":"completed_experimental","root":str(root),"recipe":recipe,"source":source,
            "reader_signature":reader.signature,"catalog_id":catalog.signature,"profile_id":profile["id"],
            "readings":len(rows),"occurrences":len(events),"production_intake_modified":False,"sampling_fps":16}
    atomic_json(root/"report.json",report)
    if not store.path("manifest.json").exists():
        atomic_json(store.path("manifest.json"),{"source":source,"interval":[0,source["duration"]],"experimental_only":True})
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--run-dir",type=Path,required=True)
    p.add_argument("--video",type=Path,required=True);p.add_argument("--profile",type=Path,required=True)
    p.add_argument("--catalog",type=Path,required=True);p.add_argument("--start",type=float,required=True);p.add_argument("--end",type=float,required=True)
    p.add_argument("--identities",type=Path);args=p.parse_args();store=Store(args.run_dir);store.install_signals()
    print(json.dumps(run(store,args.video,args.profile,args.catalog,args.start,args.end,
                         read_rows(args.identities) if args.identities else []),indent=2))


if __name__=="__main__":main()
