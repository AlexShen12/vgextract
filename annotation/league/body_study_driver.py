"""Isolated preflight and CPU cast pilot for the November body comparison diagnostic."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from .core import Store,atomic_json,read_json


def preflight(spec,output):
    import numpy as np
    from PIL import Image
    from .body_identity import FrozenEncoder
    from .body_study import extract
    start=time.monotonic()
    inputs=extract(spec,output/"source-check")
    encoder=FrozenEncoder(spec["encoder_lock"],"cpu")
    sample=next(r for r in inputs["samples"] if r["status"]=="available")
    with Image.open(sample["image"]) as image,Image.open(sample["mask"]) as mask:vector=encoder.encode(image,mask)
    if vector.shape!=(384,) or not np.isfinite(vector).all() or abs(np.linalg.norm(vector)-1)>.001:
        raise ValueError("Pinned encoder smoke failed")
    blender=subprocess.run([spec["blender"],"--version"],check=True,capture_output=True,text=True).stdout.splitlines()[0]
    result={"status":"passed","encoder_signature":encoder.signature,"embedding_dimensions":384,
            "source_samples":len(inputs["samples"]),"blender":blender,"seconds":time.monotonic()-start,"quality_gate_passed":False}
    atomic_json(output/"report.json",result);return result


def casts(spec,output):
    from importlib.util import find_spec

    if find_spec("av") is None:
        raise RuntimeError("Cast-bar preflight requires PyAV in the isolated job environment")
    from .cast_bar_intake import run
    reports=[]
    for start,end in spec["cast_windows"]:
        store=Store(output/(str(start)+"-"+str(end)));store.install_signals()
        reports.append(run(store,spec["video"],spec["cast_profile"],spec["ability_catalog"],start,end))
        atomic_json(output/"report.json",{"status":"running","windows":reports,"quality_gate_passed":False})
    result={"status":"completed_development_pilot","windows":reports,"quality_gate_passed":False,"human_verified":False}
    atomic_json(output/"report.json",result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("stage",choices=("preflight","casts"))
    p.add_argument("--spec",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    try:result=(preflight if args.stage=="preflight" else casts)(read_json(args.spec),args.output)
    except BaseException as exc:
        atomic_json(args.output/"failure.json",{"error":str(exc)});raise
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
