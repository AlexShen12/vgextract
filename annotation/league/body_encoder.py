"""Prepare an exact offline DINOv2 ViT-S/14 code/weight pin, without GPU inference."""
import argparse
import json
from pathlib import Path
import subprocess
from urllib.request import urlopen

from .core import atomic_json, read_json
from .media import file_hash

REPO="https://github.com/facebookresearch/dinov2.git"
WEIGHTS="https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"


def cache_encoder(root,revision=None):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    lock=root/"encoder-lock.json"
    if lock.exists():
        data=read_json(lock)
        if file_hash(Path(data["weights"]))!=data["weights_sha256"]:raise ValueError("Frozen encoder weights changed")
        if revision and revision!=data["code_revision"]:raise ValueError("Encoder pin already exists; use a new directory")
        return data
    if revision is None:
        revision=subprocess.run(["git","ls-remote",REPO,"HEAD"],check=True,capture_output=True,text=True).stdout.split()[0]
    import re
    if not re.fullmatch(r"[a-f0-9]{40}",revision):raise ValueError("Encoder requires an exact Git commit")
    code=root/"code"
    if not (code/".git").exists():
        subprocess.run(["git","init",str(code)],check=True,capture_output=True)
        subprocess.run(["git","-C",str(code),"remote","add","origin",REPO],check=True,capture_output=True)
    subprocess.run(["git","-C",str(code),"fetch","--depth=1","origin",revision],check=True,capture_output=True)
    subprocess.run(["git","-C",str(code),"checkout","--detach",revision],check=True,capture_output=True)
    weights=root/"dinov2_vits14_pretrain.pth"
    if not weights.exists():
        temporary=weights.with_suffix(".tmp")
        with urlopen(WEIGHTS,timeout=60) as response,temporary.open("wb") as stream:
            if response.url!=WEIGHTS:raise ValueError("Unexpected encoder weight redirect")
            total=0
            while block:=response.read(1024*1024):
                total+=len(block)
                if total>128*1024*1024:raise ValueError("Encoder checkpoint exceeds size limit")
                stream.write(block)
        temporary.replace(weights)
    data={"architecture":"dinov2_vits14","code_root":str(code),"code_revision":revision,"code_source":REPO,
          "weights":str(weights),"weights_source":WEIGHTS,"weights_sha256":file_hash(weights)}
    atomic_json(lock,data);return data


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",type=Path,required=True);p.add_argument("--revision")
    args=p.parse_args();print(json.dumps(cache_encoder(args.root,args.revision),indent=2))


if __name__=="__main__":main()
