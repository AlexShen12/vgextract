"""Discover pinned skin mesh/skeleton/texture/animation inputs from CommunityDragon BIN exports."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .core import atomic_json, digest, now, read_json
from .knowledge import CD, DD, JsonCache
from .media import file_hash
from .reference_bundle import versions


def strings(value):
    if isinstance(value,str):yield value
    elif isinstance(value,dict):
        for child in value.values():yield from strings(child)
    elif isinstance(value,list):
        for child in value:yield from strings(child)


def fetch_file(root,version,path,role,offline):
    path=path.lower().replace("\\","/")
    if not path.startswith("assets/") or ".." in path or not re.fullmatch(r"[a-z0-9_./ -]+",path):
        raise ValueError("Invalid model asset path")
    # CDragon exports game textures as PNG, not the client's proprietary TEX payload.
    if path.endswith((".tex",".dds")):path=path.rsplit(".",1)[0]+".png"
    url=f"{CD}/{version}/game/{path}"
    target=root/"files"/path
    receipt=root/"receipts"/(hashlib.sha256(url.encode()).hexdigest()+".json")
    if receipt.exists():
        record=read_json(receipt)
        if file_hash(target)!=record["sha256"]:raise ValueError("Cached model asset changed")
        return {**record,"role":role,"path":str(target.resolve())}
    if offline:raise ValueError("Model asset missing offline: "+url)
    try:
        with urlopen(Request(url, headers={"User-Agent": "gameplay-reference-resolver/0.1"}), timeout=40) as response:
            if response.url != url:
                raise ValueError(f"{role} asset redirected: {url}")
            raw = response.read(64 * 1024 * 1024 + 1)
    except HTTPError as error:
        raise ValueError(f"{role} asset returned HTTP {error.code}: {url}") from error
    if len(raw)>64*1024*1024:raise ValueError("Model asset exceeds size limit")
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_suffix(target.suffix+".tmp");temporary.write_bytes(raw);temporary.replace(target)
    record={"url":url,"sha256":hashlib.sha256(raw).hexdigest(),"retrieved_at":now()}
    atomic_json(receipt,record);return {**record,"role":role,"path":str(target.resolve())}


def cache_assets(cache,skins,dd="16.22.1",cd="16.22",offline=False):
    versions(dd,cd);root=Path(cache)/"body-assets"/(dd+"_"+cd)
    metadata=JsonCache(root/"raw",offline=offline)
    roster,source=metadata.get(f"{DD}/cdn/{dd}/data/en_US/champion.json")
    result=[]
    for request in skins:
        name=request["champion"];files=[];sources=[source]
        if name not in roster["data"]:raise ValueError("Unknown pilot champion")
        skin_id=int(request.get("skin_id",int(roster["data"][name]["key"])*1000))
        if skin_id//1000!=int(roster["data"][name]["key"]):raise ValueError("Skin belongs to another champion")
        if skin_id%1000 and not request.get("source_identity_evidence"):
            raise ValueError("Footage skin references require independent identity evidence")
        try:
            internal=name.lower();number=skin_id%1000
            data,source=metadata.get(f"{CD}/{cd}/game/data/characters/{internal}/skins/skin{number}.bin.json");sources.append(source)
            record=next(v for v in data.values() if isinstance(v,dict) and v.get("championSkinId")==skin_id)
            mesh=record["skinMeshProperties"]
            files.append(fetch_file(root,cd,mesh["simpleSkin"],"mesh",offline))
            files.append(fetch_file(root,cd,mesh["skeleton"],"skeleton",offline))
            if not mesh.get("texture"):raise ValueError("Skin requires explicit material bindings")
            files.append({**fetch_file(root,cd,mesh["texture"],"texture",offline),"material":"__auto_single__"})
            animation,source=metadata.get(f"{CD}/{cd}/game/data/characters/{internal}/animations/skin{number}.bin.json");sources.append(source)
            paths=sorted({s for s in strings(animation) if s.lower().endswith(".anm")})
            if not paths:raise ValueError("No supported animation payloads in the pinned graph")
            for path in paths:
                if any(token in Path(path).stem.lower() for token in ("idle","run","walk","attack","spell","cast")):
                    files.append(fetch_file(root,cd,path,"animation",offline))
            if not any(f["role"]=="animation" for f in files):raise ValueError("No supported pose animations")
            status="cached"
            result.append({"champion":name,"skin_id":str(skin_id),"files":files,"sources":sources,"status":status})
        except (OSError,ValueError,KeyError,RuntimeError,StopIteration) as exc:
            result.append({"champion":name,"skin_id":str(skin_id),"files":files,"sources":sources,"status":"unsupported","error":str(exc)})
        atomic_json(root/"discovery-report.json",{"reference_version":dd,"communitydragon_version":cd,"skins":result})
    return {"reference_version":dd,"communitydragon_version":cd,"skins":result,"manifest_hash":digest(result)}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--cache",type=Path,default=Path("data/league_assets"))
    p.add_argument("--skins",type=Path,required=True);p.add_argument("--offline",action="store_true");args=p.parse_args()
    print(json.dumps(cache_assets(args.cache,read_json(args.skins),offline=args.offline),indent=2))


if __name__=="__main__":main()
