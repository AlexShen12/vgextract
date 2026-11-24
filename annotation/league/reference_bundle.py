"""Immutable reference bundles. Asset versions never establish a footage patch."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
from pathlib import Path
import re

from .core import atomic_json, digest, now, read_json
from .inventory_references import asset_bytes, write_icon
from .knowledge import CD, DD, JsonCache, plain_text
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def download(url):
    allowed = {"ddragon.leagueoflegends.com", "raw.communitydragon.org"}
    if urlsplit(url).scheme != "https" or urlsplit(url).hostname not in allowed:
        raise ValueError("Untrusted reference host")
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers={"User-Agent": "gameplay-reference-resolver/0.1"}), timeout=30) as response:
                if response.url != url:
                    raise ValueError("Unexpected reference redirect")
                raw = response.read(8*1024*1024+1)
            break
        except OSError:
            if attempt == 2:
                raise
    if len(raw) > 8*1024*1024:
        raise ValueError("Reference image exceeds size limit")
    return raw

DD_VERSION = "16.22.1"
CD_VERSION = "16.22"


def resolve_versions(cache, dd="latest", cd="latest", *, offline=False):
    """Resolve mutable aliases only at preparation time and archive the discovery evidence."""
    if dd!="latest" and cd!="latest":return versions(dd,cd)
    if offline:raise ValueError("Offline preparation requires explicit reference versions")
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as directory:
        temporary=Path(directory);metadata=JsonCache(temporary)
        if dd=="latest":
            available,_=metadata.get(f"{DD}/api/versions.json");dd=available[0]
        if cd=="latest":cd=dd.rsplit(".",1)[0]
        versions(dd,cd)
        release,_=metadata.get(f"{CD}/{cd}/content-metadata.json")
        if not release.get("version","").startswith(cd+"."):
            raise ValueError("Published CommunityDragon family differs from the resolved pin")
        sources=[]
        for path in temporary.glob("*.json"):
            envelope=read_json(path);sources.append({k:envelope[k] for k in ("url","sha256","retrieved_at")})
            target=Path(cache)/"reference-resolution"/"sources"/(envelope["sha256"]+".json")
            if not target.exists():atomic_json(target,envelope)
        result={"reference_version":dd,"communitydragon_version":cd,"sources":sources,"resolved_at":now()}
        atomic_json(Path(cache)/"reference-resolution"/(digest(result)+".json"),result)
    return dd,cd


def versions(dd=DD_VERSION, cd=CD_VERSION):
    if not re.fullmatch(r"\d+\.\d+\.\d+", dd) or cd != dd.rsplit(".", 1)[0]:
        raise ValueError("Explicit Data Dragon and matching CommunityDragon family required")
    return dd, cd


def cd_url(version, path):
    if not isinstance(path, str) or not path.startswith("/lol-game-data/assets/") or ".." in path:
        raise ValueError("Invalid CommunityDragon asset path")
    return f"{CD}/{version}/plugins/rcp-be-lol-game-data/global/default/" + path.split("/assets/", 1)[1].lower()


def image_asset(root, url, identity, offline=False):
    from PIL import Image
    key = hashlib.sha256(url.encode()).hexdigest()
    record_path = root/"downloads"/(key+".json")
    if record_path.exists():
        record = read_json(record_path)
        if record.get("url") != url:
            raise ValueError("Reference URL cache mismatch")
        asset_bytes(root, record)
        return {**record, **identity}
    if offline:
        raise ValueError("Reference asset missing offline: "+url)
    raw = download(url)
    source_hash = hashlib.sha256(raw).hexdigest()
    with Image.open(io.BytesIO(raw)) as im:
        im = im.convert("RGB")
        if min(im.size) < 16:
            raise ValueError("Reference portrait too small")
        artwork = hashlib.sha256(im.resize((48, 48)).tobytes()).hexdigest()
        encoded = io.BytesIO(); im.save(encoded, format="PNG")
    raw = encoded.getvalue(); sha = hashlib.sha256(raw).hexdigest()
    path = "images/"+sha+".png"
    (root/"images").mkdir(parents=True, exist_ok=True)
    write_icon(root/path, raw)
    record = {"url": url, "path": path, "sha256": sha, "source_sha256": source_hash,
              "artwork": artwork, "retrieved_at": now()}
    atomic_json(record_path, record)
    return {**record, **identity}


def skin_root(cache, dd, cd):
    versions(dd, cd)
    return Path(cache)/"skin-references"/(dd+"_"+cd)


def load_skins(root):
    data = read_json(root/"manifest.json")
    versions(data["reference_version"], data["communitydragon_version"])
    if data.get("status") != "complete" or not data.get("assets"):
        raise ValueError("Skin references are incomplete")
    for asset in data["assets"]:
        asset_bytes(root, asset)
    return data


def cache_skins(cache, dd=DD_VERSION, cd=CD_VERSION, *, offline=False):
    root = skin_root(cache, dd, cd)
    if (root/"manifest.json").exists():
        return {"root": str(root), **load_skins(root)}
    raw = JsonCache(root/"raw", offline=offline)
    roster, roster_source = raw.get(f"{DD}/cdn/{dd}/data/en_US/champion.json")
    if roster.get("version") != dd:
        raise ValueError("Skin roster version mismatch")
    sources, assets, supplements, failures = [roster_source], [], {}, {}

    def champion(pair):
        identifier, champion = pair
        result, provenance, errors = [], [], {}
        url = f"{CD}/{cd}/plugins/rcp-be-lol-game-data/global/default/v1/champions/{champion['key']}.json"
        try:
            data, source = raw.get(url)
            if str(data["id"]) != str(champion["key"]):
                raise ValueError("CommunityDragon champion ID mismatch")
            provenance.append(source)
        except (OSError, ValueError, RuntimeError) as exc:
            return identifier, [], [], {}, {url: str(exc)}
        descriptions = {s["spellKey"].upper(): {"name": s.get("name"),
            "description": plain_text(s.get("description", "")), "source": source}
            for s in data.get("spells", []) if s.get("spellKey")}
        for skin in data.get("skins", []):
            for key, shape in (("squarePortraitPath", "square"), ("tilePath", "square"),
                               ("circleIconPath", "round"), ("circlePortraitPath", "round")):
                if not skin.get(key):
                    continue
                image_url = cd_url(cd, skin[key])
                try:
                    result.append(image_asset(root, image_url, {"id": identifier, "kind": "champion",
                        "name": champion["name"], "skin_id": str(skin["id"]),
                        "skin_name": skin["name"], "shape": shape}, offline))
                except (OSError, ValueError, RuntimeError) as exc:
                    errors[image_url] = str(exc)
        # In-game circular portraits are not consistently referenced by LCU skin JSON.
        listing_url = f"{CD}/json/{cd}/game/assets/characters/{identifier.lower()}/hud/"
        try:
            listing, source = raw.get(listing_url); provenance.append(source)
            for entry in listing:
                name = entry.get("name", "")
                match = re.fullmatch(r"[a-z0-9_]+_circle(?:_skin|_)?(\d+)\.png", name, re.I)
                if not match:
                    continue
                number = int(match[1])
                skin = next((s for s in data.get("skins", []) if int(s["id"]) % 1000 == number), None)
                if skin:
                    image_url = f"{CD}/{cd}/game/assets/characters/{identifier.lower()}/hud/{name}"
                    result.append(image_asset(root, image_url, {"id": identifier, "kind": "champion",
                        "name": champion["name"], "skin_id": str(skin["id"]),
                        "skin_name": skin["name"], "shape": "round"}, offline))
        except (OSError, ValueError, RuntimeError) as exc:
            errors[listing_url] = str(exc)
        if not result:
            errors[identifier] = "No supported skin portraits found"
        return identifier, result, provenance, descriptions, errors

    with ThreadPoolExecutor(max_workers=4) as pool:
        for identifier, images, provenance, descriptions, errors in pool.map(champion, sorted(roster["data"].items())):
            assets.extend(images); sources.extend(provenance); supplements[identifier] = descriptions; failures.update(errors)
    # Deduplicate metadata aliases pointing at exactly the same image.
    assets = list({(a["id"], a["skin_id"], a["shape"], a["sha256"]): a for a in assets}.values())
    data = {"schema_version": 1, "reference_version": dd, "communitydragon_version": cd,
        "gameplay_patch": None, "mechanics_enabled": False, "created_at": now(), "sources": sources,
        "assets": assets, "ability_supplements": supplements, "failures": failures,
        "status": "partial" if failures else "complete"}
    atomic_json(root/("partial.json" if failures else "manifest.json"), data)
    return {"root": str(root), **data}


def cache_bundle(cache, metadata_cache, dd=DD_VERSION, cd=CD_VERSION, *, offline=False):
    from .ability_catalog import cache_abilities
    from .inventory_references import cache_inventory_references
    versions(dd, cd)
    abilities = cache_abilities(cache, dd, offline=offline)
    inventory = cache_inventory_references(cache, dd, metadata_cache, offline=offline)
    skins = cache_skins(cache, dd, cd, offline=offline)
    data = {"schema_version": 1, "reference_version": dd, "communitydragon_version": cd,
            "gameplay_patch": None, "created_at": now(), "abilities": abilities, "inventory": inventory,
            "skins": {k: skins[k] for k in ("root", "status", "failures")},
            "status": "complete" if abilities["status"] == "complete_all" and skins["status"] == "complete" else "partial"}
    data["catalog_hashes"] = {"abilities": digest(read_json(Path(abilities["catalog"]))),
                              "inventory": digest(read_json(Path(inventory["manifest"]))),
                              "skins": digest({k:v for k,v in skins.items() if k!="root"})}
    data["bundle_id"] = digest({"reference_version":dd,"communitydragon_version":cd,"catalog_hashes":data["catalog_hashes"]})
    path=Path(cache)/"reference-bundles"/(dd+"_"+cd)/(data["bundle_id"]+".json")
    if path.exists():return read_json(path)
    atomic_json(path, data)
    return data
