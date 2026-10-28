"""Patch-pinned binary identity references, independent of mechanics tooltips."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen

from .core import atomic_json, read_json, uid
from .knowledge import Patch, DD


def download(url):
    if (
        urlsplit(url).scheme != "https"
        or urlsplit(url).hostname != "ddragon.leagueoflegends.com"
    ):
        raise ValueError("Visual assets must come from the pinned Data Dragon host")
    with urlopen(url, timeout=40) as response:
        if urlsplit(response.url).hostname != "ddragon.leagueoflegends.com":
            raise ValueError("Asset redirected outside Data Dragon")
        raw = response.read(8 * 1024 * 1024 + 1)
        if len(raw) > 8 * 1024 * 1024:
            raise ValueError("Visual asset exceeds the download limit")
        return raw


def cache_assets(cache: Path, patch: dict) -> dict:
    from PIL import Image

    pin = Patch(**patch)
    cache.mkdir(parents=True, exist_ok=True)
    base = f"{DD}/cdn/{pin.ddragon_version}"
    metadata = cache / "metadata"
    metadata.mkdir(exist_ok=True)

    def data(name):
        path = metadata / (pin.ddragon_version + "-" + name.replace("/", "-"))
        if not path.exists():
            raw = download(f"{base}/data/en_US/{name}")
            value = json.loads(raw)
            atomic_json(path, value)
        return read_json(path)["data"]

    references = []
    champions = data("champion.json")
    for name, champion in champions.items():
        references.append(
            {
                "kind": "champion",
                "id": name,
                "file": "champion/" + champion["image"]["full"],
            }
        )
        detail = data("champion/" + name + ".json")[name]
        references.append(
            {
                "kind": "passive",
                "id": name + ":passive",
                "owner": name,
                "file": "passive/" + detail["passive"]["image"]["full"],
            }
        )
        for slot, spell in zip("QWER", detail["spells"]):
            references.append(
                {
                    "kind": "spell",
                    "id": name + ":" + slot,
                    "owner": name,
                    "file": "spell/" + spell["image"]["full"],
                }
            )
    for name, value in data("summoner.json").items():
        references.append(
            {"kind": "spell", "id": name, "file": "spell/" + value["image"]["full"]}
        )
    for name, value in data("item.json").items():
        if value.get("maps", {}).get("11", True):
            references.append(
                {"kind": "item", "id": name, "file": "item/" + value["image"]["full"]}
            )

    def fetch(reference):
        url = base + "/img/" + reference["file"]
        relative = (
            "images/"
            + pin.ddragon_version
            + "/"
            + uid("asset", reference["kind"], reference["id"])
            + ".png"
        )
        path = cache / relative
        if not path.exists():
            raw = download(url)
            image = Image.open(io.BytesIO(raw))
            image.verify()
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(raw)
            temporary.replace(path)
        raw = path.read_bytes()
        with Image.open(io.BytesIO(raw)) as image:
            if image.width < 16 or image.height < 16:
                raise ValueError("Reference image is too small")
        return {
            **reference,
            "url": url,
            "path": relative,
            "sha256": hashlib.sha256(raw).hexdigest(),
        }

    with ThreadPoolExecutor(max_workers=6) as pool:
        assets = list(pool.map(fetch, references))
    manifest = {"schema_version": 2, "patch": patch, "assets": assets}
    atomic_json(cache / "manifest.json", manifest)
    return {"assets": len(assets), "cache": str(cache), "patch": patch}


def verify_assets(cache: Path, patch: dict) -> dict:
    path = cache / "manifest.json"
    if not path.exists():
        raise ValueError(
            f"Visual assets missing at {cache}; run cache-assets on a login node"
        )
    manifest = read_json(path)
    if manifest.get("patch") != patch or not manifest.get("assets"):
        raise ValueError("Visual asset manifest does not match the requested patch")
    kinds = set()
    for asset in manifest["assets"]:
        image = (cache / asset["path"]).resolve()
        if not image.is_relative_to(cache.resolve()) or not image.is_file():
            raise ValueError("Visual asset missing or outside the cache")
        if hashlib.sha256(image.read_bytes()).hexdigest() != asset["sha256"]:
            raise ValueError(f"Visual asset hash mismatch: {image}")
        kinds.add(asset["kind"])
    if not {"champion", "item", "spell", "passive"} <= kinds:
        raise ValueError("Visual asset manifest lacks required identity categories")
    return manifest
