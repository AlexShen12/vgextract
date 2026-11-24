"""Build and fetch the patch-pinned champion body inventory for metric learning."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import hashlib
from http.client import HTTPException
import json
from pathlib import Path
import re
import tempfile
from threading import Lock
from time import sleep
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from .body_assets import strings
from .core import atomic_json, digest, read_json
from .knowledge import CD, DD, JsonCache
from .media import file_hash

DD_PATCH = "16.22.1"
CD_PATCH = "16.22"
PILOT = ("Aatrox", "Ahri", "Lux")
EXPECTED = {"base": 173, "skin": 1964, "chroma": 7126}
DATA = f"{CD}/{CD_PATCH}/game/data/characters"
GAME = f"{CD}/{CD_PATCH}/game/"
ANIMATION = re.compile(r"characters/([a-z0-9_]+)/animations/skin(\d+)$", re.I)
MATERIAL = re.compile(r"characters/([a-z0-9_]+)/skins/skin(\d+)/materials/", re.I)
SKIN_KEY = re.compile(r"characters/([a-z0-9_]+)/skins/skin(\d+)$", re.I)


def inventory(cache):
    raw = JsonCache(cache / "raw", offline=True)
    roster, source = raw.get(f"{DD}/cdn/{DD_PATCH}/data/en_US/champion.json")
    if roster["version"] != DD_PATCH:
        raise ValueError("Unexpected Data Dragon version")
    rows = []
    for champion, metadata in sorted(roster["data"].items()):
        key = int(metadata["key"])
        url = f"{CD}/{CD_PATCH}/plugins/rcp-be-lol-game-data/global/default/v1/champions/{key}.json"
        catalog, receipt = raw.get(url)
        if catalog["id"] != key:
            raise ValueError(f"Wrong catalog for {champion}")
        base_count = 0
        for skin in catalog["skins"]:
            sid = skin["id"]
            base_count += sid == key * 1000
            variants = [(skin, "base" if sid == key * 1000 else "skin")]
            variants.extend((chroma, "chroma") for chroma in skin.get("chromas") or [])
            for variant, kind in variants:
                identity = variant["id"]
                if identity // 1000 != key or identity == sid and kind == "chroma":
                    raise ValueError(f"Invalid identity {identity} for {champion}")
                rows.append({"id": identity, "parent_skin_id": sid, "kind": kind, "name": variant["name"],
                             "champion": champion, "champion_id": key, "catalog": receipt,
                             "definition_url": skin_url(champion, identity % 1000)})
        if base_count != 1:
            raise ValueError(f"Expected one base skin for {champion}")
    counts = Counter(row["kind"] for row in rows)
    if counts != EXPECTED or len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"Pinned identity inventory changed: {counts}")
    return {"schema": 1, "data_dragon": DD_PATCH, "community_dragon": CD_PATCH,
            "roster": source, "counts": dict(counts), "identities": rows}


def skin_url(champion, number):
    return f"{DATA}/{champion.lower()}/skins/skin{number}.bin.json"


def normalized_path(path):
    path = path.lower().replace("\\", "/")
    if not re.fullmatch(r"assets/[a-z0-9_./ -]+", path) or ".." in Path(path).parts:
        raise ValueError(f"Invalid body asset path: {path}")
    if path.endswith((".tex", ".dds")):
        path = path.rsplit(".", 1)[0] + ".png"
    return path


def body_url(path):
    return GAME + quote(normalized_path(path), safe="/")


class AssetCache:
    """Keep portable URL receipts and store each successful payload once by SHA-256."""

    def __init__(self, root, offline=False):
        self.root = root
        self.offline = offline
        self.locks = [Lock() for _ in range(127)]

    def path(self, receipt):
        sha = receipt["sha256"]
        if not re.fullmatch(r"[a-f0-9]{64}", sha):
            raise ValueError("Invalid asset checksum")
        return self.root / "objects" / sha[:2] / sha

    def get(self, url):
        if not url.startswith(GAME):
            raise ValueError(f"Unexpected asset URL: {url}")
        key = hashlib.sha256(url.encode()).hexdigest()
        target = self.root / "urls" / key[:2] / f"{key}.json"
        with self.locks[int(key, 16) % len(self.locks)]:
            if target.exists():
                receipt = read_json(target)
                if receipt["url"] != url:
                    raise ValueError(f"Wrong URL receipt: {target}")
                if receipt["status"] == "missing":
                    return receipt
                cached = self.path(receipt)
                if cached.is_file() and file_hash(cached) == receipt["sha256"]:
                    return receipt
            if self.offline:
                return {"url": url, "status": "not_cached"}
            for attempt in range(3):
                try:
                    result = self.download(url)
                    atomic_json(target, result)
                    return result
                except HTTPError as exc:
                    if exc.code in (404, 410):
                        result = {"url": url, "status": "missing", "http_status": exc.code}
                        atomic_json(target, result)
                        return result
                    if exc.code not in (408, 429, 500, 502, 503, 504):
                        return {"url": url, "status": "error", "error": str(exc)}
                    error = str(exc)
                except (URLError, TimeoutError, ConnectionError, HTTPException) as exc:
                    error = str(exc)
                except ValueError as exc:
                    return {"url": url, "status": "error", "error": str(exc)}
                if attempt < 2:
                    sleep(2 ** attempt)
            return {"url": url, "status": "transient_error", "error": error}

    def download(self, url):
        request = Request(url, headers={"User-Agent": "league-body-dataset/1.0"})
        with urlopen(request, timeout=35) as response:
            if response.url != url:
                raise ValueError(f"Unexpected asset redirect: {response.url}")
            data = response.read(64 * 1024 * 1024 + 1)
        if not data or len(data) > 64 * 1024 * 1024 or data.lstrip().startswith(b"<"):
            raise ValueError("Empty, oversized, or HTML asset response")
        if url.endswith(".json"):
            json.loads(data)
        receipt = {"url": url, "status": "cached", "sha256": hashlib.sha256(data).hexdigest(),
                   "bytes": len(data)}
        path = self.path(receipt)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                stream.write(data)
                stream.close()
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        return receipt

    @lru_cache(maxsize=16)
    def definition(self, url):
        receipt = self.get(url)
        return (read_json(self.path(receipt)) if receipt["status"] == "cached" else {}), receipt


def skin_record(document, identity):
    matches = [(name, value) for name, value in document.items()
               if isinstance(value, dict) and value.get("championSkinId") == identity
               and "skinMeshProperties" in value]
    if len(matches) != 1:
        raise ValueError(f"Expected one mesh record for {identity}, found {len(matches)}")
    return matches[0]


def material_nodes(document, properties):
    """Follow named material links from the body mesh, excluding unrelated VFX records."""
    seen = set()
    links = [s for s in strings(properties) if s.casefold() in document]
    while links:
        key = links.pop().casefold()
        if key in seen:
            continue
        seen.add(key)
        record = document[key]
        yield record
        links.extend(s for s in strings(record) if s.casefold() in document)


def animation_assets(cache, graph):
    pending = [graph] if graph else []
    seen = set()
    while pending:
        graph = pending.pop()
        if graph.casefold() in seen:
            continue
        seen.add(graph.casefold())
        match = ANIMATION.fullmatch(graph)
        if not match:
            continue
        champion, number = match.groups()
        url = f"{DATA}/{champion.lower()}/animations/skin{int(number)}.bin.json"
        document, receipt = cache.definition(url)
        yield {"role": "animation_graph", **receipt}
        if not document:
            continue
        yield {"role": "animation_definition", **cache.get(url.removesuffix(".json"))}
        for value in sorted(set(strings(document))):
            if value.lower().endswith(".anm") and value.lower().startswith("assets/"):
                yield {"role": "animation", **cache.get(body_url(value))}
            elif ANIMATION.fullmatch(value):
                pending.append(value)


def material_documents(cache, document, properties):
    merged = {key.casefold(): value for key, value in document.items()}
    receipts = []
    seen = set()
    pending = list(strings(properties))
    while pending:
        link = pending.pop().casefold()
        if link in seen:
            continue
        seen.add(link)
        if link in merged:
            pending.extend(strings(merged[link]))
            continue
        match = MATERIAL.match(link)
        if not match:
            continue
        alias, number = match.groups()
        other, receipt = cache.definition(skin_url(alias, int(number)))
        receipts.append(receipt)
        for key, value in other.items():
            merged.setdefault(key.casefold(), value)
        if link in merged:
            pending.extend(strings(merged[link]))
    return merged, receipts


def classify(row, cache):
    files = []
    document, receipt = cache.definition(row["definition_url"])
    files.append({"role": "definition", **receipt})
    if receipt["status"] != "cached":
        return {"id": row["id"], "status": receipt["status"], "files": files}
    files.append({"role": "definition_binary", **cache.get(row["definition_url"].removesuffix(".json"))})
    try:
        _, skin = skin_record(document, row["id"])
        properties = skin["skinMeshProperties"]
        documents, linked = material_documents(cache, document, properties)
        files.extend({"role": "material_definition", **item} for item in linked)
        mesh = properties.get("simpleSkin")
        skeleton = properties.get("skeleton")
        if mesh:
            files.append({"role": "mesh", **cache.get(body_url(mesh))})
        if skeleton:
            files.append({"role": "skeleton", **cache.get(body_url(skeleton))})
        textures = {s for node in (properties, *material_nodes(documents, properties)) for s in strings(node)
                    if s.lower().startswith("assets/") and s.lower().endswith((".tex", ".dds", ".png"))}
        for url in sorted({body_url(path) for path in textures}):
            files.append({"role": "texture", **cache.get(url)})
        graph = skin.get("skinAnimationProperties", {}).get("animationGraphData")
        files.extend(animation_assets(cache, graph))
        status = "discovered" if mesh else "unsupported"
        error = None if mesh else "Definition has no explicit body mesh"
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        status, error = "unsupported", str(exc)
    unique = {(file["role"], file["url"]): file for file in files}
    return {"id": row["id"], "status": status, "error": error, "files": list(unique.values())}


def champion_rows(catalog, champion):
    rows = [row for row in catalog["identities"] if row["champion"] == champion]
    if not rows:
        raise ValueError(f"No identities for {champion}")
    return rows


def fetch_champion(root, catalog, cache, champion):
    path = root / "downloads" / f"{champion}.json"
    rows = champion_rows(catalog, champion)
    catalog_sha = digest(rows)
    previous = read_json(path) if path.exists() else {}
    if previous and previous["inventory_sha256"] != catalog_sha:
        raise ValueError(f"Download inventory changed for {champion}")
    prior = {row["id"]: row for row in previous.get("identities", [])}
    for row in rows:
        existing = prior.get(row["id"])
        if existing and existing["status"] in ("discovered", "missing", "unsupported") and all(
            file["status"] in ("cached", "missing") and cache.get(file["url"]) == {
                k: v for k, v in file.items() if k != "role"} for file in existing["files"]
        ):
            continue
        prior[row["id"]] = classify(row, cache)
        atomic_json(path, {"inventory_sha256": catalog_sha, "identities": list(prior.values())})
    print(champion, dict(Counter(row["status"] for row in prior.values())), flush=True)
    return sum(file["status"] in ("transient_error", "error", "not_cached")
               for row in prior.values() for file in row["files"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("inventory")
    build.add_argument("--catalog-cache", type=Path, required=True)
    fetch = sub.add_parser("fetch")
    fetch.add_argument("--champion", action="append")
    fetch.add_argument("--workers", type=int, default=4, choices=range(1, 9))
    fetch.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "inventory":
        data = inventory(args.catalog_cache)
        atomic_json(root / "inventory.json", data)
        print(data["counts"])
        return
    catalog = read_json(root / "inventory.json")
    champions = args.champion or sorted({row["champion"] for row in catalog["identities"]})
    if len(set(champions)) != len(champions):
        parser.error("Champion filters must be unique")
    cache = AssetCache(root / "assets", offline=args.offline)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        errors = sum(pool.map(lambda name: fetch_champion(root, catalog, cache, name), champions))
    if errors:
        raise SystemExit(f"{errors} asset requests need repair or retry")


if __name__ == "__main__":
    main()
