"""Versioned visual references for inventory recognition and offline review."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
from pathlib import Path
import re

from .core import atomic_json, digest, now, read_json
from .knowledge import DD, JsonCache, plain_text
from .visual_assets import download


REFERENCE_VERSION = "16.22.1"


def reference_root(cache, version):
    if not isinstance(version, str) or not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError("Inventory references require an explicit Data Dragon version")
    return Path(cache) / "inventory-references" / version


def asset_bytes(root, asset):
    path = (root / asset["path"]).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("Missing or invalid inventory reference path")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != asset["sha256"]:
        raise ValueError("Inventory reference icon hash mismatch")
    return raw


def load_inventory_references(root, version):
    path = root / "manifest.json"
    if not path.is_file():
        raise ValueError(f"Inventory references missing at {root}. Run cache-inventory-references first.")
    manifest = read_json(path)
    if manifest.get("schema_version") != 2 or manifest.get("reference_version") != version:
        raise ValueError("Inventory reference manifest version mismatch")
    keys = set()
    for asset in manifest["assets"]:
        key = (asset["kind"], asset["id"])
        if key in keys or not all(isinstance(asset[k], str) for k in ("id", "name", "artwork")):
            raise ValueError("Invalid or duplicate inventory reference")
        keys.add(key)
        asset_bytes(root, asset)
    if {kind for kind, _ in keys} != {"champion", "item"}:
        raise ValueError("Inventory references require champion portraits and item icons")
    return manifest


def cache_inventory_references(cache, version, metadata_cache, *, offline=False):
    from PIL import Image

    cache = Path(cache)
    root = reference_root(cache, version)
    if (root / "manifest.json").exists():
        manifest = load_inventory_references(root, version)
        return {"manifest": str(root / "manifest.json"), "assets": len(manifest["assets"])}

    legacy_path = cache / "manifest.json"
    legacy = read_json(legacy_path) if legacy_path.exists() else {}
    legacy_assets = {
        entry["url"]: entry for entry in legacy.get("assets", []) if "url" in entry
    }
    base = f"{DD}/cdn/{version}"
    references, sources = [], []
    json_cache = JsonCache(Path(metadata_cache), offline=offline)
    for kind in ("champion", "item"):
        document, source = json_cache.get(f"{base}/data/en_US/{kind}.json")
        sources.append(source)
        if document.get("version") != version:
            raise ValueError("Inventory metadata version mismatch")
        for identifier, record in sorted(document["data"].items()):
            if kind == "item" and not record.get("maps", {}).get("11", True):
                continue
            filename = record["image"]["full"]
            if not re.fullmatch(r"[A-Za-z0-9_-]+\.png", filename):
                raise ValueError("Invalid Data Dragon icon filename")
            references.append({
                "kind": kind, "id": identifier, "name": plain_text(record["name"]),
                "description": plain_text(record.get("description", "")).strip(),
                "url": f"{base}/img/{kind}/{filename}",
            })
            if kind == "item":
                if record.get("hideFromAll") or (record.get("inStore") is False and not references[-1]["description"]):
                    references[-1]["slot_class"] = "internal"
                elif "Trinket" in record.get("tags", []):
                    references[-1]["slot_class"] = "trinket"

    (root / "images").mkdir(parents=True, exist_ok=True)

    def fetch(reference):
        url = reference["url"]
        if url in legacy_assets:
            raw = asset_bytes(cache, legacy_assets[url])
        else:
            from .reference_bundle import image_asset
            return image_asset(root, url, reference, offline)
        with Image.open(io.BytesIO(raw)) as source:
            image = source.convert("RGB")
        if min(image.size) < 16:
            raise ValueError("Inventory reference icon is too small")
        artwork = hashlib.sha256(image.resize((48, 48)).tobytes()).hexdigest()
        encoded = io.BytesIO()
        image.save(encoded, format="PNG")
        raw = encoded.getvalue()
        sha256 = hashlib.sha256(raw).hexdigest()
        path = f"images/{sha256}.png"
        write_icon(root / path, raw)
        return {**reference, "artwork": artwork, "path": path, "sha256": sha256}

    with ThreadPoolExecutor(max_workers=6) as pool:
        assets = list(pool.map(fetch, references))
    manifest = {"schema_version": 2, "reference_version": version, "created_at": now(),
                "gameplay_patch": None, "sources": sources, "assets": assets}
    atomic_json(root / "manifest.json", manifest)
    return {"manifest": str(root / "manifest.json"), "assets": len(assets)}


def write_icon(path, raw):
    from tempfile import NamedTemporaryFile

    if path.exists() and path.read_bytes() == raw:
        return
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(raw)
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def publish_catalog(store, root, manifest):
    catalog_id = digest(manifest)
    output = store.path(f"inventory-catalogs/{catalog_id}")
    if not (output / "manifest.json").exists():
        (output / "images").mkdir(parents=True, exist_ok=True)
        items = [asset for asset in manifest["assets"] if asset["kind"] == "item"]
        for asset in items:
            write_icon(output / asset["path"], asset_bytes(root, asset))
        champions = {asset["id"]: asset["name"] for asset in manifest["assets"] if asset["kind"] == "champion"}
        atomic_json(output / "manifest.json", {"reference_version": manifest["reference_version"], "items": items,
                                               "champions": champions})
    return catalog_id
