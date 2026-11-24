"""Render and audit static champion views for the skin and chroma pilot."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import re
import os
import struct
import tempfile

import numpy as np
from PIL import Image, ImageDraw

from .body_render import render_mask
from .body_static_skn import read_mesh
from .skin_metric_raster import implementation as raster_implementation, render_view
from .core import atomic_json, digest, implementation_hash, read_json
from .media import file_hash
from .skin_metric_assets import (AssetCache, PILOT, body_url, champion_rows, material_documents,
                                 skin_record)

VIEWS = (
    [("train", elevation, heading) for elevation in (45, 60) for heading in range(0, 360, 45)]
    + [("validation", 50, 22.5 + index * 90) for index in range(4)]
    + [("test", 55, 67.5 + index * 90) for index in range(4)]
)
SLOTS = ("maintexture", "diffusetexture", "diffusecolor", "difftex", "diffuse", "basetexture", "basecolortexture")
RENDER_FILES = ["body_static_skn.py", "body_render.py", "skin_metric_assets.py",
                "skin_metric_dataset.py", "skin_metric_raster.py"]


def renderer_hash():
    return digest((implementation_hash(RENDER_FILES), raster_implementation()[1], np.__version__))


class MissingAssets(ValueError):
    pass


def material_texture(material, documents):
    original = material
    seen = set()
    while material:
        key = material.casefold()
        if key in seen or key not in documents:
            break
        seen.add(key)
        record = documents[key]
        samplers = record.get("samplerValues", []) or []
        for slot in SLOTS:
            textures = {sample["texturePath"] for sample in samplers if sample.get("texturePath")
                        and re.sub(r"[^a-z0-9]", "", sample.get("TextureName", "").casefold()) == slot}
            if len(textures) == 1:
                return textures.pop()
            if len(textures) > 1:
                raise ValueError(f"Ambiguous diffuse sampler in {material}")
        material = record.get("parent", record.get("Parent"))
    raise ValueError(f"No supported diffuse sampler for {original}")


def material_bindings(properties, documents, visible):
    default = properties.get("texture")
    overrides = {}
    for override in properties.get("materialOverride", []) or []:
        name = override["submesh"].casefold()
        if name in visible:
            if name in overrides and overrides[name] != override:
                raise ValueError(f"Conflicting material overrides for {name}")
            overrides[name] = override
    bindings = {}
    for name in sorted(visible):
        override = overrides.get(name, {})
        texture = override.get("texture")
        if not texture and override.get("Material"):
            texture = material_texture(override["Material"], documents)
        if not texture:
            texture = default
        if not texture and properties.get("Material"):
            texture = material_texture(properties["Material"], documents)
        if not texture:
            raise ValueError(f"No diffuse texture for {name}")
        bindings[name] = body_url(texture)
    return bindings


def checked_file(cache, files, url, role):
    receipt = files.get((role, url))
    if not receipt or receipt["status"] != "cached":
        status = receipt["status"] if receipt else "not_discovered"
        raise MissingAssets(f"{role}: {status}: {url}")
    path = cache.path(receipt)
    if not path.is_file() or file_hash(path) != receipt["sha256"]:
        raise MissingAssets(f"Missing or changed cached {role}: {url}")
    return path, receipt["sha256"]


def prepare_skin(cache, row, downloaded):
    files = {(item["role"], item["url"]): item for item in downloaded["files"]}
    path, definition_hash = checked_file(cache, files, row["definition_url"], "definition")
    document = read_json(path)
    record_name, skin = skin_record(document, row["id"])
    properties = skin["skinMeshProperties"]
    if not properties.get("simpleSkin"):
        raise ValueError("Definition has no explicit body mesh")
    mesh_path, mesh_hash = checked_file(cache, files, body_url(properties["simpleSkin"]), "mesh")
    xyz, uv, groups = read_mesh(mesh_path)
    hidden = set(re.split(r"[\s,]+", properties.get("initialSubmeshToHide", "").casefold()))
    groups = {name: faces for name, faces in groups.items() if name not in hidden}
    if not groups:
        raise ValueError("No visible body submeshes")
    documents, linked = material_documents(cache, document, properties)
    bindings = material_bindings(properties, documents, groups)
    textures = {}
    diffuse_hashes = {}
    for name, url in bindings.items():
        if url not in textures:
            path, sha = checked_file(cache, files, url, "texture")
            with Image.open(path) as image:
                textures[url] = np.array(image.convert("RGBA")), sha
        diffuse_hashes[name] = {"url": url, "sha256": textures[url][1]}
    material_sources = [{"url": item["url"], "sha256": item["sha256"]} for item in linked
                        if item["status"] == "cached"]
    evidence = {"definition_sha256": definition_hash, "record": record_name, "mesh_sha256": mesh_hash,
                "linked_materials": material_sources, "diffuse_by_submesh": diffuse_hashes}
    return (xyz, uv, groups, {name: textures[url][0] for name, url in bindings.items()}), evidence


def save_image(image, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            image.save(stream, format="PNG", compress_level=3)
            stream.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def validate_views(root, views):
    geometry = [(row["split"], row["elevation"], row["heading"]) for row in views]
    return len(views) == len(VIEWS) and set(geometry) == set(VIEWS) and all(
        (root / row["path"]).is_file() and file_hash(root / row["path"]) == row["sha256"] for row in views)


def render_skin(root, cache, row, downloaded, renderer, prior):
    label = row["id"]
    inputs = digest((downloaded, VIEWS, renderer))
    if prior and prior["inputs_sha256"] == inputs and prior["status"] == "renderable" \
            and validate_views(root, prior["views"]):
        return prior
    result = {"id": label, "inputs_sha256": inputs}
    try:
        model, evidence = prepare_skin(cache, row, downloaded)
        views = []
        seen_pixels = []
        for split, elevation, heading in VIEWS:
            image = render_view(*model, heading, elevation)
            render_mask(image)
            relative = f"renders/{row['champion']}/{label}/{elevation}-{heading}.png"
            path = root / relative
            save_image(image, path)
            sha = file_hash(path)
            views.append({"split": split, "elevation": elevation, "heading": heading, "path": relative,
                          "sha256": sha})
            seen_pixels.append(sha)
        hashes_by_split = {split: {view["sha256"] for view in views if view["split"] == split}
                           for split in ("train", "validation", "test")}
        if sum(map(len, hashes_by_split.values())) != len(set(seen_pixels)):
            raise ValueError("Identical rendered image crosses the camera splits")
        result.update(status="renderable", views=views, assets=evidence,
                      appearance_sha256=digest(seen_pixels))
    except MissingAssets as exc:
        result.update(status="missing_assets", error=str(exc))
    except (ValueError, KeyError, OSError, TypeError, AttributeError, struct.error) as exc:
        result.update(status="unsupported", error=str(exc))
    return result


def render_champion(root, catalog, champion):
    roster = champion_rows(catalog, champion)
    download_path = root / "downloads" / f"{champion}.json"
    download = read_json(download_path)
    if download["inventory_sha256"] != digest(roster):
        raise ValueError(f"Stale download inventory for {champion}")
    receipts = {row["id"]: row for row in download["identities"]}
    if receipts.keys() != {row["id"] for row in roster}:
        raise ValueError(f"Incomplete download receipts for {champion}")
    path = root / "shards" / f"{champion}.json"
    renderer = renderer_hash()
    old = read_json(path) if path.exists() else {}
    prior = {row["id"]: row for row in old.get("identities", [])}
    cache = AssetCache(root / "assets", offline=True)
    for row in roster:
        prior[row["id"]] = render_skin(root, cache, row, receipts[row["id"]], renderer, prior.get(row["id"]))
        atomic_json(path, {"renderer_sha256": renderer, "download_sha256": digest(download),
                           "identities": list(prior.values())})
    print(champion, dict(Counter(row["status"] for row in prior.values())), flush=True)


def pilot_sheets(root, catalog):
    directory = root / "pilot"
    directory.mkdir(parents=True, exist_ok=True)
    champions = {}
    for champion in PILOT:
        shard = read_json(root / "shards" / f"{champion}.json")
        champions[champion] = digest(shard)
        lookup = {row["id"]: row for row in shard["identities"]}
        roster = sorted(champion_rows(catalog, champion), key=lambda row: (row["parent_skin_id"], row["id"]))
        for start in range(0, len(roster), 24):
            sheet = Image.new("RGB", (1080, 760), (44, 47, 52))
            draw = ImageDraw.Draw(sheet)
            draw.text((12, 8), f"{champion} • part {start // 24 + 1}", fill="white")
            for index, label in enumerate(roster[start:start + 24]):
                x, y = index % 6 * 180, index // 6 * 180 + 22
                row = lookup[label["id"]]
                draw.text((x + 5, y), f'{label["id"]} / {label["parent_skin_id"]}', fill="white")
                draw.text((x + 5, y + 16), label["name"][:26], fill="white")
                if row["status"] != "renderable":
                    draw.text((x + 5, y + 47), row["status"], fill="#f4adad")
                    continue
                for offset, view in enumerate((row["views"][0], row["views"][19])):
                    with Image.open(root / view["path"]) as source:
                        image = source.copy()
                    image.thumbnail((86, 132))
                    sheet.paste(image, (x + 2 + 88 * offset, y + 39), image)
            save_image(sheet, directory / f"{champion}-{start // 24 + 1:02}.png")
    atomic_json(directory / "sheets.json", {"shards": champions})


def approve_pilot(root, reviewer, reject):
    source = read_json(root / "pilot" / "sheets.json")
    checked = [read_json(root / "shards" / f"{champion}.json") for champion in PILOT]
    if source["shards"] != {champion: digest(shard) for champion, shard in zip(PILOT, checked, strict=True)}:
        raise ValueError("Pilot changed since sheets were generated")
    allowed = {row["id"] for shard in checked for row in shard["identities"] if row["status"] == "renderable"}
    rejected = set(reject)
    if not reviewer.strip() or not rejected <= allowed:
        raise ValueError("Review requires a reviewer and valid rejected render IDs")
    roster = {row["id"]: row for row in read_json(root / "inventory.json")["identities"]}
    qualified = {(roster[identity]["champion"], roster[identity]["kind"]) for identity in allowed - rejected}
    required = {(champion, kind) for champion in PILOT for kind in ("base", "chroma")}
    if not required <= qualified:
        raise ValueError(f"Pilot needs a valid base and chroma for every champion: {sorted(required - qualified)}")
    atomic_json(root / "pilot" / "review.json", {"reviewer": reviewer, **source, "rejected": sorted(rejected)})


def pilot_review(root):
    review = read_json(root / "pilot" / "review.json")
    for champion in PILOT:
        shard = read_json(root / "shards" / f"{champion}.json")
        if review["shards"][champion] != digest(shard) or shard["renderer_sha256"] != renderer_hash():
            raise ValueError("Pilot review does not match the current rendered pilot")
    return review


def merge_dataset(root, catalog, champions, reviewed):
    rejection = set(pilot_review(root)["rejected"]) if reviewed else set()
    identities = []
    signatures = {}
    renderer = renderer_hash()
    for champion in champions:
        shard = read_json(root / "shards" / f"{champion}.json")
        download = read_json(root / "downloads" / f"{champion}.json")
        roster = champion_rows(catalog, champion)
        if shard["renderer_sha256"] != renderer or shard["download_sha256"] != digest(download) \
                or download["inventory_sha256"] != digest(roster):
            raise ValueError(f"Stale render shard: {champion}")
        rendered = {row["id"]: row for row in shard["identities"]}
        if rendered.keys() != {row["id"] for row in roster} or len(rendered) != len(shard["identities"]):
            raise ValueError(f"Incomplete render shard: {champion}")
        for label in roster:
            row = rendered[label["id"]].copy()
            if row["id"] in rejection:
                row.update(status="unsupported", error="Rejected in pilot contact sheet review")
            if row["status"] == "renderable":
                if not validate_views(root, row["views"]):
                    raise ValueError(f"Invalid views for {row['id']}")
                signatures.setdefault(row["appearance_sha256"], []).append(row["id"])
            identities.append({**label, **row})
    duplicate = {identity: group for group in signatures.values() if len(group) > 1 for identity in group}
    for row in identities:
        if row["id"] in duplicate:
            row.update(status="visually_indistinguishable", equivalent_ids=duplicate[row["id"]])
    counts = {kind: dict(Counter(row["status"] for row in identities if row["kind"] == kind))
              for kind in ("base", "skin", "chroma")}
    dataset = {"schema": 1, "domain": "synthetic_bind_pose", "catalog_sha256": digest(catalog),
               "renderer_sha256": renderer, "counts": counts, "identities": identities}
    atomic_json(root / ("dataset.json" if reviewed else "pilot/dataset.json"), dataset)
    print(counts, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    render = sub.add_parser("render")
    selection = render.add_mutually_exclusive_group(required=True)
    selection.add_argument("--pilot", action="store_true")
    selection.add_argument("--champion")
    selection.add_argument("--index", type=int)
    sub.add_parser("sheets")
    approve = sub.add_parser("approve-pilot")
    approve.add_argument("--reviewer", required=True)
    approve.add_argument("--reject", type=int, action="append", default=[])
    merge = sub.add_parser("merge")
    merge.add_argument("--pilot-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    catalog = read_json(root / "inventory.json")
    champions = sorted({row["champion"] for row in catalog["identities"]})
    if args.command == "render":
        if args.index is not None:
            pilot_review(root)
            if not 0 <= args.index < len(champions):
                parser.error("Invalid champion shard index")
            selected = (champions[args.index],)
        else:
            selected = PILOT if args.pilot else (args.champion,)
        for champion in selected:
            render_champion(root, catalog, champion)
    elif args.command == "sheets":
        pilot_sheets(root, catalog)
    elif args.command == "approve-pilot":
        approve_pilot(root, args.reviewer, args.reject)
    else:
        merge_dataset(root, catalog, PILOT if args.pilot_only else champions, not args.pilot_only)


if __name__ == "__main__":
    main()
