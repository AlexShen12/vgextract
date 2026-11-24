"""Build static body references directly from CommunityDragon SKN meshes."""

import argparse
import math
from pathlib import Path
import struct

import numpy as np
from PIL import Image

from .body_assets import fetch_file
from .body_render import render_mask
from .core import atomic_json, digest, implementation_hash, read_json, uid
from .knowledge import CD, JsonCache
from .media import file_hash
from .reference_bundle import load_skins, skin_root, versions


def read_mesh(path):
    raw = path.read_bytes()
    magic, major, minor = struct.unpack_from("<IHH", raw)
    if magic != 0x00112233 or minor != 1 or major not in {0, 1, 2, 4}:
        raise ValueError("Unsupported SKN mesh version")
    ranges = []
    offset = 8
    if major:
        count, = struct.unpack_from("<I", raw, offset)
        offset += 4
        if not 0 < count <= (len(raw) - offset) // 80:
            raise ValueError("Invalid SKN submesh count")
        ranges = [struct.unpack_from("<64s4I", raw, offset + 80 * index) for index in range(count)]
        offset += 80 * count
    flags, stride = 0, 52
    if major == 4:
        flags, index_count, vertex_count, stride, vertex_type = struct.unpack_from("<5I", raw, offset)
        if flags & ~3 or {0: 52, 1: 56, 2: 72, 3: 104}.get(vertex_type) != stride:
            raise ValueError("Unsupported SKN flags or vertex layout")
        offset += 60
        if flags & 1:
            block_size, = struct.unpack_from("<H", raw, offset)
            offset += 2 + block_size
    else:
        index_count, vertex_count = struct.unpack_from("<2I", raw, offset)
        offset += 8
        if major == 0:
            ranges = [(b"base", 0, vertex_count, 0, index_count)]
    vertex_start = offset + index_count * 2
    if not index_count or not vertex_count or vertex_start + vertex_count * stride > len(raw):
        raise ValueError("Truncated SKN geometry")
    indices = np.frombuffer(raw, dtype="<u2", count=index_count, offset=offset)
    xyz = np.ndarray((vertex_count, 3), "<f4", raw, vertex_start, (stride, 4))
    uv = np.ndarray((vertex_count, 2), "<f4", raw, vertex_start + 44, (stride, 4))
    if not np.isfinite(xyz).all() or not np.isfinite(uv).all():
        raise ValueError("Nonfinite SKN vertex data")
    groups = {}
    for name, first_vertex, vertices, first_index, length in ranges:
        name = name.split(b"\0", 1)[0].decode().casefold()
        faces = indices[first_index:first_index + length]
        if flags & 2:
            faces = faces.astype(np.int64) + first_vertex
        if (not length or length % 3 or len(faces) != length or first_vertex + vertices > vertex_count
                or faces.min() < first_vertex or faces.max() >= first_vertex + vertices):
            raise ValueError(f"Invalid SKN submesh range for {name}")
        groups.setdefault(name, []).append(faces.reshape(-1, 3))
    return xyz, uv, {name: np.concatenate(faces) for name, faces in groups.items()}


def diffuse_textures(document, properties, visible_groups, reviewed_textures):
    textures = {name: properties.get("texture") for name in visible_groups}
    for override in properties.get("materialOverride", []) or []:
        name = override["submesh"].casefold()
        if name not in textures or name in reviewed_textures:
            continue
        path = override.get("texture")
        if not path and override.get("Material"):
            material = document.get(override["Material"], {})
            samplers = material.get("samplerValues", [])
            candidates = set()
            for slot in ("maintexture", "diffusetexture", "difftex"):
                candidates = {s["texturePath"] for s in samplers if s.get("texturePath")
                              and s.get("TextureName", "").casefold().replace("_", "") == slot}
                if candidates:
                    break
            if len(candidates) != 1:
                raise ValueError(f"Missing or ambiguous diffuse texture for {name}")
            path = candidates.pop()
        textures[name] = path or textures[name]
    if not reviewed_textures.keys() <= textures.keys():
        raise ValueError("Mesh review cites a texture for a hidden or unknown submesh")
    textures.update(reviewed_textures)
    if not all(textures.values()):
        raise ValueError("Visible submesh is missing a diffuse texture")
    return textures


def render_view(xyz, uv, groups, textures, heading, elevation, size=224):
    theta, phi = map(math.radians, (heading, elevation))
    view = np.array([math.sin(theta) * math.cos(phi), math.sin(phi), math.cos(theta) * math.cos(phi)])
    right = np.array([math.cos(theta), 0, -math.sin(theta)])
    up = np.cross(view, right)
    projected = xyz @ np.stack([right, -up, view], axis=1)
    used = np.concatenate([faces.ravel() for faces in groups.values()])
    lower, upper = projected[used, :2].min(axis=0), projected[used, :2].max(axis=0)
    span = max(upper - lower)
    if span <= 0:
        raise ValueError("SKN geometry projects to an empty view")
    projected[:, :2] = (projected[:, :2] - (upper + lower) / 2) * (.82 * size / span) + size / 2

    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    depth = np.full((size, size), -np.inf)
    for name, faces in groups.items():
        texture = textures[name]
        height, width = texture.shape[:2]
        for face in faces:
            a, b, c = projected[face]
            ab, ac = b - a, c - a
            det = ab[0] * ac[1] - ab[1] * ac[0]
            if abs(det) < 1e-8:
                continue
            left, top = np.floor(np.minimum.reduce([a, b, c])[:2]).astype(int)
            right_edge, bottom = np.ceil(np.maximum.reduce([a, b, c])[:2]).astype(int)
            y, x = np.ogrid[top:bottom, left:right_edge]
            dx, dy = x + .5 - a[0], y + .5 - a[1]
            wb = (dx * ac[1] - dy * ac[0]) / det
            wc = (ab[0] * dy - ab[1] * dx) / det
            z = a[2] + wb * ab[2] + wc * ac[2]
            region = depth[top:bottom, left:right_edge]
            rows, cols = np.nonzero((wb >= -1e-8) & (wc >= -1e-8) & (wb + wc <= 1 + 1e-8) & (z > region))
            if not len(rows):
                continue
            base, second, third = uv[face]
            coords = base + wb[rows, cols, None] * (second - base) + wc[rows, cols, None] * (third - base)
            # SKN UVs and the published texture PNG share a top-left origin.
            tx = np.clip((coords[:, 0] * width).astype(int), 0, width - 1)
            ty = np.clip((coords[:, 1] * height).astype(int), 0, height - 1)
            colors = texture[ty, tx]
            opaque = colors[:, 3] > 127
            rows, cols = rows[opaque], cols[opaque]
            region[rows, cols] = z[rows, cols]
            rgba[top + rows, left + cols] = colors[opaque]
    return Image.fromarray(rgba)


def render_skin(root, metadata, output, champion, skin_id, name, patch, offline, mesh_review):
    url = f"{CD}/{patch}/game/data/characters/{champion.lower()}/skins/skin{skin_id % 1000}.bin.json"
    document, source = metadata.get(url)
    properties = next(row["skinMeshProperties"] for row in document.values()
                      if isinstance(row, dict) and row.get("championSkinId") == skin_id)
    mesh = fetch_file(root, patch, properties["simpleSkin"], "mesh", offline)
    xyz, uv, groups = read_mesh(Path(mesh["path"]))
    hidden = set(properties.get("initialSubmeshToHide", "").casefold().split())
    if mesh_review:
        if mesh_review["mesh_sha256"] != mesh["sha256"]:
            raise ValueError("Mesh review applies to different geometry")
        extra_hidden = set(mesh_review.get("exclude_submeshes", []))
        if not extra_hidden <= groups.keys():
            raise ValueError("Mesh review cites an unknown submesh")
        hidden |= extra_hidden
    groups = {key: faces for key, faces in groups.items() if key not in hidden}
    if mesh_review and "x_range" in mesh_review:
        minimum, maximum = mesh_review["x_range"]
        retained = {}
        for key, faces in groups.items():
            within = (xyz[faces, 0] >= minimum) & (xyz[faces, 0] <= maximum)
            if np.any(within.any(axis=1) & ~within.all(axis=1)):
                raise ValueError(f"Mesh review would cut across faces in {key}")
            body = faces[within.all(axis=1)]
            if len(body):
                retained[key] = body
        groups = retained
    if not groups:
        raise ValueError("No visible SKN submeshes")
    bindings = diffuse_textures(document, properties, groups, (mesh_review or {}).get("textures_by_submesh", {}))
    texture_files = {path: fetch_file(root, patch, path, "texture", offline) for path in sorted(set(bindings.values()))}
    images = {}
    for path, asset in texture_files.items():
        with Image.open(asset["path"]) as image:
            images[path] = np.array(image.convert("RGBA"))
    textures = {key: images[path] for key, path in bindings.items()}
    assets = {"metadata": source, "mesh": mesh, "textures": texture_files, "diffuse_by_submesh": bindings,
              "hidden_submeshes": sorted(hidden), "mesh_review": mesh_review}
    target = output / champion / str(skin_id)
    target.mkdir(parents=True, exist_ok=True)
    renders = []
    for elevation in (45, 60):
        for heading in range(0, 360, 45):
            image = render_view(xyz, uv, groups, textures, heading, elevation)
            mask = render_mask(image)
            path = target / f"{elevation}-{heading}.png"
            mask_path = target / f"{elevation}-{heading}-mask.png"
            image.save(path)
            mask.save(mask_path)
            sha, mask_sha = file_hash(path), file_hash(mask_path)
            renders.append({"id": uid("body-render", patch, champion, skin_id, elevation, heading, sha, mask_sha),
                            "champion": champion, "skin_id": str(skin_id), "skin_name": name,
                            "reference_kind": "rendered", "pose_kind": "bind_pose", "heading": heading,
                            "elevation": elevation, "path": str(path), "mask_path": str(mask_path),
                            "sha256": sha, "mask_sha256": mask_sha, "source_assets": assets})
    return renders


def build(cache, output, champions, skins, dd, cd, offline, mesh_reviews, *, all_skins=()):
    versions(dd, cd)
    catalog = load_skins(skin_root(cache, dd, cd))
    if (catalog["reference_version"], catalog["communitydragon_version"]) != (dd, cd):
        raise ValueError("Skin catalog does not match the requested patch")
    assets = {(row["id"], int(row["skin_id"])): row["skin_name"] for row in catalog["assets"]
              if row["kind"] == "champion"}
    requests = set(skins)
    for champion in all_skins:
        available = {pair for pair in assets if pair[0] == champion}
        if not available:
            raise ValueError(f"No skins in the pinned catalog for {champion}")
        requests.update(available)
    for champion in champions:
        base = [pair for pair in assets if pair[0] == champion and pair[1] % 1000 == 0]
        if len(base) != 1:
            raise ValueError(f"No unique base skin in the pinned catalog for {champion}")
        requests.add(base[0])
    if not requests or not requests <= assets.keys():
        raise ValueError("Choose skin IDs belonging to their champions in the pinned catalog")
    output = output.resolve()
    root = cache / "body-assets" / f"{dd}_{cd}"
    metadata = JsonCache(root / "raw", offline=offline)
    report = {"tools": {"renderer": "static_skn", "implementation": implementation_hash(["body_static_skn.py", "body_render.py"])},
              "reference_version": dd, "communitydragon_version": cd, "skin_catalog_sha256": digest(catalog),
              "supports_animation": False, "skins": []}
    for champion, skin_id in sorted(requests):
        record = {"champion": champion, "skin_id": str(skin_id)}
        try:
            rendered = render_skin(root, metadata, output, champion, skin_id, assets[champion, skin_id], cd, offline,
                                   mesh_reviews.get(str(skin_id)))
            record.update(status="needs_visual_audit", renders=rendered)
        except (OSError, RuntimeError, ValueError, KeyError, StopIteration, struct.error) as exc:
            record.update(status="unsupported", error=str(exc))
        report["skins"].append(record)
        atomic_json(output / "render-report.json", report)
    for record in report["skins"]:
        print(record["champion"], record["skin_id"], record.get("error", f'{len(record.get("renders", []))} views'), flush=True)
    return all(record["status"] == "needs_visual_audit" for record in report["skins"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("data/league_assets"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--champion", action="append", default=[], help="Champion whose base skin to render")
    parser.add_argument("--all-skins", action="append", default=[], metavar="CHAMPION",
                        help="Render every unique skin ID for this champion in the pinned catalog")
    parser.add_argument("--skin", action="append", nargs=2, default=[], metavar=("CHAMPION", "SKIN_ID"))
    parser.add_argument("--reference-version", default="16.22.1")
    parser.add_argument("--communitydragon-version", default="16.22")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--mesh-review", type=Path, help="Optional per-skin mesh exclusions tied to asset hashes")
    args = parser.parse_args()
    mesh_reviews = read_json(args.mesh_review) if args.mesh_review else {}
    success = build(args.cache, args.output, args.champion, [(name, int(skin)) for name, skin in args.skin],
                    args.reference_version, args.communitydragon_version, args.offline, mesh_reviews, all_skins=args.all_skins)
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
