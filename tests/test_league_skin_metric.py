import hashlib
import io
from json import dumps
import struct
from urllib.error import HTTPError

import numpy as np
from PIL import Image
import pytest

from annotation.league import skin_metric_assets as assets
from annotation.league import skin_metric_dataset as dataset
from annotation.league import skin_metric_raster as raster
from annotation.league.core import atomic_json, read_json


def put(cache, url, content, role):
    sha = hashlib.sha256(content).hexdigest()
    receipt = {"url": url, "sha256": sha, "status": "cached", "bytes": len(content)}
    destination = cache.path(receipt)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    key = hashlib.sha256(url.encode()).hexdigest()
    atomic_json(cache.root / "urls" / key[:2] / f"{key}.json", receipt)
    return {"role": role, **receipt}


def tetrahedron():
    vertices = bytearray(4 * 52)
    positions = ((0, 0, 0), (2, 0, 0), (0, 3, 0), (0, 0, 4))
    for i, position in enumerate(positions):
        struct.pack_into("<3f", vertices, 52 * i, *position)
        struct.pack_into("<2f", vertices, 52 * i + 44, (i % 2) * .9, (i // 2) * .9)
    return (struct.pack("<IHHII", 0x00112233, 0, 1, 12, 4)
            + struct.pack("<12H", 0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3) + vertices)


def test_download_distinguishes_missing_and_retryable(tmp_path, monkeypatch):
    cache = assets.AssetCache(tmp_path)
    calls = []
    monkeypatch.setattr(assets, "sleep", lambda _: None)

    def download(url):
        calls.append(url)
        code = 503 if "retry" in url else 404
        raise HTTPError(url, code, "fixture", None, None)

    monkeypatch.setattr(cache, "download", download)
    missing = assets.GAME + "missing"
    retry = assets.GAME + "retry"
    assert cache.get(missing)["status"] == "missing"
    assert cache.get(missing)["status"] == "missing"
    assert cache.get(retry)["status"] == "transient_error"
    assert cache.get(retry)["status"] == "transient_error"
    assert calls.count(missing) == 1
    assert calls.count(retry) == 6


def test_submesh_diffuse_binding_uses_exact_override_and_explicit_parent():
    properties = {"texture": "assets/base.tex", "materialOverride": [
        {"submesh": "Wings", "texture": "assets/chroma_wings.tex"},
        {"submesh": "Weapon", "Material": "Child"},
    ]}
    document = {"child": {"parent": "Parent"}, "parent": {"samplerValues": [
        {"TextureName": "Diffuse_Color", "texturePath": "assets/shared_weapon.tex"},
        {"TextureName": "NormalTexture", "texturePath": "assets/normal.tex"},
    ]}}
    actual = dataset.material_bindings(properties, document, {"body", "wings", "weapon"})
    assert actual == {key: assets.body_url(path) for key, path in {
        "body": "assets/base.tex", "wings": "assets/chroma_wings.tex", "weapon": "assets/shared_weapon.tex",
    }.items()}
    with pytest.raises(ValueError, match="No supported diffuse"):
        dataset.material_bindings(properties, {}, {"weapon"})


def test_accelerated_raster_matches_alpha_textured_reference(tmp_path):
    mesh = tmp_path / "body.skn"
    mesh.write_bytes(tetrahedron())
    xyz, uv, groups = dataset.read_mesh(mesh)
    texture = np.random.default_rng(7).integers(0, 256, size=(16, 16, 4), dtype=np.uint8)
    for heading in (0, 45, 90, 112.5, 180, 292.5):
        args = (xyz, uv, groups, {"base": texture}, heading, 50, 48)
        np.testing.assert_array_equal(raster.accelerated_render(*args, raster.rasterize),
                                      raster.reference_render(*args))


def test_render_resumes_and_full_merge_excludes_duplicate_appearances(tmp_path, monkeypatch):
    root = tmp_path
    cache = assets.AssetCache(root / "assets", offline=True)
    mesh = put(cache, assets.body_url("assets/mesh.skn"), tetrahedron(), "mesh")
    buffer = io.BytesIO()
    Image.new("RGBA", (4, 4), (9, 77, 150, 255)).save(buffer, "PNG")
    texture = put(cache, assets.body_url("assets/my_color.tex"), buffer.getvalue(), "texture")
    roster = []
    downloads = []
    records = []
    for identity, kind in ((1000, "base"), (1001, "chroma"), (1002, "chroma")):
        url = assets.skin_url("Example", identity - 1000)
        row = {"id": identity, "kind": kind, "champion": "Example", "parent_skin_id": 1000, "definition_url": url}
        roster.append(row)
        files = []
        if identity != 1002:
            doc = {"championSkinId": identity, "skinMeshProperties": {
                "simpleSkin": "assets/mesh.skn", "texture": "assets/my_color.tex"}}
            definition = put(cache, url, dumps({"skin": doc}).encode(), "definition")
            files = [definition, mesh, texture]
        downloaded = {"id": identity, "files": files}
        downloads.append(downloaded)
        renderer = dataset.renderer_hash()
        result = dataset.render_skin(root, cache, row, downloaded, renderer, None)
        records.append(result)
        if files:
            assert result["status"] == "renderable"
            assert dataset.validate_views(root, result["views"])
            assert result["assets"]["diffuse_by_submesh"]["base"]["sha256"] == texture["sha256"]
        else:
            assert result["status"] == "missing_assets"
    with monkeypatch.context() as patch:
        def unexpected_rerender(*_):
            pytest.fail("Verified render should be reused")
        patch.setattr(dataset, "render_view", unexpected_rerender)
        result = dataset.render_skin(root, cache, roster[0], downloads[0], renderer, records[0])
        assert result == records[0]
    downloaded = {"inventory_sha256": dataset.digest(roster), "identities": downloads}
    atomic_json(root / "downloads/Example.json", downloaded)
    atomic_json(root / "shards/Example.json",
                {"download_sha256": dataset.digest(downloaded), "renderer_sha256": renderer, "identities": records})
    catalog = {"identities": roster}
    dataset.merge_dataset(root, catalog, ("Example",), reviewed=False)
    coverage = read_json(root / "pilot/dataset.json")
    assert [row["status"] for row in coverage["identities"]] == [
        "visually_indistinguishable", "visually_indistinguishable", "missing_assets"]


def test_hard_batches_keep_equal_budget_and_checkpoint_reloads(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from annotation.league import skin_metric_train as training

    torch.set_num_threads(1)
    identities = [
        {"id": index, "kind": "base" if index % 2 == 0 else "chroma",
         "champion": f"Champion{index // 2}", "parent_skin_id": index // 2, "name": f"Variant {index}",
         "status": "renderable", "views": []}
        for index in range(4)
    ]
    neighbors = [[candidate for candidate in range(4) if candidate // 2 != index // 2] for index in range(4)]
    batches = list(training.training_batches(identities, 2, 2, neighbors))
    assert [len(batch) for batch in batches] == [8] * 8
    assert sorted(item for batch in batches for item in batch) == list(range(64))
    for batch in batches:
        left, right = sorted({item // 16 for item in batch})
        assert identities[left]["parent_skin_id"] == identities[right]["parent_skin_id"]
    for index, row in enumerate(identities):
        image = Image.new("RGBA", (15, 20), (50 * index + 20, 200 - 23 * index, 30 * index, 255))
        relative = f"crop-{index}.png"
        image.save(tmp_path / relative)
        row["views"] = [{"split": split, "path": relative} for split, _, _ in dataset.VIEWS]
    source = tmp_path / "pilot/dataset.json"
    atomic_json(source, {"identities": identities})
    training.train(tmp_path, source, "hard", "cpu", 2, 0, 2, scratch=True)
    out = tmp_path / "pilot/runs/hard"
    metrics = read_json(out / "metrics.json")
    assert metrics["reload_verified"] is True
    assert metrics["test"]["metrics"]["all"]["queries"] == 16
    assert all(row["views"] == 64 for row in read_json(out / "history.json"))
    training.train(tmp_path, source, "hard", "cpu", 2, 0, 2, scratch=True)
    assert read_json(out / "metrics.json") == metrics
    training.train(tmp_path, source, "random", "cpu", 2, 0, 2, scratch=True)
    (tmp_path / "runs").symlink_to(tmp_path / "pilot/runs", target_is_directory=True)
    training.compare(tmp_path)
    comparison = read_json(tmp_path / "comparison.json")
    assert comparison["runs"]["hard"] == metrics
