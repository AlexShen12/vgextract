"""Shared calibrated-intake fixture for unittest integration tests."""

import contextlib
import os
from pathlib import Path
from unittest.mock import Mock, patch

from annotation.league.core import atomic_json, digest, read_json
from annotation.league.intake_benchmark import release_policy
from annotation.league.intake_profile import load_profile

# Both packages are needed by the image-based intake integration tests.
try:
    from PIL import Image
    import numpy
except ImportError:
    Image = None


@contextlib.contextmanager
def intake_fixture(store, config, stop_after=None, pts_offset=0):
    """Exercise the real intake orchestration with injected perception boundaries."""
    manifest = read_json(store.path("manifest.json"))
    manifest["schema_version"] = 2
    atomic_json(store.path("manifest.json"), manifest)
    config.fps = 1 / 30
    config.asset_cache = str(store.path("assets"))
    config.vl_python = "fixture-vl"
    profile = load_profile("domisum-1080p-v1")
    profile["thresholds_frozen"] = True
    profile["fields"] = []
    for index, (name, reader) in enumerate(
        (
            ("identity", "text"),
            ("kills", "number"),
            ("level", "number"),
            ("inventory", "text"),
            ("player_name", "text"),
            ("game_clock", "number"),
        )
    ):
        profile["fields"].append(
            {
                "id": name,
                "route": "scoreboard",
                "slot_id": "left:row1",
                "field": name,
                "box": [0, 0, 10 + index, 10],
                "reader": reader,
                "parser": "integer" if name == "kills" else "level" if name == "level" else "clock" if name == "game_clock" else "name",
                "kind": "champion",
            }
        )
    profile_path = store.path("fixture-profile.json")
    atomic_json(profile_path, profile)
    config.layout_profile = str(profile_path)
    atomic_json(store.path("config.json"), config.to_dict())
    signature = {"engine": "hybrid-v1", "tesseract": "fixture", "driver_packages": {}, "vl": {"runtime": {"gpu": "fixture", "total_memory_bytes": 48_000_000_000}}}
    release = (
        Path(config.asset_cache)
        / "releases"
        / (
            digest({"source": manifest["source"]["sha256"], "profile": profile})
            + ".json"
        )
    )
    atomic_json(store.path("intake_runtime.json"), signature)
    atomic_json(
        release,
        {
            "passed": True,
            "runtime": signature,
            "extraction_key": store.fingerprint(config),
            "release_policy": release_policy(),
        },
    )
    atomic_json(
        release.with_name(release.stem + "-pilot.json"),
        {"passed": True, "extraction_key": store.fingerprint(config)},
    )

    class Reader:
        def __init__(self, *args):
            pass

        def read(self, image, field, directory):
            timestamp = image.getpixel((0, 0))[0]
            values = {
                "identity": "Aatrox",
                "game_clock": {"displayed_text": f"{timestamp//60}:{timestamp%60:02}", "seconds": timestamp},
                "kills": timestamp // 20,
                "level": 1 + timestamp // 30,
                "inventory": ["6610"],
                "player_name": "한글플레이어",
            }
            return {
                "acceptance": "accepted",
                "reason": "fixture recognition",
                "value": values[field["field"]],
                "raw_text": str(values[field["field"]]),
                "reader": {"engine": "fixture"},
            }

    class VL:
        def __init__(self, *args):
            pass

        def read_group(self, entries, directory):
            directory.mkdir(parents=True, exist_ok=True)
            readings = {}
            for field, path in entries:
                with Image.open(path) as image:
                    readings[field["id"]] = Reader().read(image, field, directory)
            return readings

    def frames(*args):
        lower, upper = manifest["interval"]
        for index, timestamp in enumerate(range(int(lower), int(upper), 30)):
            store.check_stop()
            yield float(timestamp) + pts_offset, Image.new("RGB", (1920, 1080), (timestamp, 0, 0))
            if stop_after and index + 1 >= stop_after:
                store.stopping = True
                store.check_stop()

    worker = Mock()
    worker.identity = signature["vl"]
    worker.tokens = {}
    worker.peak_memory_bytes = 0
    with patch.dict(os.environ), patch(
        "annotation.league.hybrid_intake.runtime", return_value=(signature, (None, signature["vl"]))
    ), patch(
        "annotation.league.hybrid_intake.LocalVLRunner", return_value=worker
    ), patch("annotation.league.hybrid_intake.NumericReader", Reader), patch(
        "annotation.league.hybrid_intake.VLReader", VL
    ), patch("annotation.league.hybrid_intake.decoded_frames", frames), patch(
        "annotation.league.hybrid_intake.layout_matches", return_value=(True, [])
    ), patch("annotation.league.hybrid_intake._panel_state", return_value="present"):
        os.environ.pop("LEAGUE_ASSET_CACHE", None)
        yield worker
