"""FFmpeg decoding with actual source PTS and fixed source-pixel geometry."""
from __future__ import annotations

import hashlib
import math
import re
import subprocess
from pathlib import Path

from .core import Config, Route, SCHEMA, Store, atomic_json, now, read_json, uid


def probe(video: Path) -> dict:
    import json
    output = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate:format=duration", "-of", "json", str(video)])
    data = json.loads(output)
    stream = data["streams"][0]
    numerator, denominator = map(float, stream["r_frame_rate"].split("/"))
    return {"width": stream["width"], "height": stream["height"],
            "fps": numerator / denominator, "duration": float(data["format"]["duration"])}


def file_hash(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def windows(start: float, end: float, length: float, overlap: float):
    position = start
    while position < end - 1e-6:
        yield (round(position, 6), round(min(end, position + length), 6))
        if position + length >= end:
            break
        position += length - overlap


def sample_times(start: float, end: float, fps: float) -> list[float]:
    return [round(start + n / fps, 6) for n in range(max(1, math.ceil((end - start) * fps)))
            if start + n / fps < end]


class Media:
    def __init__(self, store: Store, config: Config):
        self.store, self.config = store, config
        self.manifest = read_json(store.path("manifest.json"))
        self.video = Path(self.manifest["source"]["path"])

    def frame(self, requested: float, route: Route) -> dict:
        self.store.check_stop()
        source_id = uid("source", self.manifest["source"]["sha256"], round(requested, 6))
        raw = self.store.path(f"frames/{source_id}.png")
        metadata = raw.with_suffix(".json")
        if not metadata.exists() or not raw.exists():
            raw.parent.mkdir(parents=True, exist_ok=True)
            command = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "info", "-y",
                       "-ss", str(requested), "-copyts", "-i", str(self.video),
                       "-map", "0:v:0", "-vf", "showinfo", "-frames:v", "1", "-update", "1", str(raw)]
            result = subprocess.run(command, capture_output=True, text=True)
            matches = re.findall(r"\bn:\s*0\s+pts:.*?pts_time:([\d.eE+-]+)", result.stderr)
            if result.returncode or not raw.exists() or not matches:
                raise RuntimeError(f"Cannot decode frame at {requested}: {result.stderr[-1500:]}")
            timestamp = float(matches[0])
            if abs(timestamp - requested) > max(0.1, 2 / self.manifest["source"]["fps"]):
                raise RuntimeError(f"Seek PTS mismatch: requested {requested}, decoded {timestamp}")
            atomic_json(metadata, {"requested_time": requested, "time": timestamp,
                                   "source_id": uid("source", self.manifest["source"]["sha256"], timestamp)})
        source = read_json(metadata)
        frame_id = uid("frame", source["source_id"], route.box, self.config.masks)
        crop = self.store.path(f"crops/{frame_id}.png")
        if not crop.exists():
            crop.parent.mkdir(parents=True, exist_ok=True)
            x0, y0, x1, y1 = route.box
            filters = [f"drawbox=x={x}:y={y}:w={xx-x}:h={yy-y}:color=black:t=fill"
                       for x, y, xx, yy in self.config.masks]
            filters.append(f"crop={x1-x0}:{y1-y0}:{x0}:{y0}")
            subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(raw),
                "-vf", ",".join(filters), "-frames:v", "1", "-update", "1", str(crop)], check=True)
        return {"id": frame_id, **source, "route": route.name, "box": route.box,
                "path": str(crop.relative_to(self.store.root))}

    def frames(self, start: float, end: float, fps: float, route: Route) -> list[dict]:
        frames = [self.frame(t, route) for t in sample_times(start, end, fps)]
        # Different requested times may decode to the same native frame.
        return list({(f["time"], f["route"]): f for f in frames}.values())


def prepare(store: Store, config: Config, video: Path, start=0.0, end=None) -> dict:
    config.validate()
    video = video.resolve()
    info = probe(video)
    if (info["width"], info["height"]) != (config.width, config.height):
        raise ValueError("Source resolution does not match layout; calibrate an explicit new profile")
    end = info["duration"] if end is None else min(float(end), info["duration"])
    if not 0 <= start < end:
        raise ValueError("Require 0 <= start < end <= duration")
    source = {"path": str(video), "sha256": file_hash(video), **info}
    manifest = {"schema_version": SCHEMA, "created_at": now(), "source": source,
                "interval": [start, end], "patch": config.patch,
                "model_revisions": {}, "editorial_masks": config.masks,
                "editorial_policy": "Exclude intervals with obscuring/editorial outcome overlays; no audio or source sidecars in model input"}
    previous = store.path("manifest.json")
    if previous.exists():
        old = read_json(previous)
        if old["source"]["sha256"] != source["sha256"] or old["interval"] != [start, end]:
            raise ValueError("Run belongs to another source/interval; choose a new run directory")
        manifest = {**old, "source": source, "patch": config.patch, "editorial_masks": config.masks}
    atomic_json(store.path("config.json"), config.to_dict())
    atomic_json(store.path("manifest.json"), manifest)
    media = Media(store, config)
    times = sorted(set(max(start, min(t, end - 1 / info["fps"])) for t in (30.0, 780.0, 1620.0)))
    previews = [media.frame(t, route) for t in times for route in config.routes]
    store.write("previews", previews)
    atomic_json(store.path("layout_validation.json"), {
        "source_geometry_valid": True, "preview_times": times,
        "model_preprocessing_validated": False,
        "hud_legibility": "requires pilot inspection; geometry alone does not establish readability"})
    store.commit_stage("prepare", store.fingerprint(config), preview_count=len(previews))
    return manifest
