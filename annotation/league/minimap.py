"""Sparse lane-minion and tower readings for the calibrated Midbeast source."""
from __future__ import annotations

from contextlib import closing

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .core import finite, read_json, uid
from .media import file_hash, sample_times
from .scene_reference import sample_source

TIMELINE = "minimap_timeline"
SOURCE = "f4f36e1a274617ecbe224eaf7fc3f8375f757acb7a517bcc0ee7780fcd83f240"
BOX = (1620, 782, 1910, 1072)
DEFAULT_INTERVAL = 5.0
REFERENCE_TIME = 170

# Minimap-relative centers, additional standing examples, and visibly empty sites.
TOWERS = (
    ("red", "top", "outer", (20, 82), (720,), (1050, 1090, 1250, 1620)),
    ("red", "top", "inner", (31, 154), (1450, 1675), ()),
    ("red", "top", "inhibitor", (24, 200), (1450, 1680), ()),
    ("red", "mid", "outer", (114, 160), (610, 850), (1050, 1620)),
    ("red", "mid", "inner", (99, 190), (850,), (1050, 1250, 1330, 1450)),
    ("red", "mid", "inhibitor", (72, 211), (1450,), (1510, 1550, 1680)),
    ("red", "bot", "outer", (203, 262), (850,), (1050, 1250, 1620)),
    ("red", "bot", "inner", (135, 254), (1450,), (1660, 1670)),
    ("red", "bot", "inhibitor", (84, 257), (1450,), ()),
    ("blue", "top", "outer", (84, 17), (390, 610, 850), (1250, 1450, 1520, 1620)),
    ("blue", "top", "inner", (154, 25), (1450,), ()),
    ("blue", "top", "inhibitor", (203, 20), (1450,), ()),
    ("blue", "mid", "outer", (173, 120), (600, 980), (1090, 1130, 1240, 1490, 1620)),
    ("blue", "mid", "inner", (190, 90), (1450,), ()),
    ("blue", "mid", "inhibitor", (216, 68), (1450,), ()),
    ("blue", "bot", "outer", (268, 198), (1250, 1350), (1435, 1450, 1585, 1620)),
    ("blue", "bot", "inner", (258, 124), (1450,), ()),
    ("blue", "bot", "inhibitor", (263, 79), (1450,), ()),
)
LANES = {
    "top": [(24, 192), (30, 154), (25, 66), (33, 43), (80, 20), (215, 20)],
    "mid": [(78, 205), (120, 156), (142, 138), (171, 118), (215, 70)],
    "bot": [(91, 258), (220, 255), (247, 231), (262, 205), (265, 166), (260, 85)],
}
MINION_EXAMPLES = {
    "red": [(850, 253, 222), (850, 257, 214), (1250, 147, 259), (1450, 227, 244), (1450, 233, 244)],
    "blue": [(850, 128, 20), (850, 265, 154), (1450, 110, 20), (1450, 174, 113)],
}
LAYOUT_ANCHORS = (
    (slice(2, 20), slice(8, 23), .035),
    (slice(281, 290), slice(75, 235), .035),
    (slice(72, 244), slice(282, 289), .020),
)


def pixels(image):
    return np.asarray(image, dtype=np.float32) / 255


def patch(image, center, x_radius, y_radius):
    x, y = center
    return image[y-y_radius:y+y_radius+1, x-x_radius:x+x_radius+1]


def lane_pixels():
    yy, xx = np.mgrid[:BOX[3]-BOX[1], :BOX[2]-BOX[0]]
    points = np.stack((xx, yy), axis=-1)
    distances = []
    for path in LANES.values():
        segments = []
        for start, end in zip(path, path[1:]):
            direction = np.subtract(end, start)
            offset = points - start
            fraction = np.clip((offset @ direction) / (direction @ direction), 0, 1)
            segments.append(np.sum((offset - fraction[..., None] * direction) ** 2, axis=-1))
        distances.append(np.minimum.reduce(segments))
    nearest = np.argmin(distances, axis=0)
    eligible = np.minimum.reduce(distances) <= 9**2
    for _, _, _, center, _, _ in TOWERS:
        eligible &= np.sum((points - center) ** 2, axis=-1) > 10**2
    y, x = np.nonzero(eligible)
    return y, x, nearest[y, x]


def minion_groups(scores, points, lane_ids, threshold):
    candidates = np.flatnonzero(scores <= threshold)
    markers = []
    for i in candidates[np.argsort(scores[candidates])]:
        point = points[i]
        if all(np.sum((point - earlier) ** 2) >= 4**2 for _, earlier in markers):
            markers.append((lane_ids[i], point))
    groups = []
    for lane_id, lane in enumerate(LANES):
        pending = {tuple(point) for label, point in markers if label == lane_id}
        while pending:
            group = [pending.pop()]
            for x, y in group:
                nearby = {(a, b) for a, b in pending if (x-a)**2 + (y-b)**2 <= 12**2}
                group.extend(nearby)
                pending.difference_update(nearby)
            if len(group) >= 2:
                groups.append({"lane": lane, "center": np.rint(np.mean(group, axis=0)).astype(int).tolist()})
    return sorted(groups, key=lambda row: (row["lane"], row["center"]))


class Reader:
    def __init__(self, references):
        reference = references[REFERENCE_TIME]
        self.anchors = [(y, x, threshold, reference[y, x]) for y, x, threshold in LAYOUT_ANCHORS]
        self.y, self.x, self.lanes = lane_pixels()
        self.points = np.stack((self.x, self.y), axis=1)
        self.minions = {team: np.stack([patch(references[time], (x, y), 2, 2) for time, x, y in examples])
                        for team, examples in MINION_EXAMPLES.items()}
        y, x = np.mgrid[-8:9, -6:7]
        self.site_mask = (x/7)**2 + (y/10)**2 <= 1
        self.center_mask = (abs(x) <= 3) & (abs(y) <= 4)
        # Tower edges retain their shape as the number changes.
        self.outline_mask = self.site_mask & ~self.center_mask
        self.towers = []
        for team, lane, tier, center, additional, absent in TOWERS:
            standing = np.stack([patch(references[t], center, 6, 8) for t in (REFERENCE_TIME, *additional)])
            clear_sites = np.stack([patch(references[t], center, 6, 8) for t in absent]) if absent else None
            self.towers.append((team, lane, tier, center, standing, clear_sites))

    def tower_status(self, tile, standing, clear_sites):
        # Limit the effect of isolated pixels such as a camera box edge.
        standing_diff = np.abs(standing - tile).mean(axis=-1)[:, self.outline_mask]
        standing_error = np.minimum(standing_diff, .22).mean(axis=1).min()
        clear_error = float("inf")
        if clear_sites is not None:
            clear_diff = np.abs(clear_sites - tile).mean(axis=-1)
            site = np.minimum(clear_diff[:, self.site_mask], .15).mean(axis=1)
            center = np.minimum(clear_diff[:, self.center_mask], .22).mean(axis=1)
            clear_error = (.65 * site + .35 * center).min()
        if standing_error <= .075 and standing_error + .018 < clear_error:
            return "standing", "tower_icon_match"
        if clear_error <= .050 and clear_error + .018 < standing_error:
            return "icon_absent", "clear_site_match"
        if standing_error <= .075 or clear_error <= .050:
            return "unknown", "ambiguous_match"
        return "unknown", "no_confident_match"

    def read(self, image):
        layout_visible = sum(bool(np.abs(image[y, x] - template).mean() <= threshold)
                             for y, x, threshold, template in self.anchors) >= 2
        towers = []
        for team, lane, tier, center, standing, clear_sites in self.towers:
            status, reason = "unknown", "minimap_unreadable"
            if layout_visible:
                status, reason = self.tower_status(patch(image, center, 6, 8), standing, clear_sites)
            towers.append({"team": team, "lane": lane, "tier": tier, "center": center,
                           "status": status, "status_reason": reason})

        waves = None
        if layout_visible:
            windows = sliding_window_view(image, (5, 5), axis=(0, 1))
            tiles = windows[self.y-2, self.x-2].transpose(0, 2, 3, 1)
            waves = []
            for team, examples in self.minions.items():
                errors = np.abs(tiles[:, None] - examples).mean(axis=(2, 3, 4)).min(axis=1)
                for group in minion_groups(errors, self.points, self.lanes, .070 if team == "red" else .064):
                    waves.append({"team": team, **group})
        return {"layout_visible": layout_visible, "towers": towers, "visible_minion_waves": waves}


def frame_at(video, time, end):
    # Seek independently so the pipeline can skip decoding the gaps between samples.
    with closing(sample_source(video, time, min(time + 1, end), 1)) as frames:
        return next(frames, None)


def run(store, start=None, end=None, interval=DEFAULT_INTERVAL):
    manifest = read_json(store.path("manifest.json"))
    source = manifest["source"]
    if source["sha256"] != SOURCE or (source["width"], source["height"]) != (1920, 1080):
        raise ValueError("Minimap intake is calibrated for the prepared 1080p Midbeast recording OLLZBe5-fg4")
    lower, upper = manifest["interval"]
    start = lower if start is None else start
    end = upper if end is None else end
    if (not all(finite(value) for value in (start, end, interval))
            or not lower <= start < end <= upper or interval < 1):
        raise ValueError("Use a source interval inside the prepared run and a sample interval of at least one second")
    video = store.path(source["path"])
    if file_hash(video) != SOURCE:
        raise ValueError("Prepared Midbeast source video changed")

    times = {REFERENCE_TIME}
    for *_, additional, absent in TOWERS:
        times.update((*additional, *absent))
    times.update(time for examples in MINION_EXAMPLES.values() for time, _, _ in examples)
    references = {}
    for time in sorted(times):
        store.check_stop()
        sample = frame_at(video, time, source["duration"])
        if sample is None:
            raise ValueError(f"Missing minimap calibration frame at source second {time}")
        references[time] = pixels(sample[1].crop(BOX))
    reader = Reader(references)

    rows = []
    for requested in sample_times(start, end, 1 / interval):
        store.check_stop()
        sample = frame_at(video, requested, end)
        if sample is None:
            continue
        time, image = sample
        crop = image.crop(BOX)
        frame_id = uid("minimap-frame", SOURCE, time, BOX)
        crop_path = store.path(f"minimap/crops/{frame_id}.png")
        if not crop_path.exists():
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = crop_path.with_suffix(".writing")
            crop.save(temporary, format="PNG")
            temporary.replace(crop_path)
        rows.append({"id": uid("minimap-sample", frame_id), "frame_id": frame_id, "time": time,
                     "route": "minimap", "box": BOX, "path": str(crop_path.relative_to(store.root)),
                     "source_sha256": SOURCE, "sampling_interval_seconds": interval,
                     "review_status": "unreviewed", "experimental": True, "precision": "sampled",
                     **reader.read(pixels(crop))})
    if not rows:
        raise ValueError("No frames decoded in the requested minimap interval")
    store.write(TIMELINE, rows)
    return {"samples": len(rows), "visible_maps": sum(row["layout_visible"] for row in rows),
            "timeline": str(store.path(TIMELINE + ".jsonl"))}
