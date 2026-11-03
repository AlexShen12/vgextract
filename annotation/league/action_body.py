"""CPU overhead-bar localization and explicitly bounded player continuity."""
from collections import defaultdict
import numpy as np

from .core import uid
from .readers import measure_bar


def spans(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def candidates(image, profile):
    """Require a colored bar, dark frame and a champion level cell, not color alone."""
    cfg = profile["body_binder"]
    pixels = np.asarray(image.convert("RGB"), dtype=float)
    red, green, blue = pixels.transpose(2, 0, 1)
    palette = cfg["palette"]
    if palette["color"] != "green":
        return [], "unsupported_palette"
    mask = (green >= palette["min_green"]) & (green-red >= palette["green_red_gap"]) & (green-blue >= palette["green_blue_gap"])
    x0, y0, x1, y1 = cfg["search_box"]
    mask[:y0] = False; mask[y1:] = False; mask[:, :x0] = False; mask[:, x1:] = False
    for a, b, c, d in cfg["exclude_boxes"]:
        mask[b:d, a:c] = False
    width, height = cfg["bar_size"]
    result = []
    hud = measure_bar(image.crop(cfg["hud_health_box"]), {"color": "green"})
    hud_fraction = hud["value"] if hud["acceptance"] == "accepted" else None
    # Filled pixels are continuous below the bar's tick marks; tolerate one-pixel gaps.
    for y in np.flatnonzero(mask.sum(axis=1) >= cfg["minimum_fill_pixels"]):
        row = mask[y].copy()
        row[1:-1] |= row[:-2] & row[2:]
        for left, right in spans(row):
            left, right, y = int(left), int(right), int(y)
            if not cfg["minimum_fill_pixels"] <= right-left <= width+3:
                continue
            if left+width >= x1 or y+height >= y1:
                continue
            if any(abs(left-r["bar_box"][0]) < 8 and abs(y-r["bar_box"][1]) < height+5 for r in result):
                continue
            offsets = cfg["level_cell"]
            a, b, c, d = left+offsets[0], y+offsets[1], left+offsets[2], y+offsets[3]
            if min(a, b) < 0 or d > pixels.shape[0]:
                continue
            level = pixels[b:d, a:c]
            white = (level.min(axis=2) > 145) & (level.max(axis=2)-level.min(axis=2) < 65)
            dark_fraction = float((level.max(axis=2) < 100).mean())
            if white.mean() < cfg["minimum_level_white"] or dark_fraction < cfg["minimum_level_dark"]:
                continue
            # A natural green strip usually lacks the black horizontal HP-bar frame.
            border = pixels[max(0, y-3):max(1, y-1), left:left+width]
            if float((border.max(axis=2) < 80).mean()) < cfg["minimum_border_dark"]:
                continue
            fill = mask[y:y+height, left:left+width].mean(axis=0) >= .4
            fraction = float(fill.mean())
            if hud_fraction is not None and abs(fraction-hud_fraction) > cfg["health_tolerance"]:
                continue
            offsets = cfg["actor_region"]
            body = [max(x0, left+offsets[0]), max(y0, y+offsets[1]),
                    min(x1, left+offsets[2]), min(y1, y+offsets[3])]
            if any(a < body[2] and c > body[0] and b < body[3] and d > body[1]
                   for a, b, c, d in cfg["exclude_boxes"]):
                continue
            result.append({"bar_box": [left, y, left+width, y+height], "actor_region": body,
                           "health_fraction": fraction, "hud_health_fraction": hud_fraction,
                           "level_white_fraction": float(white.mean()), "level_dark_fraction": dark_fraction})
    return result, "unique_candidate" if len(result) == 1 else "ambiguous_candidates" if result else "no_supported_bar"


def bind_frames(frames, identities, profile, stop=lambda: None):
    from PIL import Image
    identity = {(r["frame_id"], r["owner_id"]): r for r in identities}
    observations, bindings = [], []
    histories = defaultdict(list)
    for frame in sorted(frames, key=lambda f: f["time"]):
        stop()
        with Image.open(frame["native_path"]) as image:
            found, reason = candidates(image, profile) if frame["layout_valid"] else ([], "invalid_layout")
        for owner in profile["owners"]:
            name = identity.get((frame["id"], owner["id"]), {}).get("hud_champion")
            row = {"id": uid("body-reading", frame["id"], owner["id"], profile["body_binder"]),
                   "frame_id": frame["id"], "frame_ids": [frame["id"]], "time": frame["time"],
                   "segment_id": frame["segment_id"], "owner_epoch": frame["owner_epochs"][owner["id"]],
                   "owner_id": owner["id"], "champion": name, "candidates": found, "reason": reason,
                   "actor_track": None, "binding_visible": False, "mode": "core", "schema_version": 3}
            history = histories[owner["id"]]
            previous = history[-1] if history else None
            contiguous = previous and all(previous[k] == row[k] for k in ("segment_id", "owner_epoch"))
            known_name = next((r["champion"] for r in reversed(history) if r.get("champion")), None)
            if name and known_name and name != known_name:
                contiguous = False
            if contiguous:
                contiguous = 0 < row["time"]-previous["time"] <= .095 and len(found) == len(previous["candidates"]) == 1
            if contiguous:
                a, b = found[0]["bar_box"], previous["candidates"][0]["bar_box"]
                contiguous = abs(a[0]-b[0])+abs(a[1]-b[1]) <= profile["body_binder"]["maximum_step_pixels"]
            if not contiguous:
                history.clear()
            if len(found) == 1:
                history.append(row)
                if len(history) >= profile["body_binder"]["confirm_samples"]:
                    track = history[-2].get("actor_track") or uid("player-track", history[0]["frame_id"], owner["id"])
                    row.update(actor_track=track, binding_visible=True, **found[0])
                    evidence = history[-profile["body_binder"]["confirm_samples"]:]
                    bindings.append({"id": uid("color-binding", *[r["id"] for r in evidence]),
                        "mode": "core", "schema_version": 3, "owner_id": owner["id"], "owner_epoch": row["owner_epoch"],
                        "segment_id": row["segment_id"], "champion": name, "actor_track": track,
                        "binding_visible": True, "start": evidence[0]["time"], "end": row["time"],
                        "bar_box": row["bar_box"], "actor_region": row["actor_region"],
                        "actor_description": "Player-style overhead HP bar with champion-level cell; approximate body region below",
                        "evidence_ids": [r["id"] for r in evidence], "frame_ids": [r["frame_id"] for r in evidence]})
                    for sample in evidence:
                        sample.update(actor_track=track, binding_visible=True, **sample["candidates"][0])
            observations.append(row)
    return observations, bindings


def reconcile_tracks(rows, bindings, previous):
    """Alias native packets only from nearby, overlapping independently detected bars."""
    votes = defaultdict(set)
    for row in rows:
        if not row.get("binding_visible"):
            continue
        neighbors = [p for p in previous if p.get("binding_visible") and all(p[k] == row[k] for k in
                     ("owner_id", "owner_epoch", "segment_id")) and abs(p["time"]-row["time"]) <= .095]
        if not neighbors:
            continue
        other = min(neighbors, key=lambda p: abs(p["time"]-row["time"]))
        a,b = row["bar_box"],other["bar_box"]
        intersection = max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
        union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
        if union > 0 and intersection/union >= .5:
            votes[row["actor_track"]].add(other["actor_track"])
    aliases = {track: next(iter(choices)) for track,choices in votes.items() if len(choices) == 1}
    return ([{**r,"actor_track":aliases.get(r.get("actor_track"),r.get("actor_track"))} for r in rows],
            [{**r,"actor_track":aliases.get(r["actor_track"],r["actor_track"])} for r in bindings])
