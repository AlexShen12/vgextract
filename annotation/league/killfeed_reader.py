"""Native Midbeast killfeed crops and independent portrait reads."""
import base64
import io
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .inventory_reader import ChampionPortraitMatcher


@dataclass
class Entry:
    top: int
    kind: str
    side: str
    box: list
    roles: dict
    signature: np.ndarray


class KillfeedReader:
    def __init__(self, profile, root, assets):
        self.profile = profile
        self.matcher = ChampionPortraitMatcher(root, {"assets": assets}, {})
        objectives = []
        for template in profile["objective_templates"]:
            with Image.open(io.BytesIO(base64.b64decode(template["png"]))) as image:
                objectives.extend(self.matcher.vector(image, inset) for inset in (.08, .18))
        self.objectives = np.asarray(objectives)

    def portrait(self, image, x, y):
        width = self.profile["portrait_size"]
        box = [x, y, x + width, y + width]
        result = self.matcher.read(image.crop(box), {})
        score = result["reader"]["score"]
        margin = result["reader"]["margin"]
        accepted = score >= self.profile["portrait_score"] and margin >= self.profile["portrait_margin"]
        return {"champion": result["value"] if accepted else None, "box": box,
                "score": score, "margin": margin, "candidates": result["candidates"],
                "reference_sha256": result["reader"]["reference_sha256"]}

    def rows(self, image):
        left, top, right, bottom = self.profile["feed_box"]
        pixels = np.asarray(image.crop((left, top, right, bottom)), dtype=np.int16)
        edge = np.median(pixels[:, -3:-1], axis=1)
        candidates = []
        for color, style in self.profile["styles"].items():
            distance = np.abs(edge - style["edge_rgb"]).mean(axis=1)
            edges = np.flatnonzero(np.diff(np.r_[False, distance <= style["edge_error"], False]))
            for first, end in zip(edges[::2], edges[1::2]):
                if not self.profile["edge_height"][0] <= end - first <= self.profile["edge_height"][1]:
                    continue
                # The solid fill just before the victim is independent of the sword,
                # assist count, portrait artwork and selected-player outline.
                x0, x1 = self.profile["fill_x"]
                fill = pixels[first + 10:first + 46, x0 - left:x1 - left]
                if np.abs(np.median(fill, axis=(0, 1)) - style["fill_rgb"]).mean() > style["fill_error"]:
                    continue
                observed = int(first) + top
                origin, step, tolerance = self.profile["row_alignment"]
                nominal = origin + step * round((observed - origin) / step)
                aligned = nominal if abs(observed - nominal) <= tolerance else observed
                candidates.append((aligned, color))
        return sorted(candidates)

    def read(self, image):
        entries = []
        for top, color in self.rows(image):
            y = top + self.profile["portrait_y"]
            roles = {role: self.portrait(image, x, y) for role, x in self.profile["portrait_x"].items()}
            left, _, right, bottom = self.profile["feed_box"]
            box = [left, top, right, min(bottom, top + self.profile["row_height"])]
            portraits = [image.crop(role["box"]) for role in roles.values()]
            signature = np.asarray([np.asarray(p.resize((16, 16), Image.Resampling.BILINEAR))
                                    for p in portraits], dtype=np.int16)
            # Objective icons are compared separately from champion portraits. A row
            # with neither a champion nor an objective witness stays unidentified.
            objective = any(self.is_objective(p) for p in portraits)
            kind = "objective" if objective else "candidate"
            entries.append(Entry(top, kind, self.profile["styles"][color]["killer_side"], box, roles, signature))
        return entries

    def is_objective(self, portrait):
        vectors = [self.matcher.vector(portrait, inset) for inset in (.08, .18)]
        return float(np.max(self.objectives @ np.asarray(vectors).T)) >= self.profile["objective_score"]

    def incoming(self, entry, current, previous, previous_entries):
        """Locate the same two portrait shapes in a fading or sliding preceding row."""
        if any(abs(old.top - entry.top) < self.profile["row_height"] / 2 for old in previous_entries):
            return None
        left, top, _, _ = self.profile["feed_box"]
        edge = self.profile["transition"]
        width = self.profile["portrait_size"]

        def reduced(image):
            size = (image.width // 2, image.height // 2)
            return np.asarray(image.convert("L").resize(size, Image.Resampling.BILINEAR), dtype=np.float32)

        source, target = reduced(current), reduced(previous)
        # Match the interiors at half resolution so one-pixel encoding or
        # alignment variation has little effect on the fade comparison.
        span = (width - 8) // 2
        views = np.lib.stride_tricks.sliding_window_view(target, (span, span))
        x_shifts = np.arange(-edge["maximum_left_shift"] // 2, 1)
        y_shifts = np.arange(-2, 3)
        y = (entry.top + self.profile["portrait_y"] + 4 - top) // 2
        if y + y_shifts[0] < 0 or y + y_shifts[-1] >= len(views):
            return None
        comparisons = []
        for x in self.profile["portrait_x"].values():
            x = (x + 4 - left) // 2
            query = source[y:y+span, x:x+span].copy()
            query -= query.mean()
            patches = views[np.ix_(y + y_shifts, x + x_shifts)].copy()
            patches -= patches.mean(axis=(-1, -2), keepdims=True)
            norms = np.linalg.norm(query) * np.linalg.norm(patches, axis=(-1, -2))
            comparisons.append(np.sum(patches * query, axis=(-1, -2)) / np.maximum(norms, 1e-6))
        agreement = np.minimum(*comparisons)
        y_index, x_index = np.unravel_index(agreement.argmax(), agreement.shape)
        if agreement[y_index, x_index] < edge["minimum_correlation"]:
            return None
        dx, dy = int(x_shifts[x_index]) * 2, int(y_shifts[y_index]) * 2
        local_top = entry.top + dy - top
        box = [left, entry.top + dy, entry.box[2] + dx, entry.box[3] + dy]
        roles = {}
        for name, role in entry.roles.items():
            x, y, right, bottom = role["box"]
            match = self.portrait(previous, x + dx - left, y + dy - top)
            match["box"] = [x + dx, y + dy, right + dx, bottom + dy]
            roles[name] = match
        crop = previous.crop((0, local_top, box[2] - left, local_top + self.profile["row_height"]))
        return box, crop, roles
