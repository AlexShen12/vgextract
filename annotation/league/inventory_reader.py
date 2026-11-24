"""Match inventory items and portraits to pinned visual references."""

import base64
from collections import Counter, defaultdict
import io

from .core import atomic_json, digest, implementation_hash, read_json
from .inventory_references import publish_catalog
from .readers import IconMatcher, unknown


INSETS = (0.08, 0.18)
CARDINAL_SHIFTS = ((-1, 0), (1, 0), (0, -1), (0, 1))
ALIGNMENT_SCORE_WINDOW = 0.08


def icon_view(image, inset):
    width, height = image.size
    inset *= .72
    return image.convert("RGB").crop((round(width * inset), round(height * inset),
                                      round(width * (1 - inset)), round(height * (1 - inset)))).resize((24, 24))


class InventoryIconMatcher(IconMatcher):
    @staticmethod
    def weights():
        import numpy as np

        weights = np.ones((24, 24))
        weights[:2, :] = weights[-2:, :] = weights[:, :2] = weights[:, -2:] = .4
        weights[15:, 15:] *= .25
        return weights

    @classmethod
    def vector(cls, image, inset=0.08):
        import numpy as np

        colors = np.asarray(icon_view(image, inset), dtype=float) / 255
        weights = cls.weights()
        mean = (colors * weights[..., None]).sum(axis=(0, 1)) / weights.sum()
        centered = (colors - mean) * weights[..., None]
        vertical, horizontal = np.gradient(colors.mean(axis=2))
        vector = np.concatenate([centered.ravel(), (vertical * weights).ravel(), (horizontal * weights).ravel()])
        return vector / max(float(np.linalg.norm(vector)), 1e-9)


class ScoreboardTrinketMatcher(InventoryIconMatcher):
    @staticmethod
    def weights():
        import numpy as np

        weights = np.ones((24, 24))
        weights[:3, :] = weights[:, :3] = weights[:, -3:] = .3
        # Vision score is rendered across the bottom of the scoreboard icon.
        weights[16:, :] = .02
        return weights


class ChampionPortraitMatcher(InventoryIconMatcher):
    """Margins compare champions, never two skins belonging to one champion."""
    def read(self, image, field):
        import numpy as np
        entries = self.references.get("champion", [])
        queries = [self.vector(image, inset) for inset in INSETS]
        values = (self.matrices["champion"] @ np.asarray(queries, dtype=np.float32).T)
        values = values.reshape(len(entries), 2, 2).max(axis=(1, 2))
        grouped = defaultdict(list)
        for score, (entry, _) in zip(values, entries):
            grouped[entry["id"]].append((float(score), entry))
        ranked = sorted(((max(v, key=lambda p: p[0])[0], name, v) for name, v in grouped.items()), reverse=True)
        score, name, variants = ranked[0]
        best = max(variants, key=lambda pair: pair[0])[1]
        skin_ids = {e.get("skin_id") for s, e in variants if abs(s-score) < 1e-6}
        return {"acceptance": "accepted", "value": name, "raw_text": None,
                "skin_id": next(iter(skin_ids)) if len(skin_ids) == 1 else None,
                "reason": "matched best portrait variant per champion",
                "reader": {"score": score, "margin": score-(ranked[1][0] if len(ranked) > 1 else 0),
                           "reference_sha256": best.get("sha256")},
                "candidates": [{"id": name, "score": score} for score, name, _ in ranked[:3]]}


def red_pixels(rgb):
    import numpy as np

    r, g, b = rgb.transpose(2, 0, 1)
    return (r >= 100) & (r - np.maximum(g, b) >= 38)


class RespawnPortraitMatcher:
    def __init__(self, root, assets):
        import numpy as np
        from PIL import Image

        self.ids, views = [], []
        for asset in assets:
            if asset["kind"] == "champion":
                self.ids.append(asset["id"])
                with Image.open(root / asset["path"]) as image:
                    views.append([np.asarray(icon_view(image, inset).convert("L"), dtype=float) for inset in INSETS])
        self.views = np.asarray(views)
        y, x = np.ogrid[:24, :24]
        self.circle = (x - 11.5) ** 2 + (y - 11.5) ** 2 < 11.5 ** 2
        # Level and row status badges overlap the bottom corners of the portrait.
        self.art = self.circle & ~((x >= 17) & (y >= 17)) & ~((x <= 5) & (y >= 17))

    def has_timer(self, image):
        import numpy as np

        rgb = np.asarray(image.convert("RGB").resize((24, 24)), dtype=float)
        red = red_pixels(rgb)
        remaining = self.circle & ~red
        return (red[5:19, 5:19].sum() >= 6 and remaining.any()
                and np.median(np.ptp(rgb, axis=2)[remaining]) <= 18)

    def read(self, image):
        import numpy as np

        scores = []
        for inset in INSETS:
            view = icon_view(image, inset)
            red = red_pixels(np.asarray(view, dtype=float))
            # Include the antialiased border of each red digit. Apply the same
            # comparison mask to the query and every grayscale reference.
            padded = np.pad(red, 1)
            masked = np.logical_or.reduce([padded[y:y + 24, x:x + 24] for y in range(3) for x in range(3)])
            keep = self.art & ~masked
            if keep.sum() < self.art.sum() / 2:
                continue
            query = np.asarray(view.convert("L"), dtype=float)[keep]
            query -= query.mean()
            references = self.views[:, :, keep]
            references -= references.mean(axis=2, keepdims=True)
            norms = np.linalg.norm(references, axis=2) * np.linalg.norm(query)
            scores.append((references @ query) / np.maximum(norms, 1e-9))
        if not scores:
            return unknown("respawn digits obscure too much of the portrait")

        similarities = np.asarray(scores).max(axis=(0, 2))
        grouped = defaultdict(list)
        for identifier, similarity in zip(self.ids, similarities):
            grouped[identifier].append(float(similarity))
        ranked = sorted(((name, max(values)) for name, values in grouped.items()), key=lambda pair: pair[1], reverse=True)
        identifier, score = ranked[0]
        margin = score - (ranked[1][1] if len(ranked) > 1 else 0)
        return {"acceptance": "accepted", "value": identifier, "raw_text": None,
                "reason": "matched grayscale portrait with red respawn digits masked",
                "reader": {"score": float(score), "margin": float(margin), "view": "grayscale_red_mask"},
                "candidates": [{"id": name, "score": float(value)} for name, value in ranked[:3]]}


class InventoryReader:
    def __init__(self, store, root, manifest, settings):
        import numpy as np
        from PIL import Image

        self.store = store
        self.settings = settings
        self.catalog_id = publish_catalog(store, root, manifest)
        portrait_assets = [a for a in manifest["assets"] if a["kind"] == "champion"]
        skin_signature = None
        if settings.get("communitydragon_version"):
            from .reference_bundle import load_skins, skin_root
            skin_directory = skin_root(root.parent.parent, manifest["reference_version"], settings["communitydragon_version"])
            skins = load_skins(skin_directory)
            skin_signature = digest(skins)
            portrait_assets += [{**a, "path": str((skin_directory/a["path"]).resolve())} for a in skins["assets"]]
        self.signature = digest({"catalog": self.catalog_id, "policy": settings,
                                 "skins": skin_signature,
                                 "implementation": implementation_hash(["inventory_reader.py", "readers.py"])})
        self.skin_signature = skin_signature
        self.items = {asset["id"]: asset for asset in manifest["assets"] if asset["kind"] == "item"}
        assets = [asset for asset in manifest["assets"] if asset["kind"] == "champion" or asset.get("slot_class") != "internal"]
        gates = {"icon": 0, "icon_margin": 0}
        self.matcher = InventoryIconMatcher(root, {"assets": assets}, gates)
        self.portraits = ChampionPortraitMatcher(root, {"assets": portrait_assets}, gates)
        trinkets = [asset for asset in assets if asset.get("slot_class") == "trinket"]
        self.trinkets = InventoryIconMatcher(root, {"assets": trinkets}, gates) if trinkets else self.matcher
        self.scoreboard_trinkets = ScoreboardTrinketMatcher(root, {"assets": trinkets}, gates) if trinkets else None
        self.respawn_portraits = RespawnPortraitMatcher(root, portrait_assets)
        self.artwork_counts = Counter(asset["artwork"] for asset in self.items.values())
        self.hidden_portraits = []
        for encoded in settings.get("hidden_portraits", []):
            with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
                self.hidden_portraits.append(self.yellow_pixels(image))
        self.selected_portrait = None
        if exemplar := settings.get("selected_portrait"):
            if exemplar["id"] not in self.respawn_portraits.ids:
                raise ValueError("Selected portrait exemplar is absent from the pinned champion catalog")
            with Image.open(io.BytesIO(base64.b64decode(exemplar["png"]))) as image:
                self.selected_portrait = self.matcher.vector(image)
        self.empty_rgb = np.asarray(settings["empty_rgb"], dtype=float)

    @staticmethod
    def yellow_pixels(crop):
        import numpy as np

        rgb = np.asarray(crop.convert("RGB").resize((24, 24)), dtype=float)
        r, g, b = rgb.transpose(2, 0, 1)
        return (r > 118) & (g > 94) & (r > 1.05 * g) & (g > 1.5 * b)

    def hidden_portrait(self, crop):
        mask = self.yellow_pixels(crop)
        return any((mask & ref).sum() >= .65 * (mask | ref).sum() for ref in self.hidden_portraits if ref.any())

    @staticmethod
    def apply_gate(reading, threshold):
        if reading["acceptance"] != "accepted":
            return reading
        score, margin = reading["reader"]["score"], reading["reader"]["margin"]
        if score < threshold["score"] or margin <= 0 or margin < threshold["margin"]:
            return {**reading, **unknown("icon score or margin below the direct-match threshold")}
        return reading

    def read_trinket(self, crop, route, gates):
        threshold = gates.get("trinket", gates["item"])
        if route == "selected_hud":
            # The selected slot includes a keybind, charge badge and bright frame.
            width, height = crop.size
            crop = crop.crop((round(width / 32), round(height / 8),
                              width - round(width * 5 / 32), height - round(height * 5 / 32)))
        elif self.scoreboard_trinkets:
            masked = self.scoreboard_trinkets.read(crop, {"kind": "item"})
            masked["reader"]["view"] = "scoreboard_vision_score_mask"
            result = self.apply_gate(masked, gates.get("masked_trinket", threshold))
            if result["acceptance"] == "accepted":
                return result
        return self.apply_gate(self.trinkets.read(crop, {"kind": "item"}), threshold)

    def match(self, crop, field):
        import numpy as np

        kind, route = field["kind"], field["route"]
        gates = self.settings["thresholds"][route]
        if kind == "item":
            pixels = np.asarray(crop.convert("RGB"), dtype=float)
            errors = np.abs(pixels - self.empty_rgb).mean(axis=2)
            if errors.mean() <= 5 and np.percentile(errors, 95) <= 11:
                return {**unknown("calibrated empty slot background"), "acceptance": "absent"}
            if float(pixels.std(axis=(0, 1)).max()) < 7:
                return unknown("slot is obscured or has too little visual detail")
            if field["field"] == "inventory.trinket":
                reading = self.read_trinket(crop, route, gates)
            else:
                reading = self.apply_gate(self.matcher.read(crop, {"kind": kind}), gates[kind])
            candidate = reading["candidates"][0]["id"]
            if self.artwork_counts[self.items[candidate]["artwork"]] > 1:
                return {**reading, **unknown("multiple item IDs share the same reference art")}
            return reading
        if route == "scoreboard":
            if self.hidden_portrait(crop):
                return unknown("scoreboard hides the portrait")
            if self.respawn_portraits.has_timer(crop):
                reading = self.respawn_portraits.read(crop)
                return self.apply_gate(reading, gates.get("respawn", gates[kind]))
        elif self.selected_portrait is not None:
            exemplar = self.settings["selected_portrait"]
            score = float(self.matcher.vector(crop) @ self.selected_portrait)
            if score >= .94:
                return {**unknown("matched the source-calibrated selected portrait"), "acceptance": "accepted",
                        "value": exemplar["id"], "reader": {"score": score, "exemplar_time": exemplar["time"],
                                                             "exemplar_evidence": exemplar["evidence"]}}
        return self.apply_gate(self.portraits.read(crop, {"kind": kind}), gates[kind])

    def read(self, crop, field, crop_hash):
        settings = {key: field[key] for key in ("route", "kind")}
        settings["trinket"] = field["field"] == "inventory.trinket"
        key = digest({"crop": crop_hash, "field": settings, "signature": self.signature})
        path = self.store.path(f"inventory-reader-cache/{key}/reading.json")
        hit = path.exists()
        if hit:
            reading = read_json(path)
        else:
            reading = self.match(crop, field)
            reading["reader"] = {**reading.get("reader", {}), "engine": "inventory-reference-v2"}
            atomic_json(path, {**reading, "catalog_id": self.catalog_id})
        return {**reading, "cache_path": str(path.relative_to(self.store.root))}, hit

    def retry_alignment(self, reading, field, shifted_crops):
        if field["route"] != "scoreboard":
            return reading
        portrait = field["kind"] == "champion"
        if portrait:
            if not self.settings.get("scoreboard_portrait_alignment"):
                return reading
            return self.align_portrait(reading, field, shifted_crops)
        if field["kind"] != "item" or field["field"] == "inventory.trinket" or reading["acceptance"] != "unknown":
            return reading
        candidates = reading.get("candidates", [])
        floor = self.settings["thresholds"]["scoreboard"]["item"]["score"]
        if not candidates or candidates[0]["score"] < floor - ALIGNMENT_SCORE_WINDOW:
            return reading
        if self.artwork_counts[self.items[candidates[0]["id"]]["artwork"]] > 1:
            return reading

        matches = []
        for offset, crop, crop_hash in shifted_crops:
            shifted, _ = self.read(crop, field, crop_hash)
            if shifted["acceptance"] == "accepted":
                matches.append((offset, shifted))
        return self.alignment_winner(reading, matches)

    @staticmethod
    def alignment_winner(reading, matches):
        by_identity = defaultdict(list)
        for offset, result in matches:
            by_identity[result["value"]].append((offset, result))
        ranked = []
        for votes in by_identity.values():
            offset, best = max(votes, key=lambda match: match[1]["reader"]["score"])
            ranked.append(((len(votes), best["reader"]["score"]), offset, best))
        ranked.sort(key=lambda entry: entry[0], reverse=True)
        if not ranked:
            return reading
        if len(ranked) > 1 and ranked[0][0] == ranked[1][0]:
            return {**reading, **unknown("alignment views tie between identities")} if reading["acceptance"] == "accepted" else reading

        (votes, _), offset, best = ranked[0]
        return {**best, "reason": "matched reference after one-pixel alignment retry",
                "reader": {**best["reader"], "pixel_shift": list(offset), "shift_votes": votes}}

    def align_portrait(self, reading, field, shifted_crops):
        # A hidden portrait or a respawn crop with insufficient usable pixels
        # provides no starting evidence. A shifted crop cannot clear that state.
        if not reading.get("candidates"):
            return reading
        view = reading.get("reader", {}).get("view")
        views = [((0, 0), reading)]
        for offset, crop, crop_hash in shifted_crops:
            shifted, _ = self.read(crop, field, crop_hash)
            views.append((offset, shifted))
        audit, passing = [], []
        for offset, result in views:
            details = result.get("reader", {})
            candidate = next(iter(result.get("candidates", [])), {})
            eligible = result["acceptance"] == "accepted" and details.get("view") == view
            if eligible:
                passing.append((offset, result))
            audit.append({"offset": list(offset), "top_id": candidate.get("id", result.get("value")),
                          "score": candidate.get("score", details.get("score")), "margin": details.get("margin"),
                          "eligible": eligible, "view": details.get("view", "color"),
                          "reason": result.get("reason")})
        result = self.alignment_winner(reading, passing)
        if result["acceptance"] == "accepted":
            result["reason"] = "matched scoreboard portrait across center and cardinal views"
        return {**result, "reader": {**result.get("reader", {}), "alignment_views": audit}}
