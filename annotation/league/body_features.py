"""Versioned crop preparation and evidence from frozen DINO reference features."""

from collections import defaultdict

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from .core import digest


def normalize_crop(image, mask, padding=.1, size=224):
    if image.size != mask.size or not 0 <= padding < .5:
        raise ValueError("Body crop and mask need aligned geometry and valid padding")
    mask = mask.convert("L").point(lambda value: 255 if value > 127 else 0)
    box = mask.getbbox()
    if box is None:
        raise ValueError("Body crop needs visible foreground")
    width, height = box[2] - box[0], box[3] - box[1]
    scale = size * (1 - 2 * padding) / max(width, height)
    scaled = max(1, round(width * scale)), max(1, round(height * scale))
    offset = tuple((size - value) // 2 for value in scaled)
    foreground = mask.crop(box)
    body = Image.composite(image.convert("RGB").crop(box), Image.new("RGB", (width, height), (127, 127, 127)), foreground)
    output = Image.new("RGB", (size, size), (127, 127, 127))
    output_mask = Image.new("L", (size, size))
    output.paste(body.resize(scaled, Image.Resampling.LANCZOS), offset)
    output_mask.paste(foreground.resize(scaled, Image.Resampling.NEAREST), offset)
    return output, output_mask, {
        "source_box": list(box), "offset": list(offset), "scale": [scaled[0] / width, scaled[1] / height],
        "native_size": [width, height], "padding": padding,
    }


def render_variant(image, mask, *, body_height=None, blur=0, brightness=1, jpeg_quality=None):
    import io

    mask = mask.convert("L")
    box = mask.getbbox()
    if not box or image.size != mask.size:
        raise ValueError("Render variant needs aligned nonempty foreground")
    image = Image.composite(image.convert("RGB"), Image.new("RGB", image.size, (127, 127, 127)), mask)
    if body_height is not None:
        ratio = body_height / (box[3] - box[1])
        size = tuple(max(1, round(n * ratio)) for n in image.size)
        image = image.resize(size, Image.Resampling.LANCZOS)
        mask = mask.resize(size, Image.Resampling.NEAREST)
    if blur:
        image = image.filter(ImageFilter.GaussianBlur(blur))
    if brightness != 1:
        image = ImageEnhance.Brightness(image).enhance(brightness)
    if jpeg_quality is not None:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=jpeg_quality)
        with Image.open(buffer) as decoded:
            image = decoded.convert("RGB")
    return image, mask


def patch_similarity(query, reference):
    if len(query) < 4 or len(reference) < 4:
        return 0.0, 0.0, []
    scores = np.asarray(query, dtype=np.float32) @ np.asarray(reference, dtype=np.float32).T
    forward = scores.argmax(axis=1)
    backward = scores.argmax(axis=0)
    pairs = [(a, int(b)) for a, b in enumerate(forward) if backward[b] == a]
    support = 2 * len(pairs) / (len(query) + len(reference))
    similarity = float(np.mean([(float(scores[a, b]) + 1) / 2 for a, b in pairs])) if pairs else 0.0
    return similarity * support, support, pairs


def silhouette_similarity(query, reference):
    first, second = np.asarray(query, dtype=bool), np.asarray(reference, dtype=bool)
    padded = np.pad(second, 1)
    scores = []
    for y in range(3):
        for x in range(3):
            shifted = padded[y:y + first.shape[0], x:x + first.shape[1]]
            intersection = (first & shifted).sum()
            union = first.sum() + second.sum() - intersection
            scores.append(float(intersection / max(1, union)))
    return max(scores)


class FeatureGallery:
    def __init__(self, entries, features, *, encoder_signature, excluded_groups=(), kind="combined"):
        from .body_identity import BodyGallery

        if len(entries) != len(features):
            raise ValueError("Each body reference requires one feature record")
        self.base = BodyGallery(entries, [f["rgb"] for f in features], encoder_signature=encoder_signature,
                                excluded_groups=excluded_groups, kind=kind)
        indices = [index for index, entry in enumerate(entries) if kind == "combined" or entry["reference_kind"] == kind]
        self.features = [features[index] for index in indices]
        self.signature = digest({"gallery": self.base.signature, "features": self.features})

    def match(self, query, policy, roster=None):
        from .body_identity import unit

        if not policy.get("development_freeze"):
            raise ValueError("Body feature matching requires a frozen policy")
        eligible = [i for i, entry in enumerate(self.base.entries) if roster is None or entry["champion"] in roster]
        if not eligible:
            return {"champion": None, "skin_id": None, "reason": "no compatible references"}
        initial = (self.base.matrix[eligible] @ unit(query["rgb"]) + 1) / 2
        by_champion = defaultdict(list)
        for index, score in zip(eligible, initial):
            by_champion[self.base.entries[index]["champion"]].append((index, float(score)))
        shortlist = sorted(by_champion, key=lambda name: (-max(score for _, score in by_champion[name]), name))[:5]
        weights = policy.get("weights", [1, 0, 0])
        local_weight = policy.get("patch_weight", 0)
        use_local = local_weight > 0 and len(query.get("patches", [])) >= 4
        threshold = policy["score"] if use_local or not local_weight else policy["global_score"]
        margin_threshold = policy["margin"] if use_local or not local_weight else policy["global_margin"]
        ranked = []
        top_k = policy.get("agreement", 1)
        quota = policy.get("template_quota")
        if quota is not None:
            quota = min(quota, *(len({self.template(i) for i, _ in by_champion[name]}) for name in shortlist))

        for name in shortlist:
            candidates = sorted(by_champion[name], key=lambda row: (-row[1], self.base.entries[row[0]]["id"]))
            if quota is not None:
                admitted = set(list(dict.fromkeys(self.template(i) for i, _ in candidates))[:quota])
                candidates = [(i, s) for i, s in candidates if self.template(i) in admitted]
            skins = defaultdict(dict)
            for index, rgb in candidates:
                entry, reference = self.base.entries[index], self.features[index]
                gray = float((np.dot(query["gray"], reference["gray"]) + 1) / 2) if weights[1] else 0.0
                shape = silhouette_similarity(query["silhouette"], reference["silhouette"]) if weights[2] else 0.0
                patch, support, pairs = patch_similarity(query["patches"], reference.get("patches", [])) if use_local else (0.0, 0.0, [])
                score = weights[0] * rgb + weights[1] * gray + weights[2] * shape
                if use_local:
                    score = (1 - local_weight) * score + local_weight * patch
                component = {"reference_id": entry["id"], "reference_sha256": entry["sha256"],
                             "skin_id": entry.get("skin_id"), "score": score, "rgb": rgb, "gray": gray,
                             "silhouette": shape, "patch": patch, "patch_support": support,
                             "patch_pairs": [[query["patch_coordinates"][a], reference["patch_coordinates"][b]]
                                             for a, b in pairs]}
                templates = skins[entry.get("skin_id")]
                key = self.template(index)
                if key not in templates or score > templates[key]["score"]:
                    templates[key] = component
            supported = []
            for templates in skins.values():
                evidence = sorted(templates.values(), key=lambda row: (-row["score"], row["reference_id"]))[:top_k]
                if len(evidence) == top_k:
                    supported.append((sum(row["score"] for row in evidence) / top_k, evidence))
            if supported:
                score, evidence = max(supported, key=lambda pair: pair[0])
                ranked.append((score, name, evidence))
        if not ranked:
            return {"champion": None, "skin_id": None, "reason": "insufficient distinct references"}
        ranked.sort(key=lambda row: (-row[0], row[1]))
        score, champion, evidence = ranked[0]
        margin = score - (ranked[1][0] if len(ranked) > 1 else 0)
        accepted = score >= threshold and margin >= margin_threshold
        return {"champion": champion if accepted else None, "skin_id": None, "score": score, "margin": margin,
                "reason": "accepted appearance" if accepted else "ambiguous appearance", "gallery_id": self.signature,
                "best_reference_skin": evidence[0]["skin_id"], "supporting_references": evidence,
                "scores_are_probabilities": False, "used_local_features": use_local,
                "candidates": [{"champion": name, "score": score} for score, name, _ in ranked[:5]]}

    def template(self, index):
        entry = self.base.entries[index]
        return entry.get("parent_template_id", entry["id"])


def encode_features(encoder, image, mask, policy):
    if policy.get("crop") == "legacy":
        foreground = mask.convert("L").point(lambda value: 255 if value > 127 else 0)
        body = Image.composite(image.convert("RGB"), Image.new("RGB", image.size, (127, 127, 127)), foreground)
        body = body.resize((224, 224))
        foreground = foreground.resize((224, 224), Image.Resampling.NEAREST)
        geometry = {"source_box": [0, 0, *image.size], "native_size": list(image.size),
                    "offset": [0, 0], "scale": [224 / value for value in image.size], "padding": 0}
    else:
        body, foreground, geometry = normalize_crop(image, mask, padding=policy.get("padding", .1))
    vector, patches = encoder.describe(body, policy.get("patch_layer"))
    result = {"rgb": vector.tolist(), "geometry": geometry, "patches": [], "patch_coordinates": []}
    if policy.get("weights", [1, 0, 0])[1]:
        result["gray"] = encoder.describe(ImageOps.grayscale(body).convert("RGB"))[0].tolist()
    result["silhouette"] = (np.asarray(foreground.resize((32, 32), Image.Resampling.NEAREST)) > 127).tolist()
    if patches is not None and min(geometry["native_size"]) >= policy.get("minimum_native_size", 24):
        occupancy = np.asarray(foreground.resize((16, 16), Image.Resampling.BOX)) >= 128
        result["patches"] = patches[occupancy.ravel()].tolist()
        result["patch_coordinates"] = np.argwhere(occupancy).tolist()
    return result
