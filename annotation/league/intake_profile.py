"""Versioned UI geometry. Profiles are data, never executable instructions."""

from __future__ import annotations

import base64
import copy
import io
import math
from pathlib import Path

from .core import digest, finite, read_json

PROFILE_DIR = Path(__file__).with_name("profiles")
UNSUPPORTED = {
    "minimap": "tracking reader not validated",
    "gameplay": "scene events not validated",
    "full_context": "scene events not validated",
    "event_feed": "notice presence reader not validated",
    "central_notice": "notice presence reader not validated",
}


def load_profile(name: str) -> dict:
    candidate = PROFILE_DIR / (name + ".json")
    path = candidate if Path(name).name == name and candidate.is_file() else Path(name)
    profile = read_json(path)
    validate_profile(profile)
    return profile


def validate_profile(profile: dict) -> None:
    if profile.get("schema_version") != 2 or profile.get("size") != [1920, 1080]:
        raise ValueError(
            "Unsupported profile version or geometry; calibrate a 1920x1080 profile"
        )
    fields = profile.get("fields", [])
    if not fields or len(fields) > 1000:
        raise ValueError("Profile requires 1..1000 field regions")
    ids = set()
    for field in fields:
        required = {"id", "route", "slot_id", "field", "box", "reader", "parser"}
        if (
            not isinstance(field, dict)
            or not required <= field.keys()
            or any(
                not isinstance(field[k], str) or not field[k]
                for k in required - {"box"}
            )
        ):
            raise ValueError(
                "Profile fields require stable IDs, binding, geometry and reader settings"
            )
        if field["id"] in ids:
            raise ValueError("Duplicate profile field ID")
        ids.add(field["id"])
        check_box(field["box"], profile["size"])
        if field["reader"] not in {"number", "text", "icon", "bar"}:
            raise ValueError("Unsupported field reader")
        if field.get("backend") not in {None, "vl", "tesseract", "bar", "inventory"}:
            raise ValueError("Unsupported field backend")
    inventory_fields = [field for field in fields if field.get("backend") == "inventory"]
    if inventory_fields:
        validate_inventory(profile, inventory_fields)
    if not 2 <= len(profile.get("anchors", [])) <= 10:
        raise ValueError("Profile requires at least two independent panel anchors")
    required = profile.get("required_anchors", len(profile["anchors"]))
    if type(required) is not int or not 2 <= required <= len(profile["anchors"]):
        raise ValueError("Layout validation must require at least two panel anchors")
    thresholds = profile.get("thresholds", {})
    if any(
        type(thresholds.get(k)) not in (int, float)
        or not math.isfinite(thresholds[k])
        or not 0 <= thresholds[k] <= 1
        for k in (
            ("number",)
            if profile.get("capabilities")
            else ("number", "text", "icon", "icon_margin")
        )
    ):
        raise ValueError("Reader thresholds must be finite proportions")
    for anchor in profile["anchors"]:
        check_box(anchor["box"], profile["size"])
        data = base64.b64decode(anchor["png"], validate=True)
        expected = (
            anchor["box"][2] - anchor["box"][0],
            anchor["box"][3] - anchor["box"][1],
        )
        if (
            data[:8] != b"\x89PNG\r\n\x1a\n"
            or (int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big"))
            != expected
        ):
            raise ValueError("Anchor reference PNG differs from its geometry")
        limit = anchor.get("maximum_error", 0.12)
        if (
            type(limit) not in (int, float)
            or not math.isfinite(limit)
            or not 0 <= limit <= 0.25
        ):
            raise ValueError("Anchor error tolerance must be in [0, .25]")


def validate_inventory(profile, fields):
    from .inventory_references import reference_root

    inventory = profile.get("inventory", {})
    reference_root(Path(), inventory.get("reference_version"))
    if inventory.get("communitydragon_version"):
        from .reference_bundle import versions
        versions(inventory["reference_version"], inventory["communitydragon_version"])
    panels, thresholds = inventory.get("panels", {}), inventory.get("thresholds", {})
    if type(inventory.get("scoreboard_portrait_alignment", False)) is not bool:
        raise ValueError("Scoreboard portrait alignment must be a boolean")
    for route in ("selected_hud", "scoreboard"):
        check_box(panels.get(route), profile["size"])
        gates = thresholds.get(route, {})
        if not {"item", "champion"} <= gates.keys():
            raise ValueError("Inventory requires item and portrait thresholds for each surface")
        for gate in gates.values():
            score, margin = gate.get("score"), gate.get("margin")
            if not finite(score) or not 0 < score <= 1 or not finite(margin) or not 0 <= margin <= 1:
                raise ValueError("Inventory thresholds require a positive score and a nonnegative margin")
    background = inventory.get("empty_rgb", [])
    if len(background) != 3 or any(type(channel) is not int or not 0 <= channel <= 255 for channel in background):
        raise ValueError("Inventory requires a calibrated empty-slot color")
    for encoded in inventory.get("hidden_portraits", []):
        if not base64.b64decode(encoded, validate=True).startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("Hidden portrait templates must be PNGs")
    exemplar = inventory.get("selected_portrait")
    if exemplar:
        if (exemplar.get("time") not in profile.get("calibration_frames", [])
                or not profile.get("calibration_source_sha256") or not exemplar.get("id")
                or len(exemplar.get("evidence", {})) < 3
                or not base64.b64decode(exemplar["png"], validate=True).startswith(b"\x89PNG\r\n\x1a\n")):
            raise ValueError("Selected portrait exemplar requires same-source development evidence")
        for box in exemplar["evidence"].values():
            check_box(box, profile["size"])
    rows = {("selected_hud", "selected")} | {("scoreboard", f"{side}:row{i}") for side in ("left", "right") for i in range(1, 6)}
    expected = {(route, row, f"inventory.{slot}") for route, row in rows for slot in (*range(6), "trinket", "champion")}
    actual = {(field["route"], field["slot_id"], field["field"]) for field in fields}
    if actual != expected or len(fields) != len(expected):
        raise ValueError("Inventory requires a portrait, six items and a trinket for each calibrated row")
    for field in fields:
        x1, y1, x2, y2 = field["box"]
        left, top, right, bottom = panels[field["route"]]
        kind = "champion" if field["field"] == "inventory.champion" else "item"
        if field.get("kind") != kind or field["reader"] != "icon" or not (left <= x1 < x2 <= right and top <= y1 < y2 <= bottom):
            raise ValueError("Inventory crop must be a valid icon region within its panel")


def check_box(box, size):
    if (
        not isinstance(box, list)
        or len(box) != 4
        or any(type(v) is not int for v in box)
        or not 0 <= box[0] < box[2] <= size[0]
        or not 0 <= box[1] < box[3] <= size[1]
    ):
        raise ValueError("Invalid native-pixel region")


def layout_matches(image, profile):
    import numpy as np
    from PIL import Image

    if list(image.size) != profile["size"]:
        return False, []
    scores = []
    for anchor in profile["anchors"]:
        reference = Image.open(io.BytesIO(base64.b64decode(anchor["png"]))).convert(
            "RGB"
        )
        actual = image.crop(anchor["box"]).convert("RGB")
        error = float(
            np.abs(
                np.asarray(actual, dtype=float) - np.asarray(reference, dtype=float)
            ).mean()
            / 255
        )
        scores.append(
            {
                "name": anchor["name"],
                "error": error,
                "matched": error <= anchor.get("maximum_error", 0.12),
            }
        )
    return (
        sum(s["matched"] for s in scores)
        >= profile.get("required_anchors", len(scores)),
        scores,
    )


def calibrated_profile(profile, fields, image):
    from PIL import Image

    result = copy.deepcopy(profile)
    known = {f["id"]: f for f in result["fields"]}
    if set(fields) - set(known):
        raise ValueError("Calibration contains unknown field IDs")
    for name, box in fields.items():
        check_box(box, result["size"])
        known[name]["box"] = box
    if list(image.size) != result["size"]:
        raise ValueError("Calibration source resolution differs from profile")
    for anchor in result["anchors"]:
        buf = io.BytesIO()
        image.crop(anchor["box"]).save(buf, format="PNG")
        anchor["png"] = base64.b64encode(buf.getvalue()).decode()
    result["parent"] = digest(profile)
    result["revision"] = profile.get("revision", 1) + 1
    result["thresholds_frozen"] = False
    result["calibrated"] = True
    result.pop("development_labels_hash", None)
    validate_profile(result)
    return result
