"""Shared frame evidence, HUD readings and validation for the v3 action experiment."""
from __future__ import annotations

from pathlib import Path

from .action_state import STATES
from .core import atomic_json, digest, finite, implementation_hash, read_json, uid
from .intake_profile import layout_matches, load_profile
from .media import file_hash, sample_times
from .playback import PlaybackTracker


def slot_schema():
    text = {"type": "string", "maxLength": 120}
    nullable = {"type": ["string", "null"], "maxLength": 80}
    number = {"type": ["number", "null"]}
    fields = {"frame_id": text, "owner_id": text, "slot_id": text, "state": {"enum": list(STATES)},
              "seconds": number, "charges": number, "recast_visible": {"type": "boolean"},
              "charge_consumption_visible": {"type": "boolean"},
              "item_effect": {"enum": ["active", "automatic", "unknown"]}, "name": nullable, "evidence": text}
    return {"type": "array", "maxItems": 26, "items": {
        "type": "object", "additionalProperties": False, "properties": fields, "required": list(fields)}}


def validate_slots(rows, frames, profile, mode, request_id):
    """Keep HUD observations tied to supplied frames, owners and displayed numerals."""
    validate_shape(rows, slot_schema())
    known = {f["id"]: f for f in frames}
    owners = {o["id"]: {s["id"] for s in o["slots"]} for o in profile["owners"]}
    slots, seen = [], set()
    for row in rows:
        key = (row["frame_id"], row["owner_id"], row["slot_id"])
        if row["owner_id"] not in owners or row["slot_id"] not in owners[row["owner_id"]] or key in seen:
            raise ValueError("Invented or duplicate action slot")
        seen.add(key)
        if row["seconds"] is not None and (not finite(row["seconds"]) or row["seconds"] <= 0):
            raise ValueError("Invalid displayed cooldown")
        if row["state"] == "cooling_down" and row["seconds"] is None:
            raise ValueError("Cooldown requires a visible numeral")
        if row["charges"] is not None and (not finite(row["charges"]) or int(row["charges"]) != row["charges"] or row["charges"] < 0):
            raise ValueError("Invalid charge reading")
        if row["frame_id"] not in known:
            raise ValueError("Invented action frame citation")
        if not row["evidence"].strip():
            raise ValueError("Invalid HUD owner/evidence")
        frame = known[row["frame_id"]]
        slots.append({**row, "id": uid("action-slot-reading", request_id, row), "time": frame["time"],
                      "segment_id": frame["segment_id"], "owner_epoch": frame["owner_epochs"][row["owner_id"]],
                      "mode": mode, "request_id": request_id, "frame_ids": [frame["id"]]})
    return slots


def validate_shape(value, schema):
    """Validate the small JSON-schema subset we emit without a new driver dependency."""
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError("Value outside action enum")
    types = schema.get("type", [])
    types = [types] if isinstance(types, str) else types
    matches = {"object": isinstance(value, dict), "array": isinstance(value, list),
               "string": isinstance(value, str), "number": finite(value),
               "boolean": type(value) is bool, "null": value is None}
    if types and not any(matches[t] for t in types):
        raise ValueError("Invalid action JSON type")
    if isinstance(value, dict):
        if set(value) != set(schema["properties"]):
            raise ValueError("Missing or invented action response fields")
        for key, child in value.items():
            validate_shape(child, schema["properties"][key])
    if isinstance(value, list):
        if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", 1000):
            raise ValueError("Invalid action list length")
        for child in value:
            validate_shape(child, schema["items"])
    if isinstance(value, str) and len(value) > schema.get("maxLength", 2000):
        raise ValueError("Action text too long")


def numeric_read(image, box, directory, confidence):
    """Read center-overlay numerals, excluding hotkey/rank/cost locations."""
    import numpy as np
    from PIL import Image, ImageOps
    from .readers import tesseract
    pixels = np.asarray(image.crop(box).convert("RGB"))
    mask = (pixels.min(axis=2) >= 165) & (pixels.max(axis=2) - pixels.min(axis=2) < 70)
    prepared = ImageOps.expand(Image.fromarray(np.where(mask, 0, 255).astype("uint8")).resize(
        (mask.shape[1] * 4, mask.shape[0] * 4), Image.Resampling.NEAREST), border=8, fill="white")
    key = digest([prepared.size, prepared.tobytes().hex()])
    path = directory / (key + ".json")
    if path.exists():
        return read_json(path)
    directory.mkdir(parents=True, exist_ok=True)
    png = directory / (key + ".png")
    prepared.save(png)
    raw = tesseract(png, whitelist="0123456789.:")
    from .readers import parse_text
    value = None
    try:
        value = parse_text(raw["raw_text"], "cooldown")["seconds"] if raw["score"] >= confidence else None
    except ValueError:
        pass
    result = {**raw, "seconds": value if value and value > 0 else None, "processed_path": str(png)}
    atomic_json(path, result)
    return result


def prepare_inputs(store, config, start, end, profile, policy, *, cache_identity=None,
                   namespace="actions-v3", max_duration=30):
    import numpy as np
    from .intake import sampled_frames
    from .media import probe
    manifest = read_json(store.path("manifest.json"))
    lower, upper = manifest["interval"]
    if (not finite(start) or not finite(end) or not lower <= start < end <= upper
            or (max_duration is not None and end-start > max_duration)):
        raise ValueError("Require an in-source interval within the preparation duration limit")
    if namespace not in {"actions-v3", "actions-v3-hud-grounded", "captions"}:
        raise ValueError("Unsupported evidence namespace")
    source = manifest["source"]
    if list((source["width"], source["height"])) != profile["size"] or file_hash(Path(source["path"])) != source["sha256"]:
        raise ValueError("Action source hash/geometry mismatch")
    if probe(Path(source["path"]))["fps"] < policy["hud_fps"]:
        raise ValueError("Source FPS is below the action HUD sampling rate")
    components = ["action_pipeline.py", "action_state.py", "action_icons.py", "playback.py", "media.py", "readers.py", "numeric_reader.py"]
    components += ["action_v3.py", "action_body.py", "action_events.py"]
    if namespace == "actions-v3-hud-grounded":
        components.append("action_hud_replay.py")
    recipe = {"schema_version": 3, "source_sha256": source["sha256"], "interval": [start, end],
              "profile": profile, "policy": policy, "masks": config.masks,
              "implementation": implementation_hash(components)}
    if cache_identity is not None:
        recipe["cache_identity"] = cache_identity
    root = store.path("experiments/" + namespace + "/" + digest(recipe)[:20])
    audit_path = root / "input-audit.json"
    if audit_path.exists():
        audit = read_json(audit_path)
        if all(Path(f["input_path"]).is_file() and file_hash(Path(f["input_path"])) == f["input_sha256"]
               and Path(f["native_path"]).is_file() and file_hash(Path(f["native_path"])) == f.get("native_sha256") for f in audit["frames"]):
            return audit
    root.mkdir(parents=True, exist_ok=True)
    anchors = load_profile(profile["anchor_profile"])
    tracker = PlaybackTracker(source["sha256"], policy["hud_fps"], scene_box=profile["scene_box"],
                              masks=config.masks, policy=policy.get("playback"), retain_signature=namespace == "captions")
    frames, last_portraits, epochs = [], {}, {o["id"]: 0 for o in profile["owners"]}
    last_clock = None
    for index, (pts, image) in enumerate(sampled_frames(source["path"], start, end, policy["hud_fps"], store.check_stop)):
        frame = save_frame(root, source["sha256"], pts, image, profile, config.masks)
        valid, scores = layout_matches(image, anchors)
        if index % policy["hud_fps"] == 0:
            # The displayed clock is continuity evidence only; never a cast timestamp.
            from .numeric_reader import numeric_preprocess
            from .readers import tesseract, parse_text
            clock_path = root / "clock.png"
            numeric_preprocess(image.crop(profile["clock_box"])).save(clock_path)
            try:
                last_clock = parse_text(tesseract(clock_path, whitelist="0123456789:")["raw_text"], "clock")["seconds"]
            except ValueError:
                last_clock = None
        playback = tracker.update(pts, last_clock, image, valid)
        for owner in profile["owners"]:
            portrait = np.asarray(image.crop(owner["identity_box"]).resize((24, 24)), dtype=float)
            if owner["id"] in last_portraits and np.abs(portrait - last_portraits[owner["id"]]).mean() / 255 > profile["thresholds"]["identity_change"]:
                epochs[owner["id"]] += 1
                last_portraits[owner["id"]] = portrait
            elif owner["id"] not in last_portraits:
                last_portraits[owner["id"]] = portrait
        frame.update(segment_id=playback["playback_segment_id"], playback=playback, layout_valid=valid,
                     source_fps=source["fps"],
                     owner_epochs=dict(epochs), scene_sample=index % (policy["hud_fps"] // policy["scene_fps"]) == 0,
                     anchors=scores)
        frames.append(frame)
    if len(frames) != len(sample_times(start, end, policy["hud_fps"])):
        raise ValueError("Action source decode incomplete")
    audit = {**recipe, "source": source, "root": str(root), "frames": frames}
    atomic_json(audit_path, audit)
    return audit


def save_frame(root, source, pts, image, profile, masks):
    from PIL import Image, ImageDraw
    frame_id = uid("action-frame", source, pts)
    native_path, input_path = root / "native" / (frame_id + ".png"), root / "inputs" / (frame_id + ".png")
    native_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.parent.mkdir(exist_ok=True)
    image.save(native_path)
    masked = image.copy()
    draw = ImageDraw.Draw(masked)
    for box in masks:
        draw.rectangle(box, fill="black")
    scene = masked.crop(profile["scene_box"])
    scene = scene.resize((960, round(scene.height * 960 / scene.width)), Image.Resampling.LANCZOS)
    hud = masked.crop(profile["hud_box"])
    combined = Image.new("RGB", (960, scene.height + hud.height + 40), "black")
    combined.paste(scene, (0, 20))
    combined.paste(hud, (0, scene.height + 40))
    draw = ImageDraw.Draw(combined)
    draw.text((4, 3), f"{frame_id} | source {pts:.6f}s", fill="white")
    draw.text((4, scene.height + 23), "same-frame native HUD", fill="white")
    combined.save(input_path)
    return {"id": frame_id, "time": pts, "native_path": str(native_path), "native_sha256": file_hash(native_path), "input_path": str(input_path),
            "input_sha256": file_hash(input_path)}


def packets(frames):
    group = []
    for frame in frames:
        if group and (group[-1]["segment_id"] != frame["segment_id"] or group[-1]["owner_epochs"] != frame["owner_epochs"]):
            yield group
            group = []
        group.append(frame)
        if len(group) == 8:
            yield group
            group = group[-1:]
    if group and (len(group) > 1 or len(frames) == 1):
        yield group


def scan_hud(store, audit, reader=None, mode="unassisted"):
    from PIL import Image
    import numpy as np
    root, profile = Path(audit["root"]), audit["profile"]
    previous, rows, pending = {}, [], {}
    for frame in audit["frames"]:
        store.check_stop()
        with Image.open(frame["native_path"]) as image:
            for owner in profile["owners"]:
                for slot in owner["slots"]:
                    pixels = np.asarray(image.crop(slot["box"]).resize((24, 24)), dtype=float)
                    key = (frame["segment_id"], frame["owner_epochs"][owner["id"]], owner["id"], slot["id"])
                    old = previous.get(key)
                    change = float(np.abs(pixels - old["pixels"]).mean() / 255) if old else 1
                    raw = numeric_read(image, slot["countdown_box"], root / "ocr", profile["thresholds"]["ocr_confidence"]) if frame["layout_valid"] else {"seconds": None, "raw_text": "", "score": 0}
                    state = "cooling_down" if raw["seconds"] is not None else "unknown"
                    row = {"id": uid("hud-reading", frame["id"], owner["id"], slot["id"], mode),
                           "frame_id": frame["id"], "frame_ids": [frame["id"]], "time": frame["time"],
                           "segment_id": frame["segment_id"], "owner_id": owner["id"],
                           "owner_epoch": frame["owner_epochs"][owner["id"]], "slot_id": slot["id"], "slot_kind": slot["kind"],
                           "state": state, "seconds": raw["seconds"], "charges": None, "item_epoch": 0,
                           "item_effect": "unknown", "name": None, "pixel_change": change, "raw": raw, "mode": mode,
                           "evidence": "OCR center-overlay numeral" if state == "cooling_down" else "No verified cooldown reading"}
                    if old and change <= profile["thresholds"]["pixel_change"] and state == "unknown":
                        # Copy only an explicitly observed available state with compatible pixels.
                        row["state"] = old["row"]["state"] if old["row"]["state"] in ("ready", "unavailable") else "unknown"
                    if state == "unknown" and frame["layout_valid"] and (not old or change > profile["thresholds"]["pixel_change"]):
                        pending.setdefault(frame["id"], set()).add((owner["id"], slot["id"]))
                    rows.append(row)
                    previous[key] = {"pixels": pixels, "row": row}
    if reader and pending:
        inspected = reader.read([f for f in audit["frames"] if f["id"] in pending], mode, "hud", slot_targets=pending)
    else:
        inspected = {"slots": [], "identities": [], "actions": [], "uncertainty": []}
    return merge_slot_reads(rows, inspected["slots"]), inspected


def merge_slot_reads(rows, proposals):
    from collections import defaultdict
    by_key = defaultdict(list)
    for p in proposals:
        by_key[(p["frame_id"], p["owner_id"], p["slot_id"])].append(p)
    result, last, item_names = [], {}, {}
    for original in rows:
        row = dict(original)
        choices = by_key.get((row["frame_id"], row["owner_id"], row["slot_id"]), [])
        states = {p["state"] for p in choices}
        if len(states) > 1 or (choices and row["state"] == "cooling_down" and choices[0]["state"] not in ("cooling_down", "unknown")):
            row.update(state="unknown", conflict=True)
        elif choices:
            chosen = choices[0]
            for field in ("state", "seconds", "charges", "item_effect", "name", "evidence", "recast_visible", "charge_consumption_visible"):
                if field in ("state", "seconds", "evidence") and chosen["state"] == "unknown" and row["state"] == "cooling_down":
                    continue
                if chosen.get(field) is not None:
                    row[field] = chosen[field]
            row["reader_evidence_ids"] = [p["id"] for p in choices]
        key = (row["segment_id"], row["owner_epoch"], row["owner_id"], row["slot_id"])
        old = last.get(key)
        if old and row["state"] == "unknown" and not row.get("conflict") and row["pixel_change"] <= .02:
            if old["state"] in ("ready", "unavailable"):
                row["state"] = old["state"]
                row["carried_from"] = old["id"]
        known_item = item_names.get(key)
        changed_item = bool(row["slot_kind"] == "item" and known_item and row.get("name") and known_item != row["name"])
        row["item_epoch"] = (old.get("item_epoch", 0) if old else 0) + int(changed_item)
        if row["slot_kind"] == "item" and row.get("name"):
            item_names[key] = row["name"]
        last[key] = row
        result.append(row)
    return result


def native_review(store, audit, window, config):
    from .intake import sampled_frames
    import numpy as np
    from PIL import Image
    root, result = Path(audit["root"]), []
    anchors = load_profile(audit["profile"]["anchor_profile"])
    gap = False
    for pts, image in sampled_frames(audit["source"]["path"], window["native_start"], window["native_end"],
                                    audit["source"]["fps"], store.check_stop):
        neighbor = min(audit["frames"], key=lambda f: abs(f["time"] - pts))
        if neighbor["segment_id"] != window["segment_id"] or not neighbor["layout_valid"]:
            gap = True
            continue
        valid, _ = layout_matches(image, anchors)
        with Image.open(neighbor["native_path"]) as sampled:
            identity_change = any(float(np.abs(
                np.asarray(image.crop(o["identity_box"]).resize((24, 24)), dtype=float)
                - np.asarray(sampled.crop(o["identity_box"]).resize((24, 24)), dtype=float)).mean())/255
                > audit["profile"]["thresholds"]["identity_change"] for o in audit["profile"]["owners"])
        if not valid or identity_change:
            audit.setdefault("native_rejections", []).append({"time": pts, "reason": "layout_or_selection_uncertain"})
            gap = True
            continue
        frame = save_frame(root, audit["source_sha256"], pts, image, audit["profile"], config.masks)
        frame.update(segment_id=neighbor["segment_id"], owner_epochs=neighbor["owner_epochs"], layout_valid=True,
                     source_fps=audit["source"]["fps"], playback=neighbor.get("playback", {}),
                     scene_sample=False, native_review=True)
        if gap:
            frame["segment_id"] = uid("native-uncertain-continuity", neighbor["segment_id"], pts)
        result.append(frame)
        if not any(f["id"] == frame["id"] for f in audit["frames"]):
            audit["frames"].append(frame)
    audit["frames"].sort(key=lambda f: f["time"])
    atomic_json(root / "input-audit.json", audit)
    return result
