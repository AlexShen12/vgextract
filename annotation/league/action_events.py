"""Evidence-bounded counters, creep credit and recall lifecycles."""
from collections import defaultdict
import re

from .core import atomic_json, digest, read_json, uid
from .numeric_reader import NumericReader, numeric_preprocess
from .readers import tesseract


def observed(row, kind, start, end, evidence):
    return {"schema_version": 3, "id": uid(kind, *[r["id"] for r in evidence]), "kind": kind,
            "mode": "core", "owner_id": row["owner_id"], "owner_epoch": row["owner_epoch"],
            "segment_id": row["segment_id"], "champion": row.get("champion"),
            "actor_track": row.get("actor_track"), "start": start, "end": end,
            "evidence_ids": [r["id"] for r in evidence], "frame_ids": [r["frame_id"] for r in evidence],
            "evidence_class": "hud-supported", "experimental": True, "review_status": "unreviewed",
            "actor_description": "Selected HUD owner", "outcome": None}


def read_counters(image, frame, profile, root, identity):
    rows = []
    for field in profile["counter_fields"]:
        crop = image.crop(field["box"])
        recipe = ["counter-v3", field, crop.size, crop.convert("RGB").tobytes().hex(), .85]
        key = digest(recipe)
        directory = root/"counter-ocr"/key
        path = directory/"reading.json"
        if path.exists():
            raw = read_json(path)
        else:
            raw = NumericReader(.85).read(crop, {"reader": "number", "parser": field["parser"]}, directory)
            atomic_json(path, raw)
        value = raw["value"] if raw["acceptance"] == "accepted" else None
        if isinstance(value, dict):
            value = value.get("amount")
        # Abbreviated gold displays cannot support exact integer changes.
        if value is not None and (not isinstance(value, (int, float)) or int(value) != value or value < 0
                                  or re.search(r"[.kK]", raw.get("raw_text") or "")):
            value = None
        rows.append({"schema_version": 3, "id": uid("counter-reading", frame["id"], field["field"], key),
            "frame_id": frame["id"], "time": frame["time"], "owner_id": identity["owner_id"],
            "owner_epoch": frame["owner_epochs"][identity["owner_id"]], "segment_id": frame["segment_id"],
            "field": field["field"], "value": int(value) if value is not None else None,
            "layout_valid": frame["layout_valid"], "champion": identity.get("hud_champion"), "raw": raw,
            "evidence_path": str(directory), "mode": "core"})
    return rows


def counter_events(rows):
    state, result = {}, []
    for row in sorted(rows, key=lambda r: (r["time"], r["field"])):
        key = (row["segment_id"], row["owner_epoch"], row["owner_id"], row["field"])
        old = state.get(key)
        if row["value"] is None or not row.get("layout_valid", True) or row.get("identity_changed"):
            state.pop(key, None)
            continue
        if old is None or row["time"]-old["last"]["time"] > .095:
            old = {"pending": [], "confirmed": None}
        pending = old["pending"]
        if not pending or pending[-1]["value"] != row["value"]:
            pending = []
        pending.append(row)
        previous = old["confirmed"]
        if len(pending) >= 2:
            if previous is not None and row["value"] != previous["value"]:
                delta = row["value"]-previous["value"]
                if row["field"] == "displayed_gold" or delta > 0:
                    kind = "gold_change" if row["field"] == "displayed_gold" else "creep_credit"
                    event = observed(row, kind, previous["time"], pending[0]["time"], [previous]+pending[:2])
                    event.update(counter_delta=delta, reward_gold=None, target_kind="unknown", count_delta=delta if kind == "creep_credit" else None)
                    result.append(event)
            old["confirmed"] = row
        state[key] = {**old, "last": row, "pending": pending[-2:]}
    return result


def read_recall(image, frame, profile, root, identity):
    import numpy as np
    cfg = profile["recall"]
    crop = image.crop(cfg["text_box"]).convert("RGB")
    key = digest(["recall-text-v3", crop.size, crop.tobytes().hex(), cfg["labels"]])
    directory = root/"recall-ocr"/key
    path = directory/"reading.json"
    if path.exists():
        raw = read_json(path)
    else:
        directory.mkdir(parents=True, exist_ok=True)
        processed = directory/"text.png"
        numeric_preprocess(crop).save(processed)
        raw = tesseract(processed)
        atomic_json(path, raw)
    text = " ".join((raw.get("raw_text") or "").casefold().split())
    explicit = text in cfg["labels"] and raw.get("score", 0) >= .7
    p = np.asarray(image.crop(cfg["progress_box"]).convert("RGB"), dtype=float)
    text_pixels = np.asarray(crop, dtype=float)
    region_occluded = bool((text_pixels.max(axis=2) < 65).mean() >= .9 and (p.max(axis=2) < 65).mean() >= .9)
    cyan = (p[..., 1] > 90) & (p[..., 2] > 90) & (p[..., 1] > p[..., 0]*1.3)
    fraction = float((cyan.mean(axis=0) >= .3).mean())
    return {"schema_version": 3, "id": uid("recall-reading", frame["id"], key, fraction), "frame_id": frame["id"],
        "time": frame["time"], "owner_id": identity["owner_id"], "owner_epoch": frame["owner_epochs"][identity["owner_id"]],
        "segment_id": frame["segment_id"], "champion": identity.get("hud_champion"), "mode": "core",
        "active": bool(explicit and fraction > .02 and frame["layout_valid"]), "progress_fraction": fraction,
        "label_visible": explicit, "channel_bar_visible": fraction > .02,
        "region_occluded": region_occluded,
        "layout_valid": frame["layout_valid"], "raw": raw, "evidence_path": str(directory),
        "base_arrival_visible": False, "post_activity_visible": False}


def recall_events(rows, scene=()):
    active, pending, result, last = {}, defaultdict(list), [], {}
    def finish(key, row=None):
        event = active.pop(key, None)
        if event:
            event.update(terminal_status="ending_unknown", right_censored=row is None)
            if row and row.get("layout_valid"):
                context = [s for s in scene if s.get("owner_id") == event["owner_id"]
                           and s["segment_id"] == event["segment_id"] and s["owner_epoch"] == event["owner_epoch"]
                           and row["time"]-.125 <= s["start"] <= row["time"]+1]
                arrivals = [s for s in context if s["kind"] == "base_arrival"]
                activity = [s for s in context if s["kind"] in ("movement", "basic_attack", "ability_cast")]
                if (row.get("base_arrival_visible") or len(arrivals) == 1) and event.get("observed_channel_end"):
                    event.update(terminal_status="completed", right_censored=False)
                elif (row.get("post_activity_visible") or activity) and event.get("last_progress", 0) > .05:
                    event.update(terminal_status="interrupted", right_censored=False)
                event["end"] = row["time"]
                event["evidence_ids"].append(row["id"])
                event["frame_ids"].append(row["frame_id"])
                for proposal in arrivals if event["terminal_status"] == "completed" else activity if event["terminal_status"] == "interrupted" else []:
                    event["evidence_ids"] += proposal["evidence_ids"]
                    event["frame_ids"] += proposal["frame_ids"]
                    event["end"] = max(event["end"], proposal["end"])
            result.append(event)
    for row in sorted(rows, key=lambda r: r["time"]):
        key = (row["owner_id"], row["segment_id"], row["owner_epoch"])
        for previous_key in list(active):
            if previous_key[0] == row["owner_id"] and previous_key != key:
                finish(previous_key)
        gap = key in last and row["time"]-last[key]["time"] > .095
        if gap or not row.get("layout_valid", True) or row.get("identity_changed"):
            finish(key); pending[key] = []
        if (not row["active"] and key in active and (row.get("channel_bar_visible") or row.get("region_occluded")) and not row.get("label_visible")
                and row["time"]-active[key].get("last_label_time", active[key]["end"]) <= .25):
            active[key]["end"] = row["time"]
            active[key].setdefault("uncertain_frame_ids", []).append(row["frame_id"])
            last[key] = row
            continue
        if row["active"] and row.get("layout_valid", True):
            if not pending[key]:
                row = {**row, "left_censored_start": key not in last or gap}
            pending[key].append(row)
            if key not in active and len(pending[key]) >= 2:
                evidence = pending[key][-2:]
                event = observed(row, "recall", evidence[0]["time"], row["time"], evidence)
                event.update(terminal_status="channeling", left_censored=pending[key][0].get("left_censored_start", False),
                             right_censored=False, last_progress=row["progress_fraction"], observed_channel_end=False)
                event["last_label_time"] = row["time"]
                active[key] = event
            elif key in active:
                event = active[key]
                event["end"] = row["time"]
                event["evidence_ids"].append(row["id"]); event["frame_ids"].append(row["frame_id"])
                event["last_progress"] = row["progress_fraction"]
                event["last_label_time"] = row["time"]
                event["observed_channel_end"] |= row["progress_fraction"] <= .05
        else:
            if key in active and row.get("raw", {}).get("raw_text", "").strip().casefold() == "recall" and row["progress_fraction"] <= .05:
                active[key]["observed_channel_end"] = True
            finish(key, row)
            pending[key] = []
        last[key] = row
    for key in list(active):
        finish(key)
    for event in result:
        matches = [s for s in scene if s["kind"] == "recall" and s["segment_id"] == event["segment_id"]
                   and s.get("owner_id") == event["owner_id"] and s["start"] <= event["end"] and s["end"] >= event["start"]]
        if len(matches) == 1:
            event.update(evidence_class="corroborated", actor_track=matches[0]["actor_track"])
    result += [{**s, "evidence_class": "scene-only", "terminal_status": "ending_unknown", "left_censored": True,
                "right_censored": True, "experimental": True, "review_status": "unreviewed"}
               for s in scene if s["kind"] == "recall" and not any(s.get("owner_id") == e["owner_id"]
                   and s["segment_id"] == e["segment_id"] and s["start"] <= e["end"] and s["end"] >= e["start"] for e in result)]
    return result


def link_farming(events, scene, bindings):
    results = [dict(e) for e in events]
    used = set()
    for event in results:
        if event["kind"] != "creep_credit":
            continue
        owners = [b for b in bindings if b["owner_id"] == event["owner_id"] and b["segment_id"] == event["segment_id"]
                  and b["owner_epoch"] == event["owner_epoch"] and b["start"] <= event["end"] and b["end"] >= event["start"]]
        tracks = {b["actor_track"] for b in owners}
        choices = [s for s in scene if s["kind"] == "minion_last_hit" and s.get("target_kind") == "lane_minion"
                   and s.get("actor_track") in tracks and s["segment_id"] == event["segment_id"]
                   and s["start"] <= event["end"]+.5 and s["end"] >= event["start"]-.5]
        if len(choices) == 1:
            match = choices[0]
            competitors = [e for e in events if e["kind"] == "creep_credit" and e["segment_id"] == match["segment_id"]
                           and match["start"] <= e["end"]+.5 and match["end"] >= e["start"]-.5]
            if len(competitors) != 1:
                continue
            event.update(kind="minion_last_hit", evidence_class="corroborated", target_kind="lane_minion",
                         actor_track=match["actor_track"], reward_gold=match.get("reward_gold"), scene_ids=[match["id"]])
            event["evidence_ids"] += match["evidence_ids"]
            event["frame_ids"] = sorted(set(event["frame_ids"]+match["frame_ids"]))
            used.add(match["id"])
    results += [{**s, "kind": "jungle_last_hit" if s.get("target_kind") == "jungle" else "creep_last_hit_candidate" if s.get("target_kind") == "unknown" else s["kind"],
                 "evidence_class": "scene-only", "experimental": True, "review_status": "unreviewed"}
                for s in scene if s["kind"] == "minion_last_hit" and s["id"] not in used]
    return results
