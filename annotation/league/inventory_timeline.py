"""Build observed inventories and same-segment last-seen views from corrected crops."""

from collections import Counter, defaultdict

from .core import uid
from .normalization import accepted


ITEM_SLOTS = tuple(str(index) for index in range(6)) + ("trinket",)


def inventory_timeline(observations):
    samples = defaultdict(list)
    names = defaultdict(dict)
    for row in observations:
        if row.get("backend") == "inventory":
            samples[(row["time"], row["task_id"])].append(row)
        elif row["field"] == "player_name" and row["route"] == "scoreboard" and accepted(row):
            names[row["task_id"]][row["slot_id"]] = row

    snapshots, timeline = [], []
    previous_segment = None
    last_seen = {}
    for (time, task_id), readings in sorted(samples.items()):
        segment = readings[0].get("playback_segment_id")
        if segment is None or segment != previous_segment:
            last_seen.clear()
        previous_segment = segment
        groups = defaultdict(dict)
        panel_states = {}
        for row in readings:
            route, owner = row["route"], row["slot_id"]
            panel_states[route] = row["panel_state"]
            if row["panel_state"] == "present":
                groups[(route, owner)][row["field"]] = row

        portraits = {}
        for key, fields in groups.items():
            portrait = fields.get("inventory.champion")
            if portrait and accepted(portrait) and isinstance(portrait["value"], str):
                portraits[key] = portrait["value"]
        counts = Counter((route, owner.split(":")[0], champion) for (route, owner), champion in portraits.items())

        current = []
        for (route, owner), fields in sorted(groups.items()):
            side = owner.split(":")[0] if route == "scoreboard" else None
            champion = portraits.get((route, owner))
            if side and counts[(route, side, champion)] > 1:
                champion = None
            name = names[task_id].get(owner) if side else None
            slots = []
            for slot in ITEM_SLOTS:
                observation = fields.get(f"inventory.{slot}")
                status, identifier = "unknown", None
                if observation and observation.get("review_status") != "rejected":
                    if accepted(observation) and isinstance(observation["value"], str):
                        status, identifier = "recognized", observation["value"]
                    elif observation["acceptance"] == "absent":
                        status = "empty"
                slots.append({"slot": slot, "status": status, "item_id": identifier,
                              "observation_id": observation["id"] if observation else None,
                              "crop_path": observation["evidence_path"] if observation else None})
            evidence = list(fields.values()) + ([name] if name else [])
            portrait = fields.get("inventory.champion")
            snapshot = {"id": uid("inventory", task_id, route, owner), "time": time, "route": route,
                        "slot_id": owner, "side": side, "champion": champion,
                        "skin_id": portrait.get("skin_id") if champion and portrait else None,
                        "skin_catalog_id": portrait.get("skin_catalog_id") if portrait else None,
                        "portrait_crop_path": portrait["evidence_path"] if portrait else None,
                        "player_name": name["value"] if name else None, "playback_segment_id": segment,
                        "catalog_id": evidence[0]["inventory_catalog_id"], "slots": slots,
                        "evidence_ids": [row["id"] for row in evidence], "frame_ids": [row["frame_id"] for row in evidence]}
            if portrait and (views := portrait.get("reader", {}).get("alignment_views")):
                snapshot["portrait_alignment_views"] = views
            current.append(snapshot["id"])
            snapshots.append(snapshot)
            if side and champion and segment is not None:
                last_seen[(side, champion)] = snapshot["id"]

        scoreboard = panel_states.get("scoreboard", "unknown")
        timeline.append({"id": uid("inventory-sample", task_id), "time": time,
                         "end": time + 1 / readings[0]["sampling_fps"], "playback_segment_id": segment,
                         "scoreboard": scoreboard, "snapshot_ids": current,
                         "last_seen_ids": sorted(last_seen.values()) if scoreboard == "absent" else []})
    return snapshots, timeline
