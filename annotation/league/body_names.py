"""Resolve confirmed player names before requesting any body appearance features."""

from collections import Counter, defaultdict
import math

from .body_identity import TrackIdentity, canonical_champion
from .body_roster import name_key
from .core import uid
from .normalization import accepted


def select_frames(samples, *, prefer_name_region=False, cadence="quality"):
    """Select a source frame per track and quarter-second window, independent of identity."""
    selected = {}
    for row in samples:
        bucket = math.floor(row["time"] * 4 + 1e-8)
        key = (row["source_sha256"], row["track_id"], row["segment_id"],
               row["owner_epoch"], row.get("track_epoch", 0), bucket)
        candidate = {**row, "vote_time": (bucket + 1) / 4}
        rank = ((-row["time"], row["id"]) if cadence == "fixed" else
                (bool(row.get("visible")), prefer_name_region and bool(row.get("name_box")),
                 row.get("quality", 0), -row["time"], row["id"]))
        if key not in selected or rank > selected[key][0]:
            selected[key] = rank, candidate
    return sorted((value[1] for value in selected.values()), key=lambda row: (row["vote_time"], row["time"], row["id"]))


def attached_name(sample, reading):
    if not reading or not sample.get("visible") or not accepted(reading):
        return ""
    if (reading.get("associated") is not True or reading.get("name_kind") != "player"
            or reading.get("source_sha256") != sample["source_sha256"]
            or reading.get("frame_id") != sample["frame_id"]
            or reading.get("track_id") != sample["track_id"]
            or reading.get("time") != sample["time"]):
        return ""
    return name_key(reading.get("value"))


def confirm_names(samples, readings):
    previous = {}
    confirmed = {}
    for row in samples:
        reading = readings.get(row["id"])
        key = attached_name(row, reading)
        context = (row["source_sha256"], row["scope"], row["track_id"],
                   row["segment_id"], row["owner_epoch"], row.get("track_epoch", 0))
        prior = previous.pop(context, None)
        if not key:
            continue
        previous[context] = (row, key, reading)
        if reading.get("review_status") == "human-verified":
            confirmed[row["id"]] = [reading]
        if prior and prior[1] == key:
            earlier, _, evidence = prior
            if (0 < row["time"] - earlier["time"] <= .5 and row["frame_id"] != earlier["frame_id"]
                    and row.get("source_frame_sha256") != earlier.get("source_frame_sha256")):
                confirmed[earlier["id"]] = [evidence, reading]
                confirmed[row["id"]] = [evidence, reading]
    return confirmed


def resolve_names(samples, rosters, read_name, match, *, cadence="quality"):
    """The reader and matcher are lazy callbacks so registry hits avoid model work."""
    rows = []
    readings = {}
    for sample in select_frames(samples, prefer_name_region=True, cadence=cadence):
        roster = rosters.at(sample["time"])
        if roster and roster["source_sha256"] != sample["source_sha256"]:
            roster = None
        row = {**sample, "scope": roster["scope"] if roster else None}
        rows.append(row)
        if roster and row.get("visible"):
            reading = read_name(row)
            if reading:
                readings[row["id"]] = reading

    confirmed = confirm_names(rows, readings)
    registries = {key: dict(roster["bindings"]) for key, roster in rosters.rosters.items()}
    conflicts = {key: set(roster["disputed_names"]) for key, roster in rosters.rosters.items()}
    names = {row["id"]: key for row in rows if (key := attached_name(row, readings.get(row["id"])))}
    simultaneous = Counter((row["scope"], names[row["id"]], row["frame_id"])
                           for row in rows if row["id"] in names)
    for (scope, key, _), count in simultaneous.items():
        if count > 1:
            conflicts[scope].add(key)

    for row in rows:
        key = names.get(row["id"])
        if key and row.get("hud_body_binding_accepted"):
            binding = registries[row["scope"]].get(key)
            if binding and row.get("accepted_hud_champion") not in {None, binding["champion"]}:
                conflicts[row["scope"]].add(key)

    tracker = TrackIdentity()
    attempts = {}
    frames_seen = defaultdict(set)
    lookup_hits = 0
    for row in rows:
        scope = row["scope"]
        key = names.get(row["id"])
        if not scope:
            continue
        roster = rosters.rosters[scope]
        track = uid("body-votes", scope, row["track_id"], row["segment_id"], row["owner_epoch"], row.get("track_epoch", 0))
        vote = {**row, "time": row["vote_time"], "track_id": track, "name_key": key, "evidence_id": row["id"]}
        if not key or key in conflicts[scope]:
            tracker.update(vote, {"champion": None})
            continue
        binding = registries[scope].get(key)
        if binding:
            tracker.update(vote, {"champion": None})
            if row.get("hud_body_binding_accepted") and row.get("accepted_hud_champion") not in {None, binding["champion"]}:
                conflicts[scope].add(key)
            else:
                lookup_hits += 1
            continue
        if row["id"] not in confirmed:
            tracker.update(vote, {"champion": None})
            continue
        frame_signature = row.get("source_frame_sha256") or row["frame_id"]
        if frame_signature in frames_seen[track]:
            tracker.update(vote, {"champion": None})
            continue
        frames_seen[track].add(frame_signature)
        result = dict(match(row, roster))
        candidate = canonical_champion(result.get("champion"))
        hud = row.get("accepted_hud_champion") if row.get("hud_body_binding_accepted") else None
        result["champion"] = candidate if candidate in roster["champions"] and hud in {None, candidate} else None
        result["skin_id"] = None
        attempts[row["id"]] = result
        result_with_votes = tracker.update(vote, result)
        champion = result_with_votes["champion"]
        if champion:
            visual_ids = result_with_votes["identity_vote_ids"]
            witnesses = sorted({item["id"] for identifier in visual_ids for item in confirmed[identifier]})
            evidence = witnesses + visual_ids
            registries[scope][key] = {
                "id": uid("body-name", roster["id"], key, champion, evidence),
                "name": readings[row["id"]]["value"], "name_key": key, "champion": champion,
                "source": "visual", "skin_id": None, "evidence_ids": evidence,
                "established_at": max(roster["confirmed_at"], row["vote_time"],
                                      *(reading["time"] for reading in confirmed[row["id"]])),
            }

    predictions = []
    for row in rows:
        scope = row["scope"]
        key = names.get(row["id"])
        binding = registries[scope].get(key) if scope and key and key not in conflicts[scope] else None
        roster = rosters.rosters.get(scope)
        reading = readings.get(row["id"], {})
        evidence = confirmed.get(row["id"], [reading] if key else [])
        name_confirmed_at = max((item["time"] for item in evidence), default=row["time"])
        available_at = max(binding["established_at"], name_confirmed_at) if binding else None
        reason = ("bound_name" if binding else "no_confirmed_roster" if not scope else
                  "name_conflict" if key in conflicts[scope] else "unbound_name" if key else "no_confirmed_name")
        predictions.append({
            **row, "champion": binding["champion"] if binding else None, "skin_id": binding["skin_id"] if binding else None,
            "player_name": reading.get("value") if key else None, "name_evidence_ids": [r["id"] for r in evidence],
            "binding_id": binding["id"] if binding else None, "identity_source": binding["source"] if binding else None,
            "roster_id": roster["id"] if roster else None, "roster_confirmed_at": roster["confirmed_at"] if roster else None,
            "roster_status": roster["status"] if roster else None,
            "identity_available_at": available_at, "backfilled": bool(binding and available_at > row["time"]),
            "visual_attempt": attempts.get(row["id"]), "reason": reason,
        })
    return {
        "predictions": predictions,
        "rosters": list(rosters.rosters.values()),
        "bindings": [{**binding, "scope": scope} for scope, registry in registries.items()
                     for key, binding in registry.items() if key not in conflicts[scope]],
        "conflicts": [{"scope": scope, "name_keys": sorted(keys)} for scope, keys in conflicts.items() if keys],
        "name_readings": list(readings.values()),
        "counts": {"observations": len(rows), "visual_attempts": len(attempts),
                   "registry_hits": lookup_hits, "skipped_visual_calls": len(rows) - len(attempts),
                   "named": sum(bool(row["champion"]) for row in predictions),
                   "backfilled": sum(row["backfilled"] for row in predictions)},
    }
