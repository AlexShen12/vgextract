"""Confirm a match roster from effective scoreboard observations."""

from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
import unicodedata

from .core import uid
from .normalization import accepted

SCOREBOARD_SLOTS = frozenset(f"{side}:row{row}" for side in ("left", "right") for row in range(1, 6))


def name_key(value):
    return unicodedata.normalize("NFC", value).strip() if isinstance(value, str) else ""


def name_bindings(rows):
    """Return only unambiguous name joins, retaining the original row evidence."""
    counts = Counter(name_key(row["name"]) for row in rows if row["name"])
    return {name_key(row["name"]): row for row in rows if row["name"] and counts[name_key(row["name"])] == 1}


@dataclass
class ScoreboardSample:
    time: float
    scheduled: float
    segment: str | None
    continuous: bool
    present: bool
    rows: list
    complete: bool
    profile_hash: str | None

    @property
    def membership(self):
        return sorted((row["side"], row["champion"]) for row in self.rows)

    @property
    def reviewed(self):
        return self.complete and all(row["portrait_reviewed"] for row in self.rows)


def scoreboard_samples(observations, source_sha256, champions, calibrated_profiles):
    grouped = defaultdict(list)
    for row in observations:
        if row.get("route") == "scoreboard" and row.get("field") in {"inventory.champion", "player_name"}:
            grouped[row["time"], row["task_id"]].append(row)
    for (time, task_id), readings in sorted(grouped.items()):
        source_id = uid("source", source_sha256, time)
        portraits = [r for r in readings if r["field"] == "inventory.champion"]
        if not portraits:
            continue
        first = portraits[0]
        segment = first.get("playback_segment_id")
        scoped = all(r.get("source_id") == source_id and r.get("playback_segment_id") == segment
                     and r.get("profile_hash") == first.get("profile_hash")
                     for r in portraits)
        profile_valid = first.get("profile_hash") in calibrated_profiles
        valid = bool(scoped and profile_valid and segment and first.get("playback_state") in {"playing", "paused"})
        present = valid and all(r.get("panel_state") == "present" for r in portraits)
        rows = []
        names = defaultdict(list)
        for row in readings:
            if (row["field"] == "player_name" and accepted(row) and name_key(row.get("value"))
                    and row.get("source_id") == source_id and row.get("playback_segment_id") == segment
                    and row.get("profile_hash") == first.get("profile_hash")):
                names[row["slot_id"]].append(row)
        for portrait in portraits:
            champion = portrait.get("value")
            if not accepted(portrait) or not isinstance(champion, str) or champion not in champions:
                continue
            slot = portrait["slot_id"]
            name = names[slot][0] if len(names[slot]) == 1 else {}
            rows.append({
                "slot": slot, "side": slot.split(":")[0], "champion": champion,
                "portrait_id": portrait["id"], "portrait_crop": portrait.get("evidence_path"),
                "portrait_reviewed": portrait.get("review_status") == "human-verified",
                "name": name.get("value"), "name_id": name.get("id"), "name_crop": name.get("evidence_path"),
                "name_reviewed": name.get("review_status") == "human-verified",
                "snapshot_id": uid("inventory", task_id, "scoreboard", slot),
                "catalog_id": portrait.get("inventory_catalog_id"), "skin_catalog_id": portrait.get("skin_catalog_id"),
            })
        members = {(row["side"], row["champion"]) for row in rows}
        complete = (present and len(portraits) == len(rows) == len(members) == 10
                    and {row["slot"] for row in rows} == SCOREBOARD_SLOTS
                    and len({row["catalog_id"] for row in rows}) == 1 and rows[0]["catalog_id"] is not None)
        yield ScoreboardSample(time, first.get("sample_time", time), segment if valid else None,
                               valid and all(r.get("continuity_verified") is True for r in portraits),
                               present, rows, complete, first.get("profile_hash"))


class RosterIndex:
    """Use observed continuous intervals, with the confirmation time recorded for backfill."""

    def __init__(self, observations, source_sha256, champions, calibrated_profiles):
        self.rosters = {}
        self.times, self.scopes = [], []
        previous = None
        scope = None
        for sample in scoreboard_samples(observations, source_sha256, champions, calibrated_profiles):
            adjacent = (previous is not None and 0 < sample.time - previous.time <= 1.25
                        and abs(sample.scheduled - previous.scheduled - 1) < .1)
            connected = (adjacent and sample.segment == previous.segment and sample.continuous
                         and sample.profile_hash == previous.profile_hash)
            if not connected:
                scope = uid("body-scope", source_sha256, sample.segment, sample.time) if sample.segment else None
            self.times.append(sample.time)
            self.scopes.append(scope)
            if scope and sample.complete:
                paired = (connected and previous.complete and previous.membership == sample.membership
                          and previous.rows[0]["catalog_id"] == sample.rows[0]["catalog_id"])
                current = self.rosters.get(scope)
                if current and current["membership"] != sample.membership:
                    current["status"] = "disputed"
                    current["conflicting_snapshots"].extend(row["snapshot_id"] for row in sample.rows)
                if current is None and (paired or sample.reviewed):
                    witnesses = [sample] if sample.reviewed else [previous, sample]
                    self.rosters[scope] = current = {
                        "id": uid("body-roster", scope, [s.rows for s in witnesses]),
                        "status": "confirmed", "source_sha256": source_sha256, "segment_id": sample.segment,
                        "scope": scope, "confirmed_at": sample.time, "membership": sample.membership,
                        "champions": sorted({row["champion"] for row in sample.rows}),
                        "snapshots": [{"time": s.time, "rows": s.rows} for s in witnesses],
                        "bindings": {}, "disputed_names": [], "conflicting_snapshots": [],
                    }
                if current and current["status"] == "confirmed":
                    self._learn_scoreboard_names(current, sample, previous if paired else None)
            previous = sample

    @staticmethod
    def _learn_scoreboard_names(roster, sample, previous):
        earlier = name_bindings(previous.rows) if previous else {}
        names = name_bindings(sample.rows)
        duplicates = Counter(name_key(row["name"]) for row in sample.rows if row["name"])
        conflicts = set(roster["disputed_names"])
        conflicts.update(name for name, count in duplicates.items() if count > 1)
        if previous:
            earlier_counts = Counter(name_key(row["name"]) for row in previous.rows if row["name"])
            conflicts.update(name for name, count in earlier_counts.items() if count > 1)
        for name, row in names.items():
            prior = earlier.get(name)
            reviewed = row["portrait_reviewed"] and row["name_reviewed"]
            if reviewed:
                witnesses = [row]
            elif prior and (row["side"], row["champion"]) == (prior["side"], prior["champion"]):
                witnesses = [prior, row]
            else:
                continue
            existing = roster["bindings"].get(name)
            if existing and (existing["side"], existing["champion"]) != (row["side"], row["champion"]):
                conflicts.add(name)
            elif existing is None:
                evidence = [item[key] for item in witnesses for key in ("name_id", "portrait_id")]
                roster["bindings"][name] = {
                    "id": uid("body-name", roster["id"], name, evidence), "name": row["name"], "name_key": name,
                    "champion": row["champion"], "side": row["side"], "skin_id": None, "source": "scoreboard",
                    "established_at": sample.time, "evidence_ids": evidence,
                }
        roster["disputed_names"] = sorted(conflicts)
        for name in conflicts:
            roster["bindings"].pop(name, None)

    def at(self, time):
        index = bisect_right(self.times, time) - 1
        if index < 0:
            return None
        scope = self.scopes[index]
        if time != self.times[index] and (index + 1 == len(self.times) or self.scopes[index + 1] != scope):
            return None
        roster = self.rosters.get(scope)
        return roster if roster and roster["status"] == "confirmed" else None
