"""Derive provisional team membership from scoreboard and killfeed evidence."""
from collections import defaultdict
from dataclasses import dataclass

from .body_roster import SCOREBOARD_SLOTS
from .core import digest, finite, read_json, uid
from .scoreboard_skim import load_skim, review_roster

SIDES = ("left", "right")
OPPOSITE = {"left": "right", "right": "left"}


@dataclass
class Scoreboard:
    members: dict
    disputed: bool
    fingerprint: str


def scoreboard_witness(portrait, time):
    return {"time": time, "id": portrait["id"], "crop": portrait["crop"],
            "crop_sha256": portrait["crop_sha256"]}


def load_scoreboard(root, source, interval, champions):
    manifest, inputs, samples, names = load_skim(root)
    start, end = manifest["interval"]
    if (manifest["source"]["sha256"] != source or not manifest["recipe"]["single_match"]
            or not start <= interval[0] < interval[1] <= end):
        raise ValueError("Killfeed requires a scoreboard skim of the same source and single-match interval")
    saved = read_json(root / "match-roster.json")
    if saved["source_sha256"] != source or saved.get("inputs") != inputs:
        raise ValueError("Regenerate the scoreboard roster from its current skim before killfeed intake")

    members = {side: {} for side in SIDES}
    reviewed_dispute = False
    if saved.get("review") is not None:
        reviewed = review_roster(root, manifest, samples, names, saved["review"])
        start, end = reviewed["interval"]
        if not start <= interval[0] < interval[1] <= end:
            raise ValueError("Killfeed interval exceeds reviewed scoreboard coverage")
        reviewed_dispute = reviewed["status"] == "disputed"
        for side, team in reviewed["members"].items():
            for champion, evidence in team.items():
                first = evidence["first"]
                members[side][champion] = scoreboard_witness(first, first["time"])
    else:
        for sample in sorted(samples, key=lambda row: row["time"]):
            if not finite(sample["time"]) or not start <= sample["time"] < end:
                raise ValueError("Scoreboard sample lies outside the skim interval")
            for slot, portrait in sample["portraits"].items():
                if slot in SCOREBOARD_SLOTS and portrait["acceptance"] == "accepted":
                    side = slot.split(":")[0]
                    members[side].setdefault(portrait["value"], scoreboard_witness(portrait, sample["time"]))
    if any(set(team) - champions for team in members.values()):
        raise ValueError("Scoreboard contains a champion outside the pinned killfeed catalog")
    disputed = (reviewed_dispute or saved["status"] == "disputed"
                or any(len(team) > 5 for team in members.values())
                or bool(members["left"].keys() & members["right"].keys()))
    return Scoreboard(members, disputed, digest({"inputs": inputs, "roster": saved}))


def scoreboard_veto(scoreboard, side, champion):
    same, other = scoreboard.members[side], scoreboard.members[OPPOSITE[side]]
    if champion in same:
        return None
    if champion in other:
        return "scoreboard_opposite_team"
    if len(same) == 5 and not scoreboard.disputed:
        return "scoreboard_complete_team"
    return None


def fill_roster(events, scoreboard, interval, provenance):
    start, end = interval
    proposals = defaultdict(list)
    observed_sides = defaultdict(set)
    conflicts = []
    for event in events:
        for name, role in event["roles"].items():
            side, champion, witness = role["side"], role["observed_champion"], role["confirmation"]
            if not side or not champion:
                continue
            if witness and not event["replay"]:
                observed_sides[champion].add(side)
            veto = scoreboard_veto(scoreboard, side, champion)
            if veto:
                role.update(champion=None, status=veto)
                ids = [scoreboard.members[OPPOSITE[side]][champion]["id"]] if veto == "scoreboard_opposite_team" else [
                    item["id"] for item in scoreboard.members[side].values()]
                conflicts.append({"event_id": event["id"], "role": name, "side": side, "champion": champion,
                                  "reason": veto, "scoreboard_evidence_ids": sorted(ids)})
            elif witness and not event["replay"]:
                proposals[side, champion].append((witness["confirmed_at"], event["id"], witness))
        event["actor"] = event["roles"]["killer"]["champion"]
        event["target"] = event["roles"]["victim"]["champion"]

    bilateral = {champion for champion, sides in observed_sides.items() if len(sides) > 1}
    competing = {
        side: {champion for team, champion in proposals
               if side == team and champion not in scoreboard.members[side] and champion not in bilateral}
        for side in SIDES
    }
    rows = []
    for (side, champion), witnesses in sorted(proposals.items()):
        reason = None
        if scoreboard.disputed:
            reason = "scoreboard_disputed"
        elif champion in bilateral:
            reason = "feed_both_teams"
        elif champion in competing[side] and len(competing[side]) + len(scoreboard.members[side]) > 5:
            reason = "competing_feed_candidates"
        if reason:
            conflicts.extend({"event_id": identifier, "side": side, "champion": champion, "reason": reason}
                             for _, identifier, _ in witnesses)
            continue
        time, identifier, witness = min(witnesses, key=lambda item: item[:2])
        scoreboard_witness = scoreboard.members[side].get(champion)
        until = min(end, scoreboard_witness["time"]) if scoreboard_witness else end
        if start <= time < until:
            rows.append({"side": side, "champion": champion, "start": time, "end": until,
                         "source": "killfeed", "status": "provisional", "observed_at": witness["observed_at"],
                         "announcement_ids": [identifier], "evidence_ids": witness["evidence_ids"]})
    if not scoreboard.disputed:
        for side, team in scoreboard.members.items():
            for champion, witness in team.items():
                if witness["time"] < end:
                    rows.append({"side": side, "champion": champion, "start": max(start, witness["time"]),
                                 "end": end, "source": "scoreboard", "status": "scoreboard",
                                 "observed_at": witness["time"], "announcement_ids": [],
                                 "evidence_ids": [witness["id"]]})
    for row in rows:
        row.update(provenance, experimental=True, id=uid("killfeed-roster", provenance, row))
    return sorted(rows, key=lambda row: (row["start"], row["side"], row["champion"])), conflicts
