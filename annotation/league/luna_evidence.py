"""Prepare a video knowledge reference and clip-scoped LUNA timeline cues."""

import argparse
from collections import defaultdict
from pathlib import Path

from .core import read_json, read_rows


MAP_ENTITIES = """## Summoner's Rift entity guide

- Champions have a roster portrait. Compare their bodies, effects, and team to the scoreboard.
- Rift Scuttler is a shelled neutral monster that patrols the river. Its Voidborn form has a purple shell. Gromp is toad-shaped.
- Raptors are a flock, wolves are a pack, and Krugs are rock creatures. Blue Sentinel and Red Brambleback guard their jungle camps.
- Dragons are winged monsters in the lower river pit. Baron Nashor, Rift Herald, and Voidgrubs use the upper river pit. Check the model and objective text before naming the monster or awarding credit.
- Wards are small placed vision objects. Lane minions arrive in waves. Turrets, inhibitors, and the Nexus are structures.
- Blast Cone is a round reddish plant that can launch nearby units. Honeyfruit drops pickups that can restore health. Scryer's Bloom sends out a reveal effect.
- Use the visible model, location, movement, or readable target panel together. A health bar or combat number alone cannot identify the unit.
- [Riot's Voidborn monster guide](https://www.leagueoflegends.com/en-us/news/game-updates/patch-14-1-notes/)
"""


def scoreboard_rosters(rows):
    """Group observed scoreboard sides by roster and omit ambiguous members."""
    memberships = defaultdict(lambda: defaultdict(set))
    for row in rows:
        if (row.get("review_status") != "rejected" and row.get("champion")
                and row.get("side") in {"left", "right"}):
            memberships[row.get("roster_id") or "scoreboard"][row["champion"]].add(row["side"])
    return {
        roster: {side: sorted(champion for champion, sides in members.items() if sides == {side})
                 for side in ("left", "right")}
        for roster, members in sorted(memberships.items())
    }


def scoreboard_team_relationships(rosters, selected_hud_champions):
    """Resolve ally and opponent lists for one unambiguous clip viewpoint."""
    owners = {champion for champion in selected_hud_champions if champion}
    if len(owners) != 1 or len(rosters) != 1:
        return None
    (owner,) = owners
    ((roster_id, teams),) = rosters.items()
    for own_side, opposing_side in (("left", "right"), ("right", "left")):
        if owner in teams[own_side]:
            return {
                "roster_id": roster_id,
                "viewpoint_champion": owner,
                "allies": [champion for champion in teams[own_side] if champion != owner],
                "opponents": teams[opposing_side],
            }
    return None


def scoreboard_relationship_by_champion(teams):
    """Index resolved scoreboard relationships by champion ID for a clip."""
    if teams is None:
        return None
    return {
        teams["viewpoint_champion"]: "self",
        **dict.fromkeys(teams["allies"], "ally"),
        **dict.fromkeys(teams["opponents"], "opponent"),
    }


def video_knowledge(manifest, roster_rows, catalog, item_catalog=None, inventory_rows=()):
    """Render a reference using this video's roster and a pinned static catalog."""
    source = manifest["source"]
    rosters = scoreboard_rosters(roster_rows)
    champions = catalog["champions"]
    lines = [
        "# League of Legends video knowledge reference", "",
        f"Source file `{source['path']}`", f"Source SHA-256 `{source['sha256']}`",
        f"Riot Data Dragon ability build `{catalog['reference_version']}` ({catalog['locale']})",
        "Treat the ability build as a reference version unless the recording patch is verified.",
        "The reference describes identities and possible effects. Use match evidence to verify casts, targets, and outcomes.",
        "", "## Observed scoreboard rosters", "",
    ]
    roster_champions = set()
    for roster, teams in rosters.items():
        for side, members in teams.items():
            roster_champions.update(members)
            names = ", ".join(champions.get(member, {}).get("name", member) for member in members)
            lines.append(f"- `{roster}` · scoreboard {side} · {names or 'unresolved'}")
    if not rosters:
        lines.append("- No scoreboard roster has been recovered for this video.")
    lines.extend(["", MAP_ENTITIES.rstrip(), "", "## Champion abilities"])
    for champion in sorted(roster_champions):
        entry = champions.get(champion)
        if entry is None:
            lines.extend(["", f"### {champion}", "- Ability reference unavailable."])
            continue
        lines.extend(["", f"### {entry['name']} ({champion})"])
        for ability in entry["abilities"]:
            description = " ".join(ability["description"].split())
            lines.append(f"- {ability['slot']} · {ability['name']} · {description}")
        if url := entry.get("provenance", {}).get("url"):
            lines.append(f"- [Riot Data Dragon source]({url})")
    if item_catalog:
        observed = {str(slot["item_id"]) for row in inventory_rows if row.get("route") == "scoreboard"
                    for slot in row["slots"] if slot["status"] == "recognized"}
        lines.extend(["", "## Item mechanics", "",
                      f"Inventory reference build `{item_catalog['reference_version']}`",
                      "Use the clip's time-scoped scoreboard reads to check which champion was last seen with an item."])
        for item in sorted(item_catalog["items"], key=lambda entry: entry["name"]):
            if str(item["id"]) in observed and item.get("description"):
                description = " ".join(item["description"].split())
                lines.append(f"- {item['name']} · {description}")
    return "\n".join(lines) + "\n"


def cast_bar_onsets(readings):
    """Bound a bar's onset when adjacent samples show absent then readable."""
    previous = {}
    onsets = {}
    for row in sorted(readings, key=lambda item: item["time"]):
        owner = row["owner_id"]
        before = previous.get(owner)
        if (before and before["visibility"] == "absent" and row["visibility"] == "readable"
                and all(before.get(field) == row.get(field) for field in ("owner_epoch", "segment_id", "catalog_id"))
                and 0 < row["time"] - before["time"] <= 2 / 16
                and before.get("review_status") != "rejected" and row.get("review_status") != "rejected"):
            onsets[row["id"]] = [before["time"], row["time"]]
        previous[owner] = row
    return onsets


def fresh_timeline_events(events, start, end, cast_onsets):
    """Return cues whose evidence bounds their onset within [start, end)."""
    fresh = []
    for event in events:
        if event.get("review_status") == "rejected":
            continue
        kind = event["kind"]
        bracket = event.get("onset_bracket")
        if bracket is None and kind == "hud_slot_activation" and event.get("onset_frame_ids"):
            bracket = [event["start"], event["end"]]
        if bracket is None and kind == "observed_channel":
            readings = event.get("evidence_ids", [])
            bracket = cast_onsets.get(readings[0]) if readings else None
        if kind not in {"kill_announcement", "hud_slot_activation", "observed_channel"} or bracket is None:
            continue
        lower, upper = bracket
        if start <= lower <= upper < end:
            fields = ("id", "kind", "actor", "target", "channel_kind", "ability_id", "observed_text",
                      "slot_id", "owner_id", "owner_epoch", "segment_id", "replay")
            fresh.append({**{field: event[field] for field in fields if field in event},
                          "onset_bracket": [lower, upper]})
    return sorted(fresh, key=lambda event: event["onset_bracket"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pass1", type=Path, required=True, help="Directory with manifest and roster timeline")
    parser.add_argument("--ability-catalog", type=Path, required=True, help="Pinned Data Dragon ability catalog")
    parser.add_argument("--inventory-catalog", type=Path, help="Pinned item catalog for scoreboard-recognized items")
    parser.add_argument("--output", type=Path, required=True, help="Output Markdown reference")
    args = parser.parse_args()
    items = read_json(args.inventory_catalog) if args.inventory_catalog else None
    inventory = read_rows(args.pass1 / "inventory_snapshots.jsonl") if items else ()
    reference = video_knowledge(read_json(args.pass1 / "manifest.json"),
                                read_rows(args.pass1 / "roster_timeline.jsonl"),
                                read_json(args.ability_catalog), items, inventory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(reference, encoding="utf-8")


if __name__ == "__main__":
    main()
