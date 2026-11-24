"""Score the saved Midbeast killfeed pilot against source-only labels."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from .core import Store, atomic_json, digest, read_json
from .intake_profile import load_profile
from .inventory_references import load_inventory_references, reference_root
from .killfeed import PROFILE, READINGS, ROSTER, TIMELINE, active_snapshot, snapshot_rows
from .killfeed_roster import Scoreboard, fill_roster, load_scoreboard

LABELS = Path(__file__).with_name("benchmarks") / "midbeast-OLLZBe5-fg4" / "killfeed-source-labels.json"


def rate(count, total):
    return count / total if total else None


def match_announcements(events, labels):
    """Pair by source timing and row order before comparing any identity."""
    events = sorted(events, key=lambda event: (event["start"], event["first_box"][1]))
    pairs = []
    used = set()
    for label in labels:
        viable = [index for index, event in enumerate(events) if index not in used
                  and label["onset"][0] - .1 < event["start"] <= label["visible_through"]
                  and event["end"] >= label["onset"][0]]
        if not viable:
            continue
        index = min(viable, key=lambda i: (abs(events[i]["start"] - label["onset"][1]), i))
        used.add(index)
        pairs.append((events[index], label))
    unmatched = [event for index, event in enumerate(events) if index not in used]
    return pairs, unmatched


def coverage_at(rows, time, truth):
    present = {(row["side"], row["champion"]) for row in rows if row["start"] <= time < row["end"]}
    baseline = {(row["side"], row["champion"]) for row in rows
                if row["source"] == "scoreboard" and row["start"] <= time < row["end"]}
    return {"time": time, "scoreboard": len(baseline & truth), "with_killfeed": len(present & truth),
            "wrong": sorted(present - truth)}


def score(store, cache, scoreboard_root):
    active = active_snapshot(store)
    if not active:
        raise ValueError("Run killfeed-intake before scoring")
    recipe = active["recipe"]
    labels = read_json(LABELS)
    profile = read_json(PROFILE)
    start, end = recipe["interval"]
    if (recipe["source_sha256"] != labels["source_sha256"]
            or recipe["profile_hash"] != digest([profile, load_profile(profile["layout_profile"])])
            or not all(start <= a < b <= end for a, b in labels["windows"])
            or any(a < d and c < b for a, b in labels["windows"] for c, d in profile["development_intervals"])):
        raise ValueError("Pilot scoring requires the frozen profile, separate source-label windows and complete coverage")
    root = reference_root(cache, profile["reference_version"])
    catalog = load_inventory_references(root, profile["reference_version"])
    champions = {asset["id"] for asset in catalog["assets"] if asset["kind"] == "champion"}
    scoreboard = load_scoreboard(scoreboard_root, recipe["source_sha256"], [start, end], champions)
    if scoreboard.disputed or scoreboard.fingerprint != recipe["scoreboard_hash"] or any(
            len(team) != 5 for team in scoreboard.members.values()):
        raise ValueError("Pilot scoring requires its original, complete scoreboard reference")
    first_complete = max(witness["time"] for team in scoreboard.members.values() for witness in team.values())
    if start > min(witness["time"] for team in scoreboard.members.values() for witness in team.values()):
        raise ValueError("Pilot scoring needs the early roster coverage interval")
    events = snapshot_rows(store, TIMELINE, active)
    roster = snapshot_rows(store, ROSTER, active)
    readings = snapshot_rows(store, READINGS, active)
    reviewed_events = [event for event in events if any(a <= event["start"] < b for a, b in labels["windows"])]
    pairs, unmatched = match_announcements(reviewed_events, labels["champion_announcements"])
    roles = {}
    for role in ("killer", "victim"):
        known = sum(event["roles"][role]["champion"] is not None for event in reviewed_events)
        correct = sum(event["roles"][role]["champion"] == label[role] for event, label in pairs)
        labeled = len(labels["champion_announcements"])
        roles[role] = {"correct": correct, "accepted": known, "labeled": labeled, "precision": rate(correct, known),
                       "coverage": rate(sum(event["roles"][role]["champion"] is not None for event, _ in pairs), labeled)}
    bracketed = [(event["onset_bracket"], label["onset"]) for event, label in pairs if event["onset_bracket"]]
    bracket_hits = sum(max(pred[0], gold[0]) + 1e-6 < min(pred[1], gold[1]) for pred, gold in bracketed)
    duplicates = [event["id"] for event in unmatched if any(
        event["start"] <= label["visible_through"] and event["end"] >= label["onset"][0]
        and all(event["roles"][name]["observed_champion"] == label[name] for name in ("killer", "victim"))
        for label in labels["champion_announcements"])]
    objective = [{"label": label, "candidate_rows": sum(
        row["classification"] == "objective" and label["onset"][0] <= row["time"] <= label["visible_through"]
        and row["roles"]["killer"]["champion"] == label["killer"] for row in readings)}
        for label in labels["non_champion_announcements"]]
    truth = {(side, champion) for side, team in scoreboard.members.items() for champion in team}
    early = [coverage_at(roster, time, truth) for time in sorted({start, *(
        row["start"] for row in roster if row["start"] < first_complete)})]
    cases = []
    participants = {label[role] for label in labels["champion_announcements"] for role in ("killer", "victim")}
    for side, champion in sorted(member for member in truth if member[1] in participants):
        hidden = deepcopy(scoreboard.members)
        del hidden[side][champion]
        recovered, _ = fill_roster(deepcopy(reviewed_events), Scoreboard(hidden, False, "withheld"), [start, end], {})
        offered = {(row["side"], row["champion"]) for row in recovered if row["source"] == "killfeed"}
        cases.append({"hidden": [side, champion], "recovered": (side, champion) in offered,
                      "wrong": sorted(offered - truth)})
    return {"schema_version": 1, "snapshot": active["revision"], "labels_sha256": digest(labels),
            "review_windows": labels["windows"], "regression_windows": labels["regression_windows"],
            "announcements": {"matched": len(pairs), "predicted": len(reviewed_events),
                              "labeled": len(labels["champion_announcements"]), "duplicates": duplicates,
                              "precision": rate(len(pairs), len(reviewed_events)),
                              "recall": rate(len(pairs), len(labels["champion_announcements"])),
                              "unmatched_predictions": [event["id"] for event in unmatched]},
            "roles": roles, "onsets": {"correct": bracket_hits, "bracketed": len(bracketed), "matched": len(pairs),
                                     "accuracy": rate(bracket_hits, len(bracketed))},
            "non_champion": objective,
            "early_roster": {"scoreboard_complete_at": first_complete, "coverage_changes": early},
            "controlled_omission": {"correct": sum(case["recovered"] for case in cases), "cases": cases},
            "passed_roster_gate": (any(case["recovered"] for case in cases) and not any(
                row["wrong"] for row in early + cases))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--scoreboard-skim", required=True, type=Path)
    parser.add_argument("--cache", type=Path, default=Path("data/league_assets"))
    args = parser.parse_args()
    store = Store(args.run_dir)
    result = score(store, args.cache, args.scoreboard_skim)
    output = store.path("killfeed") / result["snapshot"] / "benchmark.json"
    atomic_json(output, result)
    print(json.dumps(result, indent=2))
    return 0 if result["passed_roster_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
