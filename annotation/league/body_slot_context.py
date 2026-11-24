"""Load a completed single-match scoreboard skim as opt-in body roster context."""

from pathlib import Path

from .core import digest, read_json
from .media import file_hash
from .scoreboard_skim import build_match_roster, checked_evidence, load_skim, review_roster


class MatchRosterIndex:
    def __init__(self, root, source_sha256, champion_ids):
        root = Path(root).resolve()
        manifest, inputs, samples, names = load_skim(root)
        roster_path = root / "match-roster.json"
        roster = read_json(roster_path)
        if (manifest["source"]["sha256"] != source_sha256 or roster["source_sha256"] != source_sha256
                or inputs != roster["inputs"]):
            raise ValueError("Scoreboard skim evidence changed or belongs to another source")
        if roster.get("review"):
            rebuilt = review_roster(root, manifest, samples, names, roster["review"])
            if digest(roster) != digest({**rebuilt, "inputs": inputs}):
                raise ValueError("Match roster differs from the reviewed source evidence")
            method = "reviewed_cross_opening"
        else:
            # Recompute legacy skims and discard their former automatic confirmation and name bindings.
            rebuilt = build_match_roster(samples, source_sha256, manifest["interval"],
                                         single_match=manifest["recipe"]["single_match"])
            rebuilt["evidence_files"] = {}
            if rebuilt["status"] == "experimental":
                for team in rebuilt["members"].values():
                    for member in team.values():
                        for witness in (member["first"], member["best"]):
                            rebuilt["evidence_files"][witness["crop"]] = checked_evidence(
                                root, witness["crop"], witness["crop_sha256"])
            method = "unreviewed_cross_opening"
        if not set(rebuilt["champions"]) <= champion_ids:
            raise ValueError("Match roster differs from the accepted source evidence")
        self.roster = {**rebuilt, "segment_id": rebuilt["scope"], "confirmation_method": method}
        self.rosters = {rebuilt["scope"]: self.roster}
        self.revision = {"root": str(root), "roster_file": roster_path.name, "roster_sha256": file_hash(roster_path),
                         "samples_sha256": file_hash(root / "scoreboard-samples.json"),
                         "names_sha256": file_hash(root / "scoreboard-names.json"),
                         "evidence_files": rebuilt["evidence_files"],
                         "manifest_sha256": file_hash(root / "manifest.json"),
                         "profile": manifest["recipe"]["profile"]}

    def at(self, time):
        start, end = self.roster["interval"]
        return self.roster if self.roster["status"] in {"confirmed", "experimental"} and start <= time < end else None


def skim_revision_matches(revision):
    root = Path(revision["root"])
    if "evidence_files" not in revision:
        return False  # Pre-review skim results cannot qualify as reviewed evidence.
    files = {**revision["evidence_files"], revision["roster_file"]: revision["roster_sha256"],
             "scoreboard-samples.json": revision["samples_sha256"], "scoreboard-names.json": revision["names_sha256"],
             "manifest.json": revision["manifest_sha256"]}
    profile = root / "layout_profiles" / f"{revision['profile']}.json"
    return (profile.is_file() and digest(read_json(profile)) == revision["profile"]
            and all((path := (root / name).resolve()).is_relative_to(root)
                    and path.is_file() and file_hash(path) == sha for name, sha in files.items()))
