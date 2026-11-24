"""Collect gate-passing champion evidence across one source match."""

from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import argparse
import fcntl
import json
import time

from .body_roster import SCOREBOARD_SLOTS, name_key
from .core import Store, atomic_json, digest, finite, implementation_hash, read_json, uid
from .media import file_hash, probe


def build_match_roster(samples, source_sha256, interval, *, single_match=False):
    """Collect the first and strongest accepted witness for each team member."""
    members = {side: defaultdict(list) for side in ("left", "right")}
    seen = {side: set() for side in members}
    filled_at = None
    for sample in sorted(samples, key=lambda item: item["time"]):
        for slot, row in sample["portraits"].items():
            if slot not in SCOREBOARD_SLOTS or row["acceptance"] != "accepted" or not row.get("value"):
                continue
            side = slot.split(":")[0]
            champion = row["value"]
            members[side][champion].append({**row, "time": sample["time"], "slot": slot})
            seen[side].add(champion)
        if filled_at is None and all(len(names) == 5 for names in seen.values()):
            filled_at = sample["time"]
    membership = sorted((side, champion) for side, names in members.items() for champion in names)
    disputes = {side: sorted(names) for side, names in seen.items() if len(names) > 5}
    complete = all(len(names) == 5 for names in seen.values())
    witnesses = {}
    for side, by_champion in members.items():
        witnesses[side] = {name: {"accepted_reads": len(rows), "first": rows[0],
                                  "best": max(rows, key=lambda row: (row["score"], row["margin"], -row["time"]))}
                           for name, rows in sorted(by_champion.items())}
    experimental = single_match and complete and not disputes
    signature = digest([source_sha256, interval, witnesses, single_match])
    return {"id": "match-roster_" + signature[:20], "scope": "scoreboard-match_" + signature[:20],
            "source_sha256": source_sha256, "interval": list(interval), "single_match": single_match,
            "experimental": True, "bindings": {}, "disputed_names": [],
            "accepted_members": sum(map(len, seen.values())),
            "missing_members": {side: max(0, 5 - len(names)) for side, names in seen.items()},
            "mode": "team_membership", "members": witnesses, "membership_conflicts": disputes,
            "status": "disputed" if disputes else "experimental" if experimental else "review" if complete else "incomplete",
            "first_complete_at": filled_at, "confirmed_at": filled_at if experimental else None,
            "membership": membership if experimental else [],
            "champions": sorted({name for _, name in membership}) if experimental else [],
            "candidate_champions": sorted({name for _, name in membership})}


def checked_evidence(root, path, expected):
    target = (root / path).resolve()
    if not target.is_relative_to(root) or not target.is_file() or file_hash(target) != expected:
        raise ValueError(f"Scoreboard witness changed: {path}")
    return expected


def review_roster(root, manifest, samples, names, review):
    """Rebuild a segment roster using explicit human decisions about source witnesses."""
    source = manifest["source"]["sha256"]
    start, end = review["interval"]
    lower, upper = manifest["interval"]
    timestamp = datetime.fromisoformat(review["reviewed_at"])
    if (not manifest["recipe"]["single_match"] or review["source_sha256"] != source
            or review.get("samples_sha256") != digest(samples)
            or review.get("continuity_verified") is not True or not name_key(review.get("reviewer"))
            or timestamp.utcoffset() is None or not all(finite(t) for t in (start, end))
            or not lower <= start < end <= upper):
        raise ValueError("Roster review requires a reviewer, timestamp and verified source interval")
    profile_id = manifest["recipe"]["profile"]
    profile = read_json(root / "layout_profiles" / f"{profile_id}.json")
    if digest(profile) != profile_id or profile.get("calibration_source_sha256") != source:
        raise ValueError("Scoreboard layout changed or belongs to another source")
    inventory = profile["inventory"]
    catalogs = {key: value for key, value in inventory.items() if key.endswith("version")}
    by_id = {}
    frames = [sample for sample in samples if start <= sample["time"] < end]
    for sample in frames:
        for slot, row in sample["portraits"].items():
            if slot in SCOREBOARD_SLOTS:
                by_id[row["id"]] = (sample, slot, row)

    decisions = {row["portrait_id"]: row for row in review["portraits"]}
    if not decisions or len(decisions) != len(review["portraits"]) or not decisions.keys() <= by_id.keys():
        raise ValueError("Portrait decisions must cite distinct evidence within the reviewed interval")
    members = {"left": {}, "right": {}}
    name_rows = []
    evidence_files = {}
    for identifier, decision in decisions.items():
        sample, slot, portrait = by_id[identifier]
        for path, sha in ((sample["frame"], sample["frame_sha256"]), (portrait["crop"], portrait["crop_sha256"])):
            evidence_files[path] = checked_evidence(root, path, sha)
        champion = decision["champion"]  # null explicitly rejects an incorrect portrait read
        if champion is None:
            continue
        side = slot.split(":")[0]
        if not isinstance(champion, str) or not champion or champion in members[side]:
            raise ValueError("Choose one reviewed portrait witness per team member")
        witness = {**portrait, "value": champion, "time": sample["time"], "opening": sample["opening"],
                   "slot": slot, "frame": sample["frame"], "frame_sha256": sample["frame_sha256"]}
        members[side][champion] = {"first": witness, "best": witness, "accepted_reads": 1}
        if "name_id" in decision or "name" in decision:
            joined = [row for row in names if row["id"] == decision.get("name_id")
                      and (row["portrait_id"], row["slot"], row["time"], row["opening"])
                      == (identifier, slot, sample["time"], sample["opening"])]
            key = name_key(decision.get("name"))
            if len(joined) != 1 or not key:
                raise ValueError("Reviewed name must cite a name crop from the same scoreboard row and frame")
            name_path = joined[0]["crop"]
            # Legacy skims do not hash name crops. The reviewer supplies the hash of the crop they saw.
            evidence_files[name_path] = checked_evidence(root, name_path, decision["name_crop_sha256"])
            name_rows.append((key, decision["name"], side, champion, [joined[0]["id"], identifier]))

    if any(len(team) > 5 for team in members.values()):
        raise ValueError("Choose at most five distinct reviewed members per side")
    complete = all(len(team) == 5 for team in members.values())
    membership = sorted((side, champion) for side, team in members.items() for champion in team)
    conflicts = []
    for sample in frames:
        effective = []
        for slot, portrait in sample["portraits"].items():
            decision = decisions.get(portrait["id"])
            champion = decision["champion"] if decision else (
                portrait["value"] if portrait["acceptance"] == "accepted" else None)
            if champion and slot in SCOREBOARD_SLOTS:
                effective.append((slot.split(":")[0], champion))
        if complete and len(effective) == len(set(effective)) == 10 and sorted(effective) != membership:
            conflicts.append(sample["time"])

    witnesses = [evidence["first"] for team in members.values() for evidence in team.values()]
    confirmed_at = max((row["time"] for row in witnesses), default=None) if complete else None
    confirmed = complete and not conflicts
    identifier = uid("match-roster", review, catalogs, evidence_files)
    duplicates = Counter(row[0] for row in name_rows)
    bindings = {}
    for key, name, side, champion, evidence in name_rows:
        if duplicates[key] == 1 and confirmed:
            bindings[key] = {"id": uid("match-name", identifier, key, evidence), "name": name,
                             "name_key": key, "side": side, "champion": champion, "source": "scoreboard",
                             "skin_id": None, "established_at": confirmed_at, "evidence_ids": evidence}
    return {"id": identifier, "scope": uid("scoreboard-segment", source, [start, end]), "source_sha256": source,
            "interval": [start, end], "single_match": True, "experimental": True, "mode": "reviewed_cross_opening",
            "status": "disputed" if conflicts else "confirmed" if confirmed else "incomplete", "membership_conflicts": conflicts,
            "first_complete_at": confirmed_at, "confirmed_at": confirmed_at if confirmed else None,
            "membership": membership if confirmed else [], "members": members,
            "champions": sorted({champion for _, champion in membership}) if confirmed else [],
            "accepted_members": len(membership), "review": review, "catalog_versions": catalogs, "evidence_files": evidence_files,
            "bindings": bindings, "disputed_names": sorted(key for key, count in duplicates.items() if count > 1)}


def load_skim(root):
    manifest = read_json(root / "manifest.json")
    inputs = read_json(root / "skim-evidence.json")
    samples = read_json(root / "scoreboard-samples.json")
    names = read_json(root / "scoreboard-names.json")
    if (inputs["samples_sha256"] != digest(samples) or inputs["names_sha256"] != digest(names)
            or inputs["recipe"] != manifest["recipe"]):
        raise ValueError("Membership derivation requires a completed, unchanged scoreboard skim")
    return manifest, inputs, samples, names


def derive_membership(output, review=None):
    store = Store(Path(output).resolve())
    manifest, inputs, samples, names = load_skim(store.root)
    roster = (review_roster(store.root, manifest, samples, names, review) if review is not None else
              build_match_roster(samples, manifest["source"]["sha256"], manifest["interval"],
                                 single_match=manifest["recipe"]["single_match"]))
    roster["inputs"] = inputs
    timeline = []
    if roster["status"] in {"confirmed", "experimental"}:
        start, end = roster["interval"]
        confirmed = roster["confirmed_at"]
        for side, champions in roster["members"].items():
            for champion, evidence in champions.items():
                base = {"side": side, "champion": champion, "observed_at": evidence["first"]["time"],
                        "confirmed_at": confirmed, "evidence_id": evidence["first"]["id"], "roster_id": roster["id"]}
                base["experimental"] = roster["status"] == "experimental"
                for first, last, backfilled in ((start, confirmed, True), (confirmed, end, False)):
                    if first < last:
                        timeline.append({**base, "start": first, "end": last, "backfilled": backfilled})
    store.write("match_roster_timeline", timeline)
    atomic_json(store.path("match-roster.json"), roster)
    return {"status": roster["status"], "first_complete_at": roster["first_complete_at"],
            "champions": roster["champions"], "accepted_members": roster["accepted_members"],
            "candidate_champions": roster.get("candidate_champions", []),
            "membership_conflicts": roster["membership_conflicts"], "name_bindings": len(roster["bindings"]),
            "timeline_entries": len(timeline)}


def save_roster_review(output, review, *, expected_revision=None):
    root = Path(output).resolve()
    path = root / "match-roster.json"
    with (root / ".roster-review.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if expected_revision is not None and (not path.is_file() or file_hash(path) != expected_revision):
            raise ValueError("Roster changed during review. Reload the evidence before saving.")
        if review is None and path.is_file():
            review = read_json(path).get("review")
        result = derive_membership(root, review)
        return {**result, "revision": file_hash(path)}


def inspect_names(store, samples, name_fields, source):
    """Transcribe names on the clearest accepted row witnesses for review."""
    from PIL import Image
    from .cast_bars import text_preprocess
    from .readers import tesseract

    candidates = defaultdict(list)
    for sample in samples:
        for slot, row in sample["portraits"].items():
            if row["acceptance"] == "accepted" and slot in name_fields:
                side = slot.split(":")[0]
                candidates[side, row["value"]].append((sample, slot, row))
    readings = []
    for (_, champion), group in sorted(candidates.items()):
        # Use up to two separate openings for a name binding.
        ordered = sorted(group, key=lambda pair: (-pair[2]["score"], -pair[2]["margin"], pair[0]["time"]))
        used = set()
        for sample, slot, witness in ordered:
            if sample["opening"] in used:
                continue
            used.add(sample["opening"])
            with Image.open(store.path(sample["frame"])) as image:
                crop = image.crop(name_fields[slot]["box"])
            key = uid("scoreboard-name", source, sample["time"], slot)
            native = store.path(f"names/{key}.png")
            native.parent.mkdir(parents=True, exist_ok=True)
            crop.save(native)
            prepared = native.with_name(native.stem + "-ocr.png")
            text_preprocess(crop).save(prepared)
            raw = tesseract(prepared)
            readings.append({"id": key, "slot": slot, "champion": champion, "time": sample["time"],
                             "opening": sample["opening"], "value": name_key(raw["raw_text"]) or None,
                             "score": raw["score"], "crop": str(native.relative_to(store.root)),
                             "portrait_id": witness["id"]})
            if len(used) == 2:
                break
    return readings


def skim(video, output, profile_name, cache, *, single_match=False, names=True):
    from .hybrid_intake import _panel_state
    from .intake import image_hash
    from .intake_profile import layout_matches, load_profile
    from .inventory_reader import CARDINAL_SHIFTS, InventoryReader
    from .inventory_references import load_inventory_references, reference_root
    from .scene_reference import sample_source

    started = time.monotonic()
    video, output = Path(video).resolve(), Path(output).resolve()
    profile = load_profile(profile_name)
    source = {**probe(video), "path": str(video), "sha256": file_hash(video)}
    if ([source["width"], source["height"]] != profile["size"]
            or not profile.get("calibrated") or profile.get("calibration_source_sha256") != source["sha256"]):
        raise ValueError("Scoreboard skim requires the source-calibrated layout")
    store = Store(output)
    catalog = reference_root(cache, profile["inventory"]["reference_version"])
    reader = InventoryReader(store, catalog, load_inventory_references(catalog, catalog.name), profile["inventory"])
    fields = [f for f in profile["fields"] if f["route"] == "scoreboard" and f["field"] == "inventory.champion"]
    if {f["slot_id"] for f in fields} != SCOREBOARD_SLOTS:
        raise ValueError("Scoreboard layout needs ten calibrated portrait slots")
    interval = [0, source["duration"]]
    recipe = {"source_sha256": source["sha256"], "profile": digest(profile), "reader": reader.signature,
              "single_match": single_match, "names": names, "sampling_fps": 1,
              "implementation": implementation_hash(["scoreboard_skim.py", "inventory_reader.py", "intake_profile.py",
                                                     "scene_reference.py", "readers.py", "cast_bars.py"])}
    manifest = {"schema_version": 2, "source": source, "interval": interval, "experimental_only": True, "recipe": recipe}
    if store.path("manifest.json").exists() and read_json(store.path("manifest.json")) != manifest:
        raise ValueError("Use a fresh output directory for different skim inputs")
    atomic_json(store.path("manifest.json"), manifest)
    atomic_json(store.path(f"layout_profiles/{digest(profile)}.json"), profile)
    samples = []
    frames = 0
    opening = 0
    previous_open = False
    last_time = None
    for pts, image in sample_source(video, *interval, 1):
        frames += 1
        valid = layout_matches(image, profile)[0]
        present = _panel_state(image, profile, {"route": "scoreboard"}, valid) == "present"
        if not present:
            previous_open = False
        else:
            if not previous_open or (last_time is not None and pts - last_time > 1.25):
                opening += 1
            previous_open, last_time = True, pts
            frame_path = f"scoreboards/{pts:.6f}.png"
            path = store.path(frame_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path)
            rows = {}
            for field in fields:
                slot = field["slot_id"]
                x, y, right, bottom = field["box"]
                crop = image.crop(field["box"])
                direct, _ = reader.read(crop, field, image_hash(crop))

                def shifted():
                    for dx, dy in CARDINAL_SHIFTS:
                        box = x + dx, y + dy, right + dx, bottom + dy
                        if 0 <= box[0] < box[2] <= image.width and 0 <= box[1] < box[3] <= image.height:
                            view = image.crop(box)
                            yield (dx, dy), view, image_hash(view)

                result = reader.retry_alignment(direct, field, shifted())
                details = result.get("reader", {})
                dx, dy = details.get("pixel_shift", (0, 0))
                evidence = f"portraits/{pts:.6f}-{slot.replace(':', '-')}.png"
                crop_path = store.path(evidence)
                crop_path.parent.mkdir(parents=True, exist_ok=True)
                image.crop((x + dx, y + dy, right + dx, bottom + dy)).save(crop_path)
                rows[slot] = {"id": uid("slot-read", source["sha256"], pts, slot, reader.signature),
                              "value": result["value"], "acceptance": result["acceptance"], "reason": result["reason"],
                              "score": details.get("score"), "margin": details.get("margin"),
                              "crop": evidence, "crop_sha256": file_hash(crop_path),
                              "alignment_views": details.get("alignment_views", [])}
            samples.append({"time": pts, "opening": opening, "frame": frame_path, "frame_sha256": file_hash(path), "portraits": rows})
            atomic_json(store.path("scoreboard-samples.json"), samples)
        if frames % 120 == 0:
            print(f"{frames} source samples, {len(samples)} visible scoreboards, {pts:.1f}s", flush=True)
    name_fields = {field["slot_id"]: field for field in profile["fields"]
                   if field["route"] == "scoreboard" and field["field"] == "player_name"}
    name_rows = inspect_names(store, samples, name_fields, source["sha256"]) if names else []
    atomic_json(store.path("scoreboard-samples.json"), samples)
    atomic_json(store.path("scoreboard-names.json"), name_rows)
    atomic_json(store.path("skim-evidence.json"), {
        "samples_sha256": digest(samples), "names_sha256": digest(name_rows), "recipe": recipe,
    })
    membership = save_roster_review(output, None)
    report = {"source": source, "source_samples": frames, "visible_scoreboards": len(samples),
              "scoreboard_openings": opening, "match_membership": membership,
              "complete_single_frames": sum(all(row["acceptance"] == "accepted" for row in s["portraits"].values())
                                            for s in samples),
              "name_reads": len(name_rows), "seconds": time.monotonic() - started, "experimental": True}
    atomic_json(store.path("report.json"), report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", default="midbeast-1080p-inventory-v3")
    parser.add_argument("--cache", type=Path, default=Path("data/league_assets"))
    parser.add_argument("--single-match", action="store_true", help="Collect cross-opening candidates from a source containing one match")
    parser.add_argument("--skip-names", action="store_true")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--derive-membership", action="store_true", help="Build a match pool from an existing completed skim")
    actions.add_argument("--serve-review", action="store_true", help="Open the optional local scoreboard roster review")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--review", type=Path, help="Apply source-reviewed portrait decisions with --derive-membership")
    args = parser.parse_args()
    if args.review and not args.derive_membership:
        parser.error("--review requires --derive-membership")
    if args.serve_review:
        import uvicorn
        from .body_roster_review import create_app

        uvicorn.run(create_app(args.output, args.cache), host="127.0.0.1", port=args.port)
        return
    if args.derive_membership:
        result = save_roster_review(args.output, read_json(args.review) if args.review else None)
    elif args.video:
        result = skim(args.video, args.output, args.profile, args.cache, single_match=args.single_match, names=not args.skip_names)
    else:
        parser.error("--video is required for a new scoreboard skim")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
