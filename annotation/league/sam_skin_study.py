"""Run the League SAM prompt and contextual mask-verification study."""

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
from urllib.parse import urlparse

from .body_frame_matching import SkinMatcher, skin_lock_scope
from .body_operational import skin_pool_report
from .body_siglip import DEFAULT_SIGLIP_LOCK
from .body_slot_context import skim_revision_matches
from .core import atomic_json, digest, read_json
from .media import file_hash
from .sam_appearances import FPS, stage_lock
from .sam_skin_score import (MIDBEAST, SECOND, SINGED, XAYAH, by_time, draft_reviews, freeze_reviews,
                             load_reviews, replay_votes, score)
from .sam_skin_tracking import ARMS, recover_observations, track
from .sam_skin_verifier import frozen_cases, infer, prepare_shortlists

FROZEN_APPEARANCES = ("appearance_db6654019460a6101930", "appearance_43efe796f7017448bbe4")
MIDBEAST_APPEARANCES = (*FROZEN_APPEARANCES, *XAYAH, *SINGED)


def staged_file(source, target, sha):
    if target.is_file() and file_hash(target) == sha:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".staging-", delete=False) as work:
        path = Path(work.name)
    try:
        shutil.copyfile(source, path)
        if file_hash(path) != sha:
            raise ValueError(f"Study input checksum differs: {source}")
        path.replace(target)
    finally:
        path.unlink(missing_ok=True)


def source_snapshot(root):
    repo = Path(__file__).resolve().parents[2]
    records = {}
    with tempfile.TemporaryDirectory(prefix=".source-", dir=root) as work:
        staged = Path(work) / "source"
        for package in ("annotation", "ingestion"):
            if root.is_relative_to(repo / package):
                raise ValueError("Place the study outside the source packages")
            for source in sorted((repo / package).rglob("*")):
                if source.is_file() and source.suffix in {".py", ".json", ".html", ".sl", ".sh", ".txt"}:
                    relative = source.relative_to(repo)
                    sha = file_hash(source)
                    staged_file(source, staged / relative, sha)
                    records[relative] = sha
        sums = "".join(f"{sha}  {path.as_posix()}\n" for path, sha in records.items())
        (staged / "SOURCE_SHA256SUMS").write_text(sums)
        target = root / "source"
        if not target.exists():
            staged.rename(target)
        elif ((target / "SOURCE_SHA256SUMS").read_text() != sums
              or any(file_hash(target / path) != sha for path, sha in records.items())):
            raise ValueError("The existing source snapshot differs. Prepare in a new study directory.")
    return file_hash(target / "SOURCE_SHA256SUMS")


def gallery_entries(gallery, pools, patch, cache, siglip_lock):
    champions = set().union(*pools.values())
    entries = [row for row in read_json(gallery) if row["champion"] in champions
               and row.get("reference_kind") == "rendered" and row.get("skin_id") is not None]
    for row in entries:
        assets = row.get("source_assets")
        if isinstance(assets, dict):
            assets = [assets[key] for key in ("metadata", "mesh") if key in assets]
        urls = [urlparse(asset.get("url", "")) for asset in assets or []]
        if not urls or any(url.netloc != "raw.communitydragon.org" or not url.path.startswith(f"/{patch[1]}/")
                           for url in urls):
            raise ValueError(f"Render {row['id']} does not establish CommunityDragon patch {patch[1]}")
    coverage = skin_pool_report(cache, {"gallery": entries, "reference_version": patch[0],
                                       "communitydragon_version": patch[1]},
                                SimpleNamespace(rosters={"match": {"champions": sorted(champions)}}))
    if coverage["status"] != "pinned":
        raise ValueError("Stage the match's pinned skin catalog before preparing the study")
    entries = [row for row in entries
               if str(row["skin_id"]) in coverage["champions"][row["champion"]]["eligible_skin_ids"]]
    matcher = SkinMatcher(entries, cache, siglip_lock, digest(read_json(siglip_lock)), patch, encoder_name="siglip")
    if missing := sorted(champions - matcher.entries.keys()):
        raise ValueError(f"Patch-matched gallery is missing scoreboard champions: {', '.join(missing)}")
    return entries, coverage, matcher


def stage_gallery(gallery, entries, directory):
    staged = []
    for row in entries:
        relocated = dict(row)
        for key, hash_key in (("path", "sha256"), ("mask_path", "mask_sha256")):
            sha = row[hash_key]
            name = f"gallery/{sha}.png"
            staged_file(gallery.parent / row[key], directory / name, sha)
            relocated[key] = name
        staged.append(relocated)
    path = directory / "gallery.json"
    atomic_json(path, staged)
    return file_hash(path)


def find_appearances(prepared, identifiers):
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError("Select distinct prepared appearances for the match")
    observations = recover_observations(prepared)
    rows = {row["id"]: row for row in prepared["appearances"] if row["status"] == "prepared"}
    frames = prepared["frames"]
    offscreen = set(prepared["recipe"]["player_offscreen_review"]["frames"])
    chosen = []
    for identifier in identifiers:
        if identifier not in rows or identifier not in observations:
            raise ValueError(f"Cannot recover the level-badged track for {identifier}")
        row = rows[identifier]
        first, last = row["first"], row["last"]
        indexed = {str(obs["frame_index"]): obs for obs in observations[identifier]}
        seed = row["seed"]["frame_index"]
        if row["seed"] != indexed.get(str(seed)) or row["seed"]["level"] not in range(1, 19):
            raise ValueError(f"Recovered seed does not match the saved track for {identifier}")
        seed_frames = [index for index in map(int, indexed) if row["team"] == "enemy"
                       or frames[index]["player_bar"] is not None or frames[index]["id"] in offscreen]
        if seed not in seed_frames:
            raise ValueError(f"Recovered seed has unresolved player binding for {identifier}")
        review_frames = sorted({round(first + (last - first) * fraction / 4) for fraction in range(5)})
        chosen.append({**row, "observations": indexed, "seed_frames": seed_frames, "review_frames": review_frames})
    return by_time(chosen)


def prepare_match(root, run, name, identifiers, prepared, gallery, scoreboard, cache, siglip_lock):
    recipe, pools = prepared["recipe"], prepared["team_pools"]
    if (recipe["fps"] != FPS or recipe["siglip_lock_hash"] != digest(read_json(siglip_lock))
            or len(recipe["patch"]) != 2):
        raise ValueError(f"{name} needs a 16 FPS appearance run using the study SigLIP lock")
    revision = recipe["scoreboard"]
    if scoreboard:
        revision = {**revision, "root": str(scoreboard)}
    if not skim_revision_matches(revision):
        raise ValueError(f"The scoreboard evidence changed or is missing for {name}")
    rows = find_appearances(prepared, identifiers)
    directory = root / "inputs" / name
    entries, coverage, matcher = gallery_entries(gallery, pools, recipe["patch"], cache, siglip_lock)
    frames = [None] * (1 + max(row["last"] for row in rows))
    for row in rows:
        for index in range(row["first"], row["last"] + 1):
            frame = prepared["frames"][index]
            if frames[index] is not None:
                continue
            frames[index] = frame
            for key, hash_key in (("path", "sha256"), ("sam_path", "sam_sha256")):
                path = Path(frame[key])
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError("Prepared frames need relative paths within the run directory")
                staged_file(run / path, directory / path, frame[hash_key])
    return {"id": name, "source_sha256": recipe["source_sha256"], "patch": recipe["patch"],
            "appearance_manifest_sha256": file_hash(run / "appearances.json"),
            "gallery_input_sha256": file_hash(gallery),
            "gallery_sha256": stage_gallery(gallery, entries, directory), "gallery_coverage": coverage,
            "scope": skin_lock_scope(matcher, pools, recipe["source_sha256"], recipe["roster_id"]),
            "sam": recipe["sam"], "profile": recipe["profile"], "team_pools": pools,
            "appearances": rows, "frames": frames}


def historical_replay(run, scope):
    report = run / "appearance-matches-siglip.json"
    ledger = run / "skin-locks-siglip.json"
    matches, recorded = read_json(report), read_json(ledger)
    report_sha = file_hash(report)
    if (not matches["complete"] or matches["scope"] != scope or recorded["scope"] != scope
            or recorded["report_sha256"] != report_sha):
        raise ValueError("The saved Midbeast report and lock ledger have different provenance")
    replay = replay_votes(matches["appearances"])
    if any(replay[key] != recorded[key] for key in replay):
        raise ValueError("Saved Midbeast votes did not reproduce the original lock events")
    enemy = replay["assignments"]["enemy"]
    for champion, skin, expected in (("Xayah", "498038", XAYAH), ("Singed", "27006", SINGED)):
        if enemy.get(champion) != {"skin_id": skin, "supporting_appearances": list(expected)}:
            raise ValueError(f"Original {champion} regression support changed")
    return {**replay, "current_report_sha256": report_sha, "ledger_sha256": file_hash(ledger)}


def pinned_configs(args):
    configs = [(MIDBEAST, args.midbeast_run, MIDBEAST_APPEARANCES,
                args.midbeast_run / "gallery.json", args.midbeast_scoreboard)]
    if args.second_run:
        if not args.second_scoreboard or not args.second_gallery or not args.second_appearance:
            raise ValueError("The second run needs its scoreboard, gallery, and explicit appearance IDs")
        configs.append((SECOND, args.second_run, args.second_appearance, args.second_gallery, args.second_scoreboard))
    elif any((args.second_gallery, args.second_appearance, args.second_scoreboard)):
        raise ValueError("Second-match inputs need --second-run")
    return configs


def prepare(args):
    root = args.root
    configs = pinned_configs(args)
    if any(root == run for _, run, *_ in configs):
        raise ValueError("Use a separate study directory for the experiment")
    manifests = [read_json(run / "appearances.json") for _, run, *_ in configs]
    sources = {manifest["recipe"]["source_sha256"] for manifest in manifests}
    if len(sources) != len(manifests):
        raise ValueError("Use a distinct underlying match and source recording for the second cohort")
    if any(manifest["recipe"]["sam"] != manifests[0]["recipe"]["sam"] for manifest in manifests):
        raise ValueError("The match cohorts need the same SAM 3.1 checkpoint")
    request = [{"id": name, "appearance_manifest": file_hash(run / "appearances.json"),
                "gallery": file_hash(gallery), "appearances": list(ids)}
               for name, run, ids, gallery, _ in configs]
    with stage_lock(root / ".prepare.lock"):
        path = root / "study.json"
        if path.is_file():
            existing = read_json(path)
            if existing["request"] != request or existing["siglip_lock_hash"] != digest(read_json(args.siglip_lock)):
                raise ValueError("These inputs differ from the frozen study. Choose a new root.")
            return {"path": str(path), "matches": [row["id"] for row in existing["matches"]], "reused": True}
        matches = [prepare_match(root, run, name, ids, manifest, gallery, scoreboard, args.cache, args.siglip_lock)
                   for (name, run, ids, gallery, scoreboard), manifest in zip(configs, manifests, strict=True)]
        midbeast = matches[0]
        previous = historical_replay(args.midbeast_run, midbeast["scope"])
        frozen = frozen_cases(root, args.midbeast_run, manifests[0], midbeast["appearances"])
        if frozen["scope"] != midbeast["scope"] or frozen["gallery_sha256"] != midbeast["gallery_input_sha256"]:
            raise ValueError("The frozen verifier and the Midbeast gallery differ")
        study = {"schema": 1, "request": request, "matches": matches, "historical_replay": previous,
                 "frozen_verifier": frozen, "siglip_lock_hash": digest(read_json(args.siglip_lock)),
                 "source_snapshot_sha256": source_snapshot(root)}
        atomic_json(path, study)
    return {"path": str(path), "matches": [row["id"] for row in matches], "frozen_cases": len(frozen["cases"])}


def flatten_shortlists(manifest):
    return [case for row in manifest["shortlists"] for case in row["cases"]]


def verification_manifest(root, study, study_sha):
    output = root / "verification"
    with stage_lock(output / ".prepare.lock"):
        path = output / "shortlists.json"
        if path.is_file():
            manifest = read_json(path)
            if manifest["study_sha256"] != study_sha:
                raise ValueError("Shortlists were prepared for another study")
            return manifest, []
        manifest, pending = prepare_shortlists(root, study, study_sha)
        if manifest is not None:
            atomic_json(path, manifest)
        return manifest, pending


def verify(args, study, study_sha):
    cases = list(study["frozen_verifier"]["cases"])
    if not args.frozen_only:
        manifest, pending = verification_manifest(args.root, study, study_sha)
        if pending:
            return {"complete": False, "pending_tracks": pending}
        shortlist_sha = file_hash(args.root / "verification" / "shortlists.json")
        if args.prepare_only:
            return draft_reviews(args.root, study, study_sha, manifest, shortlist_sha)
        _, reviews_sha = load_reviews(args.root, study, study_sha, manifest, shortlist_sha)
        freeze_reviews(args.root, reviews_sha, create=True)
        cases += flatten_shortlists(manifest)
    if args.prepare_only:
        return {"complete": True, "frozen_cases": len(cases)}
    return {"complete": True, **infer(args.root, study_sha, cases, args.cache, args.vl_python)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Stage pinned study inputs and source")
    prep.add_argument("--midbeast-run", type=Path, required=True)
    prep.add_argument("--midbeast-scoreboard", type=Path)
    prep.add_argument("--second-run", type=Path)
    prep.add_argument("--second-scoreboard", type=Path)
    prep.add_argument("--second-gallery", type=Path)
    prep.add_argument("--second-appearance", action="append")
    tracker = commands.add_parser("track", help="Run one paired tracking arm")
    tracker.add_argument("--arm", choices=ARMS, required=True)
    tracker.add_argument("--sam21-weights", type=Path)
    tracker.add_argument("--smoke", action="store_true", help="Track one appearance per match")
    verifier = commands.add_parser("verify", help="Prepare mask reviews or run the contextual Qwen verifier")
    verifier.add_argument("--vl-python", default=sys.executable)
    verifier.add_argument("--prepare-only", action="store_true")
    scorer = commands.add_parser("score", help="Score frozen crops and replay SigLIP locks")
    scorer.add_argument("--require-complete", action="store_true")
    for stage in (prep, tracker, verifier, scorer):
        stage.add_argument("--root", type=Path, required=True)
        stage.add_argument("--cache", type=Path, default=Path("data/league_assets"))
    for stage in (prep, scorer):
        stage.add_argument("--siglip-lock", type=Path, default=DEFAULT_SIGLIP_LOCK)
    for stage in (verifier, scorer):
        stage.add_argument("--frozen-only", action="store_true")
    args = parser.parse_args(argv)
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    if args.command == "prepare":
        result = prepare(args)
    else:
        path = args.root / "study.json"
        study, study_sha = read_json(path), file_hash(path)
        if study["schema"] != 1:
            raise ValueError("Unsupported SAM skin study version")
        if file_hash(args.root / "source" / "SOURCE_SHA256SUMS") != study["source_snapshot_sha256"]:
            raise ValueError("The source snapshot checksum list differs from the prepared study")
        if args.command == "track":
            result = track(args.root, study, study_sha, args.arm, args.cache, args.sam21_weights, smoke=args.smoke)
        elif args.command == "verify":
            result = verify(args, study, study_sha)
        else:
            result = score(args, study, study_sha)
    print(json.dumps(result, indent=2))
    if ((args.command == "score" and args.require_complete or args.command == "verify")
            and not result.get("complete", True)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
