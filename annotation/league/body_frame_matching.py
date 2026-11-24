"""Match tracked bodies in one source frame to team-scoped skin renders."""

from collections import Counter, defaultdict
from functools import cached_property
from pathlib import Path

import numpy as np

from .body_features import normalize_crop
from .body_identity import FrozenEncoder, reference_approved, unit
from .body_operational import (calibrated_profiles, effective_observations, identity_inputs,
                               skin_pool_report, track_revision, track_samples, verified_crop)
from .body_siglip import DEFAULT_SIGLIP_LOCK, SiglipEncoder
from .body_slot_context import skim_revision_matches
from .core import atomic_json, digest, finite, implementation_hash, read_json


class SkinMatcher:
    def __init__(self, entries, cache, lock, lock_hash, patch, *, encoder_name="dinov2"):
        self.entries = defaultdict(lambda: defaultdict(list))
        identifiers = set()
        for entry in entries:
            if entry["reference_kind"] != "rendered" or entry.get("skin_id") is None:
                continue
            if not reference_approved(entry) or entry["id"] in identifiers:
                raise ValueError("Skin matching needs distinct, audited render references")
            identifiers.add(entry["id"])
            self.entries[entry["champion"]][str(entry["skin_id"])].append(entry)
        self.cache, self.lock, self.lock_hash, self.patch = cache, lock, lock_hash, patch
        self.encoder_name = encoder_name
        self.encoded = Counter()
        self.vectors = {}

    @cached_property
    def encoder(self):
        if self.encoder_name == "siglip":
            return SiglipEncoder(self.lock, self.lock_hash)
        if digest(read_json(self.lock)) != self.lock_hash:
            raise ValueError("Body encoder lock changed after preparation")
        return FrozenEncoder(self.lock, "cpu")

    @cached_property
    def cache_dir(self):
        recipe = digest([self.patch, self.encoder_name, self.encoder.signature,
                         implementation_hash(["body_features.py", "body_frame_matching.py"])])
        return self.cache / "body-skin-matches" / "embeddings" / recipe

    def vectors_for(self, inputs, kind):
        keys = list(inputs)
        pending = {}

        def encode_pending():
            images = [image for _, image in pending.values()]
            if self.encoder_name == "siglip":
                vectors = [unit(vector) for vector in self.encoder.describe_batch(images)]
            else:
                vectors = [self.encoder.describe(image)[0] for image in images]
            for (target, (aliases, _)), vector in zip(pending.items(), vectors, strict=True):
                atomic_json(target, vector.tolist())
                for key in aliases:
                    self.vectors[key] = vector
                self.encoded[kind] += 1
            pending.clear()

        for key in dict.fromkeys(keys):
            if key in self.vectors:
                continue
            path, mask_path, sha, mask_sha = key
            # Check the input bytes even when we can reuse their embedding from disk.
            image = verified_crop(path, sha)
            mask = verified_crop(mask_path, mask_sha)
            target = self.cache_dir / (digest([sha, mask_sha]) + ".json")
            if target.exists():
                self.vectors[key] = unit(read_json(target))
                continue
            if target in pending:
                pending[target][0].append(key)
                continue
            normalized, _, _ = normalize_crop(image, mask)
            pending[target] = [key], normalized
            if len(pending) == 16:
                encode_pending()
        if pending:
            encode_pending()
        return np.stack([self.vectors[key] for key in keys])

    def score_views(self, queries, entries):
        queries = self.vectors_for(queries, "query")
        views = self.vectors_for(((entry["path"], entry["mask_path"], entry["sha256"], entry["mask_sha256"])
                                  for entry in entries), "render")
        return np.clip(queries @ views.T, -1, 1)

    def match(self, row, pool, assigned_skins):
        eligible = {name: self.entries.get(name, {}) for name in pool}
        for name, assignment in assigned_skins.items():
            if name in eligible:
                skin = assignment["skin_id"]
                eligible[name] = {skin: eligible[name].get(skin, [])}
        entries = [entry for skins in eligible.values() for renders in skins.values() for entry in renders]
        missing = [name for name, skins in eligible.items() if not any(skins.values())]
        result = {"champion": None, "skin_id": None, "best_reference_skin": None, "reason": "missing_champion_renders",
                  "missing_champion_renders": missing, "candidate_pool": pool, "candidates": [],
                  "used_assigned_skin": False, "render_comparisons": len(entries)}
        if not entries:
            return result
        query = (row["body_path"], row["crop_mask_path"], row["body_sha256"], row["crop_mask_sha256"])
        scores = self.score_views([query], entries)[0]
        winners = {}
        for entry, score in zip(entries, scores):
            champion, skin = entry["champion"], str(entry["skin_id"])
            previous = winners.get(champion)
            candidate = {"champion": champion, "skin_id": skin, "score": float(score),
                         "reference_id": entry["id"], "reference_sha256": entry["sha256"],
                         "skin_already_assigned": champion in assigned_skins}
            if previous is None or (-score, skin, entry["id"]) < (-previous["score"], previous["skin_id"], previous["reference_id"]):
                winners[champion] = candidate
        candidates = sorted(winners.values(), key=lambda item: (-item["score"], item["champion"]))
        best, *others = candidates
        margin = best["score"] - others[0]["score"] if others else None
        accepted = not missing and margin != 0
        return {**result, "champion": best["champion"] if accepted else None,
                "skin_id": best["skin_id"] if accepted else None,
                "best_reference_skin": best["skin_id"], "score": best["score"],
                "margin": margin, "reason": result["reason"] if missing else "matched" if accepted else "tied_champions",
                "used_assigned_skin": accepted and best["skin_already_assigned"], "candidates": candidates}


def team_pools(roster, main_champion, friendly_side):
    sides = {side for side, name in roster["membership"] if name == main_champion}
    if friendly_side is None and len(sides) == 1:
        friendly_side = next(iter(sides))
    if friendly_side not in sides:
        raise ValueError("The main champion must identify the friendly scoreboard side. Set --friendly-side for a mirror.")
    pools = {team: sorted(name for side, name in roster["membership"]
                          if (side == friendly_side) == (team == "friendly")
                          and (team != "friendly" or name != main_champion))
             for team in ("friendly", "enemy")}
    return friendly_side, pools


class UnresolvedMainTrack(ValueError):
    pass


def match_bodies(rows, pools, matcher, assigned_skins, main_champion, main_track=None, main_offscreen=False):
    visible = [row for row in rows if row["visible"]]
    friendly = [row for row in visible if row["team"] == "friendly"]
    hud_tracks = {row["track_id"] for row in friendly if row.get("hud_body_binding_accepted")
                  and row.get("accepted_hud_champion") == main_champion}
    if main_track is not None:
        if main_offscreen or main_track not in {row["track_id"] for row in friendly}:
            raise ValueError("--main-track must name a visible friendly track")
    elif len(hud_tracks) == 1 and not main_offscreen:
        main_track = next(iter(hud_tracks))
    elif friendly and (not main_offscreen or hud_tracks):
        tracks = ", ".join(row["track_id"] for row in friendly)
        raise UnresolvedMainTrack(f"Main player track is unresolved. Use --main-track or --main-offscreen. Friendly tracks: {tracks}")

    results = []
    for team, pool in pools.items():
        bodies = [row for row in visible if row["team"] == team and row["track_id"] != main_track]
        matches = [{**row, **matcher.match(row, pool, assigned_skins[team])} for row in bodies]
        scores = defaultdict(list)
        for row in matches:
            if row["champion"]:
                scores[row["champion"]].append(row["score"])
        for row in matches:
            competing = sorted(scores.get(row["champion"], ()), reverse=True)
            if len(competing) > 1 and (row["score"] < competing[0] or competing[0] == competing[1]):
                row.update(champion=None, skin_id=None, used_assigned_skin=False, reason="duplicate_team_champion")
        results.extend(matches)
    return results, main_track


def render_matcher(root, cache, prepared, campaign, rosters, render_gallery, *, siglip_lock=None):
    entries = read_json(Path(render_gallery)) if render_gallery else prepared["gallery"]
    entries = [entry for entry in entries if entry["reference_kind"] == "rendered" and entry.get("skin_id") is not None]
    skins = skin_pool_report(cache, {**campaign, "gallery": entries}, rosters)
    if skins["status"] == "pinned":
        eligible = skins["champions"]
        entries = [entry for entry in entries if entry["champion"] in eligible and
                   str(entry["skin_id"]) in eligible[entry["champion"]]["eligible_skin_ids"]]
    patch = [skins["reference_version"], skins["communitydragon_version"]]
    if siglip_lock is not None:
        lock = Path(siglip_lock).resolve()
        matcher = SkinMatcher(entries, cache, lock, digest(read_json(lock)), patch, encoder_name="siglip")
    else:
        matcher = SkinMatcher(entries, cache, root / "encoder-lock.json", prepared["recipe"]["encoder_lock_hash"], patch)
    return matcher, skins


def skin_lock_scope(matcher, pools, source, roster_id):
    gallery = sorted((name, skin, reference["id"], reference["sha256"], reference["mask_sha256"])
                     for name in set().union(*pools.values())
                     for skin, views in matcher.entries.get(name, {}).items() for reference in views)
    return {"source_sha256": source, "roster_id": roster_id, "team_pools": pools, "patch": matcher.patch,
            "encoder": matcher.encoder_name, "encoder_lock_hash": matcher.lock_hash, "gallery_hash": digest(gallery)}


def verify_revisions(root, tracking, intake, observations, profiles, rosters, slot_skim):
    changed_roster = (not skim_revision_matches(rosters.revision) if slot_skim is not None else
                      observations != effective_observations(intake) or profiles != calibrated_profiles(intake))
    if changed_roster or tracking != track_revision(root):
        raise ValueError("Scoreboard or tracking evidence changed during frame matching. Rerun against a complete revision.")


def record_assignments(assigned, results):
    new_assignments = [row for row in results if row["champion"] and not row["used_assigned_skin"]]
    for row in new_assignments:
        assigned[row["team"]][row["champion"]] = {
            "skin_id": row["skin_id"], "frame_id": row["frame_id"], "track_id": row["track_id"],
            "score": row["score"], "reference_id": row["candidates"][0]["reference_id"],
        }
    return len(new_assignments)


def match_frame(root, intake_run, cache, *, frame_time, main_champion, slot_skim=None,
                friendly_side=None, main_track=None, main_offscreen=False, clip_id=None,
                render_gallery=None, siglip_lock=None, encoder="siglip", skin_assignments=None):
    root, cache = Path(root).resolve(), Path(cache).resolve()
    if not finite(frame_time) or frame_time < 0:
        raise ValueError("Frame time must be a nonnegative source timestamp")
    prepared, campaign, intake, source, _, observations, profiles, rosters = identity_inputs(
        root, intake_run, cache, slot_skim)
    frames = [(clip, frame) for clip in prepared["clips"] if clip["source_sha256"] == source
              and (clip_id is None or clip["id"] == clip_id)
              for frame in clip["frames"] if abs(frame["time"] - frame_time) <= 1e-6]
    if len(frames) != 1:
        raise ValueError("Specify one exact prepared frame time. Use --clip-id for overlapping clips.")
    clip, frame = frames[0]
    roster = rosters.at(frame["time"])
    if not roster or (roster["status"] != "confirmed" and not (skin_assignments and roster["status"] == "experimental")):
        raise ValueError("Frame needs a confirmed scoreboard or compatible appearance locks for its experimental roster")
    friendly_side, pools = team_pools(roster, main_champion, friendly_side)
    tracking = track_revision(root)
    rows = list(track_samples(root, {**prepared, "clips": [clip]}, source, frame_ids={frame["id"]}))
    if encoder not in {"siglip", "dinov2"} or (encoder == "dinov2" and siglip_lock is not None):
        raise ValueError("Choose SigLIP or DINOv2 with its matching checkpoint")
    matcher, skins = render_matcher(root, cache, prepared, campaign, rosters, render_gallery,
                                    siglip_lock=(siglip_lock or DEFAULT_SIGLIP_LOCK) if encoder == "siglip" else None)
    assigned = {"friendly": {}, "enemy": {}}
    if skin_assignments is not None:
        lock = read_json(Path(skin_assignments))
        if lock["scope"] != skin_lock_scope(matcher, pools, source, roster["id"]):
            raise ValueError("Appearance locks belong to a different match, roster, gallery, or encoder")
        assigned = lock["assignments"]
    results, excluded = match_bodies(rows, pools, matcher, assigned, main_champion, main_track, main_offscreen)
    verify_revisions(root, tracking, intake, observations, profiles, rosters, slot_skim)
    report = {"experimental": True, "frame": frame, "clip_id": clip["id"], "roster_id": roster["id"],
              "source_sha256": source, "main_champion": main_champion, "excluded_main_track": excluded,
              "friendly_scoreboard_side": friendly_side, "team_pools": pools, "skin_pool": skins,
              "assigned_skins": assigned, "skin_assignment_cache": str(skin_assignments) if skin_assignments else None,
              "scores_are_probabilities": False, "encoder": matcher.encoder_name,
              "encoder_signature": matcher.encoder.signature if "encoder" in matcher.__dict__ else None,
              "predictions": results, "counts": {
                  "visible_tracks": sum(row["visible"] for row in rows), "matched": sum(bool(row["champion"]) for row in results),
                  "new_assignments": 0, "reused_assignments": sum(row["used_assigned_skin"] for row in results),
                  "render_comparisons": sum(row["render_comparisons"] for row in results),
                  "query_encodes": matcher.encoded["query"], "render_encodes": matcher.encoded["render"],
              }}
    output = root / "frame-matches" / (digest(report)[:20] + ".json")
    atomic_json(output, report)
    return {"path": str(output), "counts": report["counts"]}
