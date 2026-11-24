"""Build an opt-in body identity timeline from a prepared campaign and effective intake."""

from collections import defaultdict
from functools import cached_property, lru_cache
import os
from pathlib import Path
import resource
import sys
import tempfile
import time

import numpy as np
from PIL import Image

from .body_identity import healthbar_proposals
from .body_names import resolve_names
from .body_roster import RosterIndex, name_key
from .core import Store, atomic_json, digest, implementation_hash, read_json, uid
from .media import file_hash
from .normalization import effective_intake


def effective_observations(store):
    with store.review_snapshot():
        corrections = store.corrections()
        rows = effective_intake(store.effective(store.rows("observations"), "observations"),
                                {row["record_id"] for row in corrections if row["target"] == "observations"})
    return rows


def track_revision(root):
    return digest({path.name: file_hash(path) for path in sorted((root / "tracks").glob("*-complete.json"))})


def skin_pool_report(cache, campaign, rosters):
    from .reference_bundle import skin_root

    version = campaign.get("reference_version", "16.22.1")
    community = campaign.get("communitydragon_version", version.rsplit(".", 1)[0])
    path = skin_root(cache, version, community) / "manifest.json"
    if not path.exists():
        return {"status": "missing_catalog", "reference_version": version, "communitydragon_version": community}
    catalog = read_json(path)
    if (catalog.get("status") != "complete" or catalog.get("reference_version") != version
            or catalog.get("communitydragon_version") != community):
        raise ValueError("Body skin pool does not match the pinned catalog")
    eligible = {name for roster in rosters.rosters.values() for name in roster["champions"]}
    available = defaultdict(set)
    for entry in catalog["assets"]:
        if entry["id"] in eligible:
            available[entry["id"]].add(str(entry["skin_id"]))
    admitted = defaultdict(set)
    for entry in campaign["gallery"]:
        if entry.get("skin_id") is not None and str(entry["skin_id"]) in available[entry["champion"]]:
            admitted[entry["champion"]].add(str(entry["skin_id"]))
    return {"status": "pinned", "reference_version": version, "communitydragon_version": community,
            "catalog_hash": digest(catalog), "champions": {
                name: {"eligible_skin_ids": sorted(available[name]), "admitted_skin_ids": sorted(admitted[name]),
                       "skin_reference_gaps": sorted(available[name] - admitted[name])} for name in sorted(eligible)}}


def calibrated_profiles(store):
    from .intake_profile import validate_profile

    result = set()
    for path in store.path("layout_profiles").glob("*.json"):
        profile = read_json(path)
        if profile.get("calibrated") is True and profile.get("inventory"):
            validate_profile(profile)
            result.add(digest(profile))
    return result


def overlap(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def body_mask(image, path, profile, bars):
    with Image.open(path) as image_mask:
        if image_mask.size != image.size:
            raise ValueError("Body mask does not use native source coordinates")
        pixels = np.asarray(image_mask.convert("L")) > 127
    keep = np.zeros_like(pixels)
    x, y, right, bottom = profile["scene_box"]
    keep[y:bottom, x:right] = True
    for x, y, right, bottom in profile["body_binder"]["exclude_boxes"]:
        keep[y:bottom, x:right] = False
    for bar in bars:
        x, y, right, bottom = bar["bar_box"]
        keep[max(0, y - 36):bottom + 3, max(0, x - 55):right + 35] = False
    cleaned = pixels & keep
    removed = 1 - cleaned.sum() / max(1, pixels.sum())
    return Image.fromarray(cleaned.astype(np.uint8) * 255), float(removed)


def body_quality(body, foreground, ui_fraction, clipped):
    pixels = np.asarray(body.convert("L"), dtype=float)
    mask = np.asarray(foreground) > 127
    horizontal = mask[:, 1:] & mask[:, :-1]
    vertical = mask[1:] & mask[:-1]
    edge = (np.abs(np.diff(pixels, axis=1))[horizontal].sum() + np.abs(np.diff(pixels, axis=0))[vertical].sum())
    sharpness = edge / max(1, horizontal.sum() + vertical.sum()) / 255
    inner_edges = (np.count_nonzero(mask[:, 1:] != mask[:, :-1]) + np.count_nonzero(mask[1:] != mask[:-1, :]))
    fragmentation = inner_edges / max(1, mask.sum())
    return (min(body.height / 100, 1) + sharpness + mask.mean() * .25
            - ui_fraction - min(fragmentation, 1) * .25 - float(clipped))


def associated_name_box(box, bars, profile, image_size, policy):
    if not policy or policy.get("display_mode") != "player" or policy.get("source_reviewed") is not True:
        return None
    matches = [bar for bar in bars if overlap(bar["actor_region"], box) >= .6 * overlap(box, box)]
    if len(matches) != 1:
        return None
    bar = matches[0]["bar_box"]
    x, y, right, bottom = policy["offsets"]
    region = [bar[0] + x, bar[1] + y, bar[2] + right, bar[1] + bottom]
    if not (0 <= region[0] < region[2] <= image_size[0] and 0 <= region[1] < region[3] <= image_size[1]):
        return None
    if any(overlap(region, other["bar_box"]) for other in bars):
        return None
    if any(overlap(region, exclude) for exclude in profile["body_binder"]["exclude_boxes"]):
        return None
    # A second plausible bar must not share the nameplate crop.
    for other in bars:
        if other is matches[0]:
            continue
        a, b, c, _ = other["bar_box"]
        if overlap(region, [a + x, b + y, c + right, b + bottom]):
            return None
    return region


def track_samples(root, prepared, source_sha256, *, frame_ids=None):
    from .action_v3 import load_profile

    jobs = {job["id"]: job for job in prepared["jobs"]}
    for clip in prepared["clips"]:
        if clip.get("source_sha256") != source_sha256 or clip.get("continuity_source_reviewed") is not True:
            continue
        profile = load_profile(clip.get("action_profile"))
        nameplate = clip.get("nameplate")
        if nameplate and (nameplate.get("source_sha256") != source_sha256
                          or nameplate.get("size") != profile["size"]
                          or not nameplate.get("reviewer")):
            raise ValueError("Nameplate calibration does not match the clip source/layout")
        for job_id in clip["jobs"]:
            job = jobs[job_id]
            path = root / "tracks" / (job_id + "-complete.json")
            if not path.exists():
                continue
            result = read_json(path)
            for record in result["frames"]:
                frame = job["frames"][record["frame_index"]]
                if frame_ids is not None and frame["id"] not in frame_ids:
                    continue
                frame_path = Path(frame["path"])
                if file_hash(frame_path) != frame["sha256"]:
                    raise ValueError("Tracked frame pixels changed")
                with Image.open(frame_path) as original:
                    image = original.convert("RGB")
                bars = healthbar_proposals(image, profile)
                for item in record["masks"]:
                    object_id = item["object_id"]
                    if not 1 <= object_id <= len(job["proposals"]):
                        raise ValueError("Mask does not refer to a prepared proposal")
                    path = Path(item["mask_path"])
                    if file_hash(path) != item["sha256"]:
                        raise ValueError("Tracked mask pixels changed")
                    mask, ui_fraction = body_mask(image, path, profile, bars)
                    box = mask.getbbox()
                    track_id = f"{job_id}-{object_id}"
                    row = {
                        "id": digest([frame["id"], track_id])[:20], "frame_id": frame["id"],
                        "source_sha256": source_sha256, "source_frame_sha256": frame["sha256"],
                        "track_id": track_id, "segment_id": clip["id"],
                        "owner_epoch": frame.get("owner_epochs", {}).get("selected", 0),
                        "time": frame["time"], "team": job["proposals"][object_id - 1]["team"],
                        "visible": box is not None, "match_group": clip["match_group"], "split": clip["split"],
                        "accepted_roster": clip.get("accepted_roster"),
                        "mask_sha256": item["sha256"], "frame_path": str(frame_path), "body_box": box,
                    }
                    if box is None:
                        yield row
                        continue
                    x, y, right, bottom = box
                    clipped = x == 0 or y == 0 or right == image.width or bottom == image.height
                    body, foreground = image.crop(box), mask.crop(box)
                    row["quality"] = body_quality(body, foreground, ui_fraction, clipped)
                    row["name_box"] = associated_name_box(box, bars, profile, image.size, nameplate)
                    row["nameplate"] = nameplate if row["name_box"] else None
                    contexts = [c for c in clip.get("hud_contexts", []) if c.get("source_frame_sha256") == frame["sha256"]
                                and c["time"] == frame["time"] and c.get("binding_visible")
                                and c.get("champion") and c.get("actor_region")]
                    if len(contexts) == 1:
                        context = contexts[0]
                        hud_box = context["actor_region"]
                        common = overlap(box, hud_box)
                        if common >= .5 * (overlap(box, box) + overlap(hud_box, hud_box) - common):
                            row.update(accepted_hud_champion=context["champion"], hud_body_binding_accepted=True)
                    identity = digest([row["id"], frame["sha256"], item["sha256"], list(box), profile])
                    crop_root = root / "identity-crops" / identity
                    crop_root.mkdir(parents=True, exist_ok=True)
                    for filename, crop in (("body.png", body), ("mask.png", foreground)):
                        target = crop_root / filename
                        if not target.exists():
                            with tempfile.TemporaryDirectory(dir=crop_root) as directory:
                                temporary = Path(directory) / filename
                                crop.save(temporary)
                                temporary.replace(target)
                    row.update(body_path=str(crop_root / "body.png"), crop_mask_path=str(crop_root / "mask.png"),
                               body_sha256=file_hash(crop_root / "body.png"), crop_mask_sha256=file_hash(crop_root / "mask.png"))
                    yield row


class NameReader:
    def __init__(self, root, reviews):
        self.root = root
        self.reviews = {row["id"]: row for row in reviews if row.get("review_status") in {"human-verified", "rejected"}}

    @cached_property
    def signature(self):
        import shutil
        import subprocess

        binary = shutil.which(os.environ.get("LEAGUE_TESSERACT", "tesseract"))
        if not binary:
            raise ValueError("Nameplate OCR requires Tesseract")
        version = subprocess.check_output([binary, "--version"], text=True, timeout=10).splitlines()[0]
        return digest([version, file_hash(Path(binary)), os.environ.get("LEAGUE_TESSDATA"),
                       implementation_hash(["body_operational.py", "readers.py", "cast_bars.py"])])

    def __call__(self, row):
        from .cast_bars import text_preprocess
        from .readers import tesseract

        if row["id"] in self.reviews:
            return self.reviews[row["id"]]
        box = row.get("name_box")
        if not box:
            return None
        policy = row["nameplate"]
        with Image.open(row["frame_path"]) as image:
            crop = image.crop(box).convert("RGB")
        key = digest([self.signature, list(box), crop.tobytes().hex(), policy])
        directory = self.root / "name-cache" / key
        directory.mkdir(parents=True, exist_ok=True)
        receipt = directory / "reading.json"
        evidence_path = directory / "source.png"
        if not receipt.exists():
            crop.save(evidence_path)
            prepared = directory / "ocr.png"
            processed = text_preprocess(crop)
            processed.save(prepared)
            glyphs = np.asarray(processed)[12:-12, 12:-12] < 128
            clipped = glyphs[0].any() or glyphs[-1].any() or glyphs[:, 0].any() or glyphs[:, -1].any()
            text = {"raw_text": "", "score": 0, "clipped": True} if clipped else tesseract(prepared, language=policy.get("language", "eng"))
            atomic_json(receipt, text)
        result = read_json(receipt)
        text = name_key(result["raw_text"])
        valid = bool(text and result["score"] >= policy["ocr_threshold"])
        return {"id": uid("body-name-read", row["id"], key), "source_sha256": row["source_sha256"],
                "frame_id": row["frame_id"], "track_id": row["track_id"], "time": row["time"],
                "name_kind": "player", "associated": True, "value": result["raw_text"] if valid else None,
                "raw_text": result["raw_text"], "score": result["score"],
                "visibility": "visible" if valid else "unknown", "acceptance": "accepted" if valid else "unknown",
                "evidence_path": str(evidence_path), "review_status": "unreviewed"}


def verified_crop(path, expected):
    if file_hash(Path(path)) != expected:
        raise ValueError("Body crop changed after extraction")
    with Image.open(path) as image:
        return image.copy()


class LazyMatcher:
    def __init__(self, root, prepared, lock, policy, gallery_kind, device, reviews=()):
        self.root, self.prepared, self.lock, self.policy = root, prepared, lock, policy
        self.kind, self.device = gallery_kind, device
        self.reviews = {row["id"]: row for row in reviews}
        self.encoder_calls = 0
        self.gallery_calls = 0

    @cached_property
    def encoder(self):
        from .body_identity import FrozenEncoder

        if digest(read_json(self.lock)) != self.prepared["recipe"]["encoder_lock_hash"]:
            raise ValueError("Body encoder lock changed after preparation")
        return FrozenEncoder(self.lock, self.device)

    def features(self, image, mask, image_sha, mask_sha, *, variant=None):
        from .body_features import encode_features, render_variant

        encoder = self.encoder
        policy = {key: value for key, value in self.policy["features"].items() if key not in {"render_variant", "weights"}}
        policy["weights"] = [1, int(bool(self.policy["features"].get("weights", [1, 0, 0])[1])), 0]
        key = digest([encoder.signature, image_sha, mask_sha, policy, variant, implementation_hash(["body_features.py"])])
        path = self.root / "identity-feature-cache" / (key + ".json")
        if path.exists():
            return read_json(path)
        self.encoder_calls += 1
        body, foreground = verified_crop(image, image_sha), verified_crop(mask, mask_sha)
        if variant:
            body, foreground = render_variant(body, foreground, **variant)
        result = encode_features(encoder, body, foreground, policy)
        atomic_json(path, result)
        return result

    @lru_cache(maxsize=16)
    def gallery(self, roster):
        from .body_features import FeatureGallery

        entries = [entry for entry in self.prepared["gallery"] if entry["champion"] in roster
                   and (self.kind == "combined" or entry["reference_kind"] == self.kind)]
        if not entries:
            return None
        features = [self.features(entry["path"], entry["mask_path"], entry["sha256"], entry["mask_sha256"]) for entry in entries]
        variant = self.policy["features"].get("render_variant")
        if variant:
            for entry in list(entries):
                if entry["reference_kind"] == "rendered":
                    features.append(self.features(entry["path"], entry["mask_path"], entry["sha256"], entry["mask_sha256"], variant=variant))
                    entries.append({**entry, "id": uid("render-variant", entry["id"], variant),
                                    "parent_template_id": entry.get("parent_template_id", entry["id"]), "variant": variant})
        return FeatureGallery(entries, features, encoder_signature=self.encoder.signature,
                              excluded_groups={clip["match_group"] for clip in self.prepared["clips"] if clip["split"] != "gallery"})

    def __call__(self, row, roster):
        review = self.reviews.get(row["id"])
        if (review and review.get("source_sha256") == row["source_sha256"]
                and review.get("frame_id") == row["frame_id"] and review.get("track_id") == row["track_id"]):
            if review.get("review_status") == "rejected":
                return {"champion": None, "reason": "rejected visual evidence"}
            if review.get("review_status") == "human-verified":
                return {"champion": review.get("champion"), "reason": "reviewed visual evidence", "review_id": review["id"]}
        if not self.policy:
            return {"champion": None, "reason": "missing frozen appearance policy"}
        gallery = self.gallery(tuple(roster["champions"]))
        if gallery is None:
            return {"champion": None, "reason": "missing admitted roster references"}
        features = self.features(row["body_path"], row["crop_mask_path"], row["body_sha256"], row["crop_mask_sha256"])
        self.gallery_calls += 1
        return {**gallery.match(features, self.policy["matching"], roster["champions"]),
                "query_geometry": features["geometry"], "native_body_box": row["body_box"]}


def identity_inputs(root, intake_run, cache, slot_skim):
    from .body_campaign import verify_campaign
    from .inventory_references import load_inventory_references, reference_root

    prepared = read_json(root / "prepared.json")
    campaign = verify_campaign(read_json(root / "campaign.json"), cache)
    if prepared["recipe"]["campaign_hash"] != digest(campaign):
        raise ValueError("Body campaign changed after preparation")
    if prepared["gallery"] != campaign["gallery"]:
        raise ValueError("Prepared gallery differs from the campaign")
    intake = Store(Path(intake_run))
    source = read_json(intake.path("manifest.json"))["source"]
    source_hash = source["sha256"]
    if not any(clip.get("source_sha256") == source_hash for clip in prepared["clips"]):
        raise ValueError("Intake source does not match any prepared body clip")
    version = campaign.get("reference_version", "16.22.1")
    catalog = load_inventory_references(reference_root(cache, version), version)
    champion_ids = {entry["id"] for entry in catalog["assets"] if entry["kind"] == "champion"}
    observations = effective_observations(intake)
    profile_hashes = calibrated_profiles(intake)
    if slot_skim is None:
        rosters = RosterIndex(observations, source_hash, champion_ids, profile_hashes)
    else:
        from .body_slot_context import MatchRosterIndex

        match_groups = {clip["match_group"] for clip in prepared["clips"] if clip.get("source_sha256") == source_hash}
        if len(match_groups) != 1:
            raise ValueError("Scoreboard roster reuse requires one verified match for the source")
        rosters = MatchRosterIndex(slot_skim, source_hash, champion_ids)
    return prepared, campaign, intake, source_hash, catalog, observations, profile_hashes, rosters


def identify(root, intake_run, cache, *, policy_path=None, gallery_kind="combined", device="cpu", slot_skim=None):
    started = time.monotonic()
    root = Path(root).resolve()
    prepared, campaign, _, source_hash, catalog, observations, profile_hashes, rosters = identity_inputs(
        root, intake_run, cache, slot_skim)
    skins = skin_pool_report(cache, campaign, rosters)
    tracking = track_revision(root)
    store = Store(root)
    with store.review_snapshot():
        reviewed_names = store.effective(store.rows("body_names"), "body_names")
        reviewed_matches = store.effective(store.rows("body_matches"), "body_matches")
    policy = None
    policy_path = Path(policy_path) if policy_path else root / "appearance-policy.json"
    if policy_path.exists():
        from .body_appearance import validate_policy

        policy = validate_policy(root, read_json(policy_path), prepared, campaign, gallery_kind)
    matcher = LazyMatcher(root, prepared, root / "encoder-lock.json", policy, gallery_kind, device, reviewed_matches)
    reader = NameReader(root, reviewed_names)
    result = resolve_names(track_samples(root, prepared, source_hash), rosters, reader, matcher,
                           cadence=policy.get("cadence", "quality") if policy else "quality")
    if slot_skim is not None:
        from .body_slot_context import skim_revision_matches

        if not skim_revision_matches(rosters.revision):
            raise ValueError("Scoreboard review changed during identification. Rerun against a complete revision.")
    if track_revision(root) != tracking:
        raise ValueError("Tracking results changed during identification. Rerun against a complete revision.")
    recipe = {
        "prepared": digest(prepared), "intake_observations": digest(observations), "profiles": sorted(profile_hashes),
        "names": digest(reviewed_names), "matches": digest(reviewed_matches), "policy": digest(policy),
        "catalog": digest(catalog), "skin_pool": digest(skins), "tracks": tracking,
        "source_sha256": source_hash, "gallery_kind": gallery_kind, "device": device,
        "name_reader": reader.__dict__.get("signature"),
        "encoder": matcher.encoder.signature if "encoder" in matcher.__dict__ else None,
        "slot_skim": rosters.revision if slot_skim is not None else None,
        "implementation": implementation_hash(["body_operational.py", "body_names.py", "body_roster.py",
                                                 "body_features.py", "body_identity.py", "body_slot_context.py",
                                                 "scoreboard_skim.py"]),
    }
    result["counts"].update(encoder_calls=matcher.encoder_calls, gallery_calls=matcher.gallery_calls)
    result.update(experimental=True, release_passed=False, recipe=recipe, skin_pool=skins, seconds=time.monotonic() - started,
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024))
    # One atomic result contains predictions and the exact roster/name evidence used by them.
    output = root / "identities" / (digest(recipe)[:20] + ".json")
    if not output.exists():
        atomic_json(output, result)
    else:
        result = read_json(output)
    atomic_json(root / "identity-latest.json", {"path": str(output), "sha256": file_hash(output), "recipe": recipe})
    return {"path": str(output), "counts": result["counts"], "rosters": len(result["rosters"]),
            "experimental": True, "release_passed": False}
