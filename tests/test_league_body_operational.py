from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image, ImageDraw

from annotation.league.body_features import FeatureGallery, normalize_crop, patch_similarity, render_variant
from annotation.league.body_names import resolve_names, select_frames
from annotation.league.body_roster import RosterIndex
from annotation.league.core import atomic_json, digest, uid, write_rows
from annotation.league.media import file_hash


CHAMPIONS = {f"C{i}" for i in range(10)} | {"Other"}
SOURCE = "a" * 64


def scoreboard(time, *, segment="s", continuous=True, state="present", missing=None,
               reviewed=False, swap=False, change=False, named=False, source=SOURCE, profile="layout"):
    rows = []
    for index in range(10):
        side = "left" if index < 5 else "right"
        number = index % 5
        if swap and index < 5:
            number = (number + 1) % 5
        slot = f"{side}:row{number + 1}"
        base = {"task_id": f"task-{time}", "time": time, "sample_time": time, "route": "scoreboard",
                "source_id": uid("source", source, time), "slot_id": slot, "profile_hash": profile,
                "playback_segment_id": segment, "continuity_verified": continuous, "playback_state": "playing",
                "acceptance": "accepted", "visibility": "visible",
                "review_status": "human-verified" if reviewed else "unreviewed"}
        visible = index != missing and state == "present"
        rows.append({**base, "id": f"portrait-{time}-{index}", "field": "inventory.champion", "panel_state": state,
                     "inventory_catalog_id": "catalog", "value": ("Other" if change and index == 0 else f"C{index}") if visible else None,
                     "acceptance": "accepted" if visible else "unknown", "visibility": "visible" if visible else "unknown"})
        if named and index == 0 and visible:
            rows.append({**base, "id": f"name-{time}", "field": "player_name", "value": "Jó#NA"})
    return rows


def rosters(rows):
    return RosterIndex(rows, SOURCE, CHAMPIONS, {"layout"})


def sample(time, track="track", owner=0, **fields):
    return {"id": f"{track}-{time}", "time": time, "frame_id": f"frame-{time}", "source_sha256": SOURCE,
            "source_frame_sha256": f"pixels-{time}", "segment_id": "clip", "owner_epoch": owner,
            "track_id": track, "visible": True, **fields}


def name(row, value="Jó#NA", **fields):
    return {**{key: row[key] for key in ("time", "source_sha256", "frame_id", "track_id")},
            "id": "name-" + row["id"], "value": value, "associated": True, "name_kind": "player",
            "visibility": "visible", "acceptance": "accepted", **fields}


class RosterTests(unittest.TestCase):
    def test_two_adjacent_open_reads_with_reordering_backfill_and_survive_close(self):
        index = rosters(scoreboard(0, state="absent", continuous=False) + scoreboard(1, named=True)
                        + scoreboard(2, named=True, swap=True) + scoreboard(3, state="absent"))
        roster = index.at(.25)
        self.assertEqual(roster["confirmed_at"], 2)
        self.assertEqual(len(roster["champions"]), 10)
        self.assertEqual(roster["bindings"]["Jó#NA"]["champion"], "C0")
        self.assertEqual(index.at(3)["id"], roster["id"])
        self.assertIsNone(index.at(3.25))

    def test_one_complete_read_and_incomplete_or_reopened_samples_do_not_confirm(self):
        for rows in (scoreboard(0), scoreboard(0) + scoreboard(1, missing=9),
                     scoreboard(0) + scoreboard(1, state="absent") + scoreboard(2),
                     scoreboard(0) + scoreboard(2), scoreboard(0) + scoreboard(1, continuous=False)):
            with self.subTest(rows=len(rows)):
                self.assertFalse(rosters(rows).rosters)

    def test_human_snapshot_and_cut(self):
        index = rosters(scoreboard(0, reviewed=True, named=True) + scoreboard(1, segment="cut", continuous=False))
        self.assertEqual(index.at(0)["bindings"]["Jó#NA"]["champion"], "C0")
        self.assertIsNone(index.at(.5))
        self.assertIsNone(index.at(1))

    def test_conflict_suspends_and_correction_rebuilds(self):
        original = scoreboard(0) + scoreboard(1)
        disputed = rosters(original + scoreboard(2, change=True))
        self.assertIsNone(disputed.at(0))
        self.assertEqual(next(iter(disputed.rosters.values()))["status"], "disputed")
        self.assertIsNotNone(rosters(original + scoreboard(2)).at(2))

    def test_wrong_source_name_cannot_seed_same_row(self):
        rows = scoreboard(0, reviewed=True, named=True)
        rows[-10]["source_id"] = "other-source"
        self.assertFalse(rosters(rows).at(0)["bindings"])

    def test_duplicate_scoreboard_names_quarantine_the_key(self):
        rows = scoreboard(0, reviewed=True, named=True)
        duplicate = dict(rows[1], id="duplicate-name", slot_id="left:row2")
        roster = rosters(rows + [duplicate]).at(0)
        self.assertEqual(roster["disputed_names"], ["Jó#NA"])
        self.assertFalse(roster["bindings"])


class NameReuseTests(unittest.TestCase):
    def test_scoreboard_binding_bypasses_matcher_on_backfill_and_reentry(self):
        index = rosters(scoreboard(0, state="absent", continuous=False) + scoreboard(1, named=True)
                        + scoreboard(2, named=True) + scoreboard(3, state="absent"))
        matcher = Mock(side_effect=AssertionError("known name reached appearance"))
        rows = [sample(.25), sample(2.5, "re-entry"), sample(3, "another")]
        result = resolve_names(rows, index, name, matcher)
        self.assertEqual([row["champion"] for row in result["predictions"]], ["C0"] * 3)
        self.assertEqual([row["backfilled"] for row in result["predictions"]], [True, False, False])
        matcher.assert_not_called()

    def test_visual_binding_uses_three_votes_then_reuses_across_tracks(self):
        index = rosters(scoreboard(0, reviewed=True) + scoreboard(1, state="absent") + scoreboard(2, state="absent"))
        matcher = Mock(return_value={"champion": "C2", "skin_id": "unverified"})
        rows = [sample(t) for t in (0, .25, .5, .75, 1)] + [sample(1.5, "re-entry")]
        result = resolve_names(rows, index, name, matcher)
        self.assertEqual(matcher.call_count, 3)
        self.assertEqual(result["counts"]["registry_hits"], 3)
        self.assertTrue(all(row["champion"] == "C2" and row["skin_id"] is None for row in result["predictions"]))
        self.assertEqual(result["bindings"][0]["source"], "visual")

    def test_unreadable_unassociated_and_wrong_case_never_use_binding(self):
        index = rosters(scoreboard(0, reviewed=True, named=True) + scoreboard(1, state="absent"))
        matcher = Mock(side_effect=AssertionError("unverified name reached appearance"))
        rows = [sample(0), sample(.25), sample(.5), sample(.75)]
        readings = [name(rows[0], "jó#NA"), name(rows[1], associated=False),
                    name(rows[2], name_kind="champion"), name(rows[3], visibility="unknown")]
        result = resolve_names(rows, index, lambda row: readings[rows.index(next(r for r in rows if r["id"] == row["id"]))], matcher)
        self.assertFalse(any(row["champion"] for row in result["predictions"]))
        matcher.assert_not_called()

    def test_disputed_same_frame_name_and_cut_do_not_transfer(self):
        index = rosters(scoreboard(0, reviewed=True, named=True) + scoreboard(1, state="absent")
                        + scoreboard(2, segment="cut", continuous=False))
        matcher = Mock(side_effect=AssertionError("disputed name reached appearance"))
        result = resolve_names([sample(0), sample(0, "overlap"), sample(2, "cut")], index, name, matcher)
        self.assertFalse(any(row["champion"] for row in result["predictions"]))
        matcher.assert_not_called()

    def test_boundary_and_duplicate_pixels_cannot_vote(self):
        index = rosters(scoreboard(0, reviewed=True) + scoreboard(1, state="absent"))
        rows = [sample(0), sample(.25, source_frame_sha256="pixels-0"), sample(.5, owner=1), sample(.75, owner=1)]
        matcher = Mock(return_value={"champion": "C0"})
        result = resolve_names(rows, index, lambda row: name(row, review_status="human-verified"), matcher)
        self.assertFalse(result["bindings"])

    def test_quality_selection_has_regular_vote_time_and_native_evidence_time(self):
        rows = [sample(0, quality=.1), sample(.125, quality=.8), sample(.25, quality=.3)]
        selected = select_frames(rows)
        self.assertEqual([(row["time"], row["vote_time"]) for row in selected], [(.125, .25), (.25, .5)])
        fixed = select_frames(rows, cadence="fixed")
        self.assertEqual([(row["time"], row["vote_time"]) for row in fixed], [(0, .25), (.25, .5)])


class AppearanceTests(unittest.TestCase):
    def test_local_and_global_routes_can_calibrate_separate_gates(self):
        from annotation.league.body_appearance import calibrate

        raw, labels = [], []
        for team in ("friendly", "enemy"):
            for kind, local, similarity, margin in (("local", True, .65, .10),
                                                    ("fallback", False, .95, .25),
                                                    ("negative", False, .70, .15)):
                for time in (.25, .5, .75):
                    row = sample(time, track=f"{team}-{kind}", team=team, vote_time=time)
                    raw.append((row, {"champion": "C0", "used_local_features": local,
                                      "score": similarity, "margin": margin}))
                labels.append({"id": row["id"], "team": team, "visibility": "readable",
                               "champion": None if kind == "negative" else "C0"})
        policy, metrics = calibrate(raw, labels, {"patch_weight": .25})
        self.assertLess(policy["score"], policy["global_score"])
        for team in ("friendly", "enemy"):
            self.assertEqual((metrics[team]["precision"], metrics[team]["coverage"]), (1, 1))

    def test_preserve_proportions_and_round_trip_geometry(self):
        image = Image.new("RGB", (40, 100), "red")
        mask = Image.new("L", image.size)
        ImageDraw.Draw(mask).rectangle((10, 10, 29, 89), fill=255)
        body, foreground, geometry = normalize_crop(image, mask)
        x, y, right, bottom = foreground.getbbox()
        self.assertEqual(body.size, (224, 224))
        self.assertAlmostEqual((right - x) / (bottom - y), .25, delta=.01)
        self.assertEqual([x, y], geometry["offset"])
        self.assertEqual(geometry["source_box"], [10, 10, 30, 90])
        variant, variant_mask = render_variant(image, mask, body_height=40, jpeg_quality=70)
        self.assertEqual(variant.size, variant_mask.size)
        self.assertEqual(variant_mask.getbbox()[3] - variant_mask.getbbox()[1], 40)

    def test_patch_support_requires_multiple_correspondences(self):
        good = [[int(i == j) for j in range(4)] for i in range(4)]
        score, support, pairs = patch_similarity(good, good)
        self.assertEqual((score, support, len(pairs)), (1, 1, 4))
        self.assertLess(patch_similarity(good, [good[0]] * 4)[1], 1)

    def test_augmented_copies_cannot_satisfy_reference_agreement(self):
        entry = {"id": "model", "champion": "C0", "skin_id": "base", "reference_kind": "rendered",
                 "sha256": "b" * 64, "reviewer": "source reviewer", "reviewer_kind": "human",
                 "verified": True, "identity_evidence": {"asset": "checked"}}
        entries = [entry, {**entry, "id": "blur", "parent_template_id": "model"},
                   {**entry, "id": "dim", "parent_template_id": "model"}]
        features = [{"rgb": [1, 0]}] * len(entries)
        gallery = FeatureGallery(entries, features, encoder_signature="encoder")
        result = gallery.match(features[0], {"development_freeze": "pin", "agreement": 3, "score": .9, "margin": .1}, ["C0"])
        self.assertIsNone(result["champion"])


class CampaignEntryPointTests(unittest.TestCase):
    def test_frame_matching_reads_compatible_appearance_locks_and_reuses_embeddings(self):
        from types import SimpleNamespace

        import numpy as np

        from annotation.league.body_frame_matching import SkinMatcher, match_frame, skin_lock_scope
        from annotation.league.body_presence import scan
        from annotation.league.body_operational import identity_inputs
        from annotation.league.core import read_json, read_rows
        from annotation.league.intake_profile import load_profile

        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            cache = directory / "cache"
            references = cache / "inventory-references/16.22.1"
            references.mkdir(parents=True)
            icon = references / "icon.png"
            Image.new("RGB", (16, 16), "blue").save(icon)
            assets = [{"kind": kind, "id": key, "name": key, "artwork": key,
                       "path": "icon.png", "sha256": file_hash(icon)}
                      for kind, ids in (("champion", CHAMPIONS), ("item", {"item"})) for key in sorted(ids)]
            atomic_json(references / "manifest.json", {"schema_version": 2, "reference_version": "16.22.1", "assets": assets})
            gallery_path = directory / "gallery.json"
            mask = directory / "render-mask.png"
            Image.new("L", (20, 20), 255).save(mask)
            gallery, vectors = [], {}
            for champion in range(1, 10):
                for skin in range(2 if champion in {1, 2} else 1):
                    key = f"C{champion}-{skin}"
                    color = (champion * 20, skin * 200, 0)
                    vector = np.eye(9, dtype=np.float32)[champion - 1]
                    if skin:
                        vector = vector * .8 + np.eye(9, dtype=np.float32)[8] * .6
                    vectors[color] = vector
                    path = directory / (key + ".png")
                    Image.new("RGB", (20, 20), color).save(path)
                    gallery.append({"id": key, "champion": f"C{champion}", "skin_id": key, "reference_kind": "rendered",
                                    "reviewer": "fixture", "reviewer_kind": "human", "verified": True,
                                    "identity_evidence": {"render_audit": "fixture"}, "path": str(path),
                                    "sha256": file_hash(path), "mask_path": str(mask), "mask_sha256": file_hash(mask)})
            atomic_json(gallery_path, gallery)

            def stage(game, frames_colors):
                root, intake = directory / game, directory / (game + "-intake")
                root.mkdir()
                video = root / "source.mp4"
                video.write_text(game)
                source = file_hash(video)
                profile = load_profile("midbeast-1080p-inventory-v2")
                atomic_json(intake / "manifest.json", {"source": {"sha256": source}})
                atomic_json(intake / "layout_profiles/profile.json", profile)
                write_rows(intake / "observations.jsonl", scoreboard(0, reviewed=True, source=source, profile=digest(profile))
                           + scoreboard(1, state="absent", source=source, profile=digest(profile)))
                frames, tracked_frames = [], []
                teams = ("friendly", "friendly", "friendly", "enemy")
                for index, colors in enumerate(frames_colors):
                    image = Image.new("RGB", (1920, 1080))
                    frame_path = root / f"frame-{index}.png"
                    masks = []
                    for slot, color in enumerate(colors):
                        if color is None:
                            continue
                        box = (350 + 180 * slot, 350, 399 + 180 * slot, 409)
                        ImageDraw.Draw(image).rectangle(box, fill=color)
                        mask_path = root / f"mask-{slot}.png"
                        foreground = Image.new("L", image.size)
                        ImageDraw.Draw(foreground).rectangle(box, fill=255)
                        foreground.save(mask_path)
                        masks.append({"object_id": slot + 1, "mask_path": str(mask_path), "sha256": file_hash(mask_path)})
                    image.save(frame_path)
                    frames.append({"id": f"native-{index}", "time": index / 4, "path": str(frame_path), "sha256": file_hash(frame_path)})
                    tracked_frames.append({"frame_index": index, "masks": masks})
                clip = {"id": "clip", "source_path": str(video), "source_sha256": source, "split": "development",
                        "match_group": game, "start": 0, "end": 1, "continuity_source_reviewed": True}
                campaign = {"pilot_champions": sorted(CHAMPIONS - {"Other"}), "clips": [clip], "gallery": [], "labels": []}
                atomic_json(root / "campaign.json", campaign)
                encoder_lock = {"code_root": str(root), "weights": str(video)}
                atomic_json(root / "encoder-lock.json", encoder_lock)
                atomic_json(root / "prepared.json", {
                    "recipe": {"campaign_hash": digest(campaign), "encoder_lock_hash": digest(encoder_lock)}, "gallery": [],
                    "clips": [{**clip, "jobs": ["job"], "frames": frames}],
                    "jobs": [{"id": "job", "frames": frames, "proposals": [{"team": team} for team in teams]}],
                })
                atomic_json(root / "tracks/job-complete.json", {"frames": tracked_frames})
                return root, intake

            def run(root, intake, time, **options):
                options.setdefault("encoder", "dinov2")
                receipt = match_frame(root, intake, cache, frame_time=time, main_champion="C0",
                                      main_track="job-1", render_gallery=gallery_path, **options)
                return read_json(Path(receipt["path"]))

            main = (255, 255, 255)
            first = stage("first-game", [(main, (20, 0, 0), None, (100, 0, 0)),
                                         (main, (40, 200, 0), (20, 200, 0), (100, 0, 0))])
            second = stage("second-game", [(main, (20, 200, 0), None, None)])
            encoder = SimpleNamespace(signature="fixture", describe=lambda image: (vectors[image.getpixel((112, 112))], None))
            with patch("annotation.league.body_frame_matching.FrozenEncoder", return_value=encoder):
                cold = run(*first, 0)
                warm = run(*first, .25)
                other_game = run(*second, 0)
                prepared_clip = read_json(first[0] / "prepared.json")["clips"][0]
                main_tracks = directory / "main-tracks.json"
                atomic_json(main_tracks, {"source_sha256": prepared_clip["source_sha256"], "frames": {
                    frame["id"]: {"frame_sha256": frame["sha256"], "track_id": "job-1"}
                    for frame in prepared_clip["frames"]}})
                output = directory / "presence-pilot"
                args = {"clip_ids": ["clip"], "main_champion": "C0", "render_gallery": gallery_path, "main_tracks": main_tracks}
                self.assertEqual(scan(*first, cache, output, **args, check_inputs=True)["sampled_frames"], 2)
                self.assertEqual(scan(*first, cache, output, **args)["reused_assignments"], 2)
                presence = read_rows(output / "presence_observations.jsonl")
                self.assertEqual(len(presence), 18)
                self.assertEqual({row["status"] for row in presence}, {"candidate_visible", "unknown"})
                changes = read_rows(output / "presence_change_candidates.jsonl")
                self.assertEqual([(row["champion"], row["before"], row["after"]) for row in changes],
                                 [("C2", "unknown", "candidate_visible")])

                siglip_lock = directory / "siglip-lock.json"
                atomic_json(siglip_lock, {"model_id": "fixture"})
                siglip = SimpleNamespace(signature=encoder.signature, describe_batch=lambda images: [
                    encoder.describe(image)[0] for image in images])
                with patch("annotation.league.body_frame_matching.SiglipEncoder", return_value=siglip):
                    siglip_cold = run(*first, 0, encoder="siglip", siglip_lock=siglip_lock)
                    _, _, _, source, _, _, _, rosters = identity_inputs(*first, cache, None)
                    lock = directory / "appearance-locks.json"
                    matcher = SkinMatcher(gallery, cache, siglip_lock, digest(read_json(siglip_lock)),
                                          ["16.22.1", "16.22"], encoder_name="siglip")
                    scope = skin_lock_scope(matcher, cold["team_pools"], source, rosters.at(0)["id"])
                    atomic_json(lock, {"scope": scope, "assignments": {"friendly": {"C1": {"skin_id": "C1-0"}}, "enemy": {}}})
                    siglip_warm = run(*first, .25, encoder="siglip", siglip_lock=siglip_lock, skin_assignments=lock)
                    with self.assertRaisesRegex(ValueError, "different match"):
                        run(*second, 0, encoder="siglip", siglip_lock=siglip_lock, skin_assignments=lock)

                atomic_json(gallery_path, [entry for entry in gallery if entry["id"] != "C1-0"])
                missing_locked_skin = run(*first, .25)

            self.assertEqual(cold["excluded_main_track"], "job-1")
            self.assertEqual(cold["counts"]["render_comparisons"], 11)
            self.assertEqual([(r["champion"], r["skin_id"]) for r in cold["predictions"]], [("C1", "C1-0"), ("C5", "C5-0")])
            self.assertEqual([(r["champion"], r["skin_id"]) for r in warm["predictions"]],
                             [("C2", "C2-1"), ("C1", "C1-1"), ("C5", "C5-0")])
            self.assertEqual(warm["counts"]["reused_assignments"], 0)
            self.assertEqual(warm["counts"]["render_comparisons"], 17)
            self.assertEqual(other_game["predictions"][0]["skin_id"], "C1-1")
            self.assertEqual(other_game["counts"]["render_encodes"], 0)
            self.assertEqual(siglip_cold["encoder"], "siglip")
            self.assertIsNone(siglip_cold["skin_assignment_cache"])
            self.assertEqual(siglip_cold["counts"]["new_assignments"], 0)
            self.assertGreater(siglip_cold["counts"]["render_encodes"], 0)
            self.assertEqual(siglip_warm["counts"]["render_encodes"], 0)
            self.assertEqual(siglip_warm["counts"]["reused_assignments"], 1)
            self.assertEqual(siglip_warm["predictions"][1]["skin_id"], "C1-0")
            self.assertEqual(missing_locked_skin["predictions"][0]["missing_champion_renders"], [])

    def test_presence_transitions_need_reviewed_visibility_on_both_sides_of_the_boundary(self):
        from annotation.league.body_presence import reviewed_transitions
        from annotation.league.timeline import fuse_events

        samples = [
            (0, "clip", "not_visible"), (.25, "clip", "visible"), (.5, "clip", None),
            (.75, "clip", "not_visible"), (1, "clip", "visible"), (1.25, "clip", "not_visible"),
            (1.5, "next-clip", "visible"), (2, "next-clip", "not_visible"),
        ]
        observations, decisions = [], []
        for index, (time, clip, status) in enumerate(samples):
            row = {"id": f"o{index}", "source_sha256": SOURCE, "frame_id": f"f{index}", "frame_sha256": f"sha{index}",
                   "clip_id": clip, "roster_id": "roster", "champion": "C2", "team": "friendly",
                   "status": "candidate_visible", "time": time}
            observations.append(row)
            if status:
                decisions.append({"observation_id": row["id"], "frame_sha256": row["frame_sha256"], "status": status})
        review = {"source_sha256": SOURCE, "reviewer": "fixture", "observations": decisions}
        events = reviewed_transitions(observations, review, SOURCE)
        self.assertEqual([(e["kind"], e["start"], e["end"]) for e in events], [
            ("champion_appeared", 0, .25), ("champion_appeared", .75, 1), ("champion_disappeared", 1, 1.25)])
        self.assertEqual(events[-1]["evidence_ids"], ["f4", "f5"])
        self.assertEqual(fuse_events(events), [])

    def test_existing_intake_and_tracks_resolve_without_a_model_and_rebuild_after_correction(self):
        from annotation.league.body_operational import identify
        from annotation.league.intake_profile import load_profile
        from annotation.league.core import read_json

        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            root, intake, cache = (directory / part for part in ("body", "intake", "cache"))
            root.mkdir()
            video = directory / "source.mp4"
            video.write_bytes(b"prepared source fixture")
            source = file_hash(video)
            profile = load_profile("midbeast-1080p-inventory-v2")
            atomic_json(intake / "manifest.json", {"source": {"sha256": source}})
            atomic_json(intake / "layout_profiles/profile.json", profile)
            observations = scoreboard(0, reviewed=True, named=True, source=source, profile=digest(profile))
            write_rows(intake / "observations.jsonl", observations)
            references = cache / "inventory-references/16.22.1"
            references.mkdir(parents=True)
            icon = references / "icon.png"
            Image.new("RGB", (16, 16), "blue").save(icon)
            assets = [{"kind": kind, "id": identifier, "name": identifier, "artwork": identifier,
                       "path": "icon.png", "sha256": file_hash(icon)}
                      for kind, ids in (("champion", CHAMPIONS), ("item", {"item"})) for identifier in sorted(ids)]
            atomic_json(references / "manifest.json", {"schema_version": 2, "reference_version": "16.22.1", "assets": assets})
            frame_path, mask_path = root / "frame.png", root / "mask.png"
            Image.new("RGB", (1920, 1080), "purple").save(frame_path)
            mask = Image.new("L", (1920, 1080))
            ImageDraw.Draw(mask).rectangle((700, 350, 745, 420), fill=255)
            mask.save(mask_path)
            frame = {"id": "native", "path": str(frame_path), "time": 0, "sha256": file_hash(frame_path)}
            clip = {"id": "clip", "source_path": str(video), "source_sha256": source, "split": "development",
                    "match_group": "match", "start": 0, "end": 1, "continuity_source_reviewed": True}
            campaign = {"pilot_champions": sorted(CHAMPIONS - {"Other"}), "clips": [clip], "gallery": [], "labels": []}
            atomic_json(root / "campaign.json", campaign)
            atomic_json(root / "prepared.json", {
                "recipe": {"campaign_hash": digest(campaign), "encoder_lock_hash": None}, "gallery": [],
                "clips": [{**clip, "jobs": ["job"]}],
                "jobs": [{"id": "job", "frames": [frame], "proposals": [{"team": "friendly"}]}],
            })
            atomic_json(root / "tracks/job-complete.json", {
                "frames": [{"frame_index": 0, "masks": [{"object_id": 1, "mask_path": str(mask_path), "sha256": file_hash(mask_path)}]}]
            })
            identifier = digest(["native", "job-1"])[:20]
            write_rows(root / "body_names.jsonl", [{
                **name({"time": 0, "source_sha256": source, "frame_id": "native", "track_id": "job-1", "id": identifier}),
                "id": identifier, "review_status": "human-verified",
            }])
            with patch("annotation.league.body_identity.FrozenEncoder", side_effect=AssertionError("encoder loaded")):
                first = identify(root, intake, cache)
                result = read_json(Path(first["path"]))
                self.assertEqual(result["predictions"][0]["champion"], "C0")
                self.assertEqual(first["counts"]["encoder_calls"], 0)
                self.assertEqual(first["counts"]["gallery_calls"], 0)
                atomic_json(intake / "corrections/00000001.json", {
                    "target": "observations", "record_id": observations[0]["id"],
                    "changes": {}, "review_status": "rejected",
                })
                second = identify(root, intake, cache)
            self.assertNotEqual(first["path"], second["path"])
            self.assertEqual(second["rosters"], 0)
            self.assertIsNone(read_json(Path(second["path"]))["predictions"][0]["champion"])
            self.assertTrue(Path(first["path"]).exists())

    def test_budget_reconciliation_deduplicates_allocations_and_preserves_failed_charges(self):
        from annotation.league.recognition_budget import reconcile, reserve, finish

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            submission = {"submitted_at": "2026-11-10", "gpu_limit_seconds": 900,
                          "jobs": [{"stage": "compare", "job_id": "1234"}]}
            self.assertEqual(reconcile(root, [submission, submission])["reserved_seconds"], 900)
            with patch.dict("os.environ", {"SLURM_JOB_ID": "1234"}):
                allocation = reserve(root, "compare", 900)
                finish(root, allocation["attempt"], 13, "failed")
                with self.assertRaises(ValueError):
                    reserve(root, "compare", 900)
            self.assertEqual(reconcile(root, [submission], [{"job_id": "1234", "state": "FAILED"}])["reserved_seconds"], 900)
