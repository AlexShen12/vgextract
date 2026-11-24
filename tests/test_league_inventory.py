"""Inventory matching, release scoring, and the correction-aware review timeline."""

import base64
from copy import deepcopy
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from annotation.league.core import Config, Store, atomic_json
from annotation.league.hybrid_intake import enabled_fields
from annotation.league.intake_benchmark import inventory_label_coverage, score
from annotation.league.intake_profile import load_profile
from annotation.league.inventory_reader import CARDINAL_SHIFTS, InventoryReader
from annotation.league.inventory_references import load_inventory_references
from annotation.league.inventory_timeline import inventory_timeline
from annotation.league.review import add_correction, create_app
from annotation.league.timeline import fuse

from league_fixtures import Image


PROFILE = "midbeast-1080p-inventory-v1"
ITEM_FIELD = {"route": "scoreboard", "field": "inventory.0", "kind": "item"}


def png(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@unittest.skipUnless(Image, "Install requirements-intake.txt")
class InventoryReaderTests(unittest.TestCase):
    def setUp(self):
        from PIL import ImageDraw

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name) / "catalog"
        self.root.mkdir()
        self.store = Store(Path(directory.name) / "run")
        self.store.root.mkdir()
        self.images = {}
        assets = []
        for identifier, kind, foreground, shape, slot_class in (
            ("10", "item", "lime", (3, 4, 22, 40), None),
            ("11", "item", "gold", (23, 4, 41, 28), None),
            ("12", "item", "aqua", (3, 18, 40, 35), "trinket"),
            ("13", "item", (255, 204, 42), (12, 12, 33, 33), "trinket"),
            ("Darius", "champion", "fuchsia", (8, 12, 41, 38), None),
            ("Talon", "champion", "navy", (4, 8, 25, 21), None),
        ):
            image = Image.new("RGB", (48, 48), (19, 21, 46))
            ImageDraw.Draw(image).rectangle(shape, fill=foreground)
            self.images[identifier] = image
            assets.append(self.reference(image, identifier, kind, slot_class))
        self.manifest = {"schema_version": 2, "reference_version": "16.13.1", "assets": assets}
        self.settings = deepcopy(load_profile(PROFILE)["inventory"])
        self.settings.pop("selected_portrait")
        self.settings["hidden_portraits"] = [base64.b64encode(png(self.images["13"])).decode()]
        self.reader = self.make_reader()

    def reference(self, image, identifier, kind, slot_class=None):
        raw = png(image)
        sha = hashlib.sha256(raw).hexdigest()
        (self.root / f"{sha}.png").write_bytes(raw)
        return {"id": identifier, "kind": kind, "name": identifier, "description": "Reference description",
                "artwork": sha, "path": f"{sha}.png", "sha256": sha, "slot_class": slot_class}

    def make_reader(self):
        return InventoryReader(self.store, self.root, self.manifest, self.settings)

    def test_visibly_empty_dark_and_stacked_items(self):
        from PIL import ImageDraw, ImageEnhance

        empty = Image.new("RGB", (32, 32), (6, 14, 13))
        obscured = Image.new("RGB", (32, 32), "black")
        self.assertEqual(self.reader.match(empty, ITEM_FIELD)["acceptance"], "absent")
        self.assertEqual(self.reader.match(obscured, ITEM_FIELD)["acceptance"], "unknown")
        darkened = ImageEnhance.Brightness(self.images["10"]).enhance(.48)
        ImageDraw.Draw(darkened).text((35, 32), "2", fill="white")
        self.assertEqual(self.reader.match(darkened, ITEM_FIELD)["value"], "10")
        trinket = {**ITEM_FIELD, "field": "inventory.trinket"}
        self.assertEqual(self.reader.match(self.images["12"], trinket)["value"], "12")
        self.assertEqual({c["id"] for c in self.reader.match(self.images["12"], trinket)["candidates"]}, {"12", "13"})

    def test_hidden_portrait_and_identical_item_art_abstain(self):
        field = {**ITEM_FIELD, "field": "inventory.champion", "kind": "champion"}
        self.assertIn("hides", self.reader.match(self.images["13"], field)["reason"])
        self.assertEqual(self.reader.match(self.images["Darius"], field)["value"], "Darius")
        self.manifest["assets"].append(self.reference(self.images["10"], "99", "item"))
        result = self.make_reader().match(self.images["10"], ITEM_FIELD)
        self.assertEqual(result["acceptance"], "unknown")
        self.assertIn("same reference art", result["reason"])

    def test_selected_top_match_accepts_a_small_margin_and_cache_is_surface_specific(self):
        selected = {**ITEM_FIELD, "route": "selected_hud"}
        crop = self.images["10"]
        match = {"acceptance": "accepted", "value": "10", "reader": {"score": .72, "margin": .01},
                 "candidates": [{"id": "10", "score": .72}]}
        with patch.object(self.reader.matcher, "read", return_value=match) as mocked:
            result, hit = self.reader.read(crop, selected, "same-crop")
            self.assertFalse(hit)
            self.assertEqual(result["value"], "10")
            self.assertTrue(self.reader.read(crop, selected, "same-crop")[1])
            self.assertEqual(self.reader.read(crop, ITEM_FIELD, "same-crop")[0]["acceptance"], "unknown")
            self.assertEqual(mocked.call_count, 2)
        for score, margin in ((.67, .12), (.75, 0)):
            with self.subTest(score=score, margin=margin), patch.object(
                self.reader.matcher, "read", return_value={**match, "reader": {"score": score, "margin": margin}}
            ):
                self.assertEqual(self.reader.match(crop, selected)["acceptance"], "unknown")

    def test_respawn_portrait_matches_grayscale_references_with_red_digits_masked(self):
        from PIL import ImageDraw, ImageOps

        crop = ImageOps.grayscale(self.images["Darius"]).convert("RGB")
        ImageDraw.Draw(crop).rectangle((21, 18, 25, 32), fill=(233, 12, 25))
        field = {**ITEM_FIELD, "field": "inventory.champion", "kind": "champion"}
        result = self.reader.match(crop, field)
        self.assertEqual(result["value"], "Darius")
        self.assertEqual(result["reader"]["view"], "grayscale_red_mask")
        self.assertNotIn("view", self.reader.match(self.images["Darius"], field)["reader"])

    def test_scoreboard_alignment_uses_agreement_before_score(self):
        direct = {"acceptance": "unknown", "candidates": [{"id": "11", "score": .65}]}
        shifts = [(offset, self.images["10"], str(offset)) for offset in CARDINAL_SHIFTS]
        missed = {"acceptance": "unknown"}

        def accepted(item, score):
            return {"acceptance": "accepted", "value": item, "reader": {"score": score}}

        for matches, expected, expected_offset, votes in (
            ([accepted("10", .81), accepted("10", .85), accepted("11", .96), missed], "10", (1, 0), 2),
            ([accepted("10", .84), missed, missed, accepted("11", .93)], "11", (0, 1), 1),
        ):
            with self.subTest(item=expected), patch.object(
                self.reader, "read", side_effect=[(match, False) for match in matches]
            ):
                result = self.reader.retry_alignment(direct, ITEM_FIELD, shifts)
                self.assertEqual(result["value"], expected)
                self.assertEqual(result["reader"]["pixel_shift"], list(expected_offset))
                self.assertEqual(result["reader"]["shift_votes"], votes)

        with patch.object(self.reader, "read", side_effect=[(accepted("10", .90), False), (accepted("11", .90), False)]):
            self.assertIs(self.reader.retry_alignment(direct, ITEM_FIELD, shifts[:2]), direct)
        with patch.object(self.reader, "read") as read:
            faint = {"acceptance": "unknown", "candidates": [{"id": "10", "score": .45}]}
            self.assertIs(self.reader.retry_alignment(faint, ITEM_FIELD, shifts), faint)
            self.assertIs(self.reader.retry_alignment(direct, {**ITEM_FIELD, "route": "selected_hud"}, shifts), direct)
            read.assert_not_called()

    def test_portrait_alignment_is_profile_specific_and_keeps_hidden_portraits_unknown(self):
        field = {**ITEM_FIELD, "field": "inventory.champion", "kind": "champion"}
        direct = {"acceptance": "unknown", "candidates": [{"id": "Talon", "score": .67}]}
        passing = {"acceptance": "accepted", "value": "Darius", "reader": {"score": .79}}
        shifts = [((1, 0), self.images["Darius"], "right")]
        with patch.object(self.reader, "read", return_value=(passing, False)) as read:
            self.assertIs(self.reader.retry_alignment(direct, field, shifts), direct)
            read.assert_not_called()
            self.reader.settings["scoreboard_portrait_alignment"] = True
            hidden = self.reader.match(self.images["13"], field)
            self.assertIs(self.reader.retry_alignment(hidden, field, shifts), hidden)
            read.assert_not_called()
            result = self.reader.retry_alignment(direct, field, shifts)
            self.assertEqual(result["value"], "Darius")
            self.assertEqual(result["reader"]["pixel_shift"], [1, 0])
            self.assertEqual(result["reader"]["shift_votes"], 1)

    def test_respawn_alignment_keeps_the_respawn_view(self):
        field = {**ITEM_FIELD, "field": "inventory.champion", "kind": "champion"}
        self.reader.settings["scoreboard_portrait_alignment"] = True
        reading = {"acceptance": "unknown", "candidates": [{"id": "Darius", "score": .61}],
                   "reader": {"view": "grayscale_red_mask"}}
        passing = {"acceptance": "accepted", "value": "Darius", "reader": {"score": .79}}
        shifts = [((1, 0), self.images["Darius"], "right")]
        with patch.object(self.reader, "read", return_value=(passing, False)) as read:
            result = self.reader.retry_alignment(reading, field, shifts)
            self.assertEqual(result["acceptance"], "unknown")
            self.assertFalse(result["reader"]["alignment_views"][1]["eligible"])
            read.assert_called_once()

    def test_center_is_one_of_five_votes_and_raw_top_ids_are_saved(self):
        field = {**ITEM_FIELD, "field": "inventory.champion", "kind": "champion"}
        self.reader.settings["scoreboard_portrait_alignment"] = True
        def result(champion, score, passing=True):
            return {"acceptance": "accepted" if passing else "unknown", "value": champion if passing else None,
                    "reader": {"score": score, "margin": .1 if passing else .02},
                    "candidates": [{"id": champion, "score": score}]}
        center = result("Talon", .78)
        readings = [result("Darius", .87), result("Darius", .86), result("Talon", .8, False), result("Talon", .79, False)]
        shifts = [(offset, self.images["Darius"], str(offset)) for offset in CARDINAL_SHIFTS]
        with patch.object(self.reader, "read", side_effect=[(reading, False) for reading in readings]):
            actual = self.reader.retry_alignment(center, field, shifts)
        self.assertEqual(actual["value"], "Darius")
        self.assertEqual(actual["reader"]["shift_votes"], 2)
        self.assertEqual([row["top_id"] for row in actual["reader"]["alignment_views"]],
                         ["Talon", "Darius", "Darius", "Talon", "Talon"])

    def test_catalog_verifies_reference_file_integrity(self):
        atomic_json(self.root / "manifest.json", self.manifest)
        self.assertEqual(load_inventory_references(self.root, "16.13.1"), self.manifest)
        (self.root / self.manifest["assets"][0]["path"]).write_bytes(b"unexpected change")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            load_inventory_references(self.root, "16.13.1")


class InventoryTimelineTests(unittest.TestCase):
    @staticmethod
    def row(time, owner, field, value=None, *, status=None, panel="present", segment="first"):
        status = status or ("accepted" if value is not None else "unknown")
        return {"schema_version": 2, "id": f"{time}-{owner}-{field}", "time": time, "sampling_fps": 1,
                "frame_id": f"{owner}-{field}-crop", "evidence_path": "source.png", "task_id": f"task-{time}",
                "route": "selected_hud" if owner == "selected" else "scoreboard", "slot_id": owner,
                "backend": "vl" if field == "player_name" else "inventory", "field": field, "entity": "unknown",
                "binding": owner, "value": value, "acceptance": status, "panel_state": panel,
                "playback_segment_id": segment, "inventory_catalog_id": "catalog",
                "visibility": {"accepted": "visible", "unknown": "unknown", "absent": "not_shown"}[status]}

    def test_rows_and_trinkets_stay_bound_and_last_seen_stops_at_cut(self):
        r = self.row
        observations = [r(1, "left:row1", "inventory.champion"), r(1, "left:row1", "inventory.0", "10"),
                        r(1, "left:row1", "inventory.1", status="absent"),
                        r(1, "right:row1", "inventory.champion", "Darius"), r(1, "right:row1", "inventory.0", "11"),
                        r(1, "right:row1", "inventory.trinket", "12"), r(1, "right:row1", "player_name", "Name"),
                        r(1, "selected", "inventory.champion", "Darius"), r(1, "selected", "inventory.0", "13"),
                        r(2, "right:row1", "inventory.0", panel="absent", status="absent"),
                        r(3, "right:row1", "inventory.0", panel="absent", status="absent", segment="cut")]
        snapshots, samples = inventory_timeline(observations)
        left, right, selected = snapshots
        self.assertIsNone(left["champion"])
        self.assertEqual([slot["status"] for slot in left["slots"][:3]], ["recognized", "empty", "unknown"])
        self.assertEqual((right["champion"], right["player_name"], right["slots"][-1]["item_id"]), ("Darius", "Name", "12"))
        self.assertEqual(selected["slots"][0]["item_id"], "13")
        self.assertEqual(samples[1]["last_seen_ids"], [right["id"]])
        self.assertEqual(samples[2]["last_seen_ids"], [])

    def test_duplicate_portraits_stay_unresolved(self):
        r = self.row
        snapshots, _ = inventory_timeline([r(1, f"right:row{i}", "inventory.champion", "Darius") for i in (1, 2)])
        self.assertEqual([row["champion"] for row in snapshots], [None, None])

    def test_review_follows_rebuilt_corrections(self):
        from fastapi.testclient import TestClient

        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory))
            config = Config(patch={}, routes=[], layout_profile=PROFILE)
            atomic_json(store.path("manifest.json"), {"schema_version": 2, "source": {"sha256": "source"}, "interval": [0, 5]})
            items = [{"id": identifier, "name": "Item " + identifier, "description": "Description", "path": identifier + ".png"}
                     for identifier in ("10", "11")]
            atomic_json(store.path("inventory-catalogs/catalog/manifest.json"),
                        {"reference_version": "16.13.1", "items": items, "champions": {"Darius": "Darius"}})
            item = self.row(1, "right:row1", "inventory.0", "10")
            portrait = self.row(1, "right:row1", "inventory.champion", "Darius")
            views = [{"offset": [0, 0], "top_id": "Darius", "score": .9, "margin": .2, "eligible": True}]
            portrait["reader"] = {"alignment_views": views}
            store.write("observations", [item, portrait,
                                         self.row(2, "right:row1", "inventory.0", panel="absent", status="absent")])
            fuse(store, config)
            with TestClient(create_app(store)) as client:
                data = client.get("/api/inventory?time=2").json()["snapshots"][0]
                self.assertTrue(data["last_seen"])
                self.assertEqual(data["time"], 1)
                self.assertEqual(data["slots"][0]["reference"]["name"], "Item 10")
                self.assertEqual(data["portrait_alignment_views"], views)
                add_correction(store, "observations", item["id"], {"value": "11"}, "human-verified")
                fuse(store, config)
                updated = client.get("/api/inventory?time=2").json()["snapshots"][0]
                self.assertEqual(updated["slots"][0]["reference"]["name"], "Item 11")
                self.assertEqual(client.get("/api/inventory?time=4").json()["status"], "unsampled")
                self.assertEqual(client.get("/inventory").status_code, 200)


class InventoryBenchmarkTests(unittest.TestCase):
    def test_reference_matching_scores_each_surface_and_rejects_missing_coverage(self):
        fields = {f["id"]: f for f in enabled_fields(load_profile(PROFILE))}
        item = "scoreboard/right:row1/inventory.0"
        hidden = "scoreboard/left:row1/inventory.champion"
        frames = [{"time": 4, "case_tags": ["inventory_scoreboard_open", "inventory_hidden_portrait"],
                   "labels": [{"field_id": item, "availability": "visible", "value": "11"},
                              {"field_id": hidden, "availability": "unreadable"}]}]
        predictions = {(4, item): {"value": "11", "acceptance": "accepted"},
                       (4, hidden): {"value": "Darius", "acceptance": "accepted"}}
        result = score(frames, predictions, fields)
        self.assertFalse(result["passed"])
        self.assertTrue(result["inventory"]["scoreboard/item"]["passed"])
        self.assertFalse(result["inventory"]["selected_hud/item"]["passed"])
        self.assertEqual(result["inventory"]["scoreboard/champion"]["false_on_unreadable"], 1)
        missing = inventory_label_coverage(frames, fields)
        self.assertIn("inventory_scoreboard_closed", missing)
        self.assertNotIn("inventory_hidden_portrait", missing)
