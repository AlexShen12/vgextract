"""SAM pilot seeds, HUD reuse, independent masks and pending result semantics."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from annotation.league.core import Config, Store, atomic_json
from annotation.league.sam_tracking import choose_seed, collect, portrait_rows, validate_labels, hud_revision, check_seed_mask, bundle
from annotation.league.sam_tracking_review import create_app


class SAMTrackingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.store = Store(self.root)
        self.frame = {"id": "f1", "time": 1.0}
        self.clip = {"seed": {"time": 1.0, "point": [50, 60], "labeler": "source-only"}}

    def test_both_arms_use_same_source_frame_but_automatic_needs_binding(self):
        manual, reason = choose_seed(self.clip, [self.frame], [], "verified")
        self.assertEqual((manual["point"], reason), ([50, 60], None))
        auto, reason = choose_seed(self.clip, [self.frame], [], "automatic")
        self.assertIsNone(auto)
        self.assertEqual(reason, "automatic_body_seed_unavailable")
        body = [{"id": "b", "frame_id": "f1", "binding_visible": True, "actor_track": "track",
                 "actor_region": [10, 20, 50, 80]}]
        auto, reason = choose_seed(self.clip, [self.frame], body, "automatic")
        self.assertEqual(auto["frame_id"], manual["frame_id"])
        self.assertEqual(auto["point"], [30, 50])
        body[0]["review_status"] = "rejected"
        self.assertIsNone(choose_seed(self.clip, [self.frame], body, "automatic")[0])

    def test_missing_seed_and_misaligned_frame_remain_unresolved(self):
        self.assertEqual(choose_seed({}, [self.frame], [], "verified")[1], "no_independent_seed")
        self.assertEqual(choose_seed(self.clip, [{"id": "wrong", "time": 2}], [], "verified")[1], "seed_frame_missing")

    def test_source_only_labels_and_native_mask_geometry_are_required(self):
        from PIL import Image
        Image.new("L", (8, 8), 255).save(self.root / "mask.png")
        clips = [{"id": f"event-development-{i:02d}", "start": 0, "end": 12, "split": "development",
                  "seed": None, "references": [{"time": 1, "visibility": "visible", "mask_path": "mask.png", "labeler": "independent"}]}
                 for i in (0, 1, 2, 3, 6)]
        data = {"source_sha256": "source", "source_only": True, "clips": clips}
        source = {"sha256": "source", "width": 8, "height": 8}
        self.assertEqual(len(validate_labels(data, source, self.root)), 5)
        with self.assertRaisesRegex(ValueError, "geometry"):
            validate_labels(data, {**source, "width": 16}, self.root)
        data["source_only"] = False
        with self.assertRaisesRegex(ValueError, "source-only"):
            validate_labels(data, source, self.root)

    def test_collect_does_not_score_unsubmitted_jobs_as_zero_quality(self):
        path = self.root / "prepared.json"
        atomic_json(path, {"root": str(self.root), "clips": []})
        summary = collect(path)
        self.assertEqual(summary["arms"]["verified"], {"status": "pending", "metrics": None})

    def test_hud_adapter_uses_existing_inventory_reader_and_honors_rejected_correction(self):
        from PIL import Image
        path = self.root / "native.png"
        Image.new("RGB", (20, 20), "red").save(path)
        frame = {**self.frame, "native_path": str(path)}
        field = {"route": "selected_hud", "field": "inventory.champion", "box": [1, 1, 10, 10]}
        layout = {"fields": [field], "inventory": {"reference_version": "16.13.1"}}
        cfg = Config(routes=[], patch={}, mechanics=False, evaluation_enabled=False)
        cfg.layout_profile = "fixture"
        fake = MagicMock(signature="reader", catalog_id="catalog")
        fake.read.return_value = ({"acceptance": "accepted", "value": "Darius"}, True)
        prior = {"id": "portrait", "field": "inventory.champion", "route": "selected_hud", "box": field["box"],
                 "time": 1.0, "inventory_catalog_id": "catalog", "review_status": "rejected", "value": "Darius"}
        with patch("annotation.league.sam_tracking.load_layout", return_value=layout), \
             patch("annotation.league.sam_tracking.load_inventory_references", return_value={}), \
             patch("annotation.league.sam_tracking.InventoryReader", return_value=fake), \
             patch("annotation.league.sam_tracking.layout_matches", return_value=(True, [])), \
             patch("annotation.league.sam_tracking._panel_state", return_value="present"), \
             patch.object(self.store, "rows", return_value=[prior]), \
             patch.object(self.store, "effective", return_value=[prior]):
            output = portrait_rows(self.store, cfg, [frame])
        self.assertEqual(fake.read.call_count, 1)
        self.assertIsNone(output[0]["hud_champion"])
        self.assertEqual(output[0]["prior_observation_id"], "portrait")

    def test_hud_revision_changes_with_corrections(self):
        with patch.object(self.store, "corrections", return_value=[]):
            first = hud_revision(self.store)
        with patch.object(self.store, "corrections", return_value=[{"target": "observations", "record_id": "portrait", "value": "Unknown"}]):
            self.assertNotEqual(first, hud_revision(self.store))

    def test_preflight_requires_localized_mask_containing_source_seed(self):
        from PIL import Image
        path = self.root / "mask.png"
        selected = {"width": 20, "height": 20, "prompts": [{"frame_index": 0, "point": [10, 10]}]}
        result = {"frames": [{"frame_index": 0, "masks": [{"object_id": 1, "mask_path": str(path)}]}]}
        Image.new("L", (20, 20), 255).save(path)
        with self.assertRaisesRegex(ValueError, "localized"):
            check_seed_mask(result, selected)
        mask = Image.new("L", (20, 20))
        mask.putpixel((10, 10), 255)
        mask.save(path)
        self.assertTrue(check_seed_mask(result, selected)["contains_seed"])

    def test_review_restricts_asset_paths_and_handles_pending_masks(self):
        from fastapi.testclient import TestClient
        path = self.root / "prepared.json"
        atomic_json(path, {"root": str(self.root), "source": {"path": str(self.root / "source.mp4")},
                    "clips": [{"id": "clip", "start": 0, "end": 12, "arms": {
                        "verified": {"status": "no_seed"}, "automatic": {"status": "no_seed"}}}]})
        client = TestClient(create_app(path))
        self.assertEqual(client.get("/asset/unknown").status_code, 404)
        self.assertEqual(client.get("/data").json()["clips"][0]["arms"]["verified"]["run_status"], "pending")
        self.assertIn("SAM tracking", client.get("/").text)

    def test_bundle_reuses_existing_importer_and_projects_label_paths(self):
        from annotation.league.caption_batch import import_bundle
        from annotation.league.media import file_hash
        source = self.root / "source.mp4"
        source.write_bytes(b"exact pinned source fixture")
        manifest = {"source": {"path": str(source), "sha256": file_hash(source)}}
        atomic_json(self.root / "manifest.json", manifest)
        root = self.root / "experiments/sam-tracking/fixture"
        labels = root / "source-labels"
        atomic_json(labels / "labels.json", {"source_only": True})
        prepared = root / "prepared.json"
        atomic_json(prepared, {"root": str(root), "source": manifest["source"], "labels_dir": str(labels), "clips": []})
        archive = self.root / "results.tar.gz"
        bundle(prepared, archive)
        destination = self.root / "imported"
        import_bundle(archive, destination, source)
        from annotation.league.core import read_json
        imported = read_json(destination / prepared.relative_to(self.root))
        self.assertTrue((Path(imported["labels_dir"]) / "labels.json").is_file())


if __name__ == "__main__":
    unittest.main()
