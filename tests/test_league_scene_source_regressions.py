"""Real source HUD pixels through OCR, transition discovery and saved action output.

The fixture is development source evidence reviewed by models. These tests check
known regressions; they do not estimate accuracy on independent footage.
"""
from pathlib import Path
import shutil
import tempfile
import unittest

from annotation.league.action_hud_replay import (associate_scene_casts, cooldown_candidates,
                                                 hud_activation_actions)
from annotation.league.action_events import read_recall, recall_events
from annotation.league.action_pipeline import scan_hud
from annotation.league.action_review import effective_actions
from annotation.league.core import Store, read_json, write_rows
from annotation.league.scene_reference import file_hash

FIXTURES = Path(__file__).parent/"fixtures/league_scene_understanding"


@unittest.skipUnless(shutil.which("tesseract"), "Source OCR regression requires Tesseract")
class SceneSourceRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.fixture = read_json(FIXTURES/"apprehend-hud.json")
        frames = []
        for row in cls.fixture["frames"]:
            path = FIXTURES/row["path"]
            if file_hash(path) != row["sha256"]:
                raise AssertionError("Source fixture pixels changed without review")
            frames.append({"id": row["id"], "time": row["time"], "native_path": str(path),
                           "layout_valid": True, "segment_id": "negative-01",
                           "owner_epochs": {"selected": 0}})
        cls.readings, _ = scan_hud(Store(cls.root), {
            "root": str(cls.root), "profile": cls.fixture["profile"], "frames": frames}, mode="core")
        cls.candidates = cooldown_candidates(cls.readings, cls.fixture["profile"]["thresholds"])

    def test_upper_cost_is_not_center_countdown_and_real_onset_is_retained(self):
        expected = self.fixture["expected"]
        self.assertEqual([row["slot_id"] for row in self.candidates], [expected["slot"]])
        self.assertEqual([self.candidates[0]["start"], self.candidates[0]["end"]], expected["onset_bracket"])
        e = [row for row in self.readings if row["slot_id"] == "E" and row["seconds"] is not None]
        self.assertEqual({row["seconds"] for row in e}, {expected["centered_cooldown"]})
        self.assertNotIn(expected["upper_cost"], [row["seconds"] for row in e])
        w = [row for row in self.readings if row["slot_id"] == "W"]
        self.assertTrue(all(row["seconds"] is None for row in w))

    def test_starting_inside_same_countdown_never_creates_another_activation(self):
        after = [row for row in self.readings if row["time"] >= self.candidates[0]["end"]]
        self.assertEqual(cooldown_candidates(after, self.fixture["profile"]["thresholds"]), [])

    def test_scene_abstention_preserves_hud_activation_in_saved_output(self):
        actions = hud_activation_actions(self.candidates, [{
            "hud_candidate_id": self.candidates[0]["id"], "status": "abstained", "actor_track": None}], [])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_rows(root/"core/actions.jsonl", actions)
            saved = effective_actions(root)
        self.assertEqual(len(saved), 1)
        self.assertEqual((saved[0]["slot_id"], saved[0]["evidence_class"]), ("E", "hud-supported"))
        self.assertEqual(saved[0]["scene_ids"], [])
        self.assertIsNone(saved[0]["actor_track"])
        self.assertIsNone(saved[0].get("target"))
        self.assertIsNone(saved[0]["outcome"])

    def test_nearby_other_actor_proposal_cannot_borrow_selected_hud_slot(self):
        candidate = self.candidates[0]
        proposal = {"id": "other-visible-cast", "kind": "ability_cast", "actor_track": "other-body",
                    "owner_id": "selected", "owner_epoch": 0, "segment_id": "negative-01",
                    "start": candidate["start"]-.1, "end": candidate["end"]+.1, "slot_id": None}
        binding = {"actor_track": "selected-body", "owner_id": "selected", "owner_epoch": 0,
                   "segment_id": "negative-01", "start": 93, "end": 95}
        result = associate_scene_casts([proposal], self.candidates, [binding])[0]
        self.assertEqual(result["status"], "scene_only")
        self.assertIsNone(result["linked_candidate"])
        self.assertIsNone(result["hud_slot_suggestion"])

    def test_apprehend_progress_bar_does_not_become_a_recall_attempt(self):
        from PIL import Image
        readings = []
        for row in self.fixture["frames"]:
            path = FIXTURES/row["cast_bar_path"]
            self.assertEqual(file_hash(path), row["cast_bar_sha256"])
            frame = {"id": row["id"], "time": row["time"], "layout_valid": True,
                     "segment_id": "negative-01", "owner_epochs": {"selected": 0}}
            with Image.open(path) as image:
                readings.append(read_recall(image, frame, self.fixture["recall_profile"], self.root,
                                            {"owner_id": "selected", "hud_champion": None}))
        self.assertFalse(any(row["active"] for row in readings))
        self.assertEqual(recall_events(readings), [])
        # The same production geometry/reader must accept actual Recall pixels;
        # rejecting every bar would not establish this hard-negative behavior.
        positive = []
        for row in self.fixture["positive_recall_frames"]:
            path = FIXTURES/row["path"]
            self.assertEqual(file_hash(path), row["sha256"])
            frame = {"id": row["id"], "time": row["time"], "layout_valid": True,
                     "segment_id": "negative-12", "owner_epochs": {"selected": 0}}
            with Image.open(path) as image:
                positive.append(read_recall(image, frame, self.fixture["recall_profile"], self.root,
                                           {"owner_id": "selected", "hud_champion": None}))
        self.assertTrue(all(row["active"] for row in positive))
        recalls = recall_events(positive)
        self.assertEqual(len(recalls), 1)
        self.assertEqual(recalls[0]["terminal_status"], "ending_unknown")


if __name__ == "__main__":
    unittest.main()
