from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from annotation.league.body_roster import SCOREBOARD_SLOTS
from annotation.league.body_slot_context import MatchRosterIndex, skim_revision_matches
from annotation.league.core import atomic_json, digest
from annotation.league.media import file_hash
from annotation.league.scoreboard_skim import build_match_roster, derive_membership, save_roster_review


SLOTS = sorted(SCOREBOARD_SLOTS)
SOURCE = "a" * 64
NAMES = {f"Champion{index}" for index in range(10)}


def sample(time, indices, *, changes=None, crop=None):
    changes = changes or {}
    evidence = {"frame": crop.name, "frame_sha256": file_hash(crop)} if crop else {}
    return {"time": time, "opening": time, **evidence, "portraits": {
        SLOTS[index]: {"id": f"portrait-{time}-{index}", "value": changes.get(index, f"Champion{index}"),
                       "acceptance": "accepted", "score": .81, "margin": .14,
                       **({"crop": crop.name, "crop_sha256": file_hash(crop)} if crop else {})}
        for index in indices
    }}


class ScoreboardSkimTests(unittest.TestCase):
    def test_partial_openings_fill_experimental_pool_and_preserve_first_evidence(self):
        observations = [sample(3, range(7)), sample(12, range(5, 10)), sample(20, [0])]
        roster = build_match_roster(observations, SOURCE, [0, 25], single_match=True)
        self.assertEqual(roster["status"], "experimental")
        self.assertEqual(roster["confirmed_at"], 12)
        self.assertEqual(set(roster["champions"]), NAMES)
        evidence = roster["members"]["left"]["Champion0"]
        self.assertEqual(evidence["first"]["time"], 3)
        self.assertEqual(evidence["accepted_reads"], 2)
        self.assertEqual(build_match_roster(observations, SOURCE, [0, 25])["status"], "review")

    def test_pool_deduplicates_champions_and_rejects_a_sixth_team_member(self):
        initial = [sample(3, range(7)), sample(4, range(7, 10))]
        reordered = sample(10, [0, 1], changes={0: "Champion1", 1: "Champion0"})
        roster = build_match_roster(initial + [reordered], SOURCE, [0, 15], single_match=True)
        self.assertEqual(roster["status"], "experimental")
        self.assertEqual(set(roster["champions"]), NAMES)
        self.assertEqual(roster["confirmed_at"], 4)
        disputed = build_match_roster(initial + [sample(11, [0], changes={0: "Extra"})], SOURCE, [0, 15], single_match=True)
        self.assertEqual(disputed["status"], "disputed")
        self.assertEqual(disputed["champions"], [])

    def test_unknown_portraits_do_not_add_champions(self):
        observations = [sample(2, range(10))]
        observations[0]["portraits"][SLOTS[0]].update(acceptance="unknown")
        roster = build_match_roster(observations, SOURCE, [0, 10], single_match=True)
        self.assertEqual(roster["status"], "incomplete")
        self.assertEqual(roster["missing_members"], {"left": 1, "right": 0})

    def fixture(self, root, *, conflict=False):
        crop = root / "crop.png"
        crop.write_bytes(b"source crop fixture")
        samples = [sample(10, range(7), crop=crop), sample(12, range(7, 10), crop=crop)]
        if conflict:
            samples.append(sample(13, range(10), changes={0: "Extra"}, crop=crop))
        profile = {"calibration_source_sha256": SOURCE, "inventory": {"reference_version": "16.22.1"}}
        profile_id = digest(profile)
        recipe = {"single_match": True, "profile": profile_id}
        name = {"id": "name", "portrait_id": "portrait-10-0", "slot": SLOTS[0], "opening": 10,
                "time": 10, "crop": crop.name}
        names = [name]
        inputs = {"samples_sha256": digest(samples), "names_sha256": digest(names), "recipe": recipe}
        atomic_json(root / "layout_profiles" / f"{profile_id}.json", profile)
        atomic_json(root / "manifest.json", {"source": {"sha256": SOURCE, "path": str(root / "missing.mp4")},
                                            "interval": [0, 25], "recipe": recipe})
        atomic_json(root / "scoreboard-samples.json", samples)
        atomic_json(root / "scoreboard-names.json", names)
        atomic_json(root / "skim-evidence.json", inputs)
        derive_membership(root)
        portraits = [{"portrait_id": row["id"], "champion": row["value"]}
                     for frame in samples[:2] for row in frame["portraits"].values()]
        portraits[0].update(name_id="name", name="Jo\u0301#NA", name_crop_sha256=file_hash(crop))
        review = {"source_sha256": SOURCE, "samples_sha256": digest(samples), "reviewer": "Test reviewer",
                  "reviewed_at": "2026-11-12T10:00:00-05:00", "interval": [4, 20], "continuity_verified": True,
                  "portraits": portraits}
        return crop, review

    def test_unreviewed_pool_can_supply_experimental_ids_without_name_bindings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            roster = MatchRosterIndex(root, SOURCE, NAMES).at(1)
            self.assertEqual(roster["confirmation_method"], "unreviewed_cross_opening")
            self.assertEqual(set(roster["champions"]), NAMES)
            self.assertFalse(roster["bindings"])

    def test_reviewed_pool_scopes_backfill_and_rejects_changed_witness(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            crop, review = self.fixture(root)
            derive_membership(root, review)
            index = MatchRosterIndex(root, SOURCE, NAMES)
            self.assertIsNone(index.at(3.9))
            self.assertEqual(index.at(4)["confirmed_at"], 12)
            self.assertEqual(index.at(19)["bindings"]["Jó#NA"]["champion"], "Champion0")
            self.assertIsNone(index.at(20))
            self.assertTrue(skim_revision_matches(index.revision))
            with self.assertRaises(ValueError):
                MatchRosterIndex(root, "b" * 64, NAMES)
            crop.write_bytes(b"changed")
            self.assertFalse(skim_revision_matches(index.revision))
            with self.assertRaisesRegex(ValueError, "witness changed"):
                MatchRosterIndex(root, SOURCE, NAMES)

    def test_conflicting_complete_read_suspends_roster_until_corrected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, review = self.fixture(root, conflict=True)
            derive_membership(root, review)
            self.assertIsNone(MatchRosterIndex(root, SOURCE, NAMES).at(10))
            review["portraits"].append({"portrait_id": "portrait-13-0", "champion": None})
            derive_membership(root, review)
            self.assertEqual(MatchRosterIndex(root, SOURCE, NAMES).at(10)["status"], "confirmed")

    def test_wrong_row_name_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, review = self.fixture(root)
            first, second = review["portraits"][:2]
            second.update({key: first.pop(key) for key in ("name", "name_id", "name_crop_sha256")})
            with self.assertRaisesRegex(ValueError, "same scoreboard row"):
                derive_membership(root, review)

    def test_rejecting_a_member_with_no_replacement_suspends_the_roster(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, review = self.fixture(root)
            derive_membership(root, review)
            self.assertEqual(MatchRosterIndex(root, SOURCE, NAMES).at(10)["status"], "confirmed")
            review["portraits"][-1]["champion"] = None
            self.assertEqual(derive_membership(root, review)["status"], "incomplete")
            self.assertIsNone(MatchRosterIndex(root, SOURCE, NAMES).at(10))

    def test_optional_review_api_rejects_stale_edits_and_saves_evidence(self):
        from fastapi.testclient import TestClient
        from annotation.league.body_roster_review import create_app

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, review = self.fixture(root)
            assets = {"assets": [{"id": name, "kind": "champion"} for name in NAMES]}
            with patch("annotation.league.body_roster_review.load_inventory_references", return_value=assets):
                app = create_app(root, root)
            with TestClient(app) as client:
                evidence = client.get("/api/roster").json()
                self.assertEqual(evidence["status"], "experimental")
                self.assertEqual(client.get("/evidence/portrait-10-0/portrait").status_code, 200)
                self.assertEqual(client.post("/api/roster", json={**review, "revision": "0" * 64}).status_code, 422)
                saved = client.post("/api/roster", json={**review, "revision": evidence["revision"]})
                self.assertEqual(saved.status_code, 200, saved.text)
                self.assertEqual(client.get("/api/roster").json()["status"], "confirmed")
                self.assertEqual(save_roster_review(root, None)["revision"], saved.json()["revision"])
