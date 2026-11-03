"""Temporal inference, adaptive review, benchmark and pipeline integration contracts."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from PIL import Image

from annotation.league.action_state import (deduplicate_scene, escalation_windows,
                                          link_actions, load_action_profile, transitions)
from annotation.league.action_pipeline import packets, validate_slots
from annotation.league.action_benchmark import detection_metrics, initialize_actions, validate_labels
from annotation.league.action_review import effective_actions
from annotation.league.core import Config, Store, atomic_json, read_json, write_rows
from annotation.league.review import create_app
from annotation.league.action_icons import AbilityIdentity, guard_kit_changes


def reading(t, state, slot="Q", **extra):
    return {"id": f"r-{t}-{slot}", "frame_id": f"f-{t}", "time": t, "state": state, "slot_id": slot,
            "owner_id": "selected", "owner_epoch": 0, "segment_id": "segment-a", "slot_kind": "ability",
            "seconds": 5 if state == "cooling_down" else None, "charges": None, "pixel_change": 1,
            "item_effect": "unknown", "name": None, **extra}


class TemporalTests(unittest.TestCase):
    def test_multi_icon_identity_tolerates_obscured_slot_but_abstains_on_conflict(self):
        def hit(name, slot):
            return {"acceptance": "accepted", "value": name+":"+slot}
        rows = {slot: hit("Example", slot) for slot in ("W", "E", "R")}
        rows["Q"] = {"acceptance": "unknown", "value": None}
        self.assertEqual(AbilityIdentity.consensus(rows), "Example")
        rows["Q"] = hit("Other", "Q")
        self.assertIsNone(AbilityIdentity.consensus(rows))
        rows.pop("Q")
        rows.pop("W")
        self.assertIsNone(AbilityIdentity.consensus(rows))

    def test_changed_ability_kit_is_not_a_cast(self):
        hud = [reading(0, "ready"), reading(.0625, "cooling_down"), reading(.125, "cooling_down")]
        identity = [{"time": r["time"], "frame_id": r["frame_id"], "segment_id": r["segment_id"],
                     "owner_epoch": 0, "owner_id": "selected", "hud_champion": "First" if i == 0 else "Other"}
                    for i, r in enumerate(hud)]
        candidates = transitions(guard_kit_changes(hud, identity))
        self.assertFalse(any(t["activation_supported"] for t in candidates))

    def test_ability_identity_rechecks_and_does_not_carry_across_selections(self):
        reader = object.__new__(AbilityIdentity)
        reader.signature = "reference-signature"
        reader.identify = Mock(side_effect=[{"champion": "First"}, {"champion": None}, {"champion": "Other"}])
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp)/"frame.png"
            Image.new("RGB", (10, 10)).save(image)
            frames = [{"id": str(i), "time": t, "native_path": str(image), "layout_valid": True,
                       "segment_id": "s", "owner_epochs": {"selected": epoch}}
                      for i, (t, epoch) in enumerate([(0, 0), (.125, 0), (.25, 0), (.3125, 1)])]
            rows = reader.read_frames(frames, {"owners": [{"id": "selected"}]}, "unassisted")
        self.assertEqual([r["hud_champion"] for r in rows], ["First", "First", None, "Other"])
        self.assertTrue(all(r["reference_assisted"] and not r["binding_visible"] for r in rows))
        self.assertEqual(reader.identify.call_count, 3)

    def test_countdown_persistence_and_ready_return_do_not_count_again(self):
        rows = [reading(0, "ready"), reading(.1, "cooling_down"), reading(.2, "cooling_down", seconds=4),
                reading(1, "ready"), reading(1.1, "cooling_down")]
        self.assertEqual(len(transitions(rows)), 2)

    def test_gaps_cuts_selection_swaps_and_initial_cooldown_do_not_invent_uses(self):
        for middle in (reading(.1, "unknown"), reading(.1, "cooling_down", segment_id="new"),
                       reading(.1, "cooling_down", owner_epoch=1), reading(.1, "cooling_down", item_epoch=1)):
            self.assertEqual(transitions([reading(0, "ready"), middle]), [])
        self.assertEqual(transitions([reading(0, "cooling_down")]), [])
        self.assertEqual(transitions([reading(0, "ready"), reading(1, "cooling_down")]), [])
        uncertain = transitions([reading(0, "unknown"), reading(.1, "cooling_down")])
        self.assertFalse(uncertain[0]["activation_supported"])
        self.assertEqual(link_actions(uncertain, [], [])[0]["evidence_class"], "unresolved")

    def test_simultaneous_slots_and_explicit_charge_recast(self):
        rows = [reading(0, "ready"), reading(0, "ready", "W"), reading(.1, "cooling_down"), reading(.1, "cooling_down", "W")]
        self.assertEqual({r["slot_id"] for r in transitions(rows)}, {"Q", "W"})
        self.assertEqual(len(transitions([reading(0, "charge", charges=2), reading(.1, "charge", charges=1,
                                                 charge_consumption_visible=True)])), 1)
        self.assertEqual(len(transitions([reading(0, "recast"), reading(.1, "recast", recast_visible=True)])), 1)
        self.assertEqual(len(transitions([reading(0, "ready"), reading(.05, "cooling_down"),
                                         reading(.1, "ready"), reading(.15, "cooling_down")])), 2)

    def test_linking_requires_binding_and_compatible_slot_and_scene_actor(self):
        hud = transitions([reading(0, "ready"), reading(.1, "cooling_down")])
        binding = {"id": "binding", "mode": "unassisted", "owner_id": "selected", "owner_epoch": 0,
                   "segment_id": "segment-a", "champion": "Example", "actor_track": "blue-armored",
                   "actor_description": "blue-bar armored body", "start": 0, "end": .5, "evidence_ids": ["identity1", "identity2"]}
        scene = {"id": "scene", "mode": "unassisted", "owner_epoch": 0, "segment_id": "segment-a",
                 "actor_track": "blue-armored", "kind": "ability_cast", "slot_id": "Q", "name": None,
                 "start": .05, "end": .2, "frame_ids": ["f-0.1"], "evidence_ids": ["request"]}
        self.assertEqual(link_actions(hud, [], [binding])[0]["evidence_class"], "hud-supported")
        self.assertEqual(link_actions(hud, [scene], [binding])[0]["evidence_class"], "corroborated")
        self.assertEqual(link_actions(hud, [scene], [])[0]["evidence_class"], "unresolved")
        wrong = {**scene, "slot_id": "W"}
        result = link_actions(hud, [wrong], [binding])
        self.assertEqual({r["evidence_class"] for r in result}, {"hud-supported", "scene-only"})

    def test_item_transition_is_not_use_without_active_effect_identity(self):
        hud = transitions([reading(0, "ready", "item_1", slot_kind="item"), reading(.1, "cooling_down", "item_1", slot_kind="item")])
        result = link_actions(hud, [], [])
        self.assertEqual(result[0]["kind"], "item_transition")
        self.assertEqual(result[0]["evidence_class"], "unresolved")

    def test_overlap_merges_only_shared_frames_of_same_actor_slot(self):
        row = {"id": "a", "mode": "unassisted", "segment_id": "s", "owner_epoch": 0, "actor_track": "body",
               "kind": "ability_cast", "slot_id": "Q", "name": None, "start": 1, "end": 2,
               "frame_ids": ["f1", "f2"], "evidence_ids": ["request1"]}
        duplicate = {**row, "id": "b", "start": 2, "end": 3, "frame_ids": ["f2", "f3"]}
        distinct = {**row, "id": "c", "start": 3, "end": 4, "frame_ids": ["f4"]}
        self.assertEqual(len(deduplicate_scene([row, duplicate, distinct])), 2)

    def test_bursts_escalate_with_segment_bounds(self):
        hud = transitions([reading(0, "ready"), reading(.1, "ready", "W"), reading(.1, "cooling_down"), reading(.2, "cooling_down", "W")])
        frames = [{"segment_id": "segment-a", "time": t/16} for t in range(16)]
        windows = escalation_windows(hud, link_actions(hud, [], []), frames)
        self.assertEqual(len([w for w in windows if "rapid_sequence" in w["reasons"]]), 2)
        self.assertTrue(all(w["start"] >= 0 and w["end"] <= 1 for w in windows))


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = Store(Path(self.temp.name))
        self.profile = load_action_profile()
        self.root = self.store.path("experiments/actions-v2/"+"a"*20)
        self.root.mkdir(parents=True)
        image = self.root / "input.png"
        self.frames = [{"id": f"frame-{i}", "time": 30+i/16, "input_path": str(image), "native_path": str(image),
                        "input_sha256": "hash", "segment_id": "s", "owner_epochs": {"selected": 0},
                        "layout_valid": True, "scene_sample": i % 4 == 0} for i in range(16)]
        self.audit = {"root": str(self.root), "source_sha256": "hash", "interval": [30, 31], "profile": self.profile,
                      "source": {"fps": 60}, "frames": self.frames}
        atomic_json(self.root / "input-audit.json", self.audit)
        self.config = Config({}, [])

    def test_invalid_slot_schema_and_citations_rejected(self):
        slot = {"frame_id": "invented", "owner_id": "selected", "slot_id": "Q", "state": "ready", "seconds": None,
                "charges": None, "recast_visible": False, "charge_consumption_visible": False,
                "item_effect": "unknown", "name": None, "evidence": "visible"}
        with self.assertRaisesRegex(ValueError, "Invented"):
            validate_slots([slot], self.frames, self.profile, "core", "req")
        slot.update(frame_id="frame-0", state="cooling_down")
        with self.assertRaisesRegex(ValueError, "numeral"):
            validate_slots([slot], self.frames, self.profile, "core", "req")
        slot["invented_field"] = True
        with self.assertRaises(ValueError):
            validate_slots([slot], self.frames, self.profile, "core", "req")

    def test_packet_boundaries(self):
        self.assertEqual([len(g) for g in packets(self.frames)], [8, 8, 2])
        frames = copy.deepcopy(self.frames)
        frames[4]["owner_epochs"]["selected"] = 1
        self.assertTrue(all(len({json.dumps(f["owner_epochs"]) for f in p}) == 1 for p in packets(frames)))
        frames[8]["segment_id"] = "cut"
        self.assertTrue(all(len({f["segment_id"] for f in p}) == 1 for p in packets(frames)))

    def test_review_is_integrated_and_corrections_are_isolated(self):
        from fastapi.testclient import TestClient
        atomic_json(self.root / "report.json", {"status": "completed_experimental", "interval": [30, 31]})
        row = {"id": "action", "mode": "unassisted", "start": 30, "end": 30.1, "evidence_class": "unresolved"}
        write_rows(self.root / "unassisted/actions.jsonl", [row])
        self.store.path("events.jsonl").write_text("untouched\n")
        client = TestClient(create_app(self.store))
        self.assertEqual(client.get("/actions").status_code, 200)
        self.assertEqual(client.get("/api/actions").json()["experiments"][0]["id"], "a"*20)
        self.assertEqual(client.get("/api/actions/"+"a"*20).status_code, 200)
        self.assertEqual(client.get("/api/actions/invalid").status_code, 404)
        response = client.post("/api/actions/"+"a"*20+"/corrections", json={"record_id": "action", "changes": {"champion": "Example"}, "review_status": "human-verified"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(effective_actions(self.root)[0]["champion"], "Example")
        self.assertEqual(self.store.path("events.jsonl").read_text(), "untouched\n")
        self.assertFalse(self.store.path("corrections").exists())
        self.assertEqual(client.post("/api/actions/"+"a"*20+"/corrections", headers={"Origin": "https://elsewhere.test"}, json={}).status_code, 403)

    def test_initialization_is_unverified_and_calibration_is_excluded(self):
        atomic_json(self.store.path("manifest.json"), {"source": {"sha256": "hash"}, "interval": [0, 1732]})
        directory = self.store.path("labels")
        initialize_actions(self.store, self.config, directory)
        data = read_json(directory/"actions.json")
        self.assertEqual(len(validate_labels(self.store, data, self.profile, draft=True)), 20)
        with self.assertRaisesRegex(ValueError, "verified"):
            validate_labels(self.store, data, self.profile)
        with self.assertRaisesRegex(ValueError, "overwrite"):
            initialize_actions(self.store, self.config, directory)

    def test_metrics_penalize_duplicates_and_preserve_empty_reference(self):
        row = {"id": "a", "kind": "ability_cast", "slot_id": "Q", "start": 1, "end": 1.2, "champion": "Example"}
        result = detection_metrics([row], [row, {**row, "id": "b"}], ("kind", "slot_id"))
        self.assertEqual(result["precision"], .5)
        self.assertEqual(result["recall"], 1)
        self.assertEqual(result["duplicate_predictions"], 1)
        self.assertIsNone(detection_metrics([], [], ("kind",))["recall"])


if __name__ == "__main__":
    unittest.main()
