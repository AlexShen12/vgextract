"""Reference admission and leakage guards for source-backed development cases."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from annotation.league.core import atomic_json, read_json
from annotation.league.scene_reference import (prepare, record_exposure, summarize,
                                                validate_label, validate_manifest)


def manifest():
    return {"schema_version": 1, "stage": 2, "split": "development", "guard_seconds": 8,
            "source": {"path": "source.mp4", "sha256": "a"*64, "width": 1920, "height": 1080},
            "held_out_exclusions": [{"id": "reserved", "start": 40, "end": 50}],
            "clips": [{"id": "negative-00", "start": 10, "end": 20, "split": "development"}]}


def label():
    return {"clip_id": "negative-00", "source_sha256": "a"*64,
            "review_status": "source_reviewed_model_draft", "exhaustive": False,
            "review": {"reviewer": "source-reviewer", "prediction_outputs_seen": False,
                       "human_reviewed": False, "full_native_review": False,
                       "method": "Source overview plus decisive native frames", "inspected_sheets": ["sheet.jpg"]},
            "events": [], "unknown_intervals": [],
            "cases": [{"tag": "initial_cooldown", "status": "supported", "start": 10, "end": 10.5,
                       "scope": "selected E initial state", "evidence": [{"time": 10, "cue": "Already counting down"}],
                       "reason": "No pre-entry ready state establishes an onset", "forbidden_claim": "New E at crop entry"}]}


def event(kind="hud_activation"):
    return {"id": "event-1", "kind": kind, "actor": "selected", "slot": "E",
            "start": 11, "end": 11.5, "onset_bracket": [11, 11.1],
            "ability_name": None, "target": None, "outcome": None,
            "evidence": [{"time": 11.1, "cue": "Native visible transition"}],
            "phases": [{"phase": "hud_activation", "start": 11, "end": 11.1}]}


class SceneReferenceTests(unittest.TestCase):
    def validate(self, value):
        data = manifest()
        return validate_label(value, data["clips"][0], data["source"]["sha256"])

    def test_held_out_guard_uses_entire_prepared_context(self):
        data = manifest()
        validate_manifest(data)
        data["clips"][0].update(start=25, end=32)
        validate_manifest(data)  # Exact eight-second separation is allowed.
        data["clips"][0]["end"] = 32.001
        with self.assertRaisesRegex(ValueError, "held-out guard"):
            validate_manifest(data)
        data["guard_seconds"] = 1
        with self.assertRaisesRegex(ValueError, "eight seconds"):
            validate_manifest(data)

    def test_duplicate_ids_and_held_out_processing_are_rejected(self):
        data = manifest()
        data["clips"].append(copy.deepcopy(data["clips"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_manifest(data)
        data = manifest()
        data["split"] = "held_out"
        with self.assertRaisesRegex(ValueError, "development"):
            validate_manifest(data)

    def test_prediction_exposure_and_model_verification_cannot_be_hidden(self):
        value = label()
        self.assertTrue(self.validate(value))
        value["review"]["prediction_outputs_seen"] = True
        with self.assertRaisesRegex(ValueError, "prediction-exposure"):
            self.validate(value)
        value = label()
        value["review_status"] = "human_verified"
        with self.assertRaisesRegex(ValueError, "human verification"):
            self.validate(value)

    def test_quiet_needs_full_review_not_just_empty_event_array(self):
        value = label()
        value["cases"][0]["tag"] = "quiet"
        with self.assertRaisesRegex(ValueError, "Quiet requires"):
            self.validate(value)
        value["exhaustive"] = True
        value["review"]["full_native_review"] = True
        self.assertTrue(self.validate(value))

    def test_later_countdown_is_not_an_initial_clip_state(self):
        value = label()
        value["cases"][0].update(start=14, end=14.5)
        with self.assertRaisesRegex(ValueError, "clip entry"):
            self.validate(value)

    def test_sequence_tags_require_the_physical_evidence_they_describe(self):
        for tag in ("isolated_ability", "rapid_attacks", "empowered_attack", "observable_miss", "obscured_outcome", "recall_lifecycle"):
            value = label()
            value["cases"][0].update(tag=tag, start=11, end=11.5)
            value["events"] = [event()]
            with self.subTest(tag=tag), self.assertRaises(ValueError):
                self.validate(value)
        value = label()
        value["cases"][0].update(tag="rapid_attacks", start=11, end=12.5)
        first, second = event("basic_attack"), event("basic_attack")
        second.update(id="attack-2", start=12, end=12.5, onset_bracket=[12, 12.1],
                      phases=[{"phase": "attack", "start": 12, "end": 12.5}])
        value["events"] = [first, second]
        self.assertTrue(self.validate(value))

    def test_hud_activation_survives_unknown_animation_without_invented_outcome(self):
        value = label()
        value["events"] = [event()]
        self.assertTrue(self.validate(value))
        value["events"][0].update(target="nearby enemy", target_evidence=[{"time": 11.1}])
        with self.assertRaisesRegex(ValueError, "HUD activation alone"):
            self.validate(value)

    def test_name_target_and_outcome_require_separate_evidence(self):
        for field in ("ability_name", "target", "outcome"):
            value = label()
            value["events"] = [event("ability_cast")]
            value["events"][0][field] = "assertion"
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "separate evidence"):
                self.validate(value)

    def test_interrupted_recall_is_a_positive_attempt_and_negative_completion(self):
        value = label()
        value["cases"][0]["tag"] = "interrupted_recall"
        with self.assertRaisesRegex(ValueError, "positive observed attempt"):
            self.validate(value)
        attempt = event("recall")
        attempt.update(slot=None, terminal_status="interrupted")
        value["events"] = [attempt]
        self.assertTrue(self.validate(value))
        attempt["terminal_status"] = "ending_unknown"
        with self.assertRaisesRegex(ValueError, "positive observed attempt"):
            self.validate(value)

    def test_source_time_phase_and_uncertainty_brackets_cannot_escape_clip(self):
        for field in ("phase", "onset", "evidence", "unknown"):
            value = label()
            value["events"] = [event()]
            if field == "phase":
                value["events"][0]["phases"][0]["end"] = 12
            elif field == "onset":
                value["events"][0]["onset_bracket"] = [10.5, 11.1]
            elif field == "evidence":
                value["events"][0]["evidence"][0]["time"] = 20
            else:
                value["unknown_intervals"] = [{"start": 19, "end": 21, "reason": "off camera"}]
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validate(value)

    def test_exposure_ledger_retains_rejected_candidates_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"exposure-ledger.json"
            for _ in range(2):
                record_exposure(path, "a"*64, "rejected-clip", 10, 20, "prepared_source_review")
            record_exposure(path, "a"*64, "rejected-clip", 11, 12, "native_or_dense_source_review")
            rows = read_json(path)["entries"]
            self.assertEqual(len(rows), 2)
            self.assertEqual((rows[0]["start"], rows[0]["end"]), (10, 20))

    def test_resume_refuses_changed_views_and_preserves_exposure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root/"manifest.json"
            atomic_json(path, manifest())
            atomic_json(root/"prepared/negative-00/source-inputs.json", {
                "source_sha256": "a"*64, "start": 10, "end": 20, "sampling_fps": 4, "frames": []})
            with patch("annotation.league.scene_reference.file_hash", return_value="a"*64):
                with self.assertRaisesRegex(ValueError, "Changed preparation"):
                    prepare(path, root/"prepared", fps=8)
            self.assertEqual(len(read_json(root/"exposure-ledger.json")["entries"]), 1)

    def test_missing_labels_and_untested_cases_are_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root/"manifest.json"
            atomic_json(path, manifest())
            result = summarize(path, root/"labels")
            self.assertEqual(result["missing_labels"], ["negative-00"])
            self.assertEqual(result["coverage"]["quiet"]["supported_clips"], [])
            atomic_json(root/"labels/negative-00.json", label())
            result = summarize(path, root/"labels")
            self.assertEqual(result["missing_labels"], [])
            self.assertEqual(result["coverage"]["initial_cooldown"]["supported_clips"], ["negative-00"])


if __name__ == "__main__":
    unittest.main()
