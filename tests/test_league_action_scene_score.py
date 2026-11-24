"""Scene benchmark behavior with independent reference and inference fixtures."""
from pathlib import Path
from contextlib import redirect_stdout
import io
import tempfile
import unittest
from unittest.mock import Mock, patch

from annotation.league.action_benchmark_v3 import experiment_actions
from annotation.league.__main__ import main
from annotation.league.action_scene_score import scene_ledger, summarize_scene
from annotation.league.action_v3 import KINDS, action_window
from annotation.league.core import Config, Store, atomic_json, read_json, write_rows


def event(kind, start, end, *, role="selected", slot=None):
    row = {"kind": kind, "start": start, "end": end, "actor_role": role, "slot_id": slot,
           "owner_id": "selected" if role == "selected" else None, "evidence_note": "Visible action",
           "boundary_resolvable": True}
    if kind == "recall":
        row["terminal_status"] = "ending_unknown"
    return row


class SceneBenchmarkTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.store = Store(Path(temporary.name))
        self.config = Config.load(Path(__file__).resolve().parents[1]/"annotation/league/midbeast.json")
        self.config.asset_cache = str(self.store.path("assets"))

    def test_successful_scene_cli_phases_release_slurm_dependencies(self):
        args = ["--run-dir", str(self.store.root), "experiment", "--suite", "actions", "--infer"]
        with patch("annotation.league.action_benchmark_v3.experiment_actions") as experiment, redirect_stdout(io.StringIO()):
            for status, expected in (("completed_experimental", 0), ("development_frozen", 0),
                                     ("unverified_scene_labels_created", 0), ("failed_experimental", 1)):
                experiment.return_value = {"experimental": True, "status": status}
                with self.subTest(status=status):
                    self.assertEqual(main([*args, "--scene-baseline"]), expected)
            experiment.return_value = {"experimental": True, "status": "completed_experimental"}
            self.assertEqual(main(args), 2)

    def test_all_categories_points_intervals_duplicates_and_attribution(self):
        refs = [event(kind, i, i+1 if kind == "movement" else i, slot="Q" if kind == "ability_cast" else None)
                for i, kind in enumerate(KINDS)]
        refs[0]["left_censored"] = True
        predictions = [{**r, "id": str(i), "evidence": [{"frame_id": str(i), "cue": "example"}]}
                       for i, r in enumerate(refs)]
        predictions[1].update(owner_id=None, actor_track="other-a", start=1.25, end=1.25)
        predictions[2].update(owner_id=None, slot_id=None)
        predictions.extend([dict(predictions[1]), {**predictions[0], "start": 0, "end": 8}])
        audit, proposals = self.store.path("audit.json"), self.store.path("proposals.jsonl")
        atomic_json(audit, {"frames": [{"id": str(i), "time": i} for i in range(len(refs))]})
        write_rows(proposals, predictions)
        ledger = scene_ledger({"id": "clip", "actions": refs}, {"input_audit": str(audit),
            "results": {"scene_baseline": {"proposals": str(proposals)}}, "requests": [{"status": "accepted"}, {"status": "rejected"}]})
        scores = summarize_scene([ledger])
        self.assertEqual(set(scores["categories"]), set(KINDS))
        self.assertTrue(all(row["matched"] == 1 for row in scores["categories"].values()))
        self.assertEqual(scores["categories"]["basic_attack"]["duplicates"], 1)
        self.assertEqual(scores["categories"]["movement"]["onset_matches"], 0)
        self.assertEqual(scores["categories"]["movement"]["false_positives"], 1)
        self.assertEqual(scores["actor_role"]["incorrect"], 1)
        self.assertEqual(scores["actor_role"]["unresolved"], 1)
        self.assertEqual(scores["ability_slot"]["unresolved"], 1)
        self.assertEqual(scores["request_statuses"], {"accepted": 1, "rejected": 1})
        self.assertEqual(ledger["proposals"][2]["cited_evidence"][0]["source_time"], 2)

    def test_quiet_clip_and_missing_category_denominators(self):
        result = summarize_scene([{"references": [], "proposals": [], "request_statuses": {"accepted": 1}}])
        self.assertTrue(all(not row["tested"] and row["recall"] is None for row in result["categories"].values()))

    def test_scene_mode_skips_hud_fusion_and_resampling(self):
        root = self.store.path("experiments/actions-v3-scene-baseline/clip")
        root.mkdir(parents=True)
        audit = {"root": str(root), "frames": [{"scene_sample": True, "layout_valid": True},
                                                {"scene_sample": False, "layout_valid": True}]}
        reader = Mock(requests=[], uncertainty=[])
        reader.read.return_value = {"actions": [{"id": "first-pass"}]}
        runner = Mock(identity={"model": "fixture"}, peak_memory_bytes=0)
        with patch("annotation.league.action_v3.prepare_inputs", return_value=audit) as prepare, \
             patch("annotation.league.action_v3.bind_frames", return_value=([], [])), \
             patch("annotation.league.action_v3.SceneReader", return_value=reader), \
             patch("annotation.league.action_v3.scan_hud", side_effect=AssertionError("HUD inference")):
            # With no icon references, action_window still builds a placeholder identity per frame.
            for i, frame in enumerate(audit["frames"]):
                frame.update(id=str(i), time=i*.0625, owner_epochs={"selected": 0}, segment_id="segment")
            result = action_window(self.store, self.config, 0, 8, infer=True, scene_baseline=True, runner=runner)
        self.assertEqual(prepare.call_args.kwargs["namespace"], "actions-v3-scene-baseline")
        reader.read.assert_called_once_with(audit["frames"][:1])
        self.assertEqual(result["status"], "completed_experimental")
        self.assertTrue(Path(result["results"]["scene_baseline"]["proposals"]).exists())

    def test_independence_development_freeze_and_held_out_gates(self):
        atomic_json(self.store.path("manifest.json"), {"source": {"sha256": "fixture"}, "interval": [0, 1732]})
        experiment_actions(self.store, self.config, initialize=True)
        options = {"infer": True, "scene_baseline": True, "runner": Mock(identity={"model": "fixture"})}
        manifest = Path(experiment_actions(self.store, self.config, initialize=True, **options)["labels"])
        with self.assertRaisesRegex(ValueError, "Independent verified development"):
            experiment_actions(self.store, self.config, draft=True, **options)
        data = read_json(manifest)
        attestation = {"labeler": "blind", "independent_of_predictions": True}
        data.update(attestation)
        for clip in data["clips"]:
            clip.update(attestation, verified=True, inspection_note="Reviewed source")
        sample = data["clips"][-1]
        sample["actions"] = [event("base_arrival", sample["start"], sample["start"], role="unknown")]
        atomic_json(manifest, data)
        with self.assertRaisesRegex(ValueError, "before freezing"):
            experiment_actions(self.store, self.config, freeze=True, **options)
        with self.assertRaisesRegex(ValueError, "Freeze unchanged"):
            experiment_actions(self.store, self.config, **options)
        proposals, audit = self.store.path("p.jsonl"), self.store.path("a.json")
        write_rows(proposals, [{"id": "pred", "kind": "movement", "start": 0, "end": 0, "evidence": []}])
        atomic_json(audit, {"frames": []})
        result = {"report": "clip.json", "resource_metrics": {}, "input_audit": str(audit),
                  "requests": [{"status": "accepted"}], "results": {"scene_baseline": {"proposals": str(proposals), "sampled_frames": 32}}}
        with patch("annotation.league.action_benchmark_v3.action_window", return_value=result) as run:
            draft = experiment_actions(self.store, self.config, draft=True, **options)
            self.assertIsNone(draft["quality_metrics"])
            self.assertEqual(list(draft["clips"]), [c["id"] for c in data["clips"][:10]])
            self.assertEqual(experiment_actions(self.store, self.config, freeze=True, **options)["status"], "development_frozen")
            scored = experiment_actions(self.store, self.config, **options)
            self.assertEqual(list(scored["clips"]), [c["id"] for c in data["clips"][10:]])
            self.assertEqual(scored["quality_metrics"]["categories"]["base_arrival"]["misses"], 1)
            self.assertEqual(run.call_count, 20)
        data["clips"][0]["inspection_note"] = "Changed development label"
        atomic_json(manifest, data)
        with self.assertRaisesRegex(ValueError, "Freeze unchanged"):
            experiment_actions(self.store, self.config, **options)
