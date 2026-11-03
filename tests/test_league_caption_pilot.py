"""Caption pilot evidence, lifecycle, review and independent acceptance contracts."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from annotation.league.__main__ import parser
from annotation.league.caption_benchmark import aggregate, initialize_labels, validate_labels, experiment_captions, score_clip, CASES
from annotation.league.caption_pilot import caption_pilot, counter_observations, regenerate, _cpu_detection
from annotation.league.caption_review import assess_caption, correct_pilot
from annotation.league.caption_scene import CaptionSceneReader, accepted_events, assemble_fragments
from annotation.league.caption_segments import POLICY, contexts, deduplicate_events, link_repeats, segment_captions
from annotation.league.captions import construct_captions
from annotation.league.core import Config, Store, atomic_json, read_json, read_rows, write_rows
from annotation.league.media import file_hash
from annotation.league.playback import PlaybackTracker
from annotation.league.review import create_app


def frames(start=0, end=6, fps=16, segment="seg", epoch=0):
    return [{"id": f"f{i}", "time": i/fps, "segment_id": segment, "owner_epochs": {"selected": epoch},
             "layout_valid": True, "scene_sample": i % (fps//4) == 0, "source_fps": 60,
             "playback": {"playback_state": "playing", "continuity_verified": True}}
            for i in range(round(start*fps), round(end*fps))]


def body_rows(fs):
    return [{"id": "body-"+f["id"], "frame_id": f["id"], "time": f["time"], "owner_id": "selected",
             "owner_epoch": f["owner_epochs"]["selected"], "segment_id": f["segment_id"], "actor_track": "player-track",
             "binding_visible": True, "champion": "Aatrox", "actor_region": [200, 200, 300, 400]} for f in fs]


def event(fs, kind="ability_hit"):
    ids = [f["id"] for f in fs]
    required = {"ability_hit": [("ability_cast_visible", ids[:2]), ("target_impact_visible", ids[-1:]), ("cast_linked_to_impact", ids)],
                "cast": [("cast_visible", ids)], "targeted_attack": [("attack_visible", ids), ("targeting_visible", ids)],
                "attack": [("attack_visible", ids)],
                "minion_last_hit": [("minion_death_visible", ids[-1:]), ("actor_action_linked_to_death", ids)]}[kind]
    target = "minion:1" if kind == "minion_last_hit" else "champion:target"
    row = {"id": "event", "kind": kind, "actor": "Aatrox", "actor_label": "Aatrox", "target": target,
        "target_label": "the observed champion", "ability": "Q", "start": fs[0]["time"], "end": fs[-1]["time"],
        "owner_id": "selected", "owner_epoch": 0, "playback_segment_id": "seg", "sampling_fps": 16,
        "evidence_ids": ids, "visual_evidence": [{"criterion": c, "frame_ids": f} for c, f in required],
        "actor_binding": {"entity": "Aatrox", "binding": "player-track", "frame_ids": ids},
        "target_binding": {"entity": target, "binding": target, "frame_ids": ids,
                           "regions": [{"frame_id": f, "box": [500, 200, 600, 400]} for f in ids]},
        "experimental": True, "caption_pilot": True, "acceptance": "grounded", "review_status": "unreviewed"}
    if kind in {"cast", "attack"}:
        row["target"] = None
        row.pop("target_binding")
    return row


def caption(t, end=None, kind="attack", epoch=0):
    return {"id": f"caption-{t}", "start": t, "end": t+.1 if end is None else end, "text": f"Claim at {t}.",
            "playback_segment_id": "seg", "owner_id": "selected", "owner_epoch": epoch,
            "claims": [{"kind": kind}], "evidence_refs": [{"table": "events", "id": f"event-{t}"}]}


class GroundingTests(unittest.TestCase):
    def test_counters_never_supply_a_last_hit_and_unknowns_break_deltas(self):
        fs = frames(0, .5)
        rows = [{"id": f["id"], "frame_id": f["id"], "time": f["time"], "owner_id": "selected", "owner_epoch": 0,
                 "segment_id": "seg", "field": "cs", "value": 10 if i < 4 else 11, "layout_valid": True} for i, f in enumerate(fs)]
        from annotation.league.timeline import fuse_observations
        obs = counter_observations(rows, fs, body=body_rows(fs))
        states, _ = fuse_observations(obs, caption_pilot=True)
        cs, _ = construct_captions(states, [], obs, interval=[0, 1], source_sha256="source", pilot=True)
        self.assertIn("cs_increased", [c["claims"][0]["kind"] for c in cs])
        self.assertFalse(any("last hit" in c["text"] for c in cs))
        rows[3]["value"] = None
        obs = counter_observations(rows, fs, body=body_rows(fs))
        states, _ = fuse_observations(obs, caption_pilot=True)
        cs, _ = construct_captions(states, [], obs, interval=[0, 1], source_sha256="source", pilot=True)
        self.assertNotIn("cs_increased", [c["claims"][0]["kind"] for c in cs])

    def test_unbound_counter_owner_stays_a_hud_slot_without_a_delta(self):
        fs = frames(0, .5)
        rows = [{"id": f["id"], "frame_id": f["id"], "time": f["time"], "owner_id": "selected", "owner_epoch": 0,
                 "segment_id": "seg", "field": "cs", "value": 10 if i < 4 else 11, "layout_valid": True} for i, f in enumerate(fs)]
        from annotation.league.timeline import fuse_observations
        obs = counter_observations(rows, fs)
        states, _ = fuse_observations(obs, caption_pilot=True)
        captions, _ = construct_captions(states, [], obs, interval=[0, 1], source_sha256="source", pilot=True)
        self.assertTrue(captions)
        self.assertTrue(all(c["claims"][0]["kind"] == "displayed_state" for c in captions))
        self.assertTrue(all(c["actor_bindings"][0]["entity"] is None for c in captions))

    def test_grounded_impact_and_production_isolation(self):
        fs = frames(1, 1.25)
        row = event(fs)
        accepted, omitted = accepted_events([row], fs, body_rows(fs))
        self.assertEqual(len(accepted), 1)
        self.assertEqual(omitted, [])
        kwargs = dict(interval=[0, 2], source_sha256="source", frame_times={f["id"]: f["time"] for f in fs})
        self.assertEqual(construct_captions([], accepted, [], **kwargs)[0], [])
        captions, _ = construct_captions([], accepted, [], **kwargs, pilot=True,
            frame_context={f["id"]: f for f in fs}, body={r["frame_id"]: r for r in body_rows(fs)})
        self.assertIn("observed champion", captions[0]["text"])
        self.assertEqual(captions[0]["review_status"], "unreviewed")
        self.assertTrue(captions[0]["experimental"])

    def test_missing_impact_retains_only_visible_cast(self):
        fs = frames(1, 1.25)
        row = event(fs)
        row["visual_evidence"] = [c for c in row["visual_evidence"] if c["criterion"] != "target_impact_visible"]
        accepted, omitted = accepted_events([row], fs, body_rows(fs))
        self.assertEqual([r["kind"] for r in accepted], ["cast"])
        self.assertEqual(omitted[0]["reason"], "missing_visual_criteria")
        self.assertIsNone(accepted[0]["target"])

    def test_sparse_wrong_owner_occluded_target_or_late_cast_cannot_establish_hit(self):
        original = frames(1, 1.25)
        for case in ("sparse", "owner", "body", "target", "ordering"):
            with self.subTest(case=case):
                fs, row, body = copy.deepcopy(original), event(original), body_rows(original)
                if case == "sparse":
                    row["sampling_fps"] = 4
                if case == "owner":
                    fs[1]["owner_epochs"]["selected"] = 1
                if case == "body":
                    body[1]["binding_visible"] = False
                if case == "target":
                    row["target_binding"]["regions"].pop()
                if case == "ordering":
                    row["visual_evidence"][0]["frame_ids"] = [fs[-1]["id"]]
                    row["visual_evidence"][1]["frame_ids"] = [fs[0]["id"]]
                accepted, _ = accepted_events([row], fs, body)
                self.assertNotIn("ability_hit", [r["kind"] for r in accepted])

    def test_native_pts_quantization_is_allowed_but_sparse_chain_is_not(self):
        fs = frames(1, 1.25)
        fs[1]["time"] += .005
        self.assertIn("ability_hit", [r["kind"] for r in accepted_events([event(fs)], fs, body_rows(fs))[0]])
        fs[1]["time"] += .2
        self.assertNotIn("ability_hit", [r["kind"] for r in accepted_events([event(fs)], fs, body_rows(fs))[0]])
        native = frames(1, 1.2, fps=60)
        skipped = event(native[::2])
        skipped["sampling_fps"] = 60
        self.assertNotIn("ability_hit", [r["kind"] for r in accepted_events([skipped], native, body_rows(native))[0]])

    def test_fragment_assembly_and_distinct_rapid_actions(self):
        fs = frames(1, 1.5)
        a, b = event(fs[:5]), event(fs[4:])
        a["visual_evidence"] = [c for c in a["visual_evidence"] if c["criterion"] != "target_impact_visible"]
        b["visual_evidence"] = [c for c in b["visual_evidence"] if c["criterion"] != "ability_cast_visible"]
        a["request_ids"], b["request_ids"] = ["one"], ["two"]
        a["id"], b["id"] = "a", "b"
        merged = assemble_fragments([a, b])
        self.assertEqual(len(merged), 1)
        self.assertIn("ability_hit", [r["kind"] for r in accepted_events(merged, fs, body_rows(fs))[0]])
        a, b = event(fs[:4], "attack"), event(fs[4:], "attack")
        b["id"] = "b"
        self.assertEqual(len(deduplicate_events([a, b])[0]), 2)
        self.assertEqual(len(deduplicate_events([a, {**a, "id": "duplicate"}])[0]), 1)

    def test_competing_target_attributions_abstain(self):
        fs = frames(1, 1.25)
        a = event(fs, "targeted_attack")
        b = copy.deepcopy(a)
        b.update(id="other", target="champion:other")
        b["target_binding"].update(entity=b["target"], binding=b["target"])
        accepted, omissions = deduplicate_events([a, b])
        self.assertEqual(accepted, [])
        self.assertEqual(len(omissions), 2)

    def test_generic_action_cannot_attribute_a_target_without_direction_evidence(self):
        fs = frames(1, 1.25)
        row = event(fs, "targeted_attack")
        row["kind"] = "attack"
        accepted, omissions = accepted_events([row], fs, body_rows(fs))
        self.assertEqual(accepted, [])
        self.assertEqual(omissions[0]["reason"], "unqualified_target_attribution")


class TimelineTests(unittest.TestCase):
    def test_clock_ticks_at_16fps_are_not_pauses_and_masks_hide_facecam(self):
        tracker = PlaybackTracker("s", 16, scene_box=[0, 0, 64, 36], masks=[[0, 0, 16, 36]])
        for i in range(48):
            pixels = np.full((36, 64, 3), 100, dtype=np.uint8)
            pixels[:, :16] = i*5 % 255
            reading = tracker.update(i/16, i//16, Image.fromarray(pixels))
            self.assertEqual(reading["playback_state"], "playing")
        still = Image.new("RGB", (64, 36), (100, 100, 100))
        for i in range(48, 96):
            reading = tracker.update(i/16, 2, still)
        self.assertEqual(reading["playback_state"], "paused")
        no_image = PlaybackTracker("s")
        for i in range(5):
            self.assertNotEqual(no_image.update(i, 10)["playback_state"], "paused")

    def test_context_padding_grouping_and_unknown_result(self):
        fs = frames(0, 10)
        clips = segment_captions([caption(3), caption(4)], fs, [0, 10], "s")
        self.assertEqual(len(clips), 1)
        self.assertEqual((clips[0]["start"], clips[0]["end"]), (1, 7.1))
        self.assertEqual(clips[0]["text"], "Claim at 3. Claim at 4.")
        last = segment_captions([caption(9.9, end=9.9375)], fs, [0, 10], "s")[0]
        self.assertTrue(last["right_censored"])
        self.assertEqual(last["result_status"], "unknown")

    def test_hud_readings_do_not_erase_physical_action_setup_and_result(self):
        fs = frames(0, 10)
        clips = segment_captions([caption(1, end=1, kind="displayed_state"),
                                  caption(3), caption(5, end=5, kind="cs_increased")], fs, [0, 10], "s")
        self.assertEqual(len(clips), 1)
        clip = clips[0]
        self.assertEqual((clip["start"], clip["end"]), (1, 6.1))
        self.assertEqual(clip["setup_status"], "observed_context")
        self.assertEqual(clip["result_status"], "observed_context")
        self.assertFalse(clip["left_censored"])
        self.assertEqual(clip["phases"]["action"], caption(3)["evidence_refs"])
        self.assertEqual(clip["caption_ids"], ["caption-1", "caption-3", "caption-5"])

    def test_clip_cap_continuations_owner_change_and_pause_barriers(self):
        fs = frames(0, 70)
        clips = segment_captions([caption(i) for i in range(3, 66)], fs, [0, 70], "s")
        self.assertGreater(len(clips), 2)
        self.assertTrue(all(c["end"]-c["start"] <= 30 for c in clips))
        self.assertEqual(clips[0]["next_clip_id"], clips[1]["id"])
        ids = [key for c in clips for key in c["caption_ids"]]
        self.assertEqual(len(ids), len(set(ids)))
        fs = frames(0, 6)
        for f in fs:
            if f["time"] >= 3:
                f["owner_epochs"]["selected"] = 1
        clips = segment_captions([caption(2.5), caption(3.2, epoch=1)], fs, [0, 6], "s")
        self.assertEqual(len(clips), 2)

        self.assertLessEqual(clips[0]["end"], 3)
        self.assertGreaterEqual(clips[1]["start"], 3)
        fs = frames(0, 6)
        for f in fs:
            if 2.5 <= f["time"] <= 3:
                f["playback"]["playback_state"] = "paused"
        clips = segment_captions([caption(2), caption(3.5)], fs, [0, 6], "s")
        self.assertEqual(len(clips), 2)

    def test_native_pts_padding_does_not_cross_a_cut(self):
        fs = frames(0, .5)
        fs[1]["time"], fs[2]["time"] = .0667, .1167
        for f in fs[2:]:
            f["segment_id"] = "cut"
        groups = contexts(fs, [0, .5])
        self.assertLessEqual(groups[0]["end"], groups[1]["start"])

    def test_repeat_requires_moving_multi_frame_correspondence(self):
        fs = frames(0, 2)+frames(5, 7, segment="replay")
        for f in fs:
            value = round((f["time"] % 5)*100)
            f["playback"]["scene_signature"] = bytes([value]*576).hex()
        clips = [{"id": "one", "start": 0, "end": 2, "playback_segment_id": "seg", "repeat_group_id": None},
                 {"id": "two", "start": 5, "end": 7, "playback_segment_id": "replay", "repeat_group_id": None}]
        linked = link_repeats(copy.deepcopy(clips), fs)
        self.assertEqual(linked[1]["earlier_occurrence_id"], "one")
        self.assertEqual(linked[0]["repeat_group_id"], linked[1]["repeat_group_id"])
        for f in fs:
            f["playback"]["scene_signature"] = bytes([100]*576).hex()
        self.assertIsNone(link_repeats(copy.deepcopy(clips), fs)[1].get("earlier_occurrence_id"))


class PilotIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(Path(self.tmp.name))
        self.root = self.store.path("experiments/captions/"+"a"*20)
        self.root.mkdir(parents=True)
        self.config = Config(routes=[], patch={}, mechanics=False, evaluation_enabled=False, asset_cache=str(self.store.path("assets")))
        self.fs = frames(0, 6)
        path = self.root/"input.png"
        Image.new("RGB", (960, 600), "gray").save(path)
        for f in self.fs:
            f.update(input_path=str(path), native_path=str(path), input_sha256=file_hash(path), native_sha256=file_hash(path))
        from annotation.league.action_v3 import load_profile
        self.profile = load_profile()
        self.audit = {"schema_version": 3, "root": str(self.root), "frames": self.fs, "interval": [0, 6],
            "source_sha256": "source", "source": {"path": "fixture.mp4", "fps": 60, "sha256": "source"},
            "profile": self.profile, "policy": POLICY}
        atomic_json(self.root/"input-audit.json", self.audit)
        atomic_json(self.store.path("manifest.json"), {"source": self.audit["source"], "interval": [0, 1732]})
        self.store.write("events", [{"id": "production"}])
        self.production_events = self.store.path("events.jsonl").read_bytes()
        self.body = body_rows(self.fs)

    def detection(self):
        identities = [{"id": f["id"], "frame_id": f["id"], "time": f["time"], "owner_id": "selected", "hud_champion": "Aatrox"} for f in self.fs]
        write_rows(self.root/"identities.jsonl", identities)
        write_rows(self.root/"body_readings.jsonl", self.body)
        write_rows(self.root/"normalized_observations.jsonl", [])
        return identities, self.body

    def runner(self, failure=False):
        runner = Mock(identity={"model": "pinned-fixture"}, peak_memory_bytes=123)
        def complete(messages, **kw):
            context = json.loads('{"frames":'+messages[0]["content"][0]["text"].split('{"frames":', 1)[1])
            fs = context["frames"]
            near = [f for f in fs if 2 <= f["time"] <= 2.25]
            if failure and kw.get("schema") and len(fs) > 1 and fs[1]["time"]-fs[0]["time"] < .1:
                raise RuntimeError("fixture GPU failure")
            actions = []
            if len(near) >= 2:
                actions = [{"kind": "cast", "actor_track": near[0]["player_track"], "target_id": None, "slot_id": "Q",
                            "start_frame": near[0]["frame_id"], "end_frame": near[-1]["frame_id"],
                            "evidence": [{"criterion": "cast_visible", "frame_ids": [f["frame_id"] for f in near]}]}]
            return {"finish_reason": "stop", "text": json.dumps({"targets": [], "actions": actions, "uncertainty": []}),
                    "usage": {"completion_tokens": 10}, "peak_memory_bytes": 123, "wall_seconds": .1}
        runner.complete.side_effect = complete
        return runner

    def run_pilot(self, runner=None, infer=False):
        with patch("annotation.league.caption_pilot.prepare_inputs", return_value=self.audit), patch(
                "annotation.league.caption_pilot._cpu_detection", side_effect=lambda *args: self.detection()):
            return caption_pilot(self.store, self.config, 0, 6, infer=infer, runner=runner)

    def test_cpu_default_reports_missing_scene_and_does_not_load_model(self):
        with patch("annotation.league.caption_pilot.LocalVLRunner", side_effect=AssertionError("GPU loaded")):
            report = self.run_pilot()
        self.assertEqual(report["status"], "completed_experimental")
        self.assertTrue(report["scene_uncertainty"])
        self.assertEqual(self.store.path("events.jsonl").read_bytes(), self.production_events)
        self.assertTrue((self.root/"caption_clips.jsonl").exists())

    def test_unavailable_layout_has_explicit_unknown_identity_rows(self):
        self.audit["frames"] = self.fs[:4]
        self.fs[0]["layout_valid"] = False
        icon = Mock()
        icon.read_frames.return_value = [{"id": "known-kit", "frame_id": self.fs[1]["id"], "time": self.fs[1]["time"],
            "owner_id": "selected", "owner_epoch": 0, "segment_id": "seg", "hud_champion": "Aatrox"}]
        with patch("annotation.league.caption_pilot.read_counters", return_value=[]):
            identities, body = _cpu_detection(self.store, self.config, self.root, self.audit, icon)
        self.assertEqual(len(identities), 4)
        self.assertIsNone(identities[0]["hud_champion"])
        self.assertEqual(identities[0]["reason"], "unsupported_layout")
        self.assertFalse(body[0]["binding_visible"])

    def test_duplicate_point_readings_are_omitted_without_extending_visibility(self):
        self.detection()
        self.fs[0]["playback"]["continuity_verified"] = False
        rows = [{"id": f["id"], "frame_id": f["id"], "time": f["time"], "owner_id": "selected", "owner_epoch": 0,
                 "segment_id": "seg", "field": "cs", "value": 10 if i < 4 else 11, "layout_valid": True}
                for i, f in enumerate(self.fs[:8])]
        write_rows(self.root/"normalized_observations.jsonl", counter_observations(rows, self.fs, body=self.body))
        regenerate(self.root)
        captions = read_rows(self.root/"captions.jsonl")
        self.assertEqual([c["claims"][0]["kind"] for c in captions], ["displayed_state", "cs_increased"])
        self.assertEqual((captions[0]["start"], captions[0]["end"]), (0, 0))
        self.assertIn("duplicate_display_reading", {r["reason"] for r in read_rows(self.root/"caption_omissions.jsonl")})

    def test_inference_resume_review_corrections_and_independent_assessment(self):
        runner = self.runner()
        report = self.run_pilot(runner, infer=True)
        self.assertGreater(report["coverage"]["generated"], 0)
        original = (self.root/"captions.jsonl").read_bytes()
        count = runner.complete.call_count
        self.run_pilot(runner, infer=True)
        self.assertEqual(count, runner.complete.call_count)
        self.assertEqual(original, (self.root/"captions.jsonl").read_bytes())
        report = self.run_pilot()
        self.assertEqual(original, (self.root/"captions.jsonl").read_bytes())
        self.assertGreater(report["resource_metrics"]["cache_reused"], 0)
        from fastapi.testclient import TestClient
        client = TestClient(create_app(self.store))
        self.assertEqual(client.get('/captions').status_code, 200)
        self.assertEqual(client.get('/api/caption-pilots/'+'a'*20).status_code, 200)
        c = read_rows(self.root/"captions.jsonl")[0]
        assessment = {"caption_id": c["id"], "reviewer": "independent-fixture", "independent_of_generation": True,
                      "supported": True, "actor_correct": True, "target_correct": True, "evidence_valid": True}
        self.assertEqual(client.post('/api/caption-pilots/'+'a'*20+'/assessments', json=assessment).status_code, 200)
        proposals = [r for r in read_rows(self.root/"scene_proposals.jsonl") if r["sampling_fps"] >= 16]
        for proposal in proposals:
            correct_pilot(self.root, "scene_proposals", proposal["id"], {}, "rejected", "fixture-reviewer")
        self.assertEqual(read_rows(self.root/"captions.jsonl"), [])
        self.assertFalse(self.store.path("corrections").exists())
        self.assertEqual(len(self.store.rows("events")), 1)
        self.assertEqual(client.post('/api/caption-pilots/'+'a'*20+'/corrections', json={}, headers={'Origin':'https://outside.invalid'}).status_code, 403)

    def test_gpu_failure_preserves_cpu_and_completed_requests(self):
        with self.assertRaisesRegex(RuntimeError, "GPU failure"):
            self.run_pilot(self.runner(failure=True), infer=True)
        report = read_json(self.root/"report.json")
        self.assertEqual(report["status"], "failed_experimental")
        self.assertTrue((self.root/"caption_coverage.json").exists())
        self.assertTrue(read_rows(self.root/"scene_proposals.jsonl"))
        self.assertTrue(list((self.root/"requests").glob('*/response.json')))

    def test_resume_excludes_unvisited_foreign_request_proposals(self):
        runner = self.runner()
        self.run_pilot(runner, infer=True)
        foreign = event(self.fs[64:68], "cast")
        foreign["id"] = "foreign-runtime-proposal"
        atomic_json(self.root/"requests/foreign/parsed.json", {"proposals": [foreign], "uncertainty": []})
        self.run_pilot(runner, infer=True)
        self.assertNotIn(foreign["id"], {p["id"] for p in read_rows(self.root/"scene_proposals.jsonl")})

    def test_compatible_request_cache_is_reused_after_derived_root_changes(self):
        from annotation.league.local_models import LOCK
        pin = read_json(LOCK)["models"]["intake"]
        self.audit["cache_identity"] = {"model": pin}
        atomic_json(self.root/"input-audit.json", self.audit)
        runner = self.runner()
        runner.identity = {"checkpoint": {"specification": pin}, "runtime": {"gpu": "fixture"}}
        self.run_pilot(runner, infer=True)
        self.root = self.root.parent/("b"*20)
        self.root.mkdir()
        self.audit["root"] = str(self.root)
        atomic_json(self.root/"input-audit.json", self.audit)
        with patch("annotation.league.caption_pilot.LocalVLRunner", side_effect=AssertionError("GPU loaded")):
            report = self.run_pilot()
        self.assertGreater(report["resource_metrics"]["cache_reused"], 0)
        self.assertGreater(report["coverage"]["generated"], 0)

    def test_empty_dense_scene_does_not_skip_native_review_of_counter_bracket(self):
        rows = [{"id": f["id"], "frame_id": f["id"], "time": f["time"], "owner_id": "selected", "owner_epoch": 0,
                 "segment_id": "seg", "field": "cs", "value": 10 if i < 2 else 11, "layout_valid": True}
                for i, f in enumerate(self.fs[32:36])]
        write_rows(self.root/"counter_readings.jsonl", rows)
        runner = Mock(identity={"model": "fixture"}, peak_memory_bytes=0)
        runner.complete.return_value = {"finish_reason": "stop", "text": json.dumps({"targets": [], "actions": [], "uncertainty": ["unresolved"]})}
        with patch("annotation.league.caption_pilot.native_review", return_value=[]) as native:
            self.run_pilot(runner, infer=True)
        self.assertEqual(native.call_count, 1)
        self.assertIn("native", [r["stage"] for r in read_rows(self.root/"resampling.jsonl")])
        self.assertEqual(read_rows(self.root/"captions.jsonl"), [])

    def test_benchmark_scores_references_and_requires_current_independent_assessments(self):
        self.run_pilot(self.runner(), infer=True)
        c = read_rows(self.root/"captions.jsonl")[0]
        clip = {"claims": [{"id": "reference-cast", "kind": "cast", "start": c["start"], "end": c["end"], "values": {"ability": "Q"}}],
                "bindings": [{"claim_id": "reference-cast", "role": "actor", "time": c["start"], "box": [200, 200, 300, 400]}],
                "phases": [{"phase": "action", "start": c["start"], "end": c["end"]}], "playback": [], "repeats": [], "case_tags": []}
        result = score_clip(clip, self.root)
        self.assertEqual(result["by_kind"]["cast"]["matched"], 1)
        self.assertEqual(result["by_kind"]["cast"]["binding_correct"], 1)
        self.assertEqual(result["by_kind"]["cast"]["reviewed"], 0)
        assess_caption(self.root, {"caption_id": c["id"], "reviewer": "independent-fixture", "independent_of_generation": True,
            "supported": True, "actor_correct": True, "target_correct": True, "evidence_valid": True})
        self.assertEqual(score_clip(clip, self.root)["by_kind"]["cast"]["supported"], 1)
        row = next(p for p in read_rows(self.root/"scene_proposals.jsonl") if p["sampling_fps"] >= 16)
        correct_pilot(self.root, "scene_proposals", row["id"], {}, "rejected", "fixture-reviewer")
        self.assertEqual(score_clip(clip, self.root)["by_kind"]["cast"]["reviewed"], 0)

    def test_rejected_chain_fragments_cannot_be_resurrected(self):
        fs = frames(1, 1.5)
        a, b = event(fs[:5]), event(fs[4:])
        a["visual_evidence"] = [c for c in a["visual_evidence"] if c["criterion"] != "target_impact_visible"]
        b["visual_evidence"] = [c for c in b["visual_evidence"] if c["criterion"] != "ability_cast_visible"]
        a["request_ids"], b["request_ids"] = ["one"], ["two"]
        b.update(id="rejected", review_status="rejected")
        accepted, _ = accepted_events([a, b], fs, body_rows(fs))
        self.assertNotIn("ability_hit", [r["kind"] for r in accepted])

    def test_scene_parser_rejects_target_outside_scene_and_actor_override(self):
        reader = CaptionSceneReader(self.store, self.root, self.profile, POLICY, self.body, [], None)
        a = {"kind": "attack", "actor_track": "other", "target_id": None, "slot_id": None,
             "start_frame": "f0", "end_frame": "f1", "evidence": [{"criterion": "attack_visible", "frame_ids": ["f0", "f1"]}]}
        value = {"actions": [a], "targets": [], "uncertainty": []}
        with self.assertRaisesRegex(ValueError, "independent"):
            reader.validate({"finish_reason": "stop", "text": json.dumps(value)}, self.fs[:2], "req")
        value["actions"] = []
        value["targets"] = [{"local_id": "target", "kind": "champion", "regions": [{"frame_id": "f0", "box": [0, 0, 20, 20]}]}]
        with self.assertRaisesRegex(ValueError, "outside"):
            reader.validate({"finish_reason": "stop", "text": json.dumps(value)}, self.fs[:2], "req")

    def test_cli_and_benchmark_freeze_guards(self):
        args = parser().parse_args(["caption-pilot", "--start", "1", "--end", "60"])
        self.assertFalse(args.infer)
        directory = self.store.path("benchmarks/captions")
        initialize_labels(self.store, directory, self.profile)
        data = read_json(directory/"captions.json")
        self.assertEqual(len(validate_labels(self.store, data, self.profile)), 20)
        with self.assertRaisesRegex(ValueError, "Independent"):
            experiment_captions(self.store, self.config, freeze=True)
        for c in data["clips"]:
            if c["split"] == "development":
                c.update(verified=True, inspected=True, explicit_negative=True, labeler="fixture", independent_of_predictions=True)
        with self.assertRaisesRegex(ValueError, "category negatives"):
            validate_labels(self.store, data, self.profile, required_splits=("development",))
        for c in data["clips"]:
            c["negative_annotations"] = {k: True for k in ("claims", "bindings", "phases", "playback", "repeats")}
        atomic_json(directory/"captions.json", data)
        self.assertEqual(experiment_captions(self.store, self.config, freeze=True)["status"], "development_frozen")
        data["clips"][0]["case_tags"] = ["quiet"]
        atomic_json(directory/"captions.json", data)
        with self.assertRaisesRegex(ValueError, "already frozen"):
            experiment_captions(self.store, self.config, freeze=True)
        self.assertFalse(aggregate([], data["clips"])["passed"])
        from fastapi.testclient import TestClient
        client = TestClient(create_app(self.store))
        self.assertEqual(client.get('/caption-labels').status_code, 200)
        self.assertNotIn('/api/caption-pilots', client.get('/caption-labels').text)
        self.assertEqual(client.post('/api/caption-labels/'+data['clips'][0]['id'], json={"labeler":"fixture", "independent_of_predictions":True,"annotations":{}}).status_code, 422)


if __name__ == "__main__":
    unittest.main()
