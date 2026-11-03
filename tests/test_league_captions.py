from __future__ import annotations

import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest

from annotation.league.__main__ import ensure_captions, main
from annotation.league.captions import DEPENDENCIES, construct_captions
from annotation.league.core import Config, Route, Store, atomic_json, read_json
from annotation.league.review import add_correction, create_app
from annotation.league.timeline import fuse_events, fuse_observations, normalize_output


def observation(key, time, value, **changes):
    return {"schema_version": 2, "id": key, "task_id": "task", "frame_id": "frame_" + key,
            "time": time, "route": "selected_hud", "slot_id": "selected", "binding": "selected",
            "entity": "Aatrox", "field": "cs", "value": value, "visibility": "visible",
            "acceptance": "accepted", "confidence": 0, "review_status": "unreviewed",
            "playback_segment_id": "segment", "continuity_verified": True, **changes}


def grounded_event(kind="ability_hit"):
    criteria = ({"ability_cast_visible": ["f1"], "target_impact_visible": ["f2"],
                 "cast_linked_to_impact": ["f1", "f2"]} if kind == "ability_hit" else
                {"minion_death_visible": ["f2"], "actor_action_linked_to_death": ["f1", "f2"]})
    target = "Jax" if kind == "ability_hit" else "minion:17"
    return {"id": "event", "kind": kind, "actor": "Aatrox", "target": target,
            "ability": "Q", "start": 1, "end": 1.0625, "sampling_fps": 16,
            "confidence": .9, "review_status": "unreviewed", "evidence_ids": ["f1", "f2"],
            "routes": ["gameplay"], "task_ids": ["task"], "criteria": sorted(criteria),
            "actor_binding": {"entity": "Aatrox", "binding": "player:1", "frame_ids": ["f1"]},
            "target_binding": {"entity": target, "binding": "unit:17", "frame_ids": ["f2"]},
            "visual_evidence": [{"criterion": name, "frame_ids": ids} for name, ids in criteria.items()]}


def construct(observations=(), events=(), **kwargs):
    states, _ = fuse_observations(list(observations))
    return construct_captions(states, events, observations, interval=[0, 120], source_sha256="source",
                              frame_times=kwargs.pop("frame_times", {"f1": 1, "f2": 1.0625}), **kwargs)


class ClaimTests(unittest.TestCase):
    def test_cs_delta_never_becomes_last_hit(self):
        captions, _ = construct([observation("a", 1, 10), observation("b", 2, 13)])
        self.assertEqual([c["claims"][0]["kind"] for c in captions], ["displayed_state", "cs_increased"])
        delta = captions[-1]
        self.assertEqual(delta["claims"][0]["values"]["delta"], 3)
        self.assertEqual((delta["start"], delta["end"]), (1, 2))
        self.assertNotIn("last hit", delta["text"])
        self.assertEqual(delta["source_sha256"], "source")
        self.assertEqual({r["id"] for r in delta["evidence_refs"] if r["table"] == "frames"}, {"frame_a", "frame_b"})

    def test_slot_only_readings_do_not_invent_a_champion_or_owner_continuity(self):
        captions, _ = construct([observation("a", 1, 10, entity="unknown"),
                                  observation("b", 2, 11, entity="unknown")])
        self.assertEqual(len(captions), 2)
        self.assertTrue(all(c["claims"][0]["kind"] == "displayed_state" for c in captions))
        self.assertIn("HUD slot selected", captions[0]["text"])
        self.assertIsNone(captions[0]["actor_bindings"][0]["entity"])

    def test_cs_delta_requires_identity_continuity_and_no_intervening_uncertainty(self):
        for change in ({"playback_segment_id": "other"}, {"continuity_verified": False},
                       {"entity": "Jax"}, {"playback_segment_id": None}):
            with self.subTest(change=change):
                captions, _ = construct([observation("a", 1, 10), observation("b", 2, 11, **change)])
                self.assertFalse(any(c["claims"][0]["kind"] == "cs_increased" for c in captions))
        captions, _ = construct([observation("a", 1, 10),
                                  observation("u", 1.5, None, visibility="unknown", acceptance="unknown"),
                                  observation("b", 2, 11)])
        self.assertFalse(any(c["claims"][0]["kind"] == "cs_increased" for c in captions))

    def test_same_value_is_not_continuous_visibility_and_resets_at_a_cut(self):
        captions, omissions = construct([observation("a", 1, 10), observation("b", 2, 10),
                                          observation("c", 3, 10, playback_segment_id="cut")])
        self.assertEqual([(c["start"], c["end"]) for c in captions], [(1, 1), (3, 3)])
        self.assertIn("unchanged_reading", [o["reason"] for o in omissions])

    def test_counter_decrease_is_a_reading_not_a_negative_farm_event(self):
        captions, _ = construct([observation("a", 1, 10), observation("b", 2, 9)])
        self.assertTrue(all(c["claims"][0]["kind"] == "displayed_state" for c in captions))

    def test_gold_changes_preserve_precision_without_inventing_a_reward(self):
        captions, _ = construct([observation("a", 1, {"displayed_text": "1.2k", "amount": 1200}, field="displayed_gold"),
                                  observation("b", 2, {"displayed_text": "1.3k", "amount": 1300}, field="displayed_gold")])
        self.assertEqual(len(captions), 2)
        self.assertIn('"1.3k"', captions[-1]["text"])
        self.assertNotIn("reward", captions[-1]["text"])

    def test_experimental_audio_rejected_conflicted_and_low_acceptance_are_omitted(self):
        for changes in ({"experimental": True}, {"evidence_modality": "audio"},
                        {"review_status": "rejected"}, {"acceptance": "unknown"}):
            with self.subTest(changes=changes):
                self.assertEqual(construct([observation("a", 1, 10, **changes)])[0], [])
        self.assertEqual(construct([observation("a", 1, 10), observation("b", 1, 11)])[0], [])

    def test_overlay_and_identity_issues_block_affected_claims_and_comparisons(self):
        rows = [observation("a", 1, 10), observation("b", 3, 11)]
        captions, _ = construct(rows, issues=[{"kind": "layout_mismatch", "time": 2}])
        self.assertTrue(all(c["claims"][0]["kind"] == "displayed_state" for c in captions))
        self.assertEqual(construct(rows, issues=[{"kind": "editorial_overlay", "start": 0, "end": 4}])[0], [])

    def test_hit_and_last_hit_have_separate_grounding_contracts(self):
        for kind in ("ability_hit", "minion_last_hit"):
            with self.subTest(kind=kind):
                captions, omissions = construct(events=[grounded_event(kind)])
                self.assertFalse(omissions)
                claim = captions[0]["claims"][0]
                self.assertEqual(claim["kind"], kind)
                self.assertEqual(len(captions[0]["actor_bindings"]), 2)
                self.assertIsNotNone(claim["target_binding_id"])
                self.assertGreaterEqual(len(claim["criteria"]), 2)

    def test_hit_requires_impact_binding_temporal_evidence_and_named_ability(self):
        for changes in ({"visual_evidence": []}, {"target_binding": None}, {"actor_binding": None},
                        {"sampling_fps": 4}, {"ability": None}, {"experimental": True},
                        {"confidence": .2}, {"evidence_ids": ["invented"]},
                        {"target_binding": {"entity": "other", "binding": "t", "frame_ids": ["f2"]}}):
            with self.subTest(changes=changes):
                captions, omissions = construct(events=[{**grounded_event(), **changes}])
                self.assertEqual(captions, [])
                self.assertEqual(len(omissions), 1)

    def test_criterion_names_and_high_confidence_alone_are_not_evidence(self):
        event = grounded_event("minion_last_hit")
        del event["visual_evidence"]
        self.assertEqual(construct(events=[event])[0], [])

    def test_declared_fps_cannot_replace_dense_frames_or_outcome_order(self):
        event = {**grounded_event(), "end": 2}
        captions, omissions = construct(events=[event], frame_times={"f1": 1, "f2": 2})
        self.assertFalse(captions)
        self.assertEqual(omissions[0]["reason"], "insufficient_temporal_evidence")
        event = grounded_event()
        event["visual_evidence"][0]["frame_ids"] = ["f2"]
        event["visual_evidence"][1]["frame_ids"] = ["f1"]
        captions, omissions = construct(events=[event])
        self.assertFalse(captions)
        self.assertEqual(omissions[0]["reason"], "outcome_precedes_action")

    def test_minion_last_hit_cannot_target_a_champion(self):
        event = grounded_event("minion_last_hit")
        event.update(target="Jax", target_binding={"entity": "Jax", "binding": "target:1", "frame_ids": ["f2"]})
        self.assertFalse(construct(events=[event])[0])

    def test_current_intake_health_clock_notice_and_bar_shapes(self):
        readings = [("health", {"displayed_text": "450/900", "current": 450, "maximum": 900}),
                    ("game_clock", {"displayed_text": "12:34", "seconds": 754}),
                    ("notice", "An ally has been slain"), ("health_fraction", .5)]
        for field, value in readings:
            with self.subTest(field=field):
                captions, _ = construct([observation("a", 1, value, field=field)])
                self.assertEqual(len(captions), 1)
                self.assertEqual(captions[0]["schema_version"], 1)
                self.assertEqual(captions[0]["claims"][0]["values"]["value"], value)

    def test_cast_does_not_establish_hit_and_recall_does_not_establish_completion(self):
        for kind, cue in (("cast", "cast_visible"), ("recall", "recall_channel_visible")):
            event = {**grounded_event(), "kind": kind, "target": None,
                     "visual_evidence": [{"criterion": cue, "frame_ids": ["f1", "f2"]}]}
            captions, _ = construct(events=[event])
            self.assertEqual(captions[0]["claims"][0]["kind"], kind)
            self.assertNotIn("hitting", captions[0]["text"])
            self.assertNotIn("completed", captions[0]["text"])

    def test_missing_observation_lineage_is_not_silently_accepted(self):
        rows = [observation("a", 1, 10)]
        states, _ = fuse_observations(rows)
        captions, omissions = construct_captions(states, [], [], interval=[0, 120], source_sha256="source")
        self.assertFalse(captions)
        self.assertEqual(omissions[0]["reason"], "missing_observation_evidence")


class StageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.store = Store(Path(self.temporary.name))
        self.config = Config({}, [Route("gameplay", [0, 0, 1920, 1080], "thinking", [])])
        atomic_json(self.store.path("manifest.json"), {"source": {"sha256": "source"}, "interval": [0, 120]})
        atomic_json(self.store.path("config.json"), self.config.to_dict())
        self.store.write("observations", [observation("a", 1, 10), observation("b", 2, 12)])
        self.store.write("event_candidates", [])
        self.store.commit_stage("annotate", self.store.fingerprint(self.config))

    def test_cli_generates_timestamped_outputs_without_a_vlm(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = main(["--run-dir", str(self.store.root), "generate-captions"])
        self.assertEqual(status, 0, output.getvalue())
        self.assertEqual(len(self.store.rows("captions")), 2)
        summary = read_json(self.store.path("caption_coverage.json"))
        self.assertEqual(summary["vlm_output_tokens"], 0)
        self.assertEqual(summary["counts"]["cs_increased"], 1)
        self.assertTrue(self.store.path("stages/generate-captions.json").exists())

    def test_corrections_regenerate_captions_with_stable_ids_for_identical_inputs(self):
        ensure_captions(self.store, self.config)
        original = self.store.path("captions.jsonl").read_bytes()
        ensure_captions(self.store, self.config)
        self.assertEqual(original, self.store.path("captions.jsonl").read_bytes())
        key = self.store.stage_key("generate-captions", self.config, DEPENDENCIES)
        add_correction(self.store, "observations", "b", {"value": 11}, "human-verified")
        self.assertIn("generate-captions", read_json(self.store.path("invalidated.json"))["stages"])
        ensure_captions(self.store, self.config)
        self.assertNotEqual(key, self.store.stage_key("generate-captions", self.config, DEPENDENCIES))
        self.assertEqual(self.store.rows("captions")[-1]["claims"][0]["values"]["delta"], 1)

    def test_missing_output_is_rebuilt_even_with_current_stage_marker(self):
        ensure_captions(self.store, self.config)
        self.store.path("captions.jsonl").unlink()
        ensure_captions(self.store, self.config)
        self.assertEqual(len(self.store.rows("captions")), 2)

    def test_qa_reviews_do_not_invalidate_caption_stage(self):
        ensure_captions(self.store, self.config)
        key = self.store.stage_key("generate-captions", self.config, DEPENDENCIES)
        atomic_json(self.store.path("corrections/00000001.json"), {"target": "questions", "record_id": "q",
                    "changes": {"answer": 2}, "review_status": "human-verified"})
        self.assertEqual(key, self.store.stage_key("generate-captions", self.config, DEPENDENCIES))

    def test_review_exposes_caption_evidence(self):
        from fastapi.testclient import TestClient
        ensure_captions(self.store, self.config)
        self.store.write("previews", [{"id": "frame_a", "time": 1}, {"id": "frame_b", "time": 2}])
        with TestClient(create_app(self.store)) as client:
            response = client.get("/api/records?target=captions")
            self.assertEqual(response.status_code, 200)
            caption = response.json()["records"][-1]
            evidence = client.get("/api/evidence/" + caption["id"]).json()
            self.assertEqual({f["id"] for f in evidence["frames"]}, {"frame_a", "frame_b"})

    def test_review_cannot_promote_a_cast_to_a_hit_without_impact(self):
        event = grounded_event()
        self.store.write("events", [event])
        self.store.write("previews", [{"id": "f1", "time": 1}, {"id": "f2", "time": 1.0625}])
        with self.assertRaisesRegex(ValueError, "missing_visual_criteria"):
            add_correction(self.store, "events", event["id"], {"visual_evidence": []}, "human-verified")
        add_correction(self.store, "events", event["id"], {"ability": "Q2"}, "human-verified")
        self.assertEqual(self.store.effective(self.store.rows("events"), "events")[0]["ability"], "Q2")

    def test_review_can_reject_an_event_with_incomplete_grounding(self):
        event = {**grounded_event(), "visual_evidence": []}
        self.store.write("events", [event])
        add_correction(self.store, "events", event["id"], {}, "rejected")
        self.assertEqual(self.store.effective(self.store.rows("events"), "events")[0]["review_status"], "rejected")


class NormalizationTests(unittest.TestCase):
    def test_rich_events_survive_normalization_and_fusion(self):
        event = grounded_event()
        event.update(start_frame="f1", end_frame="f2")
        task = {"id": "t", "model": {}, "fps": 16, "pass": 0,
                "frames": [{"id": f, "time": t, "source_id": f} for f, t in (("f1", 1), ("f2", 1.0625))]}
        route = Route("gameplay", [0, 0, 1920, 1080], "thinking", [])
        _, events, issues = normalize_output({"events": [event]}, task, route)
        self.assertFalse(issues)
        fused = fuse_events(events)
        self.assertEqual(fused[0]["actor_binding"], event["actor_binding"])
        captions, _ = construct(events=fused)
        self.assertEqual(captions[0]["claims"][0]["kind"], "ability_hit")
        event["visual_evidence"] = []
        _, events, issues = normalize_output({"events": [event]}, task, route)
        self.assertFalse(events)
        self.assertTrue(issues)

    def test_distinct_binding_chains_do_not_merge(self):
        first = grounded_event()
        second = copy.deepcopy(first)
        second["id"] = "other"
        second["actor_binding"]["binding"] = "player:2"
        self.assertEqual(len(fuse_events([first, second])), 2)


if __name__ == "__main__":
    unittest.main()
