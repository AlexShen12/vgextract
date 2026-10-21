from __future__ import annotations

import json
import contextlib
import io
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from annotation.league.core import (CAPABILITIES, Config, Route, StopRequested, Store,
                                    atomic_json, digest, read_json, validate_question)
from annotation.league.evaluation import (evaluate, exact_answer, report, temporal_metrics,
                                          uniform_times)
from annotation.league.inference import InputBudgetError, ScopedResolver, parse_output
from annotation.league.media import Media, prepare
from annotation.league.observers import annotate, transition_reason
from annotation.league.prompts import solver_messages
from annotation.league.questions import construct, generate_qa
from annotation.league.review import add_correction, create_app
from annotation.league.timeline import fuse, fuse_events, fuse_observations, normalize_output

PIN = {"gameplay_patch": "26.13", "ddragon_version": "16.13.1", "cdragon_version": "16.13"}


def config():
    return Config(PIN, [
        Route("scoreboard", [0, 0, 100, 100], "instruct", ["kills", "inventory", "player_name"]),
        Route("minimap", [100, 0, 200, 100], "instruct", ["position", "structure_status"]),
        Route("full_context", [0, 0, 1920, 1080], "thinking", ["identity"])],
        window_seconds=8, overlap_seconds=2, quiet_audit_fraction=1, solver_max_frames=8)


def init_store(path, end=120):
    store = Store(Path(path))
    atomic_json(store.path("manifest.json"), {"source": {"sha256": "fixture", "path": str(Path(path)/"source.mp4"),
        "fps": 60, "duration": end, "width": 1920, "height": 1080}, "interval": [0, end], "model_revisions": {}})
    atomic_json(store.path("layout_validation.json"), {})
    return store


def state(id, time, field, value, entity="Aatrox"):
    return {"id": id, "entity": entity, "field": field, "time": time, "value": value,
            "status": "confirmed", "evidence_ids": ["obs_"+id], "frame_ids": ["frame_"+id],
            "review_status": "unreviewed"}


def event(id, time, kind="recall", actor="Aatrox"):
    return {"id": id, "kind": kind, "actor": actor, "target": None, "object": None,
            "location": "top_lane", "start": time, "end": time+2,
            "evidence_ids": ["frame_"+id], "routes": ["gameplay"], "task_ids": [id],
            "criteria": [], "confidence": .95, "sampling_fps": 4, "review_status": "unreviewed"}


class StaticFixture:
    def roster(self):
        return {"champions": [{"internal_id": "Aatrox", "numeric_id": "266", "name": "Aatrox"},
                               {"internal_id": "Jax", "numeric_id": "24", "name": "Jax"}]}

    def system_context(self):
        return "Patch 26.13. Champions: Aatrox|266|Aatrox, Jax|24|Jax. Static identity only."

    def handle_query(self, request):
        return {"status": "unresolved"}


class FixtureMedia:
    def __init__(self, store, config):
        self.store = store
        self.manifest = read_json(store.path("manifest.json"))

    def frame(self, requested, route):
        self.store.check_stop()
        return {"id": "frame_" + digest([requested, route.name])[:12], "source_id": str(requested),
                "time": requested, "route": route.name, "box": route.box, "path": "fixture.png"}

    def frames(self, start, end, fps, route):
        return [self.frame(start+i/fps, route) for i in range(int((end-start)*fps))]


class FixtureRunner:
    """Only tests inject this runner. The production CLI always uses Qwen."""
    def __init__(self, store, stop_after=None):
        self.store, self.stop_after, self.calls = store, stop_after, []

    def infer(self, messages, frames, route, directory, *, fps, visual_limit=None):
        cost = len(frames)*10
        if visual_limit is not None and cost > visual_limit:
            raise InputBudgetError("fixture review budget exhausted")
        self.calls.append((route.name, frames[0]["time"], fps, str(directory.relative_to(self.store.root))))
        if "retrospective visual-memory" in json.dumps(messages):
            output = {"answer": "Aatrox", "abstain": False,
                      "evidence": [{"start": frames[0]["time"], "end": frames[0]["time"], "claim": "visible"}]}
        else:
            observations = []
            for frame in frames:
                fields = {"kills": int(frame["time"]//20), "inventory": ["6610"],
                          "position": {"region": "top_lane"}, "structure_status": "standing",
                          "player_name": "한글플레이어", "identity": "Aatrox"}
                observations += [{"frame_id": frame["id"], "entity": "Aatrox", "field": field,
                                  "value": fields[field], "visibility": "visible", "confidence": .95,
                                  "identity_evidence": "portrait"} for field in route.fields]
            events = []
            if route.name == "full_context":
                events.append({"kind": "recall" if frames[0]["time"] < 60 else "death",
                    "actor": "Aatrox", "start_frame": frames[0]["id"], "end_frame": frames[-1]["id"],
                    "location": "top_lane", "confidence": .95})
            output = {"observations": observations, "events": events, "issues": []}
        metrics = {"visual_tokens": cost, "input_tokens": cost+10, "output_tokens": 20,
                   "peak_gpu_memory_bytes": 1024, "wall_seconds": .01}
        directory.mkdir(parents=True, exist_ok=True)
        atomic_json(directory/"completion.json", {"metrics": metrics, "raw": json.dumps(output)})
        if self.stop_after and len(self.calls) >= self.stop_after:
            self.store.stopping = True
        return {"status": "ok", "output": output, "metrics": metrics}


class FusionTests(unittest.TestCase):
    def test_compact_hud_fields_expand_and_wrong_sidebar_bindings_stay_unknown(self):
        route = Route("right_sidebar", [1810, 125, 1920, 645], "instruct", ["ultimate", "player_name"])
        task = {"id": "t", "model": {}, "fps": 1, "pass": 0,
                "frames": [{"id": "f", "source_id": "s", "time": 1}]}
        compact = {"frame_id": "f", "entity": "Aatrox", "binding": "right:row1",
                   "values": {"ultimate": None, "player_name": "한글"},
                   "field_visibility": {"ultimate": "not_shown"}, "confidence": .9}
        obs, _, issues = normalize_output({"observations": [compact]}, task, route)
        self.assertEqual(len(obs), 2)
        self.assertEqual(obs[0]["visibility"], "not_shown")
        self.assertEqual(obs[1]["value"], "한글")
        obs, _, issues = normalize_output({"observations": [{**compact, "binding": "left:row1"}]}, task, route)
        self.assertTrue(all(o["entity"] == "unknown" for o in obs))
        self.assertTrue(issues)
        with self.assertRaises(ValueError):
            normalize_output({"observations": None}, task, route)

    def test_conflict_occlusion_and_selection_are_not_silent_state(self):
        cfg = config()
        route = Route("selected_hud", [0, 0, 100, 100], "instruct", ["health", "identity"])
        frames = [{"id": "f1", "time": 1, "source_id": "s1"}, {"id": "f2", "time": 2, "source_id": "s2"}]
        task = {"id": "t", "frames": frames, "model": {}, "pass": 0, "fps": 1}
        output = {"observations": [
            {"frame_id": "f1", "entity": "Aatrox", "field": "health", "value": 100,
             "visibility": "visible", "confidence": .95, "identity_evidence": "portrait"},
            {"frame_id": "f2", "entity": "Jax", "field": "health", "value": 120,
             "visibility": "visible", "confidence": .95}], "events": [], "issues": []}
        observations, _, issues = normalize_output(output, task, route)
        self.assertEqual(observations[1]["entity"], "unknown")
        self.assertIsNone(observations[1]["value"])
        conflict = {**observations[0], "id": "different", "value": 101, "route": "other"}
        states, conflicts = fuse_observations(observations+[conflict])
        self.assertEqual(states[0]["status"], "conflict")
        self.assertIsNone(states[0]["value"])
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(issues[0]["kind"], "identity_conflict")

    def test_last_confirmed_timestamp_does_not_fill_unobserved_value(self):
        base = {"id": "a", "entity": "ward:top", "field": "vision_status", "time": 1,
                "value": "visible", "visibility": "visible", "confidence": .95,
                "frame_id": "f", "review_status": "unreviewed"}
        later = {**base, "id": "b", "time": 10, "value": None, "visibility": "occluded"}
        states, _ = fuse_observations([base, later])
        self.assertEqual(states[-1]["last_confirmed_time"], 1)
        self.assertIsNone(states[-1]["value"])
        reviewed = {**base, "confidence": .1, "review_status": "human-verified"}
        verified, _ = fuse_observations([reviewed])
        self.assertEqual(verified[0]["status"], "confirmed")

    def test_overlapping_notices_deduplicate_but_separate_events_remain(self):
        a = event("a", 10, "death")
        b = {**event("b", 11, "death"), "routes": ["central_notice"]}
        c = event("c", 50, "death")
        fused = fuse_events([a, b, c])
        self.assertEqual(len(fused), 2)
        self.assertEqual(set(fused[0]["evidence_ids"]), {"frame_a", "frame_b"})
        b["location"] = "river"
        conflicting = fuse_events([a, b])
        self.assertIsNone(conflicting[0]["location"])
        self.assertIn("conflicting_location", conflicting[0]["ambiguity_flags"])
        channels = fuse_events([event("r1", 10), event("r2", 11)])
        self.assertEqual((channels[0]["start"], channels[0]["end"]), (10, 13))

    def test_objective_live_does_not_create_a_contest(self):
        route = Route("gameplay", [0, 0, 100, 100], "thinking", [])
        task = {"id": "t", "model": {}, "fps": 1, "pass": 0,
                "frames": [{"id": "f", "source_id": "f", "time": 1}]}
        _, events, issues = normalize_output({"events": [{"kind": "objective_contest", "start_frame": "f",
            "end_frame": "f", "actor": "Aatrox", "confidence": .9, "criteria": ["objective_visible"]}]}, task, route)
        self.assertFalse(events)
        self.assertEqual(issues[0]["kind"], "invalid_output")

    def test_clock_is_not_used_as_source_time_and_sidebars_remain_separate(self):
        route = Route("match_hud", [0, 0, 100, 100], "instruct", ["game_clock"])
        task = {"id": "t", "model": {}, "fps": 1, "pass": 0,
                "frames": [{"id": "f", "source_id": "f", "time": 30}]}
        obs, _, _ = normalize_output({"observations": [{"frame_id": "f", "entity": "match",
            "field": "game_clock", "value": "19:40", "visibility": "visible", "confidence": 1}]}, task, route)
        self.assertEqual(obs[0]["time"], 30)
        self.assertEqual(obs[0]["value"], "19:40")
        following = {**obs[0], "time": 31, "value": "19:41"}
        self.assertIsNone(transition_reason(obs[0], following))
        self.assertEqual(transition_reason(following, {**following, "time": 32, "value": "01:00"}), "clock_discontinuity")
        full = Config.load(Path("annotation/league/aatrox.json"))
        self.assertNotEqual(next(r.box for r in full.routes if r.name == "left_sidebar"),
                            next(r.box for r in full.routes if r.name == "right_sidebar"))


class QuestionTests(unittest.TestCase):
    def test_mechanics_question_requires_visual_identity_without_revealing_item_id(self):
        class Resolver:
            def handle_query(self, request):
                return {"status": "ok", "reference": {"identity": {"name": "Sundered Sky"},
                    "stats": {"values": {"ability_haste:flat": {"value": 10, "unit": "flat"}}}, "provenance": []}}
        questions = construct([state("s", 10, "inventory", ["6610"])], [], resolver=Resolver())
        self.assertEqual(questions[0]["knowledge_category"], "mechanics_augmented")
        self.assertNotIn("6610", questions[0]["prompt"])
        self.assertNotIn("Sundered Sky", questions[0]["prompt"])
        self.assertEqual(questions[0]["answer"]["ability_haste"], 10)

    def test_all_six_families_have_computable_answers_and_separated_evidence(self):
        states = [state("s1", 1, "inventory", ["6610"]), state("s2", 101, "inventory", ["6610", "3006"]),
                  state("n1", 1, "kills", 1), state("n2", 101, "kills", 4)]
        states += [state(f"p{i}", t, "position", {"region": p})
                   for i, (t, p) in enumerate([(1, "top_lane"), (61, "river"), (121, "base")])]
        events = [event("e1", 1), event("e2", 61), event("e3", 121), event("e4", 181, "death")]
        questions = construct(states, events)
        self.assertEqual(set(q["capability"] for q in questions), set(CAPABILITIES))
        count = next(q for q in questions if q["capability"] == "cumulative_counting")
        self.assertEqual(count["answer"], 3)
        trajectory = next(q for q in questions if q["capability"] == "trajectory_memory")
        self.assertEqual(trajectory["answer"], ["top_lane", "river", "base"])
        for q in questions:
            if q["horizon"] == "match_trajectory":
                self.assertGreaterEqual(len(q["evidence_times"]), 2)
            self.assertEqual(q["review_status"], "unreviewed")

    def test_editorial_overlays_and_sparse_casts_exclude_unsupported_questions(self):
        events = [event("e1", 10, "cast"), event("e2", 100, "death")]
        questions = construct([], events, issues=[{"kind": "editorial_overlay", "time": 100}])
        self.assertFalse(questions)
        simultaneous = construct([], [event("a", 10, "death", "Aatrox"), event("b", 10, "death", "Jax")])
        self.assertFalse(simultaneous)

    def test_solver_input_allowlist_excludes_answers_logs_and_gold_timestamps(self):
        q = construct([], [event("e", 10)])[0]
        q.update(answer="SECRET_ANSWER", construction_logs="SECRET_LOG", review_notes="SECRET_REVIEW")
        q["evidence_times"] = [999999]
        serialized = json.dumps(solver_messages(q, [{"id": "f", "time": 8}]))
        for forbidden in ("SECRET_ANSWER", "SECRET_LOG", "SECRET_REVIEW", "999999", "evidence_ids"):
            self.assertNotIn(forbidden, serialized)

    def test_sampling_does_not_consult_gold_evidence(self):
        self.assertEqual(uniform_times(0, 1200, 8), uniform_times(0, 1200, 8))
        self.assertEqual(len(uniform_times(0, 1200, 8)), 8)
        self.assertGreater(uniform_times(0, 1200, 8)[-1], 1199)

    def test_parser_ignores_thinking_json_and_reference_policy_filters_mechanics(self):
        self.assertEqual(parse_output('<think>{"answer":"wrong"}</think>{"answer":"right"}')["answer"], "right")
        class Resolver:
            patch = type("Patch", (), {"ddragon_version": "16.13.1"})()
            def _identify(self, *args):
                return {"internal_id": "Aatrox", "numeric_id": "266", "name": "Aatrox"}
        result = ScopedResolver(Resolver(), "visual_only").handle_query({"query_knowledge": "Aatrox"})
        self.assertNotIn("spells", result)
        self.assertEqual(result["scope"], "identity_only")


class EndToEndTests(unittest.TestCase):
    def test_interrupt_resume_optional_review_and_independent_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = init_store(temporary)
            cfg = config()
            cfg.validate()
            with patch("annotation.league.observers.Media", FixtureMedia), patch("annotation.league.evaluation.Media", FixtureMedia):
                first = FixtureRunner(store, stop_after=2)
                with self.assertRaises(StopRequested):
                    annotate(store, cfg, first, StaticFixture())
                committed = list(store.path("inference").glob("*/segment-*.json"))
                self.assertEqual(len(committed), 2)
                store.stopping = False
                resumed = FixtureRunner(store)
                summary = annotate(store, cfg, resumed, StaticFixture())
                self.assertLessEqual(summary["review_visual_tokens"], summary["review_token_allowance"])
                self.assertNotIn(first.calls[0], resumed.calls)
                self.assertTrue(any(o["status"] != "ok" for r in store.rows("review_requests") for o in r["outcomes"]))
                self.assertTrue(any(o["value"] == "한글플레이어" for o in store.rows("observations")))
                fuse(store, cfg)
                generate_qa(store, cfg)
                self.assertTrue(store.rows("questions"))
                self.assertTrue(all(q["review_status"] == "unreviewed" for q in store.rows("questions")))
                evaluate(store, cfg, FixtureRunner(store), StaticFixture())
                initial_report = report(store, cfg)
                self.assertEqual(initial_report["reference_strata"]["human-verified"]["eligible"], 0)
                self.assertGreater(initial_report["reference_strata"]["unreviewed"]["evaluated"], 0)
                original = store.rows("questions")[0]
                old_key = store.stage_key("evaluate", cfg, ["questions"])
                replacement = {"number": -1, "sequence": ["corrected"], "object": {"corrected": True}}.get(original["answer_type"], "corrected")
                add_correction(store, "questions", original["id"], {"answer": replacement}, "human-verified")
                self.assertNotEqual(store.stage_key("evaluate", cfg, ["questions"]), old_key)
                self.assertEqual(store.rows("questions")[0], original)
                cached_runner = FixtureRunner(store)
                evaluate(store, cfg, cached_runner, StaticFixture())
                self.assertFalse(cached_runner.calls)
                corrected_report = report(store, cfg)
                self.assertEqual(corrected_report["reference_strata"]["human-verified"]["eligible"], 1)
                self.assertEqual(len(store.corrections()), 1)
                self.assertIsNone(corrected_report["efficiency"]["gpu_cost"])

    def test_cli_refuses_stale_predictions_after_human_correction(self):
        from annotation.league.__main__ import main
        with tempfile.TemporaryDirectory() as temporary:
            store = init_store(temporary)
            cfg = config()
            atomic_json(store.path("config.json"), cfg.to_dict())
            q = construct([], [event("e", 10)])[0]
            store.write("questions", [q])
            store.write("predictions", [{"id": "p", "question_id": q["id"]}])
            store.commit_stage("evaluate", store.stage_key("evaluate", cfg, ["questions"]))
            add_correction(store, "questions", q["id"], {}, "human-verified")
            with contextlib.redirect_stderr(io.StringIO()) as error:
                status = main(["--run-dir", temporary, "--offline", "report"])
            self.assertEqual(status, 1)
            self.assertIn("stale", error.getvalue())

    def test_review_during_inference_does_not_change_a_stages_frozen_revision(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = init_store(temporary)
            q = construct([], [event("e", 10)])[0]
            store.write("questions", [q])
            cfg = config()
            with store.review_snapshot():
                original_key = store.stage_key("evaluate", cfg, ["questions"])
                add_correction(Store(Path(temporary)), "questions", q["id"], {}, "human-verified")
                self.assertEqual(store.effective(store.rows("questions"), "questions")[0]["review_status"], "unreviewed")
                self.assertEqual(store.stage_key("evaluate", cfg, ["questions"]), original_key)
            self.assertEqual(store.effective(store.rows("questions"), "questions")[0]["review_status"], "human-verified")
            self.assertNotEqual(store.stage_key("evaluate", cfg, ["questions"]), original_key)

    def test_config_changes_invalidate_cached_tasks(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = init_store(temporary)
            cfg = config()
            first = store.fingerprint(cfg)
            cfg.routes[0].max_pixels //= 2
            self.assertNotEqual(first, store.fingerprint(cfg))

    def test_question_and_memory_budgets_do_not_invalidate_visual_annotation(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = init_store(temporary)
            cfg = config()
            annotation_key = store.fingerprint(cfg)
            qa_key = store.stage_key("generate-qa", cfg, [])
            cfg.max_questions += 6
            self.assertEqual(annotation_key, store.fingerprint(cfg))
            self.assertNotEqual(qa_key, store.stage_key("generate-qa", cfg, []))
            evaluation_key = store.stage_key("evaluate", cfg, [])
            cfg.solver_max_frames //= 2
            self.assertEqual(annotation_key, store.fingerprint(cfg))
            self.assertNotEqual(evaluation_key, store.stage_key("evaluate", cfg, []))

    def test_event_metric_matching_does_not_double_count_predictions(self):
        result = temporal_metrics([[10, 12]], [[10, 12], [10, 12]])
        self.assertEqual(result["matched"], 1)
        self.assertEqual(result["precision"], .5)
        self.assertEqual(result["mean_iou"], 1)
        self.assertFalse(exact_answer(1, True))
        self.assertTrue(exact_answer({"gold": "28.5k"}, {"gold": "28.5k"}))
        self.assertFalse(exact_answer("28.5k", 28500))


class MediaTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
    def test_real_decoding_uses_source_pts_and_rejects_wrong_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            video = Path(temporary)/"source.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=1920x1080:rate=10",
                            "-t", "1", "-c:v", "libx264", "-preset", "ultrafast", str(video)], check=True)
            store = Store(Path(temporary)/"run")
            cfg = config()
            prepare(store, cfg, video)
            frame = Media(store, cfg).frame(.35, cfg.routes[0])
            self.assertAlmostEqual(frame["time"], .4)
            self.assertTrue(store.path(frame["path"]).exists())
            cfg.width = 1280
            with self.assertRaises(ValueError):
                prepare(Store(Path(temporary)/"bad"), cfg, video)


class ReviewApiTests(unittest.TestCase):
    def test_optional_review_routes_and_append_only_corrections(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("Install requirements-review.txt and httpx for API integration")
        with tempfile.TemporaryDirectory() as temporary:
            store = init_store(temporary)
            questions = construct([], [event("e", 10)])
            store.write("questions", questions)
            with TestClient(create_app(store)) as client:
                self.assertEqual(client.get("/").status_code, 200)
                records = client.get("/api/records").json()["records"]
                self.assertEqual(len(records), 1)
                response = client.post("/api/corrections", json={"target": "questions", "record_id": questions[0]["id"],
                    "changes": {}, "review_status": "human-verified"})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(store.rows("questions")[0]["review_status"], "unreviewed")
                self.assertEqual(client.get("/api/records").json()["records"][0]["review_status"], "human-verified")
                invalid = client.post("/api/corrections", json={"target": "questions", "record_id": questions[0]["id"],
                    "changes": {"id": "changed"}, "review_status": "human-verified"})
                self.assertEqual(invalid.status_code, 422)
                self.assertEqual(client.get("/media/manifest.json").status_code, 404)
                forbidden = client.post("/api/corrections", headers={"Origin": "https://other.example"},
                    json={"target": "questions", "record_id": questions[0]["id"], "changes": {}, "review_status": "rejected"})
                self.assertEqual(forbidden.status_code, 403)


if __name__ == "__main__":
    unittest.main()
