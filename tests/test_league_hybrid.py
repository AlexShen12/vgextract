"""Contract and isolation checks for the reduced Midbeast intake."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from annotation.league.core import Config, Store, StopRequested, atomic_json, read_json
from annotation.league.hybrid_intake import enabled_fields, _panel_state, pilot_windows
from annotation.league.intake_profile import load_profile
from annotation.league.local_models import ModelUnavailable, checkpoint, LocalVLRunner
from annotation.league.playback import PlaybackTracker
from annotation.league.numeric_reader import NumericReader
from annotation.league.vl_reader import VLReader, validate_transcriptions
from annotation.league.experiments import (
    validate_events,
    score_events,
    word_error_rate,
    audio_metrics,
    score_tracking,
    event_messages,
)
from annotation.league.timeline import fuse_observations
from annotation.league.questions import construct


def response(readings=None, **overrides):
    return {
        "finish_reason": "stop",
        "text": json.dumps({"readings": readings or []}),
        **overrides,
    }


class TranscriptionTests(unittest.TestCase):
    fields = [{"id": "hp", "parser": "health"}, {"id": "name", "parser": "name"}]

    def test_exact_binding_text_and_numeric_parsing(self):
        result = validate_transcriptions(
            response(
                [
                    {"field_id": "name", "status": "visible", "text": "한글 이름"},
                    {"field_id": "hp", "status": "visible", "text": "1877 / 2043"},
                ]
            ),
            self.fields,
        )
        self.assertEqual(result["hp"]["raw_text"], "1877 / 2043")
        self.assertEqual(result["hp"]["value"]["current"], 1877)
        self.assertEqual(result["name"]["value"], "한글 이름")

    def test_truncation_empty_examples_format_and_binding_fail_closed(self):
        valid = {"field_id": "hp", "status": "visible", "text": "1877/2043"}
        cases = [
            response([valid], finish_reason="length"),
            response([]),
            response([valid, valid]),
            response([{**valid, "field_id": "invented"}]),
            response([{**valid, "field_id": []}]),
            response([{**valid, "values": {"health": "example"}}]),
            response(text="```json\n{}\n```"),
            response(text="[]"),
            response(text=""),
        ]
        for candidate in cases:
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                validate_transcriptions(candidate, self.fields)

    def test_not_shown_and_unreadable_do_not_invent_values(self):
        result = validate_transcriptions(
            response(
                [
                    {"field_id": "hp", "status": "not_shown", "text": None},
                    {"field_id": "name", "status": "unreadable", "text": None},
                ]
            ),
            self.fields,
        )
        self.assertEqual(result["hp"]["acceptance"], "absent")
        self.assertEqual(result["name"]["acceptance"], "unknown")
        self.assertIsNone(result["hp"]["value"])

    def test_present_panel_name_cannot_be_accepted_as_absent(self):
        field = {"id": "name", "parser": "name", "required_when_panel_present": True}
        runner = Mock(identity={})
        runner.complete.return_value = response(
            [{"field_id": "name", "status": "not_shown", "text": None}]
        )
        with tempfile.TemporaryDirectory() as temporary:
            store = Store(Path(temporary))
            crop = store.path("crop.png")
            Image.new("RGB", (30, 15)).save(crop)
            result = VLReader(runner, store).read_group([(field, crop)], store.path("request"))
        self.assertEqual(runner.complete.call_count, 2)
        self.assertEqual(result["name"]["acceptance"], "unknown")
        self.assertIsNone(result["name"]["value"])

    def test_failed_group_retries_once_per_crop_with_same_cap(self):
        runner = Mock(identity={"revision": "pinned"})
        runner.complete.side_effect = [
            response(text="broken"),
            response([{"field_id": "hp", "status": "visible", "text": "9/10"}]),
            response([{"field_id": "name", "status": "unreadable", "text": None}]),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            store = Store(Path(temporary))
            crop = store.path("crop.png")
            Image.new("RGB", (30, 15)).save(crop)
            result = VLReader(runner, store).read_group(
                [(f, crop) for f in self.fields], store.path("request")
            )
            self.assertTrue(store.path("request/response.json").exists())
        self.assertEqual(runner.complete.call_count, 3)
        self.assertEqual(result["hp"]["value"]["current"], 9)
        self.assertEqual(result["name"]["acceptance"], "unknown")
        self.assertTrue(
            all(
                call.kwargs["max_tokens"] == 512
                for call in runner.complete.call_args_list
            )
        )
        self.assertTrue(
            all(
                len(call.args[0][0]["content"]) <= 9
                for call in runner.complete.call_args_list
            )
        )

    def test_conflicting_plausible_partial_group_remains_unknown(self):
        runner = Mock(identity={})
        runner.complete.side_effect = [
            response([{"field_id": "hp", "status": "visible", "text": "9/10"}]),
            response([{"field_id": "hp", "status": "visible", "text": "8/10"}]),
            response([{"field_id": "name", "status": "not_shown", "text": None}]),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            store = Store(Path(temporary))
            crop = store.path("crop.png")
            Image.new("RGB", (30, 15)).save(crop)
            result = VLReader(runner, store).read_group(
                [(f, crop) for f in self.fields], store.path("request")
            )
        self.assertEqual(result["hp"]["acceptance"], "unknown")
        self.assertIn("conflicting", result["hp"]["reason"])
        self.assertEqual(runner.complete.call_count, 3)

    def test_worker_oom_and_interruption_do_not_retry_or_cache_unknown(self):
        for error in (ModelUnavailable("CUDA out of memory"), StopRequested()):
            with tempfile.TemporaryDirectory() as temporary:
                store = Store(Path(temporary))
                crop = store.path("crop.png")
                Image.new("RGB", (30, 15)).save(crop)
                runner = Mock(identity={})
                runner.complete.side_effect = error
                with self.assertRaises(type(error)):
                    VLReader(runner, store).read_group(
                        [(self.fields[0], crop)], store.path("request")
                    )
                self.assertEqual(runner.complete.call_count, 1)
                self.assertFalse(store.path("request/result.json").exists())

    def test_numeric_reader_uses_one_attempt_for_success_and_blank_input(self):
        cases = (
            ("readable_level", {"raw_text": "12", "score": .99, "engine": "tesseract"}, "level", "accepted", 12),
            ("blank_integer", {"raw_text": "", "score": 0}, "integer", "unknown", None),
        )
        for name, raw, parser, acceptance, value in cases:
            with self.subTest(case=name), tempfile.TemporaryDirectory() as temporary, patch(
                "annotation.league.numeric_reader.tesseract", return_value=raw
            ) as tess:
                reading = NumericReader().read(
                    Image.new("RGB", (12, 12)), {"reader": "number", "parser": parser}, Path(temporary)
                )
                self.assertEqual(reading["acceptance"], acceptance)
                self.assertEqual(reading["value"], value)
                self.assertEqual(tess.call_count, 1)

    def test_default_field_assignments_and_deferred_assets(self):
        profile = load_profile("midbeast-1080p-v1")
        fields = enabled_fields(profile)
        self.assertEqual(len(fields), 21)
        self.assertFalse(any(f["reader"] == "icon" for f in fields))
        self.assertTrue(
            all(
                f["backend"] == "vl"
                for f in fields
                if f["parser"] in {"health", "name", "notice"}
            )
        )
        self.assertTrue(
            all(
                f["parser"] in {"clock", "timer", "gold", "integer", "level", "kda"}
                for f in fields
                if f["backend"] == "tesseract"
            )
        )
        self.assertIn("icons", profile["deferred"])
        self.assertFalse(
            _panel_state(Image.new("RGB", (1920, 1080)), profile, fields[0], False)
            == "present"
        )


class CheckpointTests(unittest.TestCase):
    def test_worker_compiler_caches_use_work_assets_without_replacing_home(self):
        from annotation.league.local_models import isolated_environment

        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            "os.environ", {"HOME": "/original-home", "LEAGUE_WORK_ROOT": temporary, "TMPDIR": "/original-home/tmp", "HF_HOME": "/original-home/hf"}
        ):
            environment = isolated_environment(temporary)
            self.assertEqual(environment["HOME"], "/original-home")
            for key in (
                "TRITON_CACHE_DIR",
                "TORCHINDUCTOR_CACHE_DIR",
                "VLLM_CACHE_ROOT",
                "XDG_CACHE_HOME",
                "NUMBA_CACHE_DIR",
                "CUDA_CACHE_PATH",
                "TMPDIR", "TMP", "TEMP", "TORCH_HOME", "XDG_CONFIG_HOME",
                "XDG_DATA_HOME", "XDG_STATE_HOME", "VLLM_CONFIG_ROOT",
                "MPLCONFIGDIR", "FLASHINFER_WORKSPACE_BASE", "HF_HOME",
                "HF_HUB_CACHE", "HF_MODULES_CACHE", "HF_ASSETS_CACHE", "PIP_CACHE_DIR", "VLLM_RPC_BASE_PATH",
            ):
                self.assertTrue(Path(environment[key]).is_dir())
                self.assertTrue(Path(environment[key]).is_relative_to(Path(temporary).resolve()))
            self.assertEqual(Path(environment["VLLM_RPC_BASE_PATH"]), Path(temporary).resolve()/"tmp")

    def test_worker_prefers_conda_native_runtime_and_preserves_other_library_paths(
        self,
    ):
        from annotation.league.local_models import isolated_environment

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "conda-meta").mkdir()
            (root / "lib").mkdir()
            with patch(
                "annotation.league.local_models.sys.base_prefix", temporary
            ), patch(
                "annotation.league.local_models.sys.platform", "linux"
            ), patch.dict(
                "os.environ",
                {
                    "LD_LIBRARY_PATH": "/other",
                    "PYTHONPATH": "/unrelated",
                    "PYTHONHOME": "/wrong",
                },
            ):
                environment = isolated_environment()
            self.assertEqual(
                environment["LD_LIBRARY_PATH"], str(root / "lib") + ":/other"
            )
            self.assertNotIn("PYTHONPATH", environment)
            self.assertNotIn("PYTHONHOME", environment)
            self.assertEqual(environment["VLLM_NO_USAGE_STATS"], "1")

    def test_torch_gpu_uuid_is_valid_for_nvidia_smi(self):
        from annotation.league.local_models import nvidia_uuid

        value = "1bc3a9e9-a20e-376b-8937-80cb8ae78dcd"
        self.assertEqual(nvidia_uuid(value), "GPU-" + value)
        self.assertEqual(nvidia_uuid("GPU-" + value), "GPU-" + value)
        self.assertEqual(nvidia_uuid(None), "")

    def test_preflight_cannot_overwrite_old_engine_marker(self):
        from annotation.league.hybrid_intake import runtime

        with tempfile.TemporaryDirectory() as temporary:
            store = Store(Path(temporary))
            atomic_json(store.path("manifest.json"), {"schema_version": 2})
            atomic_json(store.path("intake_runtime.json"), {"engine": "paddleocr"})
            before = store.path("intake_runtime.json").read_bytes()
            with self.assertRaisesRegex(ValueError, "Earlier extraction"):
                runtime(store, Config({}, [], evaluation_enabled=False))
            self.assertEqual(store.path("intake_runtime.json").read_bytes(), before)

    def test_missing_and_same_size_corrupted_checkpoint_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "Missing pinned"):
                checkpoint(root, "intake")
            import annotation.league.local_models as models

            pin = json.loads(models.LOCK.read_text())["models"]["intake"]
            directory = root / "models/intake"
            snapshot = directory / "snapshot"
            snapshot.mkdir(parents=True)
            (snapshot / "model.safetensors").write_bytes(b"abcd")
            atomic_json(
                directory / "manifest.json",
                {
                    "specification": pin,
                    "files": [
                        {
                            "path": "model.safetensors",
                            "size": 4,
                            "sha256": hashlib.sha256(b"abcd").hexdigest(),
                        }
                    ],
                },
            )
            checkpoint(root, "intake")
            (snapshot / "model.safetensors").write_bytes(b"efgh")
            with self.assertRaisesRegex(ValueError, "checksum"):
                checkpoint(root, "intake")

    def test_fatal_worker_latches_failure_and_disables_thinking(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = Store(Path(temporary))
            signature = (
                Path(temporary),
                {"runtime": {"total_memory_bytes": 48_000_000_000}},
            )
            with patch("annotation.league.local_models.socket.socket"):
                runner = LocalVLRunner(store, "unused", temporary, signature=signature)
            with patch.object(runner, "start"), patch.object(
                runner, "_json", side_effect=OSError("OOM")
            ) as request, patch.object(runner, "close"):
                with self.assertRaises(ModelUnavailable):
                    runner.complete([])
                payload = request.call_args.args[1]
                self.assertFalse(payload["chat_template_kwargs"]["enable_thinking"])
                self.assertEqual(payload["temperature"], 0)
                self.assertEqual(payload["max_tokens"], 512)
            with self.assertRaises(ModelUnavailable):
                runner.start()


class PlaybackTests(unittest.TestCase):
    def test_rewind_gap_and_pauses_use_source_time_and_new_segments(self):
        from PIL import Image
        still = Image.new("RGB", (64, 36), "gray")
        tracker = PlaybackTracker("source")
        first = tracker.update(0, 100, still)
        second = tracker.update(1, 101, still)
        rewind = tracker.update(2, 100, still)
        self.assertEqual(first["playback_segment_id"], second["playback_segment_id"])
        self.assertNotEqual(first["playback_segment_id"], rewind["playback_segment_id"])
        self.assertTrue(rewind["revisited_game_clock"])
        self.assertFalse(rewind["continuity_verified"])
        tracker.update(3, 100, still)
        pause = tracker.update(4, 100, still)
        self.assertEqual(pause["playback_state"], "paused")
        gap = tracker.update(20, 101)
        self.assertIn("unsampled_gap", gap["discontinuity_reasons"])

    def test_clock_unknown_and_layout_changes_break_continuity(self):
        tracker = PlaybackTracker("source")
        first = tracker.update(0, 1)
        missing = tracker.update(1, None)
        third = tracker.update(2, 3)
        self.assertEqual(
            len({r["playback_segment_id"] for r in (first, missing, third)}), 3
        )
        invalid = tracker.update(3, 4, layout_valid=False)
        self.assertFalse(invalid["continuity_verified"])

    def test_cumulative_questions_cannot_cross_segments_or_audio(self):
        states = [
            {
                "id": str(t),
                "entity": "Darius",
                "field": "kills",
                "time": t,
                "value": v,
                "status": "confirmed",
                "evidence_ids": [str(t)],
                "frame_ids": [str(t)],
                "playback_segment_id": seg,
                "continuity_verified": True,
            }
            for t, v, seg in [(0, 1, "a"), (60, 3, "b")]
        ]
        self.assertFalse(
            any(q["capability"] == "cumulative_state" for q in construct(states, []))
        )
        for s in states:
            s["evidence_modality"] = "audio"
        self.assertEqual(construct(states, []), [])

    def test_unknown_identity_retains_separate_slot_facts_and_audio_excluded(self):
        rows = [
            {
                "id": str(i),
                "schema_version": 2,
                "entity": "unknown",
                "slot_id": slot,
                "field": "health",
                "value": value,
                "time": 0,
                "visibility": "visible",
                "acceptance": "accepted",
                "frame_id": str(i),
                "playback_segment_id": "s",
            }
            for i, (slot, value) in enumerate([("left:row1", 10), ("left:row2", 20)])
        ]
        states, conflicts = fuse_observations(rows)
        self.assertEqual(len(states), 2)
        self.assertFalse(conflicts)
        self.assertEqual({s["value"] for s in states}, {10, 20})
        self.assertEqual(construct(states, []), [])
        rows[0]["evidence_modality"] = "audio"
        self.assertEqual(len(fuse_observations(rows)[0]), 1)

    def test_pilot_contains_five_minutes_in_disjoint_windows(self):
        windows = pilot_windows(0, 1733)
        self.assertEqual(sum(b - a for a, b in windows), 300)
        self.assertTrue(all(b <= c for (a, b), (c, d) in zip(windows, windows[1:])))


class ExperimentTests(unittest.TestCase):
    def experiment_store(self):
        from annotation.league.media import file_hash

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        store = Store(Path(temporary.name))
        video = store.path('video.mp4')
        video.write_bytes(b'source fixture')
        atomic_json(store.path('manifest.json'), {
            'source': {'path': str(video), 'sha256': file_hash(video), 'width': 1920, 'height': 1080},
            'interval': [0, 300],
        })
        store.write('observations', [{'id': 'preserved', 'value': 9}])
        return store

    def test_event_failure_continues_other_suites_but_cancellation_preserves_partial_report(self):
        from annotation.league.experiments import experiment

        for error in (ValueError('bad response'), StopRequested('cancelled')):
            with self.subTest(error=type(error).__name__):
                store = self.experiment_store()
                before = store.path('observations.jsonl').read_bytes()
                with patch('annotation.league.experiments.load_protocol', return_value={'clips': []}), patch(
                    'annotation.league.experiments.compare_events', side_effect=[error, {'status': 'completed_experimental'}]
                ) as compare, patch.dict('os.environ', {'LEAGUE_TRACKING_PYTHON': ''}):
                    if isinstance(error, StopRequested):
                        with self.assertRaises(StopRequested):
                            experiment(store, Config({}, []), 'events', draft=True)
                    else:
                        result = experiment(store, Config({}, []), 'events', draft=True)
                        self.assertEqual(result['status'], 'partial_experimental')
                        self.assertEqual(result['results']['intake']['status'], 'failed')
                        self.assertEqual(result['results']['event_alternative']['status'], 'completed_experimental')
                        self.assertEqual(result['results']['tracking']['status'], 'blocked')
                saved = read_json(next(store.path('experiments').glob('*/report.json')))
                self.assertEqual(saved['status'], 'interrupted' if isinstance(error, StopRequested) else 'partial_experimental')
                self.assertEqual(compare.call_count, 1 if isinstance(error, StopRequested) else 2)
                self.assertEqual(store.path('observations.jsonl').read_bytes(), before)

    def test_all_suites_continue_without_audio_after_crop_and_audio_failures(self):
        from annotation.league.experiments import experiment

        store = self.experiment_store()
        def protocol(store, path, kind, draft):
            if kind == 'audio':
                raise ValueError('bad audio labels')
            return {'clips': []}
        with patch('annotation.league.intake_benchmark.load_labels', side_effect=ValueError('bad crop labels')), patch(
            'annotation.league.experiments.load_protocol', side_effect=protocol
        ), patch('annotation.league.experiments.compare_events', return_value={'status': 'completed_experimental'}) as compare, patch(
            'annotation.league.experiments.run_worker', return_value={'clips': []}
        ), patch.dict('os.environ', {'LEAGUE_TRACKING_PYTHON': 'fixture'}):
            result = experiment(store, Config({}, []), 'all', draft=True)
        self.assertEqual(result['status'], 'partial_experimental')
        self.assertEqual(result['results']['crops']['status'], 'failed')
        self.assertEqual(result['results']['audio']['status'], 'failed')
        self.assertEqual(compare.call_count, 2)
        self.assertTrue(all(call.args[-2] is None for call in compare.call_args_list))
        self.assertEqual(result['results']['tracking']['clips'], [])

    def test_events_validate_and_report_one_to_one_matches(self):
        clip = {
            "id": "clip",
            "start": 100,
            "end": 108,
            "events": [{"kind": "death", "start": 102, "end": 102, "actor": "Darius"}],
        }
        response = {
            "finish_reason": "stop",
            "text": json.dumps(
                {
                    "events": [
                        {"kind": "death", "start": 2, "end": 2, "actor": "Darius"}
                    ],
                    "abstain": False,
                }
            ),
        }
        parsed = validate_events(response, clip)
        result = score_events([clip], {"clip": parsed})
        self.assertEqual(result["precision"], 1)
        self.assertEqual(result["recall"], 1)
        self.assertEqual(result["actor_accuracy"], 1)
        self.assertEqual(parsed["events"][0]["start"], 102)
        with self.assertRaises(ValueError):
            validate_events({**response, "finish_reason": "length"}, clip)
        with self.assertRaises(ValueError):
            validate_events({"finish_reason": "stop", "text": "[]"}, clip)

    def test_audio_cannot_enter_visual_prompt_or_visual_questions(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "clip.mp4"
            path.write_bytes(b"video")
            visual = json.dumps(event_messages(path))
            assisted = json.dumps(event_messages(path, "unique commentary"))
            self.assertNotIn("unique commentary", visual)
            self.assertIn("unique commentary", assisted)
        event = {"confidence": 1, "actor": "Darius", "experimental": True}
        self.assertEqual(construct([], [event]), [])

    def test_asr_metrics_include_silence_names_and_alignment(self):
        self.assertEqual(word_error_rate("one two", "one three")["wer"], 0.5)
        self.assertEqual(
            word_error_rate("", "hallucinated")["false_words_on_silence"], 1
        )
        result = audio_metrics(
            {
                "transcript": "Darius wins",
                "names": ["Darius"],
                "words": [{"text": "Darius", "start": 12}],
            },
            [{"text": "Darius wins", "words": [{"text": "Darius", "start": 12.2}]}],
        )
        self.assertEqual(result["name_recall"], 1)
        self.assertAlmostEqual(result["mean_word_onset_error_seconds"], 0.2)

    def test_out_of_span_alignment_preserves_transcription_and_rejects_word(self):
        from annotation.league.experiment_worker import bind_audio_alignment

        chunk = {"source_start": 100, "source_end": 102, "text": "Darius wins", "status": "transcribed"}
        bind_audio_alignment(chunk, [
            {"text": "Darius", "start_time": .2, "end_time": .8},
            {"text": "wins", "start_time": 1.8, "end_time": 2.4},
        ])
        self.assertEqual(chunk["status"], "transcribed")
        self.assertEqual(chunk["text"], "Darius wins")
        self.assertEqual(chunk["words"], [{"text": "Darius", "start": 100.2, "end": 100.8}])
        self.assertEqual(chunk["alignment"]["status"], "partial")
        self.assertEqual(chunk["alignment"]["rejected_words"][0]["end"], 102.4)
        self.assertEqual(len(chunk["alignment_candidates"]), 2)

    def test_experiments_release_models_sequentially_and_preserve_observations(self):
        from annotation.league.experiments import experiment
        from annotation.league.media import file_hash

        store = self.experiment_store()
        video = store.path("video.mp4")
        before = store.path("observations.jsonl").read_bytes()
        config = Config({}, [], evaluation_enabled=False)
        clips = [
            {
                "id": f"c{i}",
                "start": i * 10,
                "end": i * 10 + 8,
                "split": "development" if i < 10 else "held_out",
                "events": [],
            }
            for i in range(20)
        ]
        tracking = [
            {**c, "prompts": [{"object_id": 1, "point": [10, 10]}]}
            for c in clips[:5]
        ]
        events = []

        class Worker:
            def __init__(self, store, python, cache, role):
                self.role = role
                self.identity = {"role": role}
                self.peak_memory_bytes = 1
                events.append("load:" + role)

            def complete(self, *args, **kwargs):
                return {
                    "finish_reason": "stop",
                    "text": json.dumps({"events": [], "abstain": False}),
                    "wall_seconds": 1,
                    "peak_memory_bytes": 1,
                }

            def close(self):
                events.append("close:" + self.role)

        def protocol(store, path, kind, draft):
            return {"clips": clips if kind == "events" else tracking}

        def tracking_worker(*args):
            events.append("tracking")
            return {"clips": [], "experimental": True}

        audit = {
            "video": str(video),
            "video_sha256": file_hash(video),
            "frames_dir": str(store.root),
        }
        with patch(
            "annotation.league.experiments.load_protocol", side_effect=protocol
        ), patch(
            "annotation.league.experiments.prepare_clip", return_value=audit
        ), patch(
            "annotation.league.experiments.LocalVLRunner", Worker
        ), patch(
            "annotation.league.experiments.run_worker", side_effect=tracking_worker
        ), patch.dict(
            "os.environ", {"LEAGUE_TRACKING_PYTHON": "fixture"}
        ):
            result = experiment(store, config, "events", draft=True)
        self.assertEqual(
            events,
            [
                "load:intake",
                "close:intake",
                "load:event_alternative",
                "close:event_alternative",
                "tracking",
            ],
        )
        self.assertEqual(store.path("observations.jsonl").read_bytes(), before)
        self.assertTrue(result["experimental"])
        self.assertFalse(result["production_evidence_modified"])
        self.assertEqual(store.rows("event_candidates"), [])

    def test_tracking_measures_masks_not_game_events(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mask.png"
            Image.new("L", (16, 16), 255).save(path)
            truth = [
                {
                    "id": "c",
                    "reference_masks": [
                        {"object_id": 1, "frame_index": 0, "path": "mask.png"}
                    ],
                }
            ]
            predictions = [
                {
                    "clip_id": "c",
                    "frames": [
                        {
                            "frame_index": 0,
                            "masks": [{"object_id": 1, "mask_path": str(path)}],
                        }
                    ],
                }
            ]
            result = score_tracking(truth, predictions, temporary)
            self.assertEqual(result["mean_mask_iou"], 1)
            self.assertFalse(result["event_claims"])


if __name__ == "__main__":
    unittest.main()
