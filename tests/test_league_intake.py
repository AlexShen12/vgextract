"""Reader, contract, migration and real default-command integration checks."""

from __future__ import annotations

import contextlib
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch

from annotation.league.core import (
    Config,
    Store,
    StopRequested,
    atomic_json,
    digest,
    read_json,
)
from annotation.league.intake import (
    annotate,
    collect,
    require_v2,
    sampled_frames,
    decoded_frames,
)
from annotation.league.intake_benchmark import (
    score,
    label_coverage,
    select_development_gate,
)
from annotation.league.intake_profile import (
    calibrated_profile,
    layout_matches,
    load_profile,
)
from annotation.league.readers import (
    IconMatcher,
    parse_text,
)
from annotation.league.timeline import fuse_observations

from league_fixtures import Image, intake_fixture


class ParsingTests(unittest.TestCase):
    def test_exact_display_strings_and_parsed_numbers_coexist(self):
        self.assertEqual(
            parse_text("28.5k", "gold"), {"displayed_text": "28.5k", "amount": 28500}
        )
        self.assertEqual(
            parse_text("14:00", "clock"), {"displayed_text": "14:00", "seconds": 840}
        )
        self.assertEqual(
            parse_text("4/0/1", "kda"), {"kills": 4, "deaths": 0, "assists": 1}
        )
        self.assertEqual(
            parse_text("1326 2724", "health")["displayed_text"], "1326 2724"
        )
        self.assertEqual(parse_text("한글플레이어", "name"), "한글플레이어")
        self.assertEqual(parse_text("0", "rank"), 0)

    def test_parser_rejects_wrong_units_and_impossible_values(self):
        self.assertEqual(parse_text("19", "level"), 19)
        for text, kind in (
            ("14.0", "clock"),
            ("12:99", "clock"),
            ("100", "level"),
            ("10", "rank"),
            ("900/800", "health"),
            ("LIVE", "cooldown"),
            ("", "integer"),
            ("4 0 1", "kda"),
        ):
            with self.subTest(text=text, kind=kind), self.assertRaises(ValueError):
                parse_text(text, kind)


@unittest.skipUnless(Image, "Install requirements-intake.txt")
class ReaderTests(unittest.TestCase):

    def test_missing_or_changed_assets_stop_before_recognition(self):
        from annotation.league.visual_assets import verify_assets
        import hashlib

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            patch_pin = {"patch": "fixture"}
            with self.assertRaisesRegex(ValueError, "Visual assets missing"):
                verify_assets(root, patch_pin)
            Image.new("RGB", (32, 32), "red").save(root / "image.png")
            hashed = hashlib.sha256((root / "image.png").read_bytes()).hexdigest()
            manifest = {
                "patch": patch_pin,
                "assets": [
                    {"kind": kind, "path": "image.png", "sha256": hashed}
                    for kind in ("champion", "item", "spell", "passive")
                ],
            }
            atomic_json(root / "manifest.json", manifest)
            self.assertEqual(len(verify_assets(root, patch_pin)["assets"]), 4)
            (root / "image.png").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_assets(root, patch_pin)

    def test_asset_refresh_keeps_patch_versions_in_different_paths(self):
        from annotation.league.visual_assets import cache_assets

        def download(url):
            if url.endswith("/data/en_US/champion.json"):
                value = {"Aatrox": {"image": {"full": "Aatrox.png"}}}
            elif url.endswith("/data/en_US/champion/Aatrox.json"):
                value = {
                    "Aatrox": {
                        "passive": {"image": {"full": "passive.png"}},
                        "spells": [
                            {"image": {"full": slot + ".png"}} for slot in "QWER"
                        ],
                    }
                }
            elif url.endswith("/data/en_US/summoner.json"):
                value = {"SummonerFlash": {"image": {"full": "flash.png"}}}
            elif url.endswith("/data/en_US/item.json"):
                value = {"1001": {"image": {"full": "boots.png"}, "maps": {"11": True}}}
            else:
                stream = io.BytesIO()
                Image.new(
                    "RGB", (32, 32), "red" if "/16.13.1/" in url else "blue"
                ).save(stream, format="PNG")
                return stream.getvalue()
            return json.dumps({"data": value}).encode()

        with tempfile.TemporaryDirectory() as temporary, patch(
            "annotation.league.visual_assets.download", side_effect=download
        ):
            root = Path(temporary)
            first = {
                "gameplay_patch": "26.13",
                "ddragon_version": "16.13.1",
                "cdragon_version": "16.13",
            }
            second = {
                "gameplay_patch": "26.14",
                "ddragon_version": "16.14.1",
                "cdragon_version": "16.14",
            }
            cache_assets(root, first)
            old = read_json(root / "manifest.json")["assets"][0]
            before = (root / old["path"]).read_bytes()
            cache_assets(root, second)
            new = read_json(root / "manifest.json")["assets"][0]
            self.assertNotEqual(old["path"], new["path"])
            self.assertEqual((root / old["path"]).read_bytes(), before)
            self.assertNotEqual(old["sha256"], new["sha256"])

    def test_blank_and_unrelated_frames_fail_layout(self):
        profile = load_profile("domisum-1080p-v1")
        for color in ("black", "white", "green"):
            self.assertFalse(
                layout_matches(Image.new("RGB", (1920, 1080), color), profile)[0]
            )
        self.assertFalse(layout_matches(Image.new("RGB", (1280, 720)), profile)[0])

    def test_calibration_creates_new_revision_without_mutating_profile(self):
        original = load_profile("domisum-1080p-v1")
        before = copy.deepcopy(original)
        result = calibrated_profile(
            original,
            {original["fields"][0]["id"]: [930, 73, 1000, 103]},
            Image.new("RGB", (1920, 1080)),
        )
        self.assertEqual(original, before)
        self.assertEqual(result["revision"], original["revision"] + 1)
        self.assertNotEqual(digest(result), digest(original))

    def test_icon_ambiguity_and_empty_slot_are_distinct(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = Image.new("RGB", (32, 32), "red")
            for x in range(10, 25):
                for y in range(4, 25):
                    image.putpixel((x, y), (0, 240, 50))
            image.save(root / "a.png")
            matcher = IconMatcher(
                root,
                {
                    "assets": [
                        {"kind": "item", "id": "1", "path": "a.png"},
                        {"kind": "item", "id": "2", "path": "a.png"},
                    ]
                },
                {"icon": 0.9, "icon_margin": 0.08},
            )
            self.assertEqual(
                matcher.read(image, {"kind": "item"})["acceptance"], "unknown"
            )
            self.assertEqual(
                matcher.read(
                    Image.new("RGB", (32, 32)), {"kind": "item", "empty_allowed": True}
                )["acceptance"],
                "absent",
            )


class ContractTests(unittest.TestCase):
    def test_development_icon_calibration_uses_margin_and_rejects_ties(self):
        candidates = [(0.95, 0.01, False), (0.9, 0.2, True), (0.8, 0.3, True)]
        threshold, margin = select_development_gate(candidates, 0.98, icons=True)
        self.assertEqual((threshold, margin), (0.8, 0.2))
        with self.assertRaises(ValueError):
            select_development_gate([(0.99, 0, True)], 0.98, icons=True)
        with self.assertRaises(ValueError):
            select_development_gate([(0.99, 0.1, False)], 0.98, icons=True)

    def test_numeric_health_bar_crosscheck_rejects_digits_without_inventing_them(self):
        from annotation.league.intake import verify_readings

        health = {
            "acceptance": "accepted",
            "value": {"current": 50, "maximum": 1000},
            "raw_text": "50/1000",
        }
        bar = {"acceptance": "accepted", "value": 0.5}
        readings = [
            ({"slot_id": "selected", "field": "health"}, health, {}),
            ({"slot_id": "selected", "field": "health_fraction"}, bar, {}),
        ]
        verify_readings(readings)
        self.assertEqual(health["acceptance"], "unknown")
        self.assertEqual(health["value"], {"current": 50, "maximum": 1000})
        self.assertEqual(health["verification"]["status"], "conflict")
        self.assertEqual(bar["value"], 0.5)

    def test_longleaf_tesseract_four_lists_languages_without_a_directory(self):
        from annotation.league.intake import tesseract_signature

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "share/tesseract/tessdata"
            data.mkdir(parents=True)
            (data / "eng.traineddata").write_bytes(b"fixture recognition model")
            with patch.dict(
                os.environ, {"LEAGUE_TESSDATA": "", "TESSDATA_PREFIX": ""}
            ), patch(
                "annotation.league.intake.shutil.which",
                return_value=str(root / "bin/tesseract"),
            ), patch(
                "annotation.league.intake.subprocess.check_output",
                side_effect=[
                    "tesseract 4.1.1\n",
                    "List of available languages (1):\neng\n",
                ],
            ):
                signature = tesseract_signature()
                self.assertEqual(signature["version"], "tesseract 4.1.1")
                self.assertEqual(Path(os.environ["LEAGUE_TESSDATA"]), data.resolve())

    def test_corrected_native_kda_and_portrait_rebuild_slot_facts(self):
        from annotation.league.normalization import effective_intake

        base = {
            "schema_version": 2,
            "time": 1,
            "slot_id": "left:row1",
            "entity": "unknown",
            "visibility": "visible",
            "acceptance": "accepted",
            "review_status": "unreviewed",
        }
        portrait = {**base, "id": "portrait", "field": "identity", "value": "Aatrox"}
        kda = {
            **base,
            "id": "kda",
            "field": "kda",
            "value": {"kills": 7, "deaths": 1, "assists": 2},
            "raw_text": "7/1/2",
            "review_status": "human-verified",
        }
        kills = {
            **base,
            "id": "kills",
            "field": "kills",
            "value": 1,
            "derived_from": ["kda"],
        }
        rows = effective_intake([portrait, kda, kills], {"kda"})
        self.assertEqual(rows[2]["value"], 7)
        self.assertEqual(rows[2]["entity"], "Aatrox")
        self.assertEqual(rows[2]["raw_text"], "7/1/2")
        rows = effective_intake(
            [{**portrait, "review_status": "rejected"}, kda, kills], {"portrait", "kda"}
        )
        self.assertEqual(rows[2]["entity"], "unknown")
        self.assertEqual(rows[2]["value"], 7)

    def test_unknown_slots_do_not_collapse_and_engine_confidence_is_not_acceptance(
        self,
    ):
        rows = []
        for slot, value in (("left:row1", 10), ("left:row2", 11)):
            rows.append(
                {
                    "id": slot,
                    "schema_version": 2,
                    "entity": "unknown",
                    "slot_id": slot,
                    "field": "level",
                    "time": 1,
                    "value": value,
                    "visibility": "visible",
                    "confidence": 0,
                    "acceptance": "accepted",
                    "frame_id": "f",
                }
            )
        states, conflicts = fuse_observations(rows)
        self.assertEqual(len(states), 2)
        self.assertTrue(all(s["status"] == "confirmed" for s in states))
        self.assertFalse(conflicts)
        rejected = {
            **rows[0],
            "entity": "Aatrox",
            "confidence": 0.99,
            "acceptance": "unknown",
        }
        self.assertEqual(fuse_observations([rejected])[0][0]["status"], "unobserved")

    def test_version_one_cannot_resume_extraction(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = Store(Path(temporary))
            atomic_json(store.path("manifest.json"), {"schema_version": 1})
            with self.assertRaisesRegex(ValueError, "Version-1 runs are read-only"):
                require_v2(store)

    def test_empty_benchmark_cannot_pass_or_hide_missing_families(self):
        result = score([], {}, {})
        self.assertFalse(result["passed"])
        self.assertTrue(all(not m["passed"] for m in result["families"].values()))
        field = {"f": {"reader": "number", "field": "level"}}
        frame = {
            "time": 1,
            "labels": [
                {
                    "field_id": "f",
                    "availability": "visible",
                    "value": 12,
                    "raw_text": "12",
                }
            ],
        }
        result = score([frame], {}, field)
        self.assertEqual(result["families"]["number"]["eligible"], 1)
        self.assertEqual(result["families"]["number"]["coverage"], 0)

    def test_benchmark_requires_items_spells_korean_and_difficult_cases(self):
        fields = {"f": {"reader": "number", "field": "level", "parser": "level"}}
        data = {
            "frames": [
                {
                    "split": split,
                    "labels": [
                        {"field_id": "f", "availability": "visible", "value": 12}
                    ],
                }
                for split in ("development", "held_out")
            ]
        }
        result = label_coverage(data, fields)
        self.assertFalse(result["passed"])
        self.assertIn("icon:item", result["missing_field_groups"]["held_out"])
        self.assertIn("korean_text", result["missing_field_groups"]["development"])
        self.assertIn("changing_selections", result["missing_cases"])


@unittest.skipUnless(Image, "Install requirements-intake.txt")
class IntakeIntegrationTests(unittest.TestCase):
    def store(self, temporary):
        store = Store(Path(temporary))
        atomic_json(
            store.path("manifest.json"),
            {
                "schema_version": 2,
                "source": {"sha256": "fixture", "path": "fixture.mp4", "fps": 60},
                "interval": [0, 120],
            },
        )
        config = Config(
            {
                "gameplay_patch": "26.13",
                "ddragon_version": "16.13.1",
                "cdragon_version": "16.13",
            },
            [],
        )
        return store, config

    def run_cli(self, store, config):
        from annotation.league.__main__ import main

        with intake_fixture(store, config), patch(
            "annotation.league.media.prepare", return_value=read_json(store.path("manifest.json"))
        ), patch("annotation.league.media.file_hash", return_value="fixture"), patch(
            "annotation.league.inference.processor_preview", return_value={}
        ), patch("annotation.league.inference.QwenRunner", side_effect=AssertionError("Qwen initialized")), patch(
            "annotation.league.knowledge.Resolver.handle_query", side_effect=AssertionError("mechanics queried")
        ), contextlib.redirect_stdout(io.StringIO()):
            atomic_json(store.path("layout_validation.json"), {})
            return main(["--run-dir", str(store.root), "--offline", "run", "--video", "fixture.mp4"])

    @unittest.skipUnless(
        shutil.which("ffmpeg"), "FFmpeg is needed for the real PTS test"
    )
    def test_pyav_uses_source_pts_and_decode_cache_survives_reader_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = Store(Path(temporary))
            video = store.path("source.mp4")
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc2=size=64x64:rate=4",
                    "-t",
                    "2",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    str(video),
                ],
                check=True,
            )
            frames = list(sampled_frames(video, 0.1, 1.8, 1))
            self.assertEqual([t for t, _ in frames], [0.25, 1.25])
            manifest = {
                "source": {"sha256": "fixture", "path": str(video), "fps": 4},
                "interval": [0, 2],
            }
            first = list(decoded_frames(store, manifest, 1, [0, 1]))
            with patch(
                "annotation.league.intake.sampled_frames",
                side_effect=AssertionError("Decoded twice"),
            ):
                second = list(decoded_frames(store, manifest, 1, [0, 1]))
            self.assertEqual([t for t, _ in first], [t for t, _ in second])
            self.assertEqual(len(list(store.path("decoded").rglob("*.png"))), 2)

    def test_review_calibration_versions_profile_and_preserves_old_runs(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("Optional review dependencies are absent")
        from annotation.league.review import create_app

        with tempfile.TemporaryDirectory() as temporary:
            store, config = self.store(temporary)
            with intake_fixture(store, config):
                source = store.path("preview.png")
                Image.new("RGB", (1920, 1080)).save(source)
                store.write(
                    "intake_previews",
                    [{"id": "preview", "path": "preview.png", "time": 1}],
                )
                with TestClient(create_app(store)) as client:
                    self.assertTrue(client.get("/api/layout").json()["editable"])
                    response = client.post(
                        "/api/layout",
                        json={
                            "frame_id": "preview",
                            "fields": {"identity": [2, 2, 12, 12]},
                        },
                    )
                    self.assertEqual(response.status_code, 200, response.text)
                    revised = Config.load(store.path("config.json"))
                    self.assertFalse(
                        load_profile(revised.layout_profile)["thresholds_frozen"]
                    )
                    self.assertNotEqual(revised.layout_profile, config.layout_profile)
                    manifest = read_json(store.path("manifest.json"))
                    manifest["schema_version"] = 1
                    atomic_json(store.path("manifest.json"), manifest)
                    self.assertFalse(client.get("/api/layout").json()["editable"])
                    self.assertEqual(
                        client.post(
                            "/api/layout", json={"frame_id": "preview", "fields": {}}
                        ).status_code,
                        422,
                    )

    def test_annotation_uses_restricted_vl_not_legacy_observer_or_mechanics(self):
        with tempfile.TemporaryDirectory() as temporary:
            store, config = self.store(temporary)
            with intake_fixture(store, config), patch(
                "annotation.league.inference.QwenRunner",
                side_effect=AssertionError("Qwen intake"),
            ):
                resolver = Mock()
                runner = Mock()
                runner.tokens = {}
                runner.peak_memory_bytes = 0
                runner.infer.side_effect = AssertionError("legacy generation called")
                result = annotate(store, config, runner=runner, resolver=resolver)
            self.assertEqual(result["frames"], 4)
            resolver.roster.assert_not_called()
            resolver.handle_query.assert_not_called()
            runner.infer.assert_not_called()
            self.assertTrue(store.path("stages/annotate.json").exists())
            self.assertTrue(
                any(c["status"] == "unsupported" for c in store.rows("coverage"))
            )
            self.assertEqual(store.rows("event_candidates"), [])

    def test_interrupt_resume_keeps_committed_tasks(self):
        with tempfile.TemporaryDirectory() as temporary:
            store, config = self.store(temporary)
            with intake_fixture(store, config, stop_after=1), self.assertRaises(
                StopRequested
            ):
                annotate(store, config)
            task_path = next(store.path("tasks").glob("*.json"))
            before = task_path.read_bytes()
            store.stopping = False
            with intake_fixture(store, config):
                result = annotate(store, config)
            self.assertEqual(result["frames"], 4)
            self.assertEqual(task_path.read_bytes(), before)

    def test_solver_or_mechanics_settings_do_not_invalidate_intake(self):
        with tempfile.TemporaryDirectory() as temporary:
            store, config = self.store(temporary)
            with intake_fixture(store, config):
                annotate(store, config)
            original = store.fingerprint(config)
            evaluation_key = store.stage_key("evaluate", config, [])
            config.thinking_output_tokens += 1000
            config.mechanics = True
            config.models["thinking"]["revision"] = "another"
            self.assertEqual(store.fingerprint(config), original)
            self.assertNotEqual(store.stage_key("evaluate", config, []), evaluation_key)
            profile = read_json(Path(config.layout_profile))
            profile["thresholds"]["number"] += 0.01
            atomic_json(Path(config.layout_profile), profile)
            self.assertNotEqual(store.fingerprint(config), original)

    def test_fractional_source_pts_remain_distinct_from_benchmark_sample_times(self):
        from annotation.league.intake_benchmark import benchmark
        with tempfile.TemporaryDirectory() as temporary:
            store,config=self.store(temporary)
            with intake_fixture(store,config,pts_offset=.013):
                annotate(store,config,sample_times=[0,30,60,90])
                records=store.rows("intake_records")
                self.assertEqual({r["sample_time"] for r in records},{0,30,60,90})
                self.assertTrue(all(abs(r["time"]-r["sample_time"]-.013)<1e-6 for r in records))
                data={"frames":[{"time":t,"split":"development" if t<60 else "held_out","labels":[{"field_id":"kills","availability":"visible","value":t//20,"raw_text":str(t//20)}]} for t in (0,30,60,90)]}
                with patch("annotation.league.intake_benchmark.load_labels",return_value=data):
                    result=benchmark(store,config,store.path("unused-labels.json"),draft=True)
                self.assertEqual(result["development"]["families"]["number"]["correct"],2)
                self.assertEqual(result["held_out"]["families"]["number"]["correct"],2)

    def test_numeric_threshold_change_reuses_vl_requests(self):
        import annotation.league.hybrid_intake as hybrid
        with tempfile.TemporaryDirectory() as temporary:
            store, config = self.store(temporary)
            with intake_fixture(store, config):
                annotate(store, config, sample_times=[0,30,60,90])
                profile=read_json(Path(config.layout_profile))
                profile["thresholds"]["number"] += .01
                atomic_json(Path(config.layout_profile),profile)
                with patch.object(hybrid.VLReader,"read_group",side_effect=AssertionError("Unrelated VL cache invalidated")):
                    annotate(store,config,sample_times=[0,30,60,90])
                self.assertEqual(len(store.rows("intake_records")), 24)

    def test_full_run_requires_benchmark_release(self):
        with tempfile.TemporaryDirectory() as temporary:
            store, config = self.store(temporary)
            with intake_fixture(store, config):
                for path in (Path(config.asset_cache) / "releases").glob("*.json"):
                    path.unlink()
                with self.assertRaisesRegex(ValueError, "passing frozen held-out"):
                    annotate(store, config)

    def test_full_run_requires_separate_runtime_pilot(self):
        with tempfile.TemporaryDirectory() as temporary:
            store, config = self.store(temporary)
            with intake_fixture(store, config):
                for path in (Path(config.asset_cache) / "releases").glob(
                    "*-pilot.json"
                ):
                    path.unlink()
                with self.assertRaisesRegex(ValueError, "five-minute runtime pilot"):
                    annotate(store, config)

    def test_run_command_uses_replacement_and_keeps_evaluation_optional(self):
        with tempfile.TemporaryDirectory() as temporary:
            store, config = self.store(temporary)
            config.evaluation_enabled = False
            self.assertEqual(self.run_cli(store, config), 0)
            self.assertTrue(store.path("stages/annotate.json").exists())
            self.assertTrue(store.path("stages/fuse.json").exists())
            self.assertTrue(store.path("report.json").exists())
            self.assertIn("trajectory_memory", read_json(store.path("question_coverage.json"))["unsupported_capabilities"])

    def test_run_with_no_eligible_questions_does_not_initialize_solver(self):
        with tempfile.TemporaryDirectory() as temporary:
            store, config = self.store(temporary)
            with patch("annotation.league.timeline.fuse_observations", return_value=([], [])):
                self.assertEqual(self.run_cli(store, config), 0)
            self.assertEqual(store.rows("questions"), [])
            self.assertTrue(store.path("stages/evaluate.json").exists())

    def test_native_records_and_normalized_kda_have_distinct_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            store, config = self.store(temporary)
            with intake_fixture(store, config):
                annotate(store, config)
            task = read_json(next(store.path("tasks").glob("*.json")))
            source = task["observations"][1]
            task["observations"].append(
                {
                    **source,
                    "id": "derived",
                    "field_id": source["field_id"] + "#kills",
                    "derived_from": [source["id"]],
                }
            )
            collect(store, [task], config)
            self.assertNotIn("derived", {r["id"] for r in store.rows("intake_records")})
            self.assertIn("derived", {r["id"] for r in store.rows("observations")})


if __name__ == "__main__":
    unittest.main()
