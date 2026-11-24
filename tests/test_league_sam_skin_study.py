from contextlib import nullcontext
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from annotation.league.body_frame_matching import SkinMatcher, skin_lock_scope
from annotation.league.core import StopRequested, atomic_json, digest, read_json
from annotation.league.media import file_hash
from annotation.league.sam_skin_score import SINGED, XAYAH, draft_reviews, load_matcher, load_reviews, regression, score
from annotation.league.sam_skin_study import MIDBEAST_APPEARANCES, find_appearances, historical_replay, prepare, verify
from annotation.league.sam_skin_tracking import ARMS, SAM21Video, point_seed, track, track_receipt
from annotation.league.sam_skin_verifier import frozen_cases, infer, prepare_shortlists, score_frozen

BASELINE = Path(__file__).resolve().parents[1] / "data/league_recognition_v1/midbeast-sam-siglip-35-215"


class FakeSAM:
    size = (80, 90)

    def __init__(self, interrupt_after=None, short=False):
        self.requests = []
        self.sessions = 0
        self.interrupt_after = interrupt_after
        self.short = short
        self.outputs = self.mask_output(7)

    @classmethod
    def mask_output(cls, identifier):
        mask = np.zeros((cls.size[1], cls.size[0]), dtype=bool)
        mask[26:71, 15:55] = True
        return {"out_obj_ids": [identifier], "out_binary_masks": [mask]}

    def handle_request(self, request):
        self.requests.append(request)
        kind = request["type"]
        if kind == "start_session":
            if self.interrupt_after is not None and self.sessions == self.interrupt_after:
                raise StopRequested("interrupted between appearances")
            self.sessions += 1
            self.indices = list(Path(request["resource_path"]).glob("*.jpg"))
            return {"session_id": str(self.sessions)}
        if kind == "add_prompt":
            return {"outputs": self.mask_output(request.get("obj_id", 7))}
        return {}

    def handle_stream_request(self, request):
        self.requests.append(request)
        index = request["start_frame_index"]
        if request["propagation_direction"] == "forward":
            indices = range(index, len(self.indices) - int(self.short))
        else:
            indices = range(index, -1, -1)
        for index in indices:
            yield {"frame_index": index, "outputs": self.outputs}


class FakeQwen:
    identity = {"checkpoint": {"files_hash": "fixture"}, "runtime": {"gpu_name": "fixture"}}

    def __init__(self, replies, *_, **kwargs):
        self.replies = iter(replies)
        if kwargs.get("image_limit") != 2:
            raise AssertionError("Qwen needs two panels")

    def complete(self, messages, schema, **kwargs):
        panel_types = [row["type"] for row in messages[0]["content"]]
        if panel_types != ["text", "image_url", "image_url"] or len(schema["required"]) != 3:
            raise AssertionError("Qwen needs one prompt and two images with two judgments and a reason")
        accepted = next(self.replies)
        value = ({"readable": "yes", "contamination": "none" if accepted else "substantial", "reason": "fixture"}
                 if accepted is not None else {"verdict": "clear"})
        return {"text": json.dumps(value), "finish_reason": "stop", "wall_seconds": 0.,
                "peak_memory_bytes": 0, "usage": {}}

    def close(self):
        pass


def sample_match(root, name, identifiers, lock_hash):
    directory = root / "inputs" / name
    directory.mkdir(parents=True)
    for filename in ("frame.png", "mask.png", "sam.jpg"):
        Image.new("RGB", FakeSAM.size, "red").save(directory / filename)
    frame_hash = file_hash(directory / "frame.png")
    jpeg_hash = file_hash(directory / "sam.jpg")
    entries = [{"id": f"view-{champion}", "champion": champion, "skin_id": skin, "skin_name": champion,
                "reference_kind": "rendered", "reviewer": "fixture", "reviewer_kind": "human",
                "verified": True, "identity_evidence": {"review": "fixture"}, "path": "frame.png",
                "mask_path": "mask.png", "sha256": frame_hash, "mask_sha256": file_hash(directory / "mask.png")}
               for champion, skin in (("Singed", "27006"), ("Xayah", "498038"))]
    atomic_json(directory / "gallery.json", entries)
    patch_version = ["16.22.1", "16.22"]
    pools = {"enemy": ["Singed", "Xayah"], "friendly": []}
    matcher = SkinMatcher(entries, root, root / "lock.json", lock_hash, patch_version, encoder_name="siglip")
    frames = []
    appearances = []
    for i, identifier in enumerate(identifiers):
        start = len(frames)
        observations = {}
        for position in range(5):
            index = len(frames)
            frames.append({"time": i * 2 + position / 16, "path": "frame.png", "sam_path": "sam.jpg",
                           "sha256": frame_hash, "sam_sha256": jpeg_hash, "proposals": [], "raw_proposals": []})
            observations[str(index)] = {"frame_index": index, "body_box": [10, 20, 65, 75],
                                        "bar_box": [13, 14, 57, 19], "level": 3}
        appearances.append({"id": identifier, "team": "enemy", "first": start, "last": start + 4,
                            "start": frames[start]["time"], "end": frames[-1]["time"],
                            "seed": observations[str(start + 2)], "observations": observations,
                            "seed_frames": list(range(start, start + 5)), "review_frames": [start, start + 4]})
    return {"id": name, "source_sha256": name, "sam": {"fixture": True}, "patch": patch_version,
            "profile": {"size": list(FakeSAM.size), "scene_box": [0, 0, *FakeSAM.size],
                        "body_binder": {"exclude_boxes": []}},
            "gallery_sha256": file_hash(directory / "gallery.json"), "gallery_coverage": {},
            "scope": skin_lock_scope(matcher, pools, name, f"{name}-roster"), "team_pools": pools,
            "appearances": appearances, "frames": frames}


def synthetic_frozen(root):
    image = root / "frozen.png"
    Image.new("RGB", (20, 20)).save(image)
    image_sha = file_hash(image)
    cases = [{"id": f"frozen-{i:02}", "frozen_id": f"{i:02}", "context": image.name, "masked": image.name,
              "context_sha256": image_sha, "masked_sha256": image_sha, "marker": "observed",
              "label": {"verdict": "clear" if i < 17 else "muddled" if i < 36 else "uncertain"}}
             for i in range(40)]
    inference = [True] * 13 + [False] * 4 + [True] + [False] * 18 + [None] * 4
    return {"cases": cases, "historical": {}, "appearances": {}, "appearance_report_sha256": "frozen"}, inference


class SamSkinStudyTests(unittest.TestCase):
    def test_sam31_refines_bound_id_with_separate_normalized_point_request(self):
        predictor = FakeSAM()
        prompt = {"type": "add_prompt", "session_id": "s", "frame_index": 2,
                  "text": "video game character", "bounding_boxes": [[.1, .2, .7, .6]]}
        obj, iou, _ = point_seed(predictor, prompt, {"key": "target", "box_xyxy": [10, 20, 65, 75]},
                                 points=[[40, 45], [60, 54]], size=FakeSAM.size, sam21=False)
        self.assertEqual(obj, 7)
        self.assertGreater(iou, .5)
        self.assertEqual(predictor.requests, [prompt, {
            "type": "add_prompt", "session_id": "s", "frame_index": 2, "points": [[.5, .5], [.75, .6]],
            "point_labels": [1, 0], "obj_id": 7, "rel_coordinates": True,
        }])

    def test_sam21_adapter_uses_pixel_box_points_and_reverse_tracking(self):
        class Predictor:
            def init_state(self, **kwargs):
                self.init = kwargs
                return {"video_width": 80, "video_height": 90}

            def add_new_points_or_box(self, state, **kwargs):
                self.seed = kwargs
                return 1, [1], np.asarray([FakeSAM.mask_output(1)["out_binary_masks"][0]], dtype=float)

            def propagate_in_video(self, state, **kwargs):
                self.tracking = kwargs
                yield self.add_new_points_or_box(state, **self.seed)

        model = Predictor()
        adapter = SAM21Video(model)
        adapter.handle_request({"type": "start_session", "resource_path": "frames"})
        prompt = {"type": "add_prompt", "session_id": "sam21", "frame_index": 1,
                  "bounding_boxes": [[.125, .2, .625, .6]]}
        obj, _, _ = point_seed(adapter, prompt, {"key": "target", "box_xyxy": [10, 18, 60, 72]},
                               points=[[40, 45], [64, 54]], size=FakeSAM.size, sam21=True)
        self.assertEqual(obj, 1)
        self.assertEqual(model.seed, {"frame_idx": 1, "obj_id": 1, "points": [[40, 45], [64, 54]],
                                      "labels": [1, 0], "box": [10, 18, 60, 72]})
        next(adapter.handle_stream_request({"start_frame_index": 1, "propagation_direction": "backward"}))
        self.assertEqual(model.tracking, {"start_frame_idx": 1, "reverse": True})
        adapter.handle_request({"type": "close_session"})
        self.assertIsNone(adapter.state)

    def test_completed_tracks_resume_and_short_streams_are_recorded(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            match = sample_match(root, "midbeast", ("first", "second"), "lock")
            study = {"matches": [match]}
            arm = ARMS[1]
            interrupted = FakeSAM(interrupt_after=4)
            with self.assertRaises(StopRequested):
                track(root, study, "study", arm, root, loader=lambda *_: (interrupted, {}, nullcontext))
            finished, _ = track_receipt(root, "study", match, arm, match["appearances"][0])
            pending, _ = track_receipt(root, "study", match, arm, match["appearances"][1])
            self.assertEqual(len(finished["seed_trials"]), 3)
            self.assertEqual(finished["selected_seed"], 2)
            self.assertEqual([row["frame_index"] for row in finished["samples"]], list(range(5)))
            self.assertIsNone(pending)
            resumed = FakeSAM(short=True)
            totals = track(root, study, "study", arm, root, loader=lambda *_: (resumed, {}, nullcontext))
            self.assertEqual(resumed.sessions, 3)
            self.assertEqual(totals["statuses"], {"completed": 1, "propagation_incomplete": 1})
            self.assertIn("backward", [row.get("propagation_direction") for row in resumed.requests])

    def test_end_to_end_paired_lock_scorecard(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            lock = root / "lock.json"
            atomic_json(lock, {"fixture": True})
            lock_hash = digest(read_json(lock))
            midbeast = sample_match(root, "midbeast", (*XAYAH, *SINGED), lock_hash)
            second = sample_match(root, "second", ("second-1", "second-2"), lock_hash)
            frozen, replies = synthetic_frozen(root)
            study = {"matches": [midbeast, second], "frozen_verifier": frozen, "siglip_lock_hash": lock_hash,
                     "historical_replay": {"assignments": {"enemy": {}}, "events": []}}
            args = SimpleNamespace(root=root, cache=root, siglip_lock=lock, frozen_only=False)
            self.assertFalse(score(args, study, "study")["complete"])
            for arm in ARMS:
                predictor = FakeSAM()
                if arm == ARMS[2]:
                    predictor.outputs = FakeSAM.mask_output(1)
                track(root, study, "study", arm, root, loader=lambda *_: (predictor, {}, nullcontext))
            manifest, pending = prepare_shortlists(root, study, "study")
            self.assertFalse(pending)
            path = root / "verification/shortlists.json"
            atomic_json(path, manifest)
            shortlist_sha = file_hash(path)
            output = draft_reviews(root, study, "study", manifest, shortlist_sha)
            review = read_json(Path(output["review_template"]))
            review.update(predictions_seen=False, reviewer="fixture")
            for match in review["matches"].values():
                for label in match["appearances"].values():
                    label.update(champion="Singed", skin_id="27006", reason="fixture")
                    for frame in label["frames"].values():
                        frame.update(status="visible", polygons=[[[15, 26], [54, 26], [54, 70], [15, 70]]])
            crops = [case for row in manifest["shortlists"] for case in row["cases"]]
            for case in crops:
                clear = (case["appearance_id"] not in XAYAH and
                         (case["match"] != "second" or case["frame_index"] % 5))
                review["crops"][case["id"]].update(verdict="clear" if clear else "muddled", reason="fixture")
                replies.append(bool(clear))
            review_path = root / "verification/reviews.json"
            atomic_json(review_path, review)
            self.assertEqual(load_reviews(root, study, "study", manifest, shortlist_sha)[0]["reviewer"], "fixture")
            self.assertEqual(score(args, study, "study")["pending"], ["frozen review receipt"])
            cases = [*frozen["cases"], *crops]
            verify_args = SimpleNamespace(root=root, cache=root, vl_python="driver", frozen_only=False, prepare_only=False)
            with patch("annotation.league.sam_skin_study.infer",
                       side_effect=lambda *a: infer(*a, runner_factory=lambda *p, **kw: FakeQwen(replies, *p, **kw))):
                self.assertTrue(verify(verify_args, study, "study")["complete"])
            reused = infer(root, "study", cases, root, "driver",
                           runner_factory=lambda *_a, **_kw: self.fail("All inference should be reused"))
            self.assertEqual(reused["reused"], len(cases))

            def score_views(_self, queries, entries):
                return np.array([[.9 if (path.parts[-4] in XAYAH) == (entry["champion"] == "Xayah") else .1
                                  for entry in entries] for path, *_ in queries])

            with patch.object(SkinMatcher, "score_views", score_views):
                result = score(args, study, "study")
            self.assertTrue(result["complete"])
            self.assertTrue(result["qwen_seed_ranking_ready"])
            report = read_json(root / "scorecard.json")
            comparison = report["matches"]["midbeast"]["arms"][ARMS[0]]
            self.assertEqual(comparison["ungated"]["review"]["counts"], {"correct": 1, "false": 1, "unscorable": 0})
            self.assertEqual(comparison["gated"]["review"]["counts"], {"correct": 1, "false": 0, "unscorable": 0})
            self.assertTrue(comparison["gated"]["regression"]["original_singed_supports"])
            self.assertEqual(comparison["geometry"]["target_coverage"]["mean"], 1.)
            self.assertEqual(comparison["geometry"]["contamination"]["mean"], 0.)
            frozen_score = report["frozen_verifier"]
            self.assertTrue(frozen_score["passes_threshold"])
            self.assertEqual(frozen_score["metrics"]["invalid_responses"], 4)
            for index, changes in ((0, {"readable": "no"}), (18, {"contamination": "none"})):
                prediction_path = root / "verification/predictions" / f"frozen-{index:02}.json"
                previous = read_json(prediction_path)
                atomic_json(prediction_path, {**previous, **changes})
                self.assertFalse(score_frozen(root, "study", frozen)["passes_threshold"])
                atomic_json(prediction_path, previous)
            (root / "verification/predictions/frozen-39.json").unlink()
            self.assertFalse(score_frozen(root, "study", frozen)["complete"])

            review["matches"]["second"]["source_sha256"] = "midbeast"
            atomic_json(review_path, review)
            with self.assertRaisesRegex(ValueError, "another match"):
                load_reviews(root, study, "study", manifest, shortlist_sha)
            review["matches"]["second"]["source_sha256"] = second["source_sha256"]
            review["reviewer"] = "changed after inference"
            atomic_json(review_path, review)
            with self.assertRaisesRegex(ValueError, "changed after the verifier"):
                score(args, study, "study")
            atomic_json(root / "inputs/second/gallery.json", [])
            with self.assertRaisesRegex(ValueError, "gallery checksum"):
                load_matcher(root, second, root, lock, lock_hash)

    def test_second_source_must_be_distinct(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            atomic_json(root / "appearances.json", {"recipe": {"source_sha256": "same"}})
            with self.assertRaisesRegex(ValueError, "distinct underlying match"):
                prepare(SimpleNamespace(root=root / "study", midbeast_run=root, midbeast_scoreboard=None,
                                        second_run=root, second_gallery=root / "gallery.json",
                                        second_scoreboard=root, second_appearance=["id"]))

    @unittest.skipUnless((BASELINE / "mask-verifier/reviews.json").is_file(), "Midbeast evidence is local")
    def test_real_frozen_inputs_and_saved_lock_replay(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prepared = read_json(BASELINE / "appearances.json")
            tracks = find_appearances(prepared, MIDBEAST_APPEARANCES)
            frozen = frozen_cases(root, BASELINE, prepared, tracks)
            cases = frozen["cases"]
            self.assertEqual([row["frozen_id"] for row in cases if row["marker"] == "interpolated"], ["07"])
            labels = [row["label"]["verdict"] for row in cases]
            indices = {label: [i for i, candidate in enumerate(labels) if candidate == label]
                       for label in ("clear", "muddled")}
            accepted = set(indices["clear"][:13] + indices["muddled"][:1])
            infer(root, "study", cases, root, "driver",
                  runner_factory=lambda *a, **kw: FakeQwen([i in accepted for i in range(40)], *a, **kw))
            self.assertTrue(score_frozen(root, "study", frozen)["passes_threshold"])
            self.assertEqual(frozen["historical"]["qwen35_9b"]["metrics"]["clean_retained"], 1)
            replay = historical_replay(BASELINE, frozen["scope"])
            self.assertTrue(regression(replay)["original_xayah_supports"])
            self.assertTrue(regression(replay)["original_singed_supports"])
            self.assertNotEqual(replay["current_report_sha256"], frozen["appearance_report_sha256"])


if __name__ == "__main__":
    unittest.main()
