import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from annotation.league.media import file_hash
from annotation.league.body_campaign import stable_proposal_bindings
from annotation.league.body_level import champion_level, level_proposals, read_digit_mask
from annotation.league.sam_appearances import (AppearanceLocks, choose_masks, materialize_clean_crops, pick_seed,
                                               propagate_appearance, rank_masks, receipt_valid, sampled_indices,
                                               stitch_fragments)


class AppearanceTests(unittest.TestCase):
    @staticmethod
    def observation(frame, x=0):
        return {"frame_index": frame, "body_box": [x, 4, x + 5, 10], "bar_box": [x, 0, x + 5, 3]}

    def test_excludes_player_and_requires_evidence_for_friendly_seed(self):
        frames = [{"id": str(i), "time": i / 16, "proposals": [], "player_bar": None} for i in range(4)]
        observations = [self.observation(i) for i in range(4)]
        self.assertEqual(pick_seed("friendly", observations, frames, set())[1], "player_binding_unresolved")
        frames[1]["player_bar"] = [15, 0, 20, 3]
        frames[2]["player_bar"] = observations[2]["bar_box"]
        self.assertEqual(pick_seed("friendly", observations, frames, set())[1], "player_excluded")
        frames[2]["player_bar"] = None
        self.assertEqual(pick_seed("friendly", observations, frames, set())[0]["frame_index"], 1)
        frames[1]["player_bar"] = None
        self.assertEqual(pick_seed("friendly", observations, frames, {"2"})[0]["frame_index"], 2)
        self.assertEqual(pick_seed("enemy", observations, frames, set())[0]["frame_index"], 1)

    def test_only_unique_same_team_fragments_stitch_and_barriers_split(self):
        frames = [{"time": i / 16} for i in range(10)]
        first, last, simultaneous = [
            {"team": "enemy", "seed_frame_index": at, "observations": [self.observation(at, x)]}
            for at, x in [(1, 0), (4, 3), (4, 6)]]
        self.assertEqual(len(list(stitch_fragments([first, last], frames, 10, set()))), 1)
        self.assertEqual(len(list(stitch_fragments([first, last], frames, 10, {3}))), 2)
        self.assertEqual(len(list(stitch_fragments([first, last, simultaneous], frames, 10, set()))), 3)

    def test_seed_proposal_requires_a_readable_level_badge(self):
        image = Image.new("RGB", (170, 50), (48, 25, 25))
        draw = ImageDraw.Draw(image)
        draw.line((25, 15, 25, 22), fill=(240, 240, 240))
        draw.point((24, 16), fill=(240, 240, 240))
        cfg = {"body_binder": {"level_cell": [-25, -7, -3, 18]}}
        champion = {"bar_box": [40, 12, 145, 22]}
        unlabeled = {"bar_box": [60, 12, 165, 22]}
        with patch("annotation.league.body_level.read_digit_mask", return_value=1) as read:
            self.assertEqual(level_proposals(image, cfg, [champion, unlabeled]), [{**champion, "level": 1}])
            self.assertEqual(read.call_count, 1)
        with patch("annotation.league.body_level.read_digit_mask", return_value=None):
            self.assertIsNone(champion_level(image, champion["bar_box"], cfg["body_binder"]["level_cell"]))

    def test_digit_ocr_and_consecutive_level_consistency(self):
        glyph = b"\x00" * 16
        for confidence, numeral, expected in [(90, "18", 18), (42, "3", None), (90, "19", None), (90, "O", None)]:
            response = SimpleNamespace(stdout=f"5\t1\t{confidence}\t{numeral}\n".encode())
            with patch("annotation.league.body_level.subprocess.run", return_value=response):
                read_digit_mask.cache_clear()
                self.assertEqual(read_digit_mask(glyph, (2, 8), "ocr"), expected)
        profile = {"body_binder": {"confirm_samples": 3, "maximum_step_pixels": 10}}
        frames = [{"id": str(i), "time": i / 16} for i in range(5)]
        box = self.observation(0)
        proposals = [[{"bar_box": box["bar_box"], "actor_region": box["body_box"], "team": "enemy", "level": level}]
                     for level in (4, 2, 2, 3, 3)]
        bindings = stable_proposal_bindings(frames, profile, proposals)
        self.assertEqual([row["level"] for row in bindings[0]["observations"]], [2, 2, 3, 3])

    def test_mask_quality_and_temporal_spacing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = []
            for index, t in enumerate((0, .05, .1, .3, .6, .9, 1.2, 1.5)):
                mask = Image.new("L", (20, 20))
                draw = ImageDraw.Draw(mask)
                draw.rectangle((1, 1, 11 - index, 11), fill=255)
                if index == 7:
                    draw.point((19, 19), fill=255)
                path = root / f"{index}.png"
                mask.save(path)
                rows.append({"crop_mask_path": path.name, "crop_mask_sha256": file_hash(path),
                             "frame_index": index, "time": t})
            selected = choose_masks(rows, root)
            self.assertEqual([row["frame_index"] for row in selected], [0, 3, 4, 5, 6])
            self.assertEqual(choose_masks(rows[:3], root)[0]["foreground_fraction"], 1)
            self.assertEqual(len(choose_masks(rows[:3], root)), 3)

    def test_majority_vote_and_conflicting_lock(self):
        views = [{"skin_name": "Alpha"}, {"skin_name": "Beta"}]
        matcher = SimpleNamespace(entries={"C": {"1": views[:1], "2": views[1:]}},
                                  score_views=lambda _, __: np.array([[.8, .7], [.9, .8], [.8, .9]]))
        ranked, vote = rank_masks(matcher, [None] * 3, ["C"])
        self.assertEqual((ranked[0]["skin_id"], vote["mask_wins"]), ("1", 2))
        matcher.score_views = lambda _, __: np.array([[.8, .8], [.9, .8], [.85, .9]])
        self.assertIsNone(rank_masks(matcher, [None] * 3, ["C"])[1])

        locks = AppearanceLocks()
        rows = [{"id": str(i), "team": "enemy", "start": start, "end": start + 1}
                for i, start in enumerate((0, 1.25, 2.5, 5, 7))]
        locks.observe(rows[0], vote)
        locks.observe(rows[0], vote)
        locks.observe(rows[1], vote)
        self.assertEqual(locks.assigned["enemy"], {})
        locks.observe(rows[2], vote)
        self.assertEqual(locks.assigned["enemy"]["C"]["supporting_appearances"], ["0", "2"])
        other = {**vote, "skin_id": "2"}
        locks.observe(rows[3], other)
        self.assertEqual(locks.assigned["enemy"], {})
        locks.observe(rows[4], other)
        self.assertEqual(locks.assigned["enemy"]["C"]["supporting_appearances"], ["3", "4"])
        self.assertEqual([event["event"] for event in locks.events], ["locked", "suspended", "locked"])

    def test_bidirectional_session_keeps_seed_once_and_verifies_resume(self):
        class Predictor:
            def __init__(self, lose_forward=False):
                self.requests = []
                self.lose_forward = lose_forward

            def handle_request(self, request):
                self.requests.append(request)
                return {"session_id": "t", "outputs": outputs}

            def handle_stream_request(self, request):
                self.requests.append(request)
                forward = request["propagation_direction"] == "forward"
                for n in (1, 2) if forward and self.lose_forward else (1, 2, 3) if forward else (1, 0):
                    yield {"frame_index": n, "outputs": outputs}
                if forward and self.lose_forward:
                    raise RuntimeError("No points are provided; please add points first")

        mask = np.zeros((20, 20), dtype=bool)
        mask[5:13, 3:10] = True
        outputs = {"out_binary_masks": [mask], "out_obj_ids": [7]}
        profile = {"size": [20, 20], "scene_box": [0, 0, 20, 20], "body_binder": {"exclude_boxes": []}}
        appearance = {"first": 0, "last": 3, "seed": {"frame_index": 1, "body_box": [3, 5, 10, 13]},
                      "id": "one", "team": "enemy"}
        self.assertEqual(sampled_indices(appearance, 4), [0, 1, 3])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "frame.png"
            Image.new("RGB", (20, 20), "red").save(path)
            sam = root / "frame.jpg"
            Image.new("RGB", (20, 20), "red").save(sam)
            frames = [{"time": i / 16, "path": path.name, "sha256": file_hash(path),
                       "sam_path": sam.name, "sam_sha256": file_hash(sam), "proposals": []} for i in range(4)]
            predictor = Predictor()
            run = propagate_appearance(predictor, root, root, frames, appearance, profile, 16)
            self.assertEqual(run["status"], "completed")
            self.assertEqual([row["frame_index"] for row in run["samples"]], [0, 1, 2, 3])
            self.assertEqual(sum(req["type"] == "add_prompt" for req in predictor.requests), 1)
            self.assertEqual(predictor.requests[1]["text"], "video game character")
            self.assertEqual([req["propagation_direction"] for req in predictor.requests[2:4]], ["forward", "backward"])
            self.assertEqual(predictor.requests[-1]["type"], "close_session")
            receipt = {**run, "manifest_sha256": "prepared", "sam": {"pin": 1}}
            self.assertTrue(receipt_valid(receipt, root, "prepared", {"pin": 1}))
            self.assertFalse(receipt_valid(receipt, root, "other"))
            crops = materialize_clean_crops(root, root, run["samples"], frames, profile)
            with Image.open(root / crops[0]["image_path"]) as crop:
                self.assertEqual(crop.size, (7, 8))
                self.assertEqual(crop.getpixel((0, 0)), (255, 0, 0))
            (root / run["samples"][0]["mask_path"]).write_text("corrupt")
            self.assertFalse(receipt_valid(receipt, root, "prepared"))
            partial = propagate_appearance(Predictor(lose_forward=True), root, root, frames, appearance, profile, 16)
            self.assertEqual(partial["status"], "propagation_incomplete")
            self.assertEqual(partial["lost_directions"], ["forward"])
            self.assertEqual([row["frame_index"] for row in partial["samples"]], [0, 1, 2])
