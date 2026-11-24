from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np
from PIL import Image

from annotation.league.core import Config, Store, atomic_json, digest, write_rows
from annotation.league.killfeed import (
    ARTIFACTS, KIND, FeedPlayback, FeedTrackWriter, announcements, snapshot_dir,
)
from annotation.league.killfeed_reader import Entry
from annotation.league.killfeed_roster import Scoreboard, fill_roster
from annotation.league.media import file_hash
from annotation.league.questions import construct
from annotation.league.timeline import fuse


SOURCE = "a" * 64
PROVENANCE = {"source_sha256": SOURCE}
BOX = [0, 0, 160, 240]


def entry(y, killer="Ahri", victim="Darius", *, side="left", kind="candidate", pixels=1):
    roles = {name: {"champion": champion, "score": .9, "box": [0, y, 24, y+24]}
             for name, champion in (("killer", killer), ("victim", victim))}
    signature = np.random.default_rng(pixels).integers(0, 256, (2, 16, 16, 3), dtype=np.int16)
    return Entry(y, kind, side, [0, y, 160, y+37], roles, signature)


def frame_state(time, *, segment="s1", replay=False, connected=True):
    return {"playback_segment_id": segment, "replay": replay, "continuity_verified": connected and not replay,
            "clock": {"seconds": int(time), "observed_at": time}}


def witness(champion, time):
    return {"id": f"scoreboard-{champion}", "time": time}


class KillfeedTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.store = Store(Path(temporary.name))
        self.reader = Mock()
        self.reader.incoming.return_value = None
        self.writer = FeedTrackWriter(self.store.path("killfeed/revision"), SOURCE, BOX, self.reader)
        self.image = Image.new("RGB", (160, 240))

    def frame(self, time, entries, *, valid=True, **state):
        self.writer.update(time, self.image, entries, frame_state(time, **state), valid)

    def events(self):
        return announcements(self.writer.tracks, PROVENANCE)

    def test_stack_shifts_keep_one_track_and_repeated_pairs_remain_distinct(self):
        self.frame(0, [])
        self.frame(.25, [entry(0)])
        self.frame(.5, [entry(0, "Xayah", "Talon", pixels=2), entry(50)])
        self.frame(.75, [entry(0, "Xayah", "Talon", pixels=2), entry(50), entry(100)])
        self.frame(1, [entry(0), entry(50)])
        self.frame(1.25, [])
        self.frame(1.5, [entry(0)])
        ahri = [item for item in self.events() if item["actor"] == "Ahri"]
        self.assertEqual([(item["start"], item["end"]) for item in ahri], [(.25, 1), (.75, 1), (1.5, 1.5)])
        self.assertEqual([item["onset_bracket"] for item in ahri], [[0, .25], [.5, .75], [1.25, 1.5]])
        self.assertEqual(len({item["id"] for item in ahri}), 3)

    def test_faint_incoming_row_uses_earlier_crop_and_separate_confirmation(self):
        self.frame(0, [])
        self.frame(.25, [])
        faded = entry(50)
        self.reader.incoming.return_value = faded.box, Image.new("RGB", (160, 37)), faded.roles
        self.frame(.5, [faded])
        self.frame(.75, [faded])
        event, = self.events()
        self.assertEqual(event["start"], .25)
        self.assertEqual(event["onset_bracket"], [0, .25])
        self.assertEqual(event["roles"]["killer"]["confirmation"]["confirmed_at"], .75)

    def test_displaced_row_with_pixel_evidence_bridges_one_missed_sample(self):
        original, shifted = entry(50), entry(0)
        self.frame(0, [])
        self.frame(.25, [original])
        self.frame(.5, [])
        self.reader.incoming.return_value = shifted.box, Image.new("RGB", (160, 37)), shifted.roles
        self.frame(.75, [shifted])
        self.frame(1, [shifted])
        event, = self.events()
        self.assertEqual((event["start"], event["end"]), (.25, 1))
        self.assertEqual(len(event["evidence_ids"]), 4)

    def test_ambiguous_portrait_and_missing_side_cannot_fill(self):
        self.frame(1, [entry(0, side=None)])
        self.frame(1.25, [entry(0, killer="JarvanIV", side=None)])
        event, = self.events()
        self.assertIsNone(event["actor"])
        self.assertEqual(event["roles"]["killer"]["status"], "ambiguous")
        self.assertEqual(event["target"], "Darius")
        self.assertIsNone(event["roles"]["victim"]["confirmation"])
        rows, _ = fill_roster([event], Scoreboard({"left": {}, "right": {}}, False, "empty"), [0, 10], {})
        self.assertEqual(rows, [])

    def test_one_incidental_objective_match_cannot_hide_a_readable_kill(self):
        self.frame(0, [])
        self.frame(.25, [entry(0)])
        self.frame(.5, [entry(0)])
        self.frame(.75, [entry(0, kind="objective", killer=None, victim=None)])
        event, = self.events()
        self.assertEqual((event["actor"], event["target"], event["end"]), ("Ahri", "Darius", .75))

    def test_occlusion_cuts_and_replay_leave_onset_unknown_and_exclude_objectives(self):
        self.frame(0, [entry(0, victim=None)], connected=False)
        self.frame(.25, [], valid=False)
        self.frame(.5, [entry(0, victim=None)], segment="s2", connected=False)
        self.frame(.75, [entry(0, victim=None), entry(50, kind="objective", pixels=2)], segment="s2")
        self.frame(1, [entry(0, victim=None)], segment="s3", replay=True)
        events = self.events()
        self.assertEqual([event["start"] for event in events], [0, .5, 1])
        self.assertTrue(all(event["onset_bracket"] is None and event["target"] is None for event in events))
        self.assertTrue(events[-1]["replay"])
        self.assertIsNone(events[-1]["roles"]["killer"]["confirmation"])

    def test_playback_keeps_unknown_clock_spans_but_marks_rewind(self):
        playback = FeedPlayback(SOURCE)
        scene = Image.new("RGB", (1920, 1080))
        a = playback.update(0, (0, 70), scene, True)
        b = playback.update(.25, None, scene, True)
        c = playback.update(.5, (0, 70), scene, True)
        rewind = playback.update(.75, (.75, 65), scene, True)
        self.assertEqual(a["playback_segment_id"], b["playback_segment_id"])
        self.assertEqual(b["playback_segment_id"], c["playback_segment_id"])
        self.assertFalse(b["continuity_verified"])
        self.assertFalse(c["continuity_verified"])
        self.assertNotEqual(rewind["playback_segment_id"], c["playback_segment_id"])
        self.assertTrue(rewind["replay"])

    def test_future_scoreboard_supersedes_agreement_and_vetoes_disagreement(self):
        self.frame(1, [entry(0)])
        self.frame(1.25, [entry(0)])
        event, = self.events()
        scoreboard = Scoreboard({"left": {"Ahri": witness("Ahri", 5), "Darius": witness("Darius", 6)}, "right": {}},
                                False, "scoreboard")
        rows, conflicts = fill_roster([event], scoreboard, [0, 10], PROVENANCE)
        self.assertEqual(event["actor"], "Ahri")
        self.assertIsNone(event["target"])
        self.assertEqual(event["roles"]["victim"]["observed_champion"], "Darius")
        self.assertEqual(conflicts[0]["scoreboard_evidence_ids"], ["scoreboard-Darius"])
        self.assertEqual([(r["source"], r["start"], r["end"]) for r in rows if r["champion"] == "Ahri"],
                         [("killfeed", 1.25, 5), ("scoreboard", 5, 10)])

    def test_full_disputed_and_competing_rosters_abstain(self):
        self.frame(1, [entry(0), entry(50, "Xayah", "Talon", pixels=3)])
        self.frame(1.25, [entry(0), entry(50, "Xayah", "Talon", pixels=3)])
        events = self.events()
        fixed = {name: witness(name, 8) for name in ("Singed", "JarvanIV", "Leona", "Katarina")}
        scoreboard = Scoreboard({"left": fixed, "right": {}}, False, "scoreboard")
        rows, conflicts = fill_roster(deepcopy(events), scoreboard, [0, 10], {})
        self.assertFalse([row for row in rows if row["side"] == "left" and row["source"] == "killfeed"])
        self.assertEqual({row["reason"] for row in conflicts}, {"competing_feed_candidates"})
        scoreboard.disputed = True
        self.assertEqual(fill_roster(deepcopy(events), scoreboard, [0, 10], {})[0], [])
        scoreboard.disputed = False
        scoreboard.members["left"]["Alistar"] = witness("Alistar", 9)
        rows, conflicts = fill_roster(events, scoreboard, [0, 10], {})
        self.assertTrue(all(event["actor"] is None for event in events))
        self.assertEqual({row["reason"] for row in conflicts}, {"scoreboard_complete_team"})

    def test_bilateral_and_replay_candidates_do_not_fill(self):
        self.frame(1, [entry(0, victim=None)])
        self.frame(1.25, [entry(0, victim=None)])
        self.frame(1.5, [entry(0, victim=None, side="right")])
        self.frame(1.75, [entry(0, victim=None, side="right")])
        self.frame(2, [entry(0, killer="Xayah", victim=None)], segment="replay", replay=True)
        self.frame(2.25, [entry(0, killer="Xayah", victim=None)], segment="replay", replay=True)
        rows, conflicts = fill_roster(self.events(), Scoreboard({"left": {}, "right": {}}, False, "empty"), [0, 10], {})
        self.assertEqual(rows, [])
        self.assertEqual({row["reason"] for row in conflicts}, {"feed_both_teams"})
        scoreboard = Scoreboard({"left": {"Ahri": witness("Ahri", 5)}, "right": {}}, False, "scoreboard")
        rows, conflicts = fill_roster(self.events(), scoreboard, [0, 10], {})
        self.assertEqual([row["source"] for row in rows], ["scoreboard"])
        self.assertEqual({row["reason"] for row in conflicts}, {"feed_both_teams", "scoreboard_opposite_team"})

    def test_optional_fusion_keeps_repeated_entries_and_serves_native_crops(self):
        from annotation.league.captions import construct_captions
        from annotation.league.review import create_app
        from fastapi.testclient import TestClient

        config = Config({}, [])
        atomic_json(self.store.path("manifest.json"), {"source": {"sha256": SOURCE}, "interval": [0, 10]})
        legacy = {"id": "notice", "start": .5, "end": .5, "kind": "death", "actor": "Darius",
                  "target": None, "object": None, "location": None, "confidence": .9, "sampling_fps": 1,
                  "evidence_ids": ["source-death"], "routes": ["central_notice"], "task_ids": [], "criteria": []}
        self.store.write("event_candidates", [legacy])
        self.store.write("observations", [])
        self.assertEqual(fuse(self.store, config)["events"], 1)
        original, = self.store.rows("events")
        questions = construct([], [original])
        dependencies = ["observations", "event_candidates"]
        baseline = self.store.stage_key("fuse", config, dependencies)
        self.frame(0, [])
        self.frame(.25, [entry(0)])
        self.frame(.5, [])
        self.frame(.75, [entry(0)])
        events = self.events()

        recipe = {"source_sha256": SOURCE, "interval": [0, 10]}
        active = {"recipe": recipe, "revision": digest(recipe)[:20], "artifacts": {}}
        directory = snapshot_dir(self.store, active)
        for name, rows in zip(ARTIFACTS, (events, [], self.writer.rows)):
            path = directory / (name + ".jsonl")
            write_rows(path, rows)
            active["artifacts"][name] = file_hash(path)
        atomic_json(self.store.path("killfeed/active.json"), active)
        self.assertNotEqual(baseline, self.store.stage_key("fuse", config, dependencies))
        self.assertEqual(fuse(self.store, config)["events"], 3)
        self.assertEqual(fuse(self.store, config)["events"], 3)
        fused = self.store.rows("events")
        self.assertEqual([event["kind"] for event in fused], [KIND, "death", KIND])
        self.assertEqual(fused[1], original)
        self.assertEqual(construct([], fused), questions)
        captions, _ = construct_captions([], [fused[0], fused[2]], [], interval=[0, 10], source_sha256=SOURCE)
        self.assertEqual(captions, [])
        with TestClient(create_app(self.store)) as client:
            records = client.get("/api/records", params={"target": "killfeed_timeline"}).json()
            self.assertEqual(records["total"], 2)
            evidence = client.get(f'/api/evidence/{events[0]["id"]}').json()["frames"]
            self.assertEqual([item["time"] for item in evidence], [0, .25])
            self.assertEqual(client.get(f'/media/{evidence[-1]["path"]}').headers["content-type"], "image/png")
            reply = client.post("/api/corrections", json={"target": "events", "record_id": events[0]["id"],
                                "changes": {}, "review_status": "human-verified"})
            self.assertEqual(reply.status_code, 422)
        with (directory / "killfeed_timeline.jsonl").open("a") as stream:
            stream.write("corruption")
        with self.assertRaisesRegex(ValueError, "changed killfeed artifact"):
            fuse(self.store, config)
