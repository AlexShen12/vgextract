import unittest
from fractions import Fraction

from annotation.league.match_report import SourceFrames, Timeline


class MatchReportTimingTests(unittest.TestCase):
    def test_sample_rounding_preserves_native_pts_without_advancing_captions(self):
        frames = SourceFrames(Fraction(1, 60000), [0, 1001, 2002], 3003)
        self.assertEqual(frames.tick(.0167, sampled=True), 1001)
        self.assertEqual(frames.tick(.0167), 1002)

    def test_overlapping_cues_start_on_evidence_and_clear_at_scope_changes(self):
        frames = SourceFrames(Fraction(1, 100), list(range(0, 240, 5)), 240)
        timeline = Timeline.__new__(Timeline)
        timeline.frames = frames
        timeline.scope = lambda tick: ("scene", "first" if tick < 80 else "second")
        events = [
            {
                "id": "announcement", "kind": "kill_announcement", "start": .30, "end": .90,
                "segment_id": "scene", "actor": "Ahri", "target": "Darius",
            },
            {
                "id": "activation", "kind": "hud_slot_activation", "start": .45, "end": .50,
                "onset_frame_ids": ["before", "after"], "confirmed_readings": [{"time": .50}],
                "segment_id": "scene", "owner_epoch": "first", "slot_id": "Q",
            },
        ]
        cues, _, first_activation = timeline.event_tracks(events, [])
        self.assertEqual(first_activation, (50, "activation"))
        self.assertEqual([r["id"] for r in cues.at(45)], ["announcement"])
        self.assertEqual([r["id"] for r in cues.at(50)], ["announcement", "activation"])
        self.assertIn("onset unknown", cues.at(50)[0]["label"])
        self.assertEqual([r["id"] for r in cues.at(80)], ["announcement"])
        self.assertEqual([r["id"] for r in cues.at(90)], ["announcement"])
        self.assertEqual(cues.at(95), ())


if __name__ == "__main__":
    unittest.main()
