import unittest

from annotation.league.luna_evidence import (
    cast_bar_onsets,
    fresh_timeline_events,
    scoreboard_relationship_by_champion,
    scoreboard_rosters,
    scoreboard_team_relationships,
    video_knowledge,
)


class LunaEvidenceTests(unittest.TestCase):
    def test_team_relationships_follow_the_selected_hud_owner(self):
        teams = scoreboard_rosters([
            {"roster_id": "match", "side": "right", "champion": "Darius"},
            {"roster_id": "match", "side": "right", "champion": "Talon"},
            {"roster_id": "match", "side": "left", "champion": "JarvanIV"},
            {"roster_id": "match", "side": "left", "champion": "Leona"},
            {"roster_id": "match", "side": "left", "champion": "Talon", "review_status": "rejected"},
        ])
        self.assertEqual(scoreboard_team_relationships(teams, ["Darius", None, "Darius"]), {
            "roster_id": "match", "viewpoint_champion": "Darius",
            "allies": ["Talon"], "opponents": ["JarvanIV", "Leona"],
        })
        self.assertEqual(scoreboard_team_relationships(teams, ["JarvanIV"]), {
            "roster_id": "match", "viewpoint_champion": "JarvanIV",
            "allies": ["Leona"], "opponents": ["Darius", "Talon"],
        })

    def test_team_relationships_require_an_unambiguous_owner_and_roster(self):
        teams = {"match": {"left": ["Leona"], "right": ["Darius"]}}
        self.assertIsNone(scoreboard_team_relationships(teams, []))
        self.assertIsNone(scoreboard_team_relationships(teams, ["Teemo"]))
        self.assertIsNone(scoreboard_team_relationships(teams, ["Darius", "Leona"]))
        self.assertIsNone(scoreboard_team_relationships({**teams, "other": teams["match"]}, ["Darius"]))

    def test_per_champion_labels_preserve_self_and_omit_conflicting_members(self):
        teams = scoreboard_rosters([
            {"side": "right", "champion": "Darius"},
            {"side": "right", "champion": "Katarina"},
            {"side": "left", "champion": "JarvanIV"},
            {"side": "left", "champion": "Leona"},
            {"side": "right", "champion": "Leona"},
        ])
        self.assertEqual(scoreboard_relationship_by_champion(scoreboard_team_relationships(teams, ["Darius"])), {
            "Darius": "self", "Katarina": "ally", "JarvanIV": "opponent",
        })
        self.assertEqual(scoreboard_relationship_by_champion(scoreboard_team_relationships(teams, ["JarvanIV"])), {
            "JarvanIV": "self", "Darius": "opponent", "Katarina": "opponent",
        })
        self.assertIsNone(scoreboard_relationship_by_champion(scoreboard_team_relationships(teams, ["Leona"])))

    def test_fresh_timeline_events_use_onset_over_display_duration(self):
        events = [
            {"id": "carried", "kind": "kill_announcement", "start": 14, "end": 18,
             "onset_bracket": [13.75, 14]},
            {"id": "uncertain", "kind": "kill_announcement", "start": 17, "end": 21,
             "onset_bracket": None},
            {"id": "straddles-start", "kind": "kill_announcement", "start": 15, "end": 19,
             "onset_bracket": [14.75, 15]},
            {"id": "fresh", "kind": "kill_announcement", "start": 29.75, "end": 34,
             "onset_bracket": [29.5, 29.75]},
            {"id": "straddles-end", "kind": "kill_announcement", "start": 30, "end": 34,
             "onset_bracket": [29.75, 30]},
            {"id": "hud", "kind": "hud_slot_activation", "start": 15, "end": 15.0625,
             "onset_frame_ids": ["ready", "cooling"], "slot_id": "W"},
            {"id": "bar", "kind": "observed_channel", "start": 25, "end": 31,
             "evidence_ids": ["cast-start", "cast-continue"]},
            {"id": "in-progress", "kind": "observed_channel", "start": 15, "end": 19,
             "evidence_ids": ["first-readable"]},
        ]
        fresh = fresh_timeline_events(events, 15, 30, {"cast-start": [24.9375, 25]})
        assert [cue["id"] for cue in fresh] == ["hud", "bar", "fresh"]
        assert [cue["onset_bracket"] for cue in fresh] == [[15, 15.0625], [24.9375, 25], [29.5, 29.75]]
        assert fresh_timeline_events(events, 30, 45, {}) == []


    def test_cast_bar_onsets_require_contiguous_readable_transition_for_same_owner(self):
        reading = {"owner_id": "selected", "owner_epoch": "first", "segment_id": "one", "catalog_id": "reference"}
        readings = [
            {**reading, "id": "already-visible", "time": 10, "visibility": "readable"},
            {**reading, "id": "absent", "time": 10.0625, "visibility": "absent"},
            {**reading, "id": "onset", "time": 10.125, "visibility": "readable"},
            {**reading, "id": "absent-before-gap", "time": 10.1875, "visibility": "absent"},
            {**reading, "id": "after-gap", "time": 12, "visibility": "readable"},
            {**reading, "id": "absent-before-switch", "time": 12.0625, "visibility": "absent"},
            {**reading, "id": "after-switch", "time": 12.125, "visibility": "readable", "owner_epoch": "second"},
            {**reading, "id": "after-switch-back", "time": 12.1875, "visibility": "readable"},
        ]
        assert cast_bar_onsets(readings) == {"onset": [10.0625, 10.125]}


    def test_reference_uses_roster_and_observed_item_types_without_inventories(self):
        manifest = {"source": {"path": "video.mp4", "sha256": "source-hash"}}
        roster = [{"champion": "Darius", "side": "right"}, {"champion": "Darius", "side": "right"},
                  {"champion": "Singed", "side": "left"}]
        catalog = {"reference_version": "16.22.1", "locale": "en_US", "champions": {
            "Darius": {"name": "Darius", "abilities": [{"slot": "Q", "name": "Decimate", "description": "A swing."}]},
            "KhaZix": {"name": "Kha'Zix", "abilities": []},
        }}
        items = {"reference_version": "16.22.1", "items": [
            {"id": 7, "name": "Shield item", "description": "Grants a shield."},
            {"id": 8, "name": "Unobserved item", "description": "Grants a stat."},
        ]}
        inventory = [{"route": "scoreboard", "champion": "Darius", "time": 800,
                      "slots": [{"item_id": 7, "status": "recognized"}]}]
        reference = video_knowledge(manifest, roster, catalog, items, inventory)
        assert "source-hash" in reference and "Rift Scuttler" in reference
        assert "scoreboard right · Darius" in reference and "scoreboard left · Singed" in reference
        assert "Decimate" in reference and "Kha'Zix" not in reference
        assert "Shield item" in reference and "Unobserved item" not in reference and "800" not in reference
