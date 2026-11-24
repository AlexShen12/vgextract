import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from annotation.league.body_identity import BodyGallery, TrackIdentity, run_arms
from annotation.league.body_campaign import coverage, initialize, stable_healthbar_bindings
from annotation.league.recognition_metrics import body_score, identity_metrics, source_aligned


def references():
    return [{"id":str(i),"champion":champion,"reference_kind":kind,"verified":True,"reviewer":"source reviewer",
             "reviewer_kind":"human","identity_evidence":{"source":"independent source pixels"},
             "sha256":"a"*64,"skin_id":str(i),"match_group":"gallery" if kind=="gameplay" else None}
            for i,(champion,kind) in enumerate((("A","gameplay"),("A","rendered"),("B","gameplay"),("B","rendered")))]


class BodyIdentity(unittest.TestCase):
    def setUp(self):
        self.entries=references();self.vectors=[[1,0],[1,0],[0,1],[0,1]]
        self.policy={"score":.8,"margin":.1,"development_freeze":"dev-hash"}

    def test_two_skin_variants_do_not_compete_as_two_champions(self):
        gallery=BodyGallery(self.entries,self.vectors,encoder_signature="pin")
        row=gallery.match([1,0],self.policy)
        self.assertEqual(row["champion"],"A");self.assertEqual(row["margin"],1)
        self.assertIsNone(row["skin_id"]);self.assertFalse(row["scores_are_probabilities"])

    def test_leaked_or_unreviewed_gallery_rejected(self):
        with self.assertRaises(ValueError):BodyGallery(self.entries,self.vectors,encoder_signature="pin",excluded_groups=["gallery"])
        with self.assertRaises(ValueError):BodyGallery([{**self.entries[0],"verified":False}],[[1,0]],encoder_signature="pin")
        gallery=BodyGallery(self.entries,self.vectors,encoder_signature="pin")
        with self.assertRaises(ValueError):gallery.match([1,0],{"score":.8,"margin":.1})
        self.assertIsNone(gallery.match([1,1],self.policy)["champion"])

    def test_three_votes_conflicts_and_boundaries(self):
        track=TrackIdentity()
        def sample(t,**kw):return {"track_id":"track","segment_id":"segment","owner_epoch":0,"time":t,"visible":True,**kw}
        for t in (0,.25):self.assertIsNone(track.update(sample(t),{"champion":"A"})["champion"])
        self.assertEqual(track.update(sample(.5),{"champion":"A"})["champion"],"A")
        self.assertIsNone(track.update(sample(.75),{"champion":"B"})["champion"])
        self.assertIsNone(track.update(sample(1,segment_id="cut"),{"champion":"B"})["champion"])
        with self.assertRaises(ValueError):track.update(sample(1),{"champion":"A"})

    def test_same_tracks_and_embeddings_across_six_arms(self):
        samples=[{"id":str(i),"track_id":"t","segment_id":"s","owner_epoch":0,"time":i/4,"visible":True,
                  "team":"friendly","accepted_roster":["A","B"]} for i in range(3)]
        prepared={"samples":samples,"encoder_signature":"pin","evaluation_match_groups":["heldout"]}
        with tempfile.TemporaryDirectory() as root:
            result=run_arms(Path(root),prepared,self.entries,self.vectors,[[1,0]]*3,self.policy)
            self.assertEqual(len(result["arms"]),6)
            self.assertEqual([r["named"] for r in result["arms"]],[1]*6)
            self.assertFalse(result["release_passed"])

    def test_false_unknown_and_wrong_team_reduce_precision(self):
        labels=[{"id":"a","champion":"A","visibility":"readable","team":"friendly"},
                {"id":"b","champion":None,"visibility":"unknown","team":"enemy"}]
        rows=[{"id":"a","champion":"A","team":"enemy","arm":"arm","track_id":"t","segment_id":"s","owner_epoch":0,"time":0},
              {"id":"b","champion":"B","team":"enemy","arm":"arm","track_id":"u","segment_id":"s","owner_epoch":0,"time":0}]
        result=body_score(labels,rows)["arm"]
        self.assertEqual(result["friendly"]["precision"],0);self.assertEqual(result["enemy"]["false_matches_on_unknown"],1)
        self.assertFalse(identity_metrics([],[])["passed"])

    def test_source_boxes_keep_unmatched_labels_in_coverage_and_match_once(self):
        frame = {"source_sha256": "source", "time": 2.5}
        label = {**frame, "id": "independent", "champion": "A", "visibility": "readable", "team": "friendly",
                 "body_box": [10, 10, 30, 30]}
        labels = [label, {**label, "id": "missed", "time": 3}]
        prediction = {**label, "id": "generated", "arm": "arm", "track_id": "t", "segment_id": "s", "owner_epoch": 0}
        candidates = [prediction, {**prediction, "id": "duplicate", "body_box": [11, 10, 31, 30]},
                      {**prediction, "id": "wrong-source", "source_sha256": "other"}]
        matched = [row for _, row in source_aligned(labels, candidates)]
        self.assertEqual([(row["id"], row["localization_iou"]) for row in matched], [("independent", 1.0)])
        metrics = body_score(labels, matched)["arm"]["friendly"]
        self.assertEqual((metrics["coverage"], metrics["localization_misses"]), (.5, 1))

    def test_out_of_gallery_label_is_a_negative(self):
        label = {"id": "x", "champion": "Outside", "visibility": "readable", "negative_kind": "out_of_gallery"}
        metrics = identity_metrics([label], [{"id": "x", "champion": "Outside"}])
        self.assertEqual((metrics["eligible"], metrics["correct"], metrics["false_matches_on_unknown"]), (0, 0, 1))

    def test_empty_campaign_cannot_claim_measured_quality(self):
        with tempfile.TemporaryDirectory() as root:
            result=coverage(initialize(root))
            self.assertEqual(result["segment_gap"],300);self.assertFalse(result["quality_measured"])

    def test_unknown_placeholders_never_become_identity_votes_or_precision_credit(self):
        track=TrackIdentity()
        for i in range(5):
            self.assertIsNone(track.update({"track_id":"t","time":i/4,"segment_id":"s","owner_epoch":0,"visible":True},{"champion":"Unknown"})["champion"])
        metrics=identity_metrics([{"id":"x","champion":"Unknown","visibility":"readable"}],[{"id":"x","champion":"Unknown"}])
        self.assertEqual(metrics["accepted"],0);self.assertEqual(metrics["eligible"],0);self.assertFalse(metrics["passed"])

    def test_friendly_and_enemy_proposals_remain_anonymous_and_exclude_bars(self):
        from PIL import Image,ImageDraw
        from annotation.league.action_v3 import load_profile
        from annotation.league.body_identity import healthbar_proposals
        profile=load_profile()
        for color,team in (((70,180,70),"friendly"),((70,70,180),"friendly"),((180,70,70),"enemy")):
            with self.subTest(color=color):
                image=Image.new("RGB",(1920,1080),(30,30,30));draw=ImageDraw.Draw(image)
                draw.rectangle((700,220,804,235),fill=(8,8,8));draw.rectangle((700,223,783,232),fill=color)
                draw.rectangle((675,216,697,241),fill=(10,20,30));draw.rectangle((685,221,687,230),fill=(220,220,220))
                found=healthbar_proposals(image,profile)
                self.assertTrue(found);self.assertEqual({r["team"] for r in found},{team})
                self.assertTrue(all(r["champion"] is None and r["actor_region"][1]>r["bar_box"][3] for r in found))

    def test_stable_bindings_retry_a_camera_jump_and_keep_the_other_team(self):
        from PIL import Image

        def bar(x, team):
            return {"bar_box": [x, 100, x + 105, 110], "actor_region": [x, 115, x + 105, 200], "team": team}

        profile = {"body_binder": {"confirm_samples": 3, "maximum_step_pixels": 120}}
        observations = [[bar(x, "friendly"), bar(120, "enemy")] for x in (100, 98, 96, 500, 503, 501)]
        with tempfile.TemporaryDirectory() as root:
            image = Path(root) / "frame.png"
            Image.new("RGB", (1, 1)).save(image)
            frames = [{"id": str(i), "time": i / 16, "path": image} for i in range(len(observations))]
            with patch("annotation.league.body_identity.healthbar_proposals", side_effect=observations):
                bindings = stable_healthbar_bindings(frames, profile)
        self.assertEqual(
            [(row["team"], row["seed_frame_index"], row["confirmation_frame_index"], row["end_frame_index"])
             for row in bindings],
            [("friendly", 0, 2, 3), ("enemy", 0, 2, 6), ("friendly", 3, 5, 6)],
        )
        self.assertEqual(bindings[-1]["evidence_frame_ids"], ["3", "4", "5"])
        self.assertEqual([row["frame_index"] for row in bindings[-1]["observations"]], [3, 4, 5])

    def test_one_champion_bar_with_offset_fill_rows_creates_one_proposal(self):
        from PIL import Image, ImageDraw
        from annotation.league.action_body import candidates
        from annotation.league.action_v3 import load_profile

        image = Image.new("RGB", (1920, 1080), (30, 30, 30))
        draw = ImageDraw.Draw(image)
        for x in (700, 900):
            draw.rectangle((x - 30, 209, x + 115, 240), fill=(8, 8, 8))
            draw.rectangle((x + 11, 220, x + 85, 220), fill=(70, 180, 70))
            draw.rectangle((x, 224, x + 95, 232), fill=(70, 180, 70))
            draw.rectangle((x - 10, 221, x - 7, 229), fill=(220, 220, 220))
        bars, _ = candidates(image, load_profile())
        self.assertEqual([row["bar_box"][:2] for row in bars], [[711, 220], [911, 220]])
