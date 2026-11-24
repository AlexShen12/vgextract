import tempfile
import unittest
import shutil
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from annotation.league.core import atomic_json, read_json, write_rows
from annotation.league.cast_bars import AbilityCatalog, CastBarReader, channel_occurrences, correct_reading, rebuild
from annotation.league.cast_bar_benchmark import reader_contract, score_occurrences, validate_predictions
from annotation.league.recognition_dataset import initialize, coverage, validate, validate_label, save_label


def catalog(root):
    data={"reference_version":"16.22.1","status":"complete_all","champions":{
        "Darius":{"abilities":[{"slot":"E","id":"Darius:E","name":"Apprehend","description":"Sweeps enemies."},
                                {"slot":"R","id":"Darius:R","name":"Noxian Guillotine","description":"Leaps."}]},
        "Other":{"abilities":[{"slot":"Q","id":"Other:Q","name":"Shared name","description":"Other."}]},
        "Third":{"abilities":[{"slot":"Q","id":"Third:Q","name":"Shared name","description":"Third."}]}}}
    path=root/"catalog.json";atomic_json(path,data);return AbilityCatalog(path)


def reading(t=0, **updates):
    return {"id":str(t),"time":t,"owner_id":"selected","owner_epoch":0,"segment_id":"s1",
            "catalog_id":"catalog","visibility":"readable","raw_text":"Apprehend","channel_kind":"ability",
            "ability_id":"Darius:E","progress_fraction":.5,"frame_id":"f"+str(t),**updates}


class CastBars(unittest.TestCase):
    def setUp(self):
        directory=tempfile.TemporaryDirectory();self.addCleanup(directory.cleanup)
        self.root=Path(directory.name);self.catalog=catalog(self.root)

    def test_exact_names_punctuation_and_owner_conflicts(self):
        self.assertEqual(self.catalog.resolve(" NOXIAN—Guillotine ","Darius")["ability_id"],"Darius:R")
        self.assertEqual(self.catalog.resolve("Apprehend","Other")["resolution"],"champion_conflict")
        self.assertIsNone(self.catalog.resolve("Shared name")["ability_id"])
        self.assertEqual(self.catalog.resolve("Shared name","Other")["ability_id"],"Other:Q")
        self.assertEqual(self.catalog.resolve("Recall","Darius")["channel_kind"],"recall")
        self.assertIsNone(self.catalog.resolve("Apprehencl","Darius")["ability_id"])

    def test_fuzzy_requires_freeze_and_reviewed_alias(self):
        with self.assertRaises(ValueError):self.catalog.resolve("Apprehencl",fuzzy={"score":.8,"margin":.1})
        self.assertEqual(self.catalog.resolve("Apprehencl",fuzzy={"score":.8,"margin":.1,"development_freeze":"hash"})["ability_id"],"Darius:E")
        with self.assertRaises(ValueError):AbilityCatalog(self.root/"catalog.json",[{"text":"Pull","ability_id":"Darius:E"}])

    def test_pixel_cache_does_not_cache_wrong_champion_or_timestamp(self):
        image=Image.new("RGB",(60,30),"black");image.paste((0,200,200),(0,20,60,30))
        geometry={"text_box":[0,0,60,20],"progress_box":[0,20,60,30]}
        reader=CastBarReader(self.root,self.catalog,geometry)
        frame={"id":"f1","time":1,"segment_id":"s1","owner_epochs":{"selected":0}}
        with patch("annotation.league.readers.tesseract",return_value={"raw_text":"Apprehend","score":.99}) as ocr:
            first=reader.read(image,frame,{"owner_id":"selected","hud_champion":"Darius"})
            second=reader.read(image,{**frame,"id":"f2","time":2},{"owner_id":"selected","hud_champion":"Other"})
        self.assertEqual(ocr.call_count,1);self.assertEqual(first["ability_id"],"Darius:E")
        self.assertIsNone(second["ability_id"]);self.assertEqual(second["time"],2)
        self.assertTrue((self.root/first["text_path"]).exists())

    def test_absence_and_occlusion_do_not_ocr(self):
        image=Image.new("RGB",(60,30),"black")
        cfg={"text_box":[0,0,60,20],"progress_box":[0,20,60,30]}
        frame={"id":"f","time":0,"segment_id":"s","owner_epochs":{"selected":0}}
        with patch("annotation.league.readers.tesseract",side_effect=AssertionError("must not OCR")):
            self.assertEqual(CastBarReader(self.root,self.catalog,cfg).read(image,frame,{"owner_id":"selected"})["visibility"],"absent")
            self.assertEqual(CastBarReader(self.root,self.catalog,cfg,[[0,0,60,30]]).read(image,frame,{"owner_id":"selected"})["visibility"],"unknown")

    def test_channels_continue_without_claiming_completion(self):
        rows=[reading(0),reading(.0625),reading(.125,visibility="unreadable"),reading(.1875),reading(.25,visibility="absent")]
        events=channel_occurrences(rows)
        self.assertEqual(len(events),1);self.assertEqual(events[0]["end"],.1875)
        self.assertEqual(events[0]["terminal_status"],"ending_unknown");self.assertIsNone(events[0]["outcome"])

    def test_owner_playback_and_repeated_cast_boundaries(self):
        for update in ({"segment_id":"s2"},{"owner_epoch":1},{"ability_id":"Darius:R"},{"progress_fraction":.01}):
            with self.subTest(update=update):self.assertEqual(len(channel_occurrences([reading(0),reading(.0625,**update)])),2)
        self.assertEqual(len(channel_occurrences([reading(0),reading(.3)])),2)

    def test_review_rebuilds_timeline_and_rejects_wrong_owner_ability(self):
        self.catalog.publish(self.root)
        row=reading(0,catalog_id=self.catalog.signature,champion="Darius",review_status="unreviewed")
        write_rows(self.root/"cast_bar_readings.jsonl",[row])
        correct_reading(self.root,row["id"],{"ability_id":"Darius:R","raw_text":"Noxian Guillotine"},"reviewer")
        self.assertEqual(rebuild(self.root)[1][0]["ability_id"],"Darius:R")
        with self.assertRaises(ValueError):correct_reading(self.root,row["id"],{"ability_id":"Other:Q"},"reviewer")
        correct_reading(self.root,row["id"],{"review_status":"rejected"},"reviewer")
        self.assertEqual(rebuild(self.root)[1],[])

    def test_duplicate_emissions_reduce_precision(self):
        label={"id":"label","clip_id":"clip","owner_id":"selected","start":1,"end":2,
               "ability_id":"Darius:E","visibility":"readable","channel_kind":"ability"}
        predictions=[{**label,"id":"p1"},{**label,"id":"p2"}]
        score=score_occurrences([label],predictions)
        self.assertEqual(score["exact_id_precision"],.5);self.assertEqual(score["duplicate_candidates"],1)
        self.assertFalse(score["thresholds_passed"])
        self.assertFalse(score_occurrences([],[])["thresholds_passed"])

    def test_layout_specific_contracts_and_clip_boundaries(self):
        with self.assertRaises(ValueError):reader_contract({"reader_signature":"single"},{"1080p","720p"})
        readers={"1080p":"one","720p":"two"}
        self.assertTrue(reader_contract({"reader_signatures":readers},set(readers)))
        with self.assertRaises(ValueError):reader_contract({"reader_signatures":readers,"reader_signature":"wrong"},set(readers))
        with self.assertRaises(ValueError):validate_predictions([{"id":"p","clip_id":"clip","start":0,"end":20,"owner_id":"selected","channel_kind":"ability"}],[{"id":"clip","start":1,"end":2}])

    @unittest.skipUnless(shutil.which("tesseract"),"Native bar regression requires Tesseract")
    def test_real_apprehend_and_recall_use_different_vertical_positions(self):
        fixture=Path(__file__).parent/"fixtures/league_scene_understanding"
        data=read_json(fixture/"apprehend-hud.json")
        reader=CastBarReader(self.root,self.catalog,data["recall_profile"]["recall"])
        for name,kind,identifier in (("apprehend-cast-bar-03.png","ability","Darius:E"),("recall-positive-00.png","recall",None)):
            with self.subTest(name=name),Image.open(fixture/name) as image:
                row=reader.read(image,{"id":name,"time":0,"segment_id":"s","owner_epochs":{"selected":0}},
                                {"owner_id":"selected","hud_champion":"Darius"})
                self.assertEqual(row["visibility"],"readable");self.assertEqual(row["channel_kind"],kind)
                self.assertEqual(row["ability_id"],identifier)


class Dataset(unittest.TestCase):
    def test_empty_campaign_and_split_leakage_cannot_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            initialize(directory);self.assertFalse(coverage(directory)["release_ready"])
        source={"id":"source","sha256":"a"*64,"match_group":"match","split":"development"}
        data={"sources":[source,{**source,"id":"second","split":"held_out"}],"clips":[]}
        with self.assertRaisesRegex(ValueError,"crosses splits"):validate(data)
        data={"sources":[{**source,"previously_exposed":True,"split":"held_out"}],"clips":[]}
        with self.assertRaises(ValueError):validate(data)

    def test_model_cannot_mark_its_labels_human_verified(self):
        label={"reviewer":"model","reviewer_kind":"model","source_only":True,"verified":True,
               "human_attestation":True,"exhaustive":True,"events":[]}
        with self.assertRaises(ValueError):validate_label(label,{"start":0,"end":1})

    def test_verified_case_tags_require_observed_support(self):
        label={"reviewer":"human reviewer","reviewer_kind":"human","source_only":True,"verified":True,
               "human_attestation":True,"exhaustive":True,"events":[]}
        with self.assertRaises(ValueError):validate_label(label,{"start":0,"end":1,"case_tags":["recall_complete"]})
        clip={"start":0,"end":1,"case_tags":["resource_bar"]}
        with self.assertRaises(ValueError):validate_label(label,clip)
        label["case_evidence"]=[{"case_tag":"resource_bar","start":.1,"end":.5,"box":[0,0,20,5],"source_note":"Resource bar visible; no channel label."}]
        validate_label(label,clip)
