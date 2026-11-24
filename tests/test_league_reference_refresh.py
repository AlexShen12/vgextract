import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

from annotation.league.core import Store, atomic_json, digest, read_json
from annotation.league.inventory_reader import ChampionPortraitMatcher, RespawnPortraitMatcher
from annotation.league.reference_bundle import cache_skins, load_skins, versions, cd_url
from annotation.league.reference_rollout import promote
from annotation.league.recognition_budget import reserve, finish
from annotation.league.review import create_app


class ReferenceRefresh(unittest.TestCase):
    def test_skin_cache_offline_integrity_and_circular_file_names(self):
        buf=io.BytesIO();Image.new("RGB",(32,32),"green").save(buf,format="PNG");png=buf.getvalue()
        def get(url):
            if url.endswith("champion.json"):
                value={"version":"16.22.1","data":{"Example":{"key":"1","name":"Example"}}}
            elif "/json/" in url:value=[{"name":"example_circle_0.png","type":"file"}]
            else:value={"id":1,"skins":[{"id":1000,"name":"Base","tilePath":"/lol-game-data/assets/ASSETS/Characters/Example/tile.png"}],
                        "spells":[{"spellKey":"q","name":"Example Q","description":"<b>Supplement</b>"}]}
            return value,{"url":url,"sha256":"a"*64,"retrieved_at":"2026-11-10"}
        with tempfile.TemporaryDirectory() as tmp,patch("annotation.league.reference_bundle.JsonCache.get",side_effect=get),patch("annotation.league.reference_bundle.download",return_value=png):
            result=cache_skins(tmp)
            self.assertEqual(result["status"],"complete")
            self.assertEqual({a["shape"] for a in result["assets"]},{"square","round"})
            self.assertEqual(result["ability_supplements"]["Example"]["Q"]["description"],"Supplement")
            cached=cache_skins(tmp,offline=True)
            self.assertEqual(cached["assets"],result["assets"])
            root=Path(result["root"]);(root/result["assets"][0]["path"]).write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError,"hash mismatch"):load_skins(root)

    def test_version_family_and_host_path_validation(self):
        with self.assertRaises(ValueError):versions("latest","16.22")
        with self.assertRaises(ValueError):versions("16.22.1","16.13")
        with self.assertRaises(ValueError):cd_url("16.22","/lol-game-data/assets/../../secret")

    def test_champion_margin_ignores_duplicate_skin_references(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);assets=[];images=[]
            for i,color in enumerate(("red","red","blue")):
                image=Image.new("RGB",(48,48),"black");draw=ImageDraw.Draw(image)
                draw.rectangle((4,5,18,40) if i<2 else (23,7,43,28),fill=color)
                image.save(root/f"{i}.png");images.append(image)
                assets.append({"kind":"champion","id":"A" if i<2 else "B","skin_id":str(i),"path":f"{i}.png"})
            matcher=ChampionPortraitMatcher(root,{"assets":assets},{"icon":0,"icon_margin":0})
            reading=matcher.read(images[0],{"kind":"champion"})
            self.assertEqual(reading["value"],"A");self.assertGreater(reading["reader"]["margin"],.1)
            self.assertIsNone(reading["skin_id"])

    def test_draft_comparison_cannot_promote_fresh_run_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);atomic_json(root/"config.json",{"layout_profile":"old"})
            for name in ("comparison","intake","pilot"):atomic_json(root/(name+".json"),{"passed":True})
            with self.assertRaises(ValueError):promote(root/"config.json",root/"comparison.json",root/"intake.json",root/"pilot.json")
            self.assertEqual(read_json(root/"config.json")["layout_profile"],"old")

    def test_failed_allocations_are_charged_and_recovery_is_reserved(self):
        with tempfile.TemporaryDirectory() as root:
            first=reserve(root,"preflight",1800);finish(root,first["attempt"],10,"failed")
            self.assertEqual(read_json(Path(root)/"gpu-budget.json")["attempts"][0]["reserved_seconds"],1800)
            reserve(root,"remaining",23400)
            with self.assertRaises(ValueError):reserve(root,"extra",1)
            reserve(root,"recovery",3600,recovery=True)
            with self.assertRaises(ValueError):reserve(root,"over",1,recovery=True)

    def test_failed_gpu_stage_preserves_receipt_and_reservation(self):
        from annotation.league.body_stage import execute
        with tempfile.TemporaryDirectory() as tmp,patch("annotation.league.body_campaign.track",side_effect=RuntimeError("source integrity failure")):
            with self.assertRaises(RuntimeError):execute("preflight",tmp,tmp)
            root=Path(tmp);receipt=read_json(root/"stages/00000001.json")
            self.assertEqual(receipt["status"],"failed");self.assertIn("integrity",receipt["error"])
            self.assertEqual(read_json(root/"gpu-budget.json")["attempts"][0]["reserved_seconds"],1800)

    def test_body_qualification_needs_independent_mask_and_resource_headroom(self):
        from annotation.league.recognition_budget import qualify
        from annotation.league.media import file_hash
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ref=root/"ref.png";pred=root/"pred.png"
            mask=Image.new("L",(20,20));mask.putpixel((10,10),255);mask.save(ref);mask.save(pred)
            data={"clips":[{"id":"clip","seed_reference":{"path":str(ref),"sha256":file_hash(ref),"frame_index":0,"source_only":True,"reviewer":"independent"}}],
                  "jobs":[{"id":"job","source_clip":"clip","frames":list(range(32))}]}
            atomic_json(root/"prepared.json",data)
            report={"status":"completed","encoder_smoke":{"passed":True},"peak_gpu_bytes":1024,
                    "peak_host_memory_bytes":1024,"prepared_hash":digest(data),"model_load_seconds":1,
                    "inference_wall_seconds":1,"sampled_frames":32}
            atomic_json(root/"preflight/report.json",report)
            atomic_json(root/"preflight/job-complete.json",{"frames":[{"frame_index":0,"masks":[{"mask_path":str(pred),"sha256":file_hash(pred)}]}]})
            self.assertTrue(qualify(root)["passed"])
            pred.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError,"mask changed"):qualify(root)
            self.assertFalse(read_json(root/"qualification.json")["passed"])

    def test_new_review_page_and_source_labels_are_separate(self):
        from fastapi.testclient import TestClient
        from annotation.league.recognition_dataset import initialize
        with tempfile.TemporaryDirectory() as tmp:
            store=Store(Path(tmp));initialize(store.path("benchmarks/cast-bars/v1"))
            client=TestClient(create_app(store))
            self.assertEqual(client.get("/cast-bars").status_code,200)
            self.assertEqual(client.get("/api/cast-bars").json(),[])
            labels=client.get("/api/cast-dataset").json()
            self.assertNotIn("predictions",labels);self.assertFalse(labels["coverage"]["release_ready"])
            self.assertEqual(client.get("/api/cast-bars/../../unknown").status_code,404)

    def test_review_api_serves_actual_crops_and_rebuilds_channels(self):
        from fastapi.testclient import TestClient
        from annotation.league.core import write_rows
        with tempfile.TemporaryDirectory() as tmp:
            store=Store(Path(tmp));experiment="a"*20;root=store.path("experiments/cast-bars/"+experiment)
            (root/"crops").mkdir(parents=True);Image.new("RGB",(30,20),"white").save(root/"crops/text.png")
            atomic_json(root/"cast-catalogs/catalog.json",{"reference_version":"16.22.1","abilities":{
                "Darius:E":{"champion":"Darius","name":"Apprehend","description":"E"},
                "Darius:R":{"champion":"Darius","name":"Noxian Guillotine","description":"R"}}})
            write_rows(root/"cast_bar_readings.jsonl",[{"id":"row","frame_id":"f","time":1,"owner_id":"selected","owner_epoch":0,
                "segment_id":"s","catalog_id":"catalog","visibility":"readable","raw_text":"Apprehend","channel_kind":"ability",
                "ability_id":"Darius:E","champion":"Darius","progress_fraction":.5,"review_status":"unreviewed",
                "reference_version":"16.22.1","text_path":"crops/text.png"}])
            client=TestClient(create_app(store));url="/api/cast-bars/cast-bars/"+experiment
            row=client.get(url).json()["readings"][0]
            self.assertEqual(client.get("/media/"+row["text_path"]).status_code,200)
            self.assertEqual(client.post(url+"/corrections/row",json={"reviewer":"source reviewer","changes":{"ability_id":"Darius:R","raw_text":"Noxian Guillotine"}}).status_code,200)
            self.assertEqual(client.get(url).json()["occurrences"][0]["ability_id"],"Darius:R")
            self.assertEqual(client.post(url+"/corrections/row",headers={"Origin":"https://example.com"},json={}).status_code,403)
