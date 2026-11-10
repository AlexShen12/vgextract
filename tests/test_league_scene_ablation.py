"""Information separation, honest denominators and locked evaluation gates."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image, ImageChops

from annotation.league import scene_ablation as ab
from annotation.league import scene_ablation_score as score
from annotation.league.core import atomic_json, digest, read_json, write_rows


def event(category="hud_activation",start=1,end=1,**extra):
    return {"id":category,"kind":category,"start":start,"end":end,"actor":"selected","slot":"Q",
            "evidence":[{"time":start,"cue":"visible transition"}],"left_censored":False,
            "right_censored":False,"boundary_resolvable":True,**extra}


def label(events=None):
    return {"clip_id":"negative-00","source_sha256":"a"*64,
            "review_status":"source_reviewed_model_draft",
            "review":{"reviewer":"blind source agent","prediction_outputs_seen":False,"human_reviewed":False,
                      "inspected_sheets":["source-sheet.png"]},
            "scope":{"hud_activations":"complete_observable","physical_actions":"complete_observable"},
            "events":[event()] if events is None else events,"unknown_intervals":[]}


class AblationTests(unittest.TestCase):
    def test_world_pixels_identical_and_scene_only_hud_is_black(self):
        image=Image.new("RGB",(1920,1080),(190,85,40));profile=ab.load_profile()
        scene=ab.render_view(image,profile,[],False)
        grounded=ab.render_view(image,profile,[],True)
        self.assertIsNone(ImageChops.difference(scene.crop((0,0,960,465)),grounded.crop((0,0,960,465))).getbbox())
        self.assertIsNone(scene.crop((0,485,960,635)).getbbox())
        self.assertIsNotNone(grounded.crop((0,485,960,635)).getbbox())

    def test_hud_activation_survives_scene_abstention(self):
        candidate={"id":"c","owner_id":"selected","owner_epoch":0,"segment_id":"s","slot_id":"Q",
                   "start":1,"end":1.0625,"onset_frame_ids":["f0","f1"],"confirmed_readings":[]}
        actions,links=ab.primary_actions([candidate])
        self.assertEqual(len(actions),1);self.assertFalse(actions[0]["physical_action_established"])
        self.assertEqual(actions[0]["kind"],"ability_activation");self.assertIsNone(actions[0]["outcome"])
        self.assertEqual(links,[])

    def test_all_categories_one_to_one_unknown_attribution_and_quiet(self):
        refs=[event(k,i,i+1 if k=="movement" else i,id=str(i),slot="Q" if "ability" in k else None)
              for i,k in enumerate(score.KINDS)]
        preds=[{**r,"slot_id":r["slot"]} for r in refs]
        preds.append(dict(preds[1]));preds[0].update(actor="unknown",slot_id=None)
        result=score.score_clip(label(refs),preds)
        for category in score.KINDS:self.assertEqual(result["categories"][category]["matched"],1)
        self.assertEqual(result["categories"]["movement"]["duplicates"],1)
        self.assertEqual(result["attribution"]["actor"]["unresolved"],1)
        self.assertEqual(result["attribution"]["slot"]["unresolved"],1)
        quiet=score.score_clip(label([]),[event("recall")])
        self.assertEqual(quiet["categories"]["recall"]["precision"],0)
        self.assertFalse(quiet["categories"]["recall"]["tested"])

    def test_temporal_thresholds_and_maximum_cardinality(self):
        self.assertIsNotNone(score.fit(event(),event(end=1.25)))
        self.assertIsNone(score.fit(event(),event(end=1.251)))
        self.assertIsNotNone(score.fit(event("movement",0,2),event("movement",1,2)))
        self.assertIsNone(score.fit(event("movement",0,2),event("movement",1.01,2)))
        refs=[event(start=1,end=1),event(start=1.3,end=1.3)]
        pairs,duplicates=score.matches(refs,[event(start=1.15,end=1.15),event(start=.8,end=.8)])
        self.assertEqual(len(pairs),2);self.assertFalse(duplicates)

    def test_partial_scope_unknown_intervals_and_hud_only_limits(self):
        source=label([event(),event("ability_cast",2,2)])
        source["scope"]["physical_actions"]="partial"
        result=score.score_clip(source,[event(),event("ability_cast",2,2)])
        self.assertIsNone(result["categories"]["ability_cast"]["precision"])
        self.assertEqual(result["categories"]["ability_cast"]["listed_reference_recovery"],1)
        source["unknown_intervals"]=[{"start":.9,"end":1.1,"scope":"hud_activations","reason":"unreadable"}]
        result=score.score_clip(source,[event()],False)
        self.assertEqual(result["categories"]["ability_activation"]["reference_events"],0)
        self.assertEqual(result["categories"]["ability_activation"]["predictions_in_unknown_scope"],1)
        self.assertIsNone(result["categories"]["ability_cast"]["recall"])

    def test_reference_gates_and_boundary_uncertainty(self):
        clip={"id":"negative-00","start":0,"end":4};source=label()
        self.assertTrue(score.validate_reference(source,clip,"a"*64,True))
        for mutate in (lambda x:x["review"].update(prediction_outputs_seen=True),
                       lambda x:x["scope"].update(hud_activations="partial"),
                       lambda x:x["events"][0].update(target="enemy"),
                       lambda x:x["events"][0].update(boundary_resolvable="unknown")):
            changed=copy.deepcopy(source);mutate(changed)
            with self.assertRaises(ValueError):score.validate_reference(changed,clip,"a"*64,True)
        source["events"][0]["right_censored"]=True
        self.assertEqual(score.score_clip(source,[event()])["categories"]["ability_activation"]["timing_matches"],0)

    def test_freeze_requires_receipt_complete_references_and_unchanged_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);clip={"id":"negative-00","start":0,"end":4,"split":"development"}
            study={"clips":[clip],"source":{"sha256":"a"*64}};pinned={"fixture":True}
            atomic_json(root/"study.json",study)
            with patch.object(ab,"recipe",return_value=pinned):
                with self.assertRaisesRegex(ValueError,"Freeze development"):ab.assert_frozen(root,study)
                with self.assertRaises(FileNotFoundError):ab.freeze(root,root/"refs")
                atomic_json(root/"development-review.json",{"reviewer":"r","recipe_sha256":digest(pinned),
                    "held_out_answers_used_for_tuning":False,"unresolved_infrastructure_failures":0,"reviewed_arms":list(ab.ARMS)})
                atomic_json(root/"refs/development/negative-00.json",label())
                for arm in ab.ARMS:
                    result={"arm":arm,"scope":"development","status":"completed","recipe_sha256":digest(pinned),
                            "accepted_requests":1,"clips":[{"clip_id":clip["id"],"status":"completed",
                            "split":"development","arm":arm,"recipe_sha256":digest(pinned),"runtime":{"fixture":"runtime"}}]}
                    atomic_json(root/"results/development"/arm/"report.json",result)
                paths=[]
                for name in ("body","bindings","hud_candidates"):
                    path=root/(name+".jsonl");write_rows(path,[]);paths.append(str(path))
                atomic_json(root/"prepared/negative-00/prepared.json",dict(zip(("body","bindings","hud_candidates"),paths),views={"scene-only":[],"hud-grounded":[]}))
                ab.freeze(root,root/"refs");ab.assert_frozen(root,study)
                Path(paths[0]).write_text("changed")
                with self.assertRaisesRegex(ValueError,"changed"):ab.assert_frozen(root,study)

    def test_empty_or_duplicate_complete_reports_fail(self):
        study={"clips":[{"id":"a","split":"development"}]}
        with patch.object(ab,"recipe",return_value={}):
            result={"arm":"scene-only","scope":"development","status":"completed","recipe_sha256":digest({}),"clips":[]}
            with self.assertRaisesRegex(ValueError,"every expected clip"):ab.validate_run(result,study,"scene-only","development")

    def test_hud_only_never_instantiates_world_model_or_reads_body(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);clip={"id":"negative-00","start":0,"end":4,"split":"development"}
            atomic_json(root/"study.json",{"clips":[clip]});path=root/"hud.jsonl";write_rows(path,[])
            atomic_json(root/"prepared/negative-00/prepared.json",{"recipe_sha256":digest({}),"hud_candidates":str(path),
                "hud_sha256":ab.file_hash(path),"body":"must-not-read","bindings":"must-not-read",
                "views":{"scene-only":[{"input_path":"must-not-read","input_sha256":"invalid"}]}})
            with patch.object(ab,"recipe",return_value={}),patch.object(ab,"LocalVLRunner") as runner:
                result=ab.run_arm(root,"hud-only")
                runner.assert_not_called();self.assertEqual(result["status"],"completed")

    def test_scene_reader_receives_grounding_only_in_grounded_arm(self):
        class Runner:
            identity={"fixture":"runtime"};peak_memory_bytes=0
            def __init__(self,*args,**kwargs):self.prompts=[]
            def complete(self,messages,**kwargs):
                self.prompts.append(messages[0]["content"][0]["text"])
                return {"finish_reason":"stop","text":json.dumps({"actions":[],"slots":[],"uncertainty":[]})}
            def close(self):pass
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);clip={"id":"negative-00","start":0,"end":4,"split":"development"}
            atomic_json(root/"study.json",{"clips":[clip],"profile":ab.load_profile()})
            path=root/"hud.jsonl";write_rows(path,[{"id":"candidate-secret","start":.5,"end":.5625,"slot_id":"Q",
                "owner_id":"selected","owner_epoch":0,"segment_id":"s","onset_frame_ids":["0","1"],"confirmed_readings":[]}])
            body=root/"body.jsonl";write_rows(body,[])
            image=root/"image.png";Image.new("RGB",(960,635)).save(image)
            frames=[{"id":str(i),"time":i*.25,"input_path":str(image),"input_sha256":ab.file_hash(image),
                     "segment_id":"s","owner_epochs":{"selected":0}} for i in range(4)]
            atomic_json(root/"prepared/negative-00/prepared.json",{"recipe_sha256":digest({}),"hud_candidates":str(path),
                "hud_sha256":ab.file_hash(path),"body":str(body),"bindings":str(body),
                "views":{"scene-only":frames,"hud-grounded":frames}})
            for arm,expected in (("scene-only",False),("hud-grounded",True)):
                runner=Runner()
                with patch.object(ab,"recipe",return_value={}),patch.object(ab,"LocalVLRunner",return_value=runner),\
                     patch.dict("os.environ",{"LEAGUE_VL_PYTHON":"unused","LEAGUE_ASSET_CACHE":"unused"}):
                    result=ab.run_arm(root,arm,preflight=True)
                self.assertTrue(result["qualified"])
                self.assertEqual("candidate-secret" in runner.prompts[0],expected)
                self.assertEqual("hud_grounding" in runner.prompts[0],expected)


if __name__=="__main__":unittest.main()
