"""Physical occurrence boundaries and independently supported component claims."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from annotation.league.caption_occurrences import (OccurrenceSceneReader, occurrence_schema, resolve_occurrences,
    source_box, component_events)
from annotation.league.caption_pilot import occurrence_windows, occurrence_native_brackets
from annotation.league.caption_segments import POLICY
from annotation.league.captions import construct_captions, event_grounding
from annotation.league.core import Store


def footage():
    fs = [{"id": f"f{i}", "time": i/16, "segment_id": "seg", "owner_epochs": {"selected": 0}, "layout_valid": True,
           "source_fps": 60, "playback": {"playback_state": "playing"}} for i in range(96)]
    body = [{"id": "b"+f["id"], "frame_id": f["id"], "owner_id": "selected", "owner_epoch": 0, "segment_id": "seg",
             "actor_track": "player", "binding_visible": True, "actor_region": [200, 200, 300, 400], "champion": "Darius"} for f in fs]
    return fs, body


def proposal(pid="p1", start=16, end=24, kind="attack", relation="new_onset", prior=None):
    ids = [f"f{i}" for i in range(start, end+1)]
    criterion = {"attack": "attack_visible", "cast": "cast_visible", "recall": "recall_channel_visible", "movement": "movement_visible"}[kind]
    return {"id": pid, "schema_version": 2, "scene_schema_version": 2, "kind": kind, "actor": "Darius", "actor_label": "Darius",
            "start": start/16, "end": end/16, "relation": relation, "prior_proposal_id": prior,
            "phase_frames": {"onset": ids[:2] if relation == "new_onset" else [], "active": ids[1:], "recovery": [], "end": []},
            "components": {"action": "supported", "slot": "unresolved", "targeting": "unresolved", "impact": "unresolved", "last_hit": "unresolved"},
            "ability": None, "target": None, "target_binding": {}, "target_label": "the observed minion",
            "actor_binding": {"entity": "Darius", "binding": "player", "frame_ids": ids},
            "owner_id": "selected", "owner_epoch": 0, "playback_segment_id": "seg", "sampling_fps": 16,
            "inspection_stage": "16fps", "evidence_ids": ids,
            "visual_evidence": [{"criterion": criterion, "frame_ids": ids[1:]},
                {"criterion": "new_onset_visible" if relation == "new_onset" else "continuation_visible", "frame_ids": ids[:2]}],
            "request_ids": [pid], "acceptance": "grounded", "caption_pilot": True, "experimental": True, "review_status": "unreviewed"}


class OccurrenceTests(unittest.TestCase):
    def setUp(self):
        self.fs, self.body = footage()

    def resolve(self, rows):
        return resolve_occurrences(rows, self.fs, self.body, "source")

    def test_windup_release_recovery_and_recall_continuation_are_one_occurrence(self):
        for kind in ("cast", "recall"):
            a = proposal(kind=kind)
            b = proposal("p2", 23, 30, kind, "continuation", "p1")
            occurrences, events, omissions = self.resolve([a, b])
            self.assertEqual(len(occurrences), 1)
            self.assertEqual([e["kind"] for e in events], [kind])
            self.assertEqual(omissions, [])
            self.assertTrue(occurrences[0]["right_censored"])

    def test_new_onsets_preserve_rapid_actions_and_ids_ignore_type_slot_target(self):
        a, b = proposal(), proposal("p2", 25, 30)
        occurrences, _, _ = self.resolve([a, b])
        self.assertEqual(len(occurrences), 2)
        other = proposal(kind="cast")
        other["ability"] = "Q"
        self.assertEqual(self.resolve([a])[0][0]["id"], self.resolve([other])[0][0]["id"])
        adjacent=proposal("p3",17,24)
        self.assertEqual(len(self.resolve([a,adjacent])[0]),2)
        from annotation.league.caption_pilot import regenerate
        from annotation.league.core import atomic_json,write_rows,read_rows
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/('d'*20);root.mkdir()
            atomic_json(root/'input-audit.json',{'arm':'event-v2','frames':self.fs,'policy':POLICY,'interval':[0,6],
                'source_sha256':'source','profile':{},'masks':[]})
            write_rows(root/'body_readings.jsonl',self.body)
            write_rows(root/'scene_proposals.jsonl',[a,adjacent])
            regenerate(root)
            self.assertEqual(len(read_rows(root/'captions.jsonl')),2)

    def test_recovery_without_new_onset_cannot_create_an_extra_event(self):
        a, b = proposal(), proposal("p2", 25, 30, relation="unresolved")
        b["phase_frames"]["onset"] = []
        occurrences, events, omissions = self.resolve([a, b])
        self.assertEqual((sum(o["record_type"]=="physical_occurrence" for o in occurrences), len(events)), (1, 1))
        self.assertEqual(sum(o["record_type"]=="unresolved_candidate" for o in occurrences),1)
        self.assertEqual(omissions[0]["reason"], "unresolved_occurrence_boundary")

    def test_conflicting_end_and_continuation_remain_censored(self):
        a=proposal(kind="recall")
        a["phase_frames"]["end"]=["f24"]
        a["visual_evidence"].append({"criterion":"action_end_visible","frame_ids":["f24"]})
        b=proposal("p2",23,30,"recall","continuation","p1")
        occurrence=self.resolve([a,b])[0][0]
        self.assertEqual(occurrence["end_status"],"unresolved")
        self.assertTrue(occurrence["right_censored"])
        self.assertIn("competing_end_boundaries",occurrence["unresolved"])

    def test_left_censor_uses_source_context_boundary_not_packet_edge(self):
        for f, b in zip(self.fs[16:], self.body[16:]):
            f["owner_epochs"]["selected"] = b["owner_epoch"] = 1
        ongoing = proposal(kind="recall", relation="continuation")
        ongoing["owner_epoch"] = 1
        later = proposal("p2", 23, 30, "recall", "continuation")
        later["owner_epoch"] = 1
        occurrences, events, _ = self.resolve([ongoing, later])
        physical = [o for o in occurrences if o["record_type"] == "physical_occurrence"]
        self.assertEqual(len(physical), 1)
        self.assertTrue(physical[0]["left_censored"])
        self.assertIsNone(physical[0]["onset"])
        self.assertEqual(len(events), 1)

    def test_competing_action_types_targets_and_slots_stay_unresolved(self):
        a, b = proposal(), proposal("p2", kind="cast")
        occurrences, events, _ = self.resolve([a, b])
        self.assertEqual(len(occurrences), 1)
        self.assertEqual(events, [])
        self.assertIn("competing_action_types", occurrences[0]["unresolved"])

    def test_native_onset_refinement_with_shared_evidence_is_not_a_new_action(self):
        a=proposal()
        native=[]
        for i in range(5):
            f={**copy.deepcopy(self.fs[16]),'id':f'n{i}','time':1+(i+1)/60}
            self.fs.append(f);self.body.append({**copy.deepcopy(self.body[16]),'id':f'bn{i}','frame_id':f'n{i}'})
            native.append(f['id'])
        b=proposal('p2',16,24,relation='unresolved');b.update(relation='refinement',prior_proposal_id='p1',sampling_fps=60,inspection_stage='native',end=1+5/60)
        b['evidence_ids']=['f16']+native;b['actor_binding']['frame_ids']=b['evidence_ids']
        b['phase_frames']={'onset':native[:2],'active':native[1:],'recovery':[],'end':[]}
        b['visual_evidence']=[{'criterion':'new_onset_visible','frame_ids':native[:2]},
                             {'criterion':'attack_visible','frame_ids':native[1:]},
                             {'criterion':'continuation_visible','frame_ids':['f16',native[0]]}]
        occurrences,events,_=self.resolve([a,b])
        self.assertEqual(sum(o['record_type']=='physical_occurrence' for o in occurrences),1)
        self.assertEqual(len(events),1)
        self.assertAlmostEqual(occurrences[0]['onset'],1+2/60)

    def test_lost_icon_identity_keeps_body_bound_action_anonymous(self):
        from annotation.league.captions import body_signature
        row=proposal();row['actor_binding']['body_signatures']={f:body_signature(self.body[int(f[1:])]) for f in row['evidence_ids']}
        for b in self.body:b['champion']=None
        facts=self.resolve([row])[1]
        self.assertEqual(len(facts),1)
        self.assertEqual(facts[0]['actor_label'],'The selected champion')

    def test_context_only_and_invalid_owner_onset_never_emit(self):
        a = proposal()
        a["inspection_stage"] = "context"
        self.assertEqual(self.resolve([a])[1], [])
        a["inspection_stage"] = "16fps"
        self.body[16]["binding_visible"] = False
        self.assertEqual(self.resolve([a])[1], [])

    def test_generic_hit_unknown_slot_and_missing_target_retains_cast(self):
        a = proposal(kind="cast")
        ids = a["evidence_ids"]
        a["components"]["impact"] = "supported"
        a["target"] = "minion:one"
        a["target_binding"] = {"entity": a["target"], "binding": a["target"], "frame_ids": ids,
                               "regions": [{"frame_id": f, "box": [400, 200, 440, 250]} for f in ids]}
        a["visual_evidence"] += [{"criterion": "ability_cast_visible", "frame_ids": ids[:2]},
             {"criterion": "target_impact_visible", "frame_ids": ids[-2:]}, {"criterion": "cast_linked_to_impact", "frame_ids": ids}]
        _, events, _ = self.resolve([a])
        self.assertEqual({e["kind"] for e in events}, {"cast", "ability_hit"})
        captions, _ = construct_captions([], events, [], interval=[0, 6], source_sha256="source", pilot=True,
            frame_times={f["id"]: f["time"] for f in self.fs}, frame_context={f["id"]: f for f in self.fs}, body={r["frame_id"]: r for r in self.body})
        hit = next(c for c in captions if c["claims"][0]["kind"] == "ability_hit")
        self.assertIn("An ability", hit["text"])
        self.assertNotIn("None", hit["text"])
        self.assertEqual(len({c["event_id"] for c in captions}), 1)
        legacy = copy.deepcopy(next(e for e in events if e["kind"] == "ability_hit"))
        legacy.pop("scene_schema_version")
        with self.assertRaisesRegex(ValueError, "unresolved_ability"):
            event_grounding(legacy, {f["id"]: f["time"] for f in self.fs})
        a["target_binding"]["regions"] = []
        self.assertEqual([e["kind"] for e in self.resolve([a])[1]], ["cast"])

    def test_slot_claim_is_separate_and_cannot_be_proved_by_cooldown_only(self):
        a = proposal(kind="cast")
        a["ability"] = "Q"
        a["components"]["slot"] = "supported"
        self.assertEqual([e["kind"] for e in self.resolve([a])[1]], ["cast"])
        for c in ("slot_activation_visible", "slot_linked_to_action"):
            a["visual_evidence"].append({"criterion": c, "frame_ids": a["evidence_ids"]})
        facts = self.resolve([a])[1]
        self.assertEqual({e["kind"] for e in facts}, {"cast", "ability_slot"})
        self.assertIsNone(next(e for e in facts if e["kind"] == "cast")["ability"])

    def test_last_hit_needs_individual_lane_minion_death_and_action_link(self):
        row=proposal();ids=row["evidence_ids"]
        row["components"]["last_hit"]="supported";row["target"]="minion:one"
        row["target_binding"]={"entity":"minion:one","binding":"minion:one","frame_ids":ids,
            "regions":[{"frame_id":f,"box":[400,200,440,250]} for f in ids]}
        row["visual_evidence"] += [{"criterion":"minion_death_visible","frame_ids":ids[-2:]},
                                    {"criterion":"actor_action_linked_to_death","frame_ids":ids}]
        self.assertNotIn("minion_last_hit",{e["kind"] for e in self.resolve([row])[1]})
        row["visual_evidence"].append({"criterion":"lane_minion_visible","frame_ids":ids[:2]})
        self.assertIn("minion_last_hit",{e["kind"] for e in self.resolve([row])[1]})
        row["visual_evidence"]=[c for c in row["visual_evidence"] if c["criterion"]!="minion_death_visible"]
        self.assertEqual([e["kind"] for e in self.resolve([row])[1]],["attack"])

    def test_overlapping_effect_chain_fragments_complete_but_occlusion_cannot(self):
        a=proposal(kind="cast",end=23)
        b=proposal("p2",23,30,"cast","continuation","p1")
        for row in (a,b):
            row["target"]="minion:one"
            row["target_binding"]={"entity":"minion:one","binding":"minion:one","frame_ids":row["evidence_ids"],
                "regions":[{"frame_id":f,"box":[400,200,440,250]} for f in row["evidence_ids"]]}
            row["components"]["impact"]="partial"
            row["visual_evidence"].append({"criterion":"cast_linked_to_impact","frame_ids":row["evidence_ids"]})
        a["visual_evidence"].append({"criterion":"ability_cast_visible","frame_ids":a["evidence_ids"][:2]})
        b["visual_evidence"].append({"criterion":"target_impact_visible","frame_ids":b["evidence_ids"][-2:]})
        self.assertNotIn("ability_hit",{e["kind"] for e in self.resolve([a])[1]})
        occurrences,facts,_=self.resolve([a,b])
        self.assertIn("ability_hit",{e["kind"] for e in facts})
        self.assertFalse(any(r.startswith("ability_hit:") for r in occurrences[0]["unresolved"]))
        b["target_binding"]["regions"]=[r for r in b["target_binding"]["regions"] if r["frame_id"]!="f25"]
        self.assertNotIn("ability_hit",{e["kind"] for e in self.resolve([a,b])[1]})

    def test_target_geometry_maps_view_to_source_and_rejects_hud_masks(self):
        profile = {"scene_box": [0, 40, 1920, 980]}
        self.assertEqual(source_box([200, 100, 250, 150], profile), [400, 200, 500, 300])
        for box in ([200, 0, 250, 10], [200, 500, 250, 550]):
            with self.assertRaises(ValueError):
                source_box(box, profile)
        with self.assertRaisesRegex(ValueError, "masked"):
            source_box([200, 100, 250, 150], profile, [[420, 210, 440, 250]])

    def test_context_windows_stop_at_cuts_and_native_brackets_are_bounded(self):
        for f in self.fs[48:]:
            f["segment_id"] = "cut"
        windows = occurrence_windows([proposal()], self.fs, [0, 6])
        self.assertEqual((windows[0]["start"], windows[0]["end"]), (0, 3))
        native = occurrence_native_brackets([proposal(), proposal("p2", 23, 30)], windows[0])
        self.assertTrue(all(w["native_end"]-w["native_start"] <= 1 for w in native))


if __name__ == "__main__":
    unittest.main()


class OccurrencePipelineTests(unittest.TestCase):
    def test_real_orchestration_context_ledger_cache_resume_and_production_isolation(self):
        import test_league_caption_pilot as fixtures
        from unittest.mock import Mock, patch
        from annotation.league.caption_pilot import caption_pilot
        from annotation.league.core import read_rows
        helper=fixtures.PilotIntegrationTests()
        helper.setUp();self.addCleanup(helper.doCleanups)
        runner=Mock(identity={"model":"pinned-fixture"},peak_memory_bytes=123)
        def complete(messages,**kwargs):
            context=json.loads('{"frames":'+messages[0]["content"][0]["text"].split('{"frames":',1)[1])
            fs=context['frames'];near=[f for f in fs if 2<=f['time']<=2.25]
            actions=[]
            if len(near)>=2:
                first=[f for f in near if f['time']<=2.0625]
                prior=next((e['prior_proposal_id'] for e in context['ledger'] if e['shared_frames']),None)
                relation='new_onset' if len(first)==2 else 'continuation' if prior else 'unresolved'
                ids=[f['frame_id'] for f in near]
                action={"kind":"cast","actor_track":near[0]['player_track'],"target_id":None,"slot_id":None,
                    "start_frame":ids[0],"end_frame":ids[-1],"relation":relation,"prior_proposal_id":prior if relation=='continuation' else None,
                    "onset_frames":[f['frame_id'] for f in first] if relation=='new_onset' else [],"active_frames":ids[1:] if relation=='new_onset' else ids,"recovery_frames":[],"end_frames":[],
                    "action_status":"supported","slot_status":"unresolved","targeting_status":"unresolved","impact_status":"unresolved","last_hit_status":"unresolved",
                    "evidence":[{"criterion":"cast_visible","frame_ids":ids},{"criterion":"new_onset_visible" if relation=='new_onset' else 'continuation_visible',"frame_ids":ids[:2]}]}
                actions=[action]
            return {"finish_reason":"stop","text":json.dumps({"targets":[],"actions":actions,"uncertainty":[]}),"usage":{"completion_tokens":10}}
        runner.complete.side_effect=complete
        def run():
            with patch('annotation.league.caption_pilot.prepare_inputs',return_value=helper.audit),patch('annotation.league.caption_pilot._cpu_detection',side_effect=lambda *args:helper.detection()):
                return caption_pilot(helper.store,helper.config,0,6,infer=True,runner=runner)
        first=run();calls=runner.complete.call_count
        caps=read_rows(helper.root/'captions.jsonl')
        self.assertEqual(len(caps),1)
        self.assertIn('event_id',caps[0])
        self.assertEqual(len(read_rows(helper.root/'scene_occurrences.jsonl')),1)
        self.assertEqual(first['arm'],'event-v2')
        self.assertTrue(any(r['stage']=='context' for r in first['requests']))
        run();self.assertEqual(runner.complete.call_count,calls)
        self.assertEqual(helper.store.path('events.jsonl').read_bytes(),helper.production_events)
