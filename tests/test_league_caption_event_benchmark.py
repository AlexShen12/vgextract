"""Occurrence metrics cannot be satisfied by counters, fragments or unreviewed output."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from annotation.league.caption_event_benchmark import choose_clips, match_events, score, aggregate, initialize, experiment, prediction_fingerprint, reference_adequacy
from annotation.league.caption_batch import bundle, import_bundle
from annotation.league.caption_occurrences import resolve_occurrences
from annotation.league.captions import construct_captions
from annotation.league.core import Store, atomic_json, read_json, write_rows
from annotation.league.media import file_hash
from annotation.league.action_v3 import load_profile
from test_league_caption_occurrences import footage, proposal


class EventBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.store=Store(Path(self.tmp.name)/'run');self.store.root.mkdir()
        self.video=Path(self.tmp.name)/'source.mp4';self.video.write_bytes(b'pinned source fixture')
        self.source={'path':str(self.video),'sha256':file_hash(self.video),'fps':60,'width':1920,'height':1080}
        atomic_json(self.store.path('manifest.json'),{'source':self.source,'interval':[0,2000]})
        self.root=self.store.path('experiments/captions/'+'b'*20);self.root.mkdir(parents=True)
        self.fs,self.body=footage()
        self.clip={'claims':[],'events':[],'bindings':[],'phases':[],'playback':[],'repeats':[],'case_tags':[]}

    def outputs(self,rows):
        occurrences,events,_=resolve_occurrences(rows,self.fs,self.body,self.source['sha256'])
        caps,_=construct_captions([],events,[],interval=[0,6],source_sha256=self.source['sha256'],pilot=True,
            frame_times={f['id']:f['time'] for f in self.fs},frame_context={f['id']:f for f in self.fs},body={r['frame_id']:r for r in self.body})
        atomic_json(self.root/'input-audit.json',{'frames':self.fs,'source_sha256':self.source['sha256']})
        atomic_json(self.root/'caption_coverage.json',{'key':'generation'})
        for name,rs in [('captions',caps),('events',events),('scene_occurrences',occurrences),('states',[]),('normalized_observations',[]),('caption_clips',[])]:write_rows(self.root/(name+'.jsonl'),rs)
        return caps,occurrences

    def test_one_to_one_matching_maximizes_cardinality(self):
        refs=[{'id':'r1','kind':'attack','start':0,'end':1,'onset':.2},{'id':'r2','kind':'attack','start':.2,'end':1.2,'onset':.4}]
        preds=[{'id':'p1','kind':'attack','start':0,'end':1.2,'onset':.3},{'id':'p2','kind':'attack','start':0,'end':.9,'onset':0}]
        matched,duplicates=match_events(refs,preds)
        self.assertEqual(len(matched),2);self.assertEqual(duplicates,0)

    def test_component_claims_count_once_and_unreviewed_occurrence_is_missed(self):
        row=proposal(kind='cast');row['ability']='Q';row['components']['slot']='supported'
        for criterion in ('slot_activation_visible','slot_linked_to_action'):row['visual_evidence'].append({'criterion':criterion,'frame_ids':row['evidence_ids']})
        caps,occ=self.outputs([row]);self.assertEqual(len(caps),2)
        self.clip['events']=[{'id':'r1','kind':'cast','start':occ[0]['start'],'end':occ[0]['end'],'onset':occ[0]['onset'],
                             'boundary_resolvable':True,'left_censored':False,'right_censored':False}]
        metric=score(self.clip,self.root)['event_metrics']
        self.assertEqual((metric['emitted_count'],metric['duplicates'],metric['supported_matched'],metric['missed']),(1,0,0,1))
        for i,c in enumerate(caps):atomic_json(self.root/'assessments'/f'{i}.json',{'generation_key':'generation','caption_id':c['id'],'supported':True,'actor_correct':True,'target_correct':True,'evidence_valid':True})
        self.assertEqual(score(self.clip,self.root)['event_metrics']['supported_matched'],1)

    def test_short_recall_fragments_are_duplicates_even_with_censored_reference(self):
        caps,occ=self.outputs([proposal(kind='recall'),proposal('p2',28,32,kind='recall')])
        self.clip['events']=[{'id':'r1','kind':'recall','start':.5,'end':4,'onset':None,'boundary_resolvable':False,'left_censored':True,'right_censored':True}]
        metrics=score(self.clip,self.root)['event_metrics']
        self.assertEqual(metrics['duplicates'],1)
        self.assertEqual(metrics['censored_reference_count'],1)
        self.assertEqual(metrics['reference_count'],0)

    def test_empty_and_counter_only_outputs_never_pass(self):
        result=aggregate([],[])
        self.assertFalse(result['event_stage_passed'])
        self.assertIsNone(result['event_metrics']['recall'])
        self.assertIsNone(result['event_metrics']['target_binding_accuracy'])

    def test_reference_audit_separates_empowered_attack_type_and_slot(self):
        self.clip.update(id='reference', events=[{'id':'swing','kind':'attack'}], claims=[
            {'id':'attack','kind':'attack','event_id':'swing'},
            {'id':'slot','kind':'ability_slot','event_id':'swing'},
            {'id':'cast','kind':'cast','event_id':'swing'}])
        audit=reference_adequacy([self.clip])
        self.assertEqual([row['claim_id'] for row in audit['action_kind_conflicts']],['cast'])
        self.assertIn('targeted_attack',audit['insufficient_claim_kinds'])

    def test_fresh_intervals_respect_history_and_context_length(self):
        excluded=[[50,100],[500,520]]
        rows=choose_clips([0,2000],excluded)
        self.assertEqual(len(rows),20)
        for c in rows:
            self.assertTrue(12<=c['end']-c['start']<=24)
            if c['split']=='held_out':self.assertFalse(any(c['start']<b+8 and a-8<c['end'] for a,b in excluded))
        with self.assertRaisesRegex(ValueError,'fresh'):
            choose_clips([0,500],[[0,500]])

    def test_freeze_requires_independent_development_and_measured_gate(self):
        from annotation.league.core import Config
        directory=self.store.path('labels');profile=load_profile()
        # No media/perception work is done by reference initialization.
        from unittest.mock import patch
        with patch('annotation.league.caption_event_benchmark.exclusions',return_value=[]):initialize(self.store,directory,profile)
        config=Config(routes=[],patch={},mechanics=False,evaluation_enabled=False,asset_cache=str(self.store.path('assets')))
        with self.assertRaisesRegex(ValueError,'Independent'):
            experiment(self.store,config,labels=directory,freeze=True)
        data=read_json(directory/'events.json')
        for c in data['clips']:
            c.update(verified=True,inspected=True,independent_of_predictions=True,coverage_exhaustive=True,labeler='independent-fixture')
            c['negative_annotations']={k:True for k in c['negative_annotations']}
        atomic_json(directory/'events.json',data)
        with self.assertRaisesRegex(ValueError,'development event gate'):
            experiment(self.store,config,labels=directory,freeze=True)

    def test_bundle_roundtrip_preserves_claims_and_refuses_overwrite(self):
        caps,_=self.outputs([proposal()])
        report=self.store.path('experiments/caption-events/'+'c'*20+'/report.json')
        atomic_json(report,{'draft':True,'clips':{'clip':{'root':str(self.root),'generation_key':'generation','resources':{}}}})
        atomic_json(report.parent/'reference-snapshot.json',{'source_sha256':self.source['sha256']})
        archive=Path(self.tmp.name)/'result.tar.gz';bundle(report,archive,include_images=False)
        target=Path(self.tmp.name)/'imported'
        import_bundle(archive,target,self.video)
        imported=read_json(target/'manifest.json');self.assertEqual(imported['source']['sha256'],self.source['sha256'])
        loaded=[json.loads(l) for l in (target/self.root.relative_to(self.store.root)/'captions.jsonl').read_text().splitlines()]
        self.assertEqual(loaded,caps)
        with self.assertRaisesRegex(ValueError,'overwrite'):import_bundle(archive,target,self.video)

    def test_artifact_fingerprint_detects_prose_edits_even_when_generation_key_is_unchanged(self):
        caps,_=self.outputs([proposal()]);before=prediction_fingerprint(self.root)
        caps[0]['text']='An unsupported replacement.'
        write_rows(self.root/'captions.jsonl',caps)
        self.assertNotEqual(before,prediction_fingerprint(self.root))


if __name__=='__main__':unittest.main()
