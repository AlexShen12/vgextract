"""Player binding and event contracts independent of model predictions."""
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image, ImageDraw

from annotation.league.action_body import bind_frames, candidates
from annotation.league.action_events import counter_events, link_farming, recall_events
from annotation.league.action_v3 import SceneReader, action_window, coalesce_windows, load_profile, scene_instruction, scene_schema
from annotation.league.action_benchmark_v3 import experiment_actions, score_clip
from annotation.league.ability_catalog import cache_abilities
from annotation.league.core import Config, Store, atomic_json, read_json, read_rows
from annotation.league.review import create_app
from annotation.league.media import file_hash


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(Path(self.tmp.name))
        self.config = Config.load(Path(__file__).resolve().parents[1] / 'annotation/league/midbeast.json')
        self.config.asset_cache = str(self.store.path('assets'))
        self.config.masks = [[400, 200, 500, 300]]
        self.video = self.store.path('video.mp4')
        self.video.write_bytes(b'source fixture')
        atomic_json(self.store.path('manifest.json'), {
            'source': {'path': str(self.video), 'sha256': file_hash(self.video),
                       'width': 1920, 'height': 1080, 'fps': 60},
            'interval': [0, 1],
        })
        self.decoded = [(t, Image.new('RGB', (1920, 1080), 'white')) for t in (.01, .0725)]
        for name, options in (
            ('annotation.league.intake.sampled_frames', {'side_effect': lambda *a: iter(self.decoded)}),
            ('annotation.league.media.probe', {'return_value': {'fps': 60}}),
            ('annotation.league.action_pipeline.layout_matches', {'return_value': (True, [])}),
            ('annotation.league.readers.tesseract', {'return_value': {'raw_text': '1:00'}}),
            ('annotation.league.action_v3.LocalVLRunner', {'side_effect': AssertionError('Qwen loaded')}),
        ):
            patcher = patch(name, **options)
            mock = patcher.start()
            self.addCleanup(patcher.stop)
            if name.endswith('sampled_frames'):
                self.decode = mock

    def test_prepare_only_preserves_pts_and_native_images_and_reuses_inputs(self):
        report = action_window(self.store, self.config, 0, .125)
        audit = read_json(Path(report['input_audit']))
        self.assertEqual(report['status'], 'inputs_prepared')
        self.assertEqual(audit['schema_version'], 3)
        self.assertEqual([f['time'] for f in audit['frames']], [.01, .0725])
        with Image.open(audit['frames'][0]['native_path']) as native:
            self.assertEqual(native.getpixel((450, 250)), (255, 255, 255))
        with Image.open(audit['frames'][0]['input_path']) as model_input:
            self.assertEqual(model_input.getpixel((225, 125)), (0, 0, 0))
        self.assertEqual(action_window(self.store, self.config, 0, .125)['root'], report['root'])
        self.assertEqual(self.decode.call_count, 1)

    def test_incomplete_decode_and_changed_source_are_rejected(self):
        self.decoded.pop()
        with self.assertRaisesRegex(ValueError, 'decode incomplete'):
            action_window(self.store, self.config, 0, .125)
        self.video.write_bytes(b'changed source')
        with self.assertRaisesRegex(ValueError, 'hash/geometry mismatch'):
            action_window(self.store, self.config, 0, .125)


def drawing(profile, bars=((700, 223),), fill=.8, hud=.8, level=True):
    image = Image.new('RGB', (1920,1080), (30,30,30)); d=ImageDraw.Draw(image)
    for x,y in bars:
        d.rectangle((x,y-3,x+104,y+12), fill=(8,8,8))
        d.rectangle((x,y,x+int(105*fill)-1,y+9), fill=(70,180,70))
        if level:
            d.rectangle((x-25,y-7,x-3,y+18),fill=(10,20,30))
            d.rectangle((x-15,y-2,x-13,y+7),fill=(220,220,220))
    x,y,r,b=profile['body_binder']['hud_health_box']
    d.rectangle((x,y,x+int((r-x)*hud)-1,b-1), fill=(30,200,40))
    return image


def counter(t,value,field='cs',**extra):
    return {'id':str(t)+field,'frame_id':'f'+str(t),'time':t,'value':value,'field':field,'owner_id':'selected',
            'owner_epoch':0,'segment_id':'s','layout_valid':True,'champion':'Example',**extra}


def recall(t,active,progress=.7,**extra):
    return {'id':str(t),'frame_id':'f'+str(t),'time':t,'active':active,'progress_fraction':progress,
            'owner_id':'selected','owner_epoch':0,'segment_id':'s','layout_valid':True,'champion':'Example',**extra}


class BodyTests(unittest.TestCase):
    def setUp(self): self.profile=load_profile()

    def test_bar_level_frame_and_matching_health_are_required(self):
        rows,_=candidates(drawing(self.profile),self.profile)
        self.assertEqual(len(rows),1)
        self.assertAlmostEqual(rows[0]['health_fraction'],.8,places=2)
        self.assertEqual(candidates(drawing(self.profile,level=False),self.profile)[0],[])
        self.assertEqual(candidates(drawing(self.profile,fill=.1),self.profile)[0],[])

    def test_hud_portraits_minion_bars_and_natural_green_are_excluded(self):
        self.assertEqual(candidates(drawing(self.profile,bars=((1700,700),)),self.profile)[0],[])
        self.assertEqual(candidates(drawing(self.profile,bars=((900,400),),level=False),self.profile)[0],[])
        image=drawing(self.profile,bars=());ImageDraw.Draw(image).rectangle((700,200,805,240),fill=(40,180,40))
        self.assertEqual(candidates(image,self.profile)[0],[])
        profile=copy.deepcopy(self.profile);profile['body_binder']['palette']['color']='yellow'
        self.assertEqual(candidates(image,profile)[1],'unsupported_palette')

    def test_competing_candidates_do_not_choose_a_player(self):
        self.assertEqual(candidates(drawing(self.profile,bars=((700,223),(900,400))),self.profile)[1],'ambiguous_candidates')

    def test_unknown_kit_does_not_change_a_visible_player_track(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'frame.png';drawing(self.profile).save(path)
            names=['First',None,'First','First','Other','Other','Other']
            frames=[{'id':str(i),'time':i/16,'native_path':str(path),'layout_valid':True,'segment_id':'s','owner_epochs':{'selected':0}} for i in range(len(names))]
            identities=[{'frame_id':str(i),'owner_id':'selected','hud_champion':name} for i,name in enumerate(names)]
            rows,_=bind_frames(frames,identities,self.profile)
            self.assertEqual(rows[0]['actor_track'],rows[3]['actor_track'])
            self.assertIsNotNone(rows[1]['actor_track'])
            self.assertNotEqual(rows[3]['actor_track'],rows[6]['actor_track'])

    def test_camera_motion_disappearance_cuts_and_selection_reacquire(self):
        with tempfile.TemporaryDirectory() as tmp:
            frames=[]
            for i,(x,valid,segment,epoch) in enumerate([(700,True,'s',0),(720,True,'s',0),(740,True,'s',0),
                (740,False,'s',0),(750,True,'s',0),(760,True,'s',0),(770,True,'s',0),
                (770,True,'cut',0),(770,True,'cut',1),(770,True,'cut',1),(770,True,'cut',1)]):
                path=Path(tmp)/f'{i}.png'
                drawing(self.profile,bars=((x,223),) if valid else ()).save(path)
                frames.append({'id':str(i),'time':i/16,'native_path':str(path),'layout_valid':True,'segment_id':segment,'owner_epochs':{'selected':epoch}})
            rows,bindings=bind_frames(frames,[],self.profile)
            self.assertTrue(rows[2]['binding_visible'])
            self.assertFalse(rows[3]['binding_visible'])
            self.assertFalse(rows[7]['binding_visible'])
            self.assertNotEqual(rows[2]['actor_track'],rows[6]['actor_track'])
            self.assertNotEqual(rows[6]['actor_track'],rows[10]['actor_track'])
            self.assertTrue(all(b['champion'] is None for b in bindings))


class EventTests(unittest.TestCase):
    def test_overlapping_dense_requests_keep_separate_native_brackets(self):
        windows=[{'segment_id':'s','start':1,'end':4,'native_start':2,'native_end':2.5,'reasons':['slot']},
                 {'segment_id':'s','start':3,'end':6,'native_start':4,'native_end':4.5,'reasons':['burst']},
                 {'segment_id':'cut','start':2,'end':4,'reasons':['quiet_audit']}]
        result=coalesce_windows(windows)
        merged=next(r for r in result if r['segment_id']=='s')
        self.assertEqual((merged['start'],merged['end']),(1,6))
        self.assertEqual([(r['native_start'],r['native_end']) for r in merged['windows']],[(2,2.5),(4,4.5)])
        self.assertEqual(len(result),2)

    def test_persistent_counter_one_frame_ocr_errors_and_missing_intervals(self):
        rows=[counter(i/16,v) for i,v in enumerate([10,10,11,10,10,11,11,11,11])]
        events=counter_events(rows)
        self.assertEqual(len(events),1);self.assertEqual(events[0]['counter_delta'],1)
        self.assertEqual(events[0]['kind'],'creep_credit');self.assertIsNone(events[0]['reward_gold'])
        self.assertEqual(counter_events([counter(0,10),counter(.0625,10),counter(.125,None),counter(.1875,12),counter(.25,12)]),[])
        self.assertEqual(counter_events([counter(0,10),counter(.0625,10),counter(2,12),counter(2.0625,12)]),[])

    def test_passive_gold_purchases_and_simultaneous_credit_remain_separate(self):
        gold=[counter(i/16,v,'displayed_gold') for i,v in enumerate([100,100,101,101,50,50])]
        events=counter_events(gold)
        self.assertEqual([e['counter_delta'] for e in events],[1,-51])
        self.assertTrue(all(e['kind']=='gold_change' and e['reward_gold'] is None for e in events))
        cs=[counter(i/16,v) for i,v in enumerate([10,10,13,13])]
        self.assertEqual(counter_events(cs)[0]['count_delta'],3)

    def test_lane_kills_require_unique_scene_and_actor_evidence(self):
        event=counter_events([counter(i/16,v) for i,v in enumerate([10,10,11,11])])[0]
        binding={'owner_id':'selected','owner_epoch':0,'segment_id':'s','actor_track':'player','start':0,'end':.5}
        proposal={'id':'scene','kind':'minion_last_hit','target_kind':'lane_minion','actor_track':'player','mode':'core',
                  'owner_id':'selected','owner_epoch':0,'segment_id':'s','start':.1,'end':.2,'evidence_ids':['request'],
                  'frame_ids':['f'],'reward_gold':None}
        self.assertEqual(link_farming([event],[],[binding])[0]['kind'],'creep_credit')
        self.assertEqual(link_farming([event],[proposal],[binding])[0]['evidence_class'],'corroborated')
        results=link_farming([event],[{**proposal,'target_kind':'jungle'}],[binding])
        self.assertEqual(results[0]['kind'],'creep_credit');self.assertEqual(results[1]['kind'],'jungle_last_hit')
        self.assertEqual(link_farming([event],[proposal,{**proposal,'id':'other'}],[binding])[0]['evidence_class'],'hud-supported')

    def test_recall_completion_interruption_cuts_and_clipped_attempts(self):
        rows=[recall(0,False),recall(.0625,True),recall(.125,True),recall(.1875,True,.04),
              recall(.25,False,0,base_arrival_visible=True)]
        events=recall_events(rows);self.assertEqual(len(events),1)
        self.assertEqual(events[0]['terminal_status'],'completed');self.assertFalse(events[0]['left_censored'])
        rows[-2]=recall(.1875,True,.6);rows[-1]=recall(.25,False,0,post_activity_visible=True)
        self.assertEqual(recall_events(rows)[0]['terminal_status'],'interrupted')
        rows[-1]=recall(.25,False,0)
        self.assertEqual(recall_events(rows)[0]['terminal_status'],'ending_unknown')
        self.assertTrue(recall_events([recall(0,True),recall(.0625,True)])[0]['left_censored'])
        self.assertTrue(recall_events(rows[:-1])[0]['right_censored'])
        self.assertEqual(recall_events([recall(0,False),recall(.0625,False)]),[])
        rows[-1]=recall(.25,False,0,segment_id='cut',base_arrival_visible=True)
        self.assertEqual(recall_events(rows)[0]['terminal_status'],'ending_unknown')

    def test_brief_tooltip_occlusion_preserves_one_uncertain_recall(self):
        rows=[recall(0,False),recall(.0625,True),recall(.125,True),
              recall(.1875,False,0,region_occluded=True),recall(.25,True,.5),recall(.3125,False,0)]
        events=recall_events(rows)
        self.assertEqual(len(events),1)
        self.assertEqual(events[0]['uncertain_frame_ids'],['f0.1875'])
        self.assertEqual(events[0]['terminal_status'],'ending_unknown')


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.store=Store(Path(self.tmp.name));self.profile=load_profile()
        self.config=Config.load(Path(__file__).resolve().parents[1]/'annotation/league/midbeast.json')
        self.config.asset_cache=str(self.store.path('assets'))
        self.root=self.store.path('experiments/actions-v3/'+'a'*20);self.root.mkdir(parents=True)
        path = str(self.root / 'frame.png')
        self.frames = [{'id': str(i), 'time': i / 16, 'native_path': path, 'input_path': path,
                        'input_sha256': 'fixture', 'segment_id': 's', 'owner_epochs': {'selected': 0},
                        'layout_valid': True, 'scene_sample': i == 0} for i in range(4)]
        self.audit={'root':str(self.root),'source_sha256':'source','interval':[0,.25],'profile':self.profile,'source':{'fps':60},'frames':self.frames}
        atomic_json(self.root/'input-audit.json',self.audit)

    def prepare_images(self):
        """Pixel-reading tests share one image; frame IDs and timestamps stay distinct."""
        path = Path(self.frames[0]['native_path'])
        drawing(self.profile).save(path)
        signature = file_hash(path)
        for frame in self.frames:
            frame['input_sha256'] = signature
        atomic_json(self.root / 'input-audit.json', self.audit)

    def test_cpu_core_never_loads_qwen_and_preserves_production(self):
        self.prepare_images()
        self.store.path('events.jsonl').write_text('preserved\n')
        with patch('annotation.league.action_v3.prepare_inputs',return_value=self.audit),patch('annotation.league.action_v3.LocalVLRunner',side_effect=AssertionError('Qwen loaded')),patch('annotation.league.action_v3.scan_hud',return_value=([],{})),patch('annotation.league.action_v3.read_counters',return_value=[]),patch('annotation.league.action_v3.read_recall',side_effect=lambda im,f,*a:recall(f['time'],False)):
            report=action_window(self.store,self.config,0,.25,detect=True)
        self.assertEqual(report['qwen_identity_requests'],0)
        self.assertEqual(report['status'],'completed_experimental');self.assertEqual(report['resource_metrics']['requests'],0)
        self.assertEqual(self.store.path('events.jsonl').read_text(),'preserved\n')
        self.assertTrue(read_rows(self.root/'core/body_readings.jsonl')[-1]['binding_visible'])
        with self.assertRaisesRegex(ValueError,'removed'):
            action_window(self.store,self.config,0,.25,assistance='old.json')

    def test_scene_schema_has_no_identity_and_rejects_identity_override_and_invalid_tracks(self):
        schema=scene_schema()
        self.assertNotIn('identities',schema['properties'])
        self.assertNotIn('actor',schema['properties']['actions']['items']['properties'])
        runner=Mock(identity={'model':'fixture'})
        body=[{'frame_id':f['id'],'binding_visible':True,'actor_track':'player','owner_id':'selected'} for f in self.frames]
        reader=SceneReader(self.store,self.root,self.profile,runner,body,{'hud_fps':16})
        action={'kind':'minion_last_hit','actor_track':'invented','actor_description':'Player','slot_id':None,
                'target_kind':'lane_minion','reward_gold':None,'start_frame':'0','end_frame':'1','evidence':[
                {'frame_id':'0','cue':'player_action','description':'Attack'}, {'frame_id':'1','cue':'minion_death','description':'Minion dies'}]}
        response={'finish_reason':'stop','text':json.dumps({'actions':[action],'slots':[],'uncertainty':[]})}
        with self.assertRaisesRegex(ValueError,'deterministic'):
            reader.validate(response,self.frames,'scene','req')
        action['actor_track']='player';action['reward_gold']=20
        response['text']=json.dumps({'actions':[action],'slots':[],'uncertainty':[]})
        with self.assertRaisesRegex(ValueError,'popup'):reader.validate(response,self.frames,'scene','req')
        self.assertIn('Never identify or name',scene_instruction('scene'))

    def test_review_includes_v3_body_and_counters(self):
        from fastapi.testclient import TestClient
        atomic_json(self.root/'report.json',{'schema_version':3,'status':'completed_experimental','interval':[0,.25]})
        (self.root/'core').mkdir()
        client=TestClient(create_app(self.store));data=client.get('/api/actions/'+'a'*20).json()
        self.assertEqual(data['modes'],['core']);self.assertIn('body_readings',data);self.assertIn('counter_readings',data)
        self.assertEqual(client.get('/api/actions/'+'a'*20+'/abilities').status_code,200)

    def test_source_only_labeling_is_revisioned_and_separate(self):
        from fastapi.testclient import TestClient
        atomic_json(self.store.path('manifest.json'),{'source':{'sha256':'source'},'interval':[0,1732]})
        experiment_actions(self.store,self.config,initialize=True)
        client=TestClient(create_app(self.store));data=client.get('/api/action-labels').json();clip=data['clips'][0]
        body={'labeler':'independent-test-agent','independent_of_predictions':True,
              'annotations':{'verified':False,'bindings':[],'identities':[],'transitions':[],'actions':[],'case_tags':[]}}
        self.assertEqual(client.get('/action-labels').status_code,200)
        self.assertEqual(client.post('/api/action-labels/'+clip['id'],json=body).status_code,200)
        self.assertEqual(len(list(self.store.path('benchmarks/actions-v3/revisions').glob('*.json'))),1)
        body['independent_of_predictions']=False
        self.assertEqual(client.post('/api/action-labels/'+clip['id'],json=body).status_code,422)
        self.assertFalse(self.store.path('corrections').exists())

    def test_scene_cache_resume_never_requests_identity(self):
        self.prepare_images()
        runner=Mock(identity={'model':'fixture'})
        runner.complete.return_value={'finish_reason':'stop','text':json.dumps({'actions':[],'slots':[],'uncertainty':[]}),
                                      'usage':{},'wall_seconds':.1,'peak_memory_bytes':0}
        body,bindings=bind_frames(self.frames,[],self.profile)
        reader=SceneReader(self.store,self.root,self.profile,runner,body,{'hud_fps':16})
        reader.read(self.frames);reader.read(self.frames)
        self.assertEqual(runner.complete.call_count,1)
        request=read_json(next((self.root/'requests').glob('*/input.json')))
        self.assertNotIn('identities',request['schema']['properties'])
        self.assertEqual(request['task'],'scene')
        self.assertLessEqual(sum(c['type']=='image_url' for c in runner.complete.call_args.args[0][0]['content']),8)

    def test_scene_validation_rejects_truncation_bad_citations_and_discontinuities(self):
        reader = SceneReader(self.store, self.root, self.profile, Mock(), [], {})
        action = {'kind': 'ability_cast', 'actor_track': 'other-body', 'actor_description': 'Visible body',
                  'slot_id': None, 'target_kind': 'unknown', 'reward_gold': None,
                  'start_frame': '0', 'end_frame': '1',
                  'evidence': [{'frame_id': '1', 'cue': 'cast_animation', 'description': 'Visible cast'}]}
        response = {'finish_reason': 'stop', 'text': json.dumps({'actions': [action], 'slots': [], 'uncertainty': []})}
        self.assertEqual(len(reader.validate(response, self.frames, 'scene', 'req')['actions']), 1)
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            reader.validate({**response, 'finish_reason': 'length'}, self.frames, 'scene', 'req')
        for changes in ({'segment_id': 'cut'}, {'owner_epochs': {'selected': 1}}, {'time': -1}):
            frames = copy.deepcopy(self.frames)
            frames[1].update(changes)
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, 'discontinuity'):
                reader.validate(response, frames, 'scene', 'req')
        action['evidence'][0]['frame_id'] = 'invented'
        response['text'] = json.dumps({'actions': [action], 'slots': [], 'uncertainty': []})
        with self.assertRaises(KeyError):
            reader.validate(response, self.frames, 'scene', 'req')

    def test_hud_validation_enforces_requested_slots_and_rejects_duplicates(self):
        reader = SceneReader(self.store, self.root, self.profile, Mock(), [], {})
        slot = {'frame_id': '0', 'owner_id': 'selected', 'slot_id': 'Q', 'state': 'cooling_down',
                'seconds': 5, 'charges': None, 'recast_visible': False, 'charge_consumption_visible': False,
                'item_effect': 'unknown', 'name': None, 'evidence': 'Visible numeral'}
        response = {'finish_reason': 'stop', 'text': json.dumps({'actions': [], 'slots': [slot], 'uncertainty': []})}
        parsed = reader.validate(response, self.frames, 'hud', 'req', {'0': [('selected', 'Q')]})
        self.assertEqual(parsed['slots'][0]['frame_ids'], ['0'])
        with self.assertRaisesRegex(ValueError, 'Unrequested'):
            reader.validate(response, self.frames, 'hud', 'req', {'0': [('selected', 'W')]})
        response['text'] = json.dumps({'actions': [], 'slots': [slot, slot], 'uncertainty': []})
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            reader.validate(response, self.frames, 'hud', 'req')

    def test_equivalent_gpu_resume_keeps_actual_runtime_provenance(self):
        self.prepare_images()
        response={'finish_reason':'stop','text':json.dumps({'actions':[],'slots':[],'uncertainty':[]}),
                  'usage':{},'wall_seconds':.1,'peak_memory_bytes':0}
        first=Mock(identity={'model':'fixture','runtime':{'gpu':'L40S','gpu_uuid':'one','packages':{'vllm':'fixed'}}})
        first.complete.return_value=response
        SceneReader(self.store,self.root,self.profile,first,[],{}).read(self.frames)
        second=Mock(identity={'model':'fixture','runtime':{'gpu':'L40S','gpu_uuid':'two','packages':{'vllm':'fixed'}}})
        other=self.store.path('experiments/actions-v3/'+'b'*20);other.mkdir()
        reader=SceneReader(self.store,other,self.profile,second,[],{});reader.read(self.frames)
        second.complete.assert_not_called()
        request=read_json(next((other/'requests').glob('*/input.json')))
        self.assertEqual(request['actual_runtime']['runtime']['gpu_uuid'],'two')
        self.assertNotIn('gpu_uuid',request['runtime']['runtime'])

    def test_scene_failure_preserves_completed_cpu_results(self):
        self.prepare_images()
        from annotation.league.local_models import ModelUnavailable
        runner=Mock(identity={'model':'fixture'});runner.complete.side_effect=ModelUnavailable('GPU context failure')
        with patch('annotation.league.action_v3.prepare_inputs',return_value=self.audit),patch('annotation.league.action_v3.scan_hud',return_value=([],{})),patch('annotation.league.action_v3.read_counters',return_value=[]),patch('annotation.league.action_v3.read_recall',side_effect=lambda im,f,*a:recall(f['time'],False)):
            with self.assertRaises(ModelUnavailable):action_window(self.store,self.config,0,.25,infer=True,runner=runner)
        self.assertTrue((self.root/'core/core_actions.jsonl').exists())
        self.assertEqual(read_json(self.root/'report.json')['status'],'failed_experimental')

    def test_benchmark_forks_unverified_labels_and_requires_independence(self):
        atomic_json(self.store.path('manifest.json'),{'source':{'sha256':'source'},'interval':[0,1732]})
        result=experiment_actions(self.store,self.config,initialize=True)
        data=read_json(Path(result['labels']));self.assertEqual(data['schema_version'],3)
        self.assertEqual(len(data['clips']),20);self.assertFalse(data['independent_of_predictions'])
        self.assertFalse((Path(result['labels']).parent / 'assistance.json').exists())
        with self.assertRaisesRegex(ValueError,'Independent'):
            experiment_actions(self.store,self.config,freeze=True)

    def test_benchmark_forks_old_clip_locations_without_labels_or_verification(self):
        atomic_json(self.store.path('manifest.json'), {'source': {'sha256': 'source'}, 'interval': [0, 1732]})
        result = experiment_actions(self.store, self.config, initialize=True)
        path = Path(result['labels'])
        old = read_json(path)
        old['schema_version'] = 2
        old['clips'][0].update(verified=True, labeler='old-labeler', independent_of_predictions=True,
                               identities=[{'hud_champion': 'Old'}], actions=[{'kind': 'ability_cast'}])
        prior = self.store.path('benchmarks/actions-v2/actions.json')
        atomic_json(prior, old)
        path.unlink()
        experiment_actions(self.store, self.config, initialize=True)
        fork = read_json(path)
        self.assertEqual(fork['forked_from'], str(prior))
        self.assertEqual([(c['id'], c['start'], c['end'], c['split']) for c in fork['clips']],
                         [(c['id'], c['start'], c['end'], c['split']) for c in old['clips']])
        for clip in fork['clips']:
            self.assertFalse(clip['verified'])
            self.assertFalse(clip['independent_of_predictions'])
            self.assertIsNone(clip['labeler'])
            self.assertTrue(all(clip[k] == [] for k in ('identities', 'bindings', 'transitions', 'actions', 'case_tags')))
        self.assertEqual(read_json(prior), old)

    def test_independent_scoring_compares_cpu_credit_with_visible_minion_evidence(self):
        from annotation.league.core import write_rows
        directory=self.root/'core';directory.mkdir()
        cpu={'id':'cpu','kind':'creep_credit','owner_id':'selected','start':.1,'end':.15,'count_delta':2,'evidence_class':'hud-supported'}
        linked={**cpu,'id':'linked','kind':'minion_last_hit','evidence_class':'corroborated'}
        for name,rows in [('actions',[linked]),('core_actions',[cpu]),('body_readings',[]),('identities',[]),('transitions',[]),('resampling',[])]:write_rows(directory/(name+'.jsonl'),rows)
        clip={'bindings':[],'identities':[],'transitions':[],'actions':[linked]}
        report={'results':{'core':{'directory':str(directory)}}}
        metrics=score_clip(clip,report)
        self.assertEqual(metrics['core_only']['creep_credit']['recall'],1)
        self.assertEqual(metrics['core_only']['minion_last_hit']['recall'],0)
        self.assertEqual(metrics['with_scene']['minion_last_hit']['recall'],1)
        self.assertEqual(metrics['with_scene']['creep_credit']['aggregate_count_error'],0)

    def test_verified_independent_freeze_rejects_later_development_changes(self):
        atomic_json(self.store.path('manifest.json'),{'source':{'sha256':'source'},'interval':[0,1732]})
        result=experiment_actions(self.store,self.config,initialize=True);path=Path(result['labels']);data=read_json(path)
        data.update(labeler='independent-fixture',independent_of_predictions=True)
        for clip in data['clips']:
            if clip['split']=='development':clip.update(verified=True,labeler='independent-fixture',independent_of_predictions=True)
        atomic_json(path,data)
        self.assertEqual(experiment_actions(self.store,self.config,freeze=True)['status'],'development_frozen')
        data['clips'][0]['case_tags']=['new-choice'];atomic_json(path,data)
        with self.assertRaisesRegex(ValueError,'already frozen'):experiment_actions(self.store,self.config,freeze=True)

    def test_unresolved_cooldown_candidates_are_not_scored_as_activations(self):
        from annotation.league.core import write_rows
        directory=self.root/'core';directory.mkdir()
        candidate={'id':'unknown','owner_id':'selected','slot_id':'Q','start':.1,'end':.15,'activation_supported':False}
        for name,rows in [('actions',[]),('core_actions',[]),('body_readings',[]),('identities',[]),('transitions',[candidate]),('resampling',[])]:
            write_rows(directory/(name+'.jsonl'),rows)
        clip={'bindings':[],'identities':[],'transitions':[candidate],'actions':[]}
        metrics=score_clip(clip,{'results':{'core':{'directory':str(directory)}}})
        self.assertEqual(metrics['transitions']['recall'],0)
        self.assertEqual(metrics['transitions']['prediction_count'],0)
        self.assertEqual(metrics['transition_candidates']['matched'],1)
        self.assertEqual(metrics['unresolved_transition_candidates'],1)

    def test_interruption_keeps_partial_and_resource_report(self):
        with patch('annotation.league.action_v3.prepare_inputs',return_value=self.audit),patch('annotation.league.action_v3.bind_frames',side_effect=RuntimeError('decoder interrupted')):
            with self.assertRaises(RuntimeError):action_window(self.store,self.config,0,.25,detect=True)
        report=read_json(self.root/'report.json');self.assertEqual(report['status'],'failed_experimental')
        self.assertIsInstance(report['resource_metrics'],dict)


class CatalogTests(unittest.TestCase):
    def test_official_catalog_versions_tooltip_gaps_and_icon_integrity(self):
        stream=io.BytesIO();Image.new('RGB',(32,32),'green').save(stream,format='PNG');png=stream.getvalue()
        roster={'version':'16.13.1','data':{'Example':{'id':'Example'}}}
        spell={'id':'ExampleQ','name':'Example ability','description':'<b>Official</b> description',
               'tooltip':'Damage {{ missing }}','image':{'full':'Example.png'}}
        detail={'version':'16.13.1','data':{'Example':{'name':'Example','passive':{'name':'Passive','description':'Passive text','image':{'full':'Passive.png'}},'spells':[spell]*4}}}
        def get(url):return (roster if url.endswith('/champion.json') else detail,{'url':url,'sha256':'test','retrieved_at':'now'})
        with tempfile.TemporaryDirectory() as tmp,patch('annotation.league.ability_catalog.JsonCache.get',side_effect=get),patch('annotation.league.ability_catalog.download',return_value=png):
            report=cache_abilities(tmp);catalog=read_json(Path(report['catalog']))
            self.assertEqual(catalog['champions']['Example']['abilities'][1]['tooltip']['unresolved'],['missing'])
            self.assertIsNone(catalog['source_patch']);self.assertFalse(catalog['mechanics_enabled'])
            icon=Path(report['catalog']).parent/'icons/spell/Example.png';icon.write_bytes(b'tampered')
            self.assertEqual(cache_abilities(tmp)['status'],'partial')


if __name__=='__main__':unittest.main()
