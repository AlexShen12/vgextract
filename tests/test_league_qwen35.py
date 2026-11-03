"""Behavioral tests for the Qwen3.5 study; inference is injected at the GPU boundary."""
import copy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import pytest

from annotation.league.core import Store, Config, atomic_json, read_json, uid, digest
from annotation.league.local_models import LOCK, ModelUnavailable
from annotation.league.qwen35_common import Requests, ROLES, native_box, box_ok, POLICY
from annotation.league.qwen35_identity import track_detections, resolve_identities, canonical
from annotation.league.qwen35_sensors import hp_value, hp_metrics, overlay_views, align_abilities, consensus
from annotation.league.action_state import transitions
from annotation.league.qwen35_study import choose_neighborhoods, experiment_qwen35
from annotation.league.qwen35_scoring import box_metrics, action_metrics
from annotation.league.action_v3 import load_profile
from annotation.league.media import file_hash


def detection(frame='f1',time=1,track='t',name='alpha',champion_guess=None,color='green',segment='s'):
    return {'id':'d-'+frame,'frame_id':frame,'time':time,'track_id':track,'local_id':'p1','segment_id':segment,
            'owner_epoch':0,'body_box':[100,150,200,300],'hp_box':[100,120,200,130],'name_box':[90,90,210,115],
            'name':name,'name_kind':'player','bar_color':color,'champion_guess':champion_guess,'verified':True,'main_player':True,'green_bar_verified':True}


def score_row(frame='s1',time=2,name='alpha',champion='Darius'):
    return {'id':'score-'+frame,'frame_id':frame,'time':time,'name':name,'champion':champion,
            'portrait_confirmed':True,'reference_champion':champion,'row_id':'left:row1'}


def test_coordinate_conversion_and_rejection():
    assert native_box([10,20,40,50],[0,40,1920,980],[1280,627])==[15.0,69.984,60.0,114.96]
    with pytest.raises(ValueError):native_box([0,0,2000,1],[0,0,1920,1080],[1280,627])
    assert not box_ok([0,0,float('nan'),20],[1920,1080])


def test_track_continuity_and_palette_ambiguity():
    a=detection();b=detection('f2',1.25);c=detection('f3',1.5,color='red')
    rows=[{'frame_id':f,'time':t,'segment_id':s,'champions':ds} for f,t,s,ds in [
        ('f1',1,'s',[a]),('f2',1.25,'s',[b]),('f3',1.5,'cut',[c])]]
    out=track_detections(rows)
    assert out[0]['track_id']==out[1]['track_id']
    assert out[2]['track_id']!=out[1]['track_id']
    assert not out[2]['main_player']
    other={**a,'local_id':'p2','body_box':[300,100,400,250],'hp_box':[300,90,400,100]}
    out=track_detections([dict(rows[0],champions=[a,other])])
    assert not any(d['main_player'] for d in out)


def test_disappearance_and_crossing_do_not_invent_tracks():
    a=detection();rows=[{'frame_id':'f1','time':1,'segment_id':'s','champions':[a]},
        {'frame_id':'f2','time':1.25,'segment_id':'s','champions':[]},
        {'frame_id':'f3','time':1.5,'segment_id':'s','champions':[a]}]
    out=track_detections(rows);assert out[0]['track_id']!=out[1]['track_id']


def test_two_pass_late_identity_and_reappearance():
    ds=[detection('f0',.5,name=None),detection('f1',1),detection('f2',1.25),detection('f3',5,track='new')]
    resolved,registry,conflicts=resolve_identities(ds,[score_row(time=3)])
    assert not conflicts and len(registry)==1
    assert all(d['champion']=='Darius' for d in resolved)
    assert resolved[0]['retrospective']
    before,_,_=resolve_identities(ds,[score_row(time=3)],two_pass=False)
    assert before[0]['champion'] is None


def test_name_collisions_and_champion_name_mode():
    ds=[detection('f1',1),detection('f2',1.25)]
    scores=[score_row(),score_row('s2',3,champion='Jax')]
    out,registry,conflicts=resolve_identities(ds,scores)
    assert not registry and conflicts and all(d['champion'] is None for d in out)
    ds=[dict(d,name_kind='champion') for d in ds]
    assert not resolve_identities(ds,[score_row()])[1]
    assert canonical(' A\u0301 ')=='Á'
    assert canonical('Alpha')!=canonical('alpha')


def test_duplicate_names_in_one_scoreboard_cannot_bind():
    ds=[detection('f1',1),detection('f2',1.25)]
    assert not resolve_identities(ds,[score_row(),dict(score_row(),row_id='right:row1')])[1]


def test_conflicting_visual_guess_does_not_override_roster():
    ds=[detection('f1',1,champion_guess='Jax'),detection('f2',1.25,champion_guess='Jax')]
    assert not resolve_identities(ds,[score_row()])[1]


def hud(t,state,slot='Q',segment='s'):
    return {'id':str((t,state,slot)),'frame_id':str(t),'time':t,'state':state,'slot_id':slot,
            'slot_kind':'ability','owner_id':'selected','owner_epoch':0,'segment_id':segment,
            'seconds':5 if state=='cooling_down' else None,'charges':None,'charge_consumption_visible':False,'recast_visible':False}


def test_fresh_activation_vs_tick_unknown_and_cut():
    rows=[hud(1,'ready'),hud(1.0625,'cooling_down'),hud(1.125,'cooling_down')]
    out=transitions(rows);assert len(out)==1 and out[0]['activation_supported']
    out=transitions([hud(1,'unknown'),hud(1.0625,'cooling_down')]);assert not out[0]['activation_supported']
    assert not transitions([hud(1,'ready'),hud(1.0625,'cooling_down',segment='cut')])
    assert not transitions([hud(1,'ready'),hud(1.25,'cooling_down')])


def test_competing_slots_stay_unresolved():
    ds=[detection('f1',1),detection('f2',1.25)]
    action={'id':'a','frame_id':'f1','time':1,'action':'ability_action','track_id':'t','segment_id':'s','owner_epoch':0}
    q=transitions([hud(1,'ready'),hud(1.0625,'cooling_down')])[0]
    assert align_abilities([q],[action],ds)[0]['support']=='corroborated'
    e={**q,'id':'e','slot_id':'E'}
    assert all(r['support']=='unresolved' for r in align_abilities([q,e],[action],ds))
    assert align_abilities([q],[dict(action,track_id='other')],ds)[0]['slot_id'] is None


def test_hp_schema_rejects_mismatched_digits_and_zero_maximum():
    valid={'status':'visible','text':'100/200','current':100,'maximum':200}
    assert hp_value(valid)['current']==100
    with pytest.raises(ValueError):hp_value(dict(valid,current=101))
    with pytest.raises(ValueError):hp_value(dict(valid,text='0/0',current=0,maximum=0))
    with pytest.raises(ValueError):hp_value(dict(valid,status='unreadable'))


def test_hp_metrics_count_unreadable_acceptances_and_missing_predictions():
    refs=[{'frame_id':'a','status':'visible','text':'100/200','current':100,'maximum':200},
          {'frame_id':'b','status':'unreadable','text':None,'current':None,'maximum':None}]
    value={'accepted':True,'text':'100/200','current':100,'maximum':200}
    p=[{'frame_id':f,**{arm:value for arm in ('first','retry_policy','first_bar','retry_policy_bar')}} for f in ('a','b')]
    metrics=hp_metrics(refs,p)['first'];assert metrics['precision']==.5 and metrics['incorrect_unreadable_acceptances']==1
    assert not metrics['qualifies']
    assert hp_metrics(refs,[])['first']['readable_coverage']==0


def test_overlay_views_preserve_source_and_mark_hidden_pixels():
    image=Image.new('RGB',(43,45),(80,20,60));before=image.tobytes()
    a=np.asarray(image).copy();a[18:28,15:25]=255;a[:10]=0;image=Image.fromarray(a);before=image.tobytes()
    view,mask,meta=overlay_views(image)
    assert image.tobytes()==before and mask.any() and not meta['reconstructed_pixels']
    assert np.all(np.asarray(view)[mask]==[255,0,255])
    assert consensus({'Q':'Darius','W':'Darius','E':'Darius','R':None})=='Darius'
    assert consensus({'Q':'Darius','W':'Darius','E':'Darius','R':'Jax'}) is None


def test_neighborhoods_exclude_old_windows_and_are_grouped():
    clips=choose_neighborhoods([0,1732],[[20,60],[800,900]])
    assert len(clips)==20 and sum(c['split']=='held_out' for c in clips)==10
    assert all(not(c['start']<b and a<c['end']) for c in clips for a,b in [[20,60],[800,900]])


class FakeRunner:
    def __init__(self,role=ROLES[-1],fail_at=None):
        self.identity={'checkpoint':{'specification':read_json(LOCK)['models'][role]},'runtime':{'gpu':'NVIDIA L40S','gpu_uuid':'fixture','total_memory_bytes':48_000_000_000}}
        self.calls=0;self.fail_at=fail_at;self.closed=False
    def complete(self,messages,schema=None,**kwargs):
        self.calls+=1
        if self.fail_at==self.calls:raise ModelUnavailable('injected GPU failure')
        props=schema['properties'];content=messages[0]['content']
        labels=[r['text'] for r in content[1:] if r['type']=='text']
        if set(props)=={'status'}:v={'status':'image_received'}
        elif 'current' in props:v={'status':'visible','text':'100/200','current':100,'maximum':200}
        elif 'champions' in props:
            v={'champions':[{'local_id':'p1','body_box':[100,100,180,200],'hp_box':[100,80,170,90],
                'name_box':[90,50,180,75],'name':'alpha','name_kind':'player','bar_color':'green','champion_guess':None,'verified':True}]}
        elif 'name_kind' in props:v={'name':'alpha','name_kind':'player','champion_guess':None,'verified':True}
        elif 'frames' in props:v={'frames':[{'frame_id':label,'action':'ability_action','description':'Visible cast motion.'} for label in labels if ' ' not in label]}
        elif 'readings' in props:v={'readings':[{'frame_id':label,'slot':slot,'state':'ready' if i==0 else 'cooling_down','seconds':None if i==0 else 5,
            'charges':None,'charge_consumption_visible':False,'recast_visible':False} for i,label in enumerate(labels) for slot in 'QWER']}
        elif 'slots' in props:
            slots=sorted({label[0] for label in labels if label[0] in 'QWER'})
            v={'slots':[{'slot':s,'champion':'Darius','clean':False,'evidence':'Visible icon artwork.'} for s in slots]}
        elif 'rows' in props:v={'rows':[]}
        else:raise AssertionError(props)
        return {'text':json.dumps(v),'finish_reason':'stop','wall_seconds':.1,'usage':{'completion_tokens':20,'prompt_tokens':40},'peak_memory_bytes':100}
    def close(self):self.closed=True


@pytest.fixture
def fixture_study(tmp_path):
    store=Store(tmp_path/'run');store.root.mkdir()
    video=tmp_path/'source.bin';video.write_bytes(b'source-fixture')
    source={'path':str(video),'sha256':file_hash(video),'fps':60,'width':1920,'height':1080,'duration':40}
    atomic_json(store.path('manifest.json'),{'source':source,'interval':[0,40]})
    config=Config(patch={},routes=[],asset_cache=str(tmp_path/'assets'))
    directory=store.path('benchmarks/qwen35');directory.mkdir(parents=True)
    profile={**load_profile(),'scoreboard_portrait_x':{'left':491,'right':1030}}
    clips=[{'id':'clip-'+str(i),'start':start,'end':start+1,'split':split,'case_tags':[],'verified':False,
            'independent_of_predictions':False,'labeler':None,**{k:[] for k in ('hp','detections','scoreboard','actions','activations','icons','negative_categories')}}
           for i,(start,split) in enumerate([(1,'development'),(10,'held_out')])]
    study={'schema_version':1,'source_sha256':source['sha256'],'source':source,'profile':profile,'policy':POLICY,
           'excluded_intervals':[],'clips':clips,'independent_of_predictions':False,'labeler':None}
    atomic_json(directory/'study.json',study)
    def frames(store,config,study,clip,directory,**kwargs):
        result=[]
        for i in range(2):
            time=clip['start']+i*.0625 if kwargs.get('dense') else clip['start']+i*.25
            key=uid('q35-frame',source['sha256'],time);path=tmp_path/(key+'.png')
            Image.new('RGB',(1920,1080),(30+i,80,40)).save(path)
            result.append({'id':key,'time':time,'clip_id':clip['id'],'split':clip['split'],'segment_id':clip['id'],
                'owner_epoch':0,'layout_valid':True,'scene_box':profile['scene_box'],'scene_size':[1920,1080],
                'native_path':str(path),'scene_path':str(path),'hud_path':str(path),'hp_sample':True,
                'hp_path':str(path),'hp_processed_path':str(path),'health_fraction':.5,'hud_slot_boxes':{},
                'icon_paths':{s:str(path) for s in 'QWER'}})
        return result
    return store,config,directory,study,frames


def test_requests_reuse_raw_cache_and_invalidate_prompt_and_images(tmp_path):
    store=Store(tmp_path/'run');store.root.mkdir();image=tmp_path/'x.png';Image.new('RGB',(10,10)).save(image)
    runner=FakeRunner();req=Requests(store,None,tmp_path/'a',ROLES[-1],runner=runner)
    schema={'type':'object','properties':{'status':{'enum':['image_received']}},'required':['status'],'additionalProperties':False}
    a,k=req.call('preflight','hello',[('source',image)],schema)
    assert a and runner.calls==1
    req2=Requests(store,None,tmp_path/'b',ROLES[-1],runner=runner)
    assert req2.call('preflight','hello',[('source',image)],schema)[1]==k and runner.calls==1
    assert req2.call('preflight','changed',[('source',image)],schema)[1]!=k
    Image.new('RGB',(10,10),'red').save(image)
    assert req2.call('preflight','hello',[('source',image)],schema)[1]!=k


def test_all_suites_run_with_injected_gpu_and_no_production_writes(fixture_study):
    store,config,directory,study,frames=fixture_study
    with patch('annotation.league.qwen35_study.source_frames',side_effect=frames),patch('annotation.league.qwen35_study.burst_frames',return_value=[]):
        report=experiment_qwen35(store,config,'qwen35',draft=True,runner_factory=FakeRunner)
    assert report['status']=='completed_draft' and len(report['results'])==4
    assert all(v['development']['metrics'] is None for v in report['quality_metrics'].values())
    assert all(r['status']=='ok' for r in report['requests'])
    assert not store.path('observations.jsonl').exists() and not store.path('events.jsonl').exists()
    for suite in report['results']:
        rows=[json.loads(r) for r in Path(report['results'][suite]['path']).read_text().splitlines()]
        assert all(r['split']=='development' for r in rows)
    assert (Path(report['root'])/'report.md').exists()


def test_interrupted_gpu_preserves_partial_results(fixture_study):
    store,config,directory,study,frames=fixture_study
    with patch('annotation.league.qwen35_study.source_frames',side_effect=frames):
        with pytest.raises(ModelUnavailable):
            experiment_qwen35(store,config,'hp-vlm',draft=True,runner_factory=lambda role:FakeRunner(role,fail_at=3))
    reports=list(store.path('experiments/qwen35').glob('*/report.json'))
    assert read_json(reports[0])['status']=='failed'
    assert list(reports[0].parent.glob('requests/*/response.json'))
    assert list(reports[0].parent.glob('hp-vlm/*/development/*.jsonl'))==[] # interrupted role is not marked complete


def test_frozen_evaluation_requires_independent_labels(fixture_study):
    store,config,*_=fixture_study
    with pytest.raises(ValueError,match='Freeze independently'):
        experiment_qwen35(store,config,'hp-vlm')
    with pytest.raises(ValueError,match='ten development'):
        experiment_qwen35(store,config,'qwen35',freeze=True)


def test_source_only_review_and_media_boundary(fixture_study):
    from fastapi.testclient import TestClient
    from annotation.league.review import create_app
    store,config,directory,study,frames=fixture_study
    client=TestClient(create_app(store));sid=digest(str(directory.resolve()))
    response=client.get('/api/qwen35/labels',params={'study':sid})
    assert response.status_code==200 and 'predictions' not in response.json()
    assert client.get('/qwen35-labels').status_code==200
    assert client.get('/api/qwen35/media',params={'path':'/etc/passwd'}).status_code==403
    assert client.post('/api/qwen35/labels/'+sid+'/clip-0',json={'start':0}).status_code==400
    assert client.post('/api/qwen35/labels/'+sid+'/clip-0',headers={'Origin':'http://evil.test'},json={}).status_code==403


def test_cli_suite_choices_preserve_all_semantics():
    from annotation.league.__main__ import parser
    for s in ('hp-vlm','scene-identity','ability-alignment','cooldown-identity','qwen35','all'):
        assert parser().parse_args(['experiment','--suite',s]).suite==s
    assert parser().parse_args(['cache-models','--suite','qwen35']).suite=='qwen35'


def test_wrong_ability_slot_is_not_reported_as_perfect_accuracy():
    from annotation.league.qwen35_scoring import activation_metrics
    ref=[{'clip_id':'c','start':1,'end':1.1,'slot_id':'Q'}]
    pred=[{'clip_id':'c','start':1,'end':1.1,'slot_id':'E','activation_supported':True}]
    result=activation_metrics(ref,pred)
    assert result['precision']==0 and result['matched_slot_accuracy']==0


def test_owner_change_interrupts_spatial_track():
    a=detection();rows=[{'frame_id':'f1','time':1,'segment_id':'s','owner_epoch':0,'champions':[a]},
                       {'frame_id':'f2','time':1.25,'segment_id':'s','owner_epoch':1,'champions':[a]}]
    out=track_detections(rows)
    assert out[0]['track_id']!=out[1]['track_id']


def test_corrections_change_derived_recipe_but_preserve_source_observation(tmp_path):
    from annotation.league.qwen35_review import corrected
    directory=tmp_path/'labels';directory.mkdir()
    raw=detection()
    atomic_json(directory/'corrections/00000001.json',{'table':'detections','arm':'hybrid','record_id':raw['id'],'changes':{'name':'beta'}})
    out=corrected(directory,'detections','hybrid',[raw])
    assert out[0]['name']=='beta' and raw['name']=='alpha'
    assert corrected(directory,'detections','direct',[raw])[0]['name']=='alpha'


def test_custom_study_is_registered_for_source_only_labeling(fixture_study):
    from annotation.league.qwen35_review import studies
    store,config,directory,study,frames=fixture_study
    custom=store.root.parent/'custom-labels';custom.mkdir();atomic_json(custom/'study.json',study)
    with patch('annotation.league.qwen35_study.source_frames',side_effect=frames):
        report=experiment_qwen35(store,config,'qwen35',labels=custom,prepare_only=True)
    assert report['status']=='source_prepared'
    assert custom.resolve() in studies(store).values()
