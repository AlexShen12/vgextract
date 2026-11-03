"""Preparation and orchestration for four isolated Qwen3.5 pilot experiments."""
from __future__ import annotations

from collections import defaultdict
import copy
import json
from pathlib import Path

from .core import atomic_json, digest, implementation_hash, read_json, read_rows, uid, write_rows
from .intake import sampled_frames, image_hash
from .intake_profile import load_profile, layout_matches
from .media import file_hash
from .playback import PlaybackTracker
from .qwen35_common import SUITES, ROLES, POLICY, Requests
from .qwen35_identity import detect_frame, track_detections, resolve_identities, scoreboard_rows, describe_frames
from .qwen35_sensors import read_hp, read_hud, align_abilities, cooldown_frame, References

COMPONENTS = ['qwen35_common.py','qwen35_identity.py','qwen35_sensors.py','qwen35_study.py','qwen35_scoring.py']
REQUIRED_CASES = {'hp-vlm': {'ordinary_hp','rapid_hp','low_hp','digit_lengths','unreadable_hp'},
 'scene-identity': {'scoreboard_open','multiple_champions','occlusion','reappearance','quiet_negative'},
 'ability-alignment': {'basic_attack','ability_cast','rapid_sequence','recast','unavailable','cut','cooldown_only'},
 'cooldown-identity': {'clean_icon','cooldown_digits','radial_shading','unavailable','ownership_change'}}


def study_directory(store, labels=None):
    return Path(labels) if labels is not None else store.path('benchmarks/qwen35')


def used_intervals(store):
    """Exclude all previously submitted inference image time ranges, not label answers."""
    intervals=[]
    for p in store.path('experiments').rglob('input-audit.json'):
        if 'qwen35' in p.parts:continue
        data=read_json(p)
        if isinstance(data.get('interval'),list):intervals.append(data['interval'])
    # Imported Longleaf artifacts are outside this run's experiments tree.
    external_roots=[store.root.parents[1]/'league_caption_pilot_validation',store.root.parent/'artifacts',
                    store.root.parents[1]/'artifacts']
    for imported in external_roots:
        for p in imported.rglob('input-audit.json') if imported.exists() else []:
            data=read_json(p)
            if isinstance(data.get('interval'),list):intervals.append(data['interval'])
    for p in store.path('experiments/qwen35').glob('*/report.json'):
        report=read_json(p);snapshot=p.parent/'study-snapshot.json'
        if snapshot.exists() and any(r.get('status')=='ok' and r.get('task')!='preflight' for r in report.get('requests',[])):
            splits=report.get('inference_splits',['development'])
            intervals.extend([c['start'],c['end']] for c in read_json(snapshot)['clips'] if c['split'] in splits)
    return intervals


def choose_neighborhoods(interval, excluded):
    start,end=interval
    candidates=[]
    # Leave a guard interval between neighborhoods and all previously inspected windows.
    t=start+12
    while t+8<end-4:
        if not any(t < b+2 and a-2 < t+8 for a,b in excluded):candidates.append(round(t,6))
        t+=12
    if len(candidates)<20:raise ValueError('Need twenty uninspected, disjoint eight-second neighborhoods')
    selected=[candidates[round(i*(len(candidates)-1)/19)] for i in range(20)]
    return [{'id':f'clip-{i+1:02d}','start':t,'end':t+8,'split':'development' if i%2==0 else 'held_out',
             'case_tags':[],'verified':False,'independent_of_predictions':False,'labeler':None,
             'hp':[],'detections':[],'scoreboard':[],'actions':[],'activations':[],'icons':[],
             'negative_categories':[]} for i,t in enumerate(selected)]


def initialize(store, config, directory, profile):
    target=directory/'study.json'
    if target.exists():raise ValueError('Refusing to overwrite Qwen3.5 labels; use a new label directory')
    source=read_json(store.path('manifest.json'))
    if file_hash(Path(source['source']['path']))!=source['source']['sha256']:
        raise ValueError('Source video changed')
    excluded=used_intervals(store)
    study={'schema_version':1,'source_sha256':source['source']['sha256'],'source':source['source'],
           'profile':profile,'policy':POLICY,'excluded_intervals':excluded,
           'independent_of_predictions':False,'labeler':None,
           'clips':choose_neighborhoods(source['interval'],excluded)}
    atomic_json(target,study)
    # Preparation can resume separately if source decoding is interrupted.
    return {'status':'labels_initialized','experimental':True,'labels':str(target),'clips':20,
            'next':'experiment --suite qwen35 --prepare-only (source-only evidence for both splits)'}


def source_frames(store,config,study,clip,directory,*,dense=False):
    from PIL import Image, ImageDraw, ImageOps
    import numpy as np
    from .readers import measure_bar, parse_text
    from .numeric_reader import numeric_preprocess
    from .readers import tesseract
    profile=study['profile'];anchors=load_profile(profile['anchor_profile'])
    fps=POLICY['hud_fps'] if dense else POLICY['scene_fps']
    recipe={'source':study['source_sha256'],'clip':{k:clip[k] for k in ('id','start','end','split')},
            'profile':profile,'masks':config.masks,'fps':fps,'policy':POLICY,
            'implementation':implementation_hash(COMPONENTS+['playback.py','intake.py'])}
    root=directory/'source'/digest(recipe)
    audit_path=root/'audit.json'
    if audit_path.exists():
        audit=read_json(audit_path)
        if all(Path(p).is_file() and file_hash(Path(p))==sha for f in audit['frames'] for p,sha in f['files'].items()):
            return audit['frames']
    root.mkdir(parents=True,exist_ok=True)
    tracker=PlaybackTracker(study['source_sha256'],fps,scene_box=profile['scene_box'],masks=config.masks)
    owner=0;prior_portrait=None;last_clock=None;result=[];seen_hp=set()
    for index,(pts,image) in enumerate(sampled_frames(study['source']['path'],clip['start'],clip['end'],fps,store.check_stop)):
        stem=uid('q35-frame',study['source_sha256'],pts)
        valid,_=layout_matches(image,anchors)
        if index%fps==0:
            clock=root/'clock.png';numeric_preprocess(image.crop(profile['clock_box'])).save(clock)
            try:last_clock=parse_text(tesseract(clock,whitelist='0123456789:')['raw_text'],'clock')['seconds']
            except (ValueError,RuntimeError,OSError):last_clock=None
        playback=tracker.update(pts,last_clock,image,valid)
        portrait=np.asarray(image.crop(profile['owners'][0]['identity_box']).resize((24,24)),dtype=float)
        if prior_portrait is not None and np.abs(portrait-prior_portrait).mean()/255>profile['thresholds']['identity_change']:
            owner+=1;prior_portrait=portrait
        elif prior_portrait is None:prior_portrait=portrait
        f={'id':stem,'time':pts,'split':clip['split'],'clip_id':clip['id'],'owner_epoch':owner,
           'segment_id':uid('q35-segment',clip['id'],playback['playback_segment_id']),
           'layout_valid':valid,'playback':playback,'scene_box':profile['scene_box'],'files':{}}
        def save(im,label):
            path=root/f'{stem}-{label}.png';im.save(path)
            f['files'][str(path)]=file_hash(path)
            return str(path)
        f['hud_path']=save(image.crop(profile['hud_box']),'hud')
        hx,hy,_,_=profile['hud_box']
        slots=[s for s in profile['owners'][0]['slots'] if s['id'] in 'QWER' and len(s['id'])==1]
        f['hud_slot_boxes']={s['id']:[s['box'][0]-hx,s['box'][1]-hy,s['box'][2]-hx,s['box'][3]-hy] for s in slots}
        if not dense:
            f['native_path']=save(image,'native')
            masked=image.copy();draw=ImageDraw.Draw(masked)
            for box in config.masks:draw.rectangle(box,fill='black')
            scene=masked.crop(profile['scene_box']);width=POLICY['scene_width']
            scene=scene.resize((width,round(scene.height*width/scene.width)),Image.Resampling.LANCZOS)
            f['scene_path']=save(scene,'scene');f['scene_size']=list(scene.size)
            hp=image.crop([867,1041,972,1059]);hp_hash=image_hash(hp)
            f['hp_sample']=hp_hash not in seen_hp and len(seen_hp)<12
            if f['hp_sample']:
                seen_hp.add(hp_hash)
                f['hp_path']=save(hp,'hp')
                processed=ImageOps.expand(hp.resize((hp.width*4,hp.height*4),Image.Resampling.NEAREST),border=8,fill='black')
                f['hp_processed_path']=save(processed,'hp-processed')
                measured=measure_bar(image.crop([756,1044,1057,1045]),{'color':'green'})
                f['health_fraction']=measured['value'] if measured['acceptance']=='accepted' else None
            f['icon_paths']={s['id']:save(image.crop(s['box']),'icon-'+s['id']) for s in slots}
        result.append(f)
    expected=round((clip['end']-clip['start'])*fps)
    if len(result)!=expected:raise ValueError(f'Incomplete source decode: {len(result)}/{expected}')
    atomic_json(audit_path,{'recipe':recipe,'frames':result})
    return result


def load_study(store,directory):
    study=read_json(directory/'study.json');manifest=read_json(store.path('manifest.json'))
    if study.get('schema_version')!=1 or study['source_sha256']!=manifest['source']['sha256']:
        raise ValueError('Qwen3.5 study source/schema mismatch')
    if study.get('policy')!=POLICY:raise ValueError('Study policy changed; initialize a new benchmark revision')
    if file_hash(Path(study['source']['path']))!=study['source_sha256']:raise ValueError('Source video checksum mismatch')
    ids=set();intervals=[]
    for clip in study['clips']:
        if clip['id'] in ids or clip['split'] not in ('development','held_out'):
            raise ValueError('Duplicate clip or invalid split')
        if not manifest['interval'][0]<=clip['start']<clip['end']<=manifest['interval'][1] or clip['end']-clip['start']>30:
            raise ValueError('Invalid clip bounds')
        if any(clip['start']<b and a<clip['end'] for a,b in intervals):raise ValueError('Overlapping benchmark neighborhoods')
        ids.add(clip['id']);intervals.append((clip['start'],clip['end']))
        if clip['split']=='held_out' and any(clip['start']<b and a<clip['end'] for a,b in study['excluded_intervals']):
            raise ValueError('Held-out clip intersects historical inference')
    return study


def choices(study,reference_signature):
    from .local_models import LOCK
    return {'source':study['source_sha256'],'policy':POLICY,'profile':study['profile'],
            'models':{r:read_json(LOCK)['models'][r] for r in ROLES},'reference_signature':reference_signature,
            'implementation':implementation_hash(COMPONENTS+['local_models.py','readers.py','action_state.py']),
            'clips':[{k:c[k] for k in ('id','start','end','split')} for c in study['clips']]}


def freeze_development(directory,study,recipe):
    from .qwen35_scoring import validate_labels
    validate_labels(study,'development',SUITES)
    path=directory/'development-freeze.json'
    record={'choices':recipe,'development_labels':digest([c for c in study['clips'] if c['split']=='development'])}
    if path.exists() and read_json(path)!=record:raise ValueError('Development is already frozen; use a new benchmark revision')
    atomic_json(path,record)
    return {'status':'development_frozen','experimental':True,'freeze':str(path)}


def experiment_qwen35(store,config,suite,*,labels=None,initialize_labels=False,draft=False,freeze=False,
                       infer=False,preflight=False,prepare_only=False,icon_references=None,action_profile=None,runner_factory=None):
    from .action_v3 import load_profile
    from .action_state import transitions
    from .qwen35_scoring import score_study, validate_labels, write_report
    directory=study_directory(store,labels);directory.mkdir(parents=True,exist_ok=True)
    profile=load_profile(action_profile)
    # Explicit development geometry, shown in the source-only labeling viewer.
    profile={**profile,'scoreboard_portrait_x':{'left':491,'right':1030}}
    if initialize_labels:return initialize(store,config,directory,profile)
    study=load_study(store,directory)
    reference_path=Path(icon_references) if icon_references else Path(config.asset_cache)/'qwen35-references'/'16.13.1'/'manifest.json'
    if not reference_path.exists() and icon_references:raise ValueError('Explicit identity reference manifest is missing')
    references=References(reference_path) if reference_path.exists() else None
    if prepare_only:
        counts={}
        for clip in study['clips']:
            counts[clip['id']]=len(source_frames(store,config,study,clip,directory))
        return {'status':'source_prepared','experimental':True,'frames':counts,'labels':str(directory/'study.json')}
    recipe=choices(study,references.signature if references else None)
    recipe['corrections']=digest([read_json(p) for p in sorted((directory/'corrections').glob('*.json'))])
    if freeze:return freeze_development(directory,study,recipe)
    suites=SUITES if suite=='qwen35' else (suite,)
    if not draft:
        freeze_path=directory/'development-freeze.json'
        if not freeze_path.exists():raise ValueError('Freeze independently labeled development before held-out scoring; use --draft to explore')
        frozen=read_json(freeze_path)
        if frozen['choices']!=recipe or frozen['development_labels']!=digest([c for c in study['clips'] if c['split']=='development']):
            raise ValueError('Frozen development choices/labels changed')
        for split in ('development','held_out'):validate_labels(study,split,suites)
    selected=[c for c in study['clips'] if not draft or c['split']=='development']
    root=store.path('experiments/qwen35')/digest(recipe);root.mkdir(parents=True,exist_ok=True)
    atomic_json(root/'study-snapshot.json',study);atomic_json(root/'recipe.json',recipe)
    atomic_json(root/'label-directory.json',{'path':str(directory.resolve())})
    report={'experimental':True,'production_evidence_modified':False,'root':str(root),'status':'running',
            'suites':list(suites),'inference_splits':sorted({c['split'] for c in selected}),'reference_status':'available' if references else 'missing',
            'quality_metrics':None,'acceptance':False,'requests':[],'results':{},'missing_cases':{}}
    atomic_json(root/'report.json',report)
    import time
    started=time.monotonic()
    all_frames={};all_predictions={};requests=None
    try:
        for clip in selected:
            frames=source_frames(store,config,study,clip,directory)
            all_frames[clip['id']]=frames
        if preflight:
            for role in ROLES:
                requests=Requests(store,config,root,role,infer=infer,
                                  runner=runner_factory(role) if runner_factory else None)
                try:requests.qualify(next(iter(all_frames.values()))[0]['hud_path'])
                finally:report['requests']+=requests.records;requests.close();requests=None
            report['status']='preflight_completed'
            return report
        for current in suites:
            roles=ROLES if current=='hp-vlm' else (ROLES[-1],)
            suite_rows=[]
            for role in roles:
                requests=Requests(store,config,root,role,infer=infer,
                                  runner=runner_factory(role) if runner_factory else None)
                try:
                    if infer or runner_factory:requests.qualify(next(iter(all_frames.values()))[0]['hud_path'])
                    hp_seen={}
                    # Reset registries and temporal icon memory between arms and splits.
                    for split in ('development','held_out'):
                        clips=[c for c in selected if c['split']==split]
                        if current=='hp-vlm':
                            for clip in clips:
                                frames=copy.deepcopy(all_frames[clip['id']])
                                for f in frames:
                                    if f.get('hp_sample'):
                                        h=file_hash(Path(f['hp_path']))
                                        # Identical input cannot appear in both benchmark splits.
                                        if h in hp_seen:f['hp_sample']=False
                                        else:hp_seen[h]=split
                                out=read_hp(requests,frames)
                                for r in out:r.update(clip_id=clip['id'],split=split)
                                suite_rows+=out
                                write_rows(root/current/role/split/(clip['id']+'.jsonl'),out)
                        elif current in ('scene-identity','ability-alignment'):
                            for arm in ('direct','hybrid') if current=='scene-identity' else ('hybrid',):
                                detections=[];scores=[];frame_set=[];raw_rows=[]
                                for clip in clips:
                                    frames=all_frames[clip['id']];raw=[]
                                    for f in frames:
                                        store.check_stop()
                                        row=detect_frame(requests,f,study['profile'],arm) if f['layout_valid'] else {
                                            'frame_id':f['id'],'time':f['time'],'segment_id':f['segment_id'],'champions':[]}
                                        raw.append(row);raw_rows.append({**row,'clip_id':clip['id'],'split':split,'arm':arm,'table':'boxes'})
                                        scores+=scoreboard_rows(requests,f,study['profile'],references)
                                    from .qwen35_review import corrected
                                    for row in raw:
                                        champions=[{**c,'id':uid('q35-detection',row['frame_id'],c['local_id'])} for c in row['champions']]
                                        row['champions']=corrected(directory,'detections',arm,champions)
                                    scores=corrected(directory,'scoreboard',arm,scores)
                                    tracked=track_detections(raw)
                                    meta={f['id']:f for f in frames}
                                    for d in tracked:d.update(owner_epoch=meta[d['frame_id']]['owner_epoch'],clip_id=clip['id'])
                                    detections+=tracked;frame_set+=frames
                                resolved,registry,conflicts=resolve_identities(detections,scores)
                                before,_,_=resolve_identities(detections,scores,two_pass=False)
                                write_rows(root/current/role/split/arm/'registry.jsonl',registry)
                                write_rows(root/current/role/split/arm/'conflicts.jsonl',conflicts)
                                write_rows(root/current/role/split/arm/'scoreboard.jsonl',scores)
                                for table,rows in [('detections',resolved),('before_memory',before),('scoreboard',scores)]:
                                    for r in rows:
                                        frame=next((f for f in frame_set if f['id']==r['frame_id']),None)
                                        if frame:suite_rows.append({**r,'table':table,'arm':arm,'split':split,'clip_id':frame['clip_id']})
                                suite_rows+=raw_rows
                                for clip in clips:
                                    frames=all_frames[clip['id']];clip_d=[d for d in resolved if d['clip_id']==clip['id']]
                                    actions=describe_frames(requests,frames,clip_d)
                                    for r in actions:r.update(table='actions',arm=arm,split=split,clip_id=clip['id'])
                                    suite_rows+=actions
                                    if current=='ability-alignment':
                                        dense=source_frames(store,config,study,clip,directory,dense=True)
                                        hud=read_hud(requests,dense);activation=transitions(hud)
                                        # Bind temporal HUD source to this clip's sparse visual track IDs.
                                        combined=align_abilities(activation,actions,clip_d)
                                        assisted=describe_frames(requests,frames,clip_d,assisted=True,hints=[t for t in activation if t['activation_supported']])
                                        for table,rows in [('hud',hud),('activations',activation),('associations',combined),('assisted',assisted)]:
                                            for r in rows:r.update(table=table,arm=arm,split=split,clip_id=clip['id'])
                                            suite_rows+=rows
                                        # Dense scene descriptions are limited to fresh activation neighborhoods.
                                        bursts=burst_frames(store,config,study,clip,directory,activation)
                                        burst_raw=[]
                                        for f in bursts:
                                            nearest=min(dense,key=lambda d:abs(d['time']-f['time']))
                                            f.update(segment_id=nearest['segment_id'],owner_epoch=nearest['owner_epoch'])
                                            burst_raw.append(detect_frame(requests,f,study['profile'],'hybrid'))
                                        burst_d=track_detections(burst_raw)
                                        for d in burst_d:
                                            near=[p for p in clip_d if p.get('main_player') and d.get('main_player')
                                                  and p['segment_id']==d['segment_id'] and abs(p['time']-d['time'])<=.1]
                                            from .qwen35_common import iou
                                            same=[p for p in near if iou(p['hp_box'],d['hp_box'])>=.5]
                                            if len({p['track_id'] for p in same})==1:
                                                d.update(track_id=same[0]['track_id'],champion=same[0].get('champion'))
                                        burst_actions=describe_frames(requests,bursts,burst_d) if bursts else []
                                        for r in burst_actions:r.update(table='bursts',arm=arm,split=split,clip_id=clip['id'])
                                        suite_rows+=burst_actions
                        else:
                            memory={}
                            for clip in clips:
                                for f in all_frames[clip['id']]:
                                    if not f['layout_valid']:continue
                                    out=cooldown_frame(requests,f,references,memory)
                                    out.update(clip_id=clip['id'],split=split)
                                    suite_rows.append(out)
                                natural=[f for f in all_frames[clip['id']] if f['layout_valid']]
                                if natural:
                                    from .qwen35_sensors import synthetic_icon_frame
                                    synthetic=synthetic_icon_frame(natural[0],root)
                                    out=cooldown_frame(requests,synthetic,references,{})
                                    out.update(clip_id=clip['id'],split=split,natural=False,
                                               source_frame_id=natural[0]['id'],synthetic_recipe='brightness-.45-radial-150-digit-5')
                                    suite_rows.append(out)
                finally:
                    report['requests']+=[dict(r,suite=current) for r in requests.records];requests.close();requests=None
                    write_rows(root/current/'predictions.jsonl',suite_rows)
            all_predictions[current]=suite_rows
            report['results'][current]={'records':len(suite_rows),'path':str(root/current/'predictions.jsonl')}
            atomic_json(root/'report.json',report)
        report.update(score_study(study,selected,all_frames,all_predictions,report['requests'],draft=draft))
        report['status']='completed_draft' if draft else 'completed_evaluation'
        return report
    except BaseException as exc:
        report['status']='interrupted' if store.stopping else 'failed'
        report['error']=str(exc)
        raise
    finally:
        if requests:
            report['requests']+=requests.records;requests.close()
        report['wall_seconds']=time.monotonic()-started
        write_report(root,report)
        for current in suites:
            if current in report['results']:
                write_report(root/current,{**report,'suites':[current],
                    'results':{current:report['results'][current]},
                    'quality_metrics':(report.get('quality_metrics') or {}).get(current)})


def burst_frames(store,config,study,clip,directory,activations):
    """Decode 16 FPS scene context only in merged activation windows."""
    windows=[]
    for a in activations:
        if not a.get('activation_supported'):continue
        start=max(clip['start'],a['start']-1);end=min(clip['end'],a['end']+1)
        if windows and start<=windows[-1][1]:windows[-1][1]=max(end,windows[-1][1])
        else:windows.append([start,end])
    output=[]
    for i,(start,end) in enumerate(windows):
        burst={**clip,'id':clip['id']+f'-burst-{i}','start':start,'end':end}
        # Dense preparation stores HUD only. Scene preparation uses a temporary policy override through explicit fps.
        output+=scene_burst_source(store,config,study,burst,directory)
    return output


def scene_burst_source(store,config,study,clip,directory):
    from PIL import Image,ImageDraw
    profile=study['profile'];root=directory/'bursts'/digest([study['source_sha256'],clip,profile,config.masks])
    audit=root/'audit.json'
    if audit.exists():return read_json(audit)['frames']
    root.mkdir(parents=True,exist_ok=True);result=[]
    for pts,image in sampled_frames(study['source']['path'],clip['start'],clip['end'],16,store.check_stop):
        key=uid('q35-frame',study['source_sha256'],pts);native=root/(key+'-native.png');image.save(native)
        masked=image.copy();draw=ImageDraw.Draw(masked)
        for box in config.masks:draw.rectangle(box,fill='black')
        scene=masked.crop(profile['scene_box']);scene=scene.resize((1280,round(scene.height*1280/scene.width)),Image.Resampling.LANCZOS)
        path=root/(key+'-scene.png');scene.save(path)
        hud=root/(key+'-hud.png');image.crop(profile['hud_box']).save(hud)
        result.append({'id':key,'time':pts,'native_path':str(native),'scene_path':str(path),'hud_path':str(hud),
                       'scene_box':profile['scene_box'],'scene_size':list(scene.size),'owner_epoch':0,
                       'segment_id':clip['id'],'clip_id':clip['id'],'split':clip['split']})
    atomic_json(audit,{'frames':result})
    return result
