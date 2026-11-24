"""Refine unreadable clocks without repeating the dense slot or cast recognizers."""
from pathlib import Path
import time
from .core import atomic_json, digest, read_json, read_rows, uid, write_rows
from .media import file_hash
from .pass1_timeline import ContextIndex


def selection_reviews(store):
    corrections={c['record_id']:c for c in store.corrections() if c['target']=='observations'}
    if not corrections:return []
    result=[]
    from .pass1 import records
    for row in records(store.path('observations.jsonl')):
        if row['id'] in corrections and row['route']=='selected_hud' and row['field'] in {'inventory.champion','identity'}:
            result.append({'time':row['time'],'end':row['time']+1,'record_id':row['id'],
                           'review':corrections[row['id']]})
    return result


def context_recipe(store,windows,clocks):
    return digest({'windows':windows,'clocks':clocks,
        'profile':read_json(store.path('profiles/midbeast-pass1-1080p-v1.json')),
        'selection':file_hash(store.path('producers/dense/selection.jsonl')),
        'selection_reviews':selection_reviews(store),'code':file_hash(Path(__file__))})


def refine(store, windows, clocks):
    from .intake import evidence_crop
    from .intake_profile import layout_matches, load_profile
    from .numeric_reader import NumericReader
    from .playback import PlaybackTracker
    from .scene_reference import sample_source
    root=store.path('producers/context');root.mkdir(parents=True,exist_ok=True)
    profile=load_profile(str(store.path('profiles/midbeast-pass1-1080p-v1.json')))
    action=read_json(store.path('profiles/midbeast-actions-1080p-v2.json'))
    source=read_json(store.path('manifest.json'))['source']
    recipe=context_recipe(store,windows,clocks)
    if (root/'report.json').exists() and read_json(root/'report.json').get('recipe')==recipe:
        return read_json(root/'report.json')
    index=ContextIndex(clocks)
    selections=ContextIndex(read_rows(store.path('producers/dense/selection.jsonl')))
    reviews=selection_reviews(store)
    data={key:[] for key in ('playback','selection','clock_readings','frames')}
    reader=NumericReader(.85);started=time.monotonic()
    field=next(f for f in profile['fields'] if f['field']=='game_clock')
    for a,b in windows:
        tracker=PlaybackTracker(source['sha256'],16,scene_box=action['scene_box'])
        recovered=None;until=-1;recovery_ids=[]
        for i,(pts,image) in enumerate(sample_source(store.path(source['path']),a,b,16)):
            store.check_stop();scope=index.at(pts);clock=scope.get('clock') if scope else None
            valid,_=layout_matches(image,profile);evidence=scope.get('evidence_ids',[]) if scope else []
            if clock is None and valid and i%4==0:
                sha,path=evidence_crop(store,image.crop(field['box']));cached=root/'ocr'/sha/'result.json'
                if cached.exists():result=read_json(cached)
                else:
                    result=reader.read(image.crop(field['box']),field,cached.parent);atomic_json(cached,result)
                value=result.get('value');frame_id=uid('pass1-clock-frame',source['sha256'],pts)
                data['frames'].append({'id':frame_id,'time':pts,'path':path,'box':field['box'],
                    'source_sha256':source['sha256'],'presentation_timestamp':pts})
                row={'schema_version':2,'id':uid('pass1-clock-recovery',frame_id),'time':pts,
                    'task_id':uid('clock-refinement',a),'source_id':uid('source',source['sha256'],pts),
                    'frame_id':frame_id,'field':'game_clock','field_id':field['id'],'slot_id':'match',
                    'route':'match_hud','entity':'match','backend':'tesseract','box':field['box'],
                    'evidence_path':path,'value':value,'parsed_value':value,'raw_text':result.get('raw_text'),
                    'acceptance':result['acceptance'],'visibility':'visible' if value else 'unknown',
                    'review_status':'unreviewed','sampling_fps':4,'confidence':result.get('reader',{}).get('score',0),
                    'acceptance_reason':result.get('reason'),'reader':result.get('reader',{})}
                data['clock_readings'].append(row)
                recovered=value.get('seconds') if isinstance(value,dict) and result['acceptance']=='accepted' else None
                until=pts+.25;recovery_ids=[row['id']]
            if clock is None and recovered is not None and pts<until:clock=recovered;evidence=recovery_ids
            context=tracker.update(pts,clock,image,valid)
            frame_id=uid('pass1-hud-frame',source['sha256'],pts)
            common={'start':pts,'end':min(pts+1/16,b),**context,'evidence_ids':[frame_id]+evidence}
            data['playback'].append({'id':uid('pass1-canonical-playback',frame_id),**common})
            selection=selections.at(pts)
            if selection:
                effective={**selection,**common}
                for review in reviews:
                    if review['time']<=pts<review['end']:
                        # A reviewed contradiction suspends the model association;
                        # it cannot silently propagate a replacement identity.
                        effective.update(champion=None,identity_conflict=True,
                            owner_epoch=uid('reviewed-selection-boundary',review['record_id'],selection['owner_epoch']),
                            identity_review=review['review']['id'])
                data['selection'].append(effective)
    for key,rows in data.items():write_rows(root/(key+'.jsonl'),rows)
    report={'recipe':recipe,'windows':windows,'status':'completed_experimental','frames':len(data['playback']),
        'recovery_attempts':len(data['clock_readings']),
        'recovered':sum(r['acceptance']=='accepted' for r in data['clock_readings']),
        'wall_seconds':time.monotonic()-started,'experimental':True,'release_accepted':False}
    atomic_json(root/'report.json',report)
    return report
