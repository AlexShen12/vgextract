"""HP transcription, HUD activation alignment and cooldown-resistant icon views."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from .core import atomic_json, digest, finite, read_json, uid
from .media import file_hash
from .qwen35_common import TEXT, NULL_TEXT, POLICY, arr, obj, enum, rate
from .readers import parse_text, measure_bar, IconMatcher

HP_SCHEMA = obj(status=enum('visible','unreadable','absent'),text=NULL_TEXT,
                current={'type':['number','null']},maximum={'type':['number','null']})
HP_PROMPT = ('Transcribe ONLY numeric HP exactly as displayed. Return status, exact text, current and maximum. '
             'Use unreadable for ambiguous digits, absent only if visibly absent. Do not infer digits from the bar, '
             'previous readings, champion identity or game knowledge. Nonvisible values and text must be null.')
HUD_SCHEMA = obj(readings=arr(obj(frame_id=TEXT, slot=enum('Q','W','E','R'),
    state=enum('ready','cooling_down','charge','recast','unavailable','unknown'),
    seconds={'type':['number','null']}, charges={'type':['number','null']},
    charge_consumption_visible={'type':'boolean'},recast_visible={'type':'boolean'}),32,1))
ICON_SCHEMA = obj(slots=arr(obj(slot=enum('Q','W','E','R'),champion=NULL_TEXT,
                               clean={'type':'boolean'},evidence=TEXT),4,4))


def hp_value(value):
    if value['status']!='visible':
        if any(value[k] is not None for k in ('text','current','maximum')):
            raise ValueError('nonvisible_hp_has_values')
        return None
    parsed=parse_text(value['text'],'health')
    if parsed['current']!=value['current'] or parsed['maximum']!=value['maximum']:
        raise ValueError('hp_text_and_numbers_disagree')
    return parsed


def read_hp(requests, frames):
    rows=[]
    for frame in frames:
        if not frame.get('hp_sample'):continue
        attempts=[]
        for attempt in range(2):
            v,key=requests.call('hp-'+str(attempt),HP_PROMPT,[('hp',frame['hp_processed_path'])],HP_SCHEMA,
                                max_tokens=128,context={'retry':attempt},validate=hp_value)
            accepted=bool(v and v['status']=='visible')
            attempts.append({'request_id':key,'accepted':accepted,'status':v['status'] if v else 'invalid',
                             'text':v['text'] if v else None,'current':v['current'] if v else None,
                             'maximum':v['maximum'] if v else None})
            if v and v['status']!='unreadable':break
        first=attempts[0];final=attempts[-1]
        bar=frame.get('health_fraction')
        def checked(r):
            difference=abs(r['current']/r['maximum']-bar) if r['accepted'] and bar is not None else None
            return {**r,'accepted':r['accepted'] and (difference is None or difference<=.06),
                    'bar_difference':difference,'bar_checked':difference is not None}
        rows.append({'id':uid('q35-hp',requests.role,frame['id']),'frame_id':frame['id'],'time':frame['time'],
                     'role':requests.role,'attempts':attempts,'first':first,'retry_policy':final,
                     'first_bar':checked(first),'retry_policy_bar':checked(final)})
    return rows


def hp_metrics(references, predictions):
    ref={r['frame_id']:r for r in references};results={}
    indexed={r['frame_id']:r for r in predictions}
    for arm in ('first','retry_policy','first_bar','retry_policy_bar'):
        readable=accepted=correct=text_correct=covered=bad_unreadable=unreadable=0
        for key,r in ref.items():
            visible=r.get('status')=='visible';readable+=visible;unreadable+=not visible
            p=indexed.get(key,{}).get(arm,{'accepted':False})
            if not p['accepted']:continue
            accepted+=1;covered+=visible
            pair=visible and p.get('current')==r.get('current') and p.get('maximum')==r.get('maximum')
            correct+=pair;text_correct+=visible and p.get('text')==r.get('text')
            bad_unreadable+=not visible
        results[arm]={'references':len(ref),'readable':readable,'accepted':accepted,
                     'precision':rate(correct,accepted),'readable_coverage':rate(covered,readable),
                     'pair_accuracy':rate(correct,readable),'exact_text_accuracy':rate(text_correct,readable),
                     'incorrect_unreadable_acceptances':bad_unreadable,'unreadable_references':unreadable,
                     'incorrect_unreadable_rate':rate(bad_unreadable,unreadable),
                     'qualifies':bool(accepted and readable and correct/accepted>=POLICY['precision_min']
                                      and covered/readable>=POLICY['coverage_min'])}
    return results


def read_hud(requests, frames):
    output=[]
    for offset in range(0,len(frames),8):
        batch=frames[offset:offset+8];ids={f['id'] for f in batch}
        def validate(v):
            keys=[(r['frame_id'],r['slot']) for r in v['readings']]
            if len(keys)!=4*len(ids) or set(keys)!={(i,s) for i in ids for s in 'QWER'}:
                raise ValueError('missing_or_duplicate_hud_slot')
            for r in v['readings']:
                if r['state']=='cooling_down' and (not finite(r['seconds']) or r['seconds']<=0):
                    raise ValueError('cooldown_without_visible_seconds')
                if r['charges'] is not None and (not finite(r['charges']) or r['charges']<0 or int(r['charges'])!=r['charges']):
                    raise ValueError('invalid_charges')
        value,key=requests.call('ability-hud','Read Q/W/E/R availability and displayed cooldown numerals in source-time order. '
            'Do not confuse costs, hotkeys or ranks with cooldowns. A dark icon does not establish unavailable or cooldown. '
            'charge_consumption_visible and recast_visible require temporal visible evidence. Return every slot at every frame. '
            +str([{'frame_id':f['id'],'time':f['time'],'slot_boxes':f['hud_slot_boxes']} for f in batch]),
            [(f['id'],f['hud_path']) for f in batch],HUD_SCHEMA,max_tokens=3072,validate=validate)
        found={(r['frame_id'],r['slot']):r for r in (value or {}).get('readings',[])}
        for f in batch:
            for slot in 'QWER':
                row=found.get((f['id'],slot),{'state':'unknown','seconds':None,'charges':None,
                                          'charge_consumption_visible':False,'recast_visible':False})
                output.append({**row,'id':uid('q35-hud',f['id'],slot),'frame_id':f['id'], 'time':f['time'],
                               'slot_id':slot,'slot_kind':'ability','owner_id':'selected','owner_epoch':f['owner_epoch'],
                               'segment_id':f['segment_id'],'request_id':key})
    return output


def action_intervals(rows):
    result=[]
    for r in sorted(rows,key=lambda x:x['time']):
        if r['action']!='ability_action' or not r.get('track_id'):continue
        old=result[-1] if result else None
        if old and all(old[k]==r[k] for k in ('track_id','segment_id','owner_epoch')) and r['time']-old['end']<=.30:
            old['end']=r['time'];old['frame_ids'].append(r['frame_id'])
        else:
            result.append({'id':uid('q35-action-interval',r['id']),'start':r['time'],'end':r['time'],
                           'track_id':r['track_id'],'segment_id':r['segment_id'],'owner_epoch':r['owner_epoch'],
                           'frame_ids':[r['frame_id']]})
    return result


def align_abilities(transitions, scene_actions, detections):
    events=action_intervals(scene_actions);candidates={};owners={}
    for t in transitions:
        endpoint_tracks=[]
        for time in (t['start'],t['end']):
            nearby=[d for d in detections if d.get('main_player') and d['segment_id']==t['segment_id']
                    and d.get('owner_epoch')==t['owner_epoch'] and abs(d['time']-time)<=.30]
            if nearby:
                closest=min(abs(d['time']-time) for d in nearby)
                endpoint_tracks.append({d['track_id'] for d in nearby if abs(d['time']-time)<=closest+1e-6})
        tracks=set.intersection(*endpoint_tracks) if len(endpoint_tracks)==2 else set()
        owners[t['id']]=next(iter(tracks)) if len(tracks)==1 else None
        candidates[t['id']]=[e for e in events if t.get('activation_supported') and owners[t['id']]
                             and e['track_id']==owners[t['id']] and e['segment_id']==t['segment_id']
                             and e['owner_epoch']==t['owner_epoch']
                             and e['start']<=t['end']+POLICY['link_seconds']
                             and e['end']>=t['start']-POLICY['link_seconds']]
    result=[]
    for t in transitions:
        matches=candidates[t['id']]
        unique=len(matches)==1 and sum(any(e['id']==matches[0]['id'] for e in v) for v in candidates.values())==1
        result.append({**t,'id':uid('q35-alignment',t['id']), 'track_id':owners[t['id']],
                       'scene_action_id':matches[0]['id'] if unique else None,
                       'slot_id':t['slot_id'] if unique else None,
                       'candidate_slot':t['slot_id'],'support':'corroborated' if unique else 'unresolved',
                       'competing_scene_ids':[e['id'] for e in matches], 'hit':None})
    return result


class References:
    """Versioned, checksum-verified portrait/spell references; no mechanics lookup."""
    def __init__(self,path):
        import numpy as np
        from PIL import Image
        path=Path(path);manifest=read_json(path)
        self.version=manifest.get('reference_version') or manifest.get('patch',{}).get('ddragon_version')
        if not self.version:raise ValueError('Identity references need an explicit asset version')
        self.signature=digest(manifest);self.entries=[]
        for entry in manifest['assets']:
            if entry['kind'] not in ('champion','spell'):continue
            p=(path.parent/entry['path']).resolve()
            if not p.is_relative_to(path.parent.resolve()) or file_hash(p)!=entry['sha256']:
                raise ValueError('Identity reference path/hash mismatch')
            with Image.open(p) as im: image=im.convert('RGB').resize((32,32))
            self.entries.append({**entry,'pixels':np.asarray(image,dtype=float),'absolute_path':str(p)})
        self.matchers={slot:IconMatcher(path.parent,{'assets':[e for e in manifest['assets'] if e['kind']=='spell' and e['id'].endswith(':'+slot)]},
            {'icon':POLICY['icon_score'],'icon_margin':POLICY['icon_margin']}) for slot in 'QWER'}

    def match(self,image,kind,slot=None,mask=None):
        import numpy as np
        from PIL import Image
        query=np.asarray(image.convert('RGB').resize((32,32)),dtype=float)
        visible=np.ones((32,32),dtype=bool) if mask is None else np.asarray(Image.fromarray(mask.astype('uint8')*255).resize((32,32),Image.Resampling.NEAREST))==0
        if visible.mean()<.25 or query[visible].mean()<8:return None
        def vector(a):
            a=a[visible];a=a-a.mean(axis=0);flat=a.ravel()
            return flat/max(float(np.linalg.norm(flat)),1e-8)
        q=vector(query);scores=[]
        for e in self.entries:
            if e['kind']!=kind or slot and not e['id'].endswith(':'+slot):continue
            # Centered normalized color tolerates a global brightness factor.
            scores.append((float(np.dot(q,vector(e['pixels']))),e))
        scores.sort(key=lambda x:x[0],reverse=True)
        if len(scores)<2:return None
        score,e=scores[0];margin=score-scores[1][0]
        if score<POLICY['icon_score'] or margin<POLICY['icon_margin']:return None
        return {'id':e['id'],'score':score,'margin':margin,'reference_sha256':e['sha256'],
                'reference_version':self.version,'path':e['absolute_path']}


def overlay_views(image):
    """Heuristic digit/shadow masks: excluded pixels are never reconstructed."""
    import numpy as np
    from PIL import Image
    a=np.asarray(image.convert('RGB'),dtype=float);h,w=a.shape[:2]
    y,x=np.mgrid[:h,:w]
    center=(x>.12*w)&(x<.88*w)&(y>.18*h)&(y<.88*h)
    digits=(a.min(axis=2)>165)&(a.max(axis=2)-a.min(axis=2)<65)&center
    shadow=a.max(axis=2)<30
    excluded=digits|shadow
    # Global contrast adjustment acts only on observed pixels and is capped.
    brightness=float(a[~excluded].mean()) if (~excluded).any() else 0
    gain=min(2.5,100/max(brightness,1))
    normalized=np.clip(a*gain,0,255).astype('uint8')
    normalized[excluded]=[255,0,255]
    severity='heavy' if excluded.mean()>.5 else 'moderate' if excluded.mean()>.15 or brightness<60 else 'light'
    return Image.fromarray(normalized),excluded,{'excluded_fraction':float(excluded.mean()),'brightness':brightness,
        'gain':gain,'severity':severity,'clean_candidate':bool(excluded.mean()<.10 and brightness>70),
        'mask_is_heuristic':True,'reconstructed_pixels':False}


def consensus(slots):
    votes=Counter(v for v in slots.values() if v)
    return next(iter(votes)) if len(votes)==1 and next(iter(votes.values()))>=3 else None


def cooldown_frame(requests,frame,references,memory):
    from PIL import Image
    import numpy as np
    paths=[];normal_paths=[];masks=[];matches={};metadata={}
    for slot,path in frame['icon_paths'].items():
        with Image.open(path) as im:
            normalized,excluded,meta=overlay_views(im)
            target=requests.root/'icon-views'/f'{frame["id"]}-{slot}';target.mkdir(parents=True,exist_ok=True)
            normalized.save(target/'normalized.png');Image.fromarray(excluded.astype('uint8')*255).save(target/'excluded.png')
            atomic_json(target/'metadata.json',meta)
            old=references.matchers[slot].read(im,{'kind':'spell'}) if references else None
            if old and old.get('acceptance')=='accepted' and not old['value'].endswith(':'+slot):old=None
            masked=references.match(im,'spell',slot,excluded) if references else None
        matches[slot]={'baseline':old['value'].rsplit(':',1)[0] if old and old.get('acceptance')=='accepted' else None,
                       'masked':masked['id'].rsplit(':',1)[0] if masked else None,'reference':masked}
        metadata[slot]={**meta,'mask_path':str(target/'excluded.png'),'normalized_path':str(target/'normalized.png')}
        paths.append((slot,path));normal_paths.append((slot+' derived, magenta pixels excluded',target/'normalized.png'))
        masks.append((slot+' exclusion mask, white excluded',target/'excluded.png'))
    prompt=('Identify the champion kit from the ordered Q/W/E/R icons. For each slot return its champion or null, '
            'clean (no cooldown shading/numbers/unavailable ambiguity), and visible evidence. Unknown is allowed. '
            'Reference versions describe icons only; never infer a cast from identity.')
    def validate(v):
        if {r['slot'] for r in v['slots']}!=set('QWER'):raise ValueError('duplicate_or_missing_icons')
    raw,rawkey=requests.call('icons-raw',prompt,paths,ICON_SCHEMA,max_tokens=768,validate=validate)
    # Two slots per request keeps raw+mask+derived views under eight images.
    normalized=[];normalized_keys=[]
    for start in (0,2):
        group=[]
        for i in range(start,start+2):group.extend([paths[i],masks[i],normal_paths[i]])
        schema=obj(slots=arr(ICON_SCHEMA['properties']['slots']['items'],2,2))
        slots=set('QWER'[start:start+2])
        def validate_pair(v):
            if {r['slot'] for r in v['slots']}!=slots:raise ValueError('missing_icon_pair')
        value,key=requests.call('icons-normalized',prompt+' Only return supplied slots. Masks and normalized views are derived; '
            'do not use magenta or excluded pixels as observed artwork.',group,schema,max_tokens=512,validate=validate_pair)
        normalized.extend((value or {}).get('slots',[]));normalized_keys.append(key)
    raw_slots={r['slot']:r for r in (raw or {}).get('slots',[])}
    normalized_slots={r['slot']:r['champion'] for r in normalized}
    # Memory key includes split, owner epoch and playback segment; no spatial extrapolation.
    key=(frame['split'],frame['segment_id'],frame['owner_epoch'])
    temporal=memory.setdefault(key,{})
    current_raw={s:raw_slots.get(s,{}).get('champion') for s in 'QWER'}
    current_consensus=consensus(current_raw)
    previous_kit=temporal.get('_kit')
    supported_kit=current_consensus if current_consensus==consensus({s:matches[s]['masked'] for s in 'QWER'}) else None
    if supported_kit and previous_kit and supported_kit!=previous_kit:
        temporal.clear()
    if supported_kit:temporal['_kit']=supported_kit
    temporal_slots={};temporal_evidence={}
    for slot in 'QWER':
        r=raw_slots.get(slot,{})
        independent=matches[slot]['masked']
        if r.get('clean') and metadata[slot]['clean_candidate'] and r.get('champion') and r['champion']==independent:
            temporal[slot]={'champion':r['champion'],'frame_id':frame['id'],'path':frame['icon_paths'][slot]}
        if slot in temporal:
            temporal_slots[slot]=temporal[slot]['champion'];temporal_evidence[slot]=temporal[slot]
        else:temporal_slots[slot]=None
    arms={'matcher_raw':{s:matches[s]['baseline'] for s in 'QWER'},
          'qwen_raw':current_raw,'matcher_masked':{s:matches[s]['masked'] for s in 'QWER'},
          'qwen_normalized':normalized_slots,'temporal':temporal_slots}
    return {'id':uid('q35-cooldown',frame['id']),'frame_id':frame['id'],'time':frame['time'],
            'segment_id':frame['segment_id'],'owner_epoch':frame['owner_epoch'],
            'slots':arms,'champions':{arm:consensus(values) for arm,values in arms.items()},
            'views':metadata,'reference_matches':matches,'temporal_evidence':temporal_evidence,
            'request_ids':[rawkey]+normalized_keys,'natural':True}


def synthetic_icon_frame(frame,root):
    """Supplementary stress view, never substituted for natural source evidence."""
    from PIL import Image,ImageDraw,ImageEnhance
    paths={}
    for slot,path in frame['icon_paths'].items():
        with Image.open(path) as source:
            image=ImageEnhance.Brightness(source.convert('RGB')).enhance(.45)
        draw=ImageDraw.Draw(image);w,h=image.size
        draw.pieslice((0,0,w-1,h-1),0,150,fill=(8,8,8))
        draw.text((w*.4,h*.35),'5',fill='white')
        target=Path(root)/'synthetic-icons'/(frame['id']+'-'+slot+'.png')
        target.parent.mkdir(parents=True,exist_ok=True);image.save(target);paths[slot]=str(target)
    return {**frame,'id':uid('q35-synthetic',frame['id']),'icon_paths':paths,'source_frame_id':frame['id']}
