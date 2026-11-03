"""Observable champion boxes, match-scoped roster memory, and scene actions."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import unicodedata

from .core import digest, uid, read_json, write_rows
from .qwen35_common import (BOX, TEXT, NULL_TEXT, POLICY, arr, obj, enum, box_ok, iou, native_box)

CHAMPION = obj(local_id=TEXT, body_box=BOX, hp_box=BOX, name_box=BOX, name=NULL_TEXT,
               name_kind=enum('player', 'champion', 'unreadable'), bar_color=enum('green', 'red', 'blue', 'other'),
               champion_guess=NULL_TEXT, verified={'type': 'boolean'})
DETECTION = obj(champions=arr(CHAMPION, 10))
ACTION_KINDS = ('movement', 'basic_attack', 'ability_action', 'recall', 'idle', 'unknown')
ACTIONS = obj(frames=arr(obj(frame_id=TEXT, action=enum(*ACTION_KINDS), description=TEXT), 8, 1))
NAME = obj(name=NULL_TEXT, name_kind=enum('player', 'champion', 'unreadable'), champion_guess=NULL_TEXT,
           verified={'type': 'boolean'})
SCOREBOARD = obj(rows=arr(obj(row_id=TEXT, name=NULL_TEXT, champion=NULL_TEXT, portrait_confirmed={'type': 'boolean'}), 10))


def canonical(name):
    # Preserve script, punctuation, tags, and case: no fuzzy identity joins.
    return unicodedata.normalize('NFC', name).strip() if isinstance(name, str) else ''


def proposal_boxes(image, profile):
    """Color proposals, with level-cell and dark-frame tests; VLM must verify them."""
    import numpy as np
    from .action_body import spans
    a = np.asarray(image.convert('RGB'), dtype=float)
    red, green, blue = a.transpose(2, 0, 1)
    masks = {'green': (green > 100)&(green-red > 25)&(green-blue > 20),
             'red': (red > 110)&(red-green > 30)&(red-blue > 20),
             'blue': (blue > 100)&(blue-red > 25)&(blue-green > 10)}
    cfg = profile['body_binder']
    x0,y0,x1,y1 = cfg['search_box']; width,height = cfg['bar_size']
    found = []
    for color, mask in masks.items():
        mask[:y0]=False;mask[y1:]=False;mask[:,:x0]=False;mask[:,x1:]=False
        for x,y,r,b in cfg['exclude_boxes']:
            mask[y:b,x:r]=False
        for y in np.flatnonzero(mask.sum(axis=1) >= cfg['minimum_fill_pixels']):
            row = mask[y].copy(); row[1:-1] |= row[:-2]&row[2:]
            for left,right in spans(row):
                left,right,y = int(left),int(right),int(y)
                if not cfg['minimum_fill_pixels'] <= right-left <= width+3 or left+width >= x1 or y+height >= y1:
                    continue
                if any(abs(left-r['hp_box'][0]) < 8 and abs(y-r['hp_box'][1]) < 15 for r in found):
                    continue
                l,t,r,b = cfg['level_cell']; cell=a[max(0,y+t):y+b,max(0,left+l):left+r]
                border=a[max(0,y-3):max(1,y-1),left:left+width]
                if not cell.size or not border.size:
                    continue
                white=(cell.min(axis=2)>145)&(cell.max(axis=2)-cell.min(axis=2)<65)
                if (white.mean()<cfg['minimum_level_white'] or (cell.max(axis=2)<100).mean()<cfg['minimum_level_dark']
                        or (border.max(axis=2)<80).mean()<cfg['minimum_border_dark']):
                    continue
                body=[max(x0,left-55),y+15,min(x1,left+145),min(y1,y+180)]
                if not box_ok(body, image.size):
                    continue
                found.append({'body_box':body,'hp_box':[left,y,left+width,y+height],
                              'name_box':[max(0,left-50),max(0,y-35),min(x1,left+width+30),max(1,y-4)],
                              'bar_color':color})
    return found[:20]


def track_detections(rows):
    """Only unique mutual matches connect adjacent visible observations."""
    previous, output = [], []
    for row in sorted(rows, key=lambda r:r['time']):
        current=[dict(c, frame_id=row['frame_id'],time=row['time'],segment_id=row['segment_id'],owner_epoch=row.get('owner_epoch',0)) for c in row['champions'] if c['verified']]
        compatible={}
        for i,c in enumerate(current):
            choices=[]
            for j,p in enumerate(previous):
                if p['segment_id'] != c['segment_id'] or p['owner_epoch'] != c['owner_epoch'] or not 0 < c['time']-p['time'] <= POLICY['track_gap_seconds']:
                    continue
                a,b=c['hp_box'],p['hp_box']
                same_names=canonical(c.get('name')) and canonical(c.get('name')) == canonical(p.get('name'))
                contradictory=(c.get('name_kind') == p.get('name_kind') == 'player' and canonical(c.get('name'))
                               and canonical(p.get('name')) and not same_names)
                step=abs(a[0]-b[0])+abs(a[1]-b[1])
                if not contradictory and c['bar_color']==p['bar_color'] and (iou(a,b)>=.2 or step <= POLICY['track_step_pixels']):
                    choices.append(j)
            compatible[i]=choices
        for i,c in enumerate(current):
            choices=compatible[i]
            unique=len(choices)==1 and sum(choices[0] in v for v in compatible.values())==1
            c['track_id']=previous[choices[0]]['track_id'] if unique else uid('q35-track',row['frame_id'],c['local_id'])
            c['id']=uid('q35-detection',row['frame_id'],c['local_id'])
        greens=[c for c in current if c['bar_color']=='green' and c.get('green_bar_verified',False)]
        for c in current:
            c['main_player']=len(greens)==1 and c is greens[0]
        output.extend(current); previous=current
    return output


def resolve_identities(detections, scoreboard, *, two_pass=True):
    """Require repeated independent name/portrait joins, then resolve earlier tracks."""
    roster=defaultdict(set); score_evidence=defaultdict(list)
    # One row per name in a scoreboard observation; duplicates cannot establish a join.
    by_frame=defaultdict(list)
    for row in scoreboard:
        by_frame[row['frame_id']].append(row)
    for frame, rows in by_frame.items():
        counts=defaultdict(int)
        for r in rows:
            counts[canonical(r.get('name'))]+=1
        for r in rows:
            name=canonical(r.get('name'))
            if name and counts[name]==1 and r.get('portrait_confirmed') and r.get('champion') and r.get('reference_champion')==r['champion']:
                roster[name].add(r['champion']); score_evidence[name].append(r)
    seen=defaultdict(dict); registry={}
    for d in sorted(detections,key=lambda r:r['time']):
        name=canonical(d.get('name'))
        if d.get('name_kind')!='player' or not name or len(roster[name])!=1:
            continue
        champ=next(iter(roster[name]))
        # A visible contradiction blocks the observation, rather than overwriting the roster.
        if d.get('champion_guess') and d['champion_guess']!=champ:
            continue
        if not two_pass and not any(s['time']<=d['time'] for s in score_evidence[name]):
            continue
        seen[name][d['frame_id']]=d
        if len(seen[name])>=2:
            first=max(sorted(v['time'] for v in seen[name].values())[1], min(s['time'] for s in score_evidence[name]))
            registry[name]={'id':uid('q35-roster',name,champ),'name':name,'champion':champ,'established_at':first,
                            'evidence_ids':sorted([v['id'] for v in seen[name].values()]+[s['id'] for s in score_evidence[name]])}
    # Any competing supported roster entries poison that name for the whole pass.
    conflicts=[{'name':n,'champions':sorted(v)} for n,v in roster.items() if len(v)>1]
    track_names=defaultdict(set)
    for d in detections:
        if d.get('name_kind')=='player' and canonical(d.get('name')) in registry:
            track_names[d['track_id']].add(canonical(d['name']))
    resolved=[]
    for d in detections:
        name=canonical(d.get('name')) if d.get('name_kind')=='player' else ''
        binding=registry.get(name)
        if not binding and len(track_names[d['track_id']])==1:
            binding=registry[next(iter(track_names[d['track_id']]))]
        if binding and (d.get('champion_guess') and d['champion_guess']!=binding['champion']):
            binding=None
        if binding and not two_pass and binding['established_at']>d['time']:
            binding=None
        resolved.append({**d,'champion':binding['champion'] if binding else None,
                         'binding_id':binding['id'] if binding else None,
                         'identity_source':'name_scoreboard_registry' if binding else 'unresolved',
                         'retrospective':bool(binding and binding['established_at']>d['time'])})
    return resolved,list(registry.values()),conflicts


def detect_frame(requests, frame, profile, arm):
    from PIL import Image, ImageDraw
    candidates=[]
    with Image.open(frame['native_path']) as image:
        if arm=='hybrid':
            candidates=proposal_boxes(image, profile)
    context={'frame_id':frame['id'],'scene_box':profile['scene_box'],'displayed_size':frame['scene_size'],
             'proposals_native':candidates if arm=='hybrid' else None}
    prompt=('Locate all visible champion bodies, overhead HP bars, and nameplates. Exclude minions and HUD panels. '
            'Return boxes in PIXELS OF THE SUPPLIED SCENE IMAGE, not native coordinates. '
            'Name is exact visible text or null; classify player versus champion name or unreadable. '
            'Champion identity is only a visual hypothesis; never invent a name from appearance. '
            'verified requires a visible champion and associated bar. '+str(context))
    def validate(value):
        ids=set()
        for c in value['champions']:
            if c['local_id'] in ids:
                raise ValueError('duplicate_local_id')
            ids.add(c['local_id'])
            for key in ('body_box','hp_box','name_box'):
                if not box_ok(c[key],frame['scene_size']):
                    raise ValueError('box_outside_scene')
    response,key=requests.call('detect-'+arm,prompt,[('scene',frame['scene_path'])],DETECTION,
                               max_tokens=2048,context=context,validate=validate)
    champions=[]
    for c in (response or {}).get('champions',[]):
        c=dict(c)
        for field in ('body_box','hp_box','name_box'):
            c[field]=native_box(c[field],profile['scene_box'],frame['scene_size'])
        # Names are re-read at native resolution, not silently trusted after scene downsampling.
        with Image.open(frame['native_path']) as source:
            box=[min(c['body_box'][0],c['name_box'][0],c['hp_box'][0]),min(c['name_box'][1],c['hp_box'][1]),
                 max(c['body_box'][2],c['name_box'][2],c['hp_box'][2]),max(c['body_box'][3],c['hp_box'][3])]
            path=requests.root/'actor-crops'/f'{digest([frame["id"],box])}.png';path.parent.mkdir(exist_ok=True)
            source.crop(box).save(path)
            import numpy as np
            bar=np.asarray(source.crop(c['hp_box']).convert('RGB'),dtype=float)
            green=(bar[...,1]>100)&(bar[...,1]-bar[...,0]>25)&(bar[...,1]-bar[...,2]>20)
            c['green_bar_verified']=bool(green.sum()>=5 and green.mean()>=.03)
        verify,vkey=requests.call('verify-actor','Verify this champion and its overhead name/bar. Transcribe the name exactly, '
            'classify name_kind, and optionally give a champion visual hypothesis. Set verified false for non-champions.',
            [('native_actor',path)],NAME,max_tokens=256)
        if verify:
            c.update(verify)
        else:
            c['verified']=False
        c['evidence_ids']=[key,vkey];c['actor_crop']=str(path)
        champions.append(c)
    overlay=requests.root/'overlays'/f'{frame["id"]}-{arm}.png';overlay.parent.mkdir(exist_ok=True)
    with Image.open(frame['native_path']) as source:
        draw=ImageDraw.Draw(source)
        for i,c in enumerate(champions):
            for field,color in [('body_box','yellow'),('hp_box','lime'),('name_box','cyan')]:
                draw.rectangle(c[field],outline=color,width=2)
            draw.text((c['body_box'][0],c['body_box'][1]),str(i+1),fill='yellow')
        source.save(overlay)
    return {'frame_id':frame['id'],'time':frame['time'],'segment_id':frame['segment_id'],'owner_epoch':frame['owner_epoch'],
            'champions':champions,'overlay_path':str(overlay),'request_id':key}


def scoreboard_rows(requests, frame, profile, references):
    """Read only visible scoreboard panels; portrait matching is independent of Qwen."""
    from PIL import Image
    from .hybrid_intake import _panel_state
    from .intake_profile import load_profile
    intake=load_profile(profile['anchor_profile'])
    with Image.open(frame['native_path']) as image:
        if _panel_state(image,intake,{'route':'scoreboard'},frame['layout_valid'])!='present':
            return []
        root=requests.root/'scoreboard-inputs';root.mkdir(exist_ok=True)
        images=[];independent={}
        for f in intake['fields']:
            if f['field']!='player_name':continue
            x,y,r,b=f['box']; name_box=[x,y,r,b]
            # Portrait offset is explicitly stored in the experiment profile.
            side=f['slot_id'].split(':')[0]
            portrait_x=profile['scoreboard_portrait_x'][side]
            portrait=[portrait_x,y+8,portrait_x+48,y+56]
            path=root/f'{frame["id"]}-{f["slot_id"].replace(":","-")}.png'
            image.crop([min(x,portrait_x),y,r,y+60]).save(path)
            independent[f['slot_id']]=references.match(image.crop(portrait),'champion') if references else None
            images.append((f['slot_id'],path))
    rows=[]
    for offset in range(0,len(images),5):
        group=images[offset:offset+5]
        candidates={k:independent[k] for k,_ in group}
        result,key=requests.call('scoreboard','Read each supplied scoreboard row: exact player name and champion portrait. '
            'Confirm or reject the separately proposed portrait identity. row_id must equal an image label. '
            'Candidate references: '+str(candidates),group,SCOREBOARD,max_tokens=1024,context=candidates)
        known={k for k,_ in group}
        found=(result or {}).get('rows',[])
        if len({r['row_id'] for r in found})!=len(found) or any(r['row_id'] not in known for r in found):
            continue
        for row in found:
            ref=independent[row['row_id']]
            rows.append({**row,'id':uid('q35-scoreboard',frame['id'],row['row_id']), 'frame_id':frame['id'],
                         'time':frame['time'],'request_id':key,'reference_champion':ref['id'] if ref else None,
                         'reference_evidence':ref})
    return rows


def describe_frames(requests, frames, detections, *, assisted=False, hints=()):
    by_frame=defaultdict(list)
    for d in detections:
        if d.get('main_player'):by_frame[d['frame_id']].append(d)
    output=[]
    # Nonoverlapping packets: exactly one output row per sampled frame.
    batches=[]; pending=[]
    for frame in frames:
        if pending and (frame['segment_id']!=pending[-1]['segment_id'] or frame['owner_epoch']!=pending[-1]['owner_epoch']
                        or frame['time']-pending[-1]['time']>.30 or len(pending)==4):
            batches.append(pending);pending=[]
        pending.append(frame)
    if pending:batches.append(pending)
    for batch in batches:
        context=[];images=[]
        for f in batch:
            main=by_frame[f['id']]
            context.append({'frame_id':f['id'],'time':f['time'], 'main_player_box':main[0]['body_box'] if len(main)==1 else None,
                            'champion':main[0].get('champion') if len(main)==1 else None})
            images.append((f['id'],f['scene_path']))
            if assisted:images.append((f['id']+' same-frame ability HUD',f['hud_path']))
        prompt=('Describe the green-bar main player separately at every supplied frame, using neighboring frames for motion. '
                'Use native main_player_box mapped through scene_box supplied below. If no unique visible main player, output unknown. '
                'Describe only visible movement, basic attack, ability action, recall, idle or uncertainty. '
                'Do not claim hits, damage, kills or intent from an activation. '+str({'frames':context,'scene_box_native':batch[0]['scene_box'],'displayed_scene_size':batch[0]['scene_size']}))
        if assisted:prompt+=' Separately observed fresh HUD activations: '+str(list(hints))
        ids={f['id'] for f in batch}
        def validate(v):
            if len(v['frames'])!=len(ids) or {r['frame_id'] for r in v['frames']}!=ids:
                raise ValueError('missing_or_duplicate_frame_description')
        value,key=requests.call('actions-assisted' if assisted else 'actions-scene-only',prompt,images,ACTIONS,
                               max_tokens=1024,context={'frames':context,'hints':list(hints) if assisted else [],
                                                      'scene_box':batch[0]['scene_box']},validate=validate)
        values={r['frame_id']:r for r in (value or {}).get('frames',[])}
        for f in batch:
            d=by_frame[f['id']]
            r=values.get(f['id'],{'action':'unknown','description':'No valid scene response.'})
            if len(d)!=1:r={'action':'unknown','description':'Main player is not uniquely visible.'}
            output.append({**r,'id':uid('q35-frame-action',f['id'],assisted),'frame_id':f['id'],
                           'time':f['time'],'segment_id':f['segment_id'], 'track_id':d[0]['track_id'] if len(d)==1 else None,
                           'owner_epoch':f['owner_epoch'], 'request_id':key,'assisted':assisted})
    return output
