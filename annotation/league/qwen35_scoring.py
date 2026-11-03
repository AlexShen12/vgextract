"""Source-label validation and separate scorecards; predictions never certify labels."""
from __future__ import annotations

from collections import Counter,defaultdict
import json

from .core import atomic_json, finite, uid
from .qwen35_common import SUITES, POLICY, box_ok, iou, rate
from .qwen35_sensors import hp_metrics

CATEGORIES={'hp-vlm':('hp',),'scene-identity':('detections','scoreboard','actions'),
            'ability-alignment':('actions','activations'), 'cooldown-identity':('icons',)}


def validate_labels(study,split,suites):
    from .qwen35_study import REQUIRED_CASES
    from .qwen35_identity import ACTION_KINDS
    clips=[c for c in study['clips'] if c['split']==split]
    if len(clips)!=10 or len(study['clips'])!=20:raise ValueError('A frozen study requires ten development and ten held-out neighborhoods')
    if not study.get('independent_of_predictions') or not study.get('labeler'):
        raise ValueError('Independent source-label provenance is required')
    categories=set(k for suite in suites for k in CATEGORIES[suite])
    cases=set()
    for c in clips:
        if not c.get('verified') or not c.get('independent_of_predictions') or not c.get('labeler'):
            raise ValueError('Unverified or prediction-assisted clip: '+c['id'])
        cases.update(c['case_tags'])
        for category in categories:
            rows=c.get(category)
            if not isinstance(rows,list):raise ValueError('Missing reference category '+category)
            if not rows and category not in c.get('negative_categories',[]):
                raise ValueError('Empty category needs an explicit inspected negative: '+category)
            seen=set()
            for r in rows:
                if not finite(r.get('time')) or not c['start']<=r['time']<c['end']:
                    raise ValueError('Reference timestamp outside clip')
                if r.get('frame_id')!=uid('q35-frame',study['source_sha256'],r['time']):
                    raise ValueError('Reference frame must use the source-only evidence frame ID')
                key=(r['frame_id'],r.get('reference_track'),r.get('slot_id'),r.get('row_id'))
                if key in seen and category not in ('activations',):raise ValueError('Duplicate reference record')
                seen.add(key)
                if category=='hp':
                    from .qwen35_sensors import hp_value
                    if r.get('status') not in ('visible','unreadable','absent'):raise ValueError('Invalid HP status')
                    hp_value(r)
                elif category=='detections':
                    if not all(box_ok(r.get(k),[1920,1080]) for k in ('body_box','hp_box','name_box')):
                        raise ValueError('Reference boxes must be native source pixels')
                    if not r.get('reference_track') or type(r.get('main_player')) is not bool:raise ValueError('Detection needs track and explicit main-player flag')
                elif category=='actions':
                    if r.get('action') not in ACTION_KINDS:raise ValueError('Invalid reference action')
                elif category=='activations':
                    if r.get('slot_id') not in 'QWER' or not finite(r.get('start')) or not finite(r.get('end')) or not c['start']<=r['start']<=r['end']<=c['end']:
                        raise ValueError('Invalid activation reference')
                elif category=='icons':
                    if not isinstance(r.get('champion'),str) or not r['champion']:raise ValueError('Icon reference needs champion')
                    if r.get('overlay') not in ('clean','digits','shading','unavailable','mixed'):raise ValueError('Icon reference needs overlay class')
                elif category=='scoreboard':
                    if not r.get('row_id') or not r.get('name') or not r.get('champion'):raise ValueError('Incomplete scoreboard reference')
    for suite in suites:
        missing=REQUIRED_CASES[suite]-cases
        if missing:raise ValueError('Missing '+suite+' reference cases: '+', '.join(sorted(missing)))


def box_metrics(refs,predictions):
    matches=[];used=set()
    for r in refs:
        choices=[(iou(r['body_box'],p['body_box']),i,p) for i,p in enumerate(predictions)
                 if i not in used and p['frame_id']==r['frame_id']]
        choices.sort(key=lambda x:x[0],reverse=True)
        if choices and choices[0][0]>=.5:
            _,i,p=choices[0];used.add(i);matches.append((r,p))
    n=len(matches);accepted=[(r,p) for r,p in matches if p.get('champion')]
    names=[(r,p) for r,p in matches if r.get('name')]
    name_correct=sum(r['name']==p.get('name') for r,p in names)
    identity_correct=sum(r.get('champion')==p['champion'] for r,p in accepted)
    known=sum(bool(r.get('champion')) for r in refs)
    coverage=sum(bool(r.get('champion') and p.get('champion')) for r,p in matches)
    tracks=defaultdict(list)
    for r,p in matches:tracks[r['reference_track']].append((r,p))
    switches=pairs=0;reappear=correct_reappear=0
    for rows in tracks.values():
        rows.sort(key=lambda rp:rp[0]['time'])
        for (a,p),(b,q) in zip(rows,rows[1:]):
            if b['time']-a['time']<=.3:
                pairs+=1;switches+=p['track_id']!=q['track_id']
            else:
                reappear+=1;correct_reappear+=bool(b.get('champion') and b['champion']==q.get('champion'))
    return {'references':len(refs),'predictions':len(predictions),'body_precision_iou_0_5':rate(n,len(predictions)),
            'body_recall_iou_0_5':rate(n,len(refs)),
            'bar_correct_given_body':rate(sum(iou(r['hp_box'],p['hp_box'])>=.5 for r,p in matches),n),
            'name_box_correct_given_body':rate(sum(iou(r['name_box'],p['name_box'])>=.5 for r,p in matches),n),
            'name_exact_accuracy':rate(name_correct,len(names)), 'identity_precision':rate(identity_correct,len(accepted)),
            'identity_coverage':rate(coverage,known),'main_player_accuracy':rate(sum(r['main_player']==p['main_player'] for r,p in matches),n),
            'false_main_players':sum(p.get('main_player',False) for i,p in enumerate(predictions) if i not in used)
                                +sum(p['main_player'] and not r['main_player'] for r,p in matches),
            'track_switches':switches,'adjacent_track_pairs':pairs,'reidentification_accuracy':rate(correct_reappear,reappear),
            'reappearance_pairs':reappear}


def action_metrics(refs,predictions):
    lookup={p['frame_id']:p for p in predictions};correct=0;confusion=Counter()
    for r in refs:
        p=lookup.get(r['frame_id'],{'action':'unknown'});correct+=p['action']==r['action']
        confusion[r['action']+' -> '+p['action']]+=1
    return {'references':len(refs),'accuracy':rate(correct,len(refs)),'confusion':dict(confusion),
            'unknown_fraction':rate(sum(p['action']=='unknown' for p in predictions),len(predictions))}


def activation_metrics(refs,predictions,*,combined=False):
    # Events are matched once, to the same slot and clip, within 250 ms.
    used=set();pairs=[]
    eligible=[p for p in predictions if p.get('support')=='corroborated'] if combined else [p for p in predictions if p.get('activation_supported')]
    for r in refs:
        choices=[(abs(p['start']-r['start']),i,p) for i,p in enumerate(eligible) if i not in used
                 and p.get('clip_id')==r.get('clip_id') and p.get('slot_id')==r['slot_id']
                 and p['start']<=r['end']+.25 and p['end']>=r['start']-.25]
        if choices:
            _,i,p=min(choices,key=lambda x:x[0]);used.add(i);pairs.append((r,p))
    errors=[abs(r['start']-p['start']) for r,p in pairs]
    timed_used=set();timed_pairs=[]
    for reference in refs:
        choices=[(abs(p['start']-reference['start']),i,p) for i,p in enumerate(eligible)
                 if i not in timed_used and p.get('clip_id')==reference.get('clip_id')
                 and p['start']<=reference['end']+.25 and p['end']>=reference['start']-.25]
        if choices:
            _,index,prediction=min(choices,key=lambda x:x[0]);timed_used.add(index);timed_pairs.append((reference,prediction))
    ordering=[(a,b) for a,b in zip(sorted(pairs,key=lambda rp:rp[0]['start']),sorted(pairs,key=lambda rp:rp[0]['start'])[1:])
              if a[0].get('clip_id')==b[0].get('clip_id') and a[0]['end']<b[0]['start']]
    return {'references':len(refs),'predictions':len(eligible),'precision':rate(len(pairs),len(eligible)),
            'recall':rate(len(pairs),len(refs)),'mean_start_error_seconds':rate(sum(errors),len(errors)),
            'matched_slot_accuracy':rate(sum(r['slot_id']==p.get('slot_id') for r,p in timed_pairs),len(timed_pairs)),
            'rapid_order_accuracy':rate(sum(a[1]['end']<b[1]['start'] for a,b in ordering),len(ordering)),
            'ordered_pairs':len(ordering)}


def icon_metrics(refs,predictions):
    indexed={p['frame_id']:p for p in predictions if p.get('natural')};arms={}
    for arm in ('matcher_raw','qwen_raw','matcher_masked','qwen_normalized','temporal'):
        groups=defaultdict(lambda:[0,0,0,0,0])
        previous={};false_changes=0
        for r in sorted(refs,key=lambda x:x['time']):
            p=indexed.get(r['frame_id'],{});champ=p.get('champions',{}).get(arm)
            slots=p.get('slots',{}).get(arm,{})
            for key in ('all',r['overlay']):
                g=groups[key];g[0]+=1;g[1]+=bool(champ);g[2]+=bool(champ==r['champion']);g[3]+=sum(bool(v) for v in slots.values())
                g[4]+=sum(v==r['champion'] for v in slots.values())
            context=(r.get('clip_id'),p.get('segment_id'),p.get('owner_epoch'))
            old=previous.get(context)
            if champ and old and champ!=old[1] and r['champion']==old[0]:false_changes+=1
            if champ:previous[context]=(r['champion'],champ)
        arms[arm]={'by_overlay':{k:{'references':v[0],'accepted':v[1],'identity_precision':rate(v[2],v[1]),
                     'coverage':rate(v[1],v[0]),'slot_precision':rate(v[4],v[3]),'slot_coverage':rate(v[3],4*v[0])} for k,v in groups.items()},
                   'false_champion_changes':false_changes}
    return arms


def resources(records):
    successful=[r for r in records if r['status']=='ok'];new=[r for r in records if not r['cache_reused']]
    total=sum(r.get('wall_seconds',0) for r in new)
    return {'requests':len(records),'valid_requests':len(successful),'failed_requests':len(records)-len(successful),
            'cache_reused':sum(r['cache_reused'] for r in records),
            'generated_tokens':sum(r.get('usage',{}).get('completion_tokens',0) for r in new),
            'prompt_tokens':sum(r.get('usage',{}).get('prompt_tokens',0) for r in new),
            'wall_seconds_new_requests':total,'valid_new_requests_per_second':rate(sum(r['status']=='ok' for r in new),total),
            'mean_valid_request_latency_seconds':rate(sum(r.get('wall_seconds',0) for r in successful),len(successful)),
            'peak_gpu_memory_bytes':max([0]+[r.get('peak_memory_bytes',0) for r in records])}


def score_study(study,selected,frames,predictions,request_records,*,draft):
    from .qwen35_study import REQUIRED_CASES
    result={'quality_metrics':{},'resources':{},'missing_cases':{},'acceptance':False}
    for suite,rows in predictions.items():
        result['quality_metrics'][suite]={}
        for split in ('development','held_out'):
            clips=[c for c in selected if c['split']==split]
            if not clips:continue
            missing=REQUIRED_CASES[suite]-set(t for c in clips for t in c['case_tags'])
            result['missing_cases'][suite+'/'+split]=sorted(missing)
            if draft:
                result['quality_metrics'][suite][split]={'status':'unverified_draft','metrics':None}
                continue
            refs={category:[dict(r,clip_id=c['id']) for c in clips for r in c.get(category,[])] for category in CATEGORIES[suite]}
            out={};subset=[r for r in rows if r['split']==split]
            if suite=='hp-vlm':
                for role in sorted({r['role'] for r in subset}):out[role]=hp_metrics(refs['hp'],[r for r in subset if r['role']==role])
                qualifying=[role for role,m in out.items() if m['retry_policy_bar']['qualifies']]
                speeds={role:resources([r for r in request_records if r['role']==role and r['task'].startswith('hp-')]) for role in out}
                out['recommendation']=min(qualifying,key=lambda r:speeds[r]['mean_valid_request_latency_seconds'] or float('inf')) if qualifying else 'no qualifying candidate'
                out['resources']=speeds
            elif suite=='scene-identity':
                for arm in ('direct','hybrid'):
                    p=[r for r in subset if r['arm']==arm]
                    scoreboard=[r for r in p if r['table']=='scoreboard'];ref_rows={(r['frame_id'],r['row_id']):r for r in refs['scoreboard']}
                    accepted=[r for r in scoreboard if r.get('champion') and r.get('name') and r.get('portrait_confirmed')]
                    out[arm]={'before_memory':box_metrics(refs['detections'],[r for r in p if r['table']=='before_memory']),
                              'after_memory':box_metrics(refs['detections'],[r for r in p if r['table']=='detections']),
                              'actions':action_metrics(refs['actions'],[r for r in p if r['table']=='actions']),
                              'scoreboard_join_precision':rate(sum((r['frame_id'],r['row_id']) in ref_rows and
                                 all(r[k]==ref_rows[(r['frame_id'],r['row_id'])][k] for k in ('name','champion')) for r in accepted),len(accepted))}
            elif suite=='ability-alignment':
                out={'scene_only':action_metrics(refs['actions'],[r for r in subset if r['table']=='actions']),
                     'hud_only':activation_metrics(refs['activations'],[r for r in subset if r['table']=='activations']),
                     'combined':activation_metrics(refs['activations'],[r for r in subset if r['table']=='associations'],combined=True),
                     'assisted_scene':action_metrics(refs['actions'],[r for r in subset if r['table']=='assisted'])}
            else:
                out=icon_metrics(refs['icons'],subset)
                synthetic=[{**p,'frame_id':p['source_frame_id'],'natural':True} for p in subset if not p.get('natural')]
                out['synthetic_stress_only']=icon_metrics([r for r in refs['icons'] if r['overlay']=='clean'],synthetic)
            result['quality_metrics'][suite][split]=out
        result['resources'][suite]=resources([r for r in request_records if r.get('suite')==suite])
    result['hp_crop_counts']={s:sum(f.get('hp_sample',False) for c in selected if c['split']==s for f in frames[c['id']]) for s in ('development','held_out')}
    result['hp_target']=240;result['recognition_acceptance']='not established; pilot evidence only'
    return result


def write_report(root,report):
    atomic_json(root/'report.json',report)
    lines=['# Qwen3.5 experiment report','',f"Status: {report['status']}",'',
           'Experimental outputs; no production evidence modified.','',
           'Recognition acceptance: not established. Draft runs have no quality scores.','']
    for key in ('results','quality_metrics','missing_cases','resources','error'):
        if key in report:
            lines += ['## '+key.replace('_',' ').title(),'','```json',json.dumps(report[key],ensure_ascii=False,indent=2),'```','']
    examples=list(root.glob('overlays/*.png'))[:3]+list(root.glob('icon-views/*/normalized.png'))[:3]
    if examples:
        lines += ['## Evidence examples','']
        lines += ['!['+p.name+']('+str(p.resolve())+')' for p in examples]
    (root/'report.md').write_text('\n'.join(lines))
