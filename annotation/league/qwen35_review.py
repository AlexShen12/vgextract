"""Qwen3.5 evidence viewer, source-only label editor, and isolated corrections."""
from pathlib import Path
import fcntl
import re

from .core import atomic_json,digest,read_json,read_rows,now
from .qwen35_common import box_ok


def studies(store):
    roots={store.path('benchmarks/qwen35').resolve()}
    for record in store.path('benchmarks/qwen35-studies').glob('*.json'):
        roots.add(Path(read_json(record)['path']).resolve())
    for p in store.path('experiments/qwen35').glob('*/label-directory.json'):
        roots.add(Path(read_json(p)['path']).resolve())
    return {digest(str(p)):p for p in roots if (p/'study.json').exists()}


def root_for(store,key):
    if not re.fullmatch(r'[a-f0-9]{64}',key):raise ValueError('Invalid experiment ID')
    p=store.path('experiments/qwen35')/key
    if not (p/'recipe.json').is_file():raise ValueError('Experiment not found')
    return p


def corrected(directory,table,arm,rows):
    rows=[dict(r) for r in rows]
    for path in sorted((directory/'corrections').glob('*.json')):
        correction=read_json(path)
        if correction['table']!=table or correction['arm']!=arm:continue
        for row in rows:
            key=row.get('id') or row.get('frame_id')
            if correction['record_id']==key:
                row.update(correction['changes'],review_status='human-corrected')
    return rows


def install_qwen35_routes(app,store):
    from fastapi import HTTPException,Request
    from fastapi.responses import FileResponse,HTMLResponse
    globals()['Request']=Request

    def local(request):
        origin=request.headers.get('origin')
        if origin and origin.rstrip('/')!=str(request.base_url).rstrip('/'):raise HTTPException(403,'Use the local review app')

    @app.get('/qwen35',response_class=HTMLResponse)
    @app.get('/qwen35-labels',response_class=HTMLResponse)
    def page():return Path(__file__).with_name('qwen35_review.html').read_text()

    @app.get('/api/qwen35/studies')
    def list_studies():
        return [{'id':key,'path':str(path)} for key,path in studies(store).items()]

    @app.get('/api/qwen35/labels')
    def labels(study:str):
        directory=studies(store).get(study)
        if not directory:raise HTTPException(404,'Study not found')
        data=read_json(directory/'study.json')
        # This endpoint never reads prediction files.
        audits=[read_json(p) for p in (directory/'source').glob('*/audit.json')]
        data['source_frames']=[f for a in audits if a['recipe']['fps']==4 for f in a['frames']]
        return data

    @app.post('/api/qwen35/labels/{study}/{clip_id}')
    async def save_labels(study:str,clip_id:str,request:Request):
        local(request);directory=studies(store).get(study)
        if not directory:raise HTTPException(404,'Study not found')
        body=await request.json()
        allowed={'case_tags','verified','independent_of_predictions','labeler','hp','detections','scoreboard','actions','activations','icons','negative_categories'}
        if not isinstance(body,dict) or set(body)-allowed:raise HTTPException(400,'Only annotation fields may be edited')
        with (directory/'.labels.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            data=read_json(directory/'study.json');clip=next((c for c in data['clips'] if c['id']==clip_id),None)
            if clip is None:raise HTTPException(404,'Clip not found')
            candidate={**clip,**body}
            if type(candidate.get('verified')) is not bool or type(candidate.get('independent_of_predictions')) is not bool:
                raise HTTPException(400,'Verification fields must be booleans')
            for category in ('hp','detections','scoreboard','actions','activations','icons'):
                if not isinstance(candidate.get(category),list):raise HTTPException(400,'Annotation categories must be arrays')
                for r in candidate[category]:
                    if not isinstance(r,dict) or not isinstance(r.get('time'),(int,float)) or not clip['start']<=r['time']<clip['end']:
                        raise HTTPException(400,'Reference time outside clip')
                    from .core import uid
                    if r.get('frame_id')!=uid('q35-frame',data['source_sha256'],r['time']):
                        raise HTTPException(400,'Reference frame ID does not match source timestamp')
            if candidate['verified'] and (not candidate['labeler'] or not candidate['independent_of_predictions']):
                raise HTTPException(400,'Verified labels need source-only provenance')
            history=directory/'label-history';history.mkdir(exist_ok=True)
            atomic_json(history/(digest([now(),data])+'.json'),data)
            clip.update(body)
            # Global attestation is explicit, not inferred from generated predictions.
            if body.get('independent_of_predictions') and body.get('labeler'):
                data['independent_of_predictions']=True;data['labeler']=body['labeler']
            atomic_json(directory/'study.json',data)
        return {'saved':clip_id,'freeze_may_be_stale':True}

    @app.get('/api/qwen35/experiments')
    def experiments():
        return [{'id':p.parent.name,'report':read_json(p)} for p in store.path('experiments/qwen35').glob('*/report.json')]

    @app.get('/api/qwen35/records')
    def records(experiment:str,suite:str,table:str='',offset:int=0):
        try:root=root_for(store,experiment)
        except ValueError as exc:raise HTTPException(404,str(exc))
        from .qwen35_common import SUITES
        if suite not in SUITES:raise HTTPException(400,'Invalid suite')
        rows=read_rows(root/suite/'predictions.jsonl')
        if table:rows=[r for r in rows if r.get('table','readings')==table]
        offset=max(0,offset)
        return {'total':len(rows),'records':rows[offset:offset+100]}

    @app.get('/api/qwen35/evidence')
    def evidence(experiment:str,request_id:str):
        try:root=root_for(store,experiment)
        except ValueError as exc:raise HTTPException(404,str(exc))
        if not re.fullmatch(r'[a-f0-9]{64}',request_id):raise HTTPException(400,'Invalid request ID')
        p=root/'requests'/request_id/'input.json'
        if not p.exists():raise HTTPException(404,'Request not found')
        return read_json(p)

    @app.post('/api/qwen35/corrections/{experiment}')
    async def correction(experiment:str,request:Request):
        local(request)
        try:root=root_for(store,experiment)
        except ValueError as exc:raise HTTPException(404,str(exc))
        directory=Path(read_json(root/'label-directory.json')['path'])
        body=await request.json()
        table=body.get('table');arm=body.get('arm');changes=body.get('changes',{})
        permitted={'detections':{'name','name_kind','champion_guess','verified','body_box','hp_box','name_box','bar_color'},
                   'scoreboard':{'name','champion','portrait_confirmed'}}
        if table not in permitted or arm not in ('direct','hybrid') or not isinstance(changes,dict) or set(changes)-permitted[table]:
            raise HTTPException(400,'Invalid correction fields')
        indexed={r.get('id'):r for suite in ('scene-identity','ability-alignment')
                 for r in read_rows(root/suite/'predictions.jsonl') if r.get('table')==table and r.get('arm')==arm}
        if body.get('record_id') not in indexed:raise HTTPException(404,'Record not found')
        for key,value in changes.items():
            if key.endswith('_box') and not box_ok(value,[1920,1080]):raise HTTPException(400,'Invalid native box')
        dest=directory/'corrections';dest.mkdir(exist_ok=True)
        with (dest/'.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            revision=len(list(dest.glob('*.json')))+1
            value={'revision':revision,'table':table,'arm':arm,'record_id':body['record_id'],'changes':changes,'created_at':now()}
            atomic_json(dest/f'{revision:08d}.json',value)
        return {'saved':True,'rerun_required':True,'invalidated':['tracks','registry','descriptions','ability_associations','scores']}

    @app.get('/api/qwen35/media')
    def media(path:str):
        resolved=Path(path).resolve()
        roots=[store.root.resolve(),*(p.resolve() for p in studies(store).values())]
        source=Path(read_json(store.path('manifest.json'))['source']['path']).resolve()
        if resolved!=source and not any(resolved.is_relative_to(p) for p in roots):raise HTTPException(403,'Outside evidence roots')
        if not resolved.is_file():raise HTTPException(404,'Evidence missing')
        return FileResponse(resolved)
