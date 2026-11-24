"""A single read-only overview of Pass 1, using the existing correction UI."""
from pathlib import Path

from .core import read_json

TABLES = {'observations','states','events','inventory_snapshots','inventory_history',
          'inventory_identity_links','scoreboard_openings','scoreboard_opening_groups',
          'playback_timeline','selection_timeline','ability_slot_timeline','cast_timeline',
          'hud_activations','roster_timeline','issues','ability_icon_timeline','roster_evidence','cast_bar_readings',
          'commentary_timeline', 'timeline'}


def install_pass1_routes(app, store):
    from fastapi import HTTPException, Request
    from fastapi.responses import HTMLResponse, FileResponse
    from .pass1 import records
    from .broadcast_pass1 import TABLES as broadcast_tables
    from .broadcast_match import TABLES as match_tables
    globals()['Request']=Request

    def broadcast():
        path = store.path('pass1_broadcast.json')
        return read_json(path) if path.exists() else None

    def tables():
        if broadcast():
            return broadcast_tables | match_tables if match_directory() else broadcast_tables
        return TABLES

    def match_directory():
        active = store.path('broadcast_match/active.json')
        if not active.is_file():
            return None
        path = store.path('broadcast_match/' + read_json(active)['revision']).resolve()
        return path if path.is_relative_to(active.parent.resolve()) and path.is_dir() else None

    def records_directory(target):
        if target in match_tables and (match := match_directory()):
            return match
        path = store.path('pass1_manifest.json')
        manifest = read_json(path) if path.exists() else {}
        revision = manifest.get('final_revision') or manifest.get('draft_revision')
        return store.path(f'revisions/{revision}') if revision else store.root

    @app.get('/pass1/clip/{clip_id}')
    def broadcast_clip(clip_id: str):
        clip = next((row for row in (broadcast() or {}).get('clips', []) if row['id'] == clip_id), None)
        path = store.path(clip['path']).resolve() if clip else None
        if path is None or not path.is_relative_to(store.root.resolve()) or not path.is_file():
            raise HTTPException(404)
        return FileResponse(path)

    @app.post('/api/pass1/channel-corrections/{record_id}')
    async def correction(record_id: str, request: Request):
        origin=request.headers.get('origin')
        if origin and origin.rstrip('/')!=str(request.base_url).rstrip('/'):
            raise HTTPException(403,'Review writes must originate from this app')
        try:
            from .pass1 import correct_channel
            body=await request.json()
            return correct_channel(store.root,record_id,body['changes'],body['reviewer'])
        except (ValueError,KeyError) as exc:
            raise HTTPException(422,str(exc)) from exc

    @app.get('/pass1', response_class=HTMLResponse)
    def overview():
        return PAGE

    @app.get('/api/pass1')
    def report():
        metadata = broadcast()
        if metadata:
            match = match_directory()
            return {'manifest': metadata, 'broadcast': metadata, 'corrections_pending': False,
                    'report': read_json(store.path('pass1_report.json')), 'tables': sorted(tables()),
                    'match_report': read_json(match / 'report.json') if match and (match / 'report.json').is_file() else None}
        if not store.path('pass1_manifest.json').exists():
            raise HTTPException(404,'This is not an assembled Pass 1 run')
        manifest=read_json(store.path('pass1_manifest.json'))
        return {'manifest':manifest,'report':read_json(store.path('pass1_report.json')),
                'corrections_pending':bool(store.corrections()),'tables':sorted(TABLES)}

    @app.get('/api/pass1/records')
    def table(target: str='observations', offset: int=0, limit: int=100, q: str=''):
        import json
        if target not in tables() or offset<0 or not 1<=limit<=500:
            raise HTTPException(400,'Invalid table or pagination')
        result=[];total=0
        directory=records_directory(target)
        if target == 'timeline':
            source_rows = sorted(({**row, 'record_table': name}
                                  for name in ('events', 'commentary_timeline')
                                  for row in records(directory/(name+'.jsonl'))),
                                 key=lambda row: (row.get('time', row.get('start', 0)), row['id']))
        else:
            source_rows = records(directory/(target+'.jsonl'))
        for row in source_rows:
            if q and q.casefold() not in json.dumps(row,ensure_ascii=False).casefold():continue
            if offset<=total<offset+limit:result.append(row)
            total+=1
        return {'records':result,'total':total}

    @app.get('/pass1/file/{relative:path}')
    def evidence_file(relative: str):
        path=store.path(relative).resolve()
        if not path.is_relative_to(store.root) or not path.is_file() or path.suffix.lower() not in {'.png','.jpg','.jpeg'}:
            raise HTTPException(404)
        return FileResponse(path)

    @app.get('/api/pass1/evidence/{target}/{record_id}')
    def resolve_evidence(target: str, record_id: str):
        if target not in tables():raise HTTPException(400,'Invalid table')
        directory=records_directory(target)
        row=next((r for r in records(directory/(target+'.jsonl')) if r.get('id')==record_id),None)
        if row is None:raise HTTPException(404,'Record not found')
        wanted=set();frames=[];seen=set()
        def visit(record):
            for key in ('frame_ids','evidence_ids','snapshot_ids'):wanted.update(record.get(key,[]))
            for key in ('frame_id','snapshot_id','evidence_id'):
                if record.get(key):wanted.add(record[key])
            for key in ('text_path','evidence_path','crop_path','path','portrait_crop_path'):
                relative=record.get(key)
                if not relative or relative in seen:continue
                path=store.path(relative).resolve()
                if path.is_relative_to(store.root) and path.is_file():
                    frames.append({'path':relative,'time':record.get('time',record.get('start')),'box':record.get('box')})
                    seen.add(relative)
        visit(row)
        for table in ('broadcast_match_evidence','cast_bar_readings','observations','evidence_frames',
                      'roster_evidence','inventory_snapshots'):
            if len(frames)>=12:break
            for record in records(directory/(table+'.jsonl')):
                if record.get('id') in wanted:visit(record)
                if len(frames)>=12:break
        return {'time':row.get('time',row.get('start',0)),'frames':frames[:12]}


PAGE=r'''<!doctype html><html lang="en"><meta charset="utf-8"><title>League Pass 1 review</title>
<style>body{font:15px system-ui;max-width:1400px;margin:24px auto;padding:0 20px;background:#f6f7fa;color:#172033}
video{width:65%;max-height:540px;background:#111}#crop{max-width:30%;max-height:400px;vertical-align:top}
button,select,input{padding:9px;margin:5px}pre{white-space:pre-wrap;overflow-wrap:anywhere}article{background:white;border:1px solid #dde2ec;border-radius:8px;padding:14px;margin:12px 0}a{color:#1356bb}</style>
<h1 id="heading">League · Pass 1</h1><p id="status">Loading completion and coverage…</p>
<p id="midbeast-controls" hidden><a href="/">HUD corrections</a> · <a href="/inventory">Inventory</a> · Select cast_bar_readings below to review or correct channels.</p>
<video id="video" controls preload="metadata"></video><img id="crop" hidden alt="Selected source evidence">
<p><select id="table"></select><input id="query" placeholder="Filter records"><button id="load">Load</button><button id="previous">Previous</button><button id="next">Next</button><span id="count"></span></p>
<div id="rows"></div><script>
let offset=0,loadVersion=0,broadcast=null;const table=document.querySelector('#table'),video=document.querySelector('#video');
function seek(time,clip){const target=clip?'/pass1/clip/'+clip.id:'/media/source';const local=clip?time-clip.start+clip.media_start:time;
const show=()=>{video.currentTime=local;video.pause();};
if(video.getAttribute('src')!==target){video.onloadedmetadata=show;video.src=target;}
else if(video.readyState)show();else video.onloadedmetadata=show;}
function cropURL(path){return '/pass1/file/'+path.split('/').map(encodeURIComponent).join('/');}
function nativeCrop(path){return path&&/\.(png|jpe?g)$/i.test(path);}
function describe(row,target){
if(target==='broadcast_draft')return [row.game_id,row.order?String(row.order):'',row.side,row.action,row.champion,row.state].filter(Boolean).join(' · ');
if(target==='broadcast_roster'||target==='broadcast_player_snapshots')return [row.game_id,row.team??row.side,row.name??row.player_name??'Unresolved player',row.champion].filter(Boolean).join(' · ');
if(target==='broadcast_match_events')return [row.game_id,row.context,row.kind,row.objective??[row.killer_name,row.victim_name].filter(Boolean).join(' → '),row.review_status].filter(Boolean).join(' · ');
return [row.kind??row.field??target,row.champion??row.entity??row.owner_id??''].filter(Boolean).join(' · ');
}
async function load(){const version=++loadVersion,target=table.value;const response=await fetch('/api/pass1/records?'+new URLSearchParams({target,offset,limit:100,q:document.querySelector('#query').value}));const data=await response.json();if(version!==loadVersion)return;
document.querySelector('#count').textContent=offset+'–'+Math.min(offset+100,data.total)+' of '+data.total;
const rows=document.querySelector('#rows');rows.replaceChildren();for(const row of data.records){const article=document.createElement('article'),button=document.createElement('button'),pre=document.createElement('pre');
const time=row.time??row.start??0;const clip=broadcast?.clips.find(item=>item.start<=time&&time<item.end);
const cropLinks=document.createElement('div');
button.textContent='View at '+time.toFixed(3)+' s';
button.onclick=async()=>{
seek(time,clip);
const direct=row.evidence_path??row.crop_path??row.path??row.text_path??row.portrait_crop_path;
const paths=new Map(nativeCrop(direct)?[[direct,time]]:[]);
if(!paths.size||row.evidence_ids?.length>1){
const response=await fetch('/api/pass1/evidence/'+encodeURIComponent(row.record_table??target)+'/'+encodeURIComponent(row.id));
if(response.ok){for(const frame of (await response.json()).frames){if(nativeCrop(frame.path))paths.set(frame.path,frame.time);}}
}
const crop=document.querySelector('#crop');
cropLinks.replaceChildren();
crop.hidden=!paths.size;
if(paths.size)crop.src=cropURL(paths.keys().next().value);
for(const [path,sourceTime] of paths){
const link=document.createElement('a');
link.href=cropURL(path);link.target='_blank';link.rel='noopener';
link.textContent='Native crop'+(Number.isFinite(sourceTime)?' at '+sourceTime.toFixed(2)+' s':'');
cropLinks.append(link,document.createElement('br'));
}
};
pre.textContent=JSON.stringify(row,null,2);const detail=document.createElement('details'),summary=document.createElement('summary'),title=document.createElement('strong');summary.textContent='Evidence record';detail.append(summary,pre);title.textContent=describe(row,target);article.append(title,document.createElement('br'),button,cropLinks,detail);
if(broadcast){const link=document.createElement('a');link.textContent='Open VOD at '+time.toFixed(1)+' s';link.href='https://www.youtube.com/watch?'+new URLSearchParams({v:broadcast.video_id,t:Math.floor(time)+'s'});if(clip)button.after(link);else button.replaceWith(link);}
if(row.summary){const note=document.createElement('p');note.textContent=row.summary;article.insertBefore(note,detail);}
if(row.kind==='commentary'){const transcript=document.createElement('p'),speaker=document.createElement('small');transcript.textContent=row.text??'Unresolved speech';const source=row.transcript_source==='youtube_auto_captions'?'YouTube auto-captions':'ASR';speaker.textContent=source+' · '+(row.speaker??'Speaker unverified')+' · '+row.start.toFixed(2)+'–'+row.end.toFixed(2)+' s · '+row.alignment.status;article.insertBefore(transcript,detail);article.insertBefore(speaker,transcript);}
if(target==='cast_bar_readings'){const input=document.createElement('textarea'),reviewer=document.createElement('input'),save=document.createElement('button');input.placeholder='Changes as JSON, e.g. {"visibility":"unreadable"}';reviewer.placeholder='Reviewer';save.textContent='Save channel correction';save.onclick=async()=>{try{const response=await fetch('/api/pass1/channel-corrections/'+encodeURIComponent(row.id),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({changes:JSON.parse(input.value),reviewer:reviewer.value})});const result=await response.json();if(!response.ok)throw Error(JSON.stringify(result));save.textContent='Saved — finalize to rebuild';}catch(error){save.textContent=error.message;}};article.append(input,reviewer,save);}
rows.append(article);}}
fetch('/api/pass1').then(r=>r.json()).then(data=>{broadcast=data.broadcast??null;document.querySelector('#heading').textContent=(broadcast?.title??'Midbeast')+' · Pass 1';document.querySelector('#midbeast-controls').hidden=!!broadcast;if(!broadcast)video.src='/media/source';document.querySelector('#status').textContent='Status: '+data.report.status+' · Experimental'+(data.corrections_pending?' · Corrections exist; finalize to rebuild dependent outputs':'');for(const name of data.tables){const option=document.createElement('option');option.textContent=name;table.append(option);}table.value=broadcast?'commentary_timeline':'timeline';load();});
document.querySelector('#load').onclick=()=>{offset=0;load();};table.onchange=()=>{offset=0;load();};document.querySelector('#next').onclick=()=>{offset+=100;load();};document.querySelector('#previous').onclick=()=>{offset=Math.max(0,offset-100);load();};
</script></html>'''
