"""Assemble a portable, explicitly experimental League Pass 1 run.

python -m annotation.league.pass1 --help
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import json
import math
import os
from pathlib import Path
import resource
import shutil
import time

from .core import Config, Store, atomic_json, digest, read_json, read_rows, uid, write_rows
from .media import file_hash, probe

REPO = Path(__file__).resolve().parents[2]
DEFAULT = REPO/'data/league_runs/midbeast-pass1-final-v1'
SOURCE_HASH = 'f4f36e1a274617ecbe224eaf7fc3f8375f757acb7a517bcc0ee7780fcd83f240'
PROFILE = 'midbeast-pass1-1080p-v1.json'
COMPONENTS = ('cpu', 'vl', 'dense', 'asr')
OUTPUTS = ('observations', 'normalized_observations', 'states', 'events', 'conflicts',
           'inventory_snapshots', 'inventory_timeline', 'inventory_identity_links', 'inventory_history',
           'playback_timeline', 'selection_timeline', 'scoreboard_visibility', 'scoreboard_openings',
           'scoreboard_opening_groups', 'ability_slot_timeline', 'cast_bar_readings', 'cast_timeline',
           'roster_timeline', 'hud_activations', 'evidence_frames', 'issues', 'ability_icon_timeline', 'roster_evidence',
           'commentary_timeline')


@contextmanager
def lock(root, name):
    path = root/'locks'/f'{name}.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        try:
            flags = fcntl.LOCK_EX if name == 'manifest' else fcntl.LOCK_EX | fcntl.LOCK_NB
            fcntl.flock(stream, flags)
        except BlockingIOError as exc:
            raise RuntimeError(f'Another writer owns {name}') from exc
        yield


def copy_file(source, destination):
    """Only immutable media/assets may share an inode; JSON checkpoints are copied."""
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if file_hash(source) != file_hash(destination):
            raise ValueError(f'Existing imported file changed: {destination}')
        return str(destination)
    if source.suffix.lower() in {'.png','.jpg','.mp4','.safetensors'}:
        try:
            os.link(source, destination)
            return str(destination)
        except OSError:
            pass
    shutil.copy2(source, destination)
    return str(destination)


def copy_tree(source, destination):
    if source.exists():
        shutil.copytree(source, destination, dirs_exist_ok=True, copy_function=copy_file,
                        ignore=shutil.ignore_patterns('__pycache__','.DS_Store','.writer.lock'))


def records(path):
    if path.exists():
        with path.open() as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)


def relocate(value, source, prefix, key='', original_source=None):
    """Adapt paths in a copy, preserving IDs and the original on-disk artifact."""
    if isinstance(value, dict):
        return {k: relocate(v,source,prefix,k,original_source) for k,v in value.items()}
    if isinstance(value, list):
        return [relocate(v,source,prefix,key,original_source) for v in value]
    if isinstance(value, str) and (key.endswith('_path') or key in {'path','crop','root','reader_artifact'}):
        p = Path(value)
        if p.is_absolute():
            try:
                p = p.relative_to(source)
            except ValueError:
                try:
                    p=p.relative_to(original_source) if original_source else p
                except ValueError:
                    return value
                if p.is_absolute():return value
        if (source/p).exists():
            return str(prefix/p)
    return value


def register(root, name, outputs, **metadata):
    # Producer locks are separate; registry updates must still serialize.
    with lock(root,'manifest'):
        manifest = read_json(root/'pass1_manifest.json')
        manifest['producers'][name] = {**metadata, 'outputs': {
            str(p.relative_to(root)): file_hash(p) for p in outputs if p.is_file()}}
        atomic_json(root/'pass1_manifest.json',manifest)


def snapshot_code(root):
    destination = root/'code'
    # This bundle contains the two packages needed by Longleaf and a runnable snapshot.
    for package in ('annotation','ingestion'):
        source = REPO/package
        for p in source.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix in {'.py','.json','.html','.sl','.sh','.txt','.md'}:
                target = destination/p.relative_to(REPO)
                target.parent.mkdir(parents=True,exist_ok=True)
                if p.resolve()!=target.resolve():shutil.copy2(p,target)
    atomic_json(root/'code_checksums.json', {str(p.relative_to(root)):file_hash(p)
        for p in sorted(destination.rglob('*')) if p.is_file()})


def prepare(root, video, imports=(), asset_cache=None, scoreboard_skim=None, producers=()):
    root = Path(root).resolve()
    root.mkdir(parents=True,exist_ok=True)
    with lock(root,'prepare'):
        video = Path(video).resolve()
        sha = file_hash(video)
        if sha != SOURCE_HASH:
            raise ValueError('This calibrated Pass 1 profile requires the pinned Midbeast source')
        existing = read_json(root/'pass1_manifest.json') if (root/'pass1_manifest.json').exists() else None
        if existing and existing['source_sha256'] != sha:
            raise ValueError('Run belongs to another source')
        copy_file(video,root/'source/video.mp4')
        for name in (PROFILE,'midbeast-actions-1080p-v2.json','midbeast-cast-bars-1080p-v1.json'):
            target=root/'profiles'/name;target.parent.mkdir(exist_ok=True)
            shutil.copy2(Path(__file__).with_name('profiles')/name,target)
        source = {**probe(video),'path':'source/video.mp4','sha256':sha}
        atomic_json(root/'manifest.json',{'schema_version':2,'source':source,'interval':[0,source['duration']],
                    'experimental_only':True,'intake_engine':'hybrid-v1'})
        config = Config.load(Path(__file__).with_name('midbeast.json'))
        config.source_video = 'source/video.mp4'
        config.layout_profile = 'profiles/'+PROFILE
        config.asset_cache = 'assets'
        config.evaluation_enabled = False
        atomic_json(root/'config.json',config.to_dict())
        atomic_json(root/'pass1_manifest.json',existing or {'schema_version':1,'experimental':True,
            'source_sha256':sha,'scope':'pass1_without_sam_windows','producers':{},'final_revision':None,
            'release_accepted':False,'full_run_budget_approved':False})
        if asset_cache:
            cache=Path(asset_cache).resolve()
            for sub in ('inventory-references','skin-references','abilities','ability-icons-16.13.1'):
                copy_tree(cache/sub,root/'assets'/sub)
            if (cache/'manifest.json').exists():
                copy_file(cache/'manifest.json',root/'assets/manifest.json')
                copy_tree(cache/'images',root/'assets/images')
        # Model weight files stay in the cluster's pinned cache; locks ship in the bundle.
        copy_file(Path(__file__).with_name('models.local.json'),root/'profiles/models.local.json')
        for index, source_run in enumerate(imports):
            source_run=Path(source_run).resolve()
            if source_run == root or root.is_relative_to(source_run):
                raise ValueError('Import must not contain the destination run')
            manifest=read_json(source_run/'manifest.json')
            if manifest['source']['sha256'] != sha:
                raise ValueError(f'Import source mismatch: {source_run}')
            name='import-'+digest(str(source_run))[:12]
            target=root/'producers'/name
            for item in ('manifest.json','config.json','intake_report.json','intake_runtime.json',
                         'observations.jsonl','intake_records.jsonl','states.jsonl','events.jsonl',
                         'inventory_snapshots.jsonl','inventory_timeline.jsonl','killfeed','field-inputs',
                         'inventory-catalogs','layout_profiles','reader-cache','corrections'):
                src=source_run/item
                if src.is_dir():copy_tree(src,target/item)
                elif src.is_file():copy_file(src,target/item)
            copy_tree(source_run/'experiments/cast-bars',target/'experiments/cast-bars')
            register(root,name,[p for p in target.rglob('*.jsonl')]+[target/'manifest.json']+
                     list((target/'corrections').glob('*.json')),
                     status='imported',origin=str(source_run),source_sha256=sha,
                     input_manifest=manifest,priority=index,provenance_preserved=True,
                     inventory_coverage=read_json(source_run/'intake_report.json') if (source_run/'intake_report.json').exists() else {})
        snapshot_code(root)
        if scoreboard_skim:
            import_roster(root,Path(scoreboard_skim))
        for spec in producers:
            name,path=spec.split('=',1)
            import_producer(root,name,Path(path))
    return audit(root)


def import_producer(root,name,source):
    """Collect a completed CPU/GPU producer without changing its raw metadata."""
    if name not in {*COMPONENTS,'context'}:raise ValueError('Invalid producer name')
    source=source.resolve();root=root.resolve()
    if name == 'asr':
        from .pass1_asr import publish
        report = read_json(source/'report.json')
        if (report.get('source_sha256') != read_json(root/'manifest.json')['source']['sha256']
                or report.get('status') != 'completed_experimental'):
            raise ValueError('Incomplete or mismatched ASR producer')
        destination = root/'producers/asr'
        if destination == source:
            raise ValueError('Producer is already in this run')
        with lock(root, 'asr'):
            for path in source.rglob('*'):
                if path.is_file() and path != source/'report.json':
                    copy_file(path, destination/path.relative_to(source))
            atomic_json(destination/'report.json', report)
            publish(root, report)
        return
    report_file=source/('pass1_stage_report.json' if name in {'cpu','vl'} else 'report.json')
    report=read_json(report_file)
    source_manifest=source/'manifest.json' if name in {'cpu','vl'} else source.parent.parent/'manifest.json'
    if read_json(source_manifest)['source']['sha256']!=read_json(root/'manifest.json')['source']['sha256']:
        raise ValueError('Producer source hash mismatch')
    if name in {'cpu','vl'}:
        manifest=read_json(source/'manifest.json')
        if manifest['source']['sha256']!=read_json(root/'manifest.json')['source']['sha256']:
            raise ValueError('Producer source hash mismatch')
        if not report.get('operational_passed') or report['frames']!=report['expected_frames']:
            raise ValueError('Only complete producer checkpoints can be imported')
    if not report.get('windows'):raise ValueError('Producer has no recorded sampling windows')
    with lock(root,name):
        destination=root/'producers'/name
        if destination==source:raise ValueError('Producer is already in this run')
        copy_tree(source,destination)
        metrics=producer_metrics(destination)
        atomic_json(destination/'pass1_metrics.json',metrics)
        register(root,name,[*destination.glob('*.jsonl'),destination/report_file.name],
                 status='completed_experimental',windows=report['windows'],report=report,
                 origin=str(source),provenance_preserved=True,metrics=metrics)


def import_roster(root, source):
    from .scoreboard_skim import load_skim
    manifest,inputs,samples,names=load_skim(source)
    if manifest['source']['sha256']!=read_json(root/'pass1_manifest.json')['source_sha256']:
        raise ValueError('Roster belongs to another recording')
    destination=root/'producers/roster'
    for name in ('manifest.json','skim-evidence.json','scoreboard-samples.json','scoreboard-names.json',
                 'match-roster.json','match_roster_timeline.jsonl','report.json'):
        if (source/name).exists():copy_file(source/name,destination/name)
    for name in ('portraits','names','scoreboards'):copy_tree(source/name,destination/name)
    register(root,'roster',[p for p in destination.glob('*.json')]+list(destination.glob('*.jsonl')),
             status='imported',source_sha256=manifest['source']['sha256'],origin=str(source),
             inputs=inputs,report=read_json(source/'report.json'))


def source_clocks(root):
    """Prefer the complete imported clock producer; retain its evidence scope."""
    selected = {}
    from .normalization import accepted
    manifest=read_json(root/'pass1_manifest.json')
    corrections={c['record_id']:c for c in Store(root).corrections() if c['target']=='observations'}
    candidates=sorted(manifest['producers'].items(),key=lambda x:x[1].get('priority',100))
    for name, meta in candidates:
        if not name.startswith('import-') and name!='cpu':continue
        source=root/'producers'/name
        for row in records(source/'observations.jsonl'):
            if row['field'] != 'game_clock':continue
            row=relocate(row,source,Path('producers')/name)
            t=row['time'];key=round(t)
            correction=corrections.get(row['id'])
            if correction:
                row.update(correction['changes']);row['review_status']=correction['review_status']
            value=row.get('value')
            clock=value.get('seconds') if isinstance(value,dict) and accepted(row) else None
            old=selected.get(key)
            if old and old['clock'] is not None and clock is not None and old['clock']!=clock:
                old.update(clock=None,evidence_ids=old['evidence_ids']+[row['id']])
            elif key not in selected or (old['clock'] is None and clock is not None):
                selected[key]={'id':uid('pass1-clock-scope',row['id']),'start':t,'end':t+1,
                    'clock':clock,
                    'playback_segment_id':row.get('playback_segment_id',uid('unknown-clock',t)),
                    'continuity_verified':row.get('continuity_verified',False),'evidence_ids':[row['id']]}
    return sorted(selected.values(),key=lambda r:r['start'])


def slow_stage(root, component, windows, asset_cache=None, model_cache=None):
    from .hybrid_intake import annotate
    profile=read_json(root/'profiles'/PROFILE)
    imported=compatible_inventory(root)
    # Inventory is retained by the original producer when a complete run is available.
    profile['fields']=[f for f in profile['fields'] if
        (f.get('backend')=='vl' if component=='vl' else f.get('backend')!='vl') and
        not (imported and f.get('backend')=='inventory')]
    profile['id']+='-'+component
    work=root/'producers'/component
    work.mkdir(parents=True,exist_ok=True)
    atomic_json(work/'profile.json',profile)
    manifest=read_json(root/'manifest.json')
    manifest['source']['path']=str(root/manifest['source']['path'])
    atomic_json(work/'manifest.json',manifest)
    config=Config.load(root/'config.json')
    config.source_video=manifest['source']['path']
    config.layout_profile=str(work/'profile.json')
    config.asset_cache=str(Path(asset_cache).resolve() if asset_cache else root/'assets')
    if component=='vl' and (model_cache or os.environ.get('LEAGUE_MODEL_CACHE')):
        config.asset_cache=str(Path(model_cache or os.environ['LEAGUE_MODEL_CACHE']).resolve())
    config.vl_python=os.environ.get('LEAGUE_VL_PYTHON')
    atomic_json(work/'config.json',config.to_dict())
    times=[float(t) for a,b in windows for t in range(math.ceil(a),math.ceil(b)) if t<b]
    child=Store(work);child.install_signals()
    started=time.monotonic()
    result=annotate(child,config,sample_times=times,experimental=True)
    result['wall_seconds_including_setup']=time.monotonic()-started
    result['windows']=windows
    atomic_json(work/'pass1_stage_report.json',result)
    metrics=producer_metrics(work)
    atomic_json(work/'pass1_metrics.json',metrics)
    register(root,component,[work/'observations.jsonl',work/'pass1_stage_report.json'],
        status='completed_experimental',windows=windows,report=result,
        implementation=file_hash(Path(__file__)),profile_hash=digest(profile),metrics=metrics)
    return result


def producer_metrics(directory):
    selected=set();requests=set()
    for row in records(directory/'observations.jsonl'):
        selected.add(row['task_id'])
        artifact=row.get('reader_artifact')
        if artifact and artifact.startswith('vl-requests/'):
            requests.update((directory/Path(artifact).parent).rglob('response.json'))
    tasks=[read_json(p) for p in (directory/'tasks').glob('*.json') if p.stem in selected]
    responses=[read_json(p) for p in sorted(requests)]
    return {'selected_samples':len(selected),'selected_sample_checkpoints':len(tasks),
        'field_cache_hits':sum(t['metrics'].get('reader_cache_hits',0) for t in tasks),
        'requests_referenced_by_selected_samples':len(responses),
        'requests_including_repeated_allocations':sum(1 for _ in (directory/'vl-requests').rglob('response.json')),
        'prompt_tokens':sum(r.get('usage',{}).get('prompt_tokens',0) for r in responses),
        'generated_tokens':sum(r.get('usage',{}).get('completion_tokens',0) for r in responses),
        'referenced_request_seconds':sum(r.get('wall_seconds',0) for r in responses),
        'producer_logical_bytes':sum(p.stat().st_size for p in directory.rglob('*') if p.is_file())}


def compatible_inventory(root):
    """A partial import must never suppress inventory extraction in a new run."""
    source=read_json(root/'manifest.json')
    expected=math.ceil(source['interval'][1]-source['interval'][0])
    profile=read_json(root/'profiles'/PROFILE)
    wanted={(f['id'],tuple(f['box'])) for f in profile['fields'] if f.get('backend')=='inventory'}
    for name,meta in read_json(root/'pass1_manifest.json')['producers'].items():
        if not name.startswith('import-'):continue
        directory=root/'producers'/name
        report=meta.get('inventory_coverage') or (read_json(directory/'intake_report.json')
            if (directory/'intake_report.json').exists() else {})
        if not (report.get('operational_passed') and report.get('frames')==expected and
                meta['input_manifest'].get('interval')==source['interval']):continue
        for p in (directory/'layout_profiles').glob('*.json'):
            old=read_json(p)
            fields={(f['id'],tuple(f['box'])) for f in old.get('fields',[]) if f.get('backend')=='inventory'}
            if fields==wanted and old.get('inventory')==profile.get('inventory'):
                return True
    return False


def execute(root, *, pilot, component='all', asset_cache=None, model_cache=None, budget_approved=False):
    from .hybrid_intake import pilot_windows
    root=Path(root).resolve()
    manifest=read_json(root/'manifest.json')
    duration=manifest['interval'][1]
    # Five fixed one-minute windows, aligned to the one-FPS slow sampling grid.
    windows=[(float(math.floor(a)),float(math.floor(a)+60)) for a,b in pilot_windows(0,duration)] if pilot else [(0.,duration)]
    if not pilot and not budget_approved:
        raise ValueError('Full-video compute requires --budget-approved after reviewing the pilot estimate')
    if not pilot:
        with lock(root,'manifest'):
            record=read_json(root/'pass1_manifest.json');record['full_run_budget_approved']=True
            atomic_json(root/'pass1_manifest.json',record)
    components=COMPONENTS if component=='all' else (component,)
    reports={}
    for name in components:
        with lock(root,name):
            if name=='asr':
                from .pass1_asr import run
                reports[name]=run(root,pilot=pilot,model_cache=model_cache)
            elif name=='context':
                from .pass1_context import refine
                child=Store(root);child.install_signals()
                reports[name]=refine(child,windows,source_clocks(root));work=root/'producers/context'
                register(root,name,[*work.glob('*.jsonl'),work/'report.json'],
                         status='completed_experimental',windows=windows,report=reports[name])
            elif name=='dense':
                from .pass1_dense import run_dense
                child=Store(root);child.install_signals()
                reports[name]=run_dense(child,windows,asset_cache or root/'assets',source_clocks(root))
                work=root/'producers/dense'
                register(root,name,[*work.glob('*.jsonl'),work/'report.json'],status='completed_experimental',
                         windows=windows,report=reports[name],implementation=file_hash(Path(__file__).with_name('pass1_dense.py')))
                from .pass1_context import refine
                with lock(root,'context'):
                    context_report=refine(child,windows,source_clocks(root))
                    work=root/'producers/context'
                    register(root,'context',[*work.glob('*.jsonl'),work/'report.json'],
                             status='completed_experimental',windows=windows,report=context_report)
            else:
                reports[name]=slow_stage(root,name,windows,asset_cache,model_cache)
    if not pilot and component in {'all','cpu','dense'}:
        ensure_auxiliary(root,asset_cache)
    report=audit(root)
    return {'stages':reports,'audit':report}


def ensure_auxiliary(root,asset_cache=None):
    """Full-run CPU prerequisites; never scan the full recording during a pilot."""
    from .scoreboard_skim import skim
    manifest=read_json(root/'manifest.json')
    cache=Path(asset_cache).resolve() if asset_cache else root/'assets'
    roster=root/'producers/roster'
    with lock(root,'roster'):
        if not (roster/'match-roster.json').exists():
            video=root/manifest['source']['path']
            result=skim(video,roster,str(root/'profiles'/PROFILE),cache,single_match=True,names=False)
            register(root,'roster',[*roster.glob('*.json'),*roster.glob('*.jsonl')],
                     status='completed_experimental',report=result)
    if audit(root,write=False)['required_producers']['killfeed']:
        return
    with lock(root,'feed'):
        from .killfeed import run
        work=root/'producers/feed';work.mkdir(parents=True,exist_ok=True)
        source_manifest=deepcopy(manifest)
        source_manifest['source']['path']=str(root/manifest['source']['path'])
        atomic_json(work/'manifest.json',source_manifest)
        child=Store(work);child.install_signals()
        result=run(child,cache,roster)
        register(root,'feed',[p for p in work.rglob('*') if p.is_file() and p.suffix in {'.json','.jsonl'}],
                 status='completed_experimental',report=result)


def imported_rows(root, filename):
    manifest=read_json(root/'pass1_manifest.json')
    for name, meta in sorted(manifest['producers'].items(),key=lambda x:x[1].get('priority',100)):
        if name.startswith('import-'):
            directory=root/'producers'/name
            corrections=Store(directory).corrections() if filename=='observations.jsonl' else []
            for row in records(directory/filename):
                for correction in corrections:
                    if correction['target']=='observations' and correction['record_id']==row['id']:
                        row.update(correction['changes']);row['review_status']=correction['review_status']
                        row['imported_corrected']=True
                yield relocate(row,directory,Path('producers')/name)


def all_observations(root):
    from .normalization import effective_intake
    # Identity is by producer record, not value; differing readings remain reviewable.
    indexed={}
    for row in imported_rows(root,'observations.jsonl'):
        if row['id'] not in indexed:indexed[row['id']]=row
    for name in ('cpu','vl'):
        directory=root/'producers'/name
        config=read_json(directory/'config.json') if (directory/'config.json').exists() else {}
        original=Path(config['layout_profile']).parent if config.get('layout_profile') else None
        for row in records(directory/'observations.jsonl'):
            row=relocate(row,directory,Path('producers')/name,original_source=original)
            indexed[row['id']]=row
    store=Store(root)
    for row in records(root/'producers/context/clock_readings.jsonl'):
        indexed[row['id']]=row
    rows=store.effective(list(indexed.values()),'observations')
    corrected={c['record_id'] for c in store.corrections() if c['target']=='observations'}
    corrected.update(r['id'] for r in rows if r.get('imported_corrected'))
    return effective_intake(rows,corrected)


def derive(root):
    from .action_hud_replay import cooldown_candidates
    from .cast_bars import channel_occurrences
    from .inventory_timeline import inventory_timeline
    from .pass1_timeline import bind_visible_portraits, inventory_bindings, item_history, map_context, scoreboard_openings
    from .timeline import fuse_observations
    raw=all_observations(root)
    dense=root/'producers/dense'
    context=root/'producers/context'
    playback=read_rows(context/'playback.jsonl') or read_rows(dense/'playback.jsonl')
    # Outside dense coverage, keep the original source-clock continuity evidence.
    for clock in source_clocks(root):
        if not any(p['start']<=clock['start']<p['end'] for p in playback):playback.append(clock)
    playback.sort(key=lambda p:p['start'])
    selection=read_rows(context/'selection.jsonl') or read_rows(dense/'selection.jsonl')
    rows=bind_visible_portraits(map_context(raw,playback,selection))
    profile=read_json(root/'profiles'/PROFILE)
    for row in rows:
        if row.get('backend')=='inventory' and row.get('panel_state')!='present':
            row['field_box']=row['box']
            row['box']=profile['inventory']['panels'][row['route']]
    states,conflicts=fuse_observations([r for r in rows if r.get('backend')!='inventory'])
    snapshots,inventory=inventory_timeline(rows)
    snapshots=map_context(snapshots,playback,selection)
    snapshots,links=inventory_bindings(snapshots,rows)
    vis=map_context(read_rows(dense/'visibility.jsonl'),playback)
    openings,groups=scoreboard_openings(vis,snapshots)
    casts=read_rows(dense/'casts.jsonl')
    if not casts:
        for name in sorted(read_json(root/'pass1_manifest.json')['producers']):
            if not name.startswith('import-'):continue
            directory=root/'producers'/name
            for p in directory.glob('experiments/cast-bars/*/effective_cast_bar_readings.jsonl'):
                casts.extend(relocate(r,directory,Path('producers')/name) for r in records(p))
    casts=Store(root).effective(map_context(casts,playback,selection),'cast_bar_readings')
    channels=channel_occurrences(casts)
    channels=[{**r,'kind':'observed_channel','experimental':True,'review_status':'unreviewed'} for r in channels]
    slots=map_context(read_rows(dense/'slots.jsonl'),playback,selection)
    activations=[{**r,'kind':'hud_slot_activation','frame_ids':r['onset_frame_ids'],
        'evidence_ids':r['onset_frame_ids'],'experimental':True,'review_status':'unreviewed','outcome':None}
        for r in cooldown_candidates(slots,{'ocr_confidence':.85,'pixel_change':.08})]
    feed=[];roster=[]
    for name in sorted(read_json(root/'pass1_manifest.json')['producers']):
        if not name.startswith('import-') and name!='feed':continue
        directory=root/'producers'/name
        active=directory/'killfeed/active.json'
        if active.exists():
            revision=read_json(active)['revision'];base=directory/'killfeed'/revision
            feed.extend(relocate(r,directory,Path('producers')/name) for r in records(base/'killfeed_timeline.jsonl'))
            roster.extend(relocate(r,directory,Path('producers')/name) for r in records(base/'killfeed_roster_timeline.jsonl'))
    feed=list({r['id']:r for r in feed}.values())
    roster_evidence=[]
    roster_root=root/'producers/roster'
    if (roster_root/'match-roster.json').exists():
        roster_document=read_json(roster_root/'match-roster.json')
        for side,members in roster_document.get('members',{}).items():
            for champion,member in members.items():
                witness=member['first']
                roster_evidence.append({'id':witness['id'],'time':witness['time'],'champion':champion,
                    'side':side,'path':str(Path('producers/roster')/witness['crop']),
                    'crop_sha256':witness['crop_sha256'],'review_status':'unreviewed'})
        roster=[{**r,'id':uid('pass1-roster',r),'evidence_ids':[r['evidence_id']],
                 'experimental':True,'review_status':'unreviewed'}
                for r in read_rows(roster_root/'match_roster_timeline.jsonl')]
    events=sorted(map_context(feed,playback)+channels+activations,key=lambda r:(r['start'],r['id']))
    issues=[{'id':uid('pass1-issue',r['id']),'time':r['time'],'field':r['field'],
             'evidence_ids':[r['id']],'reason':r.get('acceptance_reason','unknown reading')}
            for r in rows if r.get('acceptance')=='unknown']
    if (dense/'report.json').exists():
        issues.extend(read_json(dense/'report.json').get('unavailable_samples', []))
    result=dict(zip(OUTPUTS,[rows,rows,states,events,conflicts,snapshots,inventory,links,item_history(snapshots),
        playback,selection,vis,openings,groups,slots,casts,channels,roster,activations,
        read_rows(dense/'frames.jsonl')+read_rows(context/'frames.jsonl'),issues,
        read_rows(dense/'identities.jsonl'),roster_evidence]))
    from .pass1_asr import map_commentary
    from .pass1_youtube_captions import selected_commentary
    commentary_report = selected_commentary(root)
    commentary = read_rows(root/commentary_report['commentary_path']) if commentary_report.get('commentary_path') else []
    result['commentary_timeline'] = map_commentary(commentary, playback)
    source_hash=read_json(root/'manifest.json')['source']['sha256']
    for table in result.values():
        for row in table:
            if row.get('source_sha256',source_hash)!=source_hash:
                raise ValueError('Mixed source evidence in Pass 1 output')
            row.setdefault('source_sha256',source_hash)
            row.setdefault('experimental',True)
            row.setdefault('agent_verification_status','not_started')
    return result


def audit(root, *, write=True):
    root=Path(root)
    manifest=read_json(root/'pass1_manifest.json')
    duration=read_json(root/'manifest.json')['interval'][1]
    capabilities={}
    for component in COMPONENTS:
        meta=manifest['producers'].get(component,{})
        report=meta.get('report',{})
        windows=meta.get('windows',[])
        complete=(meta.get('status')=='completed_experimental' and windows==[[0.,duration]])
        capabilities[component]={'complete':complete,'status':meta.get('status','not_run'),
             'sampled_seconds':sum(b-a for a,b in windows),'windows':windows,
             'samples':report.get('frames',0),'wall_seconds':max(report.get('stage_wall_seconds',0),report.get('wall_seconds',0)),
             'gpu_peak_bytes':report.get('gpu_memory_peak_bytes',0)}
    required={name:c['complete'] for name,c in capabilities.items()
              if name != 'asr' or name in manifest['producers']}
    if 'youtube_captions' in manifest['producers']:
        required.pop('asr', None)
        required['youtube_captions'] = (
            manifest['producers']['youtube_captions'].get('status') == 'completed_experimental'
            and manifest['producers']['youtube_captions'].get('windows') == [[0., duration]])
    required['inventory']=compatible_inventory(root) or bool(capabilities['cpu']['complete'] and
        any(f.get('backend')=='inventory' for f in (read_json(root/'producers/cpu/profile.json')['fields']
            if (root/'producers/cpu/profile.json').exists() else [])))
    roster=root/'producers/roster/match-roster.json'
    required['roster']=bool(roster.exists() and read_json(roster).get('accepted_members')==10)
    required['killfeed']=False
    feed_reports=[]
    for directory in (root/'producers').glob('*'):
        active=directory/'killfeed/active.json'
        if not active.exists():continue
        snapshot=read_json(active);recipe=snapshot['recipe']
        if recipe['source_sha256']==manifest['source_sha256'] and recipe['interval']==[0.,duration]:
            base=directory/'killfeed'/snapshot['revision']
            required['killfeed']=all((base/(name+'.jsonl')).is_file() for name in
                ('killfeed_timeline','killfeed_readings','killfeed_roster_timeline'))
            if (base/'benchmark.json').exists():feed_reports.append(read_json(base/'benchmark.json'))
    context=manifest['producers'].get('context',{}).get('report',{})
    required['shared_context']=context.get('windows')==[[0.,duration]] and context.get('status')=='completed_experimental'
    capabilities['dense']['wall_seconds']+=context.get('wall_seconds',0)
    projections={name: {'measured_wall_seconds': c['wall_seconds'],
        'remaining_wall_seconds_with_30_percent_headroom': (c['wall_seconds']*(duration-c['sampled_seconds'])/c['sampled_seconds']*1.3
            if c['sampled_seconds'] else None)} for name,c in capabilities.items()}
    report={'experimental':True,'release_accepted':False,'status':'assembled_unverified' if all(required.values()) else 'incomplete',
        'source_seconds':duration,'capabilities':capabilities,'compute_projection':projections,
        'required_producers':required,'killfeed_benchmarks':feed_reports,
        'full_run_budget_approved':manifest['full_run_budget_approved'],
        'validation':'Operational coverage is not independently measured accuracy',
        'reopen_policy':{'max_gap_seconds':2,'status':'development_policy_pending_agent_verification'},
        'verification_plan':{'method':'agent_annotation','status':'not_started',
                             'manual_review_required':False,'stage':'separate_future_pass'},
        'driver_peak_rss':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    report['longleaf_jobs']=manifest.get('longleaf_jobs',{})
    from .pass1_youtube_captions import selected_commentary
    report['commentary']=selected_commentary(root)
    report['dense_unavailable_samples']=manifest['producers'].get('dense',{}).get('report',{}).get('unavailable_samples',[])
    gpu_jobs=[j for j in report['longleaf_jobs'].values() if j.get('component')=='vl']
    report['pilot_gpu_allocated_hours']=sum(j.get('elapsed_seconds',0) for j in gpu_jobs
                                           if j.get('scope')!='full_video')/3600
    report['full_gpu_allocated_hours']=sum(j.get('elapsed_seconds',0) for j in gpu_jobs
                                          if j.get('scope')=='full_video')/3600
    report['producer_metrics']={name:meta.get('metrics',{}) for name,meta in manifest['producers'].items()
                                if name in COMPONENTS}
    review=root/'reviews/reopen-development.json'
    if review.exists():report['reopen_policy']['source_diagnostic']=read_json(review)
    report['compute_notes']=['CPU timings are local pilot wall times; Longleaf CPU throughput may differ.',
        'Projections scale the five development windows with 30 percent headroom; they are not a fixed quote.',
        'Failed and repeated allocations remain in longleaf_jobs; useful sample work is reported separately.']
    if write:atomic_json(root/'pass1_report.json',report)
    lines=['# Midbeast Pass 1','',f'Status: {report["status"]}. Experimental; release acceptance pending.','',
           '| Producer | Status | Sampled seconds | Full coverage |','| --- | --- | ---: | --- |']
    for name,c in capabilities.items():lines.append(f'| {name} | {c["status"]} | {c["sampled_seconds"]:.1f} | {c["complete"]} |')
    if write:write_report_markdown(root,report)
    return report


def write_report_markdown(root,report):
    lines=['# Midbeast Pass 1','',f'Status: **{report["status"]}**. Verification by agent annotation is a separate future pass.','',
        '## Extraction coverage','', '| Producer | Samples | Source seconds | Full recording |',
        '| --- | ---: | ---: | --- |']
    for name,c in report['capabilities'].items():
        lines.append(f'| {name} | {c["samples"]} | {c["sampled_seconds"]:.1f} | {c["complete"]} |')
    dense=report.get('dense_unavailable_samples', [])
    if dense:
        lines+=['',f'{len(dense)} scheduled dense samples follow the last video frame. Their source times are listed in `issues.jsonl`.']
    lines+=['','## Required producers','']
    lines += [f'- [{"x" if complete else " "}] {name}' for name,complete in report['required_producers'].items()]
    lines+=['','## Compute','','Remaining time is projected from pilot coverage with 30% headroom.','',
        '| Producer | Measured minutes | Estimated remaining wall hours |','| --- | ---: | ---: |']
    for name,p in report['compute_projection'].items():
        hours=p['remaining_wall_seconds_with_30_percent_headroom']
        value=f'{hours/3600:.2f}' if hours is not None else 'pending pilot'
        lines.append(f'| {name} | {p["measured_wall_seconds"]/60:.1f} | {value} |')
    lines+=['']+report.get('compute_notes',[])
    if report.get('counts'):
        lines+=['','## Assembled draft' if report['draft'] else '## Assembled run',
            '','| Table | Records |','| --- | ---: |']
        lines += [f'| {name} | {count} |' for name,count in report['counts'].items()]
        lines += ['',f'Inventory snapshots without a champion: {report["unknown_inventory_identities"]}.']
    lines+=['','## Review and limitations','',
        'Open `/pass1` in the existing review app for source playback and native crops.',
        'The two-second reopen threshold is a development policy; positive source merge accuracy is unmeasured.',
        'Unknown identities, hidden items, uncertain channel endings and playback gaps remain explicit.',
        'Manual review is not a completion requirement. Agent verification is not yet started.',
        ('Full-video extraction is paused at the user-requested pilot budget checkpoint.'
         if not report['full_run_budget_approved'] else 'Full-video extraction was approved at the pilot budget checkpoint.'),
        'Detailed field coverage, source diagnostics, job receipts and checksums are in `pass1_report.json`.','']
    (root/'pass1_report.md').write_text('\n'.join(lines))


def verify(root, *, require_complete=False, check_revision=True):
    root=Path(root).resolve();manifest=read_json(root/'pass1_manifest.json');errors=[]
    source=root/read_json(root/'manifest.json')['source']['path']
    if not source.resolve().is_relative_to(root) or not source.is_file() or file_hash(source)!=manifest['source_sha256']:
        errors.append('source checksum or external source path')
    for name,meta in manifest['producers'].items():
        for relative,sha in meta.get('outputs',{}).items():
            p=root/relative
            if not p.resolve().is_relative_to(root) or not p.is_file() or file_hash(p)!=sha:
                errors.append(f'producer checksum: {name}/{relative}')
    for relative,sha in read_json(root/'code_checksums.json').items():
        p=root/relative
        if not p.resolve().is_relative_to(root) or not p.is_file() or file_hash(p)!=sha:errors.append(f'code checksum: {relative}')
    for p in root.rglob('*'):
        if p.is_symlink() and not p.resolve().is_relative_to(root):errors.append(f'external symlink: {p.relative_to(root)}')
    if require_complete and audit(root,write=False)['status']!='assembled_unverified':errors.append('required full-video producers incomplete')
    revision=manifest.get('final_revision') or manifest.get('draft_revision')
    if check_revision and revision:
        ledger=root/'revisions'/revision/'checksums.json'
        if ledger.exists():
            checks=read_json(ledger)
            if checks['inputs']!=input_signature(root):errors.append('derived revision is stale; finalize again')
            for relative,sha in checks['files'].items():
                p=root/relative
                if not p.resolve().is_relative_to(root) or not p.is_file() or file_hash(p)!=sha:
                    errors.append(f'bundle checksum: {relative}')
        else:errors.append('revision has no sealed evidence checksums; finalize again')
    return {'passed':not errors,'errors':errors,'require_complete':require_complete,
            'release_accepted':False}


def input_signature(root):
    return digest({'producers':read_json(root/'pass1_manifest.json')['producers'],
        'corrections':Store(root).corrections(),'code':read_json(root/'code_checksums.json'),
        'profiles':{p.name:file_hash(p) for p in (root/'profiles').glob('*.json')}})


def cited_files(root,data):
    paths={}
    keys={'path','evidence_path','crop_path','portrait_crop_path','text_path','progress_path','cache_path','reader_artifact'}
    def walk(value):
        if isinstance(value,list):
            for child in value:walk(child)
        elif isinstance(value,dict):
            for key,child in value.items():
                if key in keys and isinstance(child,str):
                    p=root/child
                    if p.resolve().is_relative_to(root) and p.is_file():paths[str(p.relative_to(root))]=p
                if isinstance(child,(dict,list)):walk(child)
    walk(list(data.values()))
    for p in (root/'assets').rglob('*'):
        if p.is_file():paths[str(p.relative_to(root))]=p
    return {relative:file_hash(p) for relative,p in sorted(paths.items())}


def finalize(root, *, draft=False):
    root=Path(root).resolve()
    with lock(root,'finalize'):
        snapshot_code(root)
        before=read_json(root/'pass1_manifest.json')
        # Existing inventory editors resolve reference catalogs at the run root.
        for producer in (root/'producers').glob('*'):
            copy_tree(producer/'inventory-catalogs',root/'inventory-catalogs')
        result=verify(root,require_complete=not draft,check_revision=False)
        if not result['passed']:raise ValueError(result['errors'])
        inputs=input_signature(root)
        context_report=root/'producers/context/report.json'
        if context_report.exists():
            from .pass1_context import context_recipe
            context=read_json(context_report)
            if context.get('recipe')!=context_recipe(Store(root),context['windows'],source_clocks(root)):
                raise ValueError('Shared context is stale; run pilot/run --component context before finalizing')
        revision=digest({'producers':before['producers'],'corrections':Store(root).corrections(),
                         'code':read_json(root/'code_checksums.json'),
                         'profiles':{p.name:file_hash(p) for p in (root/'profiles').glob('*.json')}})[:20]
        destination=root/'revisions'/revision
        data=derive(root)
        declared={f['id'] for f in read_json(root/'profiles'/PROFILE)['fields']}
        samples={field:set() for field in declared}
        expected=math.ceil(read_json(root/'manifest.json')['interval'][1])
        for row in data['observations']:
            if row['field_id'] in samples:
                bucket=math.floor(row.get('sample_time',row['time'])+1e-6)
                if 0<=bucket<expected:samples[row['field_id']].add(bucket)
        missing_samples={field:expected-len(times) for field,times in samples.items() if len(times)<expected}
        if not draft and missing_samples:raise ValueError({'incomplete_field_samples':missing_samples})
        destination.mkdir(parents=True,exist_ok=True)
        for name,rows in data.items():write_rows(destination/(name+'.jsonl'),rows)
        if read_json(root/'pass1_manifest.json')!=before:raise ValueError('Producers changed during fusion; retry')
        result=verify_derived(root,data)
        atomic_json(destination/'verification.json',result)
        if not result['passed']:raise ValueError(result['errors'][:20])
        files=cited_files(root,data)
        files.update({str((destination/(name+'.jsonl')).relative_to(root)):
            file_hash(destination/(name+'.jsonl')) for name in OUTPUTS})
        if input_signature(root)!=inputs:raise ValueError('Evidence changed during fusion; retry')
        for name in OUTPUTS:
            shutil.copy2(destination/(name+'.jsonl'),root/(name+'.jsonl'))
            files[name+'.jsonl']=files[str((destination/(name+'.jsonl')).relative_to(root))]
        atomic_json(destination/'checksums.json',{'inputs':inputs,'files':files})
        with lock(root,'manifest'):
            current=read_json(root/'pass1_manifest.json')
            if current!=before:raise ValueError('Producers changed during verification; retry')
            current['draft_revision' if draft else 'final_revision']=revision
            atomic_json(root/'pass1_manifest.json',current)
        report=audit(root)
        report.update(revision=revision,draft=draft,counts={k:len(v) for k,v in data.items()},
                      unknown_inventory_identities=sum(not s.get('champion') for s in data['inventory_snapshots']))
        from collections import Counter,defaultdict
        coverage=defaultdict(Counter)
        for row in data['observations']:
            coverage[row['field_id']][row['acceptance']]+=1
        report['field_coverage']={k:dict(v) for k,v in coverage.items()}
        report['declared_fields']=[f['id'] for f in read_json(root/'profiles'/PROFILE)['fields']]
        report['missing_field_outputs']=sorted(set(report['declared_fields'])-set(coverage))
        report['missing_scheduled_field_samples']=missing_samples
        atomic_json(root/'pass1_report.json',report)
        write_report_markdown(root,report)
        return report


def verify_derived(root,data):
    errors=[]
    known={r['id'] for rows in data.values() for r in rows if 'id' in r}
    # Original feed crop IDs and raw crop-frame IDs are resolved in imported evidence ledgers.
    for directory in (root/'producers').glob('import-*'):
        for p in directory.glob('killfeed/*/killfeed_readings.jsonl'):
            for r in records(p):
                known.add(r['id'])
                for entry in r.get('entries',[]):
                    if 'id' in entry:known.add(entry['id'])
    for r in data['observations']:
        known.update([r['frame_id'],r['source_id'],r['task_id']])
    for table,rows in data.items():
        for row in rows:
            for key in ('evidence_ids','frame_ids'):
                for identifier in row.get(key,[]):
                    if identifier not in known:errors.append(f'{table}/{row.get("id")}: unresolved {identifier}')
            for key in ('evidence_path','crop_path','path','text_path','progress_path'):
                if row.get(key):
                    p=root/row[key]
                    if not p.resolve().is_relative_to(root) or not p.is_file():errors.append(f'{table}: missing {row[key]}')
    return {'passed':not errors,'errors':errors,'tables':{k:len(v) for k,v in data.items()}}


def correct_channel(root, record_id, changes, reviewer):
    """Append a source review; finalization rebuilds channels and fused events."""
    from .cast_bars import AbilityCatalog
    from .core import now
    root=Path(root);allowed={'raw_text','ability_id','channel_kind','visibility'}
    if not isinstance(changes,dict) or set(changes)-allowed or not isinstance(reviewer,str) or not reviewer.strip():
        raise ValueError('Channel corrections require supported fields and a reviewer')
    store=Store(root)
    rows=store.effective(read_rows(root/'cast_bar_readings.jsonl'),'cast_bar_readings')
    original=next((r for r in rows if r['id']==record_id),None)
    if original is None:raise ValueError('Unknown channel reading')
    row={**original,**changes}
    if row['visibility'] not in {'readable','unreadable','absent','unknown'} or row['channel_kind'] not in {'ability','recall','unknown'}:
        raise ValueError('Invalid channel kind or visibility')
    if not isinstance(row['raw_text'],str) or len(row['raw_text'])>300:raise ValueError('Invalid cast text')
    catalog=AbilityCatalog(root/'assets/abilities/16.22.1/catalog.json')
    ability=catalog.abilities.get(row['ability_id'])
    if row['channel_kind']=='ability' and (not ability or (row.get('champion') and ability['champion']!=row['champion'])):
        raise ValueError('Ability conflicts with the visible owner or catalog')
    if row['channel_kind']!='ability' and row['ability_id'] is not None:raise ValueError('Non-ability channel cannot have an ability ID')
    directory=root/'corrections';directory.mkdir(exist_ok=True)
    with (directory/'.lock').open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        revision=len(store.corrections())+1
        correction={'id':uid('pass1-channel-correction',record_id,changes,reviewer,now()),
            'target':'cast_bar_readings','record_id':record_id,'changes':changes,
            'review_status':'corrected','reviewer':reviewer,'created_at':now(),'revision':revision}
        atomic_json(directory/f'{revision:08d}.json',correction)
    return correction


def main(argv=None):
    from . import broadcast_match, broadcast_pass1

    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    broadcast_pass1.add_arguments(sub.add_parser('broadcast', help='Screen and review tournament footage'))
    broadcast_match.add_arguments(sub.add_parser('broadcast-match', help='Read roster, draft, and match events'))
    for command in ('prepare','pilot','run','finalize','verify','verify-asr','import-captions'):
        s=sub.add_parser(command);s.add_argument('--run-dir',type=Path,default=DEFAULT)
        if command=='prepare':
            s.add_argument('--video',type=Path,default=REPO/'data/youtube/lol_midbeast/OLLZBe5-fg4.mp4')
            s.add_argument('--import-run',type=Path,action='append',default=[])
            s.add_argument('--asset-cache',type=Path,default=REPO/'data/league_assets')
            s.add_argument('--scoreboard-skim',type=Path)
            s.add_argument('--import-producer',action='append',default=[],metavar='NAME=PATH',help='Import a completed cpu, vl, dense, context or asr producer')
        elif command in {'pilot','run'}:
            s.add_argument('--component',choices=(*COMPONENTS,'context','all'),default='all')
            s.add_argument('--asset-cache',type=Path)
            s.add_argument('--model-cache',type=Path,help='Pinned GPU checkpoint cache; separate from bundled image references')
            if command=='run':s.add_argument('--budget-approved',action='store_true')
        elif command=='finalize':s.add_argument('--draft',action='store_true')
        elif command=='verify-asr':s.add_argument('--reference',type=Path,required=True)
        elif command=='import-captions':
            s.add_argument('--captions',type=Path,required=True)
            s.add_argument('--video-id',required=True)
            s.add_argument('--language',default='en-orig')
        else:s.add_argument('--require-complete',action='store_true')
    a=p.parse_args(argv)
    if a.command=='broadcast':result=broadcast_pass1.run(a)
    elif a.command=='broadcast-match':result=broadcast_match.run(a)
    elif a.command=='prepare':result=prepare(a.run_dir,a.video,a.import_run,a.asset_cache,a.scoreboard_skim,a.import_producer)
    elif a.command in {'pilot','run'}:result=execute(a.run_dir,pilot=a.command=='pilot',component=a.component,
        asset_cache=a.asset_cache,model_cache=a.model_cache,budget_approved=getattr(a,'budget_approved',False))
    elif a.command=='finalize':result=finalize(a.run_dir,draft=a.draft)
    elif a.command=='import-captions':
        from .pass1_youtube_captions import import_captions
        result=import_captions(a.run_dir,a.captions,video_id=a.video_id,language=a.language)
    elif a.command=='verify-asr':
        from .pass1_asr import verify_reference
        if read_json(a.reference).get('method') == 'youtube_auto_captions':
            from .pass1_asr_captions import verify_captions
            verify_reference = verify_captions
        with lock(a.run_dir.resolve(),'asr'):
            result=verify_reference(a.run_dir,a.reference)
    elif (a.run_dir/'pass1_broadcast.json').exists():
        result=broadcast_pass1.verify(a.run_dir)
        match=broadcast_match.verify_match(a.run_dir)
        if 'revision' in match:
            result.update(match=match, passed=result['passed'] and match['passed'],
                          errors=[*result['errors'], *match['errors']])
        if a.require_complete:
            result.update(passed=False, errors=[*result['errors'], 'Broadcast pilot has experimental coverage'])
    else:result=verify(a.run_dir,require_complete=a.require_complete)
    print(json.dumps(result,indent=2))
    return 0 if result.get('passed',True) else 2


if __name__=='__main__':
    raise SystemExit(main())
