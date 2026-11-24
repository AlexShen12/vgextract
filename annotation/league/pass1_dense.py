"""Resumable native HUD-only sampling, sharing the existing CPU recognizers."""
from pathlib import Path
import math
import time

from .core import atomic_json, digest, read_json, uid, write_rows
from .media import file_hash
from .pass1_timeline import ContextIndex


def unavailable_tail_samples(lower, upper, frames, last_pts):
    sampled = {math.floor((frame['time']-lower)*16+1e-6) for frame in frames}
    missing = [(i, lower+i/16) for i in range(math.ceil((upper-lower)*16)) if i not in sampled]
    if len(sampled) != len(frames) or any(t <= last_pts+1e-8 for _, t in missing):
        raise ValueError('Dense decode missed a scheduled sample before the last source frame')
    frame_id = frames[-1]['id']
    return [{'id':uid('pass1-unavailable-hud-sample', frame_id, t), 'time':t,
             'kind':'no_source_frame', 'reason':'requested sample follows the last video frame',
             'last_source_pts':last_pts, 'rates':[16, 4] if i%4==0 else [16],
             'evidence_ids':[frame_id]} for i,t in missing]


def run_dense(store, windows, asset_cache, clocks, *, icons=True):
    import numpy as np
    from .action_icons import AbilityIdentity
    from .action_pipeline import numeric_read
    from .cast_bars import AbilityCatalog, CastBarReader
    from .hybrid_intake import _panel_state
    from .intake import evidence_crop
    from .intake_profile import layout_matches, load_profile
    from .playback import PlaybackTracker
    from .scene_reference import sample_source

    manifest = read_json(store.path('manifest.json'))
    source = manifest['source']
    video = store.path(source['path']) if not Path(source['path']).is_absolute() else Path(source['path'])
    profile = load_profile(str(store.path('profiles/midbeast-pass1-1080p-v1.json')))
    action = read_json(store.path('profiles/midbeast-actions-1080p-v2.json'))
    cast = read_json(store.path('profiles/midbeast-cast-bars-1080p-v1.json'))
    root = store.path('producers/dense')
    root.mkdir(parents=True, exist_ok=True)
    asset_cache = Path(asset_cache)
    catalog = AbilityCatalog(asset_cache/'abilities/16.22.1/catalog.json')
    icon_manifest = asset_cache/'ability-icons-16.13.1/manifest.json'
    if not icon_manifest.exists():
        icon_manifest = asset_cache/'manifest.json'
    identity = AbilityIdentity(icon_manifest) if icons else None
    clock_index = ContextIndex(clocks)
    recipe = digest({'source': source['sha256'], 'profiles': [profile, action, cast],
                     'catalog': catalog.signature, 'icons': identity.signature if identity else None,
                     'clock': clocks, 'code': file_hash(Path(__file__))})
    output = {key: [] for key in ('frames', 'playback', 'selection', 'visibility', 'slots', 'casts', 'identities')}
    started = time.monotonic()
    cache_hits = 0
    unavailable = []
    for a, b in windows:
        for n in range(math.ceil((b-a)/10)):
            lower, upper = a+n*10, min(b, a+(n+1)*10)
            path = root/'chunks'/f'{recipe[:16]}-{lower:.6f}.json'
            if path.exists():
                chunk = read_json(path)
                if chunk['recipe'] != recipe or any(not store.path(p).is_file() for p in chunk['files']):
                    raise ValueError('Dense checkpoint evidence is incomplete')
                cache_hits += 1
            else:
                tracker = PlaybackTracker(source['sha256'], 16, scene_box=action['scene_box'])
                # Restore only adjacent checkpoints in this window.
                if n and previous_chunk.get('upper') == lower:
                    tracker.state = previous_chunk['tracker']
                epoch = previous_chunk['epoch'] if n else 0
                last_portrait = np.asarray(previous_chunk['portrait']) if n else None
                data = {key: [] for key in output}
                reader = CastBarReader(root, catalog, cast, cast.get('masks', []))
                owner = action['owners'][0]
                last_icon, pixels = None, {}
                files = set()
                for i, (pts, image) in enumerate(sample_source(video, lower, upper, 16)):
                    store.check_stop()
                    valid, _ = layout_matches(image, profile)
                    c = clock_index.at(pts)
                    clock = c.get('clock') if c else None
                    context = tracker.update(pts, clock, image, valid)
                    portrait = np.asarray(image.crop(owner['identity_box']).resize((24,24)), dtype=float)
                    if last_portrait is not None and np.abs(portrait-last_portrait).mean()/255 > .12:
                        epoch += 1
                        last_icon = None
                    last_portrait = portrait
                    frame_id = uid('pass1-hud-frame', source['sha256'], pts)
                    _, crop_path = evidence_crop(store, image.crop(action['hud_box']))
                    files.add(crop_path)
                    frame = {'id': frame_id, 'time': pts, 'path': crop_path, 'box': action['hud_box'],
                             'source_sha256': source['sha256'], 'source_frame': round(pts*source['fps']),
                             'presentation_timestamp': pts, 'layout_valid': valid}
                    data['frames'].append(frame)
                    scope = {'start': pts, 'end': min(pts+1/16, upper), **context,
                             'evidence_ids': [frame_id] + (c.get('evidence_ids', []) if c else [])}
                    data['playback'].append({'id': uid('pass1-playback', frame_id), **scope})
                    if identity and valid and (last_icon is None or i%4 == 0):
                        last_icon = identity.identify(image, owner)
                    champion = last_icon.get('champion') if last_icon and valid else None
                    data['selection'].append({'id': uid('pass1-selection', frame_id), **scope,
                        'owner_epoch': uid('selection-epoch', source['sha256'], a, epoch),
                        'champion': champion, 'owner_id': 'selected', 'review_status': 'unreviewed'})
                    data['identities'].append({'id': uid('pass1-kit', frame_id), 'time': pts,
                        'frame_ids': [frame_id], 'champion': champion, 'evidence': last_icon,
                        'review_status': 'unreviewed'}) if i%4 == 0 else None
                    if i%4 == 0:
                        status = _panel_state(image, profile, {'route':'scoreboard'}, valid)
                        _, board_path = evidence_crop(store, image.crop(profile['inventory']['panels']['scoreboard']))
                        files.add(board_path)
                        data['visibility'].append({'id':uid('pass1-scoreboard',frame_id),'time':pts,
                            'status':status,'path':board_path,'frame_id':frame_id,
                            'playback_segment_id':context['playback_segment_id'],
                            'continuity_verified':context['continuity_verified']})
                    cast_frame = {'id':frame_id,'time':pts,'layout_valid':valid,
                        'segment_id':context['playback_segment_id'], 'owner_epochs':{'selected':epoch}}
                    reading = reader.read(image, cast_frame, {'owner_id':'selected','hud_champion':champion})
                    for key in ('text_path','progress_path'):
                        reading[key] = str((root/reading[key]).relative_to(store.root))
                        files.add(reading[key])
                    reading['owner_epoch'] = data['selection'][-1]['owner_epoch']
                    data['casts'].append(reading)
                    for slot in owner['slots']:
                        if slot['kind'] != 'ability':
                            continue
                        rgb = np.asarray(image.crop(slot['box']).resize((24,24)),dtype=float)
                        change = float(np.abs(rgb-pixels[slot['id']]).mean()/255) if slot['id'] in pixels else 0
                        pixels[slot['id']] = rgb
                        raw = numeric_read(image, slot['countdown_box'], root/'ocr', .85) if valid else {
                            'seconds':None,'score':0,'raw_text':''}
                        seconds = raw.get('seconds')
                        data['slots'].append({'id':uid('pass1-slot',frame_id,slot['id']),
                            'frame_id':frame_id,'frame_ids':[frame_id], 'time':pts,'owner_id':'selected',
                            'owner_epoch':data['selection'][-1]['owner_epoch'],
                            'segment_id':context['playback_segment_id'], 'slot_id':slot['id'],
                            'slot_kind':'ability','state':'cooling_down' if seconds else 'unknown',
                            'seconds':seconds,'raw':{k:v for k,v in raw.items() if k!='processed_path'},
                            'pixel_change':change,'layout_valid':valid,'review_status':'unreviewed'})
                expected = math.ceil((upper-lower)*16)
                tail = []
                if len(data['frames']) != expected:
                    if upper != manifest['interval'][1] or not data['frames']:
                        raise ValueError(f'Dense decode incomplete: {len(data["frames"])}/{expected}')
                    last_pts = max(pts for pts, _ in sample_source(video, lower, upper, None))
                    tail = unavailable_tail_samples(lower, upper, data['frames'], last_pts)
                chunk = {'recipe':recipe,'upper':upper,'data':data,'tracker':tracker.state,
                         'epoch':epoch,'portrait':last_portrait.tolist(),'files':sorted(files),
                         'unavailable_samples':tail}
                atomic_json(path, chunk)
            previous_chunk = chunk
            unavailable.extend(chunk.get('unavailable_samples', []))
            for key in output:
                output[key].extend(chunk['data'][key])
    for key, rows in output.items():
        write_rows(root/(key+'.jsonl'), rows)
    report = {'recipe':recipe,'windows':windows,'frames':len(output['frames']),
              'expected_frames':sum(math.ceil((b-a)*16) for a,b in windows),
              'unavailable_samples':unavailable,
              'wall_seconds':time.monotonic()-started,'chunk_cache_hits':cache_hits,
              'experimental':True,'release_accepted':False,'status':'completed_experimental'}
    if report['frames'] + len(unavailable) != report['expected_frames']:
        raise ValueError('Dense report does not account for every scheduled sample')
    atomic_json(root/'report.json', report)
    return report
