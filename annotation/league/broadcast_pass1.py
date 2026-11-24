"""Opt-in Pass 1 entry point for profile-driven tournament broadcast intake."""

from collections import Counter
from pathlib import Path

from .broadcast_commentary import rank_commentary
from .broadcast_timeline import (analysis_windows, automatic_segments, commentary_links, overlap,
                                 replay_metrics, reviewed_segments)
from .broadcast_video import load_templates, scan_video, segment_video
from .core import Store, atomic_json, digest, finite, read_json, read_rows, uid, write_rows
from .media import file_hash, probe
from .pass1_timeline import ContextIndex
from .pass1_youtube_captions import caption_rows, estimate_caption_ends


TABLES = {'broadcast_segments', 'analysis_windows', 'commentary_timeline', 'commentary_candidates',
          'commentary_links', 'broadcast_hud', 'broadcast_events', 'evidence_frames'}


def add_arguments(parser):
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--assets', type=Path, help='Overlay reference images, relative to profile by default')
    parser.add_argument('--captions', type=Path, required=True)
    parser.add_argument('--clip', nargs=3, action='append', required=True,
                        metavar=('FILE', 'VOD_START', 'VOD_END'))
    parser.add_argument('--review', type=Path, help='Reviewed intervals, event links, and OCR labels')
    parser.add_argument('--screen-only', action='store_true', help='Produce proposals without running OCR')
    parser.add_argument('--workers', type=int, default=2)


def checked_intervals(rows, lower, upper):
    previous = lower
    for row in sorted(rows, key=lambda row: row['start']):
        start, end = row['start'], row['end']
        if not (finite(start) and finite(end) and previous <= start < end <= upper):
            raise ValueError('Expected positive, ordered, nonoverlapping source intervals')
        previous = end
        yield row


def stage_clips(root, specs, duration):
    from .pass1 import copy_file

    requested = [{'start': float(start), 'end': float(end), 'input': Path(path).resolve()}
                 for path, start, end in specs]
    clips = []
    for clip in checked_intervals(requested, 0, duration):
        source = clip['input']
        if source.suffix.lower() not in {'.mp4', '.webm'}:
            raise ValueError(f'Expected a complete MP4 or WebM excerpt: {source}')
        media = probe(source)
        start, end = clip['start'], clip['end']
        tolerance = max(.10, 2 / media['fps'])
        if abs(media['duration'] - (end - start)) > tolerance:
            raise ValueError(f'Excerpt duration {media["duration"]} differs from VOD span {start}–{end}')
        sha = file_hash(source)
        clip_id = uid('broadcast-clip', sha, start, end)
        path = f'source/{clip_id}{source.suffix.lower()}'
        copy_file(source, root / path)
        clips.append({**media, 'id': clip_id, 'path': path, 'sha256': sha,
                      'start': start, 'end': end, 'media_start': 0})
    return clips


def screen_clips(root, profile, assets, clips, phases):
    source_hashes = {name: file_hash(assets / name) for name in sorted(
        {rule['image'] for rule in profile['templates']})}
    recipe = digest({'clips': clips, 'profile': profile, 'assets': source_hashes,
                     'scanner': file_hash(Path(__file__).with_name('broadcast_video.py'))})
    cached = root / 'broadcast_scan.json'
    if cached.exists():
        value = read_json(cached)
        if value['recipe'] == recipe:
            return value

    templates = load_templates(profile, assets)
    scans, segments = [], []
    for clip in clips:
        start, end = clip['start'], clip['end']
        scan = scan_video(root / clip['path'], start, end - start, templates)
        shots = segment_video(scan)
        # Label on the declared extent after the decoder has checked coverage.
        if abs(scan['end'] - end) > max(.10, 2 / clip['fps']):
            raise ValueError(f'Screening did not cover the requested clip ending at {end}')
        shots[-1]['end'] = end
        cuts = sorted({bound for row in phases for bound in (row['start'], row['end'])})
        phase_index = ContextIndex(phases)
        for shot in shots:
            boundaries = [shot['start'], *(t for t in cuts if shot['start'] < t < shot['end']), shot['end']]
            for left, right in zip(boundaries, boundaries[1:]):
                segments.extend(automatic_segments([{**shot, 'start': left, 'end': right}],
                                                    scan['samples'], phase_index.at(left) or {}))
        scans.append({**scan, 'clip_id': clip['id']})
    value = {'recipe': recipe, 'template_hashes': source_hashes, 'clips': scans, 'segments': segments}
    atomic_json(cached, value)
    return value


def events_with_evidence(store, definitions, clips, phases, segments, video_id):
    from .intake import sampled_frames

    clip_index, phase_index, segment_index = map(ContextIndex, (clips, phases, segments))
    frames, events = [], []
    for event in definitions:
        event_id = uid('broadcast-event', video_id, event['game_id'], event['key'])
        refs = []
        for ref in event['observations']:
            target = ref['time']
            clip, phase, segment = clip_index.at(target), phase_index.at(target), segment_index.at(target)
            if not clip or not phase or phase['game_id'] != event['game_id']:
                continue
            pts = target - clip['start'] + clip['media_start']
            stop = clip['end'] - clip['start'] + clip['media_start']
            frame_time, image = next(sampled_frames(store.path(clip['path']), pts, stop, 1, targets=[pts]))
            frame_id = uid('broadcast-evidence', clip['sha256'], frame_time)
            path = f'evidence/{frame_id}.jpg'
            output = store.path(path)
            output.parent.mkdir(parents=True, exist_ok=True)
            if not output.exists():
                image.thumbnail((960, 540))
                image.save(output, quality=90)
            time = round(clip['start'] + frame_time - clip['media_start'], 6)
            frames.append({'id': frame_id, 'time': time, 'path': path, 'presentation_timestamp': frame_time,
                           'clip_id': clip['id'], 'source_sha256': clip['sha256']})
            refs.append({**ref, 'time': time, 'frame_id': frame_id, 'phase': phase['phase'],
                         'segment_id': segment['id'] if segment else None})
        if not refs:
            continue
        events.append({**event, 'id': event_id, 'kind': event['kind'], 'time': refs[0]['time'],
                       'observations': refs, 'evidence_ids': [ref['frame_id'] for ref in refs]})
    return events, list({row['id']: row for row in frames}.values())


def run(args):
    from .broadcast_ocr import read_hud
    from .pass1 import copy_file, lock

    root = args.run_dir.resolve()
    profile = read_json(args.profile)
    video_id, duration = profile['video_id'], profile['duration']
    lower, upper = profile.get('pilot', [0, duration])
    if not (finite(duration) and 0 <= lower < upper <= duration and args.workers > 0):
        raise ValueError('Invalid broadcast duration, pilot interval, or worker count')
    phases = list(checked_intervals(profile['phases'], lower, upper))
    review = read_json(args.review) if args.review else {}
    if review and review.get('video_id') != video_id:
        raise ValueError('Review belongs to a different broadcast')
    review_spans = list(checked_intervals(review.get('segments', []), lower, upper))
    assets = args.assets or args.profile.parent
    store = Store(root)
    with lock(root, 'broadcast'):
        if (root / 'pass1_manifest.json').exists() and not (root / 'pass1_broadcast.json').exists():
            raise ValueError('Select a dedicated broadcast run directory')
        clips = stage_clips(root, args.clip, duration)
        if any(not lower <= clip['start'] < clip['end'] <= upper for clip in clips):
            raise ValueError('Visual clips must be inside the pilot interval')
        caption_hash = file_hash(args.captions)
        caption_path = f'source/{caption_hash}.json3'
        copy_file(args.captions, root / caption_path)
        scan = screen_clips(root, profile, assets, clips, phases)
        automatic = scan['segments']
        segments = list(reviewed_segments(automatic, review_spans, video_id))
        events, evidence = events_with_evidence(store, review.get('events', []), clips, phases, segments, video_id)
        cues = estimate_caption_ends(caption_rows(read_json(args.captions), duration=duration,
            source_hash=None, track_hash=caption_hash, track_path=caption_path,
            video_id=video_id, language=profile.get('language', 'en-orig')), [])
        cues = [cue for cue in cues if cue['start'] < cue['end']]
        ranked = rank_commentary(cues, video_id, duration=duration, interval=(lower, upper),
                                 vocabulary=profile.get('vocabulary'))
        event_keys = {event['key'] for event in events}
        scoped_links = [link for link in review.get('links', [])
                        if (not link.get('event') or link['event'] in event_keys)]
        links = commentary_links(ranked, cues, events, scoped_links, phases, video_id)
        hud, benchmark = [], {}
        if not args.screen_only:
            hud, benchmark = read_hud(store, clips, segments, profile,
                                      review.get('ocr_labels', []), args.workers)
        windows = analysis_windows(segments, events, links, video_id)
        outputs = {'broadcast_segments': segments, 'analysis_windows': windows,
                   'commentary_timeline': cues, 'commentary_candidates': ranked,
                   'commentary_links': links, 'broadcast_hud': hud, 'broadcast_events': events,
                   'evidence_frames': evidence}
        for name, rows in outputs.items():
            write_rows(root / (name + '.jsonl'), rows)
        atomic_json(root / 'broadcast_profile.json', profile)
        if args.review:
            atomic_json(root / 'broadcast_review.json', review)
        coverage = Counter()
        for row in segments:
            coverage[row['broadcast_context']] += row['end'] - row['start']
        metadata = {'schema_version': 1, 'video_id': video_id, 'title': profile.get('title', video_id),
                    'duration': duration, 'pilot': [lower, upper], 'clips': clips,
                    'caption_path': caption_path, 'caption_sha256': caption_hash,
                    'profile_sha256': file_hash(args.profile), 'template_hashes': scan['template_hashes'],
                    'review_sha256': file_hash(args.review) if args.review else None}
        report = {'status': 'completed_experimental', 'screen_only': args.screen_only,
                  'visual_seconds': sum(coverage.values()), 'coverage_seconds': dict(coverage),
                  'visual_coverage_complete': abs(sum(coverage.values()) - (upper - lower)) < 1e-6,
                  'reviewed_visual_seconds': sum(row['end'] - row['start'] for row in segments
                                                if row['review_status'] == 'reviewed'),
                  'counts': {key: len(rows) for key, rows in outputs.items()},
                  'reviewed_commentary_links': sum(row['review_status'] == 'reviewed' for row in links),
                  'replay': replay_metrics(automatic, review_spans), 'ocr': benchmark,
                  'input_hashes': {**{clip['path']: clip['sha256'] for clip in clips},
                                   caption_path: caption_hash},
                  'output_hashes': {name: file_hash(root / name) for name in
                                    [*(key + '.jsonl' for key in outputs), 'broadcast_profile.json',
                                     'broadcast_scan.json', *(['broadcast_review.json'] if args.review else [])]}}
        atomic_json(root / 'pass1_report.json', report)
        atomic_json(root / 'pass1_broadcast.json', metadata)
        return report


def verify(root):
    report = read_json(root / 'pass1_report.json')
    errors = []
    for relative, sha in (report['input_hashes'] | report['output_hashes']).items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file() or file_hash(path) != sha:
            errors.append(f'Checksum mismatch in {relative}')
    segments = read_rows(root / 'broadcast_segments.jsonl')
    by_segment = {row['id']: row for row in segments}
    cues = {row['id'] for row in read_rows(root / 'commentary_timeline.jsonl')}
    events = {row['id'] for row in read_rows(root / 'broadcast_events.jsonl')}
    frames = {row['id']: row for row in read_rows(root / 'evidence_frames.jsonl')}
    for row in read_rows(root / 'commentary_links.jsonl'):
        if (not set(row['cue_ids']) <= cues or
                (row['review_status'] == 'reviewed' and
                 (row['event_id'] not in events or not row['evidence_ids']
                  or not set(row['evidence_ids']) <= frames.keys()))):
            errors.append(f'Broken commentary references in {row["id"]}')
    for row in [*frames.values(), *read_rows(root / 'broadcast_hud.jsonl')]:
        relative = row.get('evidence_path', row.get('path'))
        if relative:
            path = (root / relative).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file():
                errors.append(f'Missing review evidence in {row["id"]}')
    for window in read_rows(root / 'analysis_windows.jsonl'):
        segment = by_segment.get(window['segment_id'])
        expected = 'live' if window['purpose'] == 'live_gameplay' else 'replay'
        if (not segment or segment['review_status'] != 'reviewed' or segment['broadcast_context'] != expected
                or segment['visible_content'] != 'gameplay'
                or not segment['start'] <= window['start'] < window['end'] <= segment['end']
                or not set(window['event_ids']) <= events):
            errors.append(f'Invalid analysis window {window["id"]}')
    return {'passed': not errors, 'errors': errors, 'status': report['status']}
