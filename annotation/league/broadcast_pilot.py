"""Triage a tournament VOD with timestamped commentary and optional video excerpts."""

import argparse
from collections import Counter
from pathlib import Path

from .broadcast_commentary import rank_commentary, shortlist
from .broadcast_video import (label_runs, load_templates, read_template_frame, replay_links,
                              scan_video, segment_video)
from .core import atomic_json, finite, read_json, write_rows
from .media import file_hash
from .pass1_youtube_captions import caption_rows, estimate_caption_ends


def screening(args):
    if not finite(args.duration) or args.duration <= 0:
        raise ValueError('The full VOD duration must be positive')
    if bool(args.clips) != bool(args.profile):
        raise ValueError('Video screening requires both excerpts and a broadcast overlay profile')
    clips = sorted((float(start), float(end), Path(path)) for path, start, end in args.clips)
    previous_end = 0
    for start, end, _ in clips:
        if not previous_end <= start < end <= args.duration:
            raise ValueError('Clip times must be nonoverlapping intervals within the full VOD')
        previous_end = end

    caption_hash = file_hash(args.captions)
    cues = estimate_caption_ends(caption_rows(read_json(args.captions), duration=args.duration,
        source_hash=None, track_hash=caption_hash, track_path=str(args.captions.resolve()),
        video_id=args.video_id, language=args.language), [])
    cues = [cue for cue in cues if cue['start'] < cue['end']]
    ranked = rank_commentary(cues, args.video_id, duration=args.duration)

    visual, runs = [], []
    if clips:
        from scenedetect import __version__ as scenedetect_version

        profile = read_json(args.profile)
        if profile.get('video_id') not in (None, args.video_id):
            raise ValueError('The overlay profile identifies a different VOD')
        templates = load_templates(profile, args.profile.parent)
        screenshot_path = profile.get('gameplay_screenshot')
        screenshot = read_template_frame(args.profile.parent / screenshot_path) if screenshot_path else None
        for start, end, file in clips:
            scan = scan_video(file, start, end - start, templates, screenshot)
            shots = segment_video(scan, args.detector)
            clip_runs = label_runs(shots)
            runs.extend(clip_runs)
            visual.append({**scan, 'path': str(file.resolve()), 'sha256': file_hash(file),
                           'shots': shots, 'runs': clip_runs, 'replay_links': replay_links(scan, clip_runs)})
            print(f'{file.name}: {len(shots)} labeled shots, {len(clip_runs)} runs', flush=True)
    for row in ranked:
        coverage = Counter()
        for run in runs:
            overlap = min(row['end'], run['end']) - max(row['start'], run['start'])
            if overlap > 0:
                coverage[run['label']] += overlap
        row['visual_overlap_seconds'] = {label: round(value, 3) for label, value in sorted(coverage.items())}
        row['watch_url'] = f'https://www.youtube.com/watch?v={args.video_id}&t={row["start"]}s'

    write_rows(args.output / 'commentary.jsonl', cues)
    write_rows(args.output / 'commentary_candidates.jsonl', ranked)
    atomic_json(args.output / 'shortlist.json', shortlist(ranked, 30))
    if visual:
        atomic_json(args.output / 'visual.json', {'scenedetect_version': scenedetect_version,
            'segment_detector': args.detector, 'profile': profile, 'clips': visual})
        write_rows(args.output / 'live_gameplay_candidates.jsonl', [
            {**shot, 'video_id': args.video_id, 'clip_sha256': clip['sha256']}
            for clip in visual for shot in clip['shots'] if shot['live_gameplay_candidate']])
    report = {'status': 'completed_experimental', 'video_id': args.video_id,
              'duration': args.duration, 'caption_sha256': caption_hash,
              'caption_language': args.language, 'caption_cues': len(cues),
              'first_caption_start': cues[0]['start'] if cues else None,
              'last_caption_end': max((cue['end'] for cue in cues), default=None),
              'commentary_candidates': len(ranked),
              'visual_seconds': round(sum(clip['end'] - clip['start'] for clip in visual), 3),
              'visual_report': 'visual.json' if visual else None,
              'visual_label_policy': 'no_gameplay_hud needs review; an overlay can be missing in gameplay'}
    atomic_json(args.output / 'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video-id', required=True)
    parser.add_argument('--duration', type=float, required=True, help='Full VOD duration in seconds')
    parser.add_argument('--captions', type=Path, required=True, help='YouTube JSON3 caption track')
    parser.add_argument('--language', default='en-orig')
    parser.add_argument('--clip', metavar=('FILE', 'VOD_START', 'VOD_END'), nargs=3,
                        action='append', dest='clips', default=[], help='Exactly trimmed source-time excerpt')
    parser.add_argument('--profile', type=Path, help='Overlay templates and relative frame paths')
    parser.add_argument('--detector', default='content', choices=('content', 'adaptive'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(screening(args))


if __name__ == '__main__':
    main()
