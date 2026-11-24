"""Source-aligned commentary producer and its ten-second development gate."""
from __future__ import annotations

import math
import os
from pathlib import Path
import subprocess
import time
import wave

from .core import Store, atomic_json, digest, finite, read_json, read_rows, uid, write_rows
from .media import file_hash

PILOT = [30.0, 40.0]


def audio_clip(video, path, start, end):
    """Decode exactly the requested source-time interval to mono PCM16."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.writing.wav')
    subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-y', '-ss', str(start),
                    '-i', str(video), '-t', str(end-start), '-map', '0:a:0',
                    '-vn', '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le',
                    str(temporary)], check=True)
    with wave.open(str(temporary), 'rb') as source:
        actual = source.getnframes()/source.getframerate()
    if abs(actual-(end-start)) > 1/16000 + 1e-6:
        temporary.unlink()
        raise ValueError(f'Decoded audio duration {actual} differs from requested {end-start}')
    temporary.replace(path)


def commentary_rows(result, source_hash, root, recipe):
    rows = []
    for chunk in result['chunks']:
        lower, upper = chunk['source_start'], chunk['source_end']
        if not finite(lower) or not finite(upper) or lower >= upper:
            raise ValueError('Invalid ASR source span')
        words = chunk.get('words', [])
        previous = lower
        for word in words:
            if not (finite(word['start']) and finite(word['end']) and
                    lower <= word['start'] <= word['end'] <= upper and word['start'] >= previous):
                raise ValueError('Invalid ASR word alignment')
            previous = word['start']
        aligned = chunk.get('alignment', {}).get('status') == 'aligned' and bool(words)
        start, end = (words[0]['start'], words[-1]['end']) if aligned else (lower, upper)
        path = Path(chunk['path']).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError('ASR evidence must be inside the portable bundle')
        rows.append({'id': uid('commentary', source_hash, recipe, lower, upper),
                     'kind': 'commentary', 'start': start, 'end': end,
                     'text': chunk.get('text'), 'words': words,
                     'source_start': lower, 'source_end': upper,
                     'source_sha256': source_hash, 'path': str(path.relative_to(root.resolve())),
                     'audio_sha256': file_hash(path), 'evidence_modality': 'audio',
                     'speaker': None, 'expected_speaker': 'Midbeast', 'speaker_status': 'unverified',
                     'transcription_status': chunk.get('status', 'unresolved'),
                     'alignment': chunk.get('alignment', {'status': 'unavailable'}),
                     'review_status': 'unreviewed', 'experimental': True})
    return sorted(rows, key=lambda row: (row['start'], row['id']))


def audio_coverage(rows, interval):
    """Account for detected speech and VAD-negative spans, including failures."""
    start, end = interval
    cursor = start
    spans = []
    for row in sorted(rows, key=lambda r: r['source_start']):
        lower, upper = row['source_start'], row['source_end']
        if lower < cursor-1e-6 or upper > end+1e-6 or lower < start:
            raise ValueError('Overlapping or out-of-bounds ASR coverage')
        if lower > cursor:
            spans.append({'start': cursor, 'end': lower, 'status': 'vad_no_speech'})
        resolved = row.get('transcription_status') == 'transcribed' and bool(row.get('text'))
        spans.append({'start': lower, 'end': upper,
                      'status': 'transcribed' if resolved else 'unresolved_speech',
                      'commentary_id': row['id'], 'alignment_status': row['alignment']['status']})
        cursor = upper
    if cursor < end:
        spans.append({'start': cursor, 'end': end, 'status': 'vad_no_speech'})
    return spans


def map_commentary(rows, playback, frame_tolerance=1/30):
    """Bind an utterance only when one verified playback segment covers it."""
    from bisect import bisect_left
    timeline = sorted(playback, key=lambda p: p['start'])
    starts = [p['start'] for p in timeline]
    result = []
    for original in rows:
        row = dict(original)
        first = max(0, bisect_left(starts, row['start'])-1)
        candidates = []
        for span in timeline[first:]:
            if span['start'] >= row['end']:
                break
            if span['end'] > row['start']:
                candidates.append(span)
        segments = sorted({p['playback_segment_id'] for p in candidates})
        cursor = row['start']
        covered = bool(candidates)
        for span in candidates:
            if span['start'] > cursor+frame_tolerance:
                covered = False
            cursor = max(cursor, span['end'])
        verified = (covered and cursor >= row['end']-frame_tolerance and len(segments) == 1
                    and all(p.get('continuity_verified') for p in candidates))
        row.update(playback_segment_id=segments[0] if verified else None,
                   playback_segment_ids=segments, continuity_verified=verified,
                   playback_context_status='known' if verified else 'unknown_or_crosses_boundary')
        result.append(row)
    return result


def verify_reference(root, reference_path):
    """Compare a source-listening reference with the immutable pilot prediction."""
    from .experiments import word_error_rate
    root = Path(root).resolve()
    producer = root/'producers/asr'
    report = read_json(producer/'report.json')
    pilot = root/report['pilot_report_path']
    data = read_json(pilot)
    reference = read_json(Path(reference_path))
    if (reference.get('source_sha256') != data['source_sha256'] or reference.get('interval') != PILOT
            or reference.get('method') != 'source_listening' or not reference.get('reviewer')
            or not isinstance(reference.get('transcript'), str) or not reference['transcript'].strip()):
        raise ValueError('Reference requires source hash, [30,40], source_listening, reviewer and transcript')
    rows = read_rows(pilot.parent/'commentary.jsonl')
    actual = ' '.join(row.get('text') or '' for row in rows)
    metrics = word_error_rate(reference['transcript'], actual)
    anchors = reference.get('words', [])
    predicted = [w for r in rows for w in r['words']]
    used = set()
    errors = []
    for anchor in anchors:
        if not finite(anchor.get('start')) or not PILOT[0] <= anchor['start'] < PILOT[1]:
            raise ValueError('Reference word onsets must be in source-video seconds')
        matches = [(abs(w['start']-anchor['start']), i) for i, w in enumerate(predicted)
                   if i not in used and w['text'].strip('.,!?').casefold() == anchor['text'].casefold()]
        error, index = min(matches, default=(None, None))
        errors.append(error)
        used.add(index)
    names = reference.get('names', [])
    missing_names = [name for name in names if name.casefold() not in actual.casefold()]
    checks = {'word_error_rate': metrics['wer'] is not None and metrics['wer'] <= .15,
              'league_terms': not missing_names,
              'phrase_review': reference.get('no_invented_or_skipped_phrases') is True,
              'speaker_review': reference.get('speaker') == 'Midbeast',
              'word_onsets': len(errors) >= 3 and all(e is not None and e <= .35 for e in errors),
              'resources': data.get('resource_passed', False),
              'alignment': bool(rows) and all(r['alignment']['status'] == 'aligned' for r in rows)}
    result = {'passed': all(checks.values()), 'checks': checks, 'metrics': metrics,
              'word_onset_errors_seconds': errors, 'missing_names': missing_names,
              'source_sha256': data['source_sha256'], 'interval': PILOT,
              'pilot_recipe': data['recipe'], 'pilot_result_sha256': file_hash(pilot.parent/'result.json'),
              'clip_sha256': file_hash(pilot.parent/'clip.wav'), 'reference_sha256': digest(reference),
              'reviewer': reference['reviewer'], 'method': 'source_listening'}
    atomic_json(pilot.parent/'reference.json', reference)
    atomic_json(pilot.parent/'verification.json', result)
    report['pilot_verification'] = result
    atomic_json(producer/'report.json', report)
    publish(root, report)
    return result


def require_pilot(root):
    report = read_json(root/'producers/asr/report.json')
    pilot = root/report['pilot_report_path']
    verification = read_json(pilot.parent/'verification.json')
    if not (verification.get('passed') and verification['pilot_result_sha256'] == file_hash(pilot.parent/'result.json')
            and verification['clip_sha256'] == file_hash(pilot.parent/'clip.wav')
            and verification['reference_sha256'] == digest(read_json(pilot.parent/'reference.json'))
            and verification['source_sha256'] == read_json(root/'manifest.json')['source']['sha256']):
        raise ValueError('ASR requires a passing source-verified pilot before the full run')
    return report


def publish(root, report):
    from .pass1 import register
    producer = root/'producers/asr'
    register(root, 'asr', [p for p in producer.rglob('*') if p.is_file()],
             status='completed_experimental', source_sha256=report['source_sha256'],
             windows=report['windows'], report=report, implementation=file_hash(Path(__file__)))


def run(root, *, pilot, model_cache):
    from .experiments import run_worker
    root = Path(root).resolve()
    started = time.monotonic()
    manifest = read_json(root/'manifest.json')
    source_hash = manifest['source']['sha256']
    interval = PILOT if pilot else [0.0, manifest['interval'][1]]
    if manifest['interval'][1] < PILOT[1]:
        raise ValueError('The ASR pilot requires source seconds 30–40')
    previous = {} if pilot else require_pilot(root)
    video = root/manifest['source']['path']
    if file_hash(video) != source_hash:
        raise ValueError('ASR source checksum changed')
    python = os.environ.get('LEAGUE_AUDIO_PYTHON')
    if not python or model_cache is None:
        raise ValueError('Set LEAGUE_AUDIO_PYTHON and --model-cache for the isolated ASR worker')
    from .local_models import LOCK
    recipe = digest({'source': source_hash, 'interval': interval,
                     'models': {k: v for k, v in read_json(LOCK)['models'].items() if k in {'asr', 'aligner'}},
                     'producer': file_hash(Path(__file__)),
                     'worker': file_hash(Path(__file__).with_name('experiment_worker.py'))})
    if not pilot and read_json(root/previous['pilot_report_path'])['recipe'] != digest({'source': source_hash, 'interval': PILOT,
                     'models': {k: v for k, v in read_json(LOCK)['models'].items() if k in {'asr', 'aligner'}},
                     'producer': file_hash(Path(__file__)),
                     'worker': file_hash(Path(__file__).with_name('experiment_worker.py'))}):
        raise ValueError('ASR implementation changed since the verified pilot')
    directory = root/'producers/asr'/('pilot' if pilot else 'full')/recipe[:20]
    directory.mkdir(parents=True, exist_ok=True)
    clip = directory/'clip.wav'
    if not clip.exists():
        audio_clip(video, clip, *interval)
    store = Store(root)
    store.install_signals()
    result = run_worker(store, python,
                        {'kind': 'audio', 'cache': str(Path(model_cache).resolve()), 'resume': True,
                         'timeout_seconds': 24*3600,
                         'segments': [{'id': 'pilot' if pilot else 'full', 'path': str(clip),
                                       'start': interval[0], 'end': interval[1]}]}, directory)
    rows = commentary_rows(result, source_hash, root, recipe)
    coverage = audio_coverage(rows, interval)
    write_rows(directory/'commentary.jsonl', rows)
    write_rows(directory/'coverage.jsonl', coverage)
    report = {'source_sha256': source_hash, 'recipe': recipe, 'windows': [interval],
              'status': 'completed_experimental', 'scope': 'pilot' if pilot else 'full_video',
              'frames': len(rows), 'wall_seconds': result.get('wall_seconds', time.monotonic()-started),
              'gpu_memory_peak_bytes': result.get('resource_measurement', {}).get('peak_memory_bytes', 0),
              'resource_passed': result.get('resource_passed', False),
              'speech_seconds': sum(r['source_end']-r['source_start'] for r in rows),
              'unresolved_speech_seconds': sum(s['end']-s['start'] for s in coverage if s['status']=='unresolved_speech'),
              'vad_no_speech_seconds': sum(s['end']-s['start'] for s in coverage if s['status']=='vad_no_speech'),
              'unaligned_utterances': sum(r['alignment']['status'] != 'aligned' for r in rows),
              'commentary_path': str((directory/'commentary.jsonl').relative_to(root)),
              'coverage_path': str((directory/'coverage.jsonl').relative_to(root)),
              'pilot_report_path': previous.get('pilot_report_path', str((directory/'report.json').relative_to(root))),
              'pilot_verification': previous.get('pilot_verification', {'passed': False, 'status': 'pending_source_listening'}),
              'asr_model': result['asr_model'], 'aligner_model': result['aligner_model'],
              'runtime': result.get('runtime', {}), 'gpu': result.get('gpu')}
    if pilot:
        report['projected_full_wall_seconds_with_headroom'] = math.ceil(report['wall_seconds']*manifest['interval'][1]/10*1.3)
    atomic_json(directory/'report.json', report)
    atomic_json(root/'producers/asr/report.json', report)
    publish(root, report)
    return report
