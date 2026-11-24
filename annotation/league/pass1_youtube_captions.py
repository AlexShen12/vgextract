"""Import retained YouTube JSON3 captions as Pass 1 commentary."""
from pathlib import Path
from bisect import bisect_right

from .core import atomic_json, digest, finite, read_json, read_rows, uid, write_rows
from .media import file_hash


def caption_rows(document, *, duration, source_hash, track_hash, track_path, video_id, language):
    """Use caption word onsets and remove repeated timed rolling-display text."""
    events = document.get('events')
    if not isinstance(events, list) or not finite(duration) or duration <= 0:
        raise ValueError('Expected JSON3 events and a positive source duration')
    cues = []
    seen = set()
    for event_index, event in enumerate(events):
        segments = [(index, segment) for index, segment in enumerate(event.get('segs', []))
                    if segment.get('utf8', '').strip()]
        if not segments:
            continue
        start_ms = event.get('tStartMs')
        length_ms = event.get('dDurationMs')
        if not finite(start_ms) or start_ms < 0 or not finite(length_ms) or length_ms <= 0:
            raise ValueError(f'Invalid caption timing in event {event_index}')
        start = start_ms/1000
        if start >= duration:
            continue
        timed = all(index == 0 or 'tOffsetMs' in segment for index, segment in segments)
        words, parts = [], []
        previous = start
        for index, segment in segments:
            text = segment['utf8']
            if timed:
                offset = segment.get('tOffsetMs', 0)
                if not finite(offset) or offset < 0:
                    raise ValueError(f'Invalid word offset in event {event_index}')
                time = start + offset/1000
                if time < previous:
                    raise ValueError(f'Unordered word offsets in event {event_index}')
                previous = time
                if time >= duration:
                    continue
                key = (time, text.strip())
                if key in seen:
                    continue
                seen.add(key)
                words.append({'text': text.strip(), 'start': time, 'end': None,
                              'caption_segment_index': index})
            parts.append(text)
        text = ''.join(parts).strip()
        if not text:
            continue
        identity = (start, text)
        if not timed and identity in seen:
            continue
        seen.add(identity)
        end = min(duration, start + length_ms/1000)
        if words and words[-1]['start'] > end:
            raise ValueError(f'Word onset exceeds display interval in event {event_index}')
        cues.append({'id': uid('youtube-commentary', source_hash, track_hash, event_index),
                     'kind': 'commentary', 'start': words[0]['start'] if words else start,
                     'end': end, 'text': text, 'words': words,
                     'source_sha256': source_hash, 'transcript_source': 'youtube_auto_captions',
                     'video_id': video_id, 'language': language, 'caption_sha256': track_hash,
                     'path': track_path, 'caption_event_index': event_index,
                     'caption_display_start': start, 'caption_display_end': end,
                     'evidence_modality': 'automatic_captions',
                     'alignment': {'status': 'caption_word_onsets' if timed else 'caption_segment_times'},
                     'speaker': None, 'speaker_status': 'unverified',
                     'review_status': 'unreviewed', 'experimental': True})
    cues.sort(key=lambda row: (row['start'], row['caption_event_index']))
    for index, cue in enumerate(cues[:-1]):
        next_start = cues[index+1]['start']
        last_onset = cue['words'][-1]['start'] if cue['words'] else cue['start']
        if next_start > last_onset:
            cue['end'] = min(cue['end'], next_start)
    if not cues:
        raise ValueError('Caption track has no text within the source interval')
    return cues


def estimate_caption_ends(cues, coverage, evidence_path=None):
    """Trim caption tails using the speech span containing their last word onset."""
    speech = []
    for span in coverage:
        if span.get('status') not in {'transcribed', 'unresolved_speech'}:
            continue
        start, end = span['start'], span['end']
        if not finite(start) or not finite(end) or not 0 <= start < end:
            raise ValueError('Invalid speech coverage span')
        if speech and start < speech[-1][0]:
            raise ValueError('Speech coverage must be chronological')
        # Adjacent model chunks and brief pauses belong to one speech region.
        if speech and start <= speech[-1][1] + .15:
            speech[-1][1] = max(speech[-1][1], end)
        else:
            speech.append([start, end])
    starts = [span[0] for span in speech]
    result = []
    for original in cues:
        cue = dict(original)
        timing = {'estimated': True, 'method': 'caption_boundary',
                  'caption_boundary_end': cue['end']}
        if cue['words']:
            last_onset = cue['words'][-1]['start']
            index = bisect_right(starts, last_onset)-1
            if index >= 0 and last_onset < speech[index][1] < cue['end']:
                cue['end'] = speech[index][1]
                timing.update(method='vad_speech_end', speech_end=cue['end'])
                if evidence_path:
                    timing['evidence_path'] = evidence_path
        cue['end_timing'] = timing
        result.append(cue)
    return result


def import_captions(root, track, *, video_id, language):
    from .pass1 import copy_file, lock, register
    root, track = Path(root).resolve(), Path(track).resolve()
    manifest = read_json(root/'manifest.json')
    source = manifest['source']
    pinned = read_json(Path(__file__).with_name('midbeast-source.json'))
    expected_id = source.get('video_id')
    if source['sha256'] == pinned['source_sha256']:
        expected_id = pinned['selected']['id']
    if expected_id != video_id:
        raise ValueError('Caption video ID does not match the pinned source recording')
    if not language.strip():
        raise ValueError('Caption language is required')
    coverage_path = None
    asr_path = root/'producers/asr/report.json'
    if asr_path.exists():
        asr = read_json(asr_path)
        if (asr.get('source_sha256') == source['sha256']
                and asr.get('status') == 'completed_experimental'
                and asr.get('windows') == [manifest['interval']] and asr.get('coverage_path')):
            coverage_path = (root/asr['coverage_path']).resolve()
            if not coverage_path.is_relative_to(root) or not coverage_path.is_file():
                raise ValueError('Speech coverage must be inside the bundle')
    coverage_hash = file_hash(coverage_path) if coverage_path else None
    track_hash = file_hash(track)
    recipe = digest({'source': source['sha256'], 'track': track_hash, 'video_id': video_id,
                     'language': language, 'speech_coverage': coverage_hash,
                     'implementation': file_hash(Path(__file__))})
    relative = Path('producers/youtube_captions')/recipe[:20]
    directory = root/relative
    rows = caption_rows(read_json(track), duration=manifest['interval'][1],
                        source_hash=source['sha256'], track_hash=track_hash,
                        track_path=str(relative/'track.json3'), video_id=video_id, language=language)
    coverage = read_rows(coverage_path) if coverage_path else []
    rows = estimate_caption_ends(rows, coverage, str(relative/'speech_coverage.jsonl') if coverage_path else None)
    report = {'status': 'completed_experimental', 'transcript_source': 'youtube_auto_captions',
              'source_sha256': source['sha256'], 'caption_sha256': track_hash,
              'video_id': video_id, 'language': language, 'recipe': recipe,
              'windows': [manifest['interval']], 'utterances': len(rows),
              'word_onsets': sum(len(row['words']) for row in rows),
              'first_caption_start': rows[0]['start'], 'last_caption_end': rows[-1]['end'],
              'commentary_path': str(relative/'commentary.jsonl'),
              'speech_coverage_sha256': coverage_hash,
              'vad_refined_ends': sum(row['end_timing']['method'] == 'vad_speech_end' for row in rows),
              'timing_policy': 'caption boundaries trimmed to available VAD speech ends; word ends unknown'}
    with lock(root, 'youtube_captions'):
        copy_file(track, directory/'track.json3')
        if coverage_path:
            copy_file(coverage_path, directory/'speech_coverage.jsonl')
        write_rows(directory/'commentary.jsonl', rows)
        atomic_json(directory/'report.json', report)
        register(root, 'youtube_captions', list(directory.iterdir()),
                 status=report['status'], windows=report['windows'], report=report,
                 source_sha256=source['sha256'])
    return report


def selected_commentary(root):
    """Prefer a registered YouTube track and retain compatibility with ASR bundles."""
    manifest = read_json(root/'pass1_manifest.json')
    youtube = manifest.get('producers', {}).get('youtube_captions', {}).get('report')
    if youtube:
        return youtube
    path = root/'producers/asr/report.json'
    return read_json(path) if path.exists() else {'status': 'not_run'}
