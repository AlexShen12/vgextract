"""Verify the ASR pilot against a saved YouTube automatic-caption track."""
from pathlib import Path

from .core import atomic_json, digest, read_json, read_rows
from .media import file_hash
from .pass1_asr import PILOT, publish


def caption_words(document, interval=PILOT):
    words = []
    for event in document.get('events', []):
        start = event.get('tStartMs', 0)/1000
        for segment in event.get('segs', []):
            text = segment.get('utf8', '').strip()
            time = start + segment.get('tOffsetMs', 0)/1000
            if text and interval[0] <= time < interval[1]:
                words.append({'text': text, 'start': time})
    return sorted(words, key=lambda word: word['start'])


def verify_captions(root, reference_path):
    from .experiments import word_error_rate
    root = Path(root).resolve()
    producer = root/'producers/asr'
    report = read_json(producer/'report.json')
    pilot = root/report['pilot_report_path']
    data = read_json(pilot)
    reference = read_json(Path(reference_path))
    if (reference.get('method') != 'youtube_auto_captions'
            or reference.get('source_sha256') != data['source_sha256']
            or reference.get('interval') != PILOT or reference.get('video_id') != 'OLLZBe5-fg4'
            or not reference.get('reviewer')):
        raise ValueError('Caption reference must identify the pinned video, interval, method and reviewer')
    track = Path(reference['caption_path'])
    track = (track if track.is_absolute() else root/track).resolve()
    if file_hash(track) != reference.get('caption_sha256'):
        raise ValueError('Automatic caption checksum mismatch')
    words = caption_words(read_json(track))
    if not words:
        raise ValueError('Automatic captions do not cover the pilot')
    rows = read_rows(pilot.parent/'commentary.jsonl')
    actual = ' '.join(row.get('text') or '' for row in rows)
    transcript = ' '.join(word['text'] for word in words)
    metrics = word_error_rate(transcript, actual)
    predicted = [word for row in rows for word in row['words']]
    errors = []
    used = set()
    for anchor in reference.get('onset_words', []):
        source = next((word for word in words if word['text'].strip('.,!?').casefold() == anchor.casefold()), None)
        candidates = [(abs(word['start']-source['start']), index)
                      for index, word in enumerate(predicted)
                      if source and index not in used and word['text'].strip('.,!?').casefold() == anchor.casefold()]
        error, index = min(candidates, default=(None, None))
        used.add(index)
        errors.append(error)
    checks = {'word_error_rate': metrics['wer'] is not None and metrics['wer'] <= .15,
              'meaning_review': reference.get('meaning_matches') is True,
              'word_onsets': len(errors) >= 3 and all(e is not None and e <= .35 for e in errors),
              'resources': data.get('resource_passed', False),
              'alignment': bool(rows) and all(row['alignment']['status'] == 'aligned' for row in rows)}
    saved_track = pilot.parent/'youtube-auto-captions.json3'
    saved_track.write_bytes(track.read_bytes())
    reference = {**reference, 'caption_path': str(saved_track.relative_to(root)), 'transcript': transcript}
    result = {'passed': all(checks.values()), 'checks': checks, 'metrics': metrics,
              'word_onset_errors_seconds': errors, 'method': 'youtube_auto_captions',
              'speaker_verification': 'unverified', 'vocabulary_verification': 'pending_source_listening',
              'source_sha256': data['source_sha256'], 'interval': PILOT,
              'pilot_recipe': data['recipe'], 'pilot_result_sha256': file_hash(pilot.parent/'result.json'),
              'clip_sha256': file_hash(pilot.parent/'clip.wav'), 'reference_sha256': digest(reference),
              'caption_sha256': file_hash(saved_track), 'reviewer': reference['reviewer']}
    atomic_json(pilot.parent/'reference.json', reference)
    atomic_json(pilot.parent/'verification.json', result)
    report['pilot_verification'] = result
    atomic_json(producer/'report.json', report)
    publish(root, report)
    return result
