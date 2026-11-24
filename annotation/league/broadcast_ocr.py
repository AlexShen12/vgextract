"""Native spectator HUD crops and held-out OCR scoring."""

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import pairwise
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock

from .core import uid
from .intake import evidence_crop, sampled_frames
from .numeric_reader import NumericReader
from .pass1_timeline import ContextIndex


_CROP_LOCK = Lock()


def grayscale(image):
    from PIL import Image, ImageOps

    luminance = image.convert('L')
    glyphs = luminance.point(lambda value: 255 if value > 200 else 0).getbbox()
    if glyphs is None:
        return Image.new('L', (48, 48), 255)
    text = luminance.crop(glyphs)
    pixels = text.resize((max(1, int(text.width * 48 / text.height)), 48), Image.Resampling.BICUBIC)
    mask = pixels.point(lambda value: 0 if value > 200 else 255)
    return ImageOps.expand(mask, border=10, fill=255)


def native_box(image, box):
    return tuple(round(value * scale) for value, scale in zip(box, (*image.size, *image.size)))


def icon_and_text(image, crop, icon_box):
    import numpy as np

    icon = np.asarray(image.crop(native_box(image, icon_box)), dtype=float)
    color = (icon.max(axis=2) - icon.min(axis=2) > 35).mean()
    bright = (np.asarray(crop.convert('L')) > 145).mean()
    return color > .14, bright > .025


def compact_objectives(image, box):
    import numpy as np

    probe = np.asarray(image.crop(native_box(image, box)), dtype=float)
    return (probe.max(axis=2) - probe.min(axis=2) > 45).mean() > .24


def read_field(store, image, field, base, methods):
    layout = base['layout']
    compact = base['compact_objectives'] and field.get('compact')
    box = field['boxes'].get(layout)
    if box is None:
        return {**base, 'visibility': 'hidden', 'attempts': {}}
    coords = native_box(image, compact.get('box', box) if compact else box)
    crop = image.crop(coords)
    with _CROP_LOCK:
        sha, path = evidence_crop(store, crop)
    ico = compact.get('icon_box', field.get('icon_box')) if compact else field.get('icon_box')
    if ico:
        has_icon, has_text = icon_and_text(image, crop, ico)
        if compact and compact.get('icon_only'):
            has_text = False
        if not has_text:
            return {**base, 'visibility': 'icon_only' if has_icon else 'hidden',
                    'box': coords, 'evidence_path': path, 'attempts': {}}

    attempts = {}
    with TemporaryDirectory(prefix='broadcast-ocr-') as directory:
        for name, reader in methods.items():
            result = reader.read(crop, field, Path(directory) / name)
            raw = result.get('raw_text')
            confidence = result['attempts'][0].get('score', 0) if result['attempts'] else 0
            attempts[name] = {'raw_text': raw, 'value': result.get('value'), 'confidence': confidence,
                              'acceptance': result['acceptance'], 'reason': result['reason']}
    return {**base, 'visibility': 'visible' if any(row['value'] is not None for row in attempts.values())
            else 'unreadable', 'box': coords, 'evidence_path': path, 'crop_sha256': sha, 'attempts': attempts}


def evaluate_ocr(rows, labels):
    def summarize(counts):
        accepted, readable, correct = (counts[key] for key in ('accepted', 'readable', 'correct'))
        return {**counts, 'precision': correct / accepted if accepted else None,
                'readable_coverage': counts['readable_accepted'] / readable if readable else None,
                'exact_read_accuracy': correct / readable if readable else None,
                'false_accepts': accepted - correct}

    counts = defaultdict(Counter)
    by_key = {(row['sample_time'], row['field_id']): row for row in rows}
    for label in labels:
        row = by_key.get((label['time'], label['field']))
        for method in ('baseline', 'grayscale'):
            tally = counts[label['split'], label['field'], method]
            expected = label.get('text')
            attempt = (row or {}).get('attempts', {}).get(method, {})
            accepted = attempt.get('acceptance') == 'accepted'
            tally.update({'samples': 1, 'readable': int(expected is not None),
                          'accepted': int(accepted),
                          'readable_accepted': int(accepted and expected is not None),
                          'correct': int(accepted and expected is not None
                                         and attempt['raw_text'].casefold() == expected.casefold())})
    scores = [{'split': split, 'field_id': field, 'method': method, **summarize(tally)}
              for (split, field, method), tally in sorted(counts.items())]
    calibration = defaultdict(list)
    for score in scores:
        if score['split'] == 'calibration':
            calibration[score['field_id']].append(score)
    selected = {field: max(options, key=lambda row: (row['exact_read_accuracy'] or 0,
                row['precision'] or 0, row['method'] == 'grayscale'))['method']
                for field, options in calibration.items()}
    enabled = {row['field_id'] for row in scores if row['split'] == 'heldout'
               and selected.get(row['field_id']) == row['method'] and row['readable'] >= 10
               and (row['precision'] or 0) >= .95 and (row['readable_coverage'] or 0) >= .70}
    return {'scores': scores, 'selected_methods': selected, 'enabled_fields': sorted(enabled)}


def mark_effective(rows, benchmark):
    usable = set(benchmark['enabled_fields'])
    for row in rows:
        chosen = benchmark['selected_methods'].get(row['field_id'], 'grayscale')
        reading = row['attempts'].get(chosen, {})
        row.update(method=chosen, raw_text=reading.get('raw_text'), parsed_value=reading.get('value'),
                   confidence=reading.get('confidence', 0), acceptance='review_only',
                   review_status='unreviewed', continuity_corroborated=False)
        if (row['field_id'] in usable and row['layout'] == 'live' and row['segment_id']
                and reading.get('acceptance') == 'accepted'):
            row['acceptance'] = 'accepted'

    # Consecutive readings only corroborate when the same reviewed live span covers both samples.
    groups = defaultdict(list)
    for row in rows:
        if row['field_id'] == 'game_clock' and row['segment_id']:
            groups[row['segment_id']].append(row)
    for group in groups.values():
        for left, right in pairwise(group):
            if left['acceptance'] != 'accepted' or right['acceptance'] != 'accepted':
                continue
            elapsed = right['time'] - left['time']
            advance = right['parsed_value']['seconds'] - left['parsed_value']['seconds']
            if 0 < elapsed <= 2.1 and advance > 0 and abs(advance - elapsed) <= 1:
                left['continuity_corroborated'] = right['continuity_corroborated'] = True
        for row in group:
            if not row['continuity_corroborated']:
                row['acceptance'] = 'review_only'


def read_hud(store, clips, segments, profile, labels, workers):
    from math import ceil

    index = ContextIndex(segments)
    labels_by_time = {row['time'] for row in labels}
    baseline = NumericReader()
    adapted = NumericReader(preprocess=grayscale, alternate=True)
    fields = profile['hud_fields']
    compact_box = profile.get('objective_compact_region')
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for clip in clips:
            start, end = clip['start'], clip['end']
            # Sparse samples across the reviewed coverage, plus all audit frames.
            targets = sorted({*range(ceil(start), ceil(end), 2),
                              *(t for t in labels_by_time if start <= t < end)})
            target_pts = [time - start + clip['media_start'] for time in targets]
            source = store.path(clip['path'])
            frames = sampled_frames(source, clip['media_start'], clip['media_start'] + end - start,
                                    .5, targets=target_pts)
            for target, (pts, image) in zip(targets, frames, strict=True):
                time = round(start + pts - clip['media_start'], 6)
                segment = index.at(time)
                layout = 'hidden'
                if segment and segment['broadcast_context'] == 'live':
                    layout = 'live'
                elif segment and segment.get('layout') == 'breakdown':
                    layout = 'breakdown'
                fields_to_read = []
                compact = bool(layout == 'live' and compact_box and compact_objectives(image, compact_box))
                for field in fields:
                    fields_to_read.append((field, {
                        'id': uid('broadcast-hud', clip['sha256'], target, field['id']),
                        'kind': 'broadcast_hud_reading', 'time': time, 'sample_time': target,
                        'presentation_timestamp': pts, 'source_sha256': clip['sha256'],
                        'field': field['field'], 'field_id': field['id'], 'layout': layout,
                        'compact_objectives': compact,
                        'game_id': segment['game_id'] if segment else None,
                        'segment_id': segment['id'] if segment and segment['review_status'] == 'reviewed' else None,
                    }))
                methods = {'baseline': baseline, 'grayscale': adapted}
                rows.extend(pool.map(lambda item: read_field(store, image, *item, methods), fields_to_read))
    benchmark = evaluate_ocr(rows, labels)
    mark_effective(rows, benchmark)
    return rows, benchmark
