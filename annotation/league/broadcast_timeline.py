"""Conservative broadcast segments and editable evidence links on VOD time."""

import re
from collections import Counter

from .core import uid
from .pass1_timeline import ContextIndex


def overlap(left, right):
    return max(0, min(left['end'], right['end']) - max(left['start'], right['start']))


def automatic_segments(shots, samples, phase):
    """Keep shot content separate from the broadcast replay context."""
    segments = []
    for shot in shots:
        label = shot['label']
        support = [row for row in samples if shot['start'] <= row['time'] < shot['end']]
        key = phase.get('phase', 'unknown')
        if label == 'live_gameplay' and key == 'game':
            context, content = 'live', 'gameplay'
        elif label == 'replay':
            context = 'replay'
            hud = sum(row.get('replay_gameplay_visible', False) for row in support)
            content = 'gameplay' if hud > len(support) / 2 else 'unknown'
        elif label == 'non_gameplay':
            context, content = 'non_match', 'desk'
        elif label == 'draft':
            context, content = 'non_match', 'draft'
        else:
            context, content = 'unknown', 'unknown'
        segments.append({**shot, 'game_id': phase.get('game_id'), 'phase': key,
                         'broadcast_context': context, 'visible_content': content,
                         'review_status': 'unreviewed'})
    return segments


def reviewed_segments(automatic, reviews, video_id):
    """Apply bounded review intervals while retaining the original proposal."""
    prediction = ContextIndex(automatic)
    corrections = ContextIndex(reviews)
    edges = sorted({row[key] for row in [*automatic, *reviews] for key in ('start', 'end')})
    for start, end in zip(edges, edges[1:]):
        estimate, reviewed = prediction.at(start), corrections.at(start)
        if estimate is None:
            continue
        row = dict(estimate)
        if reviewed:
            row.update(broadcast_context=reviewed['broadcast_context'],
                       visible_content=reviewed['visible_content'], review_status='reviewed')
            if 'layout' in reviewed:
                row['layout'] = reviewed['layout']
            if 'review_note' in reviewed:
                row['review_note'] = reviewed['review_note']
        yield {**row, 'id': uid('broadcast-segment', video_id, start, end),
               'start': start, 'end': end}


def replay_metrics(automatic, reviews):
    """Score labeled time only and bracket the transition error in seconds."""
    tally = Counter()
    for truth in reviews:
        expected = truth['broadcast_context'] == 'replay'
        for row in automatic:
            shared = overlap(truth, row)
            if shared:
                tally[expected, row['broadcast_context'] == 'replay'] += shared
    tp, fp, fn = tally[True, True], tally[False, True], tally[True, False]

    def boundaries(rows):
        result = []
        for row in rows:
            if row['broadcast_context'] != 'replay':
                continue
            if result and result[-1][1] == row['start']:
                result[-1][1] = row['end']
            else:
                result.append([row['start'], row['end']])
        return result

    automatic_replays = boundaries(automatic)
    errors = []
    for start, end in boundaries(reviews):
        intersecting = [row for row in automatic_replays if row[0] < end and start < row[1]]
        errors.append({'start': start, 'end': end,
                       'start_error': intersecting[0][0] - start if intersecting else None,
                       'end_error': intersecting[-1][1] - end if intersecting else None})
    return {'precision': tp / (tp + fp) if tp + fp else None,
            'recall': tp / (tp + fn) if tp + fn else None,
            'true_replay_seconds': tp, 'false_replay_seconds': fp, 'missed_replay_seconds': fn,
            'reviewed_seconds': sum(tally.values()), 'boundary_errors': errors}


def commentary_links(candidates, cues, events, reviews, phases, video_id):
    by_cue = {cue['id']: cue for cue in cues}
    by_event = {event['key']: event for event in events}
    phase_index = ContextIndex(phases)
    links = []
    for row in candidates:
        phase = phase_index.at(row['start'])
        if phase is None:
            continue
        text = ' '.join(by_cue[key]['text'] for key in row['cue_ids']).casefold()
        matches = []
        for event in events:
            if event['game_id'] != phase['game_id']:
                continue
            for ref in event['observations']:
                if ref['phase'] != phase['phase'] or not row['start'] - 30 <= ref['time'] <= row['end'] + 30:
                    continue
                terms = sorted({term for term in event['entities']
                                if re.search(r'(?<!\w)' + re.escape(term.casefold()) + r'(?!\w)', text)})
                if terms:
                    matches.append((event['id'], terms))
                    break
        links.append({'id': uid('broadcast-link-candidate', row['id']), 'kind': 'commentary_link',
                      'start': row['start'], 'end': row['end'], 'cue_ids': row['cue_ids'],
                      'game_id': phase['game_id'], 'phase': phase['phase'],
                      'event_id': None, 'review_status': 'unresolved',
                      'candidates': [{'event_id': event_id, 'matched_terms': terms}
                                     for event_id, terms in matches]})
    for review in reviews:
        event = by_event.get(review.get('event'))
        start, end = review['start'], review['end']
        cue_ids = [cue['id'] for cue in cues if start <= cue['start'] < end]
        if not cue_ids or (review.get('event') and event is None):
            raise ValueError(f'Commentary review has an unknown event or no cues at {start}')
        phase = phase_index.at(start)
        if phase is None or (event and phase['game_id'] != event['game_id']):
            raise ValueError(f'Commentary review game scope mismatch at {start}')
        links.append({'id': uid('broadcast-reviewed-link', video_id, start, end, review.get('event')),
                      'kind': 'commentary_link', 'start': start, 'end': end, 'cue_ids': cue_ids,
                      'game_id': phase['game_id'], 'phase': phase['phase'], 'summary': review['summary'],
                      'event_id': event['id'] if event else None,
                      'evidence_ids': event['evidence_ids'] if event else [],
                      'review_status': 'reviewed' if event else 'unresolved'})
    return links


def analysis_windows(segments, events, links, video_id):
    linked = {link['event_id'] for link in links if link['review_status'] == 'reviewed'}
    windows = []
    for row in segments:
        if (row['review_status'] != 'reviewed' or row['broadcast_context'] != 'live'
                or row['visible_content'] != 'gameplay'):
            continue
        priority = [{'start': max(row['start'], ref['time'] - 12),
                     'end': min(row['end'], ref['time'] + 12), 'event_id': event['id']}
                    for event in events if event['id'] in linked and event['game_id'] == row['game_id']
                    for ref in event['observations'] if ref['role'] == 'live'
                    and row['start'] < ref['time'] + 12 and ref['time'] - 12 < row['end']]
        edges = sorted({row['start'], row['end'], *(item[key] for item in priority for key in ('start', 'end'))})
        for start, end in zip(edges, edges[1:]):
            event_ids = sorted({item['event_id'] for item in priority if item['start'] <= start < item['end']})
            windows.append({'id': uid('broadcast-analysis', row['id'], start, end), 'start': start,
                            'end': end, 'game_id': row['game_id'], 'purpose': 'live_gameplay',
                            'priority': 'commentary_linked' if event_ids else 'standard',
                            'event_ids': event_ids, 'segment_id': row['id']})
    index = ContextIndex(segments)
    for event in events:
        if event['id'] not in linked:
            continue
        for ref in event['observations']:
            row = index.at(ref['time'])
            if (ref['role'] != 'replay' or not row or row['review_status'] != 'reviewed'
                    or row['broadcast_context'] != 'replay' or row['visible_content'] != 'gameplay'):
                continue
            windows.append({'id': uid('broadcast-support', video_id, event['id'], ref['time']),
                            'start': max(row['start'], ref['time'] - 5),
                            'end': min(row['end'], ref['time'] + 5),
                            'game_id': row['game_id'], 'purpose': 'replay_evidence',
                            'priority': 'commentary_linked', 'event_ids': [event['id']],
                            'segment_id': row['id']})
    return sorted(windows, key=lambda row: (row['priority'] != 'commentary_linked', row['start']))
