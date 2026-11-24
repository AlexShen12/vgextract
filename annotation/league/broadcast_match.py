"""Attach game-specific draft, roster, HUD and event records to a verified broadcast."""

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import math
from pathlib import Path
import time

from .broadcast_match_timeline import (FeedTracks, announcement_rows, associate_replays, bind_player,
                                       corroborate_kills, item_history, normalized_name, parse_center_kill,
                                       parse_center_objective, score_rows)
from .core import Store, atomic_json, digest, implementation_hash, read_json, read_rows, uid, write_rows
from .inventory_references import load_inventory_references, reference_root
from .media import file_hash
from .pass1_timeline import ContextIndex


TABLES = {'broadcast_draft', 'broadcast_roster', 'broadcast_player_snapshots', 'broadcast_match_events',
          'broadcast_item_history', 'broadcast_match_evidence'}
FPS = 4


def add_arguments(parser):
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--match-profile', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--asset-cache', type=Path, required=True)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--window', nargs=2, type=float, metavar=('START', 'END'), help='Bound a smoke run')


class Footage:
    def __init__(self, store, manifest, segments, destination):
        self.store = store
        self.clips = ContextIndex(manifest['clips'])
        self.segments = ContextIndex(segments)
        self.destination = destination
        self.frames = {}

    def frame(self, time, game, *, phase=None, context=None):
        from .intake import sampled_frames

        clip, segment = self.clips.at(time), self.segments.at(time)
        if (not clip or not segment or segment['review_status'] != 'reviewed' or segment['game_id'] != game
                or (phase and segment['phase'] != phase)
                or (context and (segment['broadcast_context'] != context or segment['visible_content'] != 'gameplay'))):
            raise ValueError(f'No verified {phase or context or game} evidence at source second {time}')
        pts = time - clip['start'] + clip['media_start']
        stop = clip['end'] - clip['start'] + clip['media_start']
        frame_pts, image = next(sampled_frames(self.store.path(clip['path']), pts, stop, 1, targets=[pts]))
        return clip, frame_pts, image

    def evidence(self, clip, pts, image, box):
        identifier = uid('broadcast-match-crop', clip['sha256'], pts, list(box))
        if identifier not in self.frames:
            path = self.destination / 'evidence' / (identifier + '.png')
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                image.crop(tuple(box)).save(path)
            row = {'id': identifier, 'time': round(clip['start'] + pts - clip['media_start'], 6),
                   'presentation_timestamp': pts, 'clip_id': clip['id'], 'source_sha256': clip['sha256'],
                   'box': list(box), 'path': str(path.relative_to(self.store.root)), 'sha256': file_hash(path)}
            self.frames[identifier] = row
        return self.frames[identifier]


def roster_and_draft(footage, review, profile, catalog, video_id):
    from .broadcast_match_readers import pil_box

    ids = {entry['id'] for entry in catalog['assets'] if entry['kind'] == 'champion'}
    rosters, draft = {}, []
    game_definitions = review['games']
    for game_id, game in game_definitions.items():
        draft_frame = footage.frame(game['final_draft_time'], game_id, phase='draft')
        score_frame = footage.frame(game['scoreboard_time'], game_id, context='live')
        if len(game['players']) != 10:
            raise ValueError(f'{game_id} roster needs ten player witnesses')
        rows = []
        for player in game['players']:
            side, row, champion = player['side'], player['row'], player['champion']
            if side not in ('blue', 'red') or not 0 <= row < 5 or champion not in ids:
                raise ValueError(f'Invalid roster witness in {game_id}')
            columns = profile['scoreboard']['columns'][side]
            start, end = columns['champion'], columns['name']
            row_box = (min(start[0], end[0]), min(start[1], end[1]),
                       max(start[2], end[2]), max(start[3], end[3]))
            row_box = pil_box(row_box, round(row * profile['scoreboard']['row_step']))
            draft_index = row if side == 'blue' else row + 5
            draft_crop = footage.evidence(*draft_frame, profile['draft']['picks'][draft_index])
            name_crop = footage.evidence(*score_frame, row_box)
            team = game['teams'][side]
            identifier = uid('broadcast-player', video_id, game_id, side, normalized_name(player['name'], team))
            rows.append({**player, 'id': identifier, 'game_id': game_id, 'team': team,
                         'time': name_crop['time'], 'kind': 'player_binding', 'review_status': 'reviewed',
                         'evidence_ids': [draft_crop['id'], name_crop['id']]})
        for side in ('blue', 'red'):
            members = [r for r in rows if r['side'] == side]
            if (len(members) != 5 or {r['row'] for r in members} != set(range(5))
                    or len({r['champion'] for r in members}) != 5
                    or len({r['id'] for r in members}) != 5):
                raise ValueError(f'Duplicate roster bindings in {game_id} {side}')
        rosters[game_id] = rows
        for row in game.get('draft', []):
            state = row['state']
            if state not in {'locked', 'hover', 'prior_game_exclusion'} or row['champion'] not in ids:
                raise ValueError(f'Invalid draft decision in {game_id}')
            frame = footage.frame(row['time'], game_id, phase='draft')
            proof = footage.evidence(*frame, row.get('box', profile['draft']['panel_box']))
            draft.append({**row, 'id': uid('broadcast-draft-decision', video_id, game_id, row),
                          'game_id': game_id, 'kind': 'draft_decision', 'review_status': 'reviewed',
                          'time': proof['time'], 'evidence_ids': [proof['id']]})
    return rosters, sorted(draft, key=lambda row: (row['game_id'], row.get('order', 0) or 0, row['time']))


def reading_metrics(samples, labels):
    if not labels:
        return {}, ()
    fields = defaultdict(Counter)
    by_key = {(row['game_id'], row['sample_time'], row['side'], row['row']): row for row in samples}
    for label in labels:
        if label['split'] != 'heldout':
            continue
        key = label['game_id'], label['time'], label['side'], label['row']
        if key not in by_key:
            continue
        row = by_key[key]
        for field, value in label['fields'].items():
            read = (row['items'] | row['stats'])[field]
            counter = fields[field]
            readable = value != 'unreadable'
            accepted = read['status'] in {'accepted', 'empty'}
            correct = accepted and readable and read['value'] == value
            counter.update(readable=int(readable), accepted=int(accepted),
                           covered=int(accepted and readable), correct=int(correct),
                           known_empty=int(value is None), nonempty=int(readable and value is not None),
                           nonempty_correct=int(correct and value is not None))
    result = {}
    for field, counts in fields.items():
        result[field] = {**counts,
                         'precision': counts['correct'] / counts['accepted'] if counts['accepted'] else None,
                         'coverage': counts['covered'] / counts['readable'] if counts['readable'] else None,
                         'false_accepts': counts['accepted'] - counts['correct']}
    eligible = [field for field, scores in result.items() if (scores['precision'] or 0) >= .95
                and (scores['coverage'] or 0) >= .70 and scores['nonempty']
                and (not field.startswith('slot') and field != 'trinket'
                     or scores['nonempty_correct'] / scores['nonempty'] >= .70)]
    return result, eligible


def run_readers(footage, manifest, profile, calibration, catalog_root, catalog, rosters,
                review, window, workers):
    from .broadcast_match_readers import ScoreboardReader, SpectatorFeed, center_notices, visual_signature
    from .intake import sampled_frames
    import numpy as np

    scoreboard = ScoreboardReader(profile['scoreboard'], Store(footage.destination), catalog_root, catalog)
    feed = SpectatorFeed(profile['feed'], calibration, scoreboard.reader.portraits)
    tracker = FeedTracks()
    labels = {row['time'] for row in review.get('scoreboard_labels', [])}
    snapshots, draft_samples = [], []
    replay_notices = {}
    frame_counts = Counter()
    last_draft = {}
    last_board = False
    last_board_time = -math.inf
    last_context, last_seen = None, -math.inf
    roles = [(side, row) for side in ('blue', 'red') for row in range(5)]
    scope_epoch = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for clip in manifest['clips']:
            lower = max(clip['start'], window[0])
            upper = min(clip['end'], window[1])
            if upper <= lower:
                continue
            start_tick = math.ceil(lower * FPS)
            targets = sorted({tick / FPS for tick in range(start_tick, math.ceil(upper * FPS))
                              if lower <= tick / FPS < upper} | {t for t in labels if lower <= t < upper})
            scheduled = []
            for target in targets:
                seg = footage.segments.at(target)
                if not seg or seg['review_status'] != 'reviewed':
                    continue
                if seg['phase'] == 'draft' and seg['visible_content'] == 'draft' and target * 2 % 1 == 0:
                    scheduled.append(target)
                elif (seg['broadcast_context'] in ('live', 'replay') and seg['visible_content'] == 'gameplay'
                      and seg['game_id'] in rosters):
                    scheduled.append(target)
            source = footage.store.path(clip['path'])
            native_times = [t - clip['start'] + clip['media_start'] for t in scheduled]
            stop = clip['end'] - clip['start'] + clip['media_start']
            if not native_times:
                continue
            native = sampled_frames(source, native_times[0], stop, FPS, targets=native_times)
            for sample, (pts, image) in zip(scheduled, native, strict=True):
                if image.size != tuple(profile['size']):
                    raise ValueError('Spectator geometry requires calibrated source resolution')
                current = round(clip['start'] + pts - clip['media_start'], 6)
                intended, seg = footage.segments.at(sample), footage.segments.at(current)
                # Camera cuts can split a continuous scope without affecting overlay ownership.
                keys = ('game_id', 'phase', 'broadcast_context', 'visible_content', 'review_status')
                if seg is None or tuple(seg[k] for k in keys) != tuple(intended[k] for k in keys):
                    continue
                game, phase, context = seg['game_id'], seg['phase'], seg['broadcast_context']
                frame_counts[f'{game}:{phase if phase == "draft" else context}'] += 1
                if phase == 'draft':
                    slots = profile['draft']['bans'] + profile['draft']['picks']
                    now = [visual_signature(image.crop(tuple(box))) for box in slots]
                    old = last_draft.get(game)
                    moved = [i for i in range(len(now)) if old and np.abs(now[i] - old[i]).mean() > 20]
                    if old is None or moved:
                        draft_samples.append({'time': current, 'game_id': game, 'changed_slots': moved,
                                              'pts': pts, 'clip_id': clip['id'], 'status': 'observed_change'})
                        last_draft[game] = now
                    continue
                state = (game, context)
                if last_context != state or current - last_seen > .36:
                    scope_epoch += 1
                    last_board = False
                last_context, last_seen = state, current
                entries = feed.read(image)
                scope = (game, context, scope_epoch)
                def observe(entry):
                    proof = footage.evidence(clip, pts, image, entry.box)
                    return {**proof, 'side': entry.side, 'raw_roles': entry.roles}
                tracker.update(current, scope, entries, observe)
                # A new row can carry a center capture sentence. Poll its first second
                # only while it is being established and retain the exact OCR string.
                notice_box = profile['feed']['center_box']
                new_rows = [(entry, track) for entry, track in tracker.previous
                            if track['rows'] in (2, 4) and entry.kind != 'kill']
                if new_rows or (context == 'replay' and sample * 2 % 1 == 0):
                    text, parsed, actors = '', None, None
                    for text in center_notices(image, profile['feed'].get('center_text_box', notice_box)):
                        parsed = parse_center_objective(text)
                        actors = parse_center_kill(text, rosters[game]) if context == 'replay' else None
                        if parsed or actors:
                            break
                    for entry, track in new_rows:
                        if parsed and entry.side == parsed[0] and entry.kind in ('unresolved', 'objective:' + parsed[1]):
                            proof = footage.evidence(clip, pts, image, notice_box)
                            track['evidence'].append({**proof, 'raw_text': text})
                            track['votes']['objective:' + parsed[1]] += 4
                    if context == 'replay' and (parsed or actors):
                        killer, victim = actors or (None, None)
                        side, objective = parsed or (killer['side'], None)
                        key = game, scope_epoch, side, objective, killer and killer['id'], victim and victim['id']
                        if key not in replay_notices:
                            proof = footage.evidence(clip, pts, image, notice_box)
                            replay_notices[key] = {
                                'id': uid('broadcast-center-replay', game, proof['id']), 'game_id': game,
                                'kind': 'neutral_objective' if objective else 'kill', 'side': side,
                                'objective': objective, 'context': 'replay', 'phase': phase,
                                'start': current, 'end': current,
                                'killer': killer and killer['id'], 'victim': victim and victim['id'],
                                'killer_name': killer and killer['name'], 'victim_name': victim and victim['name'],
                                'killer_champion': killer and killer['champion'],
                                'victim_champion': victim and victim['champion'], 'raw_text': text,
                                'onset_bracket': None, 'evidence_ids': [proof['id']], 'eligible': False,
                                'sightings': 0, 'review_status': 'candidate',
                            }
                        replay_notices[key]['end'] = current
                        replay_notices[key]['sightings'] += 1
                if context != 'live':
                    continue
                visible = scoreboard.present(image)
                frame_counts[f'{game}:{"full_scoreboard" if visible else "scoreboard_hidden"}'] += 1
                due = visible and (not last_board or current - last_board_time >= 5 or sample in labels)
                last_board = visible
                if not due:
                    continue
                last_board_time = current
                for side, row, reads in pool.map(lambda role: scoreboard.row(image, role), roles):
                    name = reads.pop('name')
                    portrait = reads.pop('champion')
                    member, method = bind_player(rosters[game], side, name['value'], portrait['value'])
                    stats = {k: reads.pop(k) for k in ('level', 'cs', 'kda')}
                    x0 = min(r['box'][0] for r in [name, portrait, *reads.values(), *stats.values()])
                    y0 = min(r['box'][1] for r in [name, portrait, *reads.values(), *stats.values()])
                    x1 = max(r['box'][2] for r in [name, portrait, *reads.values(), *stats.values()])
                    y1 = max(r['box'][3] for r in [name, portrait, *reads.values(), *stats.values()])
                    proof = footage.evidence(clip, pts, image, (x0, y0, x1, y1))
                    snapshots.append({'id': uid('broadcast-player-snapshot', game, side, row, proof['id']),
                                      'game_id': game, 'time': current, 'sample_time': sample, 'side': side,
                                      'row': row, 'kind': 'scoreboard_snapshot', 'binding_method': method,
                                      'player_id': member['id'] if member else None,
                                      'player_name': member['name'] if member else None,
                                      'champion': member['champion'] if member else None, 'name_read': name,
                                      'portrait_read': portrait, 'stats': stats, 'items': reads,
                                      'evidence_id': proof['id'], 'evidence_path': proof['path'],
                                      'evidence_ids': [proof['id'], *(member['evidence_ids'] if member else [])]})
    return snapshots, tracker.tracks, list(replay_notices.values()), draft_samples, dict(frame_counts)


def verify_match(root):
    active_path = root / 'broadcast_match' / 'active.json'
    if not active_path.is_file():
        return {'passed': True, 'errors': []}
    active = read_json(active_path)
    revision = root / 'broadcast_match' / active['revision']
    errors = []
    for relative, expected in active['checksums'].items():
        path = (root / relative).resolve()
        if (not path.is_relative_to(root.resolve()) or not path.is_file() or file_hash(path) != expected):
            errors.append(f'Match checksum mismatch in {relative}')
    evidence = read_rows(revision / 'broadcast_match_evidence.jsonl')
    ids = {r['id'] for r in evidence}
    for row in evidence:
        path = (root / row['path']).resolve()
        if not path.is_relative_to(revision.resolve()) or not path.is_file() or file_hash(path) != row['sha256']:
            errors.append(f'Match crop missing or changed: {row["id"]}')
    for table in TABLES - {'broadcast_match_evidence'}:
        for row in read_rows(revision / (table + '.jsonl')):
            if not set(row.get('evidence_ids', [])) <= ids:
                errors.append(f'Invalid match evidence for {row["id"]}')
    return {'passed': not errors, 'errors': errors, 'revision': active['revision']}


def run(args):
    from .broadcast_pass1 import verify
    from .pass1 import lock

    started = time.monotonic()
    root = args.run_dir.resolve()
    with lock(root, 'broadcast-match'):
        parent = verify(root)
        if not parent['passed']:
            raise ValueError(f'Verify the parent broadcast bundle first: {parent["errors"]}')
        return build(args, root, started)


def build(args, root, started):
    manifest = read_json(root / 'pass1_broadcast.json')
    profile, review = read_json(args.match_profile), read_json(args.review)
    if review['video_id'] != manifest['video_id'] or args.workers < 1:
        raise ValueError('The review requires this video and a positive worker count')
    version = profile['reference_version']
    catalog_root = reference_root(args.asset_cache, version)
    catalog = load_inventory_references(catalog_root, version)
    clip_scope = args.window or manifest['pilot']
    if not manifest['pilot'][0] <= clip_scope[0] < clip_scope[1] <= manifest['pilot'][1]:
        raise ValueError('Match analysis window is outside the pilot')
    files = ['pass1_broadcast.json', 'broadcast_segments.jsonl', 'broadcast_hud.jsonl', 'broadcast_events.jsonl']
    parents = {file: file_hash(root / file) for file in files}
    refs = {profile['feed']['sword_template'], *(f for fs in profile['feed']['objective_templates'].values() for f in fs)}
    recipe = {'parents': parents, 'profile': digest(profile), 'review': digest(review), 'scope': clip_scope,
              'catalog': digest(catalog), 'templates': {p: file_hash(args.calibration / p) for p in sorted(refs)},
              'code': implementation_hash(['broadcast_match.py', 'broadcast_match_timeline.py',
                                           'broadcast_match_readers.py', 'inventory_reader.py', 'numeric_reader.py',
                                           'readers.py', 'broadcast_ocr.py', 'killfeed.py'])}
    revision = digest(recipe)[:20]
    destination = root / 'broadcast_match' / revision
    footage = Footage(Store(root), manifest, read_rows(root / 'broadcast_segments.jsonl'), destination)
    rosters, draft = roster_and_draft(footage, review, profile, catalog, manifest['video_id'])
    snapshots, tracks, replay_notices, draft_samples, frame_counts = run_readers(
        footage, manifest, profile, args.calibration, catalog_root, catalog, rosters, review, clip_scope, args.workers)
    events = announcement_rows(tracks, rosters, manifest['video_id'])
    events.extend(replay_notices)
    expected = [row for row in review.get('events', []) if clip_scope[0] <= row['time'] < clip_scope[1]]
    event_scores = score_rows(events, expected)
    # When a reviewed occurrence is visible, anchor it even if the reader missed it.
    # This keeps the automatic score separate from the corrected timeline.
    matched = {row['reference_id']: row['event_id'] for row in event_scores['matches']}
    automatic = {r['id']: r for r in events}
    for truth in expected:
        frame = footage.frame(truth.get('witness_time', truth['time']), truth['game_id'], context='live')
        proof = footage.evidence(*frame, truth.get('box', profile['feed']['review_box']))
        found = automatic.get(matched.get(uid('broadcast-event-label', truth)))
        if found:
            found.update(review_status='reviewed', eligible=True)
            found['evidence_ids'].append(proof['id'])
            continue
        members = rosters[truth['game_id']]
        side = truth['side']
        killer, _ = bind_player(members, side, None, truth.get('killer_champion'))
        victim, _ = bind_player(members, 'red' if side == 'blue' else 'blue', None, truth.get('victim_champion'))
        events.append({**truth, 'id': uid('broadcast-event-review', manifest['video_id'], truth),
                       'context': 'live', 'start': truth['time'], 'end': truth['time'],
                       'onset_bracket': [truth['time'] - 1 / FPS, truth['time']],
                       'killer': killer['id'] if killer else None, 'victim': victim['id'] if victim else None,
                       'killer_name': killer['name'] if killer else None, 'victim_name': victim['name'] if victim else None,
                       'review_status': 'reviewed', 'eligible': True, 'sightings': 1, 'evidence_ids': [proof['id']]})
    events.sort(key=lambda row: (row['start'], row['id']))
    associate_replays(events)
    corroborate_kills(events, read_rows(root / 'broadcast_hud.jsonl'), footage.segments)
    metric, eligible = reading_metrics(snapshots, review.get('scoreboard_labels', []))
    for row in snapshots:
        row['eligible_fields'] = [k for k in eligible if row['player_id'] and
                                  (row['items'] | row['stats'])[k]['status'] in {'accepted', 'empty'}]
    history = item_history(snapshots, set(eligible))
    outputs = {'broadcast_draft': draft, 'broadcast_roster': [row for rows in rosters.values() for row in rows],
               'broadcast_player_snapshots': snapshots, 'broadcast_match_events': sorted(events, key=lambda row: row['start']),
               'broadcast_item_history': history, 'broadcast_match_evidence': list(footage.frames.values()),
               'draft_sample_changes': draft_samples}
    for name, rows in outputs.items():
        write_rows(destination / (name + '.jsonl'), rows)
    score_totals = defaultdict(Counter)
    for row in events:
        if row['kind'] == 'kill' and row['eligible'] and row['context'] == 'live':
            score_totals[row['game_id']][row['side']] += 1
    final_comparison = {game: {'final': row['final_kills'], 'reviewed': dict(score_totals[game])}
                        for game, row in review['games'].items() if 'final_kills' in row and args.window is None}
    report = {'status': 'completed_experimental', 'revision': revision, 'recipe': recipe,
              'seconds': time.monotonic() - started, 'frames': frame_counts,
              'counts': {k: len(rows) for k, rows in outputs.items()}, 'event_benchmark': event_scores,
              'scoreboard_benchmark': metric, 'eligible_scoreboard_fields': sorted(eligible),
              'final_kill_comparison': final_comparison,
              'unresolved_snapshot_identity': sum(row['player_id'] is None for row in snapshots),
              'unresolved_kill_roles': sum(row['kind'] == 'kill' and row['context'] == 'live' and
                                           (not row['killer'] or not row['victim']) for row in events)}
    atomic_json(destination / 'report.json', report)
    paths = [destination / (name + '.jsonl') for name in outputs] + [destination / 'report.json']
    active = {'revision': revision, 'checksums': parents | {str(path.relative_to(root)): file_hash(path) for path in paths}}
    atomic_json(root / 'broadcast_match' / 'active.json', active)
    return report
