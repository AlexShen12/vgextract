"""Game-scoped identity, announcement tracking and pilot benchmarks."""

from collections import Counter, defaultdict
import re
import unicodedata

from .core import uid
from .killfeed import fit_tracks


def normalized_name(name, team):
    value = unicodedata.normalize('NFKC', name or '').casefold().strip()
    prefix = team.casefold() + ' '
    return value[len(prefix):].strip() if value.startswith(prefix) else value


def bind_player(roster, side, name, champion):
    """A row coordinate locates the evidence. Identity comes from its contents."""
    members = [row for row in roster if row['side'] == side]
    if not members:
        return None, 'unresolved'
    token = normalized_name(name, members[0]['team'])
    by_name = [row for row in members if token and normalized_name(row['name'], row['team']) == token]
    by_champion = [row for row in members if champion and row['champion'] == champion]
    other_names = {normalized_name(row['name'], row['team']) for row in roster if row['side'] != side}
    if token in other_names or len(by_name) > 1 or len(by_champion) > 1:
        return None, 'conflict'
    if by_name and champion and by_name[0]['champion'] != champion:
        return None, 'conflict'
    if by_name:
        return by_name[0], 'name_and_portrait' if by_champion else 'name'
    return (by_champion[0], 'portrait') if by_champion else (None, 'unresolved')


class FeedTracks:
    """Track a row across stack shifts. Bound onsets only from continuous samples."""

    def __init__(self, fps=4):
        self.fps = fps
        self.previous = []
        self.last_time = None
        self.scope = None
        self.tracks = []

    def update(self, time, scope, entries, observe):
        adjacent = (self.last_time is not None and self.scope == scope
                    and 0 < time - self.last_time <= 1.25 / self.fps)
        matches = fit_tracks([entry for entry, _ in self.previous], entries) if adjacent else {}
        current = []
        for index, entry in enumerate(entries):
            if index in matches:
                track = self.previous[matches[index]][1]
            else:
                track = {'first': time, 'last': time, 'scope': scope,
                         'onset_bracket': [self.last_time, time] if adjacent else None,
                         'evidence': [observe(entry)], 'votes': Counter(), 'role_votes': defaultdict(Counter),
                         'rows': 0}
                self.tracks.append(track)
            track['last'] = time
            track['rows'] += 1
            track['votes'][entry.kind] += 1
            for role, candidates in entry.roles.items():
                value = candidates.get('champion')
                if value:
                    track['role_votes'][role][value] += 1
            current.append((entry, track))
        self.previous, self.scope, self.last_time = current, scope, time


def choose(values):
    ranked = values.most_common(2)
    if not ranked or (len(ranked) > 1 and ranked[0][1] == ranked[1][1]):
        return None
    return ranked[0][0]


def announcement_rows(tracks, roster, video_id):
    output = []
    for track in tracks:
        game_id, context, side_scope = track['scope']
        evidence = track['evidence'][0]
        side = evidence['side']
        votes = Counter({k: v for k, v in track['votes'].items() if k != 'unresolved'})
        label = choose(votes) or 'unresolved'
        objective = label.removeprefix('objective:') if label.startswith('objective:') else None
        killer = choose(track['role_votes']['killer'])
        victim = choose(track['role_votes']['victim']) if not objective else None
        members = roster.get(game_id, [])
        killer_player, _ = bind_player(members, side, None, killer)
        opposite = {'blue': 'red', 'red': 'blue'}.get(side)
        victim_player, _ = bind_player(members, opposite, None, victim)
        if label == 'kill' and not (killer_player or victim_player):
            label = 'unresolved'
        kind = 'neutral_objective' if objective else label
        output.append({'id': uid('broadcast-feed', video_id, game_id, context, side_scope, evidence['id']),
                       'game_id': game_id, 'kind': kind, 'objective': objective, 'side': side,
                       'killer': killer_player['id'] if killer_player else None,
                       'victim': victim_player['id'] if victim_player else None,
                       'killer_champion': killer, 'victim_champion': victim,
                       'killer_name': killer_player['name'] if killer_player else None,
                       'victim_name': victim_player['name'] if victim_player else None,
                       'start': track['first'], 'end': track['last'], 'context': context,
                       'onset_bracket': track['onset_bracket'], 'sightings': track['rows'],
                       'evidence_ids': [row['id'] for row in track['evidence']],
                       'eligible': False, 'review_status': 'candidate' if label != 'unresolved' else 'unresolved'})
    return sorted(output, key=lambda row: (row['start'], row['id']))


def event_key(row):
    if row['kind'] == 'kill' and row['killer_champion'] and row['victim_champion']:
        return row['game_id'], row['side'], 'kill', row['killer_champion'], row['victim_champion']
    if row['kind'] == 'neutral_objective' and row['objective'] and row['side']:
        return row['game_id'], row['side'], 'objective', row['objective']
    return None


def associate_replays(events, lookback=180):
    live = defaultdict(list)
    for row in events:
        key = event_key(row)
        if key is None:
            continue
        if row['context'] == 'live' and row['eligible']:
            live[key].append(row)
        elif row['context'] == 'replay':
            matches = [event for event in live[key] if 0 < row['start'] - event['start'] <= lookback]
            if row.get('phase') == 'postgame':
                # A breakdown can revisit the whole game and reorder its angles.
                matches = live[key]
            if len(matches) == 1:
                row['live_event_id'] = matches[0]['id']
                matches[0].setdefault('replay_sighting_ids', []).append(row['id'])


def score_rows(actual, expected, covered_games=None, tolerance=2):
    """Compare reviewed occurrences one-to-one within game, with boundary error."""
    covered_games = {row['game_id'] for row in expected} if covered_games is None else covered_games
    unmatched = {row['id']: row for row in actual if row['context'] == 'live' and row['game_id'] in covered_games
                 and row['kind'] in {'kill', 'neutral_objective'} and row['sightings'] >= 2}
    matched = []
    missed = []
    for truth in sorted(expected, key=lambda row: row['time']):
        fields = ('game_id', 'kind', 'side', 'objective') if truth['kind'] == 'neutral_objective' else (
            'game_id', 'kind', 'side', 'killer_champion', 'victim_champion')
        candidates = [row for row in unmatched.values() if all(
            row.get(key) == truth.get(key) for key in fields if truth.get(key) is not None)
            and abs(row['start'] - truth['time']) <= tolerance]
        if candidates:
            found = min(candidates, key=lambda row: abs(row['start'] - truth['time']))
            del unmatched[found['id']]
            matched.append({'event_id': found['id'], 'reference_time': truth['time'],
                            'error_seconds': round(found['start'] - truth['time'], 6), 'kind': truth['kind'],
                            'reference_id': uid('broadcast-event-label', truth)})
        else:
            missed.append(truth)
    rates = {}
    for kind in ('kill', 'neutral_objective'):
        tp = sum(row['kind'] == kind for row in matched)
        fp = sum(row['kind'] == kind for row in unmatched.values())
        fn = sum(row['kind'] == kind for row in missed)
        rates[kind] = {'matched': tp, 'extra': fp, 'missed': fn,
                       'precision': tp / (tp + fp) if tp + fp else None,
                       'recall': tp / (tp + fn) if tp + fn else None}
    return {'rates': rates, 'matches': matched, 'misses': missed, 'extra_ids': sorted(unmatched)}


def corroborate_kills(events, readings, segments):
    previous = {}
    for row in readings:
        segment = segments.at(row['time'])
        if (row['field'] != 'team_kills' or row['acceptance'] != 'accepted' or not segment
                or segment['broadcast_context'] != 'live' or segment['review_status'] != 'reviewed'):
            continue
        side = row['field_id'].removesuffix('_kills')
        key = row['game_id'], side
        before = previous.get(key)
        previous[key] = row
        if not before or row['parsed_value'] <= before['parsed_value'] or row['time'] - before['time'] > 20:
            continue
        nearby = [event for event in events if event['kind'] == 'kill' and event['eligible']
                  and event['context'] == 'live' and key == (event['game_id'], event['side'])
                  and before['time'] < event['start'] <= row['time']]
        if len(nearby) == row['parsed_value'] - before['parsed_value']:
            for event in nearby:
                event['scoreboard_change'] = {'before': before['parsed_value'], 'after': row['parsed_value'],
                                              'start': before['time'], 'end': row['time'],
                                              'hud_ids': [before['id'], row['id']],
                                              'shared_occurrences': len(nearby)}


def parse_center_objective(text):
    """Recognize the explicit capture sentence. A called objective timer is insufficient."""
    text = ' '.join(text.casefold().split())
    match = re.search(r'\b(blue|red) team has (?:slain|stolen|defeated) (?:the )?([a-z ]+)', text)
    if not match:
        return None
    side, phrase = match.groups()
    for pattern, kind in (
        ('elder', 'elder_dragon'), ('baron', 'baron_nashor'), ('herald', 'rift_herald'),
        ('void ?grubs?', 'voidgrubs'), ('(cloud|infernal|mountain|ocean|chemtech|hextech) (?:drake|dragon)', None),
    ):
        if m := re.search(pattern, phrase):
            return side, kind or m[1] + '_drake'
    return None


def parse_center_kill(text, roster):
    match = re.search(r'\b(\w+) (\w+) has slain (\w+) (\w+)\b', text, re.IGNORECASE)
    if not match:
        return None
    def member(team, name):
        matches = [r for r in roster if (r['team'].casefold() == team.casefold()
                   and normalized_name(r['name'], r['team']) == normalized_name(name, team))]
        return matches[0] if len(matches) == 1 else None
    killer = member(*match.groups()[:2])
    victim = member(*match.groups()[2:])
    return (killer, victim) if killer and victim and killer['side'] != victim['side'] else None


def item_history(snapshots, eligible_slots):
    previous, changes = {}, []
    for row in snapshots:
        if not row['player_id']:
            continue
        for field, read in row['items'].items():
            key = (row['game_id'], row['player_id'], field)
            last = previous.get(key)
            if read['status'] not in {'accepted', 'empty'}:
                if last:
                    row.setdefault('last_seen', {})[field] = {'time': last[0], 'value': last[1]}
                continue
            row.setdefault('last_seen', {})[field] = {'time': row['time'], 'value': read['value']}
            if last and last[1] != read['value']:
                changes.append({'id': uid('broadcast-item-change', row['id'], field),
                                'time': row['time'], 'game_id': row['game_id'], 'player_id': row['player_id'],
                                'field': field, 'before': last[1], 'after': read['value'],
                                'change_interval': [last[0], row['time']],
                                'eligible': field in eligible_slots,
                                'evidence_ids': [last[2], row['evidence_id']]})
            previous[key] = row['time'], read['value'], row['evidence_id']
    return changes
