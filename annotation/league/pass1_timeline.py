"""Pure, conservative joins for the experimental Pass 1 bundle."""
from bisect import bisect_right
from collections import defaultdict
from copy import deepcopy

from .core import digest, uid
from .normalization import accepted


class ContextIndex:
    def __init__(self, rows):
        self.rows = sorted(rows, key=lambda r: r['start'])
        self.starts = [r['start'] for r in self.rows]

    def at(self, time):
        i = bisect_right(self.starts, time) - 1
        if i >= 0 and self.rows[i]['start'] <= time < self.rows[i]['end']:
            return self.rows[i]
        return None


def map_context(rows, playback, selection=()):
    """Keep original producer scopes; never extrapolate across an unobserved gap."""
    pi, si = ContextIndex(playback), ContextIndex(selection)
    result = []
    for original in rows:
        row = dict(original)
        time = row.get('time', row.get('start'))
        p, s = pi.at(time), si.at(time)
        row['producer_playback_segment_id'] = row.get('playback_segment_id', row.get('segment_id'))
        segment = p['playback_segment_id'] if p else uid('unknown-playback', time)
        row.update(playback_segment_id=segment, segment_id=segment,
                   continuity_verified=bool(p and p.get('continuity_verified')))
        if row.get('owner_id') == 'selected' or row.get('slot_id') == 'selected' or row.get('route') == 'selected_hud':
            row['producer_owner_epoch'] = row.get('owner_epoch')
            row['owner_epoch'] = s['owner_epoch'] if s else uid('unknown-selection', time)
            row['selection_evidence_ids'] = s.get('evidence_ids', []) if s else []
        result.append(row)
    return result


def bind_visible_portraits(rows):
    """Same-time portrait evidence names HUD and scoreboard facts, never a body."""
    portraits = defaultdict(list)
    for row in rows:
        if row['field'] == 'inventory.champion' and accepted(row):
            portraits[(row['time'], row['route'], row['slot_id'])].append(row)
    result=[]
    for original in rows:
        row=dict(original)
        evidence=portraits.get((row['time'],row['route'],row['slot_id']),[])
        names={p['value'] for p in evidence}
        if len(names)==1 and row['route'] in {'selected_hud','scoreboard'}:
            row.update(entity=next(iter(names)),resolved_identity=next(iter(names)),
                       identity_evidence=[p['id'] for p in evidence])
        result.append(row)
    return result


def inventory_bindings(snapshots, observations):
    """Two concordant name/portrait reads establish an experimental exact-name binding.

    Bindings are segment- and side-scoped, reject all contradictory names, and
    never use a row number or an inferred missing team member as identity evidence.
    """
    snapshots = deepcopy(snapshots)
    names = {(r['time'], r['slot_id']): r for r in observations
             if r['field'] == 'player_name' and accepted(r)}
    witnesses = defaultdict(list)
    for snapshot in snapshots:
        name = names.get((snapshot['time'], snapshot['slot_id']))
        if snapshot.get('side') and name and snapshot.get('champion') and snapshot.get('continuity_verified'):
            key = (snapshot['playback_segment_id'], snapshot['side'], name['value'])
            witnesses[key].append((snapshot, name))
    bindings = {}
    for key, reads in witnesses.items():
        champions = {s['champion'] for s, _ in reads}
        if len(champions) == 1 and len({s['time'] for s, _ in reads}) >= 2:
            bindings[key] = reads
    # One champion with two distinct names in a segment is also disputed.
    inverse = defaultdict(set)
    for key, reads in bindings.items():
        inverse[(key[0], key[1], reads[0][0]['champion'])].add(key[2])
    links = []
    for snapshot in snapshots:
        name = names.get((snapshot['time'], snapshot['slot_id']))
        if not name or not snapshot.get('side') or snapshot.get('champion') or not snapshot.get('continuity_verified'):
            continue
        key = (snapshot['playback_segment_id'], snapshot['side'], name['value'])
        reads = bindings.get(key)
        if not reads:
            continue
        champion = reads[0][0]['champion']
        if len(inverse[(key[0], key[1], champion)]) != 1:
            continue
        evidence = sorted({e for s, n in reads for e in s['evidence_ids'] + [n['id']]})
        link = {'id': uid('inventory-binding', snapshot['id'], evidence),
                'snapshot_id': snapshot['id'], 'champion': champion,
                'time': snapshot['time'], 'evidence_available_at': sorted(s['time'] for s, _ in reads)[1],
                'playback_segment_id': key[0], 'side': key[1], 'player_name': key[2],
                'evidence_ids': evidence + [name['id']], 'experimental': True,
                'review_status': 'unreviewed', 'method': 'two_concordant_exact_name_portrait_reads'}
        links.append(link)
        snapshot.update(champion=champion, identity_link_id=link['id'], identity_evidence_ids=link['evidence_ids'])
    return snapshots, links


def item_history(snapshots):
    result, previous = [], {}
    for snapshot in sorted(snapshots, key=lambda s: (s['time'], s['id'])):
        if not snapshot.get('champion'):
            continue
        for slot in snapshot['slots']:
            if slot['status'] not in {'recognized', 'empty'}:
                continue
            key = (snapshot['playback_segment_id'], snapshot.get('side'), snapshot['champion'], slot['slot'])
            last = previous.get(key)
            value = (slot['status'], slot['item_id'])
            row = {'id': uid('inventory-history', snapshot['id'], slot['slot']),
                   'time': snapshot['time'], 'champion': snapshot['champion'], 'side': snapshot.get('side'),
                   'slot': slot['slot'], 'status': slot['status'], 'item_id': slot['item_id'],
                   'snapshot_id': snapshot['id'], 'playback_segment_id': key[0],
                   'evidence_ids': [slot['observation_id']] + snapshot.get('identity_evidence_ids', []),
                   'last_seen_at': snapshot['time'], 'change_interval': None, 'review_status': 'unreviewed'}
            if last and last[1] != value and last[0] < snapshot['time'] and snapshot.get('continuity_verified'):
                row['change_interval'] = [last[0], snapshot['time']]
            previous[key] = (snapshot['time'], value)
            result.append(row)
    return result


def scoreboard_openings(visibility, snapshots, max_gap=2.0):
    """Retain four-FPS boundaries; only merge fully resolved ten-row inventories."""
    groups, current, prior = [], None, None
    for sample in sorted(visibility, key=lambda x: x['time']):
        present = sample['status'] == 'present'
        same = (prior and sample['time'] - prior['time'] <= .251 and
                sample['playback_segment_id'] == prior['playback_segment_id'] and
                sample.get('continuity_verified'))
        if not present or not same:
            current = None
        if present:
            if current is None:
                current = {'id': uid('scoreboard-opening', sample['id']), 'start': sample['time'],
                           'end': sample['time'] + .25, 'playback_segment_id': sample['playback_segment_id'],
                           'sample_ids': [], 'snapshot_ids': [], 'onset_known': bool(same and prior['status'] == 'absent')}
                groups.append(current)
            current['end'] = sample['time'] + .25
            current['sample_ids'].append(sample['id'])
        prior = sample
    by_time = defaultdict(list)
    for s in snapshots:
        if s['route'] == 'scoreboard':
            by_time[s['time']].append(s)
    def signature(opening):
        signatures = set()
        resolved = True
        for time, rows in by_time.items():
            if not opening['start'] <= time < opening['end']:
                continue
            opening['snapshot_ids'].extend(s['id'] for s in rows)
            if len(rows) != 10 or any(not s.get('champion') or any(
                    v['status'] not in {'recognized', 'empty'} for v in s['slots']) for s in rows):
                resolved = False
                continue
            identities = {(s['side'], s['champion']) for s in rows}
            if len(identities) != 10:
                resolved = False
                continue
            signatures.add(digest(sorted((s['side'], s['champion'], sorted(
                (v['slot'], v['status'], v['item_id']) for v in s['slots'])) for s in rows)))
        return next(iter(signatures)) if resolved and len(signatures) == 1 else None
    for g in groups:
        g['inventory_signature'] = signature(g)
        g['merged_opening_ids'] = [g['id']]
        g['closures'] = []
    merged = []
    for group in groups:
        last = merged[-1] if merged else None
        gap_rows = [s for s in visibility if last and last['end'] <= s['time'] < group['start']]
        safe_gap = bool(gap_rows) and all(s['status'] == 'absent' and s.get('continuity_verified') and
                    s['playback_segment_id'] == group['playback_segment_id'] for s in gap_rows)
        if (last and 0 <= group['start'] - last['end'] <= max_gap and safe_gap and
                last['playback_segment_id'] == group['playback_segment_id'] and
                group['inventory_signature'] and group['inventory_signature'] == last['inventory_signature']):
            last['closures'].append([last['end'], group['start']])
            last['end'] = group['end']
            for key in ('sample_ids', 'snapshot_ids', 'merged_opening_ids'):
                last[key].extend(group[key])
        else:
            merged.append(deepcopy(group))
    return groups, merged
