"""Regression coverage for game-scoped Worlds match review."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np

from annotation.league.broadcast_match import reading_metrics
from annotation.league.broadcast_match_timeline import (
    FeedTracks, announcement_rows, associate_replays, bind_player, item_history, parse_center_kill,
    parse_center_objective, score_rows,
)
from annotation.league.core import Store, atomic_json, write_rows
from annotation.league.killfeed_reader import Entry
from annotation.league.pass1_review import install_pass1_routes


def member(game, side, name, champion, row=0):
    return {'id': f'{game}-{name}', 'game_id': game, 'side': side, 'name': name,
            'team': 'GEN' if side == 'blue' else 'BLG', 'champion': champion, 'row': row}


class MatchReadersTest(unittest.TestCase):
    def test_game_scoped_binding_rejects_conflicts_and_ignores_swap_position(self):
        first_game = [member(1, 'blue', 'Chovy', 'Syndra', 2), member(1, 'blue', 'Kiin', 'Olaf', 0),
                      member(1, 'red', 'Knight', 'Locke', 2)]
        next_game = [member(2, 'blue', 'Chovy', 'Orianna', 2), member(2, 'red', 'Knight', 'Cassiopeia', 2)]
        self.assertEqual(bind_player(first_game, 'blue', ' ＧＥＮ  Chovy ', None), (first_game[0], 'name'))
        self.assertEqual(bind_player(first_game, 'blue', None, 'Olaf'), (first_game[1], 'portrait'))
        self.assertEqual(bind_player(first_game, 'blue', 'Kiin', 'Syndra'), (None, 'conflict'))
        self.assertEqual(bind_player(first_game, 'blue', 'Knight', 'Olaf'), (None, 'conflict'))
        self.assertEqual(bind_player(next_game, 'blue', 'Chovy', 'Syndra'), (None, 'conflict'))
        self.assertEqual(bind_player(next_game, 'blue', 'Chovy', 'Orianna'), (next_game[0], 'name_and_portrait'))
        self.assertEqual(parse_center_kill('GEN Chovy has slain BLG Knight!', next_game), tuple(next_game))
        self.assertIsNone(parse_center_kill('GEN Chovy is on a killing spree!', next_game))
        self.assertIsNone(parse_center_objective('BARON 0:03'))
        self.assertEqual(parse_center_objective('Blue team has slain the Baron Nashor!'), ('blue', 'baron_nashor'))
        self.assertEqual(parse_center_objective('Red team has defeated the Rift Herald!'), ('red', 'rift_herald'))
        self.assertIsNone(parse_center_objective('Red team has summoned the Rift Herald!'))

    def test_repeated_kill_simultaneous_objective_and_scope_boundaries(self):
        roster = {'g': [member('g', 'blue', 'Canyon', 'LeeSin'), member('g', 'red', 'Viper', 'Kalista')]}
        kill = Entry(200, 'kill', 'blue', [0, 0, 10, 10],
                     {'killer': {'champion': 'LeeSin'}, 'victim': {'champion': 'Kalista'}},
                     np.zeros((2, 16, 16, 3)))
        objective = Entry(244, 'objective:baron_nashor', 'blue', [0, 10, 10, 20],
                          {'killer': {'champion': 'LeeSin'}}, np.ones((2, 16, 16, 3)) * 255)
        tracker = FeedTracks()
        for time, scope, entries in [(10, ('g', 'live', 0), []), (10.25, ('g', 'live', 0), [kill, objective]),
                                     (10.5, ('g', 'live', 0), [kill, objective]), (10.75, ('g', 'live', 0), []),
                                     (11, ('g', 'live', 0), [kill]), (11.25, ('g', 'live', 0), [kill]),
                                     (15, ('g', 'replay', 1), [kill, objective])]:
            tracker.update(time, scope, entries, lambda entry: {'id': f'{scope}-{time}-{entry.top}',
                                                                'side': entry.side})
        rows = announcement_rows(tracker.tracks, roster, 'vod')
        live = [event for event in rows if event['context'] == 'live']
        self.assertCountEqual([r['kind'] for r in live], ['kill', 'kill', 'neutral_objective'])
        self.assertIn([10, 10.25], [r['onset_bracket'] for r in live])
        for event in live:
            event['eligible'] = True
        associate_replays(rows)
        replay = {row['kind']: row for row in rows if row['context'] == 'replay'}
        self.assertIsNone(replay['kill']['onset_bracket'])
        self.assertNotIn('live_event_id', replay['kill'])
        self.assertIn('live_event_id', replay['neutral_objective'])
        truth = [{'time': row['start'], 'game_id': row['game_id'], 'side': row['side'], 'kind': row['kind'],
                  'objective': row['objective'], 'killer_champion': row['killer_champion'],
                  'victim_champion': row['victim_champion']} for row in live]
        results = score_rows(rows, truth)
        self.assertEqual(results['rates']['kill']['matched'], 2)
        self.assertEqual(results['rates']['neutral_objective']['precision'], 1)

    def test_items_keep_last_seen_and_uncertain_changes_in_review(self):
        snapshots = [
            {'game_id': 'g', 'time': t, 'player_id': 'p', 'id': str(t), 'evidence_id': f'e{t}',
             'items': {'slot1': {'status': status, 'value': value}}}
            for t, status, value in [(10, 'accepted', '1055'), (15, 'unreadable', None), (20, 'empty', None)]
        ]
        changes = item_history(snapshots, set())
        self.assertEqual(snapshots[1]['last_seen']['slot1'], {'time': 10, 'value': '1055'})
        self.assertEqual(changes[0]['change_interval'], [10, 20])
        self.assertEqual(changes[0]['evidence_ids'], ['e10', 'e20'])
        self.assertFalse(changes[0]['eligible'])

    def test_postgame_replay_requires_one_live_occurrence_across_the_game(self):
        capture = {'id': 'capture', 'game_id': 'game-1', 'side': 'red', 'kind': 'neutral_objective',
                   'objective': 'elder_dragon', 'context': 'live', 'eligible': True, 'start': 100}
        replay = {**capture, 'id': 'replay', 'context': 'replay', 'phase': 'postgame',
                  'eligible': False, 'start': 600}
        other_game = {**capture, 'id': 'separate', 'game_id': 'game-2', 'start': 300}
        associate_replays([capture, other_game, replay])
        self.assertEqual(replay['live_event_id'], 'capture')
        repeated = {**capture, 'id': 'second-capture', 'start': 450}
        second_replay = {key: value for key, value in replay.items() if key != 'live_event_id'}
        associate_replays([capture, repeated, second_replay])
        self.assertNotIn('live_event_id', second_replay)

    def test_field_gate_uses_heldout_and_counts_false_accepts(self):
        def sample(time, status, value):
            return {'game_id': 'g', 'sample_time': time, 'side': 'blue', 'row': 0, 'items': {},
                    'stats': {'level': {'status': status, 'value': value}}}
        def label(time, value, split):
            return {'game_id': 'g', 'time': time, 'side': 'blue', 'row': 0, 'split': split, 'fields': {'level': value}}
        metrics, eligible = reading_metrics([sample(10, 'accepted', 3), sample(15, 'accepted', 8)],
                                           [label(10, 8, 'calibration'), label(15, 'unreadable', 'heldout')])
        self.assertEqual(metrics['level']['false_accepts'], 1)
        self.assertEqual(eligible, [])

    def test_empty_slots_cannot_authorize_automatic_item_ids(self):
        samples, labels = [], []
        for tick in range(6):
            value = '1055' if tick == 5 else None
            key = {'game_id': 'g', 'side': 'blue', 'row': 0}
            labels.append({**key, 'time': tick, 'split': 'heldout', 'fields': {'slot1': value}})
            samples.append({**key, 'sample_time': tick, 'stats': {}, 'items': {
                'slot1': {'status': 'unreadable' if value else 'empty', 'value': None},
            }})
        metrics, eligible = reading_metrics(samples, labels)
        self.assertEqual(metrics['slot1']['precision'], 1)
        self.assertGreater(metrics['slot1']['coverage'], .7)
        self.assertEqual(metrics['slot1']['nonempty_correct'], 0)
        self.assertEqual(eligible, [])


class MatchReviewTest(unittest.TestCase):
    def test_review_exposes_match_tables_nested_report_and_native_evidence(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            atomic_json(root / 'pass1_broadcast.json', {'video_id': 'vod', 'clips': []})
            atomic_json(root / 'pass1_report.json', {'status': 'completed_experimental'})
            revision = root / 'broadcast_match' / 'revision'
            revision.mkdir(parents=True)
            atomic_json(revision / 'report.json', {'status': 'completed_experimental'})
            atomic_json(revision.parent / 'active.json', {'revision': 'revision'})
            crop = revision / 'draft.png'
            crop.write_bytes(b'crop')
            write_rows(revision / 'broadcast_draft.jsonl', [
                {'id': 'd', 'game_id': 'game-1', 'champion': 'Kalista', 'state': 'hover', 'evidence_ids': ['a']},
                {'id': 'e', 'game_id': 'game-2', 'champion': 'Kalista', 'state': 'prior_game_exclusion'},
            ])
            write_rows(revision / 'broadcast_match_evidence.jsonl', [
                {'id': 'a', 'path': str(crop.relative_to(root)), 'time': 30, 'box': [0, 0, 10, 10]},
            ])
            app = FastAPI()
            install_pass1_routes(app, Store(root))
            client = TestClient(app)
            result = client.get('/api/pass1').json()
            self.assertIn('broadcast_draft', result['tables'])
            self.assertIn('commentary_timeline', result['tables'])
            self.assertEqual(result['match_report']['status'], 'completed_experimental')
            rows = client.get('/api/pass1/records', params={'target': 'broadcast_draft', 'q': 'hover'}).json()
            self.assertEqual([r['id'] for r in rows['records']], ['d'])
            frames = client.get('/api/pass1/evidence/broadcast_draft/d').json()['frames']
            self.assertEqual(frames[0]['time'], 30)
            self.assertEqual(client.get('/pass1/file/' + frames[0]['path']).content, b'crop')
