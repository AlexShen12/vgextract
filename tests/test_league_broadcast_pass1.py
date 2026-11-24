from pathlib import Path
import tempfile
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from annotation.league.broadcast_ocr import evaluate_ocr, mark_effective
from annotation.league.broadcast_timeline import (analysis_windows, commentary_links, replay_metrics,
                                                 reviewed_segments)
from annotation.league.core import Store, atomic_json, write_rows
from annotation.league.pass1_review import install_pass1_routes


class BroadcastPass1Tests(unittest.TestCase):
    def test_replay_review_blocks_live_and_preserves_replay_evidence(self):
        automatic = [{'start': 0, 'end': 20, 'broadcast_context': 'live',
                      'visible_content': 'gameplay', 'game_id': 'g1', 'review_status': 'unreviewed'}]
        truth = [
            {'start': 0, 'end': 10, 'broadcast_context': 'live', 'visible_content': 'gameplay'},
            {'start': 10, 'end': 15, 'broadcast_context': 'replay', 'visible_content': 'gameplay'},
            {'start': 15, 'end': 18, 'broadcast_context': 'replay', 'visible_content': 'reaction'},
        ]
        segments = list(reviewed_segments(automatic, truth, 'vod'))
        events = [{'id': 'kill', 'game_id': 'g1', 'observations': [{'time': 8, 'role': 'live'},
                                                 {'time': 12, 'role': 'replay'},
                                                 {'time': 16, 'role': 'replay'}]}]
        links = [{'event_id': 'kill', 'review_status': 'reviewed'}]
        windows = analysis_windows(segments, events, links, 'vod')
        self.assertEqual([(r['purpose'], r['start'], r['end']) for r in windows],
                         [('live_gameplay', 0, 10), ('replay_evidence', 10, 15)])
        score = replay_metrics(automatic, truth)
        self.assertEqual(score['missed_replay_seconds'], 8)
        self.assertEqual(score['reviewed_seconds'], 18)

    def test_clock_requires_a_heldout_pass_and_same_segment(self):
        for valid in (False, True):
            with self.subTest(valid=valid):
                labels, rows = [], []
                for n in range(14):
                    text = f'02:{n:02d}'
                    labels.append({'time': n, 'field': 'game_clock', 'text': text,
                                   'split': 'calibration' if n < 3 else 'heldout'})
                    rows.append({'field_id': 'game_clock', 'sample_time': n, 'time': n, 'layout': 'live',
                                 'segment_id': 'first' if n < 13 else 'after-replay',
                                 'attempts': {'grayscale': {'raw_text': text if valid or n != 7 else '82:07',
                                                            'value': {'seconds': 120+n},
                                                            'acceptance': 'accepted'}}})
                benchmark = evaluate_ocr(rows, labels)
                mark_effective(rows, benchmark)
                self.assertEqual(rows[0]['continuity_corroborated'], valid)
                self.assertEqual(rows[-1]['acceptance'], 'review_only')

    def test_commentary_reuses_the_event_id_across_playback_angles(self):
        phases = [{'start': 0, 'end': 40, 'game_id': 'g1', 'phase': 'game'},
                  {'start': 40, 'end': 60, 'game_id': 'g2', 'phase': 'game'}]
        cues = [{'id': 'cue', 'start': 20, 'text': 'Kiin takes the long route around Elder.'}]
        event = {'id': 'elder', 'key': 'elder', 'game_id': 'g1', 'entities': ['Elder'],
                 'observations': [{'time': 10, 'phase': 'game'}, {'time': 22, 'phase': 'game'}],
                 'evidence_ids': ['live', 'replay']}
        candidates = [{'id': f'proposal-{start}', 'start': start, 'end': start + 10, 'cue_ids': ['cue']}
                      for start in (19, 40)]
        review = [{'start': 19, 'end': 24, 'event': 'elder', 'summary': 'The caster explains the route.'}]
        proposed, other_game, linked = commentary_links(candidates, cues, [event], review, phases, 'vod')
        self.assertEqual([row['event_id'] for row in proposed['candidates']], ['elder'])
        self.assertEqual(other_game['candidates'], [])
        self.assertEqual(linked['event_id'], 'elder')
        self.assertEqual(linked['evidence_ids'], ['live', 'replay'])

    def test_broadcast_review_serves_full_transcript_clips_and_link_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'clip.mp4').write_bytes(b'fixture')
            atomic_json(root / 'pass1_broadcast.json', {'video_id': 'vod', 'clips': [
                {'id': 'clip', 'start': 100, 'end': 200, 'path': 'clip.mp4', 'media_start': 0}]})
            atomic_json(root / 'pass1_report.json', {'status': 'completed_experimental'})
            write_rows(root / 'commentary_timeline.jsonl', [{'id': 'c1', 'start': 30, 'text': 'Game preview'}])
            (root / 'fight.jpg').write_bytes(b'evidence')
            write_rows(root / 'commentary_links.jsonl', [{'id': 'link', 'start': 120, 'evidence_ids': ['frame']}])
            write_rows(root / 'evidence_frames.jsonl', [{'id': 'frame', 'time': 140, 'path': 'fight.jpg'}])
            app = FastAPI()
            install_pass1_routes(app, Store(root))
            client = TestClient(app)
            result = client.get('/api/pass1/records', params={'target': 'commentary_timeline', 'q': 'preview'})
            self.assertEqual(result.json()['records'][0]['start'], 30)
            self.assertEqual(client.get('/pass1/clip/clip', headers={'Range': 'bytes=0-2'}).content, b'fix')
            self.assertEqual(client.get('/pass1/clip/other').status_code, 404)
            evidence = client.get('/api/pass1/evidence/commentary_links/link').json()
            self.assertEqual(evidence['frames'], [{'path': 'fight.jpg', 'time': 140, 'box': None}])
            self.assertEqual(client.get('/pass1/file/fight.jpg').content, b'evidence')
