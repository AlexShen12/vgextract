"""Regression coverage for source audio, playback joins and the ASR pilot gate."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from annotation.league.core import Store, atomic_json, read_json, write_rows
from annotation.league.experiment_worker import bind_audio_alignment
from annotation.league.pass1_asr import audio_clip, audio_coverage, commentary_rows, map_commentary
from annotation.league.pass1_review import install_pass1_routes
from annotation.league.media import file_hash


class CommentaryTests(unittest.TestCase):
    def test_alignment_offsets_and_invalid_words(self):
        chunk = {'source_start': 30., 'source_end': 40.}
        bind_audio_alignment(chunk, [
            {'text': 'hello', 'start_time': .5, 'end_time': 1.},
            {'text': 'outside', 'start_time': 9., 'end_time': 11.},
            {'text': 'backwards', 'start_time': .1, 'end_time': .2}])
        self.assertEqual(chunk['words'], [{'text': 'hello', 'start': 30.5, 'end': 31.}])
        self.assertEqual(chunk['alignment']['status'], 'partial')
        bind_audio_alignment(chunk, [])
        self.assertEqual(chunk['alignment']['status'], 'unavailable')

    def test_silence_and_unresolved_speech_account_for_full_interval(self):
        self.assertEqual(audio_coverage([], [30, 40]),
                         [{'start': 30, 'end': 40, 'status': 'vad_no_speech'}])
        rows = [{'id': 'one', 'source_start': 32, 'source_end': 34,
                 'transcription_status': 'unresolved', 'alignment': {'status': 'unavailable'}}]
        spans = audio_coverage(rows, [30, 40])
        self.assertEqual([s['status'] for s in spans],
                         ['vad_no_speech', 'unresolved_speech', 'vad_no_speech'])
        self.assertEqual(sum(s['end']-s['start'] for s in spans), 10)
        with self.assertRaises(ValueError):
            audio_coverage(rows+rows, [30, 40])

    def test_utterance_across_cut_or_gap_has_unknown_context(self):
        row = {'id': 'speech', 'start': 30.5, 'end': 32.5}
        spans = [{'start': 30, 'end': 32, 'playback_segment_id': 'a', 'continuity_verified': True},
                 {'start': 32, 'end': 34, 'playback_segment_id': 'b', 'continuity_verified': True}]
        self.assertIsNone(map_commentary([row], spans)[0]['playback_segment_id'])
        spans[1]['playback_segment_id'] = 'a'
        self.assertEqual(map_commentary([row], spans)[0]['playback_segment_id'], 'a')
        spans[1]['start'] = 32.2
        self.assertFalse(map_commentary([row], spans)[0]['continuity_verified'])

    def test_evidence_ids_survive_bundle_move(self):
        import shutil
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'original'
            root.mkdir()
            clip = root/'clip.wav'
            clip.write_bytes(b'audio evidence')
            chunk = {'path': str(clip), 'source_start': 30., 'source_end': 40.,
                     'text': 'Hello', 'status': 'transcribed', 'words': [],
                     'alignment': {'status': 'unavailable'}}
            first = commentary_rows({'chunks': [chunk]}, 'source', root, 'recipe')
            moved = Path(directory)/'moved'
            shutil.copytree(root, moved)
            chunk['path'] = str(moved/'clip.wav')
            second = commentary_rows({'chunks': [chunk]}, 'source', moved, 'recipe')
            self.assertEqual(first, second)
            self.assertEqual(first[0]['alignment']['status'], 'unavailable')

    def test_timeline_interleaves_and_older_bundle_loads(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atomic_json(root/'pass1_manifest.json', {'draft_revision': 'r'})
            revision = root/'revisions/r'
            write_rows(revision/'events.jsonl', [{'id': 'event', 'start': 31, 'end': 31.1}])
            app = FastAPI()
            install_pass1_routes(app, Store(root))
            client = TestClient(app)
            self.assertEqual(client.get('/api/pass1/records?target=timeline').json()['total'], 1)
            write_rows(revision/'commentary_timeline.jsonl', [
                {'id': 'commentary', 'start': 30, 'end': 32, 'text': 'hello'}])
            rows = client.get('/api/pass1/records?target=timeline').json()['records']
            self.assertEqual([r['id'] for r in rows], ['commentary', 'event'])
            self.assertEqual(rows[0]['record_table'], 'commentary_timeline')

    def test_decoder_uses_requested_source_interval(self):
        import wave
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def decode(command, **kwargs):
                self.assertEqual(command[command.index('-ss')+1], '30')
                self.assertEqual(command[command.index('-t')+1], '10')
                with wave.open(command[-1], 'wb') as target:
                    target.setnchannels(1)
                    target.setsampwidth(2)
                    target.setframerate(16000)
                    target.writeframes(b'\0'*320000)
            with patch('annotation.league.pass1_asr.subprocess.run', side_effect=decode):
                audio_clip(root/'video.mp4', root/'clip.wav', 30, 40)
            self.assertTrue((root/'clip.wav').is_file())

class PilotGateTests(unittest.TestCase):
    def test_reference_gate_detects_changed_audio(self):
        from annotation.league.core import digest
        from annotation.league.pass1_asr import require_pilot, verify_reference
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            producer = root/'producers/asr'
            pilot = producer/'pilot/test'
            pilot.mkdir(parents=True)
            (pilot/'clip.wav').write_bytes(b'source clip')
            atomic_json(pilot/'result.json', {'result': 'original'})
            report = {'source_sha256': 'source', 'windows': [[30.,40.]], 'recipe': 'r',
                      'pilot_report_path': 'producers/asr/pilot/test/report.json',
                      'resource_passed': True}
            atomic_json(pilot/'report.json', report)
            atomic_json(producer/'report.json', report)
            atomic_json(root/'manifest.json', {'source': {'sha256': 'source'}})
            atomic_json(root/'pass1_manifest.json', {'producers': {}})
            write_rows(pilot/'commentary.jsonl', [{
                'id': 'c', 'text': 'Darius is here', 'alignment': {'status': 'aligned'},
                'words': [{'text': text, 'start': 31+i, 'end': 31.5+i}
                          for i,text in enumerate(['Darius','is','here'])]}])
            reference = {'source_sha256': 'source', 'interval': [30,40],
                         'method': 'source_listening', 'reviewer': 'test',
                         'transcript': 'Darius is here', 'speaker': 'Midbeast',
                         'names': ['Darius'], 'no_invented_or_skipped_phrases': True,
                         'words': [{'text': text, 'start': 31.1+i}
                                   for i,text in enumerate(['Darius','is','here'])]}
            atomic_json(root/'reference.json', reference)
            self.assertTrue(verify_reference(root, root/'reference.json')['passed'])
            require_pilot(root)
            (pilot/'clip.wav').write_bytes(b'changed source clip')
            with self.assertRaisesRegex(ValueError, 'passing source-verified'):
                require_pilot(root)


class AutoCaptionTests(unittest.TestCase):
    def test_tokens_use_word_offsets_and_ignore_display_duration(self):
        from annotation.league.pass1_asr_captions import caption_words
        track = {'events': [
            {'tStartMs': 29000, 'dDurationMs': 10000,
             'segs': [{'utf8': 'before'}, {'utf8': ' inside', 'tOffsetMs': 1200}]},
            {'tStartMs': 32000, 'segs': [{'utf8': '\n'}]},
            {'tStartMs': 39000, 'segs': [{'utf8': 'last'}, {'utf8': 'after', 'tOffsetMs':1000}]}]}
        self.assertEqual(caption_words(track),
                         [{'text': 'inside','start':30.2},{'text':'last','start':39.}])

    def test_caption_reference_retains_unverified_speaker(self):
        from annotation.league.pass1_asr_captions import verify_captions
        from annotation.league.core import digest
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pilot = root/'producers/asr/pilot/test'
            pilot.mkdir(parents=True)
            (pilot/'clip.wav').write_bytes(b'audio')
            atomic_json(pilot/'result.json', {})
            report = {'source_sha256':'source','recipe':'r','windows':[[30,40]],
                      'pilot_report_path':'producers/asr/pilot/test/report.json','resource_passed':True}
            atomic_json(pilot/'report.json',report)
            atomic_json(root/'producers/asr/report.json',report)
            atomic_json(root/'pass1_manifest.json',{'producers':{}})
            words=[{'text':text,'start':31+i,'end':31.5+i} for i,text in enumerate(['one','two','three'])]
            write_rows(pilot/'commentary.jsonl',[{'text':'one two three','words':words,'alignment':{'status':'aligned'}}])
            track = root/'captions.json3'
            atomic_json(track,{'events':[{'tStartMs':31000,'segs':[
                {'utf8':w['text'],'tOffsetMs':i*1000} for i,w in enumerate(words)]}]})
            reference={'method':'youtube_auto_captions','source_sha256':'source','interval':[30,40],
                       'video_id':'OLLZBe5-fg4','reviewer':'test','caption_path':'captions.json3',
                       'caption_sha256':file_hash(track),'meaning_matches':True,
                       'onset_words':['one','two','three']}
            atomic_json(root/'reference.json',reference)
            result=verify_captions(root,root/'reference.json')
            self.assertTrue(result['passed'])
            self.assertEqual(result['speaker_verification'],'unverified')
            track.write_text('{}')
            with self.assertRaisesRegex(ValueError,'checksum'):
                verify_captions(root,root/'reference.json')


if __name__ == '__main__':
    unittest.main()
