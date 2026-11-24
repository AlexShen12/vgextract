"""YouTube caption import timing and source selection regressions."""
from pathlib import Path
import tempfile
import unittest

from annotation.league.core import atomic_json, read_json
from annotation.league.pass1_youtube_captions import caption_rows, selected_commentary


def rows(events, duration=10):
    return caption_rows({'events': events}, duration=duration, source_hash='source',
                        track_hash='track', track_path='track.json3',
                        video_id='video', language='en-orig')


class YouTubeCaptionTests(unittest.TestCase):
    def test_rolling_display_overlap_keeps_onsets_and_bounds_cues(self):
        actual = rows([
            {'tStartMs':0,'dDurationMs':5000,'segs':[{'utf8':'Hello'},{'utf8':' world','tOffsetMs':500}]},
            {'tStartMs':990,'segs':[{'utf8':'\n'}]},
            {'tStartMs':1000,'dDurationMs':5000,'segs':[{'utf8':'Next'}]}])
        self.assertEqual(len(actual),2)
        self.assertEqual(actual[0]['text'],'Hello world')
        self.assertEqual(actual[0]['end'],1)
        self.assertEqual(actual[0]['words'][1]['start'],.5)
        self.assertIsNone(actual[0]['words'][1]['end'])
        self.assertEqual(actual[0]['caption_display_end'],5)

    def test_duplicate_timed_text_is_emitted_once(self):
        actual = rows([
            {'tStartMs':0,'dDurationMs':5000,'segs':[{'utf8':'Hello'},{'utf8':' world','tOffsetMs':1000}]},
            {'tStartMs':0,'dDurationMs':5000,'segs':[{'utf8':'Hello'},{'utf8':' world','tOffsetMs':1000},
                                                               {'utf8':' again','tOffsetMs':2000}]}])
        self.assertEqual([r['text'] for r in actual],['Hello world','again'])

    def test_segment_only_timing_and_source_end_clipping(self):
        actual = rows([
            {'tStartMs':9000,'dDurationMs':4000,'segs':[{'utf8':'One'},{'utf8':' two'}]},
            {'tStartMs':11000,'dDurationMs':1000,'segs':[{'utf8':'outside'}]}])
        self.assertEqual(actual[0]['words'],[])
        self.assertEqual(actual[0]['alignment']['status'],'caption_segment_times')
        self.assertEqual(actual[0]['end'],10)

    def test_malformed_word_offsets_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'Unordered'):
            rows([{'tStartMs':0,'dDurationMs':5000,'segs':[
                {'utf8':'one','tOffsetMs':1000},{'utf8':' two','tOffsetMs':500}]}])
        with self.assertRaisesRegex(ValueError,'no text'):
            rows([{'tStartMs':100,'segs':[{'utf8':'\n'}]}])

    def test_registered_youtube_track_preferred_and_legacy_asr_loads(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            atomic_json(root/'pass1_manifest.json',{'producers':{}})
            self.assertEqual(selected_commentary(root),{'status':'not_run'})
            atomic_json(root/'producers/asr/report.json',{'commentary_path':'asr.jsonl'})
            self.assertEqual(selected_commentary(root)['commentary_path'],'asr.jsonl')
            atomic_json(root/'pass1_manifest.json',{'producers':{
                'youtube_captions':{'report':{'commentary_path':'youtube.jsonl'}}}})
            self.assertEqual(selected_commentary(root)['commentary_path'],'youtube.jsonl')


    def test_import_is_repeatable_and_binds_video_identity(self):
        from annotation.league.pass1_youtube_captions import import_captions
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            atomic_json(root/'manifest.json',{'source':{'sha256':'source','video_id':'video'},
                                              'interval':[0,10]})
            atomic_json(root/'pass1_manifest.json',{'producers':{}})
            track=root/'captions.json3'
            atomic_json(track,{'events':[{'tStartMs':0,'dDurationMs':1000,
                                         'segs':[{'utf8':'Hello'}]}]})
            with self.assertRaisesRegex(ValueError,'video ID'):
                import_captions(root,track,video_id='wrong',language='en-orig')
            first=import_captions(root,track,video_id='video',language='en-orig')
            before=read_json(root/'pass1_manifest.json')
            second=import_captions(root,track,video_id='video',language='en-orig')
            self.assertEqual(first,second)
            self.assertEqual(before,read_json(root/'pass1_manifest.json'))
            self.assertEqual(selected_commentary(root)['transcript_source'],'youtube_auto_captions')


class CaptionEndTests(unittest.TestCase):
    def test_speech_end_trims_caption_without_changing_text_or_onsets(self):
        from annotation.league.pass1_youtube_captions import estimate_caption_ends
        cue={'start':0,'end':5,'text':'hello world','words':[{'text':'world','start':1,'end':None}]}
        result=estimate_caption_ends([cue],[{'start':0,'end':1.6,'status':'transcribed'}],'coverage.jsonl')[0]
        self.assertEqual(result['end'],1.6)
        self.assertEqual(result['text'],cue['text'])
        self.assertEqual(result['words'],cue['words'])
        self.assertEqual(cue['end'],5)
        self.assertEqual(result['end_timing']['evidence_path'],'coverage.jsonl')

    def test_model_chunks_and_brief_pauses_are_merged(self):
        from annotation.league.pass1_youtube_captions import estimate_caption_ends
        cue={'start':0,'end':4,'words':[{'start':.8}]}
        coverage=[{'start':0,'end':1,'status':'transcribed'},
                  {'start':1,'end':2,'status':'transcribed'},
                  {'start':2.1,'end':3,'status':'unresolved_speech'}]
        result=estimate_caption_ends([cue],coverage)[0]
        self.assertEqual(result['end'],3)

    def test_estimate_never_extends_a_caption_or_precedes_last_word(self):
        from annotation.league.pass1_youtube_captions import estimate_caption_ends
        cue={'start':0,'end':2,'words':[{'start':1}]}
        for coverage in ([],[{'start':0,'end':.5,'status':'transcribed'}],
                           [{'start':0,'end':3,'status':'transcribed'}]):
            result=estimate_caption_ends([cue],coverage)[0]
            self.assertEqual(result['end'],2)
            self.assertEqual(result['end_timing']['method'],'caption_boundary')

    def test_segment_only_captions_use_the_existing_boundary(self):
        from annotation.league.pass1_youtube_captions import estimate_caption_ends
        result=estimate_caption_ends([{'start':0,'end':5,'words':[]}],
                                    [{'start':0,'end':1,'status':'transcribed'}])[0]
        self.assertEqual(result['end'],5)


if __name__ == '__main__':
    unittest.main()
