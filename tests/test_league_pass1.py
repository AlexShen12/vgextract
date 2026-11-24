"""Pass 1 integration contracts: conservative identities, continuity and portability."""
import copy
import json
from pathlib import Path
import tempfile
import shutil
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from annotation.league.core import Store, atomic_json, write_rows
from annotation.league.pass1_timeline import (ContextIndex, inventory_bindings, item_history,
                                              map_context, scoreboard_openings)


def snapshot(time, champion='Darius', side='right', slot_id='right:row1', item='1055'):
    return {'id':f'snapshot-{time}-{slot_id}','time':time,'route':'scoreboard','slot_id':slot_id,
        'side':side,'champion':champion,'playback_segment_id':'p','continuity_verified':True,
        'slots':[{'slot':str(i),'status':'recognized' if i==0 else 'empty',
                  'item_id':item if i==0 else None,'observation_id':f'item-{time}-{slot_id}-{i}'} for i in range(7)],
        'evidence_ids':[f'portrait-{time}-{slot_id}']}


def name(time,value='Player',slot_id='right:row1'):
    return {'id':f'name-{time}-{slot_id}','time':time,'slot_id':slot_id,'field':'player_name',
            'value':value,'visibility':'visible','acceptance':'accepted','review_status':'unreviewed'}


class TimelineTests(unittest.TestCase):
    def test_dense_tail_accounts_for_samples_after_video_ends(self):
        from annotation.league.pass1_dense import unavailable_tail_samples

        frames = [{'time': time, 'id': str(i)} for i, time in enumerate([.014, .064, .131])]
        tail = unavailable_tail_samples(0, .266, frames, .181)
        self.assertEqual([row['time'] for row in tail], [.1875, .25])
        self.assertEqual([row['rates'] for row in tail], [[16], [16, 4]])
        self.assertEqual(tail[-1]['evidence_ids'], ['2'])
        with self.assertRaisesRegex(ValueError, 'before the last source frame'):
            unavailable_tail_samples(0, .266, frames[1:], .181)

    def test_context_never_extrapolates(self):
        index=ContextIndex([{'start':1,'end':2}])
        self.assertIsNone(index.at(.9));self.assertIsNone(index.at(2));self.assertIsNotNone(index.at(1.5))

    def test_context_preserves_original_and_selection_boundary(self):
        playback=[{'start':0,'end':2,'playback_segment_id':'canonical','continuity_verified':True}]
        selections=[{'start':0,'end':1,'owner_epoch':0},{'start':1,'end':2,'owner_epoch':1}]
        rows=map_context([{'time':t,'owner_id':'selected','segment_id':'original','owner_epoch':42} for t in [.5,1.5,3]],playback,selections)
        self.assertEqual([r['owner_epoch'] for r in rows[:2]],[0,1])
        self.assertEqual(rows[0]['producer_playback_segment_id'],'original')
        self.assertFalse(rows[2]['continuity_verified'])

    def test_exact_name_backfill_preserves_original_items(self):
        original=[snapshot(1,None),snapshot(2),snapshot(3)]
        fixed,links=inventory_bindings(original,[name(1),name(2),name(3)])
        self.assertEqual(fixed[0]['champion'],'Darius')
        self.assertIsNone(original[0]['champion'])
        self.assertEqual(fixed[0]['slots'],original[0]['slots'])
        self.assertEqual(links[0]['evidence_available_at'],3)

    def test_row_position_and_roster_are_not_bindings(self):
        fixed,links=inventory_bindings([snapshot(1,None),snapshot(2),snapshot(3)],[])
        self.assertIsNone(fixed[0]['champion']);self.assertEqual(links,[])

    def test_one_read_or_conflicting_name_cannot_backfill(self):
        for snapshots in ([snapshot(1,None),snapshot(2)],
                          [snapshot(1,None),snapshot(2),snapshot(3,'Ahri')]):
            fixed,links=inventory_bindings(snapshots,[name(1),name(2),name(3)])
            self.assertIsNone(fixed[0]['champion']);self.assertFalse(links)

    def test_name_binding_survives_row_swap_but_not_segment_change(self):
        rows=[snapshot(1,None),snapshot(2),snapshot(3,slot_id='right:row2')]
        fixed,links=inventory_bindings(rows,[name(1),name(2),name(3,slot_id='right:row2')])
        self.assertEqual(fixed[0]['champion'],'Darius')
        rows[-1]['playback_segment_id']='other'
        fixed,links=inventory_bindings(rows,[name(1),name(2),name(3,slot_id='right:row2')])
        self.assertIsNone(fixed[0]['champion'])

    def test_item_changes_bracket_only_same_segment(self):
        a,b,c=snapshot(1),snapshot(3,item='2003'),snapshot(5,item='3006')
        c['playback_segment_id']='other'
        result=item_history([a,b,c]);changes=[r for r in result if r['change_interval']]
        self.assertEqual([r['change_interval'] for r in changes],[[1,3]])

    def test_partial_slots_remain_independent(self):
        s=snapshot(1);s['slots'][0]['status']='unknown'
        result=item_history([s]);self.assertEqual(len(result),6)

    def visibility(self):
        return [{'id':str(i),'time':i/4,'status':'present' if i<4 or i>=8 else 'absent',
                 'playback_segment_id':'p','continuity_verified':True} for i in range(12)]

    def full_rows(self):
        return [snapshot(t,f'Champion{i}',side, f'{side}:row{i+1}') for t in [.5,2.5]
                for side in ['left','right'] for i in range(5)]

    def test_identical_scoreboard_reopens_merge_keep_closure(self):
        raw,merged=scoreboard_openings(self.visibility(),self.full_rows())
        self.assertEqual(len(raw),2);self.assertEqual(len(merged),1)
        self.assertEqual(merged[0]['closures'],[[1.,2.]])
        self.assertEqual(len(merged[0]['merged_opening_ids']),2)

    def test_change_unknown_or_gap_prevents_merge(self):
        for case in ['change','unknown','identity','continuity','too_long']:
            with self.subTest(case=case):
                rows=self.full_rows();vis=self.visibility()
                if case=='change':rows[-1]['slots'][0]['item_id']='2003'
                elif case=='unknown':rows[-1]['slots'][0]['status']='unknown'
                elif case=='identity':rows[-1]['champion']=None
                elif case=='continuity':vis[6]['continuity_verified']=False
                else:
                    for v in vis[8:]:v['time']+=3
                    for s in rows[10:]:s['time']+=3
                _,merged=scoreboard_openings(vis,rows)
                self.assertEqual(len(merged),2)


class BundleTests(unittest.TestCase):
    def test_release_gate_stays_closed_and_experimental_mode_is_explicit(self):
        from annotation.league.hybrid_intake import annotate
        with tempfile.TemporaryDirectory() as d:
            store=Store(Path(d));config=SimpleNamespace(fps=1,reader_workers=1)
            profile={'calibrated':True}
            from contextlib import ExitStack
            with ExitStack() as stack:
                for name,value in [('require_v2',{'interval':[0,1]}),('check_engine',None),
                                   ('load_profile',profile),('enabled_fields',[]),('runtime',({},None))]:
                    stack.enter_context(patch('annotation.league.hybrid_intake.'+name,return_value=value))
                config.layout_profile='profile'
                stack.enter_context(patch.object(store,'fingerprint',return_value='key'))
                gate=stack.enter_context(patch('annotation.league.hybrid_intake._require_release',side_effect=ValueError('release-gate')))
                with self.assertRaisesRegex(ValueError,'release-gate'):annotate(store,config)
                gate.assert_called_once()
                reader=stack.enter_context(patch('annotation.league.hybrid_intake._IntakeReader')).return_value
                reader.read.return_value={'id':'task','start':0,'status':'ok','playback_tracker':{}}
                reader.report.return_value={'operational_passed':True,'frames':1,'quality_benchmark_passed':False}
                stack.enter_context(patch('annotation.league.hybrid_intake.decoded_frames',return_value=[(0,None)]))
                result=annotate(store,config,experimental=True)
                self.assertTrue(result['experimental']);self.assertFalse(result['release_accepted'])
                self.assertFalse(result['quality_benchmark_passed']);gate.assert_called_once()

    def test_draft_finalization_is_repeatable_portable_and_detects_staleness(self):
        from annotation.league import pass1
        from annotation.league.media import file_hash
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'original';root.mkdir();(root/'source').mkdir()
            video=root/'source/video.mp4';video.write_bytes(b'test source')
            (root/'crop.png').write_bytes(b'test crop')
            atomic_json(root/'manifest.json',{'source':{'path':'source/video.mp4','sha256':file_hash(video)},'interval':[0,1]})
            atomic_json(root/'pass1_manifest.json',{'source_sha256':file_hash(video),'producers':{},
                'full_run_budget_approved':False,'final_revision':None})
            atomic_json(root/'profiles'/pass1.PROFILE,{'fields':[{'id':'f'}]})
            atomic_json(root/'code_checksums.json',{})
            data={k:[] for k in pass1.OUTPUTS}
            observation={'id':'o','field_id':'f','field':'clock','time':0,'frame_id':'fr','source_id':'s','task_id':'t',
                         'acceptance':'unknown','evidence_path':'crop.png'}
            data['observations']=[observation];data['normalized_observations']=[observation]
            with patch.object(pass1,'snapshot_code'),patch.object(pass1,'derive',return_value=data):
                first=pass1.finalize(root,draft=True);second=pass1.finalize(root,draft=True)
                self.assertEqual(first['revision'],second['revision'])
            self.assertTrue(pass1.verify(root)['passed'])
            moved=Path(d)/'moved bundle';shutil.copytree(root,moved)
            shutil.rmtree(root)
            self.assertTrue(pass1.verify(moved)['passed'])
            atomic_json(moved/'corrections/00000001.json',{'target':'observations','record_id':'o',
                'changes':{'value':1},'review_status':'human-verified'})
            self.assertIn('derived revision is stale; finalize again',pass1.verify(moved)['errors'])
            (moved/'crop.png').write_bytes(b'modified crop')
            self.assertTrue(any('bundle checksum' in e for e in pass1.verify(moved)['errors']))

    def test_cross_node_resume_preserves_original_task_provenance(self):
        from annotation.league.experimental_resume import reusable_tasks
        from annotation.league.core import digest
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);store=Store(root);profile={'fields':['clock']}
            old={'vl':{'runtime':{'gpu':'L40S','gpu_uuid':'old','total_memory_bytes':40,'cuda':'13'},'checkpoint':'pinned'}}
            new=copy.deepcopy(old);new['vl']['runtime'].update(gpu_uuid='new',total_memory_bytes=41)
            atomic_json(root/'intake-runtimes/r.json',old)
            task={'id':'original','run_key':'old','runtime_id':'r','status':'ok','start':1.,'sample_time':1.,
                  'profile_hash':digest(profile),'experimental_reader_signature':'reader-v1'}
            atomic_json(root/'tasks/original.json',task)
            with patch('annotation.league.experimental_resume.reader_signature',return_value='reader-v1'):
                tasks,provenance=reusable_tasks(store,profile,new,'new-key',[1.])
                self.assertEqual(len(tasks),1);self.assertEqual(next(iter(tasks.values()))['id'],'original')
                self.assertEqual(next(iter(provenance.values()))['original_run_key'],'old')
                new['vl']['checkpoint']='different-model'
                self.assertEqual(reusable_tasks(store,profile,new,'new-key',[1.])[0],{})
            with patch('annotation.league.experimental_resume.reader_signature',return_value='reader-v2'):
                self.assertEqual(reusable_tasks(store,profile,old,'new-key',[1.])[0],{})

    def test_resource_parser_and_required_vl_visibility(self):
        from annotation.league.readers import parse_text
        from annotation.league.vl_reader import validate_transcriptions
        self.assertEqual(parse_text('179/348','resource')['current'],179)
        self.assertEqual(parse_text('0/0','resource')['maximum'],0)
        with self.assertRaises(ValueError):parse_text('349/348','resource')
        field={'id':'r','parser':'resource','required_when_panel_present':True}
        response={'finish_reason':'stop','text':json.dumps({'readings':[{'field_id':'r','status':'not_shown','text':None}]})}
        self.assertEqual(validate_transcriptions(response,[field])['r']['acceptance'],'unknown')

    def test_full_run_requires_budget_even_after_pilot(self):
        from annotation.league.pass1 import execute
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);atomic_json(root/'manifest.json',{'interval':[0,1732.765896]})
            with self.assertRaisesRegex(ValueError,'budget-approved'):execute(root,pilot=False)

    def test_path_projection_keeps_original_unchanged(self):
        from annotation.league.pass1 import relocate
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'crop.png').write_bytes(b'png')
            row={'id':'stable','evidence_path':str(root/'crop.png')}
            projected=relocate(row,root,Path('producers/import'))
            self.assertEqual(projected['evidence_path'],'producers/import/crop.png')
            self.assertEqual(row['evidence_path'],str(root/'crop.png'))

    def test_import_json_is_not_hardlinked(self):
        from annotation.league.pass1 import copy_file
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);a=root/'a.json';b=root/'nested/b.json';a.write_text('{}')
            copy_file(str(a),str(b));b.write_text('{"changed":true}')
            self.assertEqual(a.read_text(),'{}')

    def test_revision_verifier_catches_missing_crops(self):
        from annotation.league.pass1 import verify_derived
        with tempfile.TemporaryDirectory() as d:
            data={'observations':[{'id':'o','frame_id':'f','source_id':'s','task_id':'t','evidence_path':'missing.png'}]}
            result=verify_derived(Path(d),data)
            self.assertFalse(result['passed'])

    def test_review_bundle_relocation_and_path_traversal(self):
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from annotation.league.pass1_review import install_pass1_routes
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);write_rows(root/'states.jsonl',[{'id':'x','time':2}])
            app=FastAPI();install_pass1_routes(app,Store(root));client=TestClient(app)
            self.assertEqual(client.get('/api/pass1/records?target=states').json()['total'],1)
            self.assertEqual(client.get('/api/pass1/records?target=../config').status_code,400)
            self.assertEqual(client.get('/pass1').status_code,200)


if __name__=='__main__':unittest.main()
