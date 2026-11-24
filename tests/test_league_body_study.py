import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image,ImageDraw

from annotation.league.body_study import extract,summarize_vectors
from annotation.league.core import atomic_json,digest
from annotation.league.media import file_hash


class BodyStudy(unittest.TestCase):
    def fixture(self,root):
        labels=root/'labels';labels.mkdir()
        image=Image.new('RGB',(40,40),'red');image.save(root/'frame.png')
        mask=Image.new('L',(40,40));ImageDraw.Draw(mask).rectangle((10,10,19,19),fill=255);mask.save(labels/'mask.png')
        refs=[{'time':0,'visibility':'visible','mask_path':'mask.png','labeler':'draft'},
              {'time':1,'visibility':'occluded','labeler':'draft'}]
        data={'clips':[{'references':refs}]};atomic_json(root/'labels.json',data)
        frame={'id':'frame','time':0,'native_path':str(root/'frame.png'),'native_sha256':file_hash(root/'frame.png')}
        clip={'id':'clip','references':refs,'frames':[frame,{**frame,'id':'frame2','time':1}],
              'arms':{'verified':{'status':'prepared','seed':{'champion':'Darius'},'frames':[frame]},
                      'automatic':{'status':'unseeded'}}}
        (root/'verified').mkdir();atomic_json(root/'verified/clip-complete.json',{'frames':[{'frame_index':0,
            'masks':[{'object_id':1,'mask_path':str(labels/'mask.png'),'sha256':file_hash(labels/'mask.png')}]}]})
        atomic_json(root/'prepared.json',{'root':str(root),'labels_dir':str(labels),'clips':[clip],
            'recipe':{'profile':{},'labels':digest({'labels':data,'masks':{'mask.png':file_hash(labels/'mask.png')}})}})
        return {'sam_prepared':str(root/'prepared.json'),'masks':[]}

    def test_shared_geometry_missing_seeds_and_occlusions_are_preserved(self):
        with tempfile.TemporaryDirectory() as name,patch('annotation.league.body_identity.healthbar_proposals',return_value=[]):
            root=Path(name);result=extract(self.fixture(root),root/'output');rows=result['samples']
            self.assertEqual(len(rows),4);self.assertEqual({tuple(r['box']) for r in rows},{(10,10,20,20)})
            self.assertEqual(rows[-1]['status'],'missing_mask')
            self.assertEqual(result['localization'][0]['iou'],1)
            self.assertEqual(result['localization'][1]['iou'],0)
            self.assertEqual(len(result['localization']),4)
            self.assertTrue(all(r['human_verified'] is False for r in rows))

    def test_source_mask_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);spec=self.fixture(root);Image.new('L',(40,40),255).save(root/'labels/mask.png')
            with self.assertRaisesRegex(ValueError,'source labels changed'):extract(spec,root/'output')

    def test_single_champion_similarity_cannot_report_precision(self):
        rows=[{'representation':'box','clip_id':'a','vector':[1,0]},
              {'representation':'box','clip_id':'b','vector':[1,0]},
              {'representation':'box','clip_id':'c','status':'missing_mask'}]
        report=summarize_vectors(rows)['box']
        self.assertEqual(report['available'],2);self.assertEqual(report['missing'],1)
        self.assertIsNone(report['precision']);self.assertFalse(report['similarity_is_probability'])
