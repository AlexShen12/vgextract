from argparse import Namespace
from copy import deepcopy
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from annotation.league.core import atomic_json, read_json
from annotation.league.media import file_hash
from annotation.league.sam_finetune import (export, foreground_points, native_points,
    rle_decode, rle_encode, seed_tracker, validate_sources, verify_labels)
from annotation.league.sam_finetune_metrics import aggregate, associate_bar, instance_metrics


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch is optional for CPU dataset tooling')
class CheckpointTests(unittest.TestCase):
    def fixture(self):
        import torch
        from annotation.league.sam_finetune import CODE_REVISION
        from annotation.league.sam_finetune_train import freeze_backbones
        model=torch.nn.Module()
        model.backbone=torch.nn.Linear(2,2)
        model.segmentation_head=torch.nn.Linear(2,1)
        freeze_backbones(model)
        data=dict(base_sha256='base',labels_sha256='labels',code_revision=CODE_REVISION,
                  heads={n:p.detach().clone()+1 for n,p in model.named_parameters() if p.requires_grad})
        return model,data

    def test_head_reload_leaves_frozen_weights_unchanged(self):
        import torch
        from annotation.league.sam_finetune_train import apply_head_checkpoint,tensor_digest
        model,data=self.fixture()
        before=tensor_digest(model.backbone.named_parameters())
        apply_head_checkpoint(model,data,base_sha256='base',labels_sha256='labels')
        self.assertEqual(before,tensor_digest(model.backbone.named_parameters()))
        for n,p in model.named_parameters():
            if p.requires_grad:torch.testing.assert_close(p,data['heads'][n])

    def test_partial_or_wrong_provenance_checkpoint_rejected(self):
        from annotation.league.sam_finetune_train import apply_head_checkpoint
        model,data=self.fixture()
        for field in ('base_sha256','labels_sha256','code_revision'):
            changed=deepcopy(data);changed[field]='wrong'
            with self.assertRaises(ValueError):apply_head_checkpoint(model,changed,base_sha256='base',labels_sha256='labels')
        data['heads'].pop(next(iter(data['heads'])))
        with self.assertRaises(ValueError):apply_head_checkpoint(model,data,base_sha256='base',labels_sha256='labels')

    def test_bad_shape_does_not_partially_apply_checkpoint(self):
        import torch
        from annotation.league.sam_finetune_train import apply_head_checkpoint,tensor_digest
        model,data=self.fixture();before=tensor_digest(model.named_parameters())
        data['heads']['segmentation_head.bias']=torch.zeros(3)
        with self.assertRaises(ValueError):apply_head_checkpoint(model,data,base_sha256='base',labels_sha256='labels')
        self.assertEqual(before,tensor_digest(model.named_parameters()))


class FineTuneTests(unittest.TestCase):
    def fixture(self, root):
        images, reviews = [], {}
        for split in ('train', 'validation', 'test'):
            for negative in (False, True):
                name = f'{split}-{negative}'
                path = root / 'images' / (name+'.png')
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new('RGB', (30, 20)).save(path)
                row = dict(id=name, split=split, match_id=split, video_id=split,
                           source_sha256=split, negative=negative, crop_box=[10, 20, 40, 40],
                           path=str(path.relative_to(root)), path_sha256=file_hash(path))
                images.append(row)
                reviews[name] = dict(status='accepted', image_sha256=row['path_sha256'],
                    all_champions_labeled=True,
                    instances=[] if negative else [dict(track_id='target', polygons=[[[2,3],[14,3],[14,16],[2,16]]])])
        atomic_json(root/'dataset.json', dict(images=images, clips=[]))
        atomic_json(root/'reviews.json', dict(dataset_sha256=file_hash(root/'dataset.json'), reviewer='Codex',
                                             reviewer_kind='agent', images=reviews))
        return images

    def test_export_preserves_negatives_and_rle(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); self.fixture(root)
            result=export(Namespace(root=root))
            self.assertEqual(result['counts']['train'], {'images':2, 'instances':1})
            doc=read_json(root/'coco/train/_annotations.coco.json')
            self.assertEqual(int(rle_decode(doc['annotations'][0]['segmentation']).sum()),13*14)
            self.assertEqual(verify_labels(root),result)

    def test_changed_frozen_reviews_rejected_before_reexport(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); self.fixture(root); export(Namespace(root=root))
            before=file_hash(root/'coco/train/_annotations.coco.json')
            doc=read_json(root/'reviews.json'); doc['reviewer']='changed';atomic_json(root/'reviews.json',doc)
            with self.assertRaises(ValueError):export(Namespace(root=root))
            self.assertEqual(file_hash(root/'coco/train/_annotations.coco.json'),before)

    def test_review_gate_and_split_leakage(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); self.fixture(root)
            path=root/'reviews.json';doc=read_json(path);doc['images']['train-False']['status']='unreviewed';atomic_json(path,doc)
            with self.assertRaisesRegex(ValueError,'Unreviewed'):export(Namespace(root=root))
            self.fixture(root)
            path=root/'dataset.json';doc=read_json(path);doc['images'][2]['match_id']='train';atomic_json(path,doc)
            path=root/'reviews.json';doc=read_json(path);doc['dataset_sha256']=file_hash(root/'dataset.json');atomic_json(path,doc)
            with self.assertRaisesRegex(ValueError,'leaks'):export(Namespace(root=root))

    def test_coco_image_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); self.fixture(root);export(Namespace(root=root))
            Image.new('RGB',(30,20),'red').save(root/'coco/train/train-False.png')
            with self.assertRaisesRegex(ValueError,'COCO image'):verify_labels(root)

    def test_coordinate_roundtrip_and_interior_sampling(self):
        mask=np.zeros((40,60),bool);mask[5:35,10:50]=True;mask[10:30,20:40]=False
        points=foreground_points(mask)
        self.assertTrue(all(mask[y,x] for x,y in points))
        box=[123,456,183,496]
        self.assertEqual(native_points(native_points(points,box),box,inverse=True),points)
        with self.assertRaises(ValueError):foreground_points(np.zeros_like(mask))
        edge=np.ones((4,4),bool)
        self.assertTrue(all(edge[y,x] for x,y in foreground_points(edge)))

    def test_tracker_seed_does_not_require_detector(self):
        class Predictor:
            def handle_request(self, request): self.request=request;return request
        predictor=Predictor();mask=np.ones((20,30),bool)
        result=seed_tracker(predictor,'session',4,7,mask,[100,200,130,220],[1920,1080])
        self.assertEqual(result['obj_id'],7)
        self.assertNotIn('text',result)
        self.assertNotIn('bounding_boxes',result)
        self.assertEqual(result['point_labels'],[1]*len(result['points']))

    def test_metrics_penalize_missing_target_and_negative_fp(self):
        mask=np.zeros((20,30),bool);mask[4:16,5:20]=True
        score=instance_metrics([mask],[])
        self.assertEqual(score['instances'][0]['iou'],0.)
        result=aggregate([score,instance_metrics([], [mask])])
        self.assertEqual(result['mean_instance_iou'],0.)
        self.assertEqual(result['negative_false_positive_images'],1)
        self.assertEqual(instance_metrics([mask],[mask])['instances'][0]['boundary_f'],1.)

    def test_rle_orientation(self):
        mask=np.array([[1,0,0],[1,1,0]],bool)
        np.testing.assert_array_equal(mask,rle_decode(rle_encode(mask)))
        from pycocotools import mask as api
        decoded=api.decode(api.frPyObjects(rle_encode(mask),2,3))
        np.testing.assert_array_equal(mask,decoded)

    def test_bar_association_abstains_on_competing_bodies(self):
        left=np.zeros((100,100),bool);left[30:65,30:50]=True
        right=np.zeros_like(left);right[30:65,50:70]=True
        self.assertEqual(associate_bar([left],[30,10,50,20]),0)
        self.assertIsNone(associate_bar([left,right],[40,10,60,20]))

    def test_source_duplicates_rejected(self):
        path=Path(__file__).parents[1]/'annotation/league/sam_finetune_sources.json'
        sources=read_json(path)['sources'];validate_sources(sources)
        sources=deepcopy(sources);sources[1]['match_id']=sources[0]['match_id']
        with self.assertRaisesRegex(ValueError,'Duplicate'):validate_sources(sources)


if __name__ == '__main__':
    unittest.main()
