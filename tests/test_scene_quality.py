"""整景质量验收：拼接、原验证口径、边界跨窗及无修改续跑。"""
import csv
import hashlib
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np
import torch
import rasterio
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'src'),str(Path(__file__).resolve().parent)]
from supervised.load.scene_windows import SceneWindows
from supervised.load.sar_scene_dataset import SARSceneDataset
from datasets.image_process import inspect_scene
from supervised.evaluation.scene_quality import run_scene_quality, clean_mask
from supervised.evaluation.boundary_eval import boundary_regions, confusion_counts
import test_sar_scene_dataset as fixtures


class QualityTests(unittest.TestCase):
    def test_window_matches_training_and_boundary_has_no_internal_seams(self):
        with TemporaryDirectory() as t:
            root=Path(t); arrays=fixtures.SceneDatasetTests().prepare(root)
            train=SARSceneDataset(root,root/'cache',window_size=32,halo=8)
            ds=SceneWindows(root/'images/val/val.tif',root/'labels/val/val.tif',window_size=32,halo=8)
            stitched=np.full((35,47),-1,np.int64); covered=np.zeros((35,47),int)
            bound=np.zeros((35,47),bool)
            mapping={train.names[i]:i for i in train.val_ids}
            for i in range(len(ds)):
                a=ds[i]; top,left,h,w=a['window'].tolist();y=a['label'].numpy()
                b=train[mapping[f'val_r{top}_c{left}']]
                torch.testing.assert_close(a['image'],b['image'])
                np.testing.assert_array_equal(y[8:8+h,8:8+w],b['label'][8:8+h,8:8+w])
                stitched[top:top+h,left:left+w]=y[8:8+h,8:8+w];covered[top:top+h,left:left+w]+=1
                region=boundary_regions(y,y>=0,(3,))[3]['inner']
                bound[top:top+h,left:left+w]=region[8:8+h,8:8+w]
            self.assertTrue((covered==1).all())
            np.testing.assert_array_equal(bound,boundary_regions(stitched,stitched>=0,(3,))[3]['inner'])
            train.close();ds.close()

    def test_end_to_end_counts_georeference_and_resume(self):
        with TemporaryDirectory() as t, patch('torch.cuda.is_available',return_value=False):
            root=Path(t); arrays=fixtures.SceneDatasetTests().prepare(root)
            model=fixtures.TinySceneNet();model.conv.weight.data.zero_();model.conv.bias.data.fill_(.2)
            c=dict(data_layout='full_scene',image_dir=str(root/'images'),label_dir=str(root/'labels'),
                split_manifest_sha256=hashlib.sha256((root/'split_manifest.csv').read_bytes()).hexdigest(),
                val_scene_files=['val.tif'],window_size=32,window_halo=8,bands=(1,),
                batch_size=2,mean=torch.tensor([0.]),std=torch.tensor([1.]),epoch=18,amp_enabled=False,
                evaluation_scope='all_active_labeled_scene_cores')
            ck=root/'model.pth';torch.save(c,ck)
            with patch('supervised.evaluation.scene_quality.load_model',return_value=model):
                output=run_scene_quality(ck,output_dir=root/'quality',thresholds=(.3,.5,.7),save_visuals=True,boundary_radii=(3,))
                with rasterio.open(output/'raw/val.tif') as src, rasterio.open(root/'images/val/val.tif') as original:
                    self.assertEqual(src.transform,original.transform);self.assertEqual(src.crs,original.crs)
                    raw=src.read(1);self.assertEqual(raw[0,0],1);self.assertEqual(raw[1,1],255)
                gt=arrays['val'][1];valid=np.isin(gt,(0,1))&np.isfinite(arrays['val'][0])
                expected=confusion_counts(gt,np.ones(gt.shape,bool),valid)
                with np.load(output/'counts/val.npz') as d:
                    np.testing.assert_array_equal(d['counts'][1],expected)
                    np.testing.assert_array_equal(d['post'],expected)
                stamp=(output/'raw/val.tif').stat().st_mtime_ns
                run_scene_quality(ck,output_dir=output,thresholds=(.3,.5,.7),save_visuals=True,boundary_radii=(3,))
                self.assertEqual(stamp,(output/'raw/val.tif').stat().st_mtime_ns)
                with self.assertRaisesRegex(ValueError,'配置'):
                    run_scene_quality(ck,output_dir=output,thresholds=(.2,.5,.7),boundary_radii=(3,))

    def test_morphology_uses_stitched_neighbors_across_windows(self):
        from supervised.evaluation.scene_quality import apply_scene_morphology
        from rasterio.transform import from_origin
        with TemporaryDirectory() as t:
            root=Path(t);raw=np.zeros((45,49),np.uint8)
            raw[8:38,8:38]=1;raw[19,19]=0;raw[4,4]=1
            valid=np.ones_like(raw,bool);valid[0,:]=False
            profile=dict(driver='GTiff',height=45,width=49,count=1,dtype='uint8',
                         crs='EPSG:32650',transform=from_origin(100,200,3,3),nodata=255)
            for name,data in [('raw',np.where(valid,raw,255).astype('uint8')),('label',raw)]:
                with rasterio.open(root/(name+'.tif'),'w',**profile) as dst:dst.write(data,1)
            windows=[(y,x,min(16,45-y),min(16,49-x)) for y in range(0,45,16) for x in range(0,49,16)]
            counts,_=apply_scene_morphology(root/'raw.tif',root/'label.tif',root/'post.tif',windows,(1,3),1,1)
            expected=clean_mask(raw==1,valid,1,1)
            with rasterio.open(root/'post.tif') as src:got=src.read(1)
            np.testing.assert_array_equal(got,np.where(valid,expected,255).astype('uint8'))
            np.testing.assert_array_equal(counts,confusion_counts(raw,expected,valid))

    def test_fixed_points_are_reproducible_and_capped(self):
        from supervised.evaluation.boundary_eval import BoundaryPointReservoir
        a=BoundaryPointReservoir('scene',3,'inner',16)
        b=BoundaryPointReservoir('scene',3,'inner',16)
        for top in (0,20,40):
            mask=np.ones((20,20),bool); label=np.ones((20,20),int)
            for reservoir in (a,b):
                reservoir.update(mask,label,label*.2,mask,mask,top,0)
        np.testing.assert_array_equal(a.points,b.points)
        self.assertEqual(len(a.points),16)
        self.assertEqual(a.seen,1200)
        self.assertEqual(len(set(map(tuple,a.points[:,:2]))),16)

    def test_invalid_neighborhood_not_filled_and_audit_keeps_zero(self):
        raw=np.ones((21,21),bool);raw[10,10]=False
        valid=np.ones_like(raw);valid[10,9]=False
        post=clean_mask(raw,valid,closing_radius=1)
        self.assertFalse(post[10,10]);self.assertFalse(post[10,9])
        with TemporaryDirectory() as t:
            root=Path(t);fixtures.SceneDatasetTests().prepare(root)
            stats=inspect_scene(root/'images/train/train.tif',root/'labels/train/train.tif')
            self.assertGreater(stats['zero_ratio'],0)
            self.assertEqual(stats['physical_domain'],'unconfirmed')


if __name__=='__main__':unittest.main()
