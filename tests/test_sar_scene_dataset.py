"""整景读取验收：完整标签覆盖、边缘上下文、缓存、训练归一化和训练闭环。"""
import csv
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.transform import from_origin
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from supervised.load.sar_scene_dataset import SARSceneDataset
from supervised.train import WaterSegmentationTrainer


class TinySceneNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1)

    def forward(self, x):
        return self.conv(x)


class SceneDatasetTests(unittest.TestCase):
    def prepare(self, root):
        records, arrays = [], {}
        for split, value in [('train', 2), ('val', 1000)]:
            name = split+'.tif'
            x = np.full((35,47), value, dtype='float32'); x[0,0] = 0; x[1,1] = np.nan
            y = np.full(x.shape, 255, dtype='uint8'); y[2:33,3:40] = 0; y[8:30,10:35] = 1
            arrays[split] = (x,y)
            for kind, data, nodata in [('images',x,np.nan), ('labels',y,255)]:
                folder = root/kind/split; folder.mkdir(parents=True)
                with rasterio.open(folder/name,'w',driver='GTiff',height=35,width=47,count=1,
                                   dtype=data.dtype,crs='EPSG:32650',transform=from_origin(100,200,3,3),nodata=nodata) as dst:
                    dst.write(data,1)
            records.append(dict(name=name,label_name=name,split=split,active=1,scene_id=split,group_id=split,
                                water_pixels=int((y==1).sum()),background_pixels=int((y==0).sum())))
        with (root/'split_manifest.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=records[0]);writer.writeheader();writer.writerows(records)
        return arrays

    def test_core_coverage_values_cache_and_train_only_stats(self):
        with TemporaryDirectory() as temp:
            root=Path(temp); arrays=self.prepare(root)
            ds=SARSceneDataset(root,root/'cache',window_size=32,halo=8,normalization_windows=100)
            for split, ids in [('train',ds.train_ids),('val',ds.val_ids)]:
                seen=np.zeros((35,47),dtype='int64')
                for i in ids:
                    sample=ds[i]; sid,top,left,h,w,*_=map(int,ds.windows[i])
                    core=sample['label'][8:8+h,8:8+w].numpy()
                    expected=arrays[split][1][top:top+h,left:left+w]
                    np.testing.assert_array_equal(core,np.where(expected==255,-1,expected.astype('int64')))
                    seen[top:top+h,left:left+w] += core!=-1
                np.testing.assert_array_equal(seen,(arrays[split][1]!=255).astype('int64'))
            sample=ds[0]
            self.assertTrue(sample['valid_mask'][8,8])
            self.assertEqual(float(sample['image'][0,8,8]),0)
            self.assertFalse(sample['valid_mask'][9,9])
            self.assertEqual(int(sample['label'][:8].max()),-1)
            mean,std,ids=ds.normalization(ds.train_ids)
            self.assertLess(mean[0],3)  # val=1000未进入统计。
            self.assertTrue(set(ids)<=set(ds.train_ids)); self.assertTrue(np.isfinite(std).all())
            with patch('supervised.load.sar_scene_dataset.rasterio.open',side_effect=AssertionError('应复用缓存')):
                cached=SARSceneDataset(root,root/'cache',window_size=32,halo=8,normalization_windows=100)
                np.testing.assert_array_equal(cached.windows,ds.windows)
                np.testing.assert_array_equal(cached.normalization(cached.train_ids)[0],mean)
            ds.close()

    def test_real_training_checkpoint_and_pixel_denominator(self):
        with TemporaryDirectory() as temp:
            root=Path(temp); arrays=self.prepare(root)
            trainer=WaterSegmentationTrainer(TinySceneNet,{},dict(model_name='tiny'),root/'images',root/'labels',
                root/'runs',bands=(1,),batch_size=2,epochs=3,loss_name='bce',sampling='hard',hard_warmup_epochs=1,
                scene_options=dict(cache_dir=root/'cache',window_size=32,halo=8,normalization_windows=2,num_workers=0,windows_per_epoch=4))
            trainer.train()
            trainer.dataset.close()
            self.assertEqual(trainer.phase_metrics['val']['valid_pixels'],int((arrays['val'][1]!=255).sum()))
            ckpt=torch.load(trainer.run_dir/'last.pth',weights_only=True,map_location='cpu')
            self.assertEqual(ckpt['data_layout'],'full_scene')
            self.assertEqual(len(trainer.sampling_plan),4)
            self.assertEqual(ckpt['epoch'],3)
            self.assertTrue((trainer.run_dir/'val_scenes_epoch_003.csv').exists())
            self.assertEqual(ckpt['val_scene_files'],['val.tif'])
            net=TinySceneNet();net.load_state_dict(ckpt['model_state_dict'])
            trainer.dataset.close()


if __name__=='__main__':
    unittest.main()
