"""多目录整景训练：来源、分组、数值协议、缓存与实验清单备份。"""
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np
import torch
import test_sar_scene_dataset as fixtures
from test_sar_scene_dataset import TinySceneNet
from supervised.load.sar_scene_dataset import SARSceneDataset, merge_scene_manifests, check_preprocessed_input
from supervised.train import WaterSegmentationTrainer


class MultiManifestTests(unittest.TestCase):
    def prepare(self, folder):
        paths = []
        for tag in ('one', 'two'):
            root = folder/tag
            fixtures.SceneDatasetTests().prepare(root)
            path = root/'split_manifest.csv'
            with path.open(encoding='utf-8-sig', newline='') as f:
                rows = list(csv.DictReader(f))
            for row in rows:
                row['scene_id'] = tag+'_'+row['scene_id']
                row['group_id'] = tag+'_'+row['group_id']
            self.write_rows(path, rows)
            (root/'preprocessing.json').write_text(json.dumps(dict(stage='normalized', complete=True,
                mean=[2.], std=[3.], bands=[1], quantity='uncalibrated_DN')), encoding='utf-8')
            ds = SARSceneDataset(root, folder/'cache', bands=(1,), window_size=32, halo=8)
            ds.close()
            paths.append(path)
        return paths

    def write_rows(self, path, rows):
        with path.open('w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    def options(self, folder):
        return dict(cache_dir=folder/'cache', bands=(1,), window_size=32, halo=8, require_prepared=True)

    def test_merge_cache_pixels_and_checkpoint_backup(self):
        with TemporaryDirectory() as tmp, patch.object(torch.cuda, 'is_available', return_value=False):
            folder=Path(tmp); paths=self.prepare(folder)
            options=self.options(folder)
            # 合并过程不读取影像像素，复用各目录窗口缓存。
            with patch('supervised.load.sar_scene_dataset.rasterio.open', side_effect=AssertionError('unexpected pixel read')):
                root=merge_scene_manifests(paths, **options)
                ds=SARSceneDataset(root, **options)
            self.assertEqual(len(ds.scenes),4)
            self.assertTrue(all(Path(r['name']).is_absolute() for r in ds.scenes))
            for i in range(len(ds)):
                sample=ds[i]
                self.assertEqual(sample['scene_id'],ds.scenes[int(ds.windows[i,0])]['scene_id'])
            ds.close()
            torch.set_num_threads(2)
            trainer=WaterSegmentationTrainer(TinySceneNet, {}, {'model_name':'tiny'},
                paths[0].parent/'images', paths[0].parent/'labels', folder/'runs',
                bands=(1,), batch_size=2, epochs=1, loss_name='bce', require_preprocessed=True,
                require_normalized=True, scene_options=dict(cache_dir=folder/'cache', window_size=32,
                halo=8, num_workers=0, windows_per_epoch=4, manifest_paths=paths))
            trainer.train()
            ckpt=torch.load(trainer.run_dir/'last.pth', map_location='cpu', weights_only=True)
            self.assertEqual(Path(ckpt['image_dir']).parent,trainer.run_dir)
            check_preprocessed_input(Path(ckpt['image_dir']),ckpt)
            saved=SARSceneDataset(trainer.run_dir, **options)
            self.assertEqual(saved.names,trainer.dataset.names)
            self.assertEqual(len(ckpt['val_scene_files']),2)
            self.assertEqual(trainer.phase_metrics['val']['valid_pixels'],2*31*37)
            saved.close()
            resumed=WaterSegmentationTrainer(TinySceneNet, {}, {'model_name':'tiny'},
                paths[0].parent/'images', paths[0].parent/'labels', folder/'runs',
                bands=(1,), batch_size=2, epochs=1, loss_name='bce', require_preprocessed=True,
                require_normalized=True, resume_checkpoint=trainer.run_dir/'last.pth',
                scene_options=dict(cache_dir=folder/'cache', window_size=32, halo=8,
                num_workers=0, windows_per_epoch=4, manifest_paths=paths))
            self.assertEqual(resumed.start_epoch,2)
            self.assertEqual(resumed.run_dir,trainer.run_dir)
            resumed.dataset.close()
            # 原始manifest未被重写为绝对路径。
            with paths[0].open(encoding='utf-8-sig') as f:
                self.assertFalse(Path(next(csv.DictReader(f))['name']).is_absolute())

    def test_reject_mismatched_normalization_and_incomplete(self):
        with TemporaryDirectory() as tmp:
            folder=Path(tmp);paths=self.prepare(folder);p=paths[1].parent/'preprocessing.json'
            record=json.loads(p.read_text());record['mean']=[99.];p.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError,'mean/std'):
                merge_scene_manifests(paths, **self.options(folder))
            record['mean']=[2.];record['complete']=False;p.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError,'已完成'):
                merge_scene_manifests(paths, **self.options(folder))

    def test_reject_duplicate_and_cross_source_group_leakage(self):
        with TemporaryDirectory() as tmp:
            folder=Path(tmp);paths=self.prepare(folder)
            with paths[1].open(encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
            rows[0]['scene_id']='one_train';self.write_rows(paths[1],rows)
            with self.assertRaisesRegex(ValueError,'重复景号'):
                merge_scene_manifests(paths, **dict(self.options(folder),require_prepared=False))
            rows[0]['scene_id']='two_train';rows[1]['group_id']='one_train';self.write_rows(paths[1],rows)
            with self.assertRaisesRegex(ValueError,'跨train/val'):
                merge_scene_manifests(paths, **dict(self.options(folder),require_prepared=False))

if __name__=='__main__':
    unittest.main()
