"""阈值与模型仅比较固定 active val，拒绝旧划分模型混入新验证。"""
import csv
import hashlib
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import rasterio
import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"),
                str(Path(__file__).resolve().parents[1] / "scripts")]
from thres_segmentation.pipeline import run_segmentation
from compare_threshold import run_comparison


class DarkWater(torch.nn.Module):
    def forward(self, image):
        return -image[:, :1]


class ValThresholdSplitTests(unittest.TestCase):
    def test_active_val_only_and_checkpoint_split_provenance(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            rows = [dict(name='train.tif', split='train', active=1),
                    dict(name='val.tif', split='val', active=1),
                    dict(name='duplicate.tif', split='val', active=0)]
            manifest = root / 'split_manifest.csv'
            with manifest.open('w', encoding='utf-8-sig', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            truth = np.zeros((17, 17), dtype='uint8')
            truth[5:12, 5:12] = 1
            for row in rows:
                data = np.where(truth, 1, 100).astype('float32')
                if row['split'] == 'train':
                    data += 10
                for kind, array in (('images', data), ('labels', truth)):
                    target = root / kind / row['split'] / row['name']
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with rasterio.open(target, 'w', driver='GTiff', width=17, height=17,
                                       count=1, dtype=str(array.dtype)) as dst:
                        dst.write(array, 1)
            options = dict(image_dir=root / 'images/val', label_dir=root / 'labels/val',
                           split_manifest=manifest, bands={'VV': 1}, use_landcover=False,
                           threshold_options=dict(lee_window=1, water_percent=25),
                           save_masks=False, max_visualizations=0)
            result = run_segmentation(output_dir=root / 'threshold', **options)
            self.assertEqual(result['threshold_VV']['valid_pixels'], 17 * 17)
            with (root / 'threshold/files.csv').open(encoding='utf-8-sig') as stream:
                self.assertEqual([r['name'] for r in csv.DictReader(stream)], ['val.tif'])
            with self.assertRaisesRegex(ValueError, 'images/val'):
                run_segmentation(**dict(options, image_dir=root / 'images/train'))
            checkpoint = dict(model_name='synthetic', epoch=1, bands=(1,),
                              mean=torch.tensor([50.]), std=torch.tensor([20.]),
                              image_dir=str(root / 'images'), label_dir=str(root / 'labels'),
                              train_files=['train.tif'], val_files=['val.tif'], ignore_index=-1,
                              split_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
            checkpoint_path = root / 'best_f1.pth'
            torch.save(checkpoint, checkpoint_path)
            comparison_options = dict(checkpoint_path=checkpoint_path, split_manifest=manifest,
                                      scene_csv=None, threshold_bands={'VV': 1}, use_amp=False,
                                      max_visualizations=0, save_masks=False,
                                      boundary_radii=(1,), boundary_points=8,
                                      threshold_options=dict(lee_window=1, water_percent=25))
            with patch('supervised.evaluation.postprocess.load_model', return_value=DarkWater()), \
                 patch.object(torch.cuda, 'is_available', return_value=False):
                results = run_comparison(output_dir=root / 'comparison', **comparison_options)
            whole = [r for r in results if r['subset'] == 'all']
            self.assertEqual(len(whole), 2)
            self.assertTrue(all(r['samples'] == 1 and r['valid_pixels'] == 289 for r in whole))
            with (root / 'comparison/samples.csv').open(encoding='utf-8-sig') as stream:
                self.assertEqual([r['name'] for r in csv.DictReader(stream)], ['val.tif'])
            for change in ({'split_manifest_sha256': ''}, {'val_files': ['val.tif', 'duplicate.tif']},
                           {'train_files': ['train.tif', 'val.tif']}):
                torch.save(dict(checkpoint, **change), checkpoint_path)
                with self.assertRaisesRegex(ValueError, '检查点与当前分组清单不一致'):
                    run_comparison(output_dir=root / 'rejected', **comparison_options)
                self.assertFalse((root / 'rejected').exists())

    def test_missing_checkpoint_requires_explicit_model(self):
        with self.assertRaisesRegex(ValueError, '新 grouped'):
            run_comparison(checkpoint_path=None)


if __name__ == '__main__':
    unittest.main()
