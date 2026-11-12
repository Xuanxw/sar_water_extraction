"""融合阈值扫描：原像素分母、三模型同范围、BC3子集与逐景续跑。"""
import csv
import hashlib
import json
import shutil
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / 'src'), str(Path(__file__).resolve().parent)]
from supervised.evaluation import model_eval as scan
import test_sar_scene_dataset as fixture


class ScanTests(unittest.TestCase):
    def test_counts_equal_direct_thresholds_and_ignore_pixels(self):
        thresholds = [.2, .5, .8]
        edges = np.log(np.array(thresholds) / (1 - np.array(thresholds)))
        scores = np.array([[-2., edges[0], 0., edges[-1]], [2., np.nan, -2., 1.]])
        labels = np.array([[0, 1, 0, 1], [1, -1, 1, 0]])
        valid = np.ones(labels.shape, bool); valid[1, -1] = False
        actual = scan.fused_threshold_counts(scores, labels, valid, edges)
        good = valid & np.isin(labels, (0, 1))
        for i, edge in enumerate(edges):
            expected = np.bincount(2 * labels[good] + (scores[good] >= edge), minlength=4)
            np.testing.assert_array_equal(actual[i], expected)
        np.testing.assert_array_equal(scan.fused_threshold_counts(scores, labels, np.zeros_like(valid), edges), 0)
        scores[0, 0] = np.nan
        with self.assertRaises(FloatingPointError):
            scan.fused_threshold_counts(scores, labels, valid, edges)

    def test_fusion_resume_subsets_and_protocol_guard(self):
        with TemporaryDirectory() as temp, patch.object(torch.cuda, 'is_available', return_value=False):
            root = Path(temp)
            fixture.SceneDatasetTests().prepare(root)
            manifest = root / 'split_manifest.csv'
            with manifest.open(encoding='utf-8-sig', newline='') as f:
                rows = list(csv.DictReader(f))
            rows[1]['scene_id'] = 'BC3-test'
            second = dict(rows[1], scene_id='BC2-test', name='second.tif', label_name='second.tif')
            rows.append(second)
            for kind in ('images', 'labels'):
                shutil.copyfile(root / kind / 'val/val.tif', root / kind / 'val/second.tif')
            with manifest.open('w', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
            model = nn.Conv2d(1, 1, 1)
            with torch.no_grad():
                model.weight.fill_(.001); model.bias.fill_(-1)
            ckpt = dict(model_state_dict=model.state_dict(), data_layout='full_scene',
                image_dir=str(root/'images'), label_dir=str(root/'labels'), bands=(1,),
                window_size=32, window_halo=8, val_scene_files=['val.tif', 'second.tif'], val_files=['saved_window'],
                split_manifest_sha256=scan.file_hash(manifest), mean=torch.tensor([0.]), std=torch.tensor([1.]),
                batch_size=2, epoch=6, validation_inference=dict(stride=8, blend='hann', num_workers=0))
            checkpoints = {}
            for name, epoch in (('six',6), ('seven',7), ('bn',7)):
                ckpt['epoch'] = epoch
                checkpoints[name] = root / (name+'.pth')
                torch.save(ckpt, checkpoints[name])
            out = root / 'scan'
            options = dict(checkpoints=checkpoints, thresholds=[.2,.5,.8], output_dir=out, expected_epochs={})
            def load(checkpoint):
                result = nn.Conv2d(1,1,1); result.load_state_dict(checkpoint['model_state_dict']); return result
            real_inference = scan.iter_scene_predictions
            calls = 0
            def interrupted(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError('模拟断盘')
                yield from real_inference(*args, **kwargs)
            with patch.object(scan, 'load_model', side_effect=load), patch.object(scan, 'iter_scene_predictions', side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError, '模拟断盘'):
                    scan.run_threshold_scan(**options)
            self.assertEqual(len(list((out/'scene_counts').rglob('*.npz'))), 1)
            with patch.object(scan, 'load_model', side_effect=load), patch.object(scan, 'iter_scene_predictions', wraps=real_inference) as infer:
                scan.run_threshold_scan(**options)
                self.assertEqual(infer.call_count, 5)
            with (out/'threshold_metrics.csv').open(encoding='utf-8-sig', newline='') as f:
                metrics = list(csv.DictReader(f))
            self.assertEqual(len(metrics), 18)
            for row in metrics:
                self.assertEqual(int(row['valid_pixels']), 2294 if row['subset']=='all' else 1147)
            before = (out/'threshold_metrics.csv').read_bytes()
            with patch.object(scan, 'load_model', side_effect=AssertionError('已完成不能加载模型')):
                scan.run_threshold_scan(**options)
            self.assertEqual((out/'threshold_metrics.csv').read_bytes(), before)
            self.assertTrue(json.loads((out/'progress.json').read_text())['complete'])
            # 删除一景缓存只补这一景，其他模型/场景不重新推理。
            (out/'scene_counts/seven/BC3-test.npz').unlink()
            with patch.object(scan, 'load_model', side_effect=load), patch.object(scan, 'iter_scene_predictions', wraps=real_inference) as infer:
                scan.run_threshold_scan(**options)
                self.assertEqual(infer.call_count, 1)
            # 更换阈值、文件或权重不可接着累计旧结果。
            with self.assertRaisesRegex(ValueError, '续跑目录'):
                scan.run_threshold_scan(**dict(options, thresholds=[.3,.5,.8]))
            ckpt['epoch']=8; torch.save(ckpt, checkpoints['seven'])
            with self.assertRaisesRegex(ValueError, '续跑目录'):
                scan.run_threshold_scan(**options)


if __name__ == '__main__':
    unittest.main()
