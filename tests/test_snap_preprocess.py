"""真实调用本机GPT的小样本验收：有效域、缓存和训练/推理输入一致性。"""
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.windows import Window
import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]/'src'), str(Path(__file__).resolve().parent)]
from datasets import snap_preprocess
from datasets.speckle import SpeckleFilter, read_filtered_window
from supervised.load.sar_scene_dataset import SARSceneDataset
from supervised.load.scene_windows import SceneWindows
from supervised.train import WaterSegmentationTrainer
from supervised.evaluation import model_eval
import test_sar_scene_dataset as fixtures

CONFIG = dict(method='snap_lee', input_domain='amplitude', window_size=3, enl=1.)


class TinySnapNet(torch.nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, num_classes, 1)

    def forward(self, image):
        return self.conv(image)


@unittest.skipUnless(snap_preprocess.GPT_PATH is not None and snap_preprocess.GPT_PATH.is_file(), '未显式配置真实SNAP GPT')
class SnapPreprocessTests(unittest.TestCase):
    def test_array_entry_and_amplitude_power_equivalence(self):
        with TemporaryDirectory() as temporary, patch.object(snap_preprocess, 'CACHE_DIR', Path(temporary)):
            amplitude = np.full((1, 32, 32), 3., dtype='float32')
            amplitude[0, 16, 16] = 30
            amplitude[0, 8:12, 8:12] = 0
            valid = np.ones((32, 32), bool); valid[:2, :2] = False
            a = SpeckleFilter(CONFIG)(amplitude, valid)
            p = SpeckleFilter(dict(CONFIG, input_domain='intensity'))(amplitude**2, valid)
            np.testing.assert_allclose(a*a, p, rtol=2e-6, atol=1e-5)
            np.testing.assert_array_equal(a[:, ~valid], 0)

    def test_gpt_masks_band_order_cache_and_window_parity(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(snap_preprocess, 'CACHE_DIR', root/'cache'):
                path = root/'source.tif'
                x = np.full((3, 64, 64), 3., dtype='float32')
                x[2] *= 2
                x[:, 20:27, 20:27] = 0  # 合法0，不能当作NoData。
                x[:, 40, 40] = 30
                x[0, :4, :4] = -9999  # 只在一个所选通道缺测，公共有效域应排除。
                with rasterio.open(path, 'w', driver='GTiff', width=64, height=64, count=3,
                                   dtype='float32', nodata=-9999, crs='EPSG:32650',
                                   transform=rasterio.transform.from_origin(500000, 3500000, 3, 3)) as dst:
                    dst.write(x)
                f = SpeckleFilter(CONFIG)
                cached = f.snap.prepare_file(path, (3, 1))
                with rasterio.open(cached) as src:
                    full = src.read(masked=True)
                    self.assertEqual(src.units, ('amplitude', 'amplitude'))
                    self.assertTrue(np.ma.getmaskarray(full)[:, :4, :4].all())
                    self.assertFalse(np.ma.getmaskarray(full)[:, 23, 23].any())
                    np.testing.assert_array_equal(full[:, 23, 23], [0, 0])
                    np.testing.assert_allclose(full[:, 10, 10], [6, 3])
                    self.assertLess(float(full[0, 40, 40]), 30)
                with patch.object(snap_preprocess.subprocess, 'run', side_effect=AssertionError('缓存不应启动GPT')):
                    reloaded = SpeckleFilter(f.config)
                    self.assertEqual(reloaded.snap.prepare_file(path, (3, 1)), cached)
                    with rasterio.open(path) as src:
                        for window in [Window(0, 0, 16, 16), Window(18, 19, 32, 32)]:
                            data, valid = read_filtered_window(src, (3, 1), window, reloaded)
                            rows, cols = window.toslices()
                            expected = full[:, rows, cols]
                            np.testing.assert_array_equal(valid, ~np.ma.getmaskarray(expected).any(axis=0))
                            np.testing.assert_array_equal(data, expected.filled(0))
                    # 源文件时间戳改变后必须重新计算；失败不能提交完成标记。
                    st = path.stat(); os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns+1000000000))
                    with self.assertRaisesRegex(AssertionError, '缓存不应启动GPT'):
                        SpeckleFilter(f.config).snap.prepare_file(path, (3, 1))
                self.assertEqual(len(list((root/'cache').glob('*/processing.json'))), 1)
                with self.assertRaisesRegex(ValueError, 'SNAP版本'):
                    SpeckleFilter(dict(f.config, snap_runtime={'different': 'version'}))

    def test_training_checkpoint_evaluation_and_scene_reader(self):
        torch.set_num_threads(2)
        with TemporaryDirectory() as temporary, patch.object(torch.cuda, 'is_available', return_value=False):
            root = Path(temporary)
            fixtures.SceneDatasetTests().prepare(root)
            with patch.object(snap_preprocess, 'CACHE_DIR', root/'snap_cache'):
                # 预处理必须显式执行，训练与复评不能再触发GPT。
                filtering = SpeckleFilter(CONFIG)
                for split in ('train', 'val'):
                    filtering.snap.prepare_file(root/'images'/split/(split+'.tif'), (1,))
                trainer = WaterSegmentationTrainer(TinySnapNet, dict(num_classes=1), dict(model_name='tiny'),
                    root/'images', root/'labels', root/'runs', bands=(1,), batch_size=2,
                    epochs=1, loss_name='bce', speckle=CONFIG,
                    scene_options=dict(cache_dir=root/'cache', window_size=32, halo=8,
                                       normalization_windows=2, num_workers=0, windows_per_epoch=2))
                self.assertLess(trainer.dataset.normalization(trainer.dataset.train_ids)[0][0], 3)
                trainer.train(); trainer.dataset.close()
                path = trainer.run_dir/'last.pth'
                checkpoint = torch.load(path, map_location='cpu', weights_only=True)
                self.assertEqual(checkpoint['speckle']['method'], 'snap_lee')
                self.assertEqual(checkpoint['speckle']['calibration'], 'not_applied')
                with patch.object(snap_preprocess.subprocess, 'run', side_effect=AssertionError('复评应复用缓存')), \
                     patch.object(model_eval, 'load_model', return_value=trainer.model), \
                     patch.object(model_eval, 'SARSceneDataset', side_effect=lambda folder, cache, **opts:
                                  SARSceneDataset(folder, root/'cache', **opts)):
                    evaluated = model_eval.evaluate(path, use_amp=False)
                self.assertAlmostEqual(evaluated['loss'], trainer.phase_metrics['val']['loss'], places=6)
                inference = SceneWindows(root/'images/val/val.tif', window_size=32, halo=8,
                                         speckle=checkpoint['speckle'])
                prediction_input = inference[0]
                training_input = trainer.dataset[trainer.dataset.val_ids[0]]
                torch.testing.assert_close(prediction_input['image'], training_input['image'], rtol=0, atol=0)
                torch.testing.assert_close(prediction_input['valid_mask'], training_input['valid_mask'])
                inference.close(); trainer.dataset.close()


if __name__ == '__main__':
    unittest.main()
