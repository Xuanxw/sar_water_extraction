"""验证原位标签替换、数值含义、源文件保护以及中断后的训练隔离。"""
import csv
import hashlib
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from datasets import image_process
from supervised.load.sar_scene_dataset import SARSceneDataset
import label_fixtures as fixtures

# 测试显式加载样例配置，正式脚本仅使用命令行提供的文件。
image_process.configure(Path(__file__).resolve().parents[1] / 'configs/image_process.yaml')


class InPlaceLabelTests(unittest.TestCase):
    def test_replace_reversed_labels_preserve_sources_and_refresh_cache(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp)
            root, expected = fixtures.RevisionTests().prepare(folder)
            with (root / 'split_manifest.csv').open(encoding='utf-8-sig') as stream:
                before = list(csv.DictReader(stream))
            protected = {path: hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in folder.rglob('*') if path.is_file()
                         and path.name != 'split_manifest.csv'}
            # 制造水陆数值相反的旧标签，重建必须从SHP纠正，而非沿用旧栅格。
            for row in before:
                target = root / 'labels' / row['split'] / row['label_name']
                target.parent.mkdir(parents=True)
                with rasterio.open(root / 'images' / row['split'] / row['name']) as src:
                    profile = src.profile.copy()
                profile.update(dtype='uint8', nodata=255)
                with rasterio.open(target, 'w', **profile) as dst:
                    dst.write(np.where(expected == 255, 255, 1 - expected).astype('uint8'), 1)

            image_process.prepare_labels(root, read_window_size=16)
            with (root / 'split_manifest.csv').open(encoding='utf-8-sig') as stream:
                after = list(csv.DictReader(stream))
            for old, new in zip(before, after):
                for field in ('name', 'label_name', 'split', 'group_id', 'source_image', 'source_label'):
                    self.assertEqual(old[field], new[field])
                with rasterio.open(root / 'labels' / new['split'] / new['label_name']) as src:
                    np.testing.assert_array_equal(src.read(1), expected)
                    self.assertEqual(src.colormap(1)[0][:3], (255, 255, 255))
                    self.assertEqual(src.colormap(1)[1][:3], (0, 0, 0))
                self.assertEqual(int(new['water_pixels']), int((expected == 1).sum()))
                self.assertEqual(int(new['background_pixels']), int((expected == 0).sum()))
            self.assertFalse(list(root.rglob('*.vrt')))
            self.assertFalse((root / '.labels_rebuilding').exists())
            for path, digest in protected.items():
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
            ds = SARSceneDataset(root, folder / 'cache', window_size=32, halo=8)
            previous_key = ds.cache_key
            ds.close()
            image_process.prepare_labels(root, read_window_size=32)
            ds = SARSceneDataset(root, folder / 'cache', window_size=32, halo=8)
            self.assertNotEqual(ds.cache_key, previous_key)
            ds.close()

    def test_interruption_blocks_training_until_rebuild_finishes(self):
        with TemporaryDirectory() as temp:
            root, _ = fixtures.RevisionTests().prepare(Path(temp))
            manifest_before = (root / 'split_manifest.csv').read_bytes()
            with patch.object(image_process, 'rebuild_scene', side_effect=RuntimeError('模拟中断')):
                with self.assertRaisesRegex(RuntimeError, '模拟中断'):
                    image_process.prepare_labels(root)
            self.assertEqual((root / 'split_manifest.csv').read_bytes(), manifest_before)
            with self.assertRaisesRegex(RuntimeError, '标签重建尚未完成'):
                SARSceneDataset(root, Path(temp) / 'cache')
            image_process.prepare_labels(root)
            self.assertFalse((root / '.labels_rebuilding').exists())


if __name__ == '__main__':
    unittest.main()
