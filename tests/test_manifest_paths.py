"""路径迁移只使用临时小文件，检查跨目录复制、搬迁、缓存及旧配置。"""
import csv
import hashlib
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from configuration import load_yaml
from datasets.manifest_paths import source_path, write_relative_manifest, manifest_fingerprint


class ManifestPathTests(unittest.TestCase):
    def write(self, path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)

    def read(self, path):
        with path.open(encoding='utf-8-sig', newline='') as stream:
            return list(csv.DictReader(stream))

    def test_convert_copy_move_and_invalidate(self):
        with TemporaryDirectory() as temp:
            root = Path(temp) / 'original'
            raw = root / 'source'; raw.mkdir(parents=True)
            image, label = raw/'原始.zip', raw/'水体.shp'
            image.write_bytes(b'SAR'); label.write_bytes(b'LABEL')
            manifest = root/'prepared/split_manifest.csv'
            rows = [dict(name='A.tif', split='train', source_image=str(image), source_label=str(label), group_id='G_a')]
            self.write(manifest, rows)
            original = hashlib.sha256(manifest.read_bytes()).hexdigest()
            self.assertEqual(write_relative_manifest(manifest), 2)
            converted = self.read(manifest)[0]
            self.assertEqual(converted['source_image'], '../source/原始.zip')
            self.assertEqual(source_path(converted['source_label'], manifest), label)
            self.assertEqual(manifest_fingerprint(manifest), original)
            saved = manifest.read_bytes()
            self.assertEqual(write_relative_manifest(manifest), 0)
            self.assertEqual(manifest.read_bytes(), saved)
            target = root/'processed/normalized/split_manifest.csv'
            target.parent.mkdir(parents=True)
            write_relative_manifest(manifest, target)
            self.assertEqual(self.read(target)[0]['source_image'], '../../source/原始.zip')
            self.assertEqual(manifest_fingerprint(target), original)
            moved = Path(temp)/'relocated'
            shutil.copytree(root, moved)
            copied = moved/'processed/normalized/split_manifest.csv'
            self.assertEqual(source_path(self.read(copied)[0]['source_image'], copied), moved/'source/原始.zip')
            self.assertEqual(source_path(self.read(copied)[0]['source_image'], copied).read_bytes(), b'SAR')
            changed = self.read(target); changed[0]['split'] = 'val'
            self.write(target, changed)
            self.assertNotEqual(manifest_fingerprint(target), original)

    def test_yaml_root_and_legacy_config(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            config = root/'settings.yaml'
            config.write_text("paths:\n  DATA_ROOT: data\n  IMAGE_DIR: images/train\n  MANIFEST_PATHS: [prepared/split_manifest.csv]\n  SOURCE_OVERRIDES: {A: {image: raw/A.zip}}\n  OPTIONAL: null\n", encoding='utf-8')
            values = load_yaml(config)['paths']
            self.assertEqual(values['DATA_ROOT'], root/'data')
            self.assertEqual(values['IMAGE_DIR'], root/'data/images/train')
            self.assertEqual(values['MANIFEST_PATHS'], [root/'data/prepared/split_manifest.csv'])
            self.assertEqual(values['SOURCE_OVERRIDES']['A']['image'], root/'data/raw/A.zip')
            self.assertIsNone(values['OPTIONAL'])
            config.write_text('paths:\n  IMAGE_DIR: images/train\n', encoding='utf-8')
            self.assertEqual(load_yaml(config)['paths']['IMAGE_DIR'], Path('images/train'))

    def test_existing_window_cache_still_reused(self):
        from test_sar_scene_dataset import SceneDatasetTests
        from supervised.load.sar_scene_dataset import SARSceneDataset
        with TemporaryDirectory() as temp:
            root = Path(temp)
            SceneDatasetTests().prepare(root)
            manifest = root/'split_manifest.csv'
            rows = self.read(manifest)
            for row in rows:
                row['source_image'] = str(root/'images'/row['split']/row['name'])
                row['source_label'] = str(root/'labels'/row['split']/row['label_name'])
            self.write(manifest, rows)
            before = SARSceneDataset(root, root/'cache', window_size=32, halo=8)
            key, windows = before.cache_key, before.windows.copy()
            before.close()
            write_relative_manifest(manifest)
            with patch('supervised.load.sar_scene_dataset.rasterio.open', side_effect=AssertionError('must reuse index')):
                after = SARSceneDataset(root, root/'cache', window_size=32, halo=8, require_prepared=True)
            self.assertEqual(after.cache_key, key)
            np.testing.assert_array_equal(after.windows, windows)
            after.close()

    def test_prepared_relative_label_can_be_rebuilt_after_move(self):
        from test_prepare_water_data import PrepareWaterDataTests
        from datasets import image_process as process
        with TemporaryDirectory() as temp:
            root = Path(temp)/'original'; root.mkdir()
            helper = PrepareWaterDataTests()
            image, label, *_ = helper.sources(root)
            output = helper.output(root)
            row = process.prepare_scene(helper.row(), image, label, output, root/'cache',
                                        bands=(1,), read_window_size=4, field=None, class_map={None: 1}, outside_label=0)[0]
            self.assertFalse(Path(row['source_label']).is_absolute())
            moved = Path(temp)/'moved'
            shutil.copytree(root, moved)
            result = process.rebuild_scene(row, moved/'prepared', read_window_size=4)
            self.assertEqual(result['water_pixels'], row['water_pixels'])
            self.assertEqual(result['background_pixels'], row['background_pixels'])


if __name__ == '__main__':
    unittest.main()
