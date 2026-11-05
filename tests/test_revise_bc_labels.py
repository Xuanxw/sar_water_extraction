"""验证面内外语义、无效像素、VRT复用及固定地理划分。"""
import csv
import hashlib
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import fiona
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import Polygon, box, mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from datasets import revise_bc_labels as revision
from datasets.prepare_bc_data import MANIFEST_FIELDS
from datasets.sar_scene_dataset import SARSceneDataset


class RevisionTests(unittest.TestCase):
    def prepare(self, folder):
        source = folder / 'old'
        transform = from_origin(500000, 3400000, 3, 3)
        def rect(c0, r0, c1, r1):
            x0, y0 = transform * (c0, r0)
            x1, y1 = transform * (c1, r1)
            return box(x0, y1, x1, y0)
        outer, hole = rect(2, 2, 18, 18), rect(6, 6, 10, 10)
        shapes = [Polygon(outer.exterior.coords, [hole.exterior.coords]),
                  rect(14, 14, 28, 28), rect(40, 40, 45, 45)]
        expected = rasterize([(g, 1) for g in shapes], out_shape=(64, 64), transform=transform, dtype='uint8')
        expected[3, 3] = 255
        records = []
        for split, value in [('train', 2), ('val', 1000)]:
            image_dir = source / 'images' / split
            image_dir.mkdir(parents=True)
            data = np.full((64, 64), value, dtype='float32')
            data[3, 3] = np.nan
            data[4, 4] = 0
            image = image_dir / (split + '.tif')
            with rasterio.open(image, 'w', driver='GTiff', count=1, width=64, height=64,
                               crs='EPSG:32650', transform=transform, dtype='float32', nodata=np.nan) as dst:
                dst.write(data, 1)
            label = folder / (split + '.shp')
            with fiona.open(label, 'w', driver='ESRI Shapefile', crs='EPSG:32650',
                            schema={'geometry': 'Polygon', 'properties': {'class': 'int'}}) as dst:
                for g, c in zip(shapes, (0, 1, 37)):
                    dst.write(dict(geometry=mapping(g), properties={'class': c}))
            records.append(dict(name=image.name, label_name=split+'.tif', split=split, active=1,
                                group_id='G_'+split, scene_id=split, scene='合成验证', source_image=str(image),
                                source_label=str(label), row_off=0, col_off=0, source_height=64, source_width=64,
                                valid_pixels=4095, water_pixels=1, background_pixels=1, ignored_pixels=4094,
                                pixel_x=3, pixel_y=3))
        with (source / 'split_manifest.csv').open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(records)
        return source, expected

    def test_union_outside_holes_invalid_values_and_reader(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source, expected = self.prepare(root)
            originals = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
            target = root / 'v2'
            revision.prepare_labels(source, target, read_window_size=16, workers=2)
            with (target / 'split_manifest.csv').open(encoding='utf-8-sig') as f:
                rows = list(csv.DictReader(f))
            self.assertEqual([(r['split'], r['group_id']) for r in rows], [('train','G_train'),('val','G_val')])
            for r in rows:
                with rasterio.open(target / 'labels' / r['split'] / r['label_name']) as src:
                    np.testing.assert_array_equal(src.read(1), expected)
                    self.assertEqual(src.colormap(1)[0][:3], (255,255,255))
                    self.assertEqual(src.colormap(1)[1][:3], (0,0,0))
                self.assertEqual(int(r['water_pixels']), int((expected == 1).sum()))
                self.assertEqual(int(r['background_pixels']), int((expected == 0).sum()))
                with rasterio.open(target / 'images' / r['split'] / r['name']) as src:
                    self.assertEqual(src.read(1)[4,4], 0)
            self.assertEqual(expected[7,7], 0)  # 水体面内部孔洞不是水体。
            self.assertEqual(expected[16,16], 1)  # 原class0与1的交叠仍为水体。
            self.assertEqual(expected[42,42], 1)  # class数值不决定水陆。
            ds = SARSceneDataset(target, root/'cache', window_size=32, halo=8, normalization_windows=100)
            try:
                self.assertEqual(int(ds.windows[:,5:7].sum()), 2*4095)
                cache_key = ds.cache_key
                mean, _, _ = ds.normalization(ds.train_ids)
                self.assertLess(float(mean[0]), 3)  # val=1000不进入归一化。
            finally:
                ds.close()
            self.assertTrue(all(hashlib.sha256(p.read_bytes()).hexdigest()==h for p,h in originals.items()))
            image = source/'images/train/train.tif'
            stat = image.stat()
            os.utime(image, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
            changed = SARSceneDataset(target, root/'cache', window_size=32, halo=8)
            self.assertNotEqual(changed.cache_key, cache_key)
            changed.close()

    def test_all_valid_fast_path_keeps_zero_and_outside_background(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source, expected = self.prepare(root)
            manifest = source/'split_manifest.csv'
            with manifest.open(encoding='utf-8-sig') as f:
                rows = list(csv.DictReader(f))
            for row in rows:
                row['valid_pixels'] = 4096
                with rasterio.open(source/'images'/row['split']/row['name'], 'r+') as f:
                    data = f.read(1)
                    data[3,3] = 0
                    f.write(data,1)
            with manifest.open('w', encoding='utf-8-sig', newline='') as f:
                writer=csv.DictWriter(f,fieldnames=MANIFEST_FIELDS)
                writer.writeheader();writer.writerows(rows)
            revision.prepare_labels(source, root/'v2', workers=2)
            expected[3,3] = 1
            with rasterio.open(root/'v2/labels/train/train.tif') as f:
                np.testing.assert_array_equal(f.read(1),expected)

    def test_interrupted_first_scene_resumes_without_duplicate_header(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source, _ = self.prepare(root)
            target = root/'v2'
            with patch.object(revision, 'rebuild_scene', side_effect=RuntimeError('中断')):
                with self.assertRaisesRegex(RuntimeError, '中断'):
                    revision.prepare_labels(source, target)
            self.assertFalse((target/'split_manifest.csv').exists())
            revision.prepare_labels(source, target)
            with (target/'split_manifest.csv').open(encoding='utf-8-sig') as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 2)


if __name__ == '__main__':
    unittest.main()
