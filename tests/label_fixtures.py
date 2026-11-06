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
from datasets.image_process import MANIFEST_FIELDS
from supervised.load.sar_scene_dataset import SARSceneDataset


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
