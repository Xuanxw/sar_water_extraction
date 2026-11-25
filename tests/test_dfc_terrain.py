"""固定 DFC 地形对照的物理尺度、有效域和训练分位数。"""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from datasets.dfc_sample.s1_dataset import WaterDataset
from datasets.dfc_terrain import (
    DFC_TERRAIN_CHANNELS, DFC_TERRAIN_INPUTS, apply_dfc_terrain, dfc_terrain_from_checkpoint,
    dfc_terrain_protocol, dfc_terrain_valid_mask, prepare_dfc_terrain,
)
from datasets.relative_normalization import LOG_REPRESENTATION


class DfcTerrainTests(unittest.TestCase):
    def test_plane_angle_at_center_and_edges(self):
        y, x = np.indices((8, 12), dtype=np.float32)
        height = -20 + 4 * x - 3 * y
        data = np.stack((np.ones_like(height), height))
        image, valid = apply_dfc_terrain(data, np.ones_like(height, dtype=bool), 'elevation_slope')
        self.assertTrue(valid.all())
        np.testing.assert_array_equal(image[1], height)
        # 30 m 的单边差分和 60 m 的中心差分都应得到 sqrt(0.4^2+0.3^2)。
        np.testing.assert_allclose(image[2], np.degrees(np.arctan(0.5)), atol=2e-6)

    def test_common_mask_at_nodata_barrier_and_valid_sea_level(self):
        height = np.zeros((9, 10), dtype=np.float32)
        support = np.ones_like(height, dtype=bool)
        support[:, 1:3] = False
        height[:, 1:3] = np.nan
        height[:, 0] = 200
        vv = np.ones_like(height)
        expected = support.copy()
        expected[:, 0] = False  # 该列没有连续 30 m 的横向高程邻域。
        np.testing.assert_array_equal(dfc_terrain_valid_mask(support), expected)
        for mode in DFC_TERRAIN_INPUTS:
            with self.subTest(mode=mode):
                channels, common = apply_dfc_terrain(np.stack((vv, height)), support, mode)
                np.testing.assert_array_equal(common, expected)
                self.assertEqual(channels.shape, (3, 9, 10))
                self.assertTrue(np.all(channels[:, ~common] == 0))
                self.assertTrue(np.all(channels[1:, common] == 0))
                self.assertTrue(np.all(channels[0, common] == 1))
        heights = np.indices(height.shape, dtype=np.float32)[1]
        data = np.stack((vv, heights))
        for mode, uses_height, uses_slope in [
            ('vv_only', False, False), ('elevation', True, False),
            ('slope', False, True), ('elevation_slope', True, True),
        ]:
            channels, common = apply_dfc_terrain(data, support, mode)
            np.testing.assert_array_equal(channels[1, common], heights[common] if uses_height else 0)
            np.testing.assert_allclose(channels[2, common], np.degrees(np.arctan(.1)) if uses_slope else 0,
                                       atol=1e-6)

    def test_dataset_fit_and_inference_share_train_only_transform(self):
        size = (9, 10)
        yy, xx = np.indices(size, dtype=np.float32)
        raw_vv = 1 + 10 * yy + xx
        raw_vv[:, 0] = -10  # 文件掩膜有效，地形差分无支持，不得参与 VV 对数转换。
        raw_dem = 3 * xx - 2 * yy
        common = xx >= 3
        with TemporaryDirectory() as root:
            images, labels = Path(root) / 'images', Path(root) / 'labels'
            images.mkdir()
            labels.mkdir()
            profile = dict(driver='GTiff', height=size[0], width=size[1], crs='EPSG:32650',
                           transform=from_origin(500000, 4000000, 10, 10))
            for name, vv in [('a', raw_vv), ('v', raw_vv + 10000)]:
                data = np.stack((vv, xx, yy, raw_dem))
                data[3, :, 1:3] = -9999
                with rasterio.open(images / f'{name}.tif', 'w', count=4, dtype='float32',
                                   nodata=-9999, **profile) as writer:
                    writer.write(data)
                    writer.scales = (1, 1, 1, 2)
                    writer.offsets = (0, 0, 0, -10)
                with rasterio.open(labels / f'{name}.tif', 'w', count=1, dtype='uint8', **profile) as writer:
                    writer.write(np.ones(size, dtype=np.uint8), 1)
            for mode in DFC_TERRAIN_INPUTS:
                ds = WaterDataset(images, labels, bands=(1, 4), patch_representation=LOG_REPRESENTATION,
                                  dfc_terrain_input=mode)
                fitted = ds.fit_relative_normalization([0])
                self.assertEqual(fitted['fit_pixels'], common.sum())
                reference = np.percentile(10 * np.log10(raw_vv[common].astype(np.float64)), [2, 98])
                np.testing.assert_allclose([fitted['p02'], fitted['p98']], reference)
                train_sample = ds[0]
                scaled_dem = 2 * raw_dem - 10
                expected, valid = prepare_dfc_terrain(np.stack((raw_vv, scaled_dem)), ~np.isin(xx, [1, 2]),
                                                      mode, fitted)
                np.testing.assert_allclose(train_sample['image'].numpy(), expected)
                np.testing.assert_array_equal(train_sample['valid_mask'].numpy(), common)
                np.testing.assert_array_equal(valid, common)
                np.testing.assert_array_equal(train_sample['label'].numpy(), np.where(common, 1, -1))
                if mode == 'elevation_slope':
                    np.testing.assert_array_equal(expected[1, common], scaled_dem[common])
                    np.testing.assert_allclose(expected[2, common], np.degrees(np.arctan(np.hypot(.6, .4))),
                                               atol=1e-5)
                val_sample = ds[1]['image'].numpy()
                np.testing.assert_array_equal(val_sample[0, common], 1)

    def test_checkpoint_rejects_incompatible_grid_or_bands(self):
        model = dict(dfc_terrain=dfc_terrain_protocol('slope'), patch_representation=LOG_REPRESENTATION,
                     in_channels=3, input_channels=DFC_TERRAIN_CHANNELS, bands=(1, 4))
        self.assertEqual(dfc_terrain_from_checkpoint(model), 'slope')
        self.assertIsNone(dfc_terrain_from_checkpoint({'bands': (1, 4)}))
        for changes in [dict(bands=(1, 2)), dict(dfc_terrain={**model['dfc_terrain'], 'grid_spacing_m': 1})]:
            with self.assertRaises(ValueError):
                dfc_terrain_from_checkpoint({**model, **changes})
