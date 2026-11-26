"""细网格/原生边界的有效性传播、分块及DEM解耦验证。"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.transform import from_origin

from test_scene_visibility import scene_config, FlatTerrain, CORE
from test_geometry_core import straight_model
from sar_geometry.scene import prepare_inputs
from sar_geometry.profiles import build_zero_doppler_terrain_profiles, sample_zero_doppler_visibility
from thres_segmentation.profile_terrain import (
    visibility_transition_band,
    refine_sar_visibility_boundaries,
)


class FineVisibilityTests(unittest.TestCase):
    def test_dem_kernel_invalidates_cache_and_preserves_nodata(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = scene_config(Path(tmp))
            source = config['paths']['DEM_TILES'][0]
            with rasterio.open(source, 'r+') as ds:
                rows, cols = np.indices(ds.shape)
                values = (600 + 90 * np.cos(cols / 2) * np.sin(rows / 4)).astype('f4')
                values[110:125, 120:135] = ds.nodata
                ds.write(values, 1)
            config['scene']['dem_resolution_m'] = 15
            path, _ = prepare_inputs(config, lambda _: None)
            with rasterio.open(path) as ds:
                original = ds.read(1)
            config['scene']['dem_resampling'] = 'cubic'
            path, _ = prepare_inputs(config, lambda _: None)
            with rasterio.open(path) as ds:
                cubic = ds.read(1)
            self.assertGreater(np.nanmax(np.abs(cubic - original)), .1)
            self.assertTrue(np.any(~np.isfinite(cubic)))

    def test_elevation_sampling_across_column_chunks(self):
        from sar_geometry.profiles import ElevationSampler
        with tempfile.TemporaryDirectory() as tmp:
            config = scene_config(Path(tmp))
            path = config['paths']['DEM_TILES'][0]
            with rasterio.open(path, 'r+') as ds:
                field = np.arange(ds.width * ds.height, dtype='f4').reshape(ds.shape)
                field[140:150, 140:150] = ds.nodata
                ds.write(field, 1)
            sampler = ElevationSampler(path)
            try:
                x = np.linspace(-.04, .06, 1050)
                y = np.linspace(5.06, 4.95, len(x))
                expected = sampler.sample(x, y)
                actual = sampler.sample(np.stack([x, x]), np.stack([y, y]))
                np.testing.assert_allclose(actual, np.stack([expected, expected]), equal_nan=True)
            finally:
                sampler.close()

    def test_transition_halo_matches_whole_and_keeps_unknown(self):
        shadow = np.zeros((20, 23), np.uint8)
        shadow[7:13, 8:15] = 1
        shadow[11:15, 3:6] = 255
        layover = np.zeros_like(shadow)
        whole = visibility_transition_band(shadow, layover)
        pieces = []
        for row in range(0, 20, 5):
            first, last = max(0, row - 1), min(20, row + 6)
            pieces.append(visibility_transition_band(shadow[first:last], layover[first:last])[
                row - first:min(row + 5, 20) - first
            ])
        np.testing.assert_array_equal(whole, np.concatenate(pieces))
        self.assertTrue(whole[11, 3])
        self.assertFalse(whole[9, 10])

    def test_dem_storage_resolution_is_independent(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = scene_config(Path(tmp))
            config['scene'].update(resolution_m=50, dem_resolution_m=200)
            dem, ref = prepare_inputs(config, lambda text: None)
            with rasterio.open(dem) as d, rasterio.open(ref) as r:
                self.assertEqual(d.res, (200, 200))
                self.assertEqual(r.res, (50, 50))
            # 参数相同可续跑，不重写派生网格。
            before = dem.stat().st_mtime_ns
            prepare_inputs(config, lambda text: None)
            self.assertEqual(dem.stat().st_mtime_ns, before)

    def test_ecef_sampling_and_unknown(self):
        model = straight_model()
        times = np.linspace(0, 4, 9)
        theta = np.linspace(.05, .08, 200)
        masks, points, _, _ = build_zero_doppler_terrain_profiles(
            CORE, model, FlatTerrain(), times, theta, [-100, 1000]
        )
        query = np.concatenate([points.reshape(-1, 3), [[np.nan, np.nan, np.nan]]])
        values, good, covered = sample_zero_doppler_visibility(CORE, model, query, times, theta, masks)
        np.testing.assert_array_equal(values[:-1], masks.reshape(-1, 2))
        self.assertTrue(good[:-1].all())
        self.assertTrue(covered[:-1].all())
        np.testing.assert_array_equal(values[-1], [255, 255])
        self.assertFalse(good[-1])

    def test_native_boundary_recalculation_and_block_equivalence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            geometry = root / 'geometry'
            cache = geometry / 'data_cache'
            cache.mkdir(parents=True)
            profile = dict(driver='GTiff', count=1, width=12, height=12, dtype='uint8',
                           crs=32650, transform=from_origin(480000, 4497000, 5, 5), nodata=255)
            shadow = np.zeros((12, 12), np.uint8)
            shadow[:, 6:] = 1
            shadow[0, 0] = 255
            for name, value in (('shadow_mask', shadow), ('layover_mask', np.zeros_like(shadow))):
                with rasterio.open(geometry / (name + '.tif'), 'w', **profile) as dst:
                    dst.write(value, 1)
            with rasterio.open(cache / 'dem_ellipsoid.tif', 'w', **dict(profile, dtype='float32')) as dst:
                dst.write(np.full(shadow.shape, 200, np.float32), 1)
            np.save(cache / 'profiles.npy', np.zeros((2, 2, 2), np.uint8))
            np.savez(cache / 'profile_axes.npz', times=[0., 1.], theta=[0., 1.])
            source = root / 'sar.tif'
            native = dict(profile, width=30, height=30, dtype='float32',
                          transform=from_origin(480000, 4497000, 2, 2), nodata=-9999)
            with rasterio.open(source, 'w', **native) as dst:
                image = np.full((30, 30), 10, np.float32)
                image[0, 0] = -9999
                dst.write(image, 1)

            def visible(exe, model, points, times, theta, masks):
                n = len(points)
                return np.zeros((n, 2), np.uint8), np.ones(n, bool), np.ones(n, bool)

            outputs = []
            for size in (7, 32):
                folder = root / str(size)
                folder.mkdir()
                with patch('thres_segmentation.profile_terrain.build_bc_range_doppler_model', return_value={}), patch(
                    'thres_segmentation.profile_terrain.sample_zero_doppler_visibility', side_effect=visible
                ):
                    products, report = refine_sar_visibility_boundaries(
                        source, geometry, folder, dict(paths=dict(CORE_PATH=CORE), model={}), {}, size, lambda text: None
                    )
                with rasterio.open(products['shadow_mask']) as dst:
                    self.assertEqual(dst.transform, native['transform'])
                    result = dst.read(1)
                self.assertEqual(result[0, 0], 255)
                self.assertEqual(result[15, 15], 0)  # 原分类边界经过重算。
                self.assertEqual(result[15, 25], 1)  # 稳定遮挡核心不被清空。
                self.assertGreater(report['refined_pixels'], 0)
                outputs.append(result)
            np.testing.assert_array_equal(*outputs)


if __name__ == '__main__':
    unittest.main()
