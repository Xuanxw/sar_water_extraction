"""正式阈值入口的自动包络接入；不调用真实几何计算或网络。"""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import rasterio
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from thres_segmentation.profile_terrain import prepare_profile_height_envelope, generate_profile_terrain
from configuration import load_yaml


class ProfileEnvelopeIntegrationTests(unittest.TestCase):
    def make_inputs(self, root):
        dem = root/'dem.tif'
        with rasterio.open(dem, 'w', driver='GTiff', width=2, height=2, count=1,
                dtype='float32', crs=4326, transform=from_origin(1, 2, .5, .5)) as dst:
            dst.write(np.full((2, 2), 100., 'float32'), 1)
        automatic = dict(mode='regional_dem', global_height_bounds_m=[-12000, 10000],
                         margin_m=250, rounding_m=500, coverage_survey=dict(time_samples=3,
                         theta_samples=32, theta_max_rad=.12, coverage_guard_m=1000))
        cfg = dict(paths=dict(AUXILIARY_CACHE_DIR=root/'shared'), org=dict(
            profile_geometry=dict(height_envelope=automatic),
            auxiliary=dict(max_tiles=8, download_missing=False, timeout=10, retries=1)))
        geometry = dict(paths=dict(CORE_PATH=root/'core', DEM_TILES=[dem], GEOID_PATH='geoid', OUTPUT_DIR=root/'output'),
                        model=dict(height_datum='EGM2008'), scene=dict(output_crs='EPSG:32631',support_buffer_m=20000))
        return cfg, geometry

    def test_fixed_mode_keeps_existing_baseline(self):
        cfg = dict(org=dict(profile_geometry={}))
        geometry = dict(scene=dict(height_bounds_m=[-12000, 10000]))
        before = deepcopy(geometry)
        self.assertIsNone(prepare_profile_height_envelope(geometry, {}, cfg, lambda s: None))
        self.assertEqual(geometry, before)

    def test_computed_values_are_passed_to_geometry_without_mutating_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, geometry = self.make_inputs(Path(tmp))
            before = deepcopy(cfg)
            domain = dict(geographic_bounds_with_guard=[1.1, 1.1, 1.9, 1.9])
            # 特意采用与密云不同的值，验证适配器传递每景派生界限。
            derived = dict(height_bounds_m=[-1000., 4500.])
            with patch('thres_segmentation.profile_terrain.bound_same_range_terrain', return_value=domain), \
                 patch('thres_segmentation.profile_terrain.scan_regional_dem_envelope', return_value=derived), \
                 patch('thres_segmentation.profile_terrain.download_tile', side_effect=AssertionError('local DEM covers region')):
                result = prepare_profile_height_envelope(geometry, {}, cfg, lambda s: None)
            self.assertEqual(geometry['scene']['height_bounds_m'], [-1000., 4500.])
            self.assertEqual(cfg, before)
            stored = json.loads((geometry['paths']['OUTPUT_DIR']/'regional_height_envelope.json').read_text(encoding='utf-8'))
            self.assertEqual(stored['height_bounds_m'], result['height_bounds_m'])

    def test_missing_tiles_do_not_fall_back_to_miyun_envelope(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, geometry = self.make_inputs(Path(tmp))
            geometry['paths']['DEM_TILES'] = []
            with patch('thres_segmentation.profile_terrain.bound_same_range_terrain', return_value=dict(
                    geographic_bounds_with_guard=[1.1, 1.1, 1.9, 1.9])), \
                 patch('thres_segmentation.profile_terrain.download_tile', side_effect=AssertionError('disabled')):
                with self.assertRaisesRegex(FileNotFoundError, '缺少DEM'):
                    prepare_profile_height_envelope(geometry, {}, cfg, lambda s: None)
            self.assertNotIn('height_bounds_m', geometry['scene'])

    def test_download_uses_shared_cache_and_derived_scene_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, geometry = self.make_inputs(Path(tmp))
            local = geometry['paths']['DEM_TILES'][0]
            geometry['paths']['DEM_TILES'] = []
            cfg['org']['auxiliary']['download_missing'] = True
            with patch('thres_segmentation.profile_terrain.bound_same_range_terrain', return_value=dict(
                    geographic_bounds_with_guard=[1.1, 1.1, 1.9, 1.9])), \
                 patch('thres_segmentation.profile_terrain.download_tile', return_value=local) as download, \
                 patch('thres_segmentation.profile_terrain.scan_regional_dem_envelope', return_value=dict(height_bounds_m=[0., 1500.])):
                prepare_profile_height_envelope(geometry, {}, cfg, lambda s: None)
            self.assertEqual(download.call_count, 1)
            self.assertEqual(download.call_args.args[1].parent, cfg['paths']['AUXILIARY_CACHE_DIR']/'tiles/copernicus')
            self.assertEqual(geometry['scene']['height_bounds_m'], [0., 1500.])

    def test_formal_geometry_call_receives_automatic_envelope(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, local = self.make_inputs(root)
            cfg = load_yaml(ROOT/'configs/北京密云区/corrected_amplitude_otsu_autoheight.yaml')
            cfg['paths'].update(AUXILIARY_CACHE_DIR=root/'aux', PROFILE_GEOMETRY_CACHE_DIR=root/'geometry',
                                PROFILE_DEM_TILES=local['paths']['DEM_TILES'])
            folder = root/'scene'
            folder.mkdir()
            (folder/'metadata.json').write_text(json.dumps(dict(source_image='BC2-SM-ORG-VV-test.tif')), encoding='utf-8')
            with patch('thres_segmentation.profile_terrain.read_bc_metadata', return_value={}), \
                 patch('thres_segmentation.profile_terrain.bound_same_range_terrain', return_value=dict(
                    geographic_bounds_with_guard=[1.1, 1.1, 1.9, 1.9])), \
                 patch('thres_segmentation.profile_terrain.scan_regional_dem_envelope', return_value=dict(height_bounds_m=[-1000., 4500.])), \
                 patch('thres_segmentation.profile_terrain.run_zero_doppler_terrain_visibility', return_value=dict(complete=True)) as run, \
                 patch('thres_segmentation.profile_terrain.align_profile_masks', return_value={}) as align:
                generate_profile_terrain(root/'reference.tif', cfg, folder)
            self.assertEqual(run.call_args.args[0]['scene']['height_bounds_m'], [-1000., 4500.])
            self.assertNotIn('height_bounds_m', cfg['org']['profile_geometry']['scene'])
            self.assertTrue(align.called)
            stored = json.loads((folder/'terrain_geometry.json').read_text(encoding='utf-8'))
            self.assertEqual(stored['geometry_quality']['height_envelope']['height_bounds_m'], [-1000., 4500.])


if __name__ == '__main__':
    unittest.main()
