"""v5.7扫描边界、编码、未知传播及同网格比较；不代替整链GAMMA验收。"""

import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import rasterio
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from sar_geometry.core import scan_gc_map2_v57_shadow_layover
from sar_geometry.gamma_reference import (
    GAMMA_VERSION, compare_gc_map2_masks, decode_gc_map2_ls_map, encode_gc_map2_ls_map,
)

CORE = Path(os.environ.get('SAR_GEOMETRY_CORE', str(ROOT / 'build/sar_geometry_gamma57/sar_geometry_core.exe')))


class Gamma57ScanTests(unittest.TestCase):
    def scan(self, angles, ranges, sign=1):
        return scan_gc_map2_v57_shadow_layover(CORE, [angles], [ranges], sign)[0]

    def test_flat_visible_first_point_is_untested(self):
        np.testing.assert_array_equal(self.scan([1, 2, 3, 4], [10, 11, 12, 13]), [0, 1, 1, 1])

    def test_shadow_horizon_equality_is_visible(self):
        result = self.scan([1, 3, 2, 2.5, 3, 4], [10, 11, 12, 13, 14, 15])
        np.testing.assert_array_equal(result, [0, 1, 16, 16, 1, 1])

    def test_zero_skips_pair_without_resetting_shadow(self):
        result = self.scan([4, 3, 0, 2, 3, 5], [10, 11, 12, 13, 14, 15])
        np.testing.assert_array_equal(result, [0, 16, 0, 0, 16, 1])

    def test_pure_shadow_inhibits_new_fold(self):
        np.testing.assert_array_equal(self.scan([1, 3, 2, 3], [10, 12, 11, 13]), [0, 1, 16, 1])

    def test_closed_fold_backfills_both_branches(self):
        result = self.scan([1, 2, 3, 4, 5, 6], [10, 12, 11, 10, 11, 13])
        np.testing.assert_array_equal(result, [4, 4, 4, 4, 4, 1])

    def test_equal_range_and_float32_quantization(self):
        result = self.scan([1, 2, 3, 4], [10, 10000000, 10000000.25, 10000002])
        np.testing.assert_array_equal(result, [0, 4, 4, 1])

    def test_unclosed_tail_does_not_backfill(self):
        np.testing.assert_array_equal(self.scan([1, 2, 3], [10, 12, 11]), [0, 1, 4])

    def test_reverse_angular_direction(self):
        np.testing.assert_array_equal(self.scan([-1, -3, -2, -3], [10, 11, 12, 13], -1), [0, 1, 16, 1])

    def test_nonfinite_input_is_outside_recovered_contract(self):
        with self.assertRaises(ValueError):
            self.scan([1, np.nan], [10, 11])


class GammaEncodingTests(unittest.TestCase):
    def test_standard_flags_with_or_without_tested_bit(self):
        shadow, layover = decode_gc_map2_ls_map([0, 1, 4, 5, 16, 17, 20, 21, 2, 8])
        np.testing.assert_array_equal(shadow, [255, 0, 0, 0, 1, 1, 1, 1, 255, 255])
        np.testing.assert_array_equal(layover, [255, 0, 1, 1, 0, 0, 1, 1, 255, 255])

    def test_display_255_is_layover(self):
        shadow, layover = decode_gc_map2_ls_map([0, 127, 255, 31, 63], 1)
        np.testing.assert_array_equal(shadow, [255, 0, 0, 1, 1])
        np.testing.assert_array_equal(layover, [255, 0, 1, 0, 1])

    def test_partial_unknown_cannot_be_encoded_as_fully_known(self):
        values, valid = encode_gc_map2_ls_map([0, 1, 0, 1, 255, 1], [0, 0, 1, 1, 0, 255])
        np.testing.assert_array_equal(values, [1, 16, 4, 20, 0, 0])
        np.testing.assert_array_equal(valid, [True, True, True, True, False, False])


class GammaRasterComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.profile = dict(driver='GTiff', width=8, height=1, count=1, dtype='uint8',
                            crs='EPSG:32650', transform=from_origin(400000, 4500000, 10, 10), nodata=255)
        for name, values in {
            'gamma': [127, 255, 31, 63, 0, 127, 255, 127],
            'shadow': [0, 1, 0, 1, 0, 255, 0, 0],
            'layover': [0, 1, 0, 1, 255, 0, 255, 0],
        }.items():
            with rasterio.open(self.folder / f'{name}.tif', 'w', **self.profile) as target:
                target.write(np.array([values], dtype='uint8'), 1)
        self.config = dict(
            paths=dict(SHADOW_PATH=str(self.folder / 'shadow.tif'), LAYOVER_PATH=str(self.folder / 'layover.tif'),
                       GAMMA_LS_MAP=str(self.folder / 'gamma.tif'), OUTPUT_DIR=str(self.folder / 'comparison')),
            reference=dict(version=GAMMA_VERSION, grid_domain='map', format='geotiff', ls_scaling=1),
            block_size=3,
        )

    def test_counts_cover_all_pixels_and_unknowns_separately(self):
        report = compare_gc_map2_masks(self.config, log=lambda _: None)
        shadow = report['metrics']['shadow']
        self.assertEqual(shadow['confusion_against_gamma'], dict(tn=3, fp=1, fn=1, tp=1))
        self.assertEqual(shadow['ours_unknown'], 1)
        self.assertEqual(shadow['gamma_unknown'], 1)
        self.assertEqual(report['metrics']['layover']['both_unknown'], 1)
        self.assertEqual(report['gamma_code_counts']['255'], 2)
        self.assertFalse(report['numerical_equivalence_verified'])
        with rasterio.open(self.folder / 'comparison/shadow_difference.tif') as source:
            np.testing.assert_array_equal(source.read(1), [[1, 2, 3, 1, 5, 4, 1, 1]])

    def test_raw_uchar_uses_explicit_exported_grid(self):
        raw = self.folder / 'ls_map'
        np.array([127, 255, 31, 63, 0, 127, 255, 127], dtype='uint8').tofile(raw)
        self.config['paths'].update(GAMMA_LS_MAP=str(raw), GAMMA_GRID_TIF=str(self.folder / 'gamma.tif'))
        self.config['reference']['format'] = 'uchar'
        report = compare_gc_map2_masks(self.config, log=lambda _: None)
        self.assertEqual(report['metrics']['shadow']['common_known'], 6)

    def test_shifted_grid_is_rejected(self):
        with rasterio.open(self.folder / 'layover.tif', 'r+') as source:
            source.transform = from_origin(400005, 4500000, 10, 10)
        with self.assertRaisesRegex(ValueError, '相同的地图网格'):
            compare_gc_map2_masks(self.config, log=lambda _: None)

    def test_radar_grid_is_not_mislabeled_as_org_grid(self):
        self.config['reference']['grid_domain'] = 'radar'
        with self.assertRaisesRegex(ValueError, 'ls_map_rdc'):
            compare_gc_map2_masks(self.config, log=lambda _: None)


if __name__ == '__main__':
    unittest.main()
