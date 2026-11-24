"""对数幅度Otsu：阈值单位、合法0、回退、幅度尺度及分块一致性。"""
import json
import importlib.util
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import rasterio
import yaml
from unittest.mock import patch
from shapely.geometry import box

from thres_segmentation.threshold import select_otsu_or_percentile
from thres_segmentation.org_pipeline import statistics, segment
from thres_segmentation.org_pipeline import run_org_scenes
import test_org_pipeline as org_fixture


class OtsuLogAmplitudeTests(unittest.TestCase):
    def test_log_threshold_restored_to_amplitude_and_scale_equivariance(self):
        values = np.r_[np.geomspace(10, 20, 10000), np.geomspace(500, 1000, 10000), 1e8]
        source = values.copy()
        _, linear = select_otsu_or_percentile(values, 30, method='otsu')
        self.assertEqual(linear['method_used'], 'percentile')
        threshold, report = select_otsu_or_percentile(values, 30, method='otsu', otsu_domain='log_amplitude')
        self.assertEqual(report['method_used'], 'otsu')
        self.assertTrue(20 < threshold < 500)
        self.assertAlmostEqual(threshold, 10 ** (report['otsu_threshold_axis'] / 20))
        self.assertTrue(report['otsu_lower_neighbor'] < threshold < report['otsu_upper_neighbor'])
        scaled, _ = select_otsu_or_percentile(values * 17, 30 * 17, method='otsu', otsu_domain='log_amplitude')
        self.assertAlmostEqual(scaled, threshold * 17)
        np.testing.assert_array_equal(values, source)

    def test_zero_retained_negative_rejected_and_fallback_in_original_units(self):
        values = np.r_[np.zeros(100), np.ones(100)]
        threshold, report = select_otsu_or_percentile(values, 0., method='otsu', otsu_domain='log_amplitude')
        self.assertEqual(report['log_floor_pixels'], 100)
        self.assertEqual(report['method_used'], 'otsu')
        self.assertGreater(threshold, 0)
        np.testing.assert_array_equal(values < threshold, np.r_[np.ones(100, bool), np.zeros(100, bool)])
        with self.assertRaises(ValueError):
            select_otsu_or_percentile(np.array([-1., 2.]), 1., method='otsu', otsu_domain='log_amplitude')
        with self.assertRaises(ValueError):
            select_otsu_or_percentile(values, 0., method='otsu', otsu_domain='log_amplitude', otsu_log_floor=0.)
        threshold, report = select_otsu_or_percentile(np.full(10, 1234.), 1234., method='otsu', otsu_domain='log_amplitude')
        self.assertEqual(threshold, 1234.)
        self.assertEqual(report['method_used'], 'percentile')
        self.assertEqual(report['fallback_reason'], 'constant_image')

    def test_histogram_range_stays_in_amplitude_units(self):
        values = np.r_[np.geomspace(10, 20, 5000), np.geomspace(500, 1000, 5000), 1e8]
        bounds = np.percentile(values, [0.1, 99.9])
        threshold, report = select_otsu_or_percentile(values, 30., method='otsu',
            otsu_domain='log_amplitude', otsu_histogram_range=bounds)
        self.assertEqual(report['histogram_min'], bounds[0])
        self.assertEqual(report['histogram_max'], bounds[1])
        self.assertAlmostEqual(report['histogram_axis_min'], 20 * np.log10(bounds[0]))
        self.assertAlmostEqual(report['histogram_axis_max'], 20 * np.log10(bounds[1]))
        self.assertTrue(20 < threshold < 500)

    def test_org_production_classifies_original_amplitude_and_blocks_match(self):
        with TemporaryDirectory() as temp, redirect_stdout(StringIO()):
            root = Path(temp)
            fixture = org_fixture.OrgProductionTests('test_morphology_matches_single_block')
            config = fixture.make_config(root)
            config['threshold'].update(method='otsu', otsu_domain='log_amplitude', water_percent=30)
            config['org']['postprocess'].update(terrain_enabled=False, landcover_enabled=False, history_enabled=False)
            source = next(config['paths']['ORG_ROOT'].glob('*.tif'))
            aoi = box(500000, 4000000, 500064, 4000064)
            masks = []
            for size in (9, 128):
                folder = root / f'blocks_{size}'
                folder.mkdir()
                config['processing']['block_size'] = size
                stats = statistics(source, aoi, config, folder)
                report = stats['threshold_selection']
                self.assertEqual(report['otsu_domain'], 'log_amplitude')
                self.assertEqual(stats['thresholds_amplitude'][0], report['otsu_threshold'])
                self.assertEqual(stats['thresholds_amplitude'][1:3],
                                 [report['otsu_lower_neighbor'], report['otsu_upper_neighbor']])
                segment(source, aoi, {}, config, folder, stats)
                with rasterio.open(source) as src, rasterio.open(folder / 'water.tif') as mask:
                    data = src.read(1, masked=True)
                    expected = np.where(~np.ma.getmaskarray(data), data.data < report['otsu_threshold'], 255).astype('uint8')
                    masks.append(mask.read(1))
                    np.testing.assert_array_equal(masks[-1], expected)
                json.dumps(stats, allow_nan=False)
            np.testing.assert_array_equal(masks[0], masks[1])

    def test_audit_explicit_single_scene_cache_preparation_and_resume(self):
        script = Path(__file__).resolve().parents[1] / 'scripts/threshold/inspect_otsu_outliers.py'
        spec = importlib.util.spec_from_file_location('log_audit', script)
        audit = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(audit)
        with TemporaryDirectory() as temp, redirect_stdout(StringIO()):
            root = Path(temp)
            fixture = org_fixture.OrgProductionTests('test_production_resume_and_optional_merge')
            config = fixture.make_config(root)
            config['threshold']['water_percent'] = 30
            baseline = run_org_scenes(config)
            record = json.loads((baseline / 'results.json').read_text(encoding='utf-8'))[0]
            options = dict(paths=dict(BASELINE_OUTPUT_DIR=str(baseline), SOURCE_CACHE_DIR=str(root / 'source_cache'),
                                      OUTPUT_DIR=str(root / 'audit')),
                           audit=dict(prepare_source=True, scene_ids=[record['scene_id']], block_size=13,
                                      patch_size=16, max_patches=1, preview_size=64,
                                      histogram_ranges=[None], otsu_domains=['linear','log_amplitude']))
            path = root / 'audit.yaml'
            path.write_text(yaml.safe_dump(options), encoding='utf-8')
            output = audit.inspect_otsu_outliers(path)
            report = json.loads((output / record['scene_id'] / 'audit.json').read_text(encoding='utf-8'))
            self.assertEqual(report['valid_pixels'], 2499)
            self.assertEqual(len(report['trials']), 2)
            self.assertEqual(report['outlier_reference_source'], 'current_linear_full_histogram')
            self.assertTrue((output / record['scene_id'] / 'log_amplitude_comparison.png').is_file())
            aoi, crs = audit.read_aoi(config['paths']['AOI_PATH'], config['org']['output_crs'])
            with patch.object(audit, 'crop_org', side_effect=AssertionError('crop repeated')):
                audit.prepare_otsu_scene_cache(baseline, root / 'source_cache', record['scene_id'], config, aoi, crs)


if __name__ == '__main__':
    unittest.main()
