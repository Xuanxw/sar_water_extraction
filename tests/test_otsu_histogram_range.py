"""稳健直方图保持样本数/原值、极亮长尾回退、块间一致性及审查图输出。"""
import importlib.util
import json
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import rasterio
import yaml
from shapely.geometry import box

from thres_segmentation.threshold import select_otsu_or_percentile, ThresholdSegmenter
from thres_segmentation.org_pipeline import statistics, run_org_scenes
from thres_segmentation.scene_pipeline import scene_threshold
from utils.files import sha256
import test_org_pipeline as org_fixture

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('otsu_audit', ROOT / 'scripts/inspect_otsu_outliers.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class OtsuHistogramRangeTests(unittest.TestCase):
    def test_bright_tail_clamped_histogram_keeps_samples_and_input(self):
        values = np.r_[np.linspace(1, 2, 5000), np.linspace(6, 7, 5000), 1e8]
        original = values.copy()
        _, full = select_otsu_or_percentile(values, 3., method='otsu')
        self.assertEqual(full['fallback_reason'], 'class_fraction_below_minimum')
        bounds = np.percentile(values, [0.1, 99.9])
        threshold, robust = select_otsu_or_percentile(values, 3., method='otsu', otsu_histogram_range=bounds)
        self.assertEqual(robust['method_used'], 'otsu')
        self.assertTrue(2 < threshold < 6)
        self.assertAlmostEqual(robust['otsu_lower_fraction'], 5000 / len(values))
        self.assertEqual(robust['histogram_clipped_lower_pixels'], 10)
        self.assertEqual(robust['histogram_clipped_upper_pixels'], 10)
        np.testing.assert_array_equal(values, original)
        _, equal = select_otsu_or_percentile(values, 3., method='otsu', otsu_histogram_range=[1, 1])
        self.assertEqual(equal['fallback_reason'], 'collapsed_histogram_range')

    def test_org_top_mode_range_percentiles_and_offline_match_array(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            fixture = org_fixture.OrgProductionTests('test_morphology_matches_single_block')
            config = fixture.make_config(root)
            source = next(config['paths']['ORG_ROOT'].glob('*.tif'))
            with rasterio.open(source) as src:
                data = src.read(1, masked=True)
            valid = ~np.ma.getmaskarray(data)
            bounds = np.percentile(data.data[valid], [0, 99])
            config['threshold'].update(method='otsu', mode='top', otsu_histogram_percentiles=[0, 99])
            for size in (9, 128):
                folder = root / f'size_{size}'
                folder.mkdir()
                config['processing']['block_size'] = size
                stats = statistics(source, box(500000, 4000000, 500064, 4000064), config, folder)
                self.assertEqual(stats['threshold_selection']['histogram_min'], bounds[0])
                self.assertEqual(stats['threshold_selection']['histogram_max'], bounds[1])
                segmenter = ThresholdSegmenter(preprocessing='offline_normalized', method='otsu',
                                              mode='top', otsu_histogram_percentiles=[0, 99])
                _, _, expected = segmenter.predict(data.data, valid)
                with rasterio.open(source) as src:
                    actual = scene_threshold(src, 1, size, segmenter, folder)[-1]
                self.assertEqual(actual, expected)
                self.assertEqual(actual, stats['thresholds_amplitude'][0])

    def test_audit_figures_exact_counts_and_source_unchanged(self):
        with TemporaryDirectory() as temp, redirect_stdout(StringIO()):
            root = Path(temp)
            fixture = org_fixture.OrgProductionTests('test_production_resume_and_optional_merge')
            config = fixture.make_config(root)
            config['threshold'].update(method='otsu', otsu_bins=1024, otsu_min_separability=0.5,
                                       otsu_min_class_fraction=0.001, water_percent=30)
            baseline = run_org_scenes(config)
            record = json.loads((baseline / 'results.json').read_text(encoding='utf-8'))[0]
            before = {p: sha256(p) for p in baseline.rglob('*') if p.is_file()}
            options = dict(paths=dict(BASELINE_OUTPUT_DIR=str(baseline), OUTPUT_DIR=str(root / 'audit')),
                           audit=dict(scene_ids=[record['scene_id']], block_size=13, patch_size=16,
                                      max_patches=2, preview_size=64, histogram_ranges=[None, [0.1, 99.9]]))
            path = root / 'audit.yaml'
            path.write_text(yaml.safe_dump(options), encoding='utf-8')
            output = audit.inspect_otsu_outliers(path)
            folder = output / record['scene_id']
            report = json.loads((folder / 'audit.json').read_text(encoding='utf-8'))
            self.assertEqual(report['valid_pixels'], 2499)
            self.assertEqual(len(report['trials']), 2)
            self.assertTrue((folder / 'patch_01.png').is_file())
            self.assertTrue((folder / 'patch_01.csv').is_file())
            self.assertTrue((folder / 'histogram_range_comparison.png').is_file())
            self.assertTrue((folder / 'outlier_locations.png').is_file())
            self.assertFalse(list(folder.glob('values_*')))
            self.assertEqual(before, {p: sha256(p) for p in before})


if __name__ == '__main__':
    unittest.main()
