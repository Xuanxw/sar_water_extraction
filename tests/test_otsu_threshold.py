"""Otsu的解析分群、退化回退、整景分块一致性和生产续跑验证。"""

import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import rasterio
from shapely.geometry import box

from thres_segmentation.threshold import ThresholdSegmenter, select_otsu_or_percentile
from thres_segmentation.scene_pipeline import scene_threshold
from thres_segmentation import org_pipeline
import test_org_pipeline as org_fixture


class OtsuThresholdTests(unittest.TestCase):
    def test_two_classes_and_polarity(self):
        image = np.array([[-3., -3., 0., 0.], [12., 12., 12., np.nan]])
        valid = np.isfinite(image)
        for mode in ('bottom', 'top'):
            segmenter = ThresholdSegmenter(method='otsu', preprocessing='offline_normalized', mode=mode)
            prediction, score, threshold = segmenter.predict(image, valid)
            expected = image < 5 if mode == 'bottom' else image >= 5
            np.testing.assert_array_equal(prediction, expected & valid)
            np.testing.assert_equal(score, image)
            self.assertTrue(0 < threshold < 12)
            self.assertEqual(segmenter.threshold_report['method_used'], 'otsu')

    def test_fallback_is_exact_percentile_and_invalid_input_is_not_hidden(self):
        constant = np.full(80, 5.)
        t, info = select_otsu_or_percentile(constant, 5., method='otsu')
        self.assertEqual(t, 5.)
        self.assertEqual(info['fallback_reason'], 'constant_image')
        # 极少异常亮值不能单独形成被接受的类别。
        values = np.r_[np.zeros(10000), 100.]
        t, info = select_otsu_or_percentile(values, 0., method='otsu')
        self.assertEqual(t, 0.)
        self.assertEqual(info['fallback_reason'], 'class_fraction_below_minimum')
        values = np.linspace(0, 100, 1001)
        t, info = select_otsu_or_percentile(values, 30., method='otsu', otsu_min_separability=0.99)
        self.assertEqual(t, 30.)
        self.assertEqual(info['fallback_reason'], 'separability_below_minimum')
        with self.assertRaises(ValueError):
            select_otsu_or_percentile(np.array([np.nan]), 0., method='otsu')
        with self.assertRaises(ValueError):
            select_otsu_or_percentile(np.array([]), 0., method='otsu')
        with self.assertRaises(ValueError):
            select_otsu_or_percentile(values, 30., method='unknown')

    def test_otsu_maximizes_histogram_between_class_variance(self):
        # 对每个分界枚举两类方差，与向量化Otsu结果独立对照。
        values = np.repeat(np.arange(6, dtype=float), [3, 6, 8, 2, 1, 5])
        bins = 6
        edges = np.linspace(0, 5, bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2
        counts = np.histogram(values, bins=edges)[0]
        scores = []
        for split in range(1, bins):
            left, right = counts[:split], counts[split:]
            a = np.dot(left, centers[:split]) / left.sum()
            b = np.dot(right, centers[split:]) / right.sum()
            scores.append(left.sum() * right.sum() / counts.sum() ** 2 * (a - b) ** 2)
        t, info = select_otsu_or_percentile(values, 1., method='otsu', otsu_bins=bins)
        self.assertAlmostEqual(t, edges[np.argmax(scores) + 1])
        self.assertEqual(info['method_used'], 'otsu')

    def test_org_block_size_and_offline_scene_use_one_global_threshold(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            fixture = org_fixture.OrgProductionTests('test_morphology_matches_single_block')
            config = fixture.make_config(root)
            config['threshold'].update(method='otsu', water_percent=30)
            config['org']['postprocess'].update(terrain_enabled=False, history_enabled=False, landcover_enabled=False)
            source = next(config['paths']['ORG_ROOT'].glob('*.tif'))
            aoi = box(500000, 4000000, 500064, 4000064)
            stats = []
            masks = []
            for size in (13, 128):
                config['processing']['block_size'] = size
                folder = root / f'blocks_{size}'
                folder.mkdir()
                stats.append(org_pipeline.statistics(source, aoi, config, folder))
                org_pipeline.segment(source, aoi, {}, config, folder, stats[-1])
                with rasterio.open(folder / 'water.tif') as mask:
                    masks.append(mask.read(1))
            self.assertEqual(stats[0]['thresholds_amplitude'], stats[1]['thresholds_amplitude'])
            np.testing.assert_array_equal(masks[0], masks[1])
            self.assertEqual(stats[0]['threshold_selection']['method_used'], 'otsu')
            with rasterio.open(source) as src:
                segmenter = ThresholdSegmenter(method='otsu', preprocessing='offline_normalized', water_percent=30)
                threshold = scene_threshold(src, 1, 9, segmenter, root)[-1]
            self.assertEqual(threshold, stats[0]['thresholds_amplitude'][0])

    def test_complete_production_records_method_fallback_and_resume(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            fixture = org_fixture.OrgProductionTests('test_morphology_matches_single_block')
            config = fixture.make_config(root)
            config['threshold'].update(method='otsu', water_percent=30)
            config['org']['postprocess']['terrain_method'] = 'slope_only'
            output = org_pipeline.run_org_scenes(config)
            records = json.loads((output / 'results.json').read_text(encoding='utf-8'))
            first = Path(records[0]['water_shp']).parent
            original = (first / 'water_polygon.shp').stat().st_mtime_ns
            quality = json.loads((first / 'quality.json').read_text(encoding='utf-8'))
            self.assertEqual(quality['extraction']['statistics']['threshold_selection']['method_used'], 'otsu')
            with patch.object(org_pipeline, 'statistics', side_effect=AssertionError('statistics repeated')), \
                 patch.object(org_pipeline, 'segment', side_effect=AssertionError('segmentation repeated')):
                org_pipeline.run_org_scenes(config)
            self.assertEqual(original, (first / 'water_polygon.shp').stat().st_mtime_ns)
            # 新目录强制测试统计判据失败，确认实际回到原Q30值，不吞掉其他阶段异常。
            fallback = copy.deepcopy(config)
            fallback['paths']['OUTPUT_DIR'] = root / 'fallback'
            fallback['paths']['CACHE_DIR'] = root / 'fallback_cache'
            fallback['threshold']['otsu_min_separability'] = 0.999999
            org_pipeline.run_org_scenes(fallback)
            files = list((root / 'fallback').glob('*/*/quality.json'))
            for path in files:
                stats = json.loads(path.read_text(encoding='utf-8'))['extraction']['statistics']
                report = stats['threshold_selection']
                self.assertEqual(report['method_used'], 'percentile')
                self.assertEqual(report['fallback_reason'], 'separability_below_minimum')
                self.assertEqual(stats['thresholds_amplitude'][0], report['percentile_threshold'])


if __name__ == '__main__':
    unittest.main()
