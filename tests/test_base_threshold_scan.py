"""阈值扫描复用固定掩膜、保持Q30基线、单调放宽及阶段续跑的合成验证。"""

import importlib.util
import json
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import rasterio
import yaml

import test_org_pipeline as org_fixture
from thres_segmentation import org_pipeline
from thres_segmentation.product_layout import scene_product
from utils.files import sha256

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('base_scan', ROOT / 'scripts/threshold/base_threshold_scan.py')
scan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scan)


class BaseThresholdScanTests(unittest.TestCase):
    def test_frozen_terrain_q30_equivalence_source_preservation_and_resume(self):
        with TemporaryDirectory() as temp, redirect_stdout(StringIO()):
            root = Path(temp)
            fixture = org_fixture.OrgProductionTests('test_production_resume_and_optional_merge')
            config = fixture.make_config(root)
            config['threshold']['water_percent'] = 30
            config['org']['postprocess'].update(history_enabled=False, history_relax_percent=0.)
            baseline = org_pipeline.run_org_scenes(config)
            record = json.loads((baseline / 'results.json').read_text(encoding='utf-8'))[0]
            original = Path(record['water_shp']).parent
            before = {p: sha256(p) for p in baseline.rglob('*') if p.is_file()}
            options = dict(paths=dict(BASELINE_OUTPUT_DIR=str(baseline), OUTPUT_DIR=str(root / 'scan'),
                                      CACHE_DIR=str(root / 'scan_cache')),
                           experiment=dict(dates=[record['date']], scene_ids=[record['scene_id']],
                                           percentiles=[30, 60], resume=True, console_color=False))
            path = root / 'scan.yaml'
            path.write_text(yaml.safe_dump(options), encoding='utf-8')
            with patch('thres_segmentation.terrain_geometry.generate_terrain_mask',
                       side_effect=AssertionError('terrain recomputed')):
                output = scan.run_base_threshold_scan(path)
            masks = []
            for q in (30, 60):
                folder = output / f'base_q{q}' / record['date'] / record['scene_id']
                with rasterio.open(folder / 'water.tif') as src:
                    masks.append(src.read(1))
                quality = json.loads((folder / 'quality.json').read_text(encoding='utf-8'))
                self.assertEqual(quality['extraction']['statistics']['water_percent'], q)
                self.assertEqual(quality['extraction']['statistics']['threshold_selection']['method_used'], 'percentile')
                self.assertTrue((folder / 'water_polygon.shp').is_file())
                self.assertIsNone(quality['accuracy']['recall'])
            with rasterio.open(scene_product(original, 'water.tif')) as src:
                np.testing.assert_array_equal(masks[0], src.read(1))
            self.assertTrue(np.any((masks[1] == 1) & (masks[0] == 0)))
            self.assertFalse(np.any((masks[1] == 0) & (masks[0] == 1)))
            self.assertEqual(before, {p: sha256(p) for p in before})
            with patch.object(scan, 'statistics', side_effect=AssertionError('statistics repeated')), \
                 patch.object(scan, 'segment', side_effect=AssertionError('segmentation repeated')), \
                 patch.object(scan, 'export_water_polygon', side_effect=AssertionError('polygon repeated')):
                scan.run_base_threshold_scan(path)


if __name__ == '__main__':
    unittest.main()
