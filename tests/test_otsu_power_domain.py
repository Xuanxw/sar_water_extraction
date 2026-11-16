"""功率口径：SNAP不平方/开方、对数还原、单位一致性及ORG生产入口。"""
import json
import unittest
import xml.etree.ElementTree as ET
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from datasets.snap_preprocess import SnapSpeckleFilter
from thres_segmentation.threshold import select_otsu_or_percentile
from thres_segmentation.org_pipeline import run_org_scenes, statistics
from thres_segmentation.org_io import read_aoi
import test_org_pipeline as org_fixture


class OtsuPowerDomainTests(unittest.TestCase):
    def test_power_graph_has_no_square_or_sqrt(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / 'graph.xml'
            filt = SnapSpeckleFilter(dict(input_domain='intensity', window_size=7, enl=1.))
            filt.write_graph(path, root / 'input.tif', root / 'output.tif', 1)
            graph = ET.parse(path)
            expressions = [e.text for e in graph.findall('.//expression')]
            self.assertTrue(any('band_1' in e for e in expressions))
            self.assertFalse(any('sqrt' in e or 'band_1 * band_1' in e for e in expressions))
            units = [e.text for e in graph.findall('.//unit')]
            self.assertTrue(units)
            self.assertTrue(all(unit == 'intensity' for unit in units))

    def test_log_power_matches_log_amplitude_of_square_root(self):
        power = np.r_[np.geomspace(100, 400, 5000), np.geomspace(1e5, 4e5, 5000), 1e12]
        p, rp = select_otsu_or_percentile(power, 200., method='otsu', otsu_domain='log_intensity')
        a, ra = select_otsu_or_percentile(np.sqrt(power), np.sqrt(200.), method='otsu', otsu_domain='log_amplitude')
        self.assertEqual(rp['method_used'], 'otsu')
        self.assertEqual(rp['log_coefficient'], 10.)
        self.assertAlmostEqual(p, 10 ** (rp['otsu_threshold_axis'] / 10))
        self.assertAlmostEqual(p, a * a)
        np.testing.assert_array_equal(power < p, np.sqrt(power) < a)
        # 对同一数组只改10/20系数不会改变划分；滤波输入域才是本次必须纠正的部分。
        same, _ = select_otsu_or_percentile(power, 200., method='otsu', otsu_domain='log_amplitude')
        self.assertAlmostEqual(p, same)

    def test_power_org_entry_and_mismatched_domain_rejected(self):
        with TemporaryDirectory() as temp, redirect_stdout(StringIO()):
            root = Path(temp)
            fixture = org_fixture.OrgProductionTests('test_production_resume_and_optional_merge')
            config = fixture.make_config(root)
            config['org']['filter']['input_domain'] = 'intensity'
            config['threshold'].update(method='otsu', otsu_domain='log_intensity', water_percent=30)
            output = run_org_scenes(config)
            quality = next(output.glob('*/*/quality.json'))
            stats = json.loads(quality.read_text(encoding='utf-8'))['extraction']['statistics']
            self.assertEqual(stats['input_domain'], 'intensity')
            self.assertEqual(stats['threshold_units'], 'linear_intensity')
            self.assertEqual(stats['threshold_selection']['otsu_domain'], 'log_intensity')
            self.assertTrue(list(config['paths']['CACHE_DIR'].glob('scenes/*/intensity.tif')))
            config['threshold']['otsu_domain'] = 'log_amplitude'
            source = next(config['paths']['ORG_ROOT'].glob('*.tif'))
            aoi, _ = read_aoi(config['paths']['AOI_PATH'], config['org']['output_crs'])
            with self.assertRaisesRegex(ValueError, '不一致'):
                statistics(source, aoi, config, root)


if __name__ == '__main__':
    unittest.main()
