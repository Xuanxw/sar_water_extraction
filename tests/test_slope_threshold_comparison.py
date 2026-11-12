"""真实小栅格验证坡度单因素对照，不下载、不调用SNAP、不处理真实密云数据。"""
import csv
import hashlib
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import rasterio
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import slope_threshold_comparison as experiment
import test_org_pipeline as fixtures
from thres_segmentation.org_pipeline import run_org_scenes, statistics, segment
from thres_segmentation.org_io import prepare_auxiliary, read_aoi


class SlopeComparisonTests(unittest.TestCase):
    def test_control_masks_fixed_points_resume_and_source_unchanged(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            helper = fixtures.OrgProductionTests()
            config = helper.make_config(root)
            config['logging'] = {'console_color': False}
            with rasterio.open(config['paths']['LANDCOVER_PATH'], 'r+') as dst:
                cover = np.full((64, 64), 80, np.float32)
                cover[12:16, 10:14] = 50
                dst.write(cover, 1)
            with patch('thres_segmentation.org_io.download_tile', side_effect=AssertionError('no network')):
                baseline = run_org_scenes(config)
                originals = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in baseline.rglob('*') if p.is_file()}
                aoi, crs = read_aoi(config['paths']['AOI_PATH'], config['org']['output_crs'])
                auxiliary = prepare_auxiliary(config, aoi, crs)
                # 横向设置5/10/15/20/25度，覆盖等于门限、被剔除以及坡度缺测。
                slope_file = root / 'test_slope.tif'
                slope = np.tile(np.resize(np.array([5,10,15,20,25], np.float32), 64), (64,1))
                slope[18:20, 20:25] = np.nan
                with rasterio.open(slope_file, 'w', **helper.profile) as dst:
                    dst.write(slope, 1)
                auxiliary['slope'] = slope_file
                options = {'paths': {'BASELINE_OUTPUT_DIR': str(baseline),
                            'BASELINE_CACHE_DIR': str(config['paths']['CACHE_DIR']),
                            'SNAP_CACHE_DIR': str(root/'snap'),
                            'AUXILIARY_CACHE_DIR': str(config['paths']['AUXILIARY_CACHE_DIR']),
                            'OUTPUT_DIR': str(root/'slope_results'), 'CACHE_DIR': str(root/'slope_cache')},
                           'experiment': {'base_percentile':10, 'history_percentile':20,
                                          'max_slope_degrees':[20,15,10], 'resume':True, 'console_color':False}}
                path = root/'experiment.yaml'
                path.write_text(yaml.safe_dump(options), encoding='utf-8')
                # 矢量阶段中断后，不重复计算该景已完成的20度分割。
                with patch.object(experiment, 'prepare_auxiliary', return_value=auxiliary), \
                     patch.object(experiment, 'vectorize', side_effect=RuntimeError('interrupted')):
                    with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                        experiment.main(path)
                first_mask = next((root/'slope_results').glob('slope_20/*/*/water.tif'))
                stamp = first_mask.stat().st_mtime_ns
                with patch.object(experiment, 'prepare_auxiliary', return_value=auxiliary):
                    result = experiment.main(path)
                self.assertEqual(stamp, first_mask.stat().st_mtime_ns)
                for record in json.loads((baseline/'results.json').read_text()):
                    sid, date = record['scene_id'], record['date']
                    masks, point_tables, thresholds = [], [], []
                    for level in (20, 15, 10):
                        folder = result/f'slope_{level}'/date/sid
                        with rasterio.open(folder/'water.tif') as src:
                            masks.append(src.read(1))
                        with (folder/'fixed_points_predictions.csv').open(encoding='utf-8-sig', newline='') as stream:
                            point_tables.append(list(csv.DictReader(stream)))
                        quality = json.loads((folder/'quality.json').read_text(encoding='utf-8'))
                        thresholds.append(quality['extraction']['statistics']['thresholds_amplitude'])
                        self.assertEqual(quality['comparison_to_slope20']['added_water_pixels'], 0)
                        self.assertIsNone(quality['accuracy']['f1'])
                        self.assertTrue((folder/'water.shp').exists())
                        self.assertFalse((folder/'slope.tif').exists())
                    self.assertEqual(thresholds[0], thresholds[1])
                    self.assertEqual(thresholds[1], thresholds[2])
                    self.assertTrue(np.all((masks[1]!=1)|(masks[0]==1)))
                    self.assertTrue(np.all((masks[2]!=1)|(masks[1]==1)))
                    self.assertGreater(int((masks[0]==1).sum()), int((masks[1]==1).sum()))
                    self.assertGreater(int((masks[1]==1).sum()), int((masks[2]==1).sum()))
                    self.assertTrue(np.array_equal(masks[0]==255, masks[2]==255))
                    for rows in point_tables[1:]:
                        self.assertEqual([(r['point_id'],r['weight']) for r in point_tables[0]],
                                         [(r['point_id'],r['weight']) for r in rows])
                    # 独立按Q10/Q20/20度直接运行生产分割，确认控制组不是原Q15结果。
                    parameters = json.loads((result/'slope_20'/date/sid/'parameters.json').read_text(encoding='utf-8'))
                    config['org']['postprocess']['history_relax_percent'] = 10
                    config['org']['postprocess']['max_slope_deg'] = 20
                    expected = root/f'expected_{sid}'; expected.mkdir()
                    stats = statistics(Path(parameters['filtered_image']), aoi, config, expected)
                    segment(Path(parameters['filtered_image']), aoi, auxiliary, config, expected, stats, save_context=False)
                    with rasterio.open(expected/'water.tif') as src:
                        np.testing.assert_array_equal(masks[0], src.read(1))
                    self.assertTrue((result/'figures'/f'{sid}.png').exists())
                with patch.object(experiment, 'prepare_auxiliary', return_value=auxiliary), \
                     patch.object(experiment, 'segment', side_effect=AssertionError('should reuse')), \
                     patch.object(experiment, 'statistics', side_effect=AssertionError('should reuse')):
                    experiment.main(path)
                self.assertTrue(all(hashlib.sha256(p.read_bytes()).hexdigest()==digest for p,digest in originals.items()))


if __name__ == '__main__':
    unittest.main()
