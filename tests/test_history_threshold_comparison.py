"""历史分位对照：小型真实栅格流程验证，禁止下载、不启动SNAP。"""
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
from history_threshold_comparison import main
import test_org_pipeline as fixtures
from thres_segmentation.org_pipeline import run_org_scenes


class HistoryComparisonTests(unittest.TestCase):
    def test_three_levels_parity_monotonicity_resume_and_source_preserved(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            helper = fixtures.OrgProductionTests()
            config = helper.make_config(root)
            config['logging'] = {'console_color': False}
            # 让较亮历史水体跨越15/20/25分位，确保测试能检出真实差异。
            with rasterio.open(config['paths']['LANDCOVER_PATH'], 'r+') as dst:
                cover = np.full((64,64),80,np.float32);cover[:15,:15]=50
                dst.write(cover,1)
            with patch('thres_segmentation.org_io.download_tile', side_effect=AssertionError('no network')):
                baseline = run_org_scenes(config)
                originals = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in baseline.rglob('*') if p.is_file()}
                options = {'paths': {'BASELINE_OUTPUT_DIR': str(baseline),
                            'BASELINE_CACHE_DIR': str(config['paths']['CACHE_DIR']),
                            'SNAP_CACHE_DIR': str(root/'snap'),
                            'AUXILIARY_CACHE_DIR': str(config['paths']['AUXILIARY_CACHE_DIR']),
                            'OUTPUT_DIR': str(root/'comparison'), 'CACHE_DIR': str(root/'comparison_cache')},
                           'experiment': {'history_percentiles':[15,20,25], 'resume':True,'console_color':False}}
                path = root/'experiment.yaml';path.write_text(yaml.safe_dump(options),encoding='utf8')
                result = main(path)
                for record in json.loads((baseline/'results.json').read_text()):
                    masks=[]; ids=[]
                    for level in (15,20,25):
                        folder=result/f'history_{level}'/record['date']/record['scene_id']
                        with rasterio.open(folder/'water.tif') as src:masks.append(src.read(1))
                        ids.append((folder/'fixed_points_predictions.csv').read_text(encoding='utf-8-sig').splitlines())
                        self.assertTrue((folder/'water.shp').exists())
                        self.assertFalse((folder/'slope.tif').exists())
                        self.assertFalse((folder/'history.tif').exists())
                    self.assertTrue(np.all((masks[0]!=1)|(masks[1]==1)))
                    self.assertTrue(np.all((masks[1]!=1)|(masks[2]==1)))
                    self.assertGreater(int((masks[2]==1).sum()),int((masks[0]==1).sum()))
                    self.assertEqual([r.split(',')[0] for r in ids[0][1:]], [r.split(',')[0] for r in ids[2][1:]])
                with patch('history_threshold_comparison.segment',side_effect=AssertionError('should resume')):
                    main(path)
                self.assertTrue(all(hashlib.sha256(p.read_bytes()).hexdigest()==value for p,value in originals.items()))

    def test_auxiliary_cached_tiles_without_download(self):
        import shutil
        from thres_segmentation.org_io import prepare_auxiliary, read_aoi
        with TemporaryDirectory() as temp:
            config = fixtures.OrgProductionTests().make_config(Path(temp))
            cached = config['paths']['AUXILIARY_CACHE_DIR']/'tiles/worldcover/cached.tif'
            cached.parent.mkdir(parents=True)
            shutil.copy2(config['paths']['LANDCOVER_PATH'], cached)
            config['paths']['LANDCOVER_PATH'] = None
            aoi, crs = read_aoi(config['paths']['AOI_PATH'], config['org']['output_crs'])
            with patch('thres_segmentation.org_io.tile_specs',return_value=[{'filename':'cached.tif','url':'not_called'}]), patch('thres_segmentation.org_io.download_tile',side_effect=AssertionError('no download')):
                result = prepare_auxiliary(config,aoi,crs)
                self.assertTrue(result['worldcover'].exists())


if __name__=='__main__':unittest.main()
