"""坡度阈值约束的边界、分块、成面保护和只读参考复用；不启动真实处理。"""
import copy
import hashlib
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from thres_segmentation.threshold import slope_otsu_rejection
from thres_segmentation import org_pipeline as pipeline
from thres_segmentation.org_io import export_water_polygon
from thres_segmentation.profile_terrain import PROFILE_METHOD_VERSION
import test_org_pipeline as org_fixture


class OtsuSlopeGuardTests(unittest.TestCase):
    def test_strict_threshold_boundary_unknown_and_fallback(self):
        values=np.array([[2.,3.,4.,4.,4.,4.,4.]])
        slope=np.array([[20.,20.,15.,15.01,np.nan,np.inf,20.]])
        valid=np.array([[True,True,True,True,True,True,False]])
        reject=slope_otsu_rejection(values,valid,slope,6.,3.,15.)
        np.testing.assert_array_equal(reject,[[False,True,False,True,False,False,False]])
        for otsu in (2.,3.):
            self.assertFalse(slope_otsu_rejection(values,valid,slope,otsu,3.,15.).any())

    def test_block_invariance_and_cleanup_cannot_refill_rejection(self):
        with TemporaryDirectory() as temp:
            root=Path(temp)
            fixture=org_fixture.OrgProductionTests()
            config=fixture.make_config(root)
            config['threshold'].update(method='otsu',water_percent=30,
                                       slope_guard=dict(enabled=True,slope_deg=15.))
            config['org']['postprocess'].update(terrain_method='radar_profiles',history_enabled=False,
                                               landcover_enabled=False,terrain_enabled=True)
            values=np.full((32,32),2.,np.float32)
            slope=np.full(values.shape,2.,np.float32)
            values[10:13,10:13]=5.
            slope[10:13,10:13]=20.
            values[5,5]=5.
            slope[5,5]=15.
            values[6,6]=5.
            slope[6,6]=np.nan
            values[0,0]=np.nan
            terrain=np.zeros(values.shape,np.uint8)
            terrain[20,20]=1
            profile=dict(fixture.profile,width=32,height=32,transform=from_origin(500000,4000032,1,1))
            def write(name,array):
                path=root/name
                with rasterio.open(path,'w',**profile) as dst:
                    dst.write(array.astype('float32'),1)
                return path
            source=write('input.tif',values)
            slope_path=write('slope.tif',slope)
            terrain_path=write('terrain.tif',terrain)
            stats=dict(thresholds_amplitude=[7.,6.5,7.5,7.],
                       threshold_selection=dict(percentile_threshold=3.,method_used='otsu'))
            masks=[]
            for size in (7,128):
                config['processing']['block_size']=size
                folder=root/f'block_{size}'
                folder.mkdir()
                pipeline.segment(source,box(500000,4000000,500032,4000032),{'slope':slope_path},
                    config,folder,stats,terrain_products={'terrain_exclusion':terrain_path})
                with rasterio.open(folder/'water.tif') as src:
                    masks.append(src.read(1))
                self.assertEqual(masks[-1][5,5],1)
                self.assertEqual(masks[-1][6,6],1)
                self.assertEqual(masks[-1][20,20],0)
                self.assertEqual(masks[-1][0,0],255)
                self.assertTrue(np.all(masks[-1][10:13,10:13]==0))
                # 此孔洞小于200像素；没有风险保护时会被填回。
                export_water_polygon(folder/'water.tif',options=dict(hole_max_pixels=200,
                    morph_kernel_size=3,min_water_pixels=1,smooth_iterations=0),
                    postprocess=dict(terrain_enabled=False,landcover_enabled=False))
                with rasterio.open(folder/'water_polygon.tif') as src:
                    final=src.read(1)
                self.assertTrue(np.all(final[10:13,10:13]==0))
                self.assertEqual(final[0,0],255)
            np.testing.assert_array_equal(*masks)

    def test_reference_reuse_selection_resume_and_source_preserved(self):
        with TemporaryDirectory() as temp:
            root=Path(temp)
            config=org_fixture.OrgProductionTests().make_config(root)
            config['threshold'].update(method='otsu',water_percent=30)
            config['org']['postprocess'].update(terrain_method='radar_profiles',history_enabled=False)
            config['org']['save_quicklook']=False
            config['org']['archive_intermediates']=True

            def flat_geometry(reference,settings,folder):
                paths={}
                with rasterio.open(reference) as src:
                    profile=dict(src.profile,driver='GTiff',dtype='uint8',nodata=255)
                    for name in ('terrain_exclusion','shadow_mask','layover_mask'):
                        path=folder/(name+'.tif')
                        with rasterio.open(path,'w',**profile) as dst:
                            dst.write(np.zeros(src.shape,np.uint8),1)
                        paths[name]=path
                (folder/'terrain_geometry.json').write_text(
                    json.dumps(dict(method=PROFILE_METHOD_VERSION)),encoding='utf-8')
                return paths

            with patch('thres_segmentation.profile_terrain.generate_profile_terrain',side_effect=flat_geometry):
                pipeline.run_org_scenes(config)
            baseline=config['paths']['OUTPUT_DIR']
            source_files=[p for directory in (baseline,config['paths']['CACHE_DIR']) for p in directory.rglob('*') if p.is_file()]
            before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
            trial=copy.deepcopy(config)
            trial['paths'].update(OUTPUT_DIR=root/'trial',OUTPUT_ROOT=root/'trial',CACHE_DIR=root/'trial_cache',
                                  REUSE_RESULTS_DIR=baseline)
            trial['threshold']['slope_guard']=dict(enabled=True,slope_deg=15.)
            selected='BC2-SM-ORG-VV-20260101T030000-000001'
            trial['org']['scene_ids']=[selected]
            with patch.object(pipeline,'crop_org',side_effect=AssertionError('must reuse crop')), \
                 patch.object(pipeline,'statistics',side_effect=AssertionError('must reuse statistics')), \
                 patch('thres_segmentation.profile_terrain.generate_profile_terrain',side_effect=AssertionError('must reuse geometry')):
                pipeline.run_org_scenes(trial)
            results=json.loads((trial['paths']['OUTPUT_DIR']/'results.json').read_text(encoding='utf-8'))
            self.assertEqual([r['scene_id'] for r in results],[selected])
            polygon_report=json.loads(Path(results[0]['water_polygon_shp']).with_suffix('.json').read_text(encoding='utf-8'))
            self.assertTrue(Path(polygon_report['slope_threshold_rejection']).is_file())
            self.assertTrue(Path(polygon_report['source']).is_file())
            with rasterio.open(trial['paths']['CACHE_DIR']/'scenes'/selected/'filter_reference.vrt') as src:
                self.assertEqual(src.driver,'VRT')
            # 续跑仅使用本轮阶段记录；源实验字节保持不变。
            with patch.object(pipeline,'segment',side_effect=AssertionError('mask must resume')):
                pipeline.run_org_scenes(trial)
            # 新几何只导入前置影像/统计，必须重算地形；不能悄悄沿用30米旧掩膜。
            fine=copy.deepcopy(trial)
            fine['paths'].update(OUTPUT_DIR=root/'fine',OUTPUT_ROOT=root/'fine',CACHE_DIR=root/'fine_cache')
            fine['org']['profile_geometry']=dict(scene=dict(resolution_m=5,dem_resolution_m=30))
            with patch.object(pipeline,'crop_org',side_effect=AssertionError('must reuse crop')), \
                 patch.object(pipeline,'statistics',side_effect=AssertionError('must reuse statistics')), \
                 patch('thres_segmentation.profile_terrain.generate_profile_terrain',side_effect=flat_geometry) as geometry_call:
                pipeline.run_org_scenes(fine)
                self.assertEqual(geometry_call.call_count,1)
            with patch.object(pipeline,'segment',side_effect=AssertionError('fine mask must resume')):
                pipeline.run_org_scenes(fine)
            self.assertEqual(before,{p:hashlib.sha256(p.read_bytes()).hexdigest() for p in before})


if __name__=='__main__':
    unittest.main()
