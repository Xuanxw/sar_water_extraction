"""整景接通测试：观测平面、外围未知域、线程/分批一致性与续跑。"""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import rasterio
from rasterio.transform import from_origin

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from sar_geometry.core import compute_radar_shadow_layover_batch, solve_range_doppler_points, interpolate_hermite_orbit_states
from sar_geometry.profiles import DemPostingGrid, build_zero_doppler_terrain_profiles, sample_zero_doppler_visibility, TO_GEODETIC
from sar_geometry.scene import run_zero_doppler_terrain_visibility, combine_shadow_layover_masks
from sar_geometry.metadata import read_bc_metadata
from test_geometry_core import straight_model, CORE
from test_metadata_and_nominal import xml_text


class FlatTerrain:
    def sample(self,lon,lat):return np.full_like(lon,100.)


class ProfileGridTests(unittest.TestCase):
    def test_dem_posting_uses_source_centers_and_terrain_height(self):
        from pyproj import Transformer
        with tempfile.TemporaryDirectory() as tmp:
            cfg = scene_config(Path(tmp))
            tiles = cfg['paths']['DEM_TILES']
            grid = DemPostingGrid(tiles, 6)
            x0, y0 = grid.center
            dx, dy = grid.step
            xyz = Transformer.from_crs(4979, 4978, always_xy=True)
            points = np.column_stack(xyz.transform([x0+.49*dx, x0+.51*dx],
                                                    [y0+.51*dy, y0+.49*dy], [250, 250]))
            result = grid.surface_points(points, FlatTerrain())
            lon, lat, h = TO_GEODETIC.transform(*result.T)
            np.testing.assert_allclose(lon, [x0, x0+dx], atol=1e-10, rtol=0)
            np.testing.assert_allclose(lat, [y0+dy, y0], atol=1e-10, rtol=0)
            np.testing.assert_allclose(h, 100, atol=1e-5)
            with self.assertRaises(ValueError):
                DemPostingGrid(tiles, .5)

    def test_posted_layover_preserves_unknowns_and_shadow(self):
        center = np.array([[1,0],[0,1],[0,255],[255,0]], dtype='uint8')
        good = np.ones(4, bool)
        posted = np.array([[0,1],[1,255]], dtype='uint8')
        points = np.ones((4,3))
        def snap(known):
            self.assertEqual(len(known), 3)
            result = known.copy()
            result[-1] = np.nan
            return result
        with patch('sar_geometry.profiles._query_profile_masks',
                   side_effect=[(center,good,good), (posted,good[:2],good[:2])]):
            result, _, _ = sample_zero_doppler_visibility(CORE, {}, points, None, None, None, post_points=snap)
        np.testing.assert_array_equal(result, [[1,1],[0,1],[0,255],[255,0]])

    def test_layover_weight_threshold_is_inclusive(self):
        masks = np.zeros((2, 2, 2), dtype='uint8')
        masks[:, 1, 1] = 1
        x = np.array([.44, .45, .46])
        with patch('sar_geometry.profiles.solve_range_doppler_points', return_value=np.zeros((3, 6))), \
                patch('sar_geometry.profiles.map_ecef_to_zero_doppler_profile',
                      return_value=(np.zeros(3), x, np.ones(3, bool))):
            args = CORE, {}, np.zeros((3, 3)), np.arange(2), np.arange(2), masks
            normal = sample_zero_doppler_visibility(*args, subpixel_layover=True)[0]
            relaxed = sample_zero_doppler_visibility(*args, subpixel_layover=True, layover_vote_threshold=.45)[0]
            np.testing.assert_array_equal(normal[:, 1], [0, 0, 0])
            np.testing.assert_array_equal(relaxed[:, 1], [0, 1, 1])
            with self.assertRaises(ValueError):
                sample_zero_doppler_visibility(*args, subpixel_layover=True, layover_vote_threshold=0)

    def test_subpixel_preserves_small_monotone_steps_and_bounds(self):
        points = np.column_stack((np.arange(5), [10, 11, 11.01, 12, 13],
                                  np.linspace(-.8, -.2, 5), np.ones(5)))[None]
        bounds = np.array([[-2., 10.5, 12.5]])
        near = compute_radar_shadow_layover_batch(CORE, points, bounds, layover_scan='ordered_subpixel')
        points[:, :, 1] += 650000
        bounds[:, 1:] += 650000
        far = compute_radar_shadow_layover_batch(CORE, points, bounds, layover_scan='ordered_subpixel')
        np.testing.assert_array_equal(near, far)
        self.assertEqual(far[0, :, 1].tolist(), [255, 0, 0, 0, 255])
        points[:, :, 3] = 0
        empty = compute_radar_shadow_layover_batch(CORE, points, bounds, layover_scan='ordered_subpixel')
        self.assertTrue(np.all(empty == 255))

    def test_subpixel_vote_preserves_unknown_shadow_and_coverage(self):
        masks = np.array([[[0, 0], [1, 1], [0, 255]],
                          [[1, 1], [1, 1], [0, 255]]], dtype='uint8')
        rows = np.array([.4, .4, .4, .4])
        cols = np.array([.4, 1.4, 1.6, 0.4])
        solved = np.zeros((4, 6))
        solved[-1, 5] = 6
        with patch('sar_geometry.profiles.solve_range_doppler_points', return_value=solved), \
                patch('sar_geometry.profiles.map_ecef_to_zero_doppler_profile',
                      return_value=(rows, cols, np.ones(4, bool))):
            value, _, _ = sample_zero_doppler_visibility(
                CORE, {}, np.zeros((4, 3)), np.arange(2), np.arange(3), masks, subpixel_layover=True)
        np.testing.assert_array_equal(value, [[0, 1], [1, 1], [0, 255], [255, 255]])

    def test_ordered_fold_endpoints_shadow_and_missing_support(self):
        # 110→130→120→125→140：地表次序中的折返，回填最远与最近两个端点。
        ranges = [100, 110, 130, 120, 125, 140, 150]
        row = np.column_stack([np.arange(7), ranges, np.linspace(-.8, -.2, 7), np.ones(7)])
        bounds = np.array([[-2, 101, 149]])
        def classify(points):
            return compute_radar_shadow_layover_batch(CORE, points[None], bounds, layover_scan='ordered')[0]
        self.assertEqual(classify(row)[:, 1].tolist(), [255, 0, 1, 1, 1, 0, 255])
        legacy = compute_radar_shadow_layover_batch(CORE, row[None], bounds)[0]
        self.assertEqual(legacy[2:4, 1].tolist(), [255, 255])
        shaded = row.copy()
        shaded[3, 2] = shaded[2, 2] - .1
        self.assertEqual(classify(shaded)[3].tolist(), [1, 0])
        hole = row.copy()
        hole[3, 3] = 0
        self.assertTrue(np.all(classify(hole)[:, 1] == 255))

    def test_surface_observation_plane_and_roundtrip(self):
        model=straight_model();theta=np.linspace(.05,.08,200);times=np.array([0.,2.,4.])
        masks,points,good,error=build_zero_doppler_terrain_profiles(CORE,model,FlatTerrain(),times,theta,[-100,1000])
        self.assertTrue(good.all());self.assertLess(error,.05)
        states=interpolate_hermite_orbit_states(CORE,model['orbit'],times)
        plane=np.sum((points-states[:,None,:3])*states[:,None,3:],axis=-1)
        np.testing.assert_allclose(plane,0,atol=1e-3)
        _,_,h=TO_GEODETIC.transform(points[...,0],points[...,1],points[...,2])
        np.testing.assert_allclose(h,100,atol=.05)
        result=solve_range_doppler_points(CORE,model,points.reshape(-1,3))
        np.testing.assert_allclose(result[:,2].reshape(3,200),np.broadcast_to(times[:,None],(3,200)),atol=1e-7)
        self.assertFalse(np.any(masks==1))
        other,_,_,_=build_zero_doppler_terrain_profiles(CORE,model,FlatTerrain(),times[1:],theta,[-100,1000])
        np.testing.assert_array_equal(other,masks[1:])

    def test_unknown_boundaries_threads_and_holes(self):
        p=np.array([[0,100,-.5,0],[1,110,-.4,1],[2,120,-.2,1],[3,130,-.3,1],[4,140,0,1],[5,150,.1,0]],float)
        batch=np.stack([p,p]);bounds=np.array([[-.25,115,145]]*2)
        with patch.dict(os.environ,{'OMP_NUM_THREADS':'1'}):one=compute_radar_shadow_layover_batch(CORE,batch,bounds)
        with patch.dict(os.environ,{'OMP_NUM_THREADS':'4'}):many=compute_radar_shadow_layover_batch(CORE,batch,bounds)
        np.testing.assert_array_equal(one,many)
        self.assertEqual(one[0,1,0],255);self.assertEqual(one[0,2,0],0)
        self.assertEqual(one[0,3,0],1);self.assertEqual(one[0,1,1],255)
        batch[:,3,3]=0;hole=compute_radar_shadow_layover_batch(CORE,batch,bounds)
        self.assertTrue(np.all(hole[:,:,1]==255));self.assertEqual(hole[0,4,0],255)

    def test_union_keeps_unknown(self):
        np.testing.assert_array_equal(combine_shadow_layover_masks(np.array([0,255,1,255]),np.array([0,0,255,1])),[0,255,1,1])


def scene_config(root):
    """构造米制平地测试景；不读取密云影像、不做网络请求。"""
    xml=root/'BC2-SM-SLC-test.xml';xml.write_text(xml_text(),encoding='utf-8')
    org=root/'org.tif';dem=root/'dem.tif'
    for path,size,transform,value in ((org,80,from_origin(0,5.02,.00025,.00025),10),
                                      (dem,300,from_origin(-.03,5.05,.0003,.0003),100)):
        with rasterio.open(path,'w',driver='GTiff',count=1,width=size,height=size,dtype='float32',
                           crs=4326,transform=transform,nodata=-9999) as dst:dst.write(np.full((size,size),value,np.float32),1)
    return dict(paths=dict(XML_PATH=xml,ORG_PATH=org,DEM_TILES=[dem],GEOID_PATH=None,OUTPUT_DIR=root/'out',CORE_PATH=CORE),
        model=dict(radar_grid_confirmed=True,doppler_mode='zero',doppler_sign=None,height_datum='WGS84_ellipsoid',
                   max_iterations=40,tolerance_hz=.01,tolerance_pixels=.0001,max_step_seconds=2.),
        scene=dict(output_crs='EPSG:32631',resolution_m=100,azimuth_step_m=100,range_step_m=100,support_buffer_m=1000,
                   height_bounds_m=[-100,1000],height_tolerance_m=.05,org_nodata=0,max_profile_points=1000000),
        execution=dict(resume=True,threads=2,profile_batch_rows=7,map_block_rows=8,time_budget_seconds=180))


class SceneTests(unittest.TestCase):
    def test_posting_chunk_and_cache_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = scene_config(root)
            cfg['scene'].update(layover_scan='ordered_subpixel', map_posting_factor=6, layover_vote_threshold=.45)
            meta = read_bc_metadata(cfg['paths']['XML_PATH'])
            first = run_zero_doppler_terrain_visibility(cfg, meta, lambda _: None)
            self.assertTrue(first['complete'])
            other = copy.deepcopy(cfg)
            other['paths']['OUTPUT_DIR'] = root/'chunk'
            other['execution'].update(profile_batch_rows=23, map_block_rows=15)
            run_zero_doppler_terrain_visibility(other, meta, lambda _: None)
            with rasterio.open(root/'out/layover_mask.tif') as a, rasterio.open(root/'chunk/layover_mask.tif') as b:
                np.testing.assert_array_equal(a.read(1), b.read(1))
            self.assertTrue(run_zero_doppler_terrain_visibility(cfg, meta, lambda _: None)['cache_reused_this_run'])
            cfg['scene']['layover_vote_threshold'] = .5
            self.assertFalse(run_zero_doppler_terrain_visibility(cfg, meta, lambda _: None)['cache_reused_this_run'])

    def test_resume_and_chunk_equivalence(self):
        from sar_geometry.profiles import build_zero_doppler_terrain_profiles as real
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config=scene_config(root);meta=read_bc_metadata(config['paths']['XML_PATH']);calls=0
            def interrupt(*args,**kw):
                nonlocal calls
                calls+=1
                if calls==2:raise RuntimeError('test interruption')
                return real(*args,**kw)
            with patch('sar_geometry.scene.build_zero_doppler_terrain_profiles',side_effect=interrupt),self.assertRaisesRegex(RuntimeError,'interruption'):
                run_zero_doppler_terrain_visibility(config,meta,lambda x:None)
            state=json.loads((root/'out/data_cache/scene_progress.json').read_text())
            self.assertEqual(state['next_profile'],7);self.assertFalse(state['complete'])
            run_zero_doppler_terrain_visibility(config,meta,lambda x:None)
            other=copy.deepcopy(config);other['paths']['OUTPUT_DIR']=root/'other'
            other['execution'].update(profile_batch_rows=40,map_block_rows=40,threads=1)
            run_zero_doppler_terrain_visibility(other,meta,lambda x:None)
            for path in (root/'out').glob('*.tif'):
                with rasterio.open(path) as a,rasterio.open(root/'other'/path.name) as b:
                    self.assertEqual((a.transform,a.crs,a.shape),(b.transform,b.crs,b.shape))
                    np.testing.assert_array_equal(a.read(),b.read())
            with patch('sar_geometry.scene.build_zero_doppler_terrain_profiles',side_effect=AssertionError('must reuse')):
                report=run_zero_doppler_terrain_visibility(config,meta,lambda x:None)
            self.assertTrue(report['complete'])


if __name__=='__main__':unittest.main()
