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
from sar_geometry.core import classify_profiles, solve_points, orbit_states
from sar_geometry.profiles import build_profiles, TO_GEODETIC
from sar_geometry.scene import run_visibility, combine_masks
from sar_geometry.metadata import read_bc_metadata
from test_geometry_core import straight_model, CORE
from test_metadata_and_nominal import xml_text


class FlatTerrain:
    def sample(self,lon,lat):return np.full_like(lon,100.)


class ProfileGridTests(unittest.TestCase):
    def test_surface_observation_plane_and_roundtrip(self):
        model=straight_model();theta=np.linspace(.05,.08,200);times=np.array([0.,2.,4.])
        masks,points,good,error=build_profiles(CORE,model,FlatTerrain(),times,theta,[-100,1000])
        self.assertTrue(good.all());self.assertLess(error,.05)
        states=orbit_states(CORE,model['orbit'],times)
        plane=np.sum((points-states[:,None,:3])*states[:,None,3:],axis=-1)
        np.testing.assert_allclose(plane,0,atol=1e-3)
        _,_,h=TO_GEODETIC.transform(points[...,0],points[...,1],points[...,2])
        np.testing.assert_allclose(h,100,atol=.05)
        result=solve_points(CORE,model,points.reshape(-1,3))
        np.testing.assert_allclose(result[:,2].reshape(3,200),np.broadcast_to(times[:,None],(3,200)),atol=1e-7)
        self.assertFalse(np.any(masks==1))
        other,_,_,_=build_profiles(CORE,model,FlatTerrain(),times[1:],theta,[-100,1000])
        np.testing.assert_array_equal(other,masks[1:])

    def test_unknown_boundaries_threads_and_holes(self):
        p=np.array([[0,100,-.5,0],[1,110,-.4,1],[2,120,-.2,1],[3,130,-.3,1],[4,140,0,1],[5,150,.1,0]],float)
        batch=np.stack([p,p]);bounds=np.array([[-.25,115,145]]*2)
        with patch.dict(os.environ,{'OMP_NUM_THREADS':'1'}):one=classify_profiles(CORE,batch,bounds)
        with patch.dict(os.environ,{'OMP_NUM_THREADS':'4'}):many=classify_profiles(CORE,batch,bounds)
        np.testing.assert_array_equal(one,many)
        self.assertEqual(one[0,1,0],255);self.assertEqual(one[0,2,0],0)
        self.assertEqual(one[0,3,0],1);self.assertEqual(one[0,1,1],255)
        batch[:,3,3]=0;hole=classify_profiles(CORE,batch,bounds)
        self.assertTrue(np.all(hole[:,:,1]==255));self.assertEqual(hole[0,4,0],255)

    def test_union_keeps_unknown(self):
        np.testing.assert_array_equal(combine_masks(np.array([0,255,1,255]),np.array([0,0,255,1])),[0,255,1,1])


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
    def test_resume_and_chunk_equivalence(self):
        from sar_geometry.profiles import build_profiles as real
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config=scene_config(root);meta=read_bc_metadata(config['paths']['XML_PATH']);calls=0
            def interrupt(*args,**kw):
                nonlocal calls
                calls+=1
                if calls==2:raise RuntimeError('test interruption')
                return real(*args,**kw)
            with patch('sar_geometry.scene.build_profiles',side_effect=interrupt),self.assertRaisesRegex(RuntimeError,'interruption'):
                run_visibility(config,meta,lambda x:None)
            state=json.loads((root/'out/data_cache/scene_progress.json').read_text())
            self.assertEqual(state['next_profile'],7);self.assertFalse(state['complete'])
            run_visibility(config,meta,lambda x:None)
            other=copy.deepcopy(config);other['paths']['OUTPUT_DIR']=root/'other'
            other['execution'].update(profile_batch_rows=40,map_block_rows=40,threads=1)
            run_visibility(other,meta,lambda x:None)
            for path in (root/'out').glob('*.tif'):
                with rasterio.open(path) as a,rasterio.open(root/'other'/path.name) as b:
                    self.assertEqual((a.transform,a.crs,a.shape),(b.transform,b.crs,b.shape))
                    np.testing.assert_array_equal(a.read(),b.read())
            with patch('sar_geometry.scene.build_profiles',side_effect=AssertionError('must reuse')):
                report=run_visibility(config,meta,lambda x:None)
            self.assertTrue(report['complete'])


if __name__=='__main__':unittest.main()
