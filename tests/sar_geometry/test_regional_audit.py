"""独立射线证据和SAR网格未知域传播测试。"""
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch
import numpy as np
import rasterio
from rasterio.transform import from_origin

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT))
from scripts.terrain.audit_terrain_disagreements import trace_sar_line_of_sight
from thres_segmentation.profile_terrain import align_profile_masks
from thres_segmentation.org_pipeline import segment
from configuration import load_yaml
from shapely.geometry import box


class Surface:
    def __init__(self,mode):self.mode=mode
    def sample(self,lon,lat):
        if self.mode=='missing':return np.full_like(lon,np.nan)
        if self.mode=='ridge':return np.where((lon<-.0002)&(lon>-.001),100.,0.)
        return np.zeros_like(lon)


class RegionalAuditTests(unittest.TestCase):
    def test_ray_flat_ridge_and_missing(self):
        p=np.array([[6378137.,0,0]]);state=np.array([[6878137.,-400000,0,0,0,0]])
        for mode,expected in [('flat','clear'),('ridge','blocked'),('missing','uncertain')]:
            value,_=trace_sar_line_of_sight(p,state,Surface(mode))
            self.assertEqual(value[0]['ray_status'],expected)
        coarse,_=trace_sar_line_of_sight(p,state,Surface('ridge'),7.5)
        fine,_=trace_sar_line_of_sight(p,state,Surface('ridge'),3.75)
        self.assertEqual(coarse[0]['ray_status'],fine[0]['ray_status'])

    def test_align_preserves_unknown_and_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);geo=root/'geo';out=root/'out';geo.mkdir();out.mkdir()
            profile=dict(driver='GTiff',width=4,height=4,count=1,dtype='uint8',crs=32650,
                         transform=from_origin(500000,4500000,30,30),nodata=255)
            data=np.array([[255,1,0,0],[0,0,1,255],[0,0,0,0],[0,0,0,0]],np.uint8)
            for name in ['shadow_mask','layover_mask','terrain_exclusion']:
                with rasterio.open(geo/(name+'.tif'),'w',**profile) as dst:dst.write(data,1)
            ref=root/'reference.tif'
            with rasterio.open(ref,'w',**profile) as dst:dst.write(np.zeros((4,4),np.uint8),1)
            paths=align_profile_masks(ref,geo,out,2)
            for path in paths.values():
                with rasterio.open(path) as src:np.testing.assert_array_equal(src.read(1),data)

    def test_segmentation_uses_profile_and_preserves_unknown_candidates(self):
        """新入口按剖面1排除；255不硬删，且不叠加原15度高坡规则。"""
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'sar.tif';terrain=root/'terrain_exclusion.tif'
            p=dict(driver='GTiff',width=4,height=4,count=1,dtype='float32',crs=32650,
                   transform=from_origin(500000,4500000,30,30))
            with rasterio.open(source,'w',**p) as dst:dst.write(np.full((4,4),.1,np.float32),1)
            flags=np.zeros((4,4),np.uint8);flags[0,0]=1;flags[0,1]=255
            with rasterio.open(terrain,'w',**dict(p,dtype='uint8',nodata=255)) as dst:dst.write(flags,1)
            c=load_yaml(ROOT/'configs/北京密云区/corrected_amplitude_otsu_autoheight.yaml')
            c['processing']['block_size']=4;post=c['org']['postprocess']
            post.update(history_enabled=False,landcover_enabled=False,opening_radius_px=0,closing_radius_px=0)
            with patch('thres_segmentation.profile_terrain.generate_profile_terrain',return_value={'terrain_exclusion':terrain}):
                segment(source,box(500000,4499880,500120,4500000),{},c,root,
                        {'thresholds_amplitude':[1,1,1,1],'mean':0,'std':1,'decision_threshold':1})
            with rasterio.open(root/'water.tif') as src:value=src.read(1)
            self.assertEqual(value[0,0],0);self.assertEqual(value[0,1],1)
            self.assertTrue((value[1:]==1).all())


if __name__=='__main__':unittest.main()
