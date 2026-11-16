"""短轨道/矩形外角回归：不能拒绝可解像元，也不能伪造域外有效根。"""
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import rasterio
from rasterio.transform import from_origin

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from sar_geometry.core import solve_range_doppler_points,interpolate_hermite_orbit_states
from sar_geometry.scene import make_zero_doppler_profile_grid
from test_geometry_core import straight_model,CORE


class ShortOrbitTests(unittest.TestCase):
    def test_overshoot_recover_and_outside_root_rejected(self):
        """解析加速轨道的根t=2在支持内；初次Newton越界不能导致提前失败。"""
        m=straight_model();m.update(time0=-1,rows=401,max_step_seconds=10)
        m['orbit']=np.array([[t,7178137,1000*t+500*t*t,0,0,1000+1000*t,0] for t in [0,2.1]])
        good=solve_range_doppler_points(CORE,m,[[6378137,4000,100000]])[0]
        self.assertEqual(good[5],0);self.assertAlmostEqual(good[2],2,places=8)
        outside=solve_range_doppler_points(CORE,m,[[6378137,7500,100000]])[0]
        self.assertEqual(outside[5],2)

    def test_grid_uses_supported_time_not_invalid_rectangle_corners(self):
        m=straight_model();m['orbit']=np.array([[t,7178137,7500*t,0,0,7500,0] for t in [0,.5]])
        options=dict(height_bounds_m=[-12000,10000],azimuth_step_m=100,range_step_m=100,
                     support_buffer_m=1000,max_profile_points=1000000)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'ref.tif'
            with rasterio.open(path,'w',driver='GTiff',count=1,width=10,height=10,dtype='uint8',crs=4326,
                               transform=from_origin(-.2,5.1,.04,.02)) as dst:dst.write(np.ones((10,10),np.uint8),1)
            times,theta=make_zero_doppler_profile_grid({'scene':options},m,path,CORE)
        self.assertEqual(times[0],0);self.assertEqual(times[-1],.5)
        self.assertTrue(np.all(np.diff(times)>0));self.assertTrue(np.all(np.diff(theta)>0))
        self.assertTrue(np.isfinite(interpolate_hermite_orbit_states(CORE,m['orbit'],times)).all())


if __name__=='__main__':unittest.main()
