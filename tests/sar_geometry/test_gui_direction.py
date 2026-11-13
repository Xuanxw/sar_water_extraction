"""以GDAL实际坡向和三维法向验证方向约定，不用新旧掩膜的重合度作为真值。"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import rasterio
from rasterio.transform import from_origin

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from thres_segmentation.terrain_geometry import classify_terrain

GDALDEM=Path(os.environ.get('GDALDEM_PATH',r'E:\Code_projects\tools\libs\gdal_3.8.5\bin\gdaldem.exe'))


class DirectionTests(unittest.TestCase):
    def test_facing_and_away_slopes_against_3d_normals(self):
        """60度坡、40度入射：朝雷达坡面点积cos20，背向坡面cos100。"""
        for heading in (0,72,192.635096,350):
            for side,offset in [('right',90),('left',-90)]:
                sensor_to_ground=(heading+offset)%360;sensor=(sensor_to_ground+180)%360
                aspects=np.array([sensor,sensor_to_ground]);slope=np.array([60.,60.])
                a,b=np.deg2rad(aspects),np.deg2rad(slope);theta=np.deg2rad(40)
                normal=np.column_stack((np.sin(b)*np.sin(a),np.sin(b)*np.cos(a),np.cos(b)))
                los=np.array([np.sin(theta)*np.sin(np.deg2rad(sensor)),np.sin(theta)*np.cos(np.deg2rad(sensor)),np.cos(theta)])
                np.testing.assert_allclose(normal@los,[np.cos(np.deg2rad(20)),np.cos(np.deg2rad(100))],atol=1e-14)
                np.testing.assert_array_equal(classify_terrain(slope,aspects,heading,40,side,15),[2,1])

    def test_gdal_actual_dem_planes(self):
        """实际运行项目GDAL 3.8.5，在四种解析坡面确认其输出是下坡朝向。"""
        if not GDALDEM.is_file():self.skipTest('未配置GDALDEM_PATH，无法运行外部GDAL实测')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rr,cc=np.mgrid[0:31,0:31];x=(cc-15)*10;y=(15-rr)*10
            env=os.environ.copy();env['PATH']=str(GDALDEM.parent)+os.pathsep+env.get('PATH','')
            for direction in (0,90,180,270):
                a=np.deg2rad(direction);z=1000-np.tan(np.deg2rad(60))*(x*np.sin(a)+y*np.cos(a))
                dem=root/'dem.tif';aspect=root/'aspect.tif';slope=root/'slope.tif'
                with rasterio.open(dem,'w',driver='GTiff',count=1,width=31,height=31,dtype='float32',crs=32650,
                                   transform=from_origin(500000,4500000,10,10)) as dst:dst.write(z.astype('float32'),1)
                for mode,path in [('aspect',aspect),('slope',slope)]:
                    subprocess.run([str(GDALDEM),mode,str(dem),str(path),'-alg','ZevenbergenThorne','-q'],env=env,check=True,capture_output=True)
                with rasterio.open(aspect) as src:observed=float(src.read(1)[15,15])
                with rasterio.open(slope) as src:beta=float(src.read(1)[15,15])
                self.assertLess(abs((observed-direction+180)%360-180),1e-3)
                self.assertAlmostEqual(beta,60,places=3)
                # 雷达位于该坡面朝向一侧；航向由左视几何反推。
                heading=(direction-90)%360
                self.assertEqual(classify_terrain(np.array([beta]),np.array([observed]),heading,40,'left',15)[0],2)

    def test_flat_unknown_and_both_sides(self):
        np.testing.assert_array_equal(classify_terrain(np.array([0.,np.nan,20.,20.]),np.array([0,0,270,90]),0,40,'right',15),[4,0,3,4])
        with self.assertRaises(ValueError):classify_terrain(np.array([0.]),np.array([0.]),0,40,'unknown',15)


if __name__=='__main__':unittest.main()
