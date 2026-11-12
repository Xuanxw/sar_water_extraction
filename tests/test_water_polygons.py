import sys,unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import rasterio,fiona
from rasterio.transform import from_origin
from scipy import ndimage
from shapely.geometry import shape
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from thres_segmentation.water_polygons import clean_mask,build_water_polygons

class WaterPolygonTests(unittest.TestCase):
    def test_holes_components_exclusion_nodata_and_boundary(self):
        a=np.zeros((70,80),np.uint8);a[5:55,5:55]=1
        a[20:22,20:22]=0;a[30:34,30:34]=0;a[2,2]=1;a[:2,:]=255
        a[5:55,50:52]=0
        protected=np.zeros_like(a,bool);protected[30:34,30:34]=True;protected[:,50:52]=True
        result,report=clean_mask(a.copy(),protected,dict(hole_max_pixels=20,morph_kernel_size=3,min_water_pixels=30))
        self.assertTrue(np.all(result[20:22,20:22]==1))
        self.assertTrue(np.all(result[protected&(a!=255)]==0))
        np.testing.assert_array_equal(result==255,a==255)
        self.assertEqual(result[2,2],0)
        self.assertGreater(report['filled_pixels'],0)
    def test_morphology_across_row_blocks(self):
        a=np.zeros((600,70),np.uint8);a[80:550,10:60]=1;a[255:258,25:28]=0
        expected=ndimage.binary_closing(ndimage.binary_opening(a==1,structure=ndimage.generate_binary_structure(2,1)),structure=ndimage.generate_binary_structure(2,1))
        result,_=clean_mask(a.copy(),np.zeros_like(a,bool),dict(hole_max_pixels=0,morph_kernel_size=3,min_water_pixels=1))
        np.testing.assert_array_equal(result==1,expected)
    def test_polygon_export_smoothing_and_original_preserved(self):
        with TemporaryDirectory() as tmp:
            p=Path(tmp);src=p/'water.tif';dst=p/'water_polygon.shp'
            a=np.zeros((80,80),np.uint8);a[10:60,10:60]=1;a[20:22,20:22]=0;a[35:45,35:45]=0;a[1,1]=1;a[0]=255
            with rasterio.open(src,'w',driver='GTiff',width=80,height=80,count=1,dtype='uint8',nodata=255,crs='EPSG:32650',transform=from_origin(500000,4000000,3,3)) as f:f.write(a,1)
            original=src.read_bytes()
            build_water_polygons(src,dst,options=dict(hole_max_pixels=10,morph_kernel_size=3,min_water_pixels=200,smooth_iterations=2))
            self.assertEqual(src.read_bytes(),original)
            with fiona.open(dst) as f:
                polygons=[shape(x['geometry']) for x in f]
            self.assertEqual(len(polygons),1);self.assertTrue(polygons[0].is_valid)
            self.assertEqual(len(polygons[0].interiors),1)
            before=dst.stat().st_mtime_ns
            build_water_polygons(src,dst,options=dict(hole_max_pixels=10,morph_kernel_size=3,min_water_pixels=200,smooth_iterations=2))
            self.assertEqual(before,dst.stat().st_mtime_ns)

if __name__=='__main__':unittest.main()
