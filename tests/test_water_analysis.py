"""Scientific contract tests: area denominators, coverage and depth validity."""
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import fiona
import numpy as np
import rasterio
from rasterio.transform import from_origin
from pyproj import Geod

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from thres_segmentation.water_analysis import analyze_water, estimate_depth, pixel_areas, read_reference_water


class WaterAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.transform = from_origin(500000, 3000000, 10, 10)
        self.crs = 'EPSG:32650'

    def raster(self, name, data, transform=None, crs=None, nodata=255, mask=None):
        path = self.root/name
        with rasterio.open(path,'w',driver='GTiff',height=data.shape[0],width=data.shape[1],
                           count=1,dtype=data.dtype,crs=crs or self.crs,
                           transform=transform or self.transform,nodata=nodata) as dst:
            dst.write(data,1)
            if mask is not None:
                dst.write_mask(mask.astype('uint8')*255)
        return path

    def vector(self, name, polygons):
        path = self.root/name
        with fiona.open(path,'w',driver='GPKG',crs=self.crs,
                        schema={'geometry':'Polygon','properties':{}}) as dst:
            for coords in polygons:
                dst.write({'geometry':{'type':'Polygon','coordinates':[coords]},'properties':{}})
        return path

    def test_area_projected_feet_rotated_and_geographic(self):
        self.assertEqual(pixel_areas(self.crs,self.transform,(2,3)),100)
        foot = pixel_areas('EPSG:2277',from_origin(0,0,10,10),(1,1))
        self.assertAlmostEqual(foot,100*(1200/3937)**2)
        affine = rasterio.Affine(3,2,0,1,-4,0)
        self.assertEqual(pixel_areas(self.crs,affine,(2,2)),14)
        transform = from_origin(110,30,.01,.01)
        areas = pixel_areas('EPSG:4326',transform,(2,2))
        expected = abs(Geod(ellps='WGS84').polygon_area_perimeter([110,110.01,110.01,110],[30,30,29.99,29.99])[0])
        self.assertAlmostEqual(areas[0,0],expected,places=3)
        self.assertGreater(areas[1,0],areas[0,0])

    def test_reference_reader_alignment_custom_codes_and_vector_coverage(self):
        target = self.raster('target.tif', np.zeros((3,3),dtype='uint8'))
        # 不同分辨率/范围，0代表水；范围外仍然未知，而非默认0水体。
        previous = self.raster('reference.tif', np.zeros((1,1),dtype='uint8'), nodata=None,
                               transform=from_origin(500010,2999990,20,20))
        water, known = read_reference_water(target, previous, (0,), (1,))
        expected = np.zeros((3,3),dtype=bool); expected[1:,1:] = True
        np.testing.assert_array_equal(water,expected)
        np.testing.assert_array_equal(known,expected)
        polygon = [(500010,2999990),(500030,2999990),(500030,2999970),(500010,2999970),(500010,2999990)]
        vector = self.vector('reference.gpkg',[polygon])
        with self.assertRaisesRegex(ValueError,'coverage'):
            read_reference_water(target,vector)
        water, known = read_reference_water(target,vector,reference_coverage_path=vector)
        np.testing.assert_array_equal(water,expected)
        np.testing.assert_array_equal(known,expected)

    def test_change_denominator_nodata_and_vector_area(self):
        current = np.array([[1,1,0],[0,255,1]],dtype='uint8')
        history = np.array([[1,0,1],[0,1,255]],dtype='uint8')
        result = analyze_water(self.raster('now.tif',current),self.root/'out',
                               reference_path=self.raster('old.tif',history),block_size=1)
        self.assertEqual(result['current_water']['m2'],300)
        for category,expected in [('comparison_valid',400),('new_water',100),('lost_water',100),
                                  ('persistent_water',100),('unknown_history',100),('net_change',0)]:
            self.assertEqual(result['change'][category]['m2'],expected)
        with rasterio.open(self.root/'out/water_change.tif') as src:
            np.testing.assert_array_equal(src.read(1),[[1,2,3],[0,255,255]])
        with fiona.open(self.root/'out/water_changes.gpkg') as src:
            self.assertEqual(sum(f['properties']['area_m2'] for f in src),200)
        self.assertTrue((self.root/'out/areas.csv').exists())
        with self.assertRaises(FileExistsError):
            analyze_water(self.root/'now.tif',self.root/'out')

    def test_reversed_history_and_uncovered_zero(self):
        current = self.raster('now.tif',np.ones((3,3),dtype='uint8'))
        # 0 is water, no declared NoData; outside the 1-pixel footprint must stay unknown.
        history = self.raster('old.tif',np.zeros((1,1),dtype='uint8'),nodata=None,
                              transform=from_origin(500010,2999990,10,10))
        result = analyze_water(current,self.root/'out',reference_path=history,
                               reference_water_values=(0,),reference_dry_values=(1,))
        self.assertEqual(result['change']['persistent_water']['m2'],100)
        self.assertEqual(result['change']['unknown_history']['m2'],800)
        self.assertEqual(result['change']['new_water']['m2'],0)

    def test_internal_mask_remains_unknown(self):
        current = self.raster('now.tif',np.ones((2,2),dtype='uint8'))
        old = self.raster('old.tif',np.zeros((2,2),dtype='uint8'),nodata=None,
                          mask=np.array([[True,False],[True,True]]))
        result = analyze_water(current,self.root/'out',reference_path=old)
        self.assertEqual(result['change']['new_water']['m2'],300)
        self.assertEqual(result['change']['unknown_history']['m2'],100)

    def test_vector_coverage_required_and_outside_unknown(self):
        current = self.raster('now.tif',np.ones((2,2),dtype='uint8'))
        water = self.vector('water.gpkg',[[(500000,3000000),(500010,3000000),
                           (500010,2999990),(500000,2999990),(500000,3000000)]])
        coverage = self.vector('coverage.gpkg',[[(500000,3000000),(500020,3000000),
                              (500020,2999990),(500000,2999990),(500000,3000000)]])
        with self.assertRaisesRegex(ValueError,'coverage'):
            analyze_water(current,self.root/'fail',reference_path=water)
        result = analyze_water(current,self.root/'out',reference_path=water,reference_coverage_path=coverage)
        self.assertEqual(result['change']['persistent_water']['m2'],100)
        self.assertEqual(result['change']['new_water']['m2'],100)
        self.assertEqual(result['change']['unknown_history']['m2'],200)
        complete = analyze_water(current,self.root/'complete',reference_path=water,vector_assume_complete=True)
        self.assertEqual(complete['change']['new_water']['m2'],300)

    def basin(self):
        water = np.zeros((9,9),dtype=bool);water[2:7,2:7]=True
        elevation = np.full((9,9),10.);elevation[water]=7.
        return water,np.ones((9,9),bool),elevation

    def test_depth_flat_basin_zero_missing_and_negative(self):
        water,valid,z = self.basin()
        z[3,3]=10;z[3,4]=11
        zvalid = valid.copy();zvalid[3,5]=False
        depth,level,quality,_,records = estimate_depth(water,valid,z,zvalid,self.transform,self.crs,scope='all_water')
        self.assertAlmostEqual(depth[4,4],3)
        self.assertEqual(depth[3,3],0)
        self.assertEqual(quality[3,3],1)
        self.assertEqual(quality[3,4],5)
        self.assertTrue(np.isnan(depth[3,4]))
        self.assertEqual(quality[3,5],2)
        self.assertEqual(records[0]['estimated_pixels'],23)

    def test_new_water_excludes_history_unknown_and_permanent(self):
        water,valid,z = self.basin()
        history = np.zeros_like(water);history[3,3]=True
        known = valid.copy();known[3,4]=False
        depth,_,quality,_,_ = estimate_depth(water,valid,z,valid,self.transform,self.crs,
                                            history,known,scope='new_water')
        self.assertEqual(quality[3,3],6);self.assertEqual(quality[3,4],7)
        self.assertTrue(np.isnan(depth[3,3]));self.assertAlmostEqual(depth[4,4],3)

    def test_invalid_boundaries_and_distance_limit(self):
        water,valid,z = self.basin()
        _,_,quality,_,_ = estimate_depth(water,valid,z,valid,self.transform,self.crs,
                                          scope='all_water',max_distance_m=5)
        self.assertTrue(np.all(quality[water]==4))
        depth,_,quality,_,_ = estimate_depth(water,water,z,valid,self.transform,self.crs,scope='all_water')
        self.assertTrue(np.isnan(depth).all())
        self.assertTrue(np.all(quality[water]==3))
        # A fully flooded scene has no observed dry shoreline.
        _,_,quality,_,_ = estimate_depth(valid,valid,z,valid,self.transform,self.crs,scope='all_water')
        self.assertTrue(np.all(quality==3))

    def test_disconnected_water_has_independent_levels(self):
        water = np.zeros((9,19),bool);water[2:7,2:7]=True;water[2:7,12:17]=True
        z = np.full(water.shape,10.);z[:,10:]=20.;z[water]-=3
        valid = np.ones_like(water)
        depth,level,_,_,records = estimate_depth(water,valid,z,valid,self.transform,self.crs,scope='all_water')
        np.testing.assert_allclose(depth[water],3,atol=1e-6)
        self.assertAlmostEqual(level[4,4],10);self.assertAlmostEqual(level[4,14],20)
        self.assertEqual(len(records),2)

    def test_depth_outputs_unit_scale_volume_and_guard(self):
        water,valid,z = self.basin()
        mask = self.raster('now.tif',water.astype('uint8'))
        dem = self.raster('dsm.tif',(z*100).astype('float32'),nodata=-9999)
        with self.assertRaisesRegex(ValueError,'depth_max_pixels'):
            analyze_water(mask,self.root/'too_big',elevation_path=dem,depth_max_pixels=1)
        with self.assertRaisesRegex(ValueError,'historical'):
            analyze_water(mask,self.root/'no_history',elevation_path=dem)
        result = analyze_water(mask,self.root/'out',elevation_path=dem,elevation_kind='DSM',
                               elevation_scale=.01,depth_scope='all_water',block_size=3)
        self.assertAlmostEqual(result['depth']['estimated_area']['m2'],2500)
        self.assertAlmostEqual(result['depth']['estimated_volume_m3'],7500)
        self.assertAlmostEqual(result['depth']['area_weighted_mean_m'],3)
        self.assertTrue(any('DSM measures' in w for w in result['warnings']))
        with rasterio.open(self.root/'out/water_depth_m.tif') as src:
            self.assertEqual(src.nodata,-9999)
            self.assertEqual(src.read(1)[0,0],-9999)
            self.assertAlmostEqual(src.read(1)[4,4],3)
        parsed = json.loads((self.root/'out/summary.json').read_text('utf-8'))
        self.assertEqual(parsed['status'],'complete')

    def test_empty_water_and_no_overlap(self):
        mask = self.raster('now.tif',np.zeros((2,2),dtype='uint8'))
        old = self.raster('old.tif',np.ones((2,2),dtype='uint8'),transform=from_origin(501000,3000000,10,10))
        result = analyze_water(mask,self.root/'out',reference_path=old)
        self.assertEqual(result['current_water']['m2'],0)
        self.assertEqual(result['change']['comparison_valid']['m2'],0)
        self.assertEqual(result['change']['lost_water']['m2'],0)
        with fiona.open(self.root/'out/water_changes.gpkg') as src:
            self.assertEqual(len(src),0)

    def test_patch_pipeline_analysis_without_saved_mask(self):
        from thres_segmentation.pipeline import run_segmentation
        water,valid,z = self.basin()
        images = self.root/'images';images.mkdir()
        self.raster('images/scene.tif',np.where(water,1,100).astype('float32'),nodata=-9999)
        history = self.raster('history.tif',np.zeros(water.shape,dtype='uint8'))
        dem = self.raster('dem.tif',z.astype('float32'),nodata=-9999)
        run_segmentation(images,bands={'VV':1},output_dir=self.root/'out',use_landcover=False,
                         save_masks=False,max_visualizations=0,
                         threshold_options=dict(lee_window=1,water_percent=40),
                         analysis_options=dict(reference_path='unused_base_reference.tif'),
                         scene_analysis_options={'scene.tif':dict(reference_path=history,elevation_path=dem)})
        self.assertFalse((self.root/'out/masks').exists())
        result = json.loads((self.root/'out/analysis/threshold_VV/scene/summary.json').read_text('utf-8'))
        self.assertEqual(result['current_water']['m2'],2500)
        self.assertEqual(result['change']['new_water']['m2'],2500)
        self.assertAlmostEqual(result['depth']['estimated_volume_m3'],7500)

    def test_reference_reprojected_from_geographic_crs(self):
        from rasterio.warp import transform_bounds
        current = self.raster('now.tif',np.ones((2,2),dtype='uint8'))
        west,south,east,north = transform_bounds(self.crs,'EPSG:4326',499900,2999880,500120,3000100)
        history = self.raster('old.tif',np.ones((20,20),dtype='uint8'),crs='EPSG:4326',
                              transform=from_origin(west,north,(east-west)/20,(north-south)/20))
        result = analyze_water(current,self.root/'out',reference_path=history)
        self.assertEqual(result['change']['persistent_water']['m2'],400)
        self.assertEqual(result['change']['unknown_history']['m2'],0)


if __name__ == '__main__':
    unittest.main()
