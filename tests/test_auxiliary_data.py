"""Offline acquisition tests use actual GeoTIFF bodies and controlled HTTP responses."""
import csv
from io import BytesIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from thres_segmentation.auxiliary_data import (
    tile_specs, download_tile, mosaic_tiles, selected_images, prepare_auxiliary_inputs)


class Response(BytesIO):
    def __init__(self, body, content_type='image/tiff', length=None):
        super().__init__(body)
        self.headers = {'Content-Type':content_type,'Content-Length':str(len(body) if length is None else length),'ETag':'test'}


class AuxiliaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bounds = (117.,30.,117.04,30.04)
        self.transform = from_origin(117,30.04,.01,.01)

    def raster(self,name,data=None,transform=None,nodata=255):
        path = self.root/name;path.parent.mkdir(parents=True,exist_ok=True)
        if data is None:
            data = np.full((4,4),80,dtype='uint8')
        with rasterio.open(path,'w',driver='GTiff',height=data.shape[0],width=data.shape[1],
                           count=1,dtype=data.dtype,crs='EPSG:4326',nodata=nodata,
                           transform=transform or self.transform) as target:
            target.write(data,1)
        return path

    def test_tile_edges_versions_and_southern_hemisphere(self):
        self.assertEqual(len(tile_specs('worldcover',(117,30,120,33))),1)
        self.assertIn('_2021_v200_N30E117_',tile_specs('worldcover',self.bounds)[0]['filename'])
        self.assertIn('/v100/2020/',tile_specs('worldcover',self.bounds,2020)[0]['url'])
        self.assertEqual(tile_specs('worldcover',(-.5,-.5,0,0))[0]['tile'],'S03W003')
        self.assertIn('_S01_00_W001_00_DEM',tile_specs('copernicus',(-.5,-.5,0,0))[0]['tile'])
        with self.assertRaises(ValueError):tile_specs('worldcover',self.bounds,2022)
        with self.assertRaises(ValueError):tile_specs('copernicus',(179,0,-179,1))

    def test_atomic_download_receipt_and_cache_reuse(self):
        body = self.raster('fixture.tif').read_bytes()
        dest = self.root/'cache/tile.tif'
        with patch('urllib.request.urlopen',return_value=Response(body)) as request:
            download_tile('https://example.test/tile.tif',dest,log=lambda _:None)
            self.assertEqual(request.call_count,1)
        receipt = json.loads(dest.with_suffix('.source.json').read_text('utf-8'))
        self.assertEqual(receipt['bytes'],len(body));self.assertEqual(len(receipt['sha256']),64)
        with patch('urllib.request.urlopen',side_effect=AssertionError('cache should avoid network')):
            self.assertEqual(download_tile(receipt['url'],dest,log=lambda _:None),dest)
        self.assertFalse(list(dest.parent.glob('*.part')))

    def test_reject_html_and_partial_body(self):
        body = self.raster('fixture.tif').read_bytes()
        for response in (Response(b'<html>login</html>','text/html'),Response(body,length=len(body)+10)):
            dest = self.root/'cache/tile.tif'
            with patch('urllib.request.urlopen',return_value=response), self.assertRaises(RuntimeError):
                download_tile('https://example.test/tile.tif',dest,retries=1,log=lambda _:None)
            self.assertFalse(dest.exists());self.assertFalse(list(dest.parent.glob('*.part')))

    def test_retry_then_success_and_cancel(self):
        body = self.raster('fixture.tif').read_bytes()
        with patch('urllib.request.urlopen',side_effect=[OSError('temporary'),Response(body)]) as request, patch('time.sleep'):
            download_tile('https://example.test/a',self.root/'a.tif',log=lambda _:None)
            self.assertEqual(request.call_count,2)
        with self.assertRaises(InterruptedError), patch('urllib.request.urlopen') as request:
            download_tile('https://example.test/b',self.root/'b.tif',cancel=lambda:True)
        request.assert_not_called()

    def test_mosaic_water_encoding_holes_and_native_grid(self):
        raw = np.array([[80,40,0,255],[80,10,80,80],[80,95,1,100],[20,80,80,80]],dtype='uint8')
        source = self.raster('worldcover.tif',raw)
        dest = mosaic_tiles([source],self.bounds,self.root/'water.tif',water=True,block_size=2)
        with rasterio.open(dest) as target:
            # The output may include a one-pixel alignment margin from floating point bounds.
            from rasterio.vrt import WarpedVRT
            with WarpedVRT(target,crs='EPSG:4326',transform=self.transform,width=4,height=4) as aligned:
                np.testing.assert_array_equal(aligned.read(1),[[1,0,255,255],[1,0,1,1],[1,0,255,0],[0,1,1,1]])
            self.assertAlmostEqual(target.res[0],.01)
        with patch('thres_segmentation.auxiliary_data.WarpedVRT',side_effect=AssertionError('reuse crop')):
            self.assertEqual(mosaic_tiles([source],self.bounds,dest,water=True,log=lambda _:None),dest)

    def test_two_tile_mosaic_preserves_zero_elevation(self):
        west = self.raster('west.tif',np.zeros((4,2),dtype='float32'),nodata=-9999)
        east = self.raster('east.tif',np.full((4,2),12,dtype='float32'),nodata=-9999,
                           transform=from_origin(117.02,30.04,.01,.01))
        dest = mosaic_tiles([west,east],self.bounds,self.root/'dem.tif',block_size=2)
        with rasterio.open(dest) as source:
            from rasterio.vrt import WarpedVRT
            with WarpedVRT(source,crs='EPSG:4326',transform=self.transform,width=4,height=4) as aligned:
                np.testing.assert_allclose(aligned.read(1),[[0,0,12,12]]*4)
        with self.assertRaisesRegex(ValueError,'max_pixels'):
            mosaic_tiles([west,east],self.bounds,self.root/'oversized.tif',max_pixels=1)
        self.assertFalse((self.root/'oversized.tif').exists())

    def test_manifest_selection_ignores_train_inactive_and_prefix(self):
        image = self.raster('images/val/BC3-a.tif')
        manifest = self.root/'split_manifest.csv'
        with manifest.open('w',newline='',encoding='utf-8') as stream:
            writer = csv.writer(stream);writer.writerow(['name','split','active'])
            writer.writerows([['BC3-a.tif','val',1],['missing.tif','train',1],
                             ['inactive.tif','val',0],['BC2-missing.tif','val',1]])
        self.assertEqual(selected_images(image.parent,manifest,'BC3-'),{'BC3-a.tif':image})

    def test_prepare_each_scene_and_normalized_analysis_contract(self):
        image1 = self.raster('one.tif');image2 = self.raster('two.tif',transform=from_origin(118,31,.01,.01))
        raw = self.raster('raw.tif')
        dem = self.raster('dem_source.tif',np.zeros((4,4),dtype='float32'),nodata=-9999)
        def acquire(url,destination,**kwargs):
            return raw if 'worldcover' in url else dem
        with patch('thres_segmentation.auxiliary_data.download_tile',side_effect=acquire), \
             patch('thres_segmentation.auxiliary_data.mosaic_tiles',side_effect=lambda paths,bounds,destination,**kwargs: destination):
            result = prepare_auxiliary_inputs({'one':image1,'two':image2},self.root/'cache',
                download_history=True,download_elevation=True,buffer_m=0,log=lambda _:None)
        self.assertNotEqual(result['one']['reference_path'],result['two']['reference_path'])
        self.assertEqual(result['one']['reference_water_values'],(1,))
        self.assertEqual(result['one']['elevation_kind'],'DSM')
        manifest = json.loads(next((self.root/'cache').glob('batch_*.json')).read_text('utf-8'))
        self.assertEqual(len(manifest['scenes']),2)

    def test_local_paths_take_precedence_and_bad_depth_preflight(self):
        local = dict(reference_path=self.raster('local_water.tif'),elevation_path=self.raster('local_dem.tif'),elevation_kind='DEM')
        with patch('urllib.request.urlopen',side_effect=AssertionError('no network')):
            result = prepare_auxiliary_inputs({'scene':self.root/'unread_image.tif'},self.root/'cache',
                analysis_options=local,download_history=True,download_elevation=True,log=lambda _:None)
        self.assertEqual(result['scene'],local)
        with self.assertRaisesRegex(ValueError,'requires historical'), patch('urllib.request.urlopen') as request:
            prepare_auxiliary_inputs({'scene':self.root/'missing.tif'},self.root/'cache',download_elevation=True)
        request.assert_not_called()

    def test_download_only_elevation_needs_no_history(self):
        image = self.raster('scene.tif')
        with patch('thres_segmentation.auxiliary_data.download_tile',return_value=image), \
             patch('thres_segmentation.auxiliary_data.mosaic_tiles',return_value=self.root/'downloaded_dem.tif'):
            result = prepare_auxiliary_inputs({'scene':image},self.root/'cache',
                download_elevation=True,require_depth_reference=False,buffer_m=0,log=lambda _:None)
        self.assertIn('elevation_path',result['scene'])
        self.assertNotIn('reference_path',result['scene'])

    def test_missing_tile_does_not_publish_batch_success(self):
        image = self.raster('scene.tif')
        with patch('thres_segmentation.auxiliary_data.download_tile',side_effect=RuntimeError('404')), self.assertRaises(RuntimeError):
            prepare_auxiliary_inputs({'scene':image},self.root/'cache',download_history=True)
        self.assertFalse(list((self.root/'cache').glob('batch_*.json')))

    def test_main_download_only_and_dataset_dispatch(self):
        from thres_segmentation import main
        image = self.raster('scene.tif')
        prepared = {'scene.tif':dict(reference_path='downloaded_water.tif',reference_water_values=(1,))}
        config=main.load_yaml(Path(__file__).resolve().parents[1]/'configs/threshold_segmentation.yaml')
        config['run'].update(mode='download',data_layout='patch')
        config['paths'].update(SPLIT_MANIFEST=None,IMAGE_DIR=self.root)
        config['auxiliary'].update(download_history=True,download_elevation=False)
        with patch.object(main,'load_yaml',return_value=config), \
             patch('thres_segmentation.auxiliary_data.prepare_auxiliary_inputs',return_value=prepared), \
             patch('thres_segmentation.pipeline.run_segmentation') as segment:
            self.assertEqual(main.main(),prepared);segment.assert_not_called()
        config['run']['mode']='dataset'
        with patch.object(main,'load_yaml',return_value=config), \
             patch('thres_segmentation.auxiliary_data.prepare_auxiliary_inputs',return_value=prepared), \
             patch('thres_segmentation.pipeline.run_segmentation') as segment:
            main.main()
            self.assertEqual(segment.call_args.kwargs['scene_analysis_options'],prepared)


if __name__ == '__main__':
    unittest.main()
