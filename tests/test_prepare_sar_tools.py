"""统一入口的任务隔离及双面板快视图回归。"""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest
import json
import numpy as np
import rasterio
from rasterio.transform import from_origin
from PIL import Image
from datasets import image_process as tools

class SarToolsTests(unittest.TestCase):
    def test_mode_executes_only_selected_task(self):
        tasks=['prepare_labels','prepare_dataset','audit','export_quicklooks']
        for mode,selected in [('labels',0),('prepare',1),('split',1),('inspect',1),('radiometry',2),('quicklook',3)]:
            with patch.object(tools,'MODE',mode), patch.object(tools,tasks[0]) as labels, patch.object(tools,tasks[1]) as prepare, patch.object(tools,tasks[2]) as audit, patch.object(tools,tasks[3]) as quick:
                tools.main()
                for i,call in enumerate([labels,prepare,audit,quick]):
                    self.assertEqual(call.call_count, int(i==selected))

    def test_comparison_keeps_raw_values_and_reuses_completed_output(self):
        with TemporaryDirectory() as temp:
            path=Path(temp)/'scene.tif'
            data=np.array([[0,1,12,255,300,7],[2,3,4,5,6,7]],dtype='float32')
            with rasterio.open(path,'w',driver='GTiff',height=2,width=6,count=1,dtype='float32',crs='EPSG:32650',transform=from_origin(0,20,3,3)) as dst:
                dst.write(data,1)
            with patch.object(tools.shutil,'disk_usage') as disk:
                disk.return_value.free=10**12
                tools.export_scene(path)
                png=path.parent/'quicklooks/scene.quick_overview.png'
                record=json.loads(png.with_suffix('.json').read_text(encoding='utf-8'))
                self.assertEqual(record['statistics']['available_stages'],['org'])
                before=png.stat().st_mtime_ns
                result=tools.export_scene(path)
                self.assertEqual(result['status'],'reused')
                self.assertEqual(png.stat().st_mtime_ns,before)
                with Image.open(png) as image:
                    self.assertEqual(record["statistics"]["overview_width"],1)
                    self.assertEqual(record["statistics"]["overview_height"],1)
                    x,y,w,h=record["panels"]["org"]
                    self.assertEqual(image.getpixel((x,y)),int(data.mean()))
                    self.assertGreater(image.height,data.shape[0])
            with rasterio.open(path) as source:
                np.testing.assert_array_equal(source.read(1),data)

    def test_average_masks_partial_edges_and_stripe_seams(self):
        with TemporaryDirectory() as temp:
            path=Path(temp)/'blocks.tif'
            data=np.arange(35,dtype='float32').reshape(5,7)
            data[:2,:2]=np.nan
            data[0,2]=np.inf
            with rasterio.open(path,'w',driver='GTiff',height=5,width=7,count=1,dtype='float32',nodata=np.nan) as dst:
                dst.write(data,1)
            with rasterio.open(path) as src, patch.multiple(tools,QUICKLOOK_LOOKS=2,QUICKLOOK_MAX_SIDE=100,STRIPE_ROWS=3):
                image,counts,looks=tools.read_overview(src)
            self.assertEqual(image.shape,(3,4))
            for row in range(3):
                for col in range(4):
                    block=data[row*2:row*2+2,col*2:col*2+2]
                    valid=block[np.isfinite(block)]
                    self.assertEqual(counts[row,col],len(valid))
                    self.assertAlmostEqual(float(image[row,col]),float(valid.mean()) if len(valid) else 0,places=5)
            with rasterio.open(path) as src, patch.multiple(tools,QUICKLOOK_LOOKS=1,QUICKLOOK_MAX_SIDE=3):
                image,_,looks=tools.read_overview(src)
                self.assertEqual(looks,3)
                self.assertLessEqual(max(image.shape),3)
