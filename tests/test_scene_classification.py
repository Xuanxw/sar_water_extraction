"""场景地物分类：合成栅格验证，不联网、不运行实际场景任务。"""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import numpy as np
import rasterio
from rasterio.transform import from_origin
from datasets import classify_scenes as module


class SceneClassificationTests(unittest.TestCase):
    def options(self):
        return dict(module.OPTIONS)

    def test_rules_and_missing_coverage(self):
        options=self.options();counts=np.zeros(11,dtype=np.int64)
        counts[3]=60;counts[4]=40
        row=module.summarize(counts,100,100,30,1500,options)
        self.assertEqual(row['主导地物'],'农田')
        self.assertEqual(row['较多地物'],'农田；建成区')
        self.assertEqual(row['地形标签'],'山地/陡坡候选')
        missing=module.summarize(counts,200,0,0,0,options)
        self.assertIn('覆盖不足',missing['主导地物'])
        self.assertEqual(missing['平均坡度_度'],'')

    def test_raster_counts_and_block_seam_slopes(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);transform=from_origin(500000,3000000,30,30)
            profile=dict(driver='GTiff',height=120,width=120,count=1,crs='EPSG:32648',transform=transform)
            sar=np.ones((120,120),dtype='float32');sar[0,:]=-9999
            cover=np.full(sar.shape,40,dtype='uint8');cover[:,60:]=50
            dem=np.broadcast_to(np.arange(120,dtype='float32')*15,sar.shape).copy()
            for name,array,nodata in [('sar',sar,-9999),('cover',cover,0),('dem',dem,-9999)]:
                with rasterio.open(root/f'{name}.tif','w',**profile,dtype=array.dtype,nodata=nodata) as dst:
                    dst.write(array,1)
            options=self.options();options['BLOCK_SIZE']=32
            with patch.object(module,'get_tiles',side_effect=lambda product,*args:[root/('cover.tif' if product=='worldcover' else 'dem.tif')]):
                result=module.classify_scene(root/'sar.tif',root,options)
            self.assertEqual(result['状态'],'完成')
            self.assertEqual(result['SAR抽样有效像素'],119*120)
            self.assertAlmostEqual(result['农田占比'],.5,places=5)
            self.assertAlmostEqual(result['建成区占比'],.5,places=5)
            self.assertAlmostEqual(result['平均坡度_度'],np.degrees(np.arctan(.5)),places=4)
            self.assertEqual(result['陡坡占比'],1)
            self.assertGreater(result['地形覆盖率'],.95)

    def test_dedup_and_split_conflict(self):
        with TemporaryDirectory() as tmp:
            a=Path(tmp)/'a.csv';b=Path(tmp)/'b.csv'
            header='scene_id,name,split,group_id,active\n'
            a.write_text(header+'test,test.tif,train,G1,1\n')
            b.write_text(a.read_text())
            self.assertEqual(len(module.read_scenes([a,b])),1)
            b.write_text(header+'test,test.tif,val,G1,1\n')
            with self.assertRaisesRegex(ValueError,'冲突'):
                module.read_scenes([a,b])

    def test_missing_product_is_unknown(self):
        options=self.options()
        row=module.summarize(np.zeros(11,dtype=np.int64),100,0,0,0,options)
        self.assertEqual(row['农田占比'],'')
        self.assertIn('覆盖不足',row['地形标签'])


if __name__=='__main__':
    unittest.main()
