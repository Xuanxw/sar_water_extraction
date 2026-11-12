"""核验GUI核心：属性保存、几何保护、撤销、影像坐标和瓦片缓存。"""
import io
from contextlib import closing
import sqlite3
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import fiona
import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_origin
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from validate_pts.review_store import ReviewStore, FIELDS
from validate_pts.map_layers import render_local, map_bounds, fetch_tile


def make_points(path):
    properties={k:'str' for k in ('point_id','scene_id','date','stratum','confidence','note','reviewer','reference','review_date')}
    properties.update(ref_class='int',pred='int',row='int',col='int',weight='float')
    with fiona.open(path,'w',driver='GPKG',layer='verification_points',crs='EPSG:4326',
                    schema={'geometry':'Point','properties':properties}) as dst:
        for i in range(2):
            p={k:'' for k in properties}
            p.update(point_id=f'p{i}',scene_id='TEST_SCENE_20250730',date='2025-07-30',stratum='shore_buffer',
                     ref_class=None,pred=1,row=10,col=10+i,weight=120.)
            dst.write({'geometry':{'type':'Point','coordinates':(117+i*.001,40.5)},'properties':p})


class PointViewerTests(unittest.TestCase):
    def test_save_undo_preserve_geometry_and_backup(self):
        with TemporaryDirectory() as temp:
            path=Path(temp)/'points.gpkg';make_points(path);store=ReviewStore(path)
            with closing(sqlite3.connect(path)) as c:
                original=list(c.execute('SELECT point_id,geom,weight,pred FROM verification_points'))
                rtree=list(c.execute('SELECT * FROM rtree_verification_points_geom'))
            values=dict(ref_class=1,confidence='高',note='已核验',reviewer='测试',reference='同期光学',review_date='2026-11-11')
            store.save('p0',values)
            self.assertEqual(ReviewStore(path).by_id['p0']['ref_class'],1)
            self.assertIsNone(ReviewStore(store.backup_path).by_id['p0']['ref_class'])
            with closing(sqlite3.connect(path)) as c:
                self.assertEqual(original,list(c.execute('SELECT point_id,geom,weight,pred FROM verification_points')))
                self.assertEqual(rtree,list(c.execute('SELECT * FROM rtree_verification_points_geom')))
                self.assertEqual(c.execute('PRAGMA integrity_check').fetchone()[0],'ok')
            store.undo();self.assertIsNone(ReviewStore(path).by_id['p0']['ref_class'])

    def test_concurrent_edit_not_overwritten(self):
        with TemporaryDirectory() as temp:
            path=Path(temp)/'points.gpkg';make_points(path);a=ReviewStore(path);b=ReviewStore(path)
            value={k:a.by_id['p0'][k] for k in FIELDS};value['ref_class']=0
            a.save('p0',value)
            with self.assertRaisesRegex(ValueError,'其他程序'):b.save('p0',value)

    def test_local_reprojection_and_outside_coverage(self):
        with TemporaryDirectory() as temp:
            path=Path(temp)/'rgb.tif'
            with rasterio.open(path,'w',driver='GTiff',crs='EPSG:4326',transform=from_origin(116.99,40.51,.0002,.0002),
                               width=100,height=100,count=3,dtype='uint8') as dst:
                dst.write(np.stack([np.full((100,100),v,np.uint8) for v in (40,120,200)]))
            from pyproj import Transformer
            center=Transformer.from_crs(4326,3857,always_xy=True).transform(117,40.5)
            image,msg=render_local(path,map_bounds(center,17,100,80),100,80,stretch=False)
            np.testing.assert_array_equal(np.asarray(image)[40,50],[40,120,200])
            image,msg=render_local(path,map_bounds((0,0),17,100,80),100,80,stretch=False)
            self.assertIn('无有效',msg)
            np.testing.assert_array_equal(np.asarray(image)[40,50],[38,44,52])

    def test_tile_cache_and_no_secret_in_error(self):
        payload=io.BytesIO();Image.new('RGB',(256,256),'blue').save(payload,format='PNG')
        with TemporaryDirectory() as temp:
            with patch('validate_pts.map_layers.urlopen',return_value=io.BytesIO(payload.getvalue())) as request:
                result=fetch_tile((1,2,4,'private-test-token',temp));self.assertIsNone(result[3]);self.assertEqual(request.call_count,1)
            with patch('validate_pts.map_layers.urlopen',side_effect=AssertionError('should use cache')):
                self.assertIsNone(fetch_tile((1,2,4,'private-test-token',temp))[3])
            with patch('validate_pts.map_layers.urlopen',side_effect=ValueError('private-test-token')):
                result=fetch_tile((2,2,4,'private-test-token',temp));self.assertNotIn('private-test-token',result[3])

    def test_gui_save_and_reload_with_offline_image(self):
        import time
        import tkinter as tk
        from validate_pts.main import PointReviewer
        with TemporaryDirectory() as temp:
            path=Path(temp)/'points.gpkg';make_points(path)
            root=tk.Tk();root.withdraw();app=PointReviewer(root,path)
            try:
                app.reference.set('同期光学 2025-07-30');app.auto_next.set(False)
                self.assertTrue(app.save(1,advance=False))
                self.assertEqual(ReviewStore(path).by_id['p0']['ref_class'],1)
                app.undo();self.assertIsNone(ReviewStore(path).by_id['p0']['ref_class'])
            finally:app.close()


if __name__=='__main__':unittest.main()
