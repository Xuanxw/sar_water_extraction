"""拼接保留重叠面，并支持日期完成后的复用和中断重写。"""
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import fiona
from shapely.geometry import box, mapping, shape
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import merge_water_shp as merge

class MergeAppendTests(unittest.TestCase):
    def test_append_and_resume(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);source=root/'source';output=root/'merged'
            geoms=[box(0,0,2,2),box(1,0,3,2),box(0,0,2,2)]
            for i,g in enumerate(geoms):
                folder=source/'2026-11-12'/str(i);folder.mkdir(parents=True)
                with fiona.open(folder/'water_polygon.shp','w',driver='ESRI Shapefile',crs='EPSG:32650',schema={'geometry':'Polygon','properties':{'id':'int'}}) as dst:
                    dst.write({'geometry':mapping(g),'properties':{'id':i}})
                (folder/'scene.json').write_text(json.dumps({'date':'2026-11-12','status':'complete','scene_id':str(i),'sensor':'BC3'}))
            with patch.object(merge,'WRITE_BATCH_SIZE',2),patch.object(merge,'CONSOLE_COLOR',False):
                with patch.object(merge,'save_quicklook',side_effect=RuntimeError('test interruption')):
                    with self.assertRaisesRegex(RuntimeError,'test interruption'):
                        merge.merge_results(source,output,'EPSG:32650')
                self.assertFalse((output/'2026-11-12_polygon_sources.json').exists())
                merge.merge_results(source,output,'EPSG:32650',save_previews=False)
                shp=output/'2026-11-12_water_polygon.shp'
                with fiona.open(shp) as src:
                    features=list(src)
                self.assertEqual(len(features),3)
                self.assertEqual(sum(shape(f['geometry']).area for f in features),12)
                for i,f in enumerate(features):
                    self.assertTrue(shape(f['geometry']).equals(geoms[i]))
                    self.assertEqual(f['properties']['scene_id'],str(i))
                    self.assertEqual(f['properties']['sensor'],'BC3')
                before=shp.stat().st_mtime_ns
                merge.merge_results(source,output,'EPSG:32650',save_previews=False)
                self.assertEqual(before,shp.stat().st_mtime_ns)
                self.assertFalse((output/'.merge_cache').exists())
                meta=json.loads((output/'2026-11-12_polygon_sources.json').read_text(encoding='utf-8'))
                self.assertEqual(meta['method'],'feature_append')
                self.assertEqual(meta['feature_count'],3)

    def test_rejects_point_input(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);folder=root/'source'/'2026-11-12'/'scene';folder.mkdir(parents=True)
            (folder/'scene.json').write_text(json.dumps({'date':'2026-11-12','status':'complete','scene_id':'scene'}))
            with fiona.open(folder/'water_polygon.shp','w',driver='ESRI Shapefile',crs='EPSG:32650',schema={'geometry':'Point','properties':{'id':'int'}}) as dst:
                dst.write({'geometry':{'type':'Point','coordinates':(0,0)},'properties':{'id':1}})
            with self.assertRaisesRegex(ValueError,'仅允许面矢量'):
                merge.merge_results(root/'source',root/'out','EPSG:32650',save_previews=False)

if __name__=='__main__':unittest.main()
