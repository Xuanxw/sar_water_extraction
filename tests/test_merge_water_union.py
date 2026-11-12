"""可选并集：保留孔洞、消除重叠、缓存续跑，与追加成果共存。"""
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import fiona
from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import merge_water_shp as merge

class MergeUnionTests(unittest.TestCase):
    def test_optional_union_and_cached_resume(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);source=root/'source';output=root/'merged'
            geoms=[box(i,0,i+2,2) for i in range(6)]
            geoms.append(box(20,0,30,10).difference(box(22,2,28,8)))
            folder=source/'2026-11-12'/'scene';folder.mkdir(parents=True)
            with fiona.open(folder/'water.shp','w',driver='ESRI Shapefile',crs='EPSG:32650',schema={'geometry':'Polygon','properties':{'id':'int'}}) as dst:
                dst.writerecords([{'geometry':mapping(g),'properties':{'id':i}} for i,g in enumerate(geoms)])
            (folder/'scene.json').write_text(json.dumps({'date':'2026-11-12','status':'complete','scene_id':'scene'}))
            with patch.object(merge,'MERGE_BATCH_SIZE',2),patch.object(merge,'CONSOLE_COLOR',False):
                merge.merge_results(source,output,'EPSG:32650',save_previews=False)
                appended=output/'2026-11-12_water.shp'
                stamp=appended.stat().st_mtime_ns
                original=merge.cached_union
                def interrupt(gs,path,log,label):
                    if path.name.startswith('level_'):raise RuntimeError('test union interruption')
                    return original(gs,path,log,label)
                with patch.object(merge,'cached_union',side_effect=interrupt):
                    with self.assertRaisesRegex(RuntimeError,'test union interruption'):
                        merge.merge_results(source,output,'EPSG:32650',save_previews=False,dissolve=True)
                leaves={p:p.stat().st_mtime_ns for p in (output/'.merge_cache').rglob('scene_*.wkb')}
                self.assertEqual(len(leaves),4)
                merge.merge_results(source,output,'EPSG:32650',save_previews=False,dissolve=True)
                self.assertTrue(all(p.stat().st_mtime_ns==t for p,t in leaves.items()))
                self.assertEqual(appended.stat().st_mtime_ns,stamp)
                with fiona.open(output/'2026-11-12_water_union.shp') as src:
                    parts=[shape(f['geometry']) for f in src]
                expected=unary_union(geoms)
                self.assertLess(unary_union(parts).symmetric_difference(expected).area,1e-9)
                self.assertAlmostEqual(sum(p.area for p in parts),expected.area)
                self.assertLess(len(parts),len(geoms))
                with patch.object(merge,'merge_date',side_effect=AssertionError('must reuse completed date')):
                    merge.merge_results(source,output,'EPSG:32650',save_previews=False,dissolve=True)
                summary=json.loads((output/'2026-11-12_union_sources.json').read_text(encoding='utf-8'))
                self.assertEqual(summary['method'],'geometric_union')
                self.assertEqual(summary['scene_ids'],['scene'])

if __name__=='__main__':unittest.main()
