"""分层合并的几何一致性及中断续跑验证。"""
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


class MergeTreeTests(unittest.TestCase):
    def test_tree_matches_direct_union_and_resumes_nodes(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);source=root/'source';output=root/'merged'
            geoms=[box(i,0,i+2,2) for i in range(12)]
            geoms.append(box(20,0,30,10).difference(box(22,2,28,8)))
            folder=source/'2026-11-12'/'scene';folder.mkdir(parents=True)
            with fiona.open(folder/'water.shp','w',driver='ESRI Shapefile',crs='EPSG:32650',
                            schema={'geometry':'Polygon','properties':{'id':'int'}}) as dst:
                dst.writerecords([{'geometry':mapping(g),'properties':{'id':i}} for i,g in enumerate(geoms)])
            (folder/'scene.json').write_text(json.dumps({'date':'2026-11-12','status':'complete','scene_id':'scene'}))
            original=merge.cached_union
            def interrupt_parent(gs,path,log,label):
                if path.name.startswith('level_'):raise RuntimeError('test interruption')
                return original(gs,path,log,label)
            with patch.object(merge,'MERGE_BATCH_SIZE',2),patch.object(merge,'CONSOLE_COLOR',False):
                with patch.object(merge,'cached_union',side_effect=interrupt_parent):
                    with self.assertRaisesRegex(RuntimeError,'test interruption'):
                        merge.merge_results(source,output,'EPSG:32650',save_previews=False)
                leaves={p:p.stat().st_mtime_ns for p in (output/'.merge_cache').rglob('scene_*.wkb')}
                self.assertEqual(len(leaves),7)
                merge.merge_results(source,output,'EPSG:32650',save_previews=False)
                self.assertTrue(all(p.stat().st_mtime_ns==stamp for p,stamp in leaves.items()))
                with fiona.open(output/'2026-11-12_water.shp') as src:
                    actual=unary_union([shape(f['geometry']) for f in src])
                self.assertLess(actual.symmetric_difference(unary_union(geoms)).area,1e-9)
                with patch.object(merge,'merge_date',side_effect=AssertionError('completed date must be reused')):
                    merge.merge_results(source,output,'EPSG:32650',save_previews=False)
                log=(output/'processing.log').read_text(encoding='utf-8')
                for text in ('归并第','SHP已写出','复用已完成','test interruption'):
                    self.assertIn(text,log)


if __name__=='__main__':unittest.main()
