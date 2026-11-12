"""日期合并与可选ROI快视图：不改变水体面积、孔洞或源文件。"""
import hashlib
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import fiona
from PIL import Image
from pyproj import Transformer
from shapely import transform
from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from merge_water_shp import merge_results


def write_polygon(path, geom, crs='EPSG:32650'):
    path.parent.mkdir(parents=True,exist_ok=True)
    with fiona.open(path,'w',driver='ESRI Shapefile',crs=crs,
                    schema={'geometry':'Polygon','properties':{'id':'int'}}) as dst:
        dst.write({'geometry':mapping(geom),'properties':{'id':1}})


class MergePreviewTests(unittest.TestCase):
    def test_roi_optional_reprojected_and_does_not_clip(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);source=root/'source'
            donut=box(500000,4000000,501000,4001000).difference(box(500300,4000300,500700,4000700))
            extra=box(500800,4000000,501200,4000200)
            for i,geom in enumerate((donut,extra)):
                folder=source/'2026-11-12'/str(i)
                write_polygon(folder/'water_polygon.shp',geom)
                (folder/'scene.json').write_text(json.dumps({'status':'complete','date':'2026-11-12','scene_id':str(i)}))
            roi=box(500000,4000000,500500,4001000)
            roi_path=root/'roi_input.shp'
            write_polygon(roi_path,transform(roi,Transformer.from_crs(32650,4326,always_xy=True).transform,interleaved=False),'EPSG:4326')
            hashes={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file()}
            for use_roi in (True,False):
                output=merge_results(source,root/str(use_roi),'EPSG:32650',roi_path if use_roi else None,preview_size=300)
                with fiona.open(output/'2026-11-12_water_polygon.shp') as src:
                    water=unary_union([shape(f['geometry']) for f in src])
                self.assertAlmostEqual(water.area,unary_union([donut,extra]).area)
                self.assertFalse(water.contains(box(500400,4000400,500500,4000500)))
                with Image.open(output/'2026-11-12_water_polygon.png') as image:
                    image.verify()
                self.assertEqual((output/'roi.shp').exists(),use_roi)
                self.assertEqual((output/'roi.png').exists(),use_roi)
                if use_roi:
                    with fiona.open(output/'roi.shp') as src:
                        restored=unary_union([shape(f['geometry']) for f in src])
                    self.assertLess(restored.symmetric_difference(roi).area,0.01)
            self.assertTrue(all(hashlib.sha256(p.read_bytes()).hexdigest()==h for p,h in hashes.items()))


if __name__=='__main__':unittest.main()
