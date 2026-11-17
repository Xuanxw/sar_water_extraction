"""过程PNG原生变化、实际SHP、图层语义及只补图续跑的合成验证。"""
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import fiona
import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box, mapping, Polygon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from thres_segmentation.org_quicklook import (
    summarize_water_stage_changes, rasterize_water_shp_preview, quicklook_is_current,
)
from thres_segmentation import org_pipeline
from utils.files import sha256
import test_org_pipeline as org_fixture


class ProcessQuicklookTests(unittest.TestCase):
    def test_native_changes_precede_aggregation_and_keep_unknown(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = dict(driver='GTiff', width=4, height=4, count=1, dtype='uint8', nodata=255,
                           crs=32650, transform=from_origin(500000, 4000004, 1, 1))
            raw = np.zeros((4, 4), np.uint8)
            raw[0, 0] = 1
            raw[1, 1] = 255
            final = np.zeros_like(raw)
            final[0, 1] = 1
            final[1, 1] = 1  # 来源未知，不计作新增。
            paths = []
            for name, values in (('raw', raw), ('corrected', final), ('final', final)):
                path = root / (name + '.tif')
                with rasterio.open(path, 'w', **profile) as dst:
                    dst.write(values, 1)
                paths.append(path)
            frozen = {path: sha256(path) for path in paths}
            for block in (1, 4):
                changes, areas, _ = summarize_water_stage_changes(paths, (1, 1), block_size=block)
                self.assertTrue(changes['total']['removed'][0, 0])
                self.assertTrue(changes['total']['added'][0, 0])
                self.assertEqual(areas['total'], dict(removed=1, added=1))
                self.assertEqual(areas['cleanup'], dict(removed=0, added=0))
            cover = np.full((4, 4), 40, np.uint8)
            cover[0, 0] = cover[1, 1] = 50
            cover[2, 2] = 255
            with rasterio.open(root / 'cover.tif', 'w', **profile) as dst:
                dst.write(cover, 1)
            _, _, rebuilt = summarize_water_stage_changes(paths, (4, 4), cover_path=root / 'cover.tif',
                                                          excluded_classes=[50], block_size=2)
            self.assertEqual(rebuilt[0, 0], 1)
            self.assertTrue(np.isnan(rebuilt[1, 1]))  # SAR未知不能重建成确认掩膜。
            self.assertTrue(np.isnan(rebuilt[2, 2]))  # 土地覆盖缺测也保持未知。
            self.assertEqual(frozen, {path: sha256(path) for path in frozen})

    def test_actual_shp_holes_and_batches_preserve_earlier_polygons(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'water_polygon.shp'
            schema = dict(geometry='Polygon', properties=dict(water='int'))
            with fiona.open(path, 'w', driver='ESRI Shapefile', schema=schema, crs='EPSG:32650') as dst:
                for number in range(260):
                    x, y = number % 20 * 3, number // 20 * 3
                    dst.write(dict(geometry=mapping(box(x + .2, y + .2, x + .8, y + .8)), properties=dict(water=1)))
                polygon = Polygon([(80, 80), (95, 80), (95, 95), (80, 95)],
                                  holes=[[(84, 84), (91, 84), (91, 91), (84, 91)]])
                dst.write(dict(geometry=mapping(polygon), properties=dict(water=1)))
            result = rasterize_water_shp_preview(path, dict(width=100, height=100, crs='EPSG:32650',
                                                           transform=from_origin(0, 100, 1, 1)))
            self.assertEqual(result[99, 0], 1)
            self.assertEqual(result[12, 87], 0)  # 实际SHP孔洞未填。
            self.assertEqual(result[18, 82], 1)

    def test_completed_scene_upgrades_only_png_and_retains_products(self):
        with TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            config = org_fixture.OrgProductionTests().make_config(root)
            config['logging']['console_color'] = False
            config['output']['process_preview_size'] = 64
            config['org']['archive_intermediates'] = True
            result = org_pipeline.run_org_scenes(config)
            folder = next(result.glob('*/*/quicklook.png')).parent
            png = folder / 'quicklook.png'
            self.assertTrue(quicklook_is_current(png, 64))
            with Image.open(png) as image:
                titles = image.info['PanelTitles']
                self.assertIn('Raw threshold candidates', titles)
                self.assertIn('Actual final SHP', titles)
                self.assertIn('Radar shadow', titles)
                self.assertIn('Total: raw to cleaned', titles)
            # 模拟旧两联图：格式标记不匹配时仅重画PNG，不能重分割或重算地形。
            Image.new('RGB', (10, 10)).save(png)
            frozen = {path: sha256(path) for directory in (result, config['paths']['CACHE_DIR'])
                      for path in directory.rglob('*') if path.suffix in ('.tif', '.shp', '.dbf', '.shx', '.prj')}
            with patch.object(org_pipeline, 'segment', side_effect=AssertionError('must not resegment')):
                org_pipeline.run_org_scenes(config)
            self.assertTrue(quicklook_is_current(png, 64))
            self.assertEqual(frozen, {path: sha256(path) for path in frozen})
            self.assertTrue(any(config['paths']['CACHE_DIR'].rglob('landcover_exclusion.tif')))


if __name__ == '__main__':
    unittest.main()
