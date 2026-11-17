"""OSM道路共享缓存、分块冲突与不改变水体的合成验证；不请求网络。"""
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import fiona
import numpy as np
from pyproj import Transformer
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from configuration import load_yaml
from thres_segmentation import osm_roads, org_pipeline
from utils.files import sha256
import test_org_pipeline as org_fixture


def settings():
    return load_yaml(ROOT / 'configs/北京密云区/corrected_amplitude_otsu_slope15_fine_geometry_roads_review.yaml')['road_review']


def network_fixture(root, polygon):
    path = root / 'roads.gpkg'
    with fiona.open(path, 'w', driver='GPKG', layer='road_buffers', crs='EPSG:32650',
                    schema=dict(geometry='Polygon', properties=dict(osm_id='str'))) as dst:
        dst.write(dict(geometry=polygon.__geo_interface__, properties=dict(osm_id='1')))
    return dict(buffers=[polygon], crs='EPSG:32650', path=path,
                metadata=dict(attribution='© OpenStreetMap contributors', license='ODbL-1.0'))


class RoadReviewTests(unittest.TestCase):
    def test_partial_reference_only_reuses_complete_stages(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'reference/2025-07-25/scene'
            source.mkdir(parents=True)
            old_cache = root / 'old_cache/scenes/scene'
            old_cache.mkdir(parents=True)
            target = root / 'new_scene'
            target.mkdir()
            reference = dict(paths=dict(CACHE_DIR=root / 'old_cache', SNAP_CACHE_DIR=root / 'snap'))
            progress = org_pipeline.SceneProgress(root / 'new_cache', 'scene', target, root / 'snap')
            record = dict(scene_id='scene', date='2025-07-25')
            log = Mock()
            reuse = lambda: org_pipeline.reuse_reference_inputs(root / 'reference', reference, record, progress, target, log)
            self.assertFalse(reuse())  # 参考景未开始，不要求用户等待整批完成。
            profile = dict(driver='GTiff', count=1, width=4, height=4, dtype='float32', crs=32650,
                           transform=from_origin(500000, 4000004, 1, 1))
            with rasterio.open(old_cache / 'crop.tif', 'w', **profile) as dst:
                dst.write(np.ones((4, 4), np.float32), 1)
            for name in ('metadata.json', 'source.xml'):
                (source / name).write_text('{}', encoding='utf-8')
            state = dict(stages=dict(crop=dict(complete=True, files=[dict(root='cache', path='crop.tif')])))
            def save():
                (old_cache / 'progress.json').write_text(json.dumps(state), encoding='utf-8')
            save()
            self.assertFalse(reuse())  # 只导入crop；未完成滤波仍由本轮处理。
            self.assertTrue(progress.completed('crop'))
            self.assertFalse(progress.completed('filter'))
            (old_cache / 'statistics.json').write_text('{}', encoding='utf-8')
            state['stages'].update(filter=dict(complete=True, files=[dict(root='cache', path='crop.tif')]),
                                   statistics=dict(complete=True, files=[dict(root='cache', path='statistics.json')]))
            save()
            self.assertFalse(reuse())  # 几何/分割未完成，不能读正在生成的掩膜。
            self.assertTrue(progress.completed('statistics'))
            for name in ('terrain_exclusion.tif', 'shadow_mask.tif', 'layover_mask.tif'):
                with rasterio.open(source / name, 'w', **profile) as dst:
                    dst.write(np.zeros((4, 4), np.float32), 1)
            (source / 'terrain_geometry.json').write_text('{}', encoding='utf-8')
            state['stages']['mask'] = dict(complete=True)
            save()
            before = {p: sha256(p) for p in source.iterdir()}
            self.assertTrue(reuse())
            self.assertTrue(progress.completed('reference_terrain'))
            self.assertEqual(before, {p: sha256(p) for p in before})

    def test_incomplete_response_not_cached_and_complete_empty_is_cached(self):
        with TemporaryDirectory() as tmp:
            options = settings()
            options.update(endpoints=['https://example.invalid/overpass'], retries=1)
            session = Mock()
            session.open.side_effect = [io.StringIO(json.dumps(dict(elements=[], remark='runtime error: Query timed out')))]
            with self.assertRaisesRegex(RuntimeError, '未完成'):
                osm_roads.download_osm_road_tile((116, 40, 116.25, 40.25), options, tmp, session, lambda text: None)
            self.assertFalse(list(Path(tmp).rglob('*.json')))
            session.open.side_effect = [io.StringIO(json.dumps(dict(elements=[])))]
            path = osm_roads.download_osm_road_tile((116, 40, 116.25, 40.25), options, tmp, session, lambda text: None)
            self.assertTrue(json.loads(path.read_text(encoding='utf-8'))['complete'])
            session.open.reset_mock()
            self.assertEqual(path, osm_roads.download_osm_road_tile(
                (116, 40, 116.25, 40.25), options, tmp, session, lambda text: None))
            session.open.assert_not_called()

    def test_shared_tiles_dedup_width_and_empty_network(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            options = settings()
            options['context_buffer_m'] = 0
            aoi = box(499900, 4500000, 500100, 4500200)  # 穿过117°固定瓦片边界。
            to_ll = Transformer.from_crs(32650, 4326, always_xy=True)
            elements = []
            for identifier, x, tags in ((1, 499930, dict(highway='primary', width='8 m', bridge='yes')),
                                        (2, 500050, dict(highway='primary', width='3;5', tunnel='yes'))):
                lon, lat = to_ll.transform([x, x], [4500010, 4500190])
                elements.append(dict(type='way', id=identifier, tags=tags,
                                     geometry=[dict(lon=xx, lat=yy) for xx, yy in zip(lon, lat)]))
            with patch.object(osm_roads, 'build_opener') as factory:
                session = factory.return_value
                payload = dict(elements=elements)
                session.open.side_effect = lambda *args, **kwargs: io.StringIO(json.dumps(payload))
                first = osm_roads.prepare_osm_road_network(aoi, 'EPSG:32650', root, options, lambda text: None)
                self.assertEqual(session.open.call_count, 2)
                self.assertEqual(len(first['buffers']), 2)
                frame = {row['osm_id']: row for row in first['attributes']}
                self.assertEqual(frame['1']['width_m'], 8)
                self.assertEqual(frame['2']['width_m'], 12)
                self.assertEqual(frame['2']['width_source'], 'class_fallback')
                self.assertEqual(frame['1']['bridge'], 'yes')
                self.assertEqual(frame['2']['tunnel'], 'yes')
                saved = first['path'].stat().st_mtime_ns
                session.open.reset_mock()
                second = osm_roads.prepare_osm_road_network(aoi, 'EPSG:32650', root, options, lambda text: None)
                self.assertEqual(second['path'].stat().st_mtime_ns, saved)
                options['extra_buffer_m'] += 1
                wider = osm_roads.prepare_osm_road_network(aoi, 'EPSG:32650', root, options, lambda text: None)
                self.assertNotEqual(first['path'], wider['path'])
                self.assertGreater(sum(g.area for g in wider['buffers']), sum(g.area for g in first['buffers']))
                session.open.assert_not_called()
                payload['elements'] = []
                empty = osm_roads.prepare_osm_road_network(aoi, 'EPSG:32650', root / 'empty', options, lambda text: None)
                self.assertEqual(len(empty['buffers']), 0)
                self.assertEqual(set(fiona.listlayers(empty['path'])), {'road_centerlines', 'road_buffers'})

    def test_block_equivalence_unknown_no_water_changes_and_empty_overlap(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = dict(driver='GTiff', count=1, width=12, height=12, dtype='uint8', nodata=255,
                           crs=32650, transform=from_origin(500000, 4000012, 1, 1))
            raw = np.zeros((12, 12), np.uint8)
            raw[3:10, 3:10] = 1
            raw[0] = 255
            corrected = raw.copy()
            corrected[:, 6] = 0
            corrected[0] = 255
            final = corrected.copy()
            final[4:8, 6] = 1
            source = []
            for name, data in (('raw', raw), ('corrected', corrected), ('final', final), ('sar', np.full(raw.shape, 20, np.uint8))):
                path = root / (name + '.tif')
                with rasterio.open(path, 'w', **profile) as dst:
                    dst.write(data, 1)
                source.append(path)
            before = {path: sha256(path) for path in source}
            network = network_fixture(root, box(500004, 4000000, 500007, 4000012))
            expected_road = rasterize([(network['buffers'][0], 1)], out_shape=raw.shape,
                                      transform=profile['transform']) == 1
            outputs = []
            for block in (3, 16):
                folder = root / str(block)
                osm_roads.export_road_water_conflicts(source[3], *source[:3], network, folder,
                                                     block_size=block, preview_size=40, log=lambda text: None)
                with rasterio.open(folder / 'road_conflict.tif') as dst:
                    result = dst.read()
                    self.assertEqual(dst.transform, profile['transform'])
                    self.assertEqual(dst.descriptions, ('raw_threshold', 'after_auxiliary', 'final_polygon'))
                outputs.append(result)
                report = json.loads((folder / 'road_review.json').read_text(encoding='utf-8'))
                self.assertFalse(report['water_modified'])
                for band, (name, mask) in enumerate(zip(report['bands'].values(), (raw, corrected, final))):
                    self.assertTrue(np.all(result[band, 0] == 255))
                    valid = raw != 255
                    expected = expected_road & valid & (mask == 1)
                    np.testing.assert_array_equal(result[band, valid], expected[valid])
                    self.assertEqual(report['conflict_area_m2'][name], int(expected.sum()))
                for name in ('raw_threshold', 'final_polygon'):
                    with fiona.open(folder / 'road_conflicts.gpkg', layer=name + '_conflicts') as layer:
                        self.assertAlmostEqual(sum(f['properties']['area_m2'] for f in layer), report['conflict_area_m2'][name])
            np.testing.assert_array_equal(*outputs)
            self.assertEqual(before, {path: sha256(path) for path in source})
            network['buffers'] = []
            osm_roads.export_road_water_conflicts(source[3], *source[:3], network, root / 'no_roads',
                                                 block_size=5, preview_size=40, log=lambda text: None)
            with rasterio.open(root / 'no_roads/road_conflict.tif') as src:
                self.assertFalse(np.any(src.read() == 1))

    def test_org_pipeline_download_once_read_only_quality_and_resume(self):
        with TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            config = org_fixture.OrgProductionTests().make_config(root)
            config['logging']['console_color'] = False
            baseline_root = org_pipeline.run_org_scenes(config)
            baseline = json.loads((baseline_root / 'results.json').read_text(encoding='utf-8'))
            baseline_files = [path for row in baseline for path in Path(row['water_shp']).parent.rglob('*') if path.is_file()]
            frozen = {path: sha256(path) for path in baseline_files}
            trial = copy.deepcopy(config)
            trial['paths'].update(OUTPUT_DIR=root / 'trial', OUTPUT_ROOT=root / 'trial', CACHE_DIR=root / 'trial_cache',
                                  OSM_CACHE_DIR=root / 'roads_cache')
            trial['org']['archive_intermediates'] = True
            trial['road_review'] = settings()
            trial['road_review'].update(block_size=13, preview_size=64)
            network = network_fixture(root, box(500030, 4000005, 500035, 4000055))
            with patch.object(osm_roads, 'prepare_osm_road_network', return_value=network) as prepare:
                result_root = org_pipeline.run_org_scenes(trial)
                self.assertEqual(prepare.call_count, 1)  # 多景只准备一次共享路网。
            rows = json.loads((result_root / 'results.json').read_text(encoding='utf-8'))
            for old, new in zip(baseline, rows):
                with rasterio.open(Path(old['water_polygon_shp']).with_suffix('.tif')) as a, rasterio.open(Path(new['water_polygon_shp']).with_suffix('.tif')) as b:
                    np.testing.assert_array_equal(a.read(), b.read())
                folder = Path(new['water_polygon_shp']).parent
                quality = json.loads((folder / 'quality.json').read_text(encoding='utf-8'))
                self.assertFalse(quality['road_review']['water_modified'])
                self.assertTrue((folder / 'road_review/road_conflicts.gpkg').is_file())
            # 完整核验阶段续跑不重复写文件，且归档后的原始候选能正常恢复。
            before = {path: path.stat().st_mtime_ns for row in rows
                      for path in (Path(row['water_polygon_shp']).parent / 'road_review').glob('*')}
            with patch.object(osm_roads, 'prepare_osm_road_network', return_value=network), patch.object(
                osm_roads, 'export_road_water_conflicts', side_effect=AssertionError('must resume road review')
            ), patch.object(org_pipeline, 'segment', side_effect=AssertionError('must resume masks')):
                org_pipeline.run_org_scenes(trial)
            self.assertEqual(before, {path: path.stat().st_mtime_ns for path in before})
            self.assertEqual(frozen, {path: sha256(path) for path in frozen})


if __name__ == '__main__':
    unittest.main()
