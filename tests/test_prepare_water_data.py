"""新传感器准备流程：真实小GeoTIFF/SHP/ZIP，验证未知标签及训练读取契约。"""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from zipfile import ZipFile, ZIP_DEFLATED
import csv
import json
import sys
import unittest

import fiona
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import transform_geom
from shapely.geometry import box, mapping, Polygon, MultiPolygon

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"),
                str(Path(__file__).resolve().parents[1] / "scripts")]
from datasets import image_process as prepare
from datasets.dfc_sample.s1_dataset import WaterDataset

# 测试显式加载样例配置，正式脚本仅使用命令行提供的文件。
prepare.configure(Path(__file__).resolve().parents[1] / 'configs/image_process.yaml')


class PrepareWaterDataTests(unittest.TestCase):
    def setUp(self):
        # 合成清单不分生产批次，避免当前YAML选择batch 2时把测试样本全部过滤。
        batch = patch.object(prepare, 'PREPARE_BATCH', None)
        batch.start()
        self.addCleanup(batch.stop)

    def sources(self, root, name='SAMPLE_A', nodata=-9999, reproject=False):
        images, labels = root/'source_images', root/'source_labels'
        images.mkdir(exist_ok=True)
        labels.mkdir(exist_ok=True)
        transform = from_origin(500000,3400000,3,3)
        data = np.arange(63,dtype='float32').reshape(7,9)
        data[2,1] = -9999
        path = images/(name+'.tiff')
        with rasterio.open(path,'w',driver='GTiff',width=9,height=7,count=1,dtype='float32',
                           crs='EPSG:32650',transform=transform,nodata=nodata) as dst:
            dst.write(data,1)
        def rectangle(c0,r0,c1,r1):
            x0,y0 = transform*(c0,r0)
            x1,y1 = transform*(c1,r1)
            return box(x0,y1,x1,y0)
        shapes=[(rectangle(0,0,4,6),0),(rectangle(3,1,7,7),1)]
        label=labels/(name+'.shp')
        with fiona.open(label,'w',driver='ESRI Shapefile',crs='EPSG:4326' if reproject else 'EPSG:32650',
                        schema={'geometry':'Polygon','properties':{'kind':'int'}}) as dst:
            for geom,kind in shapes:
                geometry=mapping(geom)
                if reproject:
                    geometry=transform_geom('EPSG:32650','EPSG:4326',geometry)
                dst.write(dict(geometry=geometry,properties={'kind':kind}))
        return path,label,data,transform,shapes

    def output(self, root):
        output=root/'prepared'
        for kind in ('images','labels'):
            for split in ('train','val'):
                (output/kind/split).mkdir(parents=True,exist_ok=True)
        return output

    def row(self, name='SAMPLE_A', split='train'):
        return dict(scene_id=name,split=split,group_id='G_a',scene='合成场景')

    def test_full_scene_zip_grid_values_unknown_and_reader(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source, label, data, transform, shapes = self.sources(root, reproject=True)
            archive = root/'SAMPLE_A.zip'
            with ZipFile(archive, 'w', ZIP_DEFLATED) as z:
                z.write(source, 'nested/SAMPLE_A.TIFF')
            output = self.output(root)
            rows = prepare.prepare_scene(self.row(), archive, label, output, root/'cache',
                                         bands=(1,), read_window_size=4, field='kind', class_map={0:0,1:1})
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row['name'], 'SAMPLE_A.tif')
            self.assertEqual((row['source_height'], row['source_width']), data.shape)
            self.assertEqual(list((root/'cache').iterdir()), [])
            expected = np.full(data.shape, 255, dtype='uint8')
            bg = rasterize([(shapes[0][0],1)], out_shape=data.shape, transform=transform).astype(bool)
            water = rasterize([(shapes[1][0],1)], out_shape=data.shape, transform=transform).astype(bool)
            valid = data != -9999
            expected[bg & ~water & valid] = 0
            expected[water & ~bg & valid] = 1
            with rasterio.open(output/'labels/train'/row['name']) as target:
                self.assertEqual(target.shape, data.shape)  # 不补齐到窗口整数倍。
                self.assertEqual(target.transform, transform)
                self.assertEqual(target.crs, rasterio.crs.CRS.from_epsg(32650))
                self.assertEqual(target.nodata, 255)
                np.testing.assert_array_equal(target.read(1), expected)
            with rasterio.open(output/'images/train'/row['name']) as target:
                self.assertEqual(target.shape, data.shape)
                self.assertEqual(target.transform, transform)
                result = target.read(1)
                np.testing.assert_array_equal(result[valid], data[valid])
                self.assertTrue(np.isnan(result[~valid]).all())
                self.assertEqual(result[0,0], 0)  # 保留有效零值。
                self.assertEqual(result[-1,-1], data[-1,-1])  # 无标签区域仍保留SAR。
            self.assertEqual(row['ignored_pixels'], int((expected==255).sum()))
            self.assertEqual(row['water_pixels'], int((expected==1).sum()))
            self.assertEqual(row['background_pixels'], int((expected==0).sum()))
            self.assertEqual(row['valid_pixels'], int(valid.sum()))
            self.assertEqual(len(list((output/'images/train').glob('*.tif'))), 1)
            self.assertFalse(list(output.rglob('*.partial')))
            dataset = WaterDataset(output/'images/train', output/'labels/train', (1,))
            sample = dataset[0]
            np.testing.assert_array_equal(sample['label'].numpy(), np.where(expected==255,-1,expected.astype('int64')))
            self.assertEqual(tuple(sample['image'].shape), (1,7,9))

    def test_undeclared_validity_requires_confirmation(self):
        with TemporaryDirectory() as temp:
            root=Path(temp)
            source,label,*_=self.sources(root,nodata=None)
            with self.assertRaisesRegex(ValueError,'没有声明无效域'):
                prepare.prepare_scene(self.row(),source,label,self.output(root),root/'cache',bands=(1,),
                                      read_window_size=4,field='kind',class_map={0:0,1:1})
            rows=prepare.prepare_scene(self.row(),source,label,self.output(root),root/'cache',bands=(1,),
                                       read_window_size=4,field='kind',class_map={0:0,1:1},input_nodata=-9999)
            self.assertTrue(rows)

    def test_mapping_and_invalid_geometry_are_not_guessed_or_repaired(self):
        with TemporaryDirectory() as temp:
            root=Path(temp)
            _,label,*_=self.sources(root)
            with self.assertRaisesRegex(ValueError,'CLASS_MAP'):
                prepare.read_polygons(label,'EPSG:32650','kind',{})
            with self.assertRaisesRegex(ValueError,'没有字段'):
                prepare.read_polygons(label,'EPSG:32650','class',{0:0,1:1})
            bad=root/'bad.shp'
            with fiona.open(bad,'w',driver='ESRI Shapefile',crs='EPSG:32650',
                            schema={'geometry':'Polygon','properties':{'kind':'int'}}) as dst:
                dst.write(dict(geometry=mapping(Polygon([(0,0),(2,2),(0,2),(2,0),(0,0)])),properties={'kind':1}))
            with patch.object(prepare,'ACCEPT_REVIEWED_GEOMETRY',False), self.assertRaisesRegex(ValueError,'几何待修复'):
                prepare.read_polygons(bad,'EPSG:32650','kind',{1:1})

    def test_degenerate_holes_preserve_water_polygon_and_real_holes(self):
        """复现两点内环：保留外环、真实孔洞和多面部件，源文件不变。"""
        outer=[(0,0),(10,0),(10,10),(0,10),(0,0)]
        hole=[(2,2),(4,2),(4,4),(2,4),(2,2)]
        other=[(20,0),(30,0),(30,10),(20,10),(20,0)]
        with TemporaryDirectory() as temp:
            for kind in ('Polygon','MultiPolygon'):
                with self.subTest(kind=kind):
                    rings=[outer,hole,[(5,5),(5,5)]]
                    coordinates=rings if kind=='Polygon' else [rings,[other]]
                    path=Path(temp)/'short_hole.geojson'
                    path.write_text(json.dumps(dict(type='FeatureCollection',features=[dict(
                        type='Feature',properties={'kind':1},geometry=dict(type=kind,coordinates=coordinates)
                    )])),encoding='utf-8')
                    before=path.read_bytes(); messages=StringIO()
                    with redirect_stdout(messages),patch.object(prepare,'ACCEPT_REVIEWED_GEOMETRY',False):
                        geometries,classes=prepare.read_polygons(path,'EPSG:4326','kind',{1:1})
                    expected=Polygon(outer,[hole])
                    if kind=='MultiPolygon':
                        expected=MultiPolygon([expected,Polygon(other)])
                    self.assertEqual(len(geometries),1)
                    self.assertTrue(geometries[0].is_valid)
                    self.assertTrue(geometries[0].equals(expected))
                    np.testing.assert_array_equal(classes,[1])
                    self.assertIn('移除1个退化内环',messages.getvalue())
                    self.assertEqual(path.read_bytes(),before)

    def test_malformed_outer_ring_still_reports_feature(self):
        """外环不能被静默删除；出错时明确打印要素编号。"""
        with TemporaryDirectory() as temp:
            path=Path(temp)/'short_outer.geojson'
            path.write_text(json.dumps(dict(type='FeatureCollection',features=[dict(
                type='Feature',properties={'kind':1},geometry=dict(type='Polygon',coordinates=[[(0,0),(0,0)]])
            )])),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'标签要素0无法构造面'):
                prepare.read_polygons(path,'EPSG:4326','kind',{1:1})

    def test_main_preserves_split_and_ignores_unapproved_scenes(self):
        with TemporaryDirectory() as temp:
            root=Path(temp)
            self.sources(root,'SAMPLE_A')
            self.sources(root,'SAMPLE_B')
            split=root/'split.csv'
            entries=[['SAMPLE_A','train','地点；G_a','已确认'],['SAMPLE_B','val','地点；G_b','是'],
                     ['NO_IMAGE','train','地点；G_c','待确认']]
            with split.open('w',encoding='utf-8-sig',newline='') as f:
                writer=csv.writer(f);writer.writerow(['影像编号','数据划分','场景','是否采用']);writer.writerows(entries)
            config=dict(MODE='prepare',SPLIT_CSV=split,IMAGE_ROOT=root/'source_images',LABEL_ROOT=root/'source_labels',
                        OUTPUT_DIR=root/'prepared',CACHE_DIR=root/'cache',READ_WINDOW_SIZE=4,CLASS_FIELD='kind',CLASS_MAP={0:0,1:1})
            with patch.multiple(prepare,**config),redirect_stdout(StringIO()):
                prepare.main()
            manifest=list(csv.DictReader((root/'prepared/split_manifest.csv').open(encoding='utf-8-sig')))
            self.assertEqual({r['scene_id'] for r in manifest},{'SAMPLE_A','SAMPLE_B'})
            self.assertEqual({(r['split'],r['group_id']) for r in manifest},{('train','G_a'),('val','G_b')})
            self.assertTrue(all(r['active']=='1' for r in manifest))
            self.assertFalse((root/'prepared/split_manifest.csv.partial').exists())
            entries[1][2]='地点；G_a'
            with split.open('w',encoding='utf-8-sig',newline='') as f:
                writer=csv.writer(f);writer.writerow(['影像编号','数据划分','场景','是否采用']);writer.writerows(entries)
            config['OUTPUT_DIR']=root/'rejected'
            with patch.multiple(prepare,**config),self.assertRaisesRegex(ValueError,'地理组跨train/val'):
                prepare.main()
            self.assertFalse(config['OUTPUT_DIR'].exists())

    def test_no_approved_scenes_and_failed_partial_are_not_published(self):
        with TemporaryDirectory() as temp:
            root=Path(temp)
            self.sources(root)
            split=root/'split.csv'
            def write_status(status):
                with split.open('w',encoding='utf-8-sig',newline='') as f:
                    writer=csv.writer(f);writer.writerow(['影像编号','数据划分','场景','是否采用'])
                    writer.writerow(['SAMPLE_A','train','地点；G_a',status])
            config=dict(MODE='prepare',SPLIT_CSV=split,IMAGE_ROOT=root/'source_images',LABEL_ROOT=root/'source_labels',
                        OUTPUT_DIR=root/'prepared',CACHE_DIR=root/'cache',READ_WINDOW_SIZE=4,CLASS_FIELD='kind',CLASS_MAP={0:0,1:1})
            write_status('待确认')
            with patch.multiple(prepare,**config),self.assertRaisesRegex(ValueError,'没有可处理场景'):
                prepare.main()
            self.assertFalse(config['OUTPUT_DIR'].exists())
            write_status('是')
            with patch.multiple(prepare,**config),self.assertRaisesRegex(ValueError,'尚未发布训练清单'):
                prepare.main()  # 只有train、没有val，不能冒充完整训练集。
            self.assertFalse((config['OUTPUT_DIR']/'split_manifest.csv').exists())
            self.assertTrue((config['OUTPUT_DIR']/'split_manifest.csv.partial').exists())


    def test_resume_skips_completed_scene_and_retries_missing_field(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            self.sources(root, 'SAMPLE_A')
            self.sources(root, 'SAMPLE_B')
            split = root/'split.csv'
            with split.open('w', encoding='utf-8-sig', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['影像编号', '数据划分', '场景', '是否采用'])
                writer.writerows([['SAMPLE_A', 'train', 'G_a', '已确认'], ['SAMPLE_B', 'val', 'G_b', '已确认']])
            config = dict(MODE='prepare', SPLIT_CSV=split, IMAGE_ROOT=root/'source_images',
                          LABEL_ROOT=root/'source_labels', OUTPUT_DIR=root/'prepared', CACHE_DIR=root/'cache',
                          READ_WINDOW_SIZE=4, CLASS_FIELD='kind', CLASS_MAP={0:0, 1:1},
                          SCENE_CLASS_RULES={'SAMPLE_B': {'field':'missing', 'map':{0:0,1:1}}})
            with patch.multiple(prepare, **config), redirect_stdout(StringIO()):
                prepare.main()
            partial = config['OUTPUT_DIR']/'split_manifest.csv.partial'
            with partial.open(encoding='utf-8-sig') as stream:
                before = list(csv.DictReader(stream))
            self.assertEqual({r['scene_id'] for r in before}, {'SAMPLE_A'})
            self.assertTrue((config['OUTPUT_DIR']/'skipped_scenes.csv').exists())
            # 字段存在但全是其他数值，也应跳过且不进入影像切块。
            config['SCENE_CLASS_RULES'] = {'SAMPLE_B': {'field':'kind', 'map':{99:1}}}
            with patch.multiple(prepare, **config), redirect_stdout(StringIO()), \
                 patch.object(prepare, 'prepare_scene', side_effect=AssertionError('不得解压切块')):
                prepare.main()
            self.assertIn('没有可映射类别', (config['OUTPUT_DIR']/'skipped_scenes.csv').read_text(encoding='utf-8-sig'))
            config['SCENE_CLASS_RULES'] = {}
            with patch.multiple(prepare, **config), redirect_stdout(StringIO()), \
                 patch.object(prepare, 'prepare_scene', wraps=prepare.prepare_scene) as run:
                prepare.main()
                self.assertEqual(run.call_count, 1)
                self.assertEqual(run.call_args.args[0]['scene_id'], 'SAMPLE_B')
            with (config['OUTPUT_DIR']/'split_manifest.csv').open(encoding='utf-8-sig') as stream:
                after = list(csv.DictReader(stream))
            self.assertEqual([r for r in after if r['scene_id']=='SAMPLE_A'], before)
            self.assertEqual(len({r['name'] for r in after}), len(after))

    def test_resume_excludes_unfinished_scene_but_protects_completed(self):
        """合成三景：A完成、B失败、C未开始；排除B后仅生成C，A的像元和记录不变。"""
        with TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ('SAMPLE_A', 'SAMPLE_B', 'SAMPLE_C'):
                self.sources(root, name)
            split = root/'split.csv'
            fields = ['影像编号', '数据划分', '场景', '是否采用']
            entries = [['SAMPLE_A', 'train', 'G_a', '已确认'],
                       ['SAMPLE_B', 'val', 'G_b', '已确认'],
                       ['SAMPLE_C', 'val', 'G_c', '已确认']]

            def write_rows(values):
                with split.open('w', encoding='utf-8-sig', newline='') as stream:
                    writer = csv.writer(stream)
                    writer.writerow(fields)
                    writer.writerows(values)

            write_rows(entries)
            config = dict(SPLIT_CSV=split, IMAGE_ROOT=root/'source_images',
                          LABEL_ROOT=root/'source_labels', OUTPUT_DIR=root/'prepared', CACHE_DIR=root/'cache',
                          READ_WINDOW_SIZE=4, CLASS_FIELD='kind', CLASS_MAP={0:0, 1:1},
                          SCENE_CLASS_RULES={}, MAX_SCENES=None)
            original = prepare.prepare_scene

            def fail_second(row, *args, **kwargs):
                if row['scene_id'] == 'SAMPLE_B':
                    raise ValueError('标签缺少坐标系')
                return original(row, *args, **kwargs)

            with patch.multiple(prepare, **config), redirect_stdout(StringIO()):
                with patch.object(prepare, 'prepare_scene', side_effect=fail_second):
                    with self.assertRaisesRegex(ValueError, '标签缺少坐标系'):
                        prepare.prepare_dataset(mode='prepare', reuse_complete=True)
                output = config['OUTPUT_DIR']
                partial = output/'split_manifest.csv.partial'
                before = partial.read_bytes()
                preserved = {p: (p.read_bytes(), p.stat().st_mtime_ns)
                             for kind in ('images', 'labels') for p in (output/kind/'train').glob('*.tif')}
                # 已完成景不得退出、改train/val或换地理组；未完成景不能顺便换分组。
                for row, column, value in ((0, 3, '否'), (0, 1, 'val'), (0, 2, 'G_d'), (1, 2, 'G_e')):
                    changed = [r.copy() for r in entries]
                    changed[row][column] = value
                    write_rows(changed)
                    with patch.object(prepare, 'prepare_scene') as build:
                        with self.assertRaisesRegex(ValueError, '已改变'):
                            prepare.prepare_dataset(mode='prepare', reuse_complete=True)
                        build.assert_not_called()
                    self.assertEqual(partial.read_bytes(), before)
                # 仅把失败景改为否，不手动修改输出目录的旧快照。
                entries[1][3] = '否'
                write_rows(entries)
                with patch.object(prepare, 'prepare_scene', wraps=original) as build:
                    prepare.prepare_dataset(mode='prepare', reuse_complete=True)
                    self.assertEqual(build.call_count, 1)
                    self.assertEqual(build.call_args.args[0]['scene_id'], 'SAMPLE_C')
                with (output/'split_manifest.csv').open(encoding='utf-8-sig', newline='') as stream:
                    result = list(csv.DictReader(stream))
                self.assertEqual([r['scene_id'] for r in result], ['SAMPLE_A', 'SAMPLE_C'])
                self.assertEqual((output/'scene_split.csv').read_bytes(), split.read_bytes())
                for path, saved in preserved.items():
                    self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), saved)
                with patch.object(prepare, 'prepare_scene') as build:
                    prepare.prepare_dataset(mode='prepare', reuse_complete=True)
                    build.assert_not_called()

    def test_split_only_does_not_read_images_or_geometry(self):
        with TemporaryDirectory() as temp:
            split = Path(temp) / 'split.csv'
            with split.open('w', encoding='utf-8-sig', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['影像编号', '数据划分', '场景', '是否采用'])
                writer.writerows([['A', 'train', 'G_a', '已确认'], ['B', 'val', 'G_b', '已确认']])
            with patch.multiple(prepare, MODE='split', SPLIT_CSV=split), \
                 patch.object(prepare, 'index_sources', side_effect=AssertionError('不得读取影像')), \
                 redirect_stdout(StringIO()) as output:
                prepare.main()
            self.assertIn("'train': 1", output.getvalue())
            self.assertIn("'val': 1", output.getvalue())


if __name__=='__main__':
    unittest.main()
