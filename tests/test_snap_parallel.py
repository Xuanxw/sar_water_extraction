"""整景并行与串行 Lee 对照。指定环境变量 SNAP_TEST_GPT 后执行真实算子。"""
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier
from time import perf_counter
from unittest.mock import patch

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from datasets import image_process, snap_preprocess


class SnapParallelTests(unittest.TestCase):
    def test_slots(self):
        with TemporaryDirectory() as temp, patch.multiple(snap_preprocess, CACHE_DIR=Path(temp), GPT_SCENE_WORKERS=2):
            barrier = Barrier(2)

            def visit(_, row):
                with snap_preprocess.gpt_slot() as userdir:
                    barrier.wait(timeout=5)
                    return row, userdir.name

            results = list(image_process.process_scene_rows(['a', 'b'], 2, visit))
            self.assertEqual({r for r, _ in results}, {'a', 'b'})
            self.assertEqual({d for _, d in results}, {'snap_userdir', 'snap_userdir_1'})

    def test_resume_serial_stage_with_two_workers(self):
        import json
        from test_image_process import ImageProcessTests
        from test_sar_scene_dataset import SceneDatasetTests

        with TemporaryDirectory() as temp:
            root = Path(temp)
            fixture = ImageProcessTests()
            options = fixture.options(root)
            SceneDatasetTests().prepare(options['OUTPUT_DIR'])

            def interrupt(source, bands, output, work, diagnostics):
                if source.name == 'val.tif':
                    raise RuntimeError('中断')
                fixture.fake_gpt(source, bands, output, work, diagnostics)

            with patch.multiple(image_process, **options), patch.object(snap_preprocess, 'snap_runtime', return_value={'test':'1'}):
                with patch.object(image_process, 'SNAP_SCENE_WORKERS', 1), patch.object(
                        snap_preprocess.SnapSpeckleFilter, 'run', side_effect=interrupt):
                    with self.assertRaisesRegex(RuntimeError, '中断'):
                        image_process.process_stage('filtered')
                target = options['FILTERED_ROOT']
                original = target/'images/train/train.tif'
                stamp = original.stat().st_mtime_ns
                with patch.object(image_process, 'SNAP_SCENE_WORKERS', 2), patch.object(
                        snap_preprocess.SnapSpeckleFilter, 'run', side_effect=fixture.fake_gpt) as gpt:
                    image_process.process_stage('filtered')
                self.assertEqual(gpt.call_count, 1)
                self.assertEqual(original.stat().st_mtime_ns, stamp)
                status = json.loads((target/'processing_status.json').read_text(encoding='utf-8'))
                self.assertEqual((status['complete'], status['reused'], status['generated']), (True, 1, 1))

    @unittest.skipUnless(os.environ.get('SNAP_TEST_GPT'), '需要指定真实 GPT 路径')
    def test_change_yaml_writer_reuses_published_scene(self):
        import json
        import yaml
        from test_sar_scene_dataset import SceneDatasetTests

        template = Path(__file__).resolve().parents[1] / 'configs/image_process.yaml'
        self.addCleanup(image_process.configure, template)
        with TemporaryDirectory() as temp:
            root = Path(temp)
            SceneDatasetTests().prepare(root/'prepared')
            config = yaml.safe_load(template.read_text(encoding='utf-8'))
            config['paths'].update(DATA_ROOT=str(root), OUTPUT_DIR='prepared', FILTERED_ROOT='filtered',
                                   SNAP_WORK_DIR='work', SNAP_CACHE_DIR='cache', SNAP_GPT_PATH=os.environ['SNAP_TEST_GPT'])
            config['run']['MODE'] = 'filter'
            config['snap'].update(SNAP_SCENE_WORKERS=1, SNAP_THREADS=2, CLEAN_SNAP_CACHE_ON_SUCCESS=True)
            config['snap'].pop('SNAP_WRITER_FORMAT')  # 已部署的旧 YAML。
            path = root/'switch.yaml'
            path.write_text(yaml.safe_dump(config), encoding='utf-8')
            image_process.main(path)

            first = root/'filtered/images/train/train.tif'
            second = root/'filtered/images/val/val.tif'
            stamp = first.stat().st_mtime_ns
            with rasterio.open(second) as src:
                self.assertEqual(src.tags()['snap_writer_format'], 'GeoTIFF-BigTIFF')
                old_values, old_mask = src.read(), src.read_masks()
                old_grid = (src.crs, src.transform, src.shape)
            # 在临时数据集中模拟一景待生成。上一轮完整成功后，滤波缓存 TIFF 应已清理。
            second.unlink()
            self.assertFalse(list((root/'cache').glob('*/filtered.tif')))
            config['snap']['SNAP_WRITER_FORMAT'] = 'BEAM-DIMAP'
            path.write_text(yaml.safe_dump(config), encoding='utf-8')
            image_process.main(path)

            status = json.loads((root/'filtered/processing_status.json').read_text(encoding='utf-8'))
            self.assertEqual((status['reused'], status['generated']), (1, 1))
            self.assertEqual(first.stat().st_mtime_ns, stamp)
            with rasterio.open(second) as src:
                self.assertEqual(src.tags()['snap_writer_format'], 'BEAM-DIMAP')
                self.assertEqual(old_grid, (src.crs, src.transform, src.shape))
                np.testing.assert_array_equal(old_mask, src.read_masks())
                np.testing.assert_array_equal(old_values, src.read())

    @unittest.skipUnless(os.environ.get('SNAP_TEST_GPT'), '需要指定真实 GPT 路径')
    def test_serial_parallel_pixel_parity(self):
        rng = np.random.default_rng(137)
        with TemporaryDirectory() as temp:
            root = Path(temp)
            sources = [root/'a.tif', root/'b.tif']
            for source in sources:
                values = rng.lognormal(2, 1, (2, 1025, 1027)).astype('float32')
                values[:, 100:104, 100:104] = 0
                values[0, :3, :] = -9999
                values[1, 255:260, 510:517] = -9999
                with rasterio.open(source, 'w', driver='GTiff', width=1027, height=1025, count=2,
                                   dtype='float32', crs='EPSG:32650', nodata=-9999,
                                   transform=rasterio.transform.from_origin(500000, 3500000, 3, 3)) as dst:
                    dst.write(values)
            outputs = []
            for workers in (1, 2):
                with patch.multiple(snap_preprocess, GPT_PATH=Path(os.environ['SNAP_TEST_GPT']),
                                    CACHE_DIR=root/f'cache_{workers}', WORK_DIR=root/'work',
                                    GPT_SCENE_WORKERS=workers, GPT_THREADS=4, GPT_TILE_CACHE='512M'):
                    filt = snap_preprocess.SnapSpeckleFilter(dict(input_domain='intensity', window_size=3, enl=1.))
                    start = perf_counter()

                    def apply(_, source):
                        return source, filt.prepare_file(source, (2, 1))

                    outputs.append(dict(image_process.process_scene_rows(sources, workers, apply)))
                    print(f'{workers} 景并行，总耗时 {perf_counter()-start:.2f} 秒', flush=True)
            for source in sources:
                with rasterio.open(source) as src, rasterio.open(outputs[0][source]) as a, rasterio.open(outputs[1][source]) as b:
                    self.assertEqual((src.shape, src.crs, src.transform), (b.shape, b.crs, b.transform))
                    np.testing.assert_array_equal(a.read(), b.read())
                    np.testing.assert_array_equal(a.read_masks(), b.read_masks())
                    self.assertTrue(np.all(b.read_masks(1)[100:104,100:104] == 255))
                    self.assertTrue(np.all(b.read_masks(2)[255:260,510:517] == 0))
