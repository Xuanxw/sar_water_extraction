"""无标签整景推理：真正模型加载、拼接网格、面矢量、未知值及中断重跑。"""

import json
from contextlib import closing
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import fiona
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from supervised import predict
from supervised.models.unet import UNet
from supervised.evaluation.mask_postprocess import clean_mask, postprocess_water_raster


class PixelModel(torch.nn.Module):
    def forward(self, image):
        return image[:, :1]


class PredictionTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        self.source = self.root / 'new_scene.tif'
        self.profile = dict(driver='GTiff', width=47, height=35, count=1,
                            crs='EPSG:32650', transform=from_origin(400000, 4500000, 10, 10))
        rows, cols = np.indices((35, 47))
        self.image = np.where((cols > 18) & (rows > 4), 2, -2).astype('float32')
        self.image[0, 0] = 0  # 合法0不能被丢弃。
        self.image[1, 1] = np.nan
        self.image[20, 30] = np.nan
        with rasterio.open(self.source, 'w', **self.profile, dtype='float32', nodata=np.nan) as target:
            target.write(self.image, 1)
        self.config = dict(
            paths=dict(CHECKPOINT_PATH=self.root / 'model.pth', IMAGE_PATH=self.source,
                       OUTPUT_DIR=self.root / 'result', EXCLUSION_MASKS=[]),
            run=dict(DEVICE='cpu', RECURSIVE=False, RESUME=True, CONSOLE_COLOR=False),
            input=dict(BANDS=None, INPUTS_NORMALIZED=False, FILTER_ALREADY_APPLIED=False, DESCRIPTION='test'),
            inference=dict(WINDOW_SIZE=32, HALO=8, STRIDE=8, BLEND='hann',
                           BATCH_SIZE=3, NUM_WORKERS=0, USE_AMP=False, THRESHOLD=0.5),
            postprocess=dict(OPENING_RADIUS=0, CLOSING_RADIUS=0, BLOCK_SIZE=11),
            output=dict(SAVE_PROBABILITY=True, QUICKLOOK_SIZE=120),
        )
        self.checkpoint = dict(bands=[1], mean=torch.tensor([0.0]), std=torch.tensor([1.0]),
                               inputs_normalized=False, patch_representation='stored', window_size=32, window_halo=8)
        self.settings, _ = predict.prediction_settings(self.checkpoint, self.config)
        self.info = dict(name='synthetic_pixel_model', epoch=0, sha256='test')

    def run_pixel_prediction(self):
        return predict.predict_scene(PixelModel(), self.source, self.root / 'scene_output', self.settings,
            self.config, checkpoint_info=self.info, log=lambda _: None)

    def test_native_mask_probabilities_and_polygon_roundtrip(self):
        report = self.run_pixel_prediction()
        folder = self.root / 'scene_output'
        expected = np.where(np.isfinite(self.image), self.image >= 0, 255).astype('uint8')
        with rasterio.open(folder / 'water.tif') as source:
            np.testing.assert_array_equal(source.read(1), expected)
            self.assertEqual(source.transform, self.profile['transform'])
            self.assertEqual(source.crs.to_epsg(), 32650)
            self.assertEqual(source.nodata, 255)
        with rasterio.open(folder / 'probability.tif') as source:
            probability = source.read(1)
            self.assertEqual(probability[0, 0], 0.5)
            self.assertEqual(probability[1, 1], -1)
            self.assertAlmostEqual(float(probability[10, 30]), 0.880797, places=6)
        with fiona.open(folder / 'water_polygon.shp') as source:
            geometries = [feature['geometry'] for feature in source]
            self.assertTrue(all(g['type'] == 'Polygon' for g in geometries))
            restored = rasterize([(g, 1) for g in geometries], out_shape=expected.shape,
                                 transform=self.profile['transform'])
            np.testing.assert_array_equal(restored, expected == 1)
        self.assertTrue(report['complete'])
        self.assertFalse(report['accuracy']['available'])
        self.assertEqual(report['counts']['valid_pixels'], self.image.size-2)
        self.assertEqual(report['counts']['water_area_m2'], int((expected == 1).sum())*100)
        self.assertTrue((folder / 'quicklook.png').is_file())

    def test_main_loads_real_checkpoint_without_any_label_metadata(self):
        model = UNet(in_channels=1, base_channels=2)
        for parameter in model.parameters():
            parameter.data.zero_()
        model.classifier.bias.data[1] = 2
        # 不提供label_dir、manifest、val名单等字段，确保新入口与训练数据清单无关。
        checkpoint = dict(self.checkpoint, model_name='unet', in_channels=1, base_channels=2,
                          model_state_dict=model.state_dict(), epoch=1)
        torch.save(checkpoint, self.config['paths']['CHECKPOINT_PATH'])
        path = self.root / 'predict.yaml'
        serializable = {**self.config, 'paths': {k: str(v) if isinstance(v, Path) else v
                                               for k, v in self.config['paths'].items()}}
        path.write_text(yaml.safe_dump(serializable), encoding='utf-8')
        reports = predict.main(path)
        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0]['counts']['water_pixels'], self.image.size-2)
        self.assertTrue((self.root / 'result/processing.log').exists())
        with patch.object(predict, 'iter_scene_predictions', side_effect=AssertionError('不应再次推理')):
            again = predict.main(path)
        self.assertEqual(again[0]['key'], reports[0]['key'])

    def test_completed_outputs_are_not_overwritten_with_new_threshold(self):
        self.run_pixel_prediction()
        marker = self.root / 'scene_output/quality.json'
        previous = marker.read_bytes()
        self.config['inference']['THRESHOLD'] = 0.9
        with self.assertRaisesRegex(ValueError, '已有不同参数'):
            self.run_pixel_prediction()
        self.assertEqual(marker.read_bytes(), previous)

    def test_failed_scene_is_not_a_completion_marker_and_can_be_rerun(self):
        with patch.object(predict, 'write_water_polygons', side_effect=RuntimeError('模拟中断')):
            with self.assertRaisesRegex(RuntimeError, '模拟中断'):
                self.run_pixel_prediction()
        marker = self.root / 'scene_output/quality.json'
        self.assertFalse(json.loads(marker.read_text(encoding='utf-8'))['complete'])
        self.assertTrue(self.run_pixel_prediction()['complete'])

    def test_zero_water_scene_still_has_valid_empty_polygon_output(self):
        self.config['inference']['THRESHOLD'] = 0.99
        report = self.run_pixel_prediction()
        self.assertEqual(report['polygon_count'], 0)
        self.assertEqual(report['water_area_km2'], 0)
        with fiona.open(self.root / 'scene_output/water_polygon.shp') as source:
            self.assertEqual(len(source), 0)

    def test_normalization_mismatch_and_unknown_representation_do_not_fallback(self):
        self.config['input']['INPUTS_NORMALIZED'] = True
        with self.assertRaisesRegex(ValueError, 'INPUTS_NORMALIZED'):
            predict.prediction_settings(self.checkpoint, self.config)
        self.config['input']['INPUTS_NORMALIZED'] = False
        self.checkpoint['patch_representation'] = 'unimplemented'
        with self.assertRaisesRegex(ValueError, '尚未适配'):
            predict.prediction_settings(self.checkpoint, self.config)

    def test_no_label_arrays_are_allocated_for_production_blocks(self):
        from supervised.evaluation.sliding_inference import iter_scene_predictions
        blocks = iter_scene_predictions(PixelModel(), self.source, bands=(1,), window_size=32,
            halo=8, stride=8, boundary_context=False, return_labels=False)
        with closing(blocks):
            for block in blocks:
                self.assertIsNone(block['label'])
                self.assertIsNone(block['label_context'])

    def test_postprocessing_has_no_block_seams_and_unknown_exclusion_is_kept(self):
        raw = np.zeros((35, 47), dtype='uint8')
        raw[3:32, 8:43] = 1
        raw[12, 16] = 0
        raw[18:20, 18:20] = 255
        raw_path = self.root / 'raw.tif'
        with rasterio.open(raw_path, 'w', **self.profile, dtype='uint8', nodata=255) as target:
            target.write(raw, 1)
        expected = clean_mask(raw == 1, raw != 255, 1, 2)
        result = self.root / 'post.tif'
        postprocess_water_raster(raw_path, result, opening_radius=1, closing_radius=2, block_size=9)
        with rasterio.open(result) as source:
            np.testing.assert_array_equal(source.read(1), np.where(raw == 255, 255, expected))
        excluded = np.zeros_like(raw)
        excluded[10:14, 20:24] = 1
        excluded[25:28, 25:28] = 255
        exclusion_path = self.root / 'exclusion.tif'
        with rasterio.open(exclusion_path, 'w', **self.profile, dtype='uint8', nodata=255) as target:
            target.write(excluded, 1)
        postprocess_water_raster(raw_path, result, exclusion_masks=[exclusion_path], block_size=9)
        with rasterio.open(result) as source:
            actual = source.read(1)
            self.assertTrue((actual[10:14, 20:24] == 0).all())
            self.assertTrue((actual[25:28, 25:28] == 1).all())
            np.testing.assert_array_equal(actual == 255, raw == 255)

    def test_output_cannot_replace_the_input_raster(self):
        self.source.rename(self.root / 'water.tif')
        with self.assertRaisesRegex(ValueError, '覆盖输入'):
            predict.predict_scene(PixelModel(), self.root / 'water.tif', self.root, self.settings,
                self.config, checkpoint_info=self.info)


if __name__ == '__main__':
    unittest.main()
