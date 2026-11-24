"""共享训练分位数在配置、检查点和中断续跑间保持一致。"""
import csv
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from zipfile import ZipFile
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.transform import from_origin
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from datasets.dfc_sample.s1_dataset import WaterDataset
from datasets.relative_normalization import LOG_REPRESENTATION
from supervised import train


REPOSITORY = Path(__file__).resolve().parents[1]
REPRESENTATION = 'vv_train_p02_p98_dem_m'


class TinyWaterModel(torch.nn.Module):
    def __init__(self, in_channels=2, num_classes=1):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 4, 1), torch.nn.BatchNorm2d(4),
            torch.nn.ReLU(), torch.nn.Dropout2d(0.1), torch.nn.Conv2d(4, num_classes, 1))

    def forward(self, image):
        return self.layers(image)


class SharedNormalizationTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        torch.set_num_threads(2)
        train.configure(REPOSITORY / 'configs/train.yaml')
        self.addCleanup(train.configure, REPOSITORY / 'configs/train.yaml')
        for mocked in (patch.object(torch.cuda, 'is_available', return_value=False),
                       patch.object(train, 'cprint')):
            mocked.start()
            self.addCleanup(mocked.stop)
        rows = []
        for number in range(6):
            split = 'train' if number < 4 else 'val'
            image_path = self.root / 'images' / split / f'{number}.tif'
            label_path = self.root / 'labels' / split / f'{number}.tif'
            image_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.parent.mkdir(parents=True, exist_ok=True)
            vv = np.arange(64, dtype=np.float32).reshape(8, 8) + 10 * number
            if split == 'val':
                vv += 100000  # 极端验证值不得进入分位数拟合。
            dem = np.arange(64, dtype=np.float32).reshape(8, 8) + 30
            profile = dict(driver='GTiff', height=8, width=8, count=2,
                           dtype='float32', transform=from_origin(100, 200, 10, 10), crs='EPSG:32650')
            with rasterio.open(image_path, 'w', **profile) as dst:
                dst.write(np.stack((vv, dem)))
            profile.update(count=1, dtype='uint8')
            with rasterio.open(label_path, 'w', **profile) as dst:
                dst.write((np.indices((8, 8))[1] < 4).astype('uint8'), 1)
            rows.append(dict(name=f'{number}.tif', split=split, active=1, group_id=f'g{number}'))
        with (self.root / 'split_manifest.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def make(self, checkpoint=None, representation=REPRESENTATION):
        return train.WaterSegmentationTrainer(
            image_dir=self.root / 'images', label_dir=self.root / 'labels', output_dir=self.root / 'outputs',
            model_class=TinyWaterModel, model_kwargs=dict(in_channels=2, num_classes=1),
            model_info=dict(model_name='tiny_water', input_channels=('VV', 'DEM'),
                            dem_band=2, dem_source='Copernicus GLO-30 DSM', dem_unit='m'),
            bands=(1, 2), batch_size=2, epochs=2, learning_rate=1e-3, seed=42,
            val_ratio=0.2, loss_name='bce', optimizer_name='adamw', weight_decay=.01,
            cosine_schedule=True, train_augmentation=True, head_loss_weights=(1.0,),
            output_dir_is_final=True, patch_representation=representation,
            resume_checkpoint=checkpoint)

    def test_four_fresh80_configs_and_factory_share_the_input_protocol(self):
        archive = ZipFile(REPOSITORY / 'docs/archive/s1_rank_configs.zip')
        self.addCleanup(archive.close)
        datasets, training_settings = [], []
        models = ('s1water_unet', 'pvt_deeplabv3plus', 'mobilenetv2_unet', 's1water_unetpp')
        for name in models:
            with self.subTest(model=name):
                path = REPOSITORY / 'configs/s1' / f'train_{name}_trainnorm80.yaml'
                current = yaml.safe_load(path.read_text(encoding='utf-8'))
                original = yaml.safe_load(archive.read(f'configs/s1/train_{name}.yaml'))
                self.assertIsNone(current['paths']['RESUME_CHECKPOINT'])
                self.assertFalse(current['model']['PRETRAINED'])
                self.assertEqual(current['training']['EPOCHS'], 80)
                self.assertNotIn('EXTENSION', current['training'])
                self.assertEqual(current['data']['PATCH_REPRESENTATION'], REPRESENTATION)
                self.assertEqual(current['paths']['OUTPUT_DIR'], original['paths']['OUTPUT_DIR'] + '_trainnorm80')
                datasets.append(current['data'])
                training_settings.append(current['training'])
                restored = json.loads(json.dumps(current))
                restored['data']['PATCH_REPRESENTATION'] = original['data']['PATCH_REPRESENTATION']
                restored['training']['EPOCHS'] = original['training']['EPOCHS']
                restored['paths']['OUTPUT_DIR'] = original['paths']['OUTPUT_DIR']
                self.assertEqual(restored, original)
                train.configure(path)
                with patch.object(train, 'WaterSegmentationTrainer') as factory:
                    train.create_trainer()
                options = factory.call_args.kwargs
                self.assertEqual(options['bands'], (1, 4))
                self.assertEqual(options['model_kwargs']['in_channels'], 2)
                self.assertEqual(options['model_info']['input_channels'], ('VV', 'DEM'))
        self.assertTrue(all(item == datasets[0] for item in datasets))
        self.assertTrue(all(item == training_settings[0] for item in training_settings))

    def test_four_log_configs_change_only_representation_and_fresh_output(self):
        folder = REPOSITORY / 'configs/s1'
        for name in ('s1water_unet', 'pvt_deeplabv3plus', 'mobilenetv2_unet', 's1water_unetpp'):
            with self.subTest(network=name):
                baseline = yaml.safe_load((folder / f'train_{name}_trainnorm80.yaml').read_text(encoding='utf-8'))
                path = folder / f'train_{name}_log_trainnorm80.yaml'
                logarithmic = yaml.safe_load(path.read_text(encoding='utf-8'))
                self.assertEqual(logarithmic['data']['PATCH_REPRESENTATION'], LOG_REPRESENTATION)
                self.assertNotEqual(logarithmic['paths']['OUTPUT_DIR'], baseline['paths']['OUTPUT_DIR'])
                self.assertIsNone(logarithmic['paths']['RESUME_CHECKPOINT'])
                logarithmic['data']['PATCH_REPRESENTATION'] = baseline['data']['PATCH_REPRESENTATION']
                logarithmic['paths']['OUTPUT_DIR'] = baseline['paths']['OUTPUT_DIR']
                self.assertEqual(logarithmic, baseline)
                train.configure(path)
                with patch.object(train, 'WaterSegmentationTrainer') as factory:
                    train.create_trainer()
                self.assertEqual(factory.call_args.kwargs['bands'], (1, 4))
                self.assertEqual(factory.call_args.kwargs['patch_representation'], LOG_REPRESENTATION)

    def test_checkpoint_restores_frozen_transform_without_refitting(self):
        continuous = self.make()
        continuous.train()
        expected = torch.load(continuous.run_dir / 'last.pth', weights_only=True, map_location='cpu')
        partial = self.make()
        original_run = partial.run_epoch

        def interrupt(epoch, phase, loader):
            if epoch == 1 and phase == 'val':
                raise RuntimeError('test validation interrupt')
            return original_run(epoch, phase, loader)

        with patch.object(partial, 'run_epoch', side_effect=interrupt):
            with self.assertRaisesRegex(RuntimeError, 'test validation interrupt'):
                partial.train()
        checkpoint_path = partial.run_dir / 'train_end.pth'
        saved = torch.load(checkpoint_path, weights_only=True, map_location='cpu')
        self.assertEqual(saved['relative_normalization'],
                         json.loads((partial.run_dir / 'relative_normalization.json').read_text(encoding='utf-8')))
        with patch.object(WaterDataset, 'fit_relative_normalization', side_effect=AssertionError('must not refit')):
            resumed = self.make(checkpoint_path)
            self.assertTrue(resumed.resume_validation)
            resumed.train()
        actual = torch.load(resumed.run_dir / 'last.pth', weights_only=True, map_location='cpu')
        self.assertEqual(actual['relative_normalization'], expected['relative_normalization'])
        self.assertEqual(actual['scheduler_state_dict'], expected['scheduler_state_dict'])
        torch.testing.assert_close(actual['mean'], expected['mean'], rtol=0, atol=0)
        torch.testing.assert_close(actual['std'], expected['std'], rtol=0, atol=0)
        for name, values in expected['model_state_dict'].items():
            torch.testing.assert_close(actual['model_state_dict'][name], values, rtol=0, atol=0)
        for pid, state in expected['optimizer_state_dict']['state'].items():
            for key, value in state.items():
                torch.testing.assert_close(actual['optimizer_state_dict']['state'][pid][key], value, rtol=0, atol=0)

    def test_log_baseline_restores_fitted_transform_on_resume(self):
        trainer = self.make(representation=LOG_REPRESENTATION)
        trainer.train()
        path = trainer.run_dir / 'last.pth'
        checkpoint = torch.load(path, weights_only=True, map_location='cpu')
        self.assertEqual(checkpoint['patch_representation'], LOG_REPRESENTATION)
        self.assertEqual(checkpoint['relative_normalization']['representation'], LOG_REPRESENTATION)
        self.assertLess(checkpoint['relative_normalization']['p98'], 25.0)
        with patch.object(WaterDataset, 'fit_relative_normalization', side_effect=AssertionError('must not refit')):
            resumed = self.make(path, representation=LOG_REPRESENTATION)
            self.assertEqual(resumed.relative_normalization, trainer.relative_normalization)
            with self.assertRaisesRegex(ValueError, '输入表示'):
                self.make(path, representation=REPRESENTATION)

    def test_missing_metadata_and_changed_representation_fail_before_fit(self):
        trainer = self.make()
        trainer.train()
        path = trainer.run_dir / 'last.pth'
        checkpoint = torch.load(path, weights_only=True, map_location='cpu')
        checkpoint.pop('relative_normalization')
        torch.save(checkpoint, path)
        with patch.object(WaterDataset, 'fit_relative_normalization', side_effect=AssertionError('must not refit')):
            with self.assertRaisesRegex(ValueError, 'relative_normalization'):
                self.make(path)
            with self.assertRaisesRegex(ValueError, '输入表示'):
                self.make(path, representation='vv_rank_dem_m')


if __name__ == '__main__':
    unittest.main()
