"""S1 扩训配置保持模型、输入、损失与固定划分不变，首次扩训独立输出。"""
from pathlib import Path
import unittest
from zipfile import ZipFile

import yaml


class S1ExtendedConfigTests(unittest.TestCase):
    def test_only_training_horizon_schedule_and_output_change(self):
        archive = ZipFile(Path(__file__).resolve().parents[1] / 'docs/archive/s1_rank_configs.zip')
        self.addCleanup(archive.close)
        for model in ('s1water_unet', 'pvt_deeplabv3plus', 'mobilenetv2_unet', 's1water_unetpp'):
            with self.subTest(model=model):
                original = yaml.safe_load(archive.read(f'configs/s1/train_{model}.yaml'))
                extended = yaml.safe_load(archive.read(f'configs/s1/train_{model}_extend80.yaml'))
                self.assertEqual(original['training']['EPOCHS'], 20)
                self.assertEqual(extended['training']['EPOCHS'], 80)
                self.assertEqual(extended['training'].pop('EXTENSION'),
                                 dict(learning_rate=0.00005, min_lr=0.000001))
                self.assertEqual(extended['paths']['OUTPUT_DIR'], original['paths']['OUTPUT_DIR'] + '_extend80')
                checkpoint = Path(extended['paths']['RESUME_CHECKPOINT'].replace('\\', '/'))
                self.assertEqual(checkpoint.name, 'last.pth')
                # The archived U-Net++ YAML resumes its extension after an interruption.
                source = extended if model == 's1water_unetpp' else original
                self.assertEqual(checkpoint.parent.parent.as_posix(), source['paths']['OUTPUT_DIR'])
                for key in ('RESUME_CHECKPOINT', 'OUTPUT_DIR'):
                    extended['paths'][key] = original['paths'][key]
                extended['training']['EPOCHS'] = original['training']['EPOCHS']
                self.assertEqual(original, extended)


if __name__ == '__main__':
    unittest.main()
