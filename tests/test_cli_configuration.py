"""命令行配置契约：显式路径、导入隔离、参数校验及配置快照，不启动生产任务。"""
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import yaml
from configuration import config_argument

ROOT = Path(__file__).resolve().parents[1]


class CliConfigurationTests(unittest.TestCase):
    def test_argument_required_and_relative_path_resolved(self):
        with self.assertRaisesRegex(SystemExit, '用法'):
            config_argument(['entry.py'])
        with self.assertRaisesRegex(SystemExit, '用法'):
            config_argument(['entry.py', 'one.yaml', 'two.yaml'])
        expected = Path('configs/中文配置.yaml').resolve()
        self.assertEqual(config_argument(['entry.py', 'configs/中文配置.yaml']), expected)
        self.assertEqual(config_argument(['entry.py', 'configs/中文配置.yaml', 'coarse.yaml'], optional_args=1), expected)
        with patch('sys.argv', ['entry.py', 'configs/中文配置.yaml']):
            self.assertEqual(config_argument(), expected)

    def test_imports_do_not_load_yaml_or_parse_external_arguments(self):
        code = '''
from unittest.mock import patch
import sys
sys.argv = ['external.py', '--unrelated-option']
with patch('configuration.load_yaml', side_effect=AssertionError('must not load YAML at import')):
    from datasets import image_process
    from supervised import train, post_process
    from thres_segmentation import main
    from sar_geometry import main as geometry
    import history_threshold_comparison, slope_threshold_comparison
    import repair_gui_terrain, export_water_polygons, preview_org_coverage
    assert image_process.CONFIG is None
    assert train.CONFIG is None
    assert post_process.CONFIG is None
'''
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT/'src'), str(ROOT/'scripts')]),
                   PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1')
        result = subprocess.run([sys.executable, '-B', '-c', code], env=env, cwd=ROOT,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_main_uses_supplied_config_and_keeps_exact_snapshot(self):
        from supervised import train
        default = ROOT/'configs/train.yaml'
        with TemporaryDirectory() as temp:
            selected = Path(temp)/'服务器配置.yaml'
            config = yaml.safe_load(default.read_text(encoding='utf-8-sig'))
            config['training']['LEARNING_RATE'] = 0.000321
            selected.write_text(yaml.safe_dump(config, allow_unicode=True), encoding='utf-8')
            trainer = Mock(run_dir=Path(temp))
            try:
                with patch.object(train, 'create_trainer', return_value=trainer):
                    train.main(selected)
                self.assertEqual(train.CONFIG_PATH, selected.resolve())
                self.assertEqual(train.LEARNING_RATE, 0.000321)
                snapshots = list(Path(temp).glob('train_config_*.yaml'))
                self.assertEqual(len(snapshots), 1)
                self.assertEqual(snapshots[0].read_bytes(), selected.read_bytes())
                trainer.train.assert_called_once()
            finally:
                train.configure(default)

    def test_preparation_main_uses_supplied_config(self):
        from datasets import image_process
        default = ROOT/'configs/image_process.yaml'
        with TemporaryDirectory() as temp:
            selected = Path(temp)/'input.yaml'
            config = yaml.safe_load(default.read_text(encoding='utf-8-sig'))
            config['run'].update(MODE='split', PREPARE_BATCH=None)
            selected.write_text(yaml.safe_dump(config, allow_unicode=True), encoding='utf-8')
            try:
                with patch.object(image_process, 'prepare_dataset') as run:
                    image_process.main(selected)
                run.assert_called_once()
                self.assertEqual(image_process.CONFIG_PATH, selected.resolve())
                self.assertEqual(image_process.MODE, 'split')
                self.assertIsNone(image_process.PREPARE_BATCH)
            finally:
                image_process.configure(default)


if __name__ == '__main__':
    unittest.main()
