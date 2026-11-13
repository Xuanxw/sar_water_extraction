"""YAML入口契约：路径/标签类型、训练配置传递和启动配置快照。"""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch, Mock

from configuration import load_yaml
from supervised import train
from datasets import snap_preprocess

# 测试显式加载样例配置，正式脚本仅使用命令行提供的文件。
train.configure(Path(__file__).resolve().parents[1] / 'configs/train.yaml')


class YamlConfigurationTests(unittest.TestCase):
    def test_paths_and_label_null_key(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            file = root / "input.yaml"
            file.write_text("paths:\n  IMAGE_DIR: 'I:\\水体数据\\images'\n  MANIFEST_PATHS: ['I:\\水体数据\\split_manifest.csv']\n  VALID_MASK_DIR: null\nlabels:\n  CLASS_MAP: {null: 1}\n", encoding="utf-8")
            config = load_yaml(file)
            self.assertIsInstance(config["paths"]["IMAGE_DIR"], Path)
            self.assertIsInstance(config["paths"]["MANIFEST_PATHS"][0], Path)
            self.assertIsNone(config["paths"]["VALID_MASK_DIR"])
            self.assertEqual(config["labels"]["CLASS_MAP"], {None: 1})

    def test_training_factory_passes_yaml_and_never_initializes_snap(self):
        with patch.object(train, "WaterSegmentationTrainer") as factory, \
             patch.object(snap_preprocess, "configure_paths", side_effect=AssertionError("训练不能配置SNAP")), \
             patch.object(snap_preprocess, "snap_runtime", side_effect=AssertionError("训练不能检查SNAP")):
            train.create_trainer()
        options = factory.call_args.kwargs
        self.assertIsNone(options["speckle"])
        self.assertEqual(options["resume_checkpoint"], train.CONFIG["paths"]["RESUME_CHECKPOINT"])
        self.assertEqual(options["learning_rate"], train.CONFIG["training"]["LEARNING_RATE"])
        self.assertEqual(options["scene_options"]["manifest_paths"], train.CONFIG["paths"]["MANIFEST_PATHS"])
        self.assertEqual(options["scene_options"]["cache_dir"], train.CONFIG["paths"]["WINDOW_CACHE_DIR"])

    def test_each_launch_keeps_a_separate_config_snapshot(self):
        with TemporaryDirectory() as temporary:
            trainer = Mock(run_dir=Path(temporary))
            with patch.object(train, "create_trainer", return_value=trainer):
                train.main()
                train.main()
            copies = list(trainer.run_dir.glob("train_config_*.yaml"))
            self.assertEqual(len(copies), 2)
            for copy in copies:
                self.assertEqual(copy.read_bytes(), train.CONFIG_PATH.read_bytes())
            self.assertEqual(trainer.train.call_count, 2)
