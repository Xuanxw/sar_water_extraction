"""检查统一入口的方法选择、实验目录和检查点名称，防止选错模型。"""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"),
                str(Path(__file__).resolve().parents[1] / "scripts")]
from supervised import train

# 测试显式加载样例配置，正式脚本仅使用命令行提供的文件。
train.configure(Path(__file__).resolve().parents[1] / 'configs/train.yaml')
class TrainingSelectionTests(unittest.TestCase):
    def test_methods_are_peers_and_keep_checkpoint_contract(self):
        cases = (
            ("unet", "UNet", "unet", "unet_baseline"),
            ("pvt", "PVTWaterNet", "pvt", "pvt_baseline"),
            ("resnet50", "ResNet50WaterNet", "resnet50", "resnet50_baseline"),
            ("uanet", "PVTUANet", "pvt_uanet", "pvt_uanet_mbdc"),
        )
        for method, class_name, checkpoint_name, directory in cases:
            with self.subTest(method=method), patch.object(train, "METHOD", method), \
                 patch.object(train, "LOSS_NAME", "ce"), \
                 patch.object(train, "SAMPLING", "random"), \
                 patch.object(train, "POLARIZATIONS", ("VV", "VH")), \
                 patch.object(train, "POLARIZATION_BANDS", {"VV": 1, "VH": 2}), \
                 patch.object(train, "CLASS_WEIGHT_MODE", "none"), \
                 patch.object(train, "STAGE", "mbdc"), patch.object(train, "OUTPUT_DIR", None), \
                 patch.object(train, "WaterSegmentationTrainer") as factory:
                self.assertIs(train.create_trainer(), factory.return_value)
                options = factory.call_args.kwargs
                self.assertEqual(options["model_class"].__name__, class_name)
                self.assertEqual(options["model_info"]["model_name"], checkpoint_name)
                self.assertEqual(options["output_dir"], train.OUTPUT_ROOT / directory)
                self.assertEqual("stage" in options["model_kwargs"], method == "uanet")

    def test_uanet_stage_controls_its_directory(self):
        for stage in ("mbdc", "fpn", "uafm", "full"):
            with self.subTest(stage=stage), patch.object(train, "METHOD", "uanet"), \
                 patch.object(train, "LOSS_NAME", "ce"), \
                 patch.object(train, "SAMPLING", "random"), \
                 patch.object(train, "POLARIZATIONS", ("VV", "VH")), \
                 patch.object(train, "POLARIZATION_BANDS", {"VV": 1, "VH": 2}), \
                 patch.object(train, "CLASS_WEIGHT_MODE", "none"), \
                 patch.object(train, "STAGE", stage), patch.object(train, "OUTPUT_DIR", None), \
                 patch.object(train, "WaterSegmentationTrainer") as factory:
                train.create_trainer()
                options = factory.call_args.kwargs
                self.assertEqual(options["model_kwargs"]["stage"], stage)
                self.assertEqual(options["output_dir"].name, f"pvt_uanet_{stage}")

    def test_unknown_method_does_not_silently_select_another_model(self):
        with patch.object(train, "METHOD", "unknown"), self.assertRaises(ValueError):
            train.create_trainer()


if __name__ == "__main__":
    unittest.main()
