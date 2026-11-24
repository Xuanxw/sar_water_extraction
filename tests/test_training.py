"""检查统一入口的方法选择、实验目录和检查点名称，防止选错模型。"""
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import yaml

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"),
                str(Path(__file__).resolve().parents[1] / "scripts")]
from supervised import train

# 测试显式加载样例配置，正式脚本仅使用命令行提供的文件。
train.configure(Path(__file__).resolve().parents[1] / 'configs/train.yaml')
class TrainingSelectionTests(unittest.TestCase):
    def test_bc_s1water_selection_preserves_pvt_training_protocol(self):
        default = Path(__file__).resolve().parents[1] / "configs/train.yaml"
        train.configure(default)
        presets = []
        for architecture, backbone, pretrained in (("deeplabv3plus", train.BACKBONE, True),
                                                    ("sentinel1_water_unet", "s1water_cnn", False)):
            with patch.multiple(train, ARCHITECTURE=architecture, BACKBONE=backbone,
                                PRETRAINED=pretrained, OUTPUT_DIR=None), \
                    patch.object(train, "WaterSegmentationTrainer") as factory:
                train.create_trainer()
            presets.append(factory.call_args.kwargs)

        pvt, s1 = presets
        self.assertEqual(pvt["model_info"]["model_name"], "pvt_deeplabv3plus")
        self.assertEqual(s1["model_info"]["model_name"], "sentinel1_water_unet")
        self.assertEqual(s1["model_kwargs"]["in_channels"], 1)
        self.assertFalse(s1["model_info"]["pretrained"])
        self.assertIsNone(s1["resume_checkpoint"])
        self.assertNotEqual(pvt["output_dir"].name, s1["output_dir"].name)
        different = {key for key in pvt if pvt[key] != s1[key]}
        self.assertEqual(different, {"model_class", "model_kwargs", "model_info", "output_dir"})

    def test_yaml_selection_and_old_server_config(self):
        default = Path(__file__).resolve().parents[1] / "configs/train.yaml"
        source = yaml.safe_load(default.read_text(encoding="utf-8"))
        self.addCleanup(train.configure, default)
        with TemporaryDirectory() as temp:
            path = Path(temp) / "train.yaml"
            source["model"].update(ARCHITECTURE="semantic_fpn", BACKBONE=train.RESNET50)
            path.write_text(yaml.safe_dump(source), encoding="utf-8")
            train.configure(path)
            self.assertEqual(train.select_model(), ("semantic_fpn", train.RESNET50))
            source["run"]["METHOD"] = "uafnet"
            path.write_text(yaml.safe_dump(source), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "旧 run.METHOD"):
                train.configure(path)
            del source["model"]["ARCHITECTURE"], source["model"]["BACKBONE"]
            source["model"]["UAFNET_BACKBONE"] = train.RESNET50
            path.write_text(yaml.safe_dump(source), encoding="utf-8")
            train.configure(path)
            self.assertEqual(train.select_model(), ("uafnet", train.RESNET50))

    def test_architecture_backbone_select_the_model(self):
        cases = [
            ("unet", "plain_cnn", "unet", "ce"),
            ("unet", "mobilenet_v2", "ai4g_mobilenetv2_unet_dfc", "bce"),
            ("semantic_fpn", train.PVT_B2, "pvt", "ce"),
            ("semantic_fpn", train.RESNET50, "resnet50", "ce"),
            ("uanet", train.PVT_B2, "pvt_uanet", "ce"),
            ("uafnet", train.RESNET50, "resnet50_uafnet", "bce"),
            ("deeplabv3plus", "pvt_v2_b3.in1k", "pvt_deeplabv3plus", "bce"),
        ]
        for architecture, backbone, name, loss in cases:
            with self.subTest(architecture=architecture, backbone=backbone), patch.multiple(
                train, METHOD=None, ARCHITECTURE=architecture, BACKBONE=backbone, LOSS_NAME=loss,
                UAFNET_REFINEMENT_DECODER="uafm", SAMPLING="random", CLASS_WEIGHT_MODE="none"
            ), patch.object(train, "WaterSegmentationTrainer") as factory:
                train.create_trainer()
                info = factory.call_args.kwargs["model_info"]
                self.assertEqual((info["architecture"], info["backbone"], info["model_name"]),
                                 (architecture, backbone, name))
                opts = factory.call_args.kwargs["model_kwargs"]
                if "backbone_name" in opts:
                    self.assertEqual(opts["backbone_name"], backbone)
        with patch.multiple(train, METHOD=None, ARCHITECTURE="deeplabv3plus", BACKBONE=train.RESNET50):
            with self.assertRaisesRegex(ValueError, "不支持"):
                train.create_trainer()

    def test_legacy_methods_keep_checkpoint_contract(self):
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
