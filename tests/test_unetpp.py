"""验证 U-Net++ 密集连接、网格、梯度、检查点及 S1 对照配置。"""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import yaml
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from supervised.models.checkpoint import load_model
from supervised.models.sentinel1_water_unet import UNetS1Water
from supervised.models.unetpp import UNetPlusPlus


class UNetPlusPlusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_encoder_matches_s1_baseline(self):
        torch.manual_seed(42)
        baseline = UNetS1Water(in_channels=2)
        torch.manual_seed(42)
        nested = UNetPlusPlus(in_channels=2)
        for name, value in baseline.state_dict().items():
            if name.startswith(("down.", "center.")):
                torch.testing.assert_close(value, nested.state_dict()[name], rtol=0, atol=0)
        self.assertEqual(baseline.drop.p, nested.drop.p)
        for module in nested.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.assertEqual((module.eps, module.momentum), (1e-3, 0.01))

    def test_single_vv_and_vv_dem_preserve_odd_grid(self):
        for channels, shape in ((1, (17, 29)), (2, (35, 49))):
            with self.subTest(channels=channels):
                model = UNetPlusPlus(in_channels=channels).eval()
                image = torch.randn(1, channels, *shape)
                original = image.clone()
                padded = nn.functional.pad(image, (0, -shape[1] % 32, 0, -shape[0] % 32))
                with torch.inference_mode():
                    actual = model(image)
                    expected = model(padded)[..., :shape[0], :shape[1]]
                self.assertEqual(actual.shape, (1, 1, *shape))
                self.assertTrue(torch.isfinite(actual).all())
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(image, original, rtol=0, atol=0)

    def test_final_node_receives_all_dense_skips(self):
        model = UNetPlusPlus(in_channels=2).eval()
        captured = []
        handles = []
        # 用不同常数标记五条同尺度跳连和下一级特征，检查拼接的来源与顺序。
        marked = [(model.down[0], 0)]
        marked += [(model.nested[f"x0_{column}"], column) for column in range(1, 5)]
        marked.append((model.nested["x1_4"], 14))
        for block, marker in marked:
            handles.append(block.register_forward_hook(
                lambda module, inputs, output, value=marker: torch.full_like(output, value)
            ))
        handles.append(model.nested["x0_5"].register_forward_pre_hook(
            lambda module, inputs: captured.append(inputs[0].detach().clone())
        ))
        try:
            with torch.inference_mode():
                model(torch.randn(1, 2, 32, 32))
        finally:
            for handle in handles:
                handle.remove()
        expected = torch.tensor([0, 1, 2, 3, 4, 14], dtype=torch.float32)
        expected = torch.repeat_interleave(expected, torch.tensor([32, 32, 32, 32, 32, 64]))
        torch.testing.assert_close(captured[0], expected[None, :, None, None].expand(1, 224, 32, 32))

    def test_bce_reaches_all_encoder_and_nested_nodes(self):
        torch.manual_seed(123)
        model = UNetPlusPlus(in_channels=2, dropout=0).train()
        logits = model(torch.randn(2, 2, 32, 32))
        target = torch.randint(0, 2, logits.shape).float()
        loss = nn.functional.binary_cross_entropy_with_logits(logits, target)
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        for name, block in [*model.nested.items(), *enumerate(model.down), ("center", model.center)]:
            self.assertGreater(block[0][0].weight.grad.abs().sum().item(), 0, str(name))

    def test_checkpoint_round_trip_and_baseline_rejection(self):
        model = UNetPlusPlus(in_channels=2).eval()
        checkpoint = dict(model_name="sentinel1_water_unetpp",
                          model_kwargs=dict(in_channels=2, num_classes=1),
                          model_state_dict=model.state_dict())
        restored = load_model(checkpoint).eval()
        image = torch.randn(1, 2, 32, 32)
        with torch.inference_mode():
            torch.testing.assert_close(model(image), restored(image), rtol=0, atol=0)
        # 禁止把原 U-Net 权重静默加载成新的解码器。
        checkpoint["model_name"] = "sentinel1_water_unet"
        with self.assertRaises(RuntimeError):
            load_model(checkpoint)


class UNetPlusPlusSelectionTests(unittest.TestCase):
    def setUp(self):
        from supervised import train
        self.train = train
        self.default = ROOT / "configs/train.yaml"
        self.train.configure(self.default)
        self.addCleanup(self.train.configure, self.default)

    def test_s1_config_changes_only_architecture_and_output(self):
        paths = [ROOT / f"configs/s1/train_s1water_{name}_trainnorm80.yaml" for name in ("unet", "unetpp")]
        original, nested = [yaml.safe_load(path.read_text(encoding="utf-8")) for path in paths]
        self.assertNotEqual(original["paths"]["OUTPUT_DIR"], nested["paths"]["OUTPUT_DIR"])
        self.assertIsNone(nested["paths"]["RESUME_CHECKPOINT"])
        nested["paths"]["OUTPUT_DIR"] = original["paths"]["OUTPUT_DIR"]
        self.assertEqual(nested["model"].pop("ARCHITECTURE"), "unetpp")
        original["model"].pop("ARCHITECTURE")
        self.assertEqual(nested, original)
        factories = []
        for path in paths:
            self.train.configure(path)
            with patch.object(self.train, "WaterSegmentationTrainer") as factory:
                self.train.create_trainer()
                factories.append(factory.call_args.kwargs)
        original, nested = factories
        self.assertIs(nested["model_class"], UNetPlusPlus)
        self.assertEqual(nested["model_info"]["model_name"], "sentinel1_water_unetpp")
        self.assertEqual(nested["model_info"]["input_channels"], ("VV", "DEM"))
        self.assertFalse(nested["model_info"]["deep_supervision"])
        self.assertEqual(nested["model_kwargs"]["in_channels"], 2)
        self.assertEqual(nested["head_loss_weights"], (1.0,))
        different = {key for key in original if original[key] != nested[key]}
        self.assertEqual(different, {"model_class", "model_info", "output_dir"})

    def test_reject_pretrained_or_wrong_backbone_and_loss(self):
        self.train.configure(ROOT / "configs/s1/train_s1water_unetpp_trainnorm80.yaml")
        for overrides, error in (({"PRETRAINED": True}, "PRETRAINED"),
                                 ({"BACKBONE": "mobilenet_v2"}, "不支持"),
                                 ({"LOSS_NAME": "ce"}, "bce")):
            with self.subTest(overrides=overrides), patch.multiple(self.train, **overrides):
                with self.assertRaisesRegex(ValueError, error):
                    self.train.create_trainer()


if __name__ == "__main__":
    unittest.main()
