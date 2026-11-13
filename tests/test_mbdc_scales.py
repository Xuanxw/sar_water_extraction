"""核对按尺度空洞率、有效邻域梯度、旧结构与新检查点重载。"""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"),
                str(Path(__file__).resolve().parents[1] / "scripts")]
from supervised import train
from supervised.models.pvt import MBDC, PVTUANet
from supervised.models.checkpoint import load_model

# 测试显式加载样例配置，正式脚本仅使用命令行提供的文件。
train.configure(Path(__file__).resolve().parents[1] / 'configs/train.yaml')

RATES = (((4, 8, 16), (2, 4, 8)), ((2, 4, 8), (1, 2, 4)),
         ((1, 2, 4), (1, 1, 2)), ((1, 1, 2), (1, 1, 1)))


class MBDCScaleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_scale_shapes_padding_and_initial_weights_stay_comparable(self):
        for size, channels, rates in zip((128, 64, 32, 16), (64, 128, 320, 512), RATES):
            torch.manual_seed(42)
            old = MBDC(channels).eval()
            torch.manual_seed(42)
            new = MBDC(channels, dilations=rates).eval()
            self.assertEqual(list(old.state_dict()), list(new.state_dict()))
            for key, value in old.state_dict().items():
                torch.testing.assert_close(value, new.state_dict()[key], rtol=0, atol=0)
            for branch, branch_rates in zip((new.branch1, new.branch2), rates):
                for layer, rate in zip(branch[1:], branch_rates):
                    self.assertEqual(layer.conv.dilation, (rate, rate))
                    self.assertEqual(layer.conv.padding, (rate, rate))
            with torch.no_grad():
                x = torch.randn(1, channels, size, size)
                self.assertEqual(tuple(new(x).shape), (1, 32, size, size))
                if size == 128:  # 最浅层保持原配置，输出应逐位一致。
                    torch.testing.assert_close(old(x), new(x), rtol=0, atol=0)

    def test_deepest_convolution_restores_off_center_sampling(self):
        x = torch.ones(1, 32, 16, 16)
        old = MBDC(512).branch1[-1].conv
        new = MBDC(512, dilations=RATES[-1]).branch1[-1].conv
        old(x).sum().backward()
        new(x).sum().backward()
        # 原 dilation=16 的周围8个核位置只接触补零；新 dilation=2 能接触有效像素。
        self.assertEqual(old.weight.grad[:, :, 0, 0].abs().sum().item(), 0)
        self.assertGreater(new.weight.grad[:, :, 0, 0].abs().sum().item(), 0)

    def test_new_checkpoint_restores_rates_and_old_default_is_unchanged(self):
        base = dict(in_channels=2, num_classes=1, pretrained=False, stage="uafm", final_residual=True)
        torch.manual_seed(42)
        old = PVTUANet(**base).eval()
        torch.manual_seed(42)
        new = PVTUANet(**base, mbdc_dilations=RATES).eval()
        self.assertEqual(sum(p.numel() for p in old.parameters()), sum(p.numel() for p in new.parameters()))
        for key, value in old.state_dict().items():
            torch.testing.assert_close(value, new.state_dict()[key], rtol=0, atol=0)
        self.assertEqual(old.conv_5[0].branch1[-1].conv.dilation, (16, 16))
        restored = load_model(dict(model_name="pvt_uafnet", model_kwargs=dict(base, mbdc_dilations=RATES),
                                   model_state_dict=new.state_dict())).eval()
        self.assertEqual(restored.conv_5[0].branch1[-1].conv.dilation, (2, 2))
        with torch.inference_mode():
            x = torch.randn(1, 2, 64, 64)
            for actual, expected in zip(restored(x), new(x)):
                self.assertEqual(tuple(actual.shape), (1, 1, 64, 64))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_factory_changes_only_structure_configuration_and_directory(self):
        choices = []
        for rates in (None, RATES):
            with patch.object(train, "METHOD", "uafnet"), patch.object(train, "UAFNET_BACKBONE", "pvt_v2_b2.in1k"), patch.object(train, "UAFNET_REFINEMENT_DECODER", "uafm"), patch.object(train, "UAFNET_HIGH_RESOLUTION_REFINEMENT", False), patch.object(train, "LOSS_NAME", "bce"), \
                 patch.object(train, "SAMPLING", "hard"), patch.object(train, "OUTPUT_DIR", None), \
                 patch.object(train, "UAFNET_FINAL_RESIDUAL", True), \
                 patch.object(train, "HEAD_LOSS_WEIGHTS", (1., 1., 1., 1.)), \
                 patch.object(train, "UAFNET_MBDC_DILATIONS", rates), \
                 patch.object(train, "WaterSegmentationTrainer") as factory:
                train.create_trainer()
                choices.append(factory.call_args.kwargs)
        old, new = choices
        for key in old:
            if key not in ("model_kwargs", "output_dir"):
                self.assertEqual(old[key], new[key], key)
        options = dict(new["model_kwargs"])
        self.assertEqual(options.pop("mbdc_dilations"), RATES)
        self.assertEqual(options, old["model_kwargs"])
        self.assertEqual(old["output_dir"].name, "pvt_uafnet_hard_sampling_residual")
        self.assertEqual(new["output_dir"].name, "pvt_uafnet_hard_sampling_residual_scale_mbdc")
        for method in ("unet", "pvt", "resnet50", "uanet"):
            with patch.object(train, "METHOD", method), patch.object(train, "LOSS_NAME", "ce"), \
                 patch.object(train, "SAMPLING", "random"), patch.object(train, "UAFNET_MBDC_DILATIONS", RATES), \
                 patch.object(train, "WaterSegmentationTrainer") as factory:
                train.create_trainer()
                options = factory.call_args.kwargs
                self.assertNotIn("mbdc_dilations", options["model_kwargs"])
                self.assertNotIn("scale_mbdc", options["output_dir"].name)


if __name__ == "__main__":
    unittest.main()
