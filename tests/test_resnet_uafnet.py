"""ResNet50 替换完整 UAFNet 编码器的离线检查，不启动正式训练或下载权重。"""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"),
                str(Path(__file__).resolve().parents[1] / "scripts")]
from supervised import train
from supervised.models.pvt import PVTUANet
from supervised.models.checkpoint import load_model
from supervised.models.losses import WaterSegmentationLoss

# 测试显式加载样例配置，正式脚本仅使用命令行提供的文件。
train.configure(Path(__file__).resolve().parents[1] / 'configs/train.yaml')


class ResNetUAFNetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_shapes_polarizations_and_checkpoint(self):
        for channels in (1, 2):
            with self.subTest(channels=channels):
                options = dict(in_channels=channels, num_classes=1, pretrained=False,
                               stage="uafm", backbone_name="resnet50.tv_in1k", final_residual=True)
                model = PVTUANet(**options).eval()
                self.assertEqual(model.backbone.feature_info.channels(), [256, 512, 1024, 2048])
                self.assertEqual(model.backbone.feature_info.reduction(), [4, 8, 16, 32])
                image = torch.randn(1, channels, 65, 97)
                with torch.no_grad():
                    outputs = model(image)
                self.assertEqual(len(outputs), 4)
                for output in outputs:
                    self.assertEqual(tuple(output.shape), (1, 1, 65, 97))
                    self.assertTrue(torch.isfinite(output).all())
                checkpoint = dict(model_name="resnet50_uafnet", model_kwargs=dict(options, pretrained=True),
                                  model_state_dict=model.state_dict())
                restored = load_model(checkpoint).eval()  # 重载不得再次下载预训练权重。
                with torch.no_grad():
                    for actual, expected in zip(restored(image), outputs):
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_full_resolution_and_residual_initialization(self):
        model = PVTUANet(num_classes=1, pretrained=False, stage="uafm",
                         backbone_name="resnet50.tv_in1k", final_residual=True).eval()
        with torch.no_grad():
            outputs = model(torch.randn(1, 2, 512, 512))
        self.assertTrue(all(tuple(x.shape) == (1, 1, 512, 512) for x in outputs))
        # 零初始化最终修正，初始预测严格等于粗预测。
        torch.testing.assert_close(outputs[-1], outputs[0], rtol=0, atol=0)

    def test_bce_gradients_reach_encoder_and_decoder(self):
        model = PVTUANet(num_classes=1, pretrained=False, stage="uafm",
                         backbone_name="resnet50.tv_in1k", final_residual=True).train()
        image = torch.randn(2, 2, 64, 64)
        label = torch.randint(0, 2, (2, 64, 64))
        label[:, :3] = -1
        loss = sum(WaterSegmentationLoss("bce")(output, label) for output in model(image))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        for module in (model.backbone, model.conv_2, model.neck, model.decoder, model.ufm_layer4):
            self.assertGreater(sum(p.grad.abs().sum().item() for p in module.parameters()), 0)

    def test_factory_changes_only_backbone_and_metadata(self):
        configs = []
        for backbone in ("pvt_v2_b2.in1k", "resnet50.tv_in1k"):
            with patch.object(train, "UAFNET_BACKBONE", backbone), patch.object(train, "UAFNET_REFINEMENT_DECODER", "uafm"), patch.object(train, "UAFNET_HIGH_RESOLUTION_REFINEMENT", False), \
                 patch.object(train, "WaterSegmentationTrainer") as factory:
                train.create_trainer("uafnet")
                configs.append(factory.call_args.kwargs)
        before, after = configs
        for key in before:
            if key not in ("model_kwargs", "model_info", "output_dir"):
                self.assertEqual(before[key], after[key], key)
        pvt_options, resnet_options = dict(before["model_kwargs"]), dict(after["model_kwargs"])
        pvt_options.pop("backbone_name")
        resnet_options.pop("backbone_name")
        self.assertEqual(pvt_options, resnet_options)
        self.assertEqual(after["model_info"]["model_name"], "resnet50_uafnet")
        self.assertEqual(after["output_dir"].name, "resnet50_uafnet_hard_sampling_residual")
        self.assertEqual(after["loss_name"], "bce")
        self.assertEqual(after["head_loss_weights"], (1., 1., 1., 1.))


if __name__ == "__main__":
    unittest.main()
