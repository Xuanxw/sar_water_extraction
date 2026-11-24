"""离线检查骨干接口、梯度和检查点；不启动正式训练或下载权重。"""
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"),
                str(Path(__file__).resolve().parents[1])]
from supervised.models.checkpoint import load_model
from supervised.models.pvt import PVTWaterNet
from supervised.models.resnet import ResNet50WaterNet
from supervised.models.common import SemanticFPNDecoder


class BackboneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_shapes_and_checkpoint_roundtrip(self):
        for name, model_class, channels in (
            ("pvt", PVTWaterNet, [64, 128, 320, 512]),
            ("resnet50", ResNet50WaterNet, [256, 512, 1024, 2048]),
        ):
            with self.subTest(model=name):
                model = model_class(pretrained=False).eval()
                self.assertIsInstance(model.decoder, SemanticFPNDecoder)
                image = torch.randn(1, 2, 65, 97)
                with torch.no_grad():
                    features = model.backbone(image)
                    output = model(image)
                self.assertEqual([list(x.shape[1:]) for x in features],
                                 [[c, h, w] for c, h, w in zip(channels, [17, 9, 5, 3], [25, 13, 7, 4])])
                self.assertEqual(tuple(output.shape), (1, 2, 65, 97))
                self.assertTrue(torch.isfinite(output).all())
                checkpoint = dict(model_name=name, model_state_dict=model.state_dict(),
                                  model_kwargs=dict(in_channels=2, num_classes=2,
                                                    decoder_channels=32, pretrained=True))
                restored = load_model(checkpoint).eval()  # 即使记录 True，也必须离线重建。
                with torch.no_grad():
                    torch.testing.assert_close(restored(image), output, rtol=0, atol=0)

    def test_gradients_reach_backbone_and_decoder(self):
        for model_class in (PVTWaterNet, ResNet50WaterNet):
            with self.subTest(model=model_class.__name__):
                model = model_class(pretrained=False).train()
                image = torch.randn(2, 2, 64, 64)
                label = torch.randint(0, 2, (2, 64, 64))
                label[:, :5] = -1
                loss = torch.nn.functional.cross_entropy(model(image), label, ignore_index=-1)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                for module in (model.backbone, model.projections, model.decoder):
                    gradients = [p.grad for p in module.parameters() if p.requires_grad]
                    self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
                    self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0)

    def test_training_entries_and_bn_calibration(self):
        from patch_training_fixture import train_patch_model

        for name, backbone in (("pvt", "pvt_v2_b2.in1k"), ("resnet50", "resnet50.tv_in1k")):
            with self.subTest(model=name), TemporaryDirectory() as folder:
                calibrator, _ = train_patch_model(Path(folder), "semantic_fpn", backbone)
                self.assertEqual(calibrator.checkpoint["model_name"], name)
                model = calibrator.model
                before = {key: value.clone() for key, value in model.state_dict().items()}
                calibrator.calibrate_layers()
                changed = []
                for key, value in model.state_dict().items():
                    self.assertTrue(torch.isfinite(value).all(), key)
                    if not torch.equal(value, before[key]):
                        changed.append(key)
                        self.assertTrue(key.endswith(("running_mean", "running_var")), key)
                self.assertTrue(changed)


if __name__ == "__main__":
    unittest.main()
