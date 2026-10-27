"""离线检查骨干接口、梯度和检查点；不启动正式训练或下载权重。"""
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from models.checkpoint import load_model
from models.pvt import PVTWaterNet
from models.resnet import ResNet50WaterNet
from models.common import SemanticFPNDecoder


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
        import train
        from calibrate_bn import BNCalibrator

        class TinyDataset(torch.utils.data.Dataset):
            def __init__(self, image_dir, label_dir=None, bands=None):
                self.image_dir = Path(image_dir)
                self.image_paths = [self.image_dir / f"{i}.tif" for i in range(4)]
            def __len__(self):
                return len(self.image_paths)
            def __getitem__(self, i):
                image = torch.randn(2, 64, 64, generator=torch.Generator().manual_seed(i))
                return dict(image=image, label=(image[0] > 0).long(),
                            valid_mask=torch.ones(64, 64, dtype=torch.bool), name=f"{i}.tif")

        for method in ("pvt", "resnet50"):
            # 临时目录固定在本项目 tests 内，结束时只清理该次测试生成的文件。
            with self.subTest(method=method), TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
                with patch.object(train, "METHOD", method), patch.object(train, "OUTPUT_DIR", Path(folder)), \
                     patch.object(train, "LOSS_NAME", "ce"), \
                     patch.object(train, "SAMPLING", "random"), \
                     patch.object(train, "PRETRAINED", False), patch.object(train, "EPOCHS", 1), \
                     patch.object(train, "WaterDataset", TinyDataset), \
                     patch.object(torch.cuda, "is_available", return_value=False):
                    trainer = train.create_trainer()
                    trainer.train()
                checkpoint = torch.load(trainer.run_dir / "last.pth", weights_only=True)
                model = load_model(checkpoint).eval()
                before = {k: v.clone() for k, v in model.state_dict().items()}
                calibrator = BNCalibrator.__new__(BNCalibrator)
                calibrator.model, calibrator.device = model, torch.device("cpu")
                calibrator.mean, calibrator.std = trainer.mean, trainer.std
                calibrator.train_loader = trainer.train_loader
                calibrator.calibrate_layers()
                changed = []
                for name, value in model.state_dict().items():
                    self.assertTrue(torch.isfinite(value).all())
                    if not torch.equal(value, before[name]):
                        changed.append(name)
                        self.assertTrue(name.endswith(("running_mean", "running_var")))
                self.assertTrue(changed)


if __name__ == "__main__":
    unittest.main()
