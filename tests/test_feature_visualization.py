"""核对采集不修改输出/模型状态，且训练入口能按轮次保存固定样本。"""
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from PIL import Image

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"),
                str(Path(__file__).resolve().parents[1] / "scripts")]
from supervised import train
from supervised.evaluation.visual import FeatureVisualizer
from supervised.models.pvt import PVTUANet
from test_uafnet_training import TinyWaterDataset


class FeatureVisualizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_capture_preserves_outputs_buffers_rng_and_removes_hooks(self):
        model = PVTUANet(pretrained=False, num_classes=1, stage="uafm").eval()
        image = torch.randn(1, 2, 64, 64)
        buffers = {name: value.clone() for name, value in model.named_buffers()}
        rng = torch.get_rng_state().clone()
        with torch.inference_mode():
            expected = model(image)
            with FeatureVisualizer(model) as viewer:
                actual = model(image)
        self.assertEqual(len(viewer.maps), 15)
        self.assertEqual(viewer.maps["PVT 1"].shape, (1, 16, 16))
        self.assertEqual(viewer.maps["PVT 4"].shape, (1, 2, 2))
        for x, y in zip(actual, expected):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        for name, value in model.named_buffers():
            torch.testing.assert_close(value, buffers[name], rtol=0, atol=0)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertFalse(any(module._forward_hooks for module in model.modules()))
        with self.assertRaises(RuntimeError):
            with FeatureVisualizer(model):
                raise RuntimeError("模拟前向中断")
        self.assertFalse(any(module._forward_hooks for module in model.modules()))

    def test_training_records_fixed_validation_sample_at_requested_epochs(self):
        with TemporaryDirectory() as folder:
            with patch.object(train, "METHOD", "uafnet"), \
                 patch.object(train, "WaterDataset", TinyWaterDataset), \
                 patch.object(train, "IMAGE_DIR", Path(folder) / "images"), \
                 patch.object(train, "LABEL_DIR", Path(folder) / "labels"), \
                 patch.object(train, "OUTPUT_DIR", Path(folder)), \
                 patch.object(train, "PRETRAINED", False), patch.object(train, "EPOCHS", 3), \
                 patch.object(train, "SAVE_FEATURE_MAPS", True), patch.object(train, "FEATURE_MAP_INTERVAL", 5), \
                 patch.object(torch.cuda, "is_available", return_value=False):
                trainer = train.create_trainer()
                trainer.train()
            feature_dir = trainer.run_dir / "features"
            self.assertEqual(sorted(p.name for p in feature_dir.iterdir()), ["epoch_001", "epoch_003"])
            for epoch in ("epoch_001", "epoch_003"):
                files = list((feature_dir / epoch).glob("*.png"))
                self.assertEqual([p.stem for p in files], [Path(trainer.val_files[0]).stem])
                with Image.open(files[0]) as image:
                    self.assertGreater(image.width, 1000)
                    image.verify()
            self.assertFalse(any(module._forward_hooks for module in trainer.model.modules()))


if __name__ == "__main__":
    unittest.main()
