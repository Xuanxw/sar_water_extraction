"""验证水体单通道训练、忽略像素、深监督以及检查点恢复后的预测。"""
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import train
from models.pvt import PVTUANet, UncertaintyRank
from models.checkpoint import load_model


class TinyWaterDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, label_dir=None, bands=None):
        self.image_dir = Path(image_dir)
        self.image_paths = [self.image_dir / f"{i}.tif" for i in range(4)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = torch.randn(2, 64, 64, generator=torch.Generator().manual_seed(index))
        label = (image[0] > 0).long()
        label[:3] = -1
        valid = torch.ones(64, 64, dtype=torch.bool)
        valid[:2] = False
        image[:, :2] = 0
        return dict(image=image, label=label, valid_mask=valid, name=f"{index}.tif")


class UAFNetTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_single_channel_outputs_and_all_head_gradients(self):
        model = PVTUANet(num_classes=1, pretrained=False, stage="uafm")
        outputs = model(torch.randn(2, 2, 64, 64))
        self.assertEqual(len(outputs), 4)
        for output in outputs:
            self.assertEqual(tuple(output.shape), (2, 1, 64, 64))
        loss = sum(torch.nn.functional.binary_cross_entropy_with_logits(
            output, torch.ones_like(output)) for output in outputs)
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        fore, back = UncertaintyRank(include_midpoint=True)(torch.zeros(1, 1, 1, 1))
        self.assertEqual((fore.item(), back.item()), (5, 5))

    def test_training_and_reload_agree_with_masked_bce(self):
        with TemporaryDirectory() as folder:
            with patch.object(train, "METHOD", "uafnet"), \
                 patch.object(train, "OUTPUT_DIR", Path(folder)), \
                 patch.object(train, "PRETRAINED", False), patch.object(train, "EPOCHS", 2), \
                 patch.object(train, "LEARNING_RATE", 1e-4), \
                 patch.object(train, "WaterDataset", TinyWaterDataset), \
                 patch.object(torch.cuda, "is_available", return_value=False):
                trainer = train.create_trainer()
                trainer.train()
            self.assertIsInstance(trainer.optimizer, torch.optim.AdamW)
            self.assertEqual(trainer.scheduler.last_epoch, 2)
            self.assertAlmostEqual(trainer.optimizer.param_groups[0]["lr"], 0)
            for filename in ("best.pth", "best_f1.pth", "last.pth"):
                self.assertTrue((trainer.run_dir / filename).is_file())
            checkpoint = torch.load(trainer.run_dir / "last.pth", weights_only=True)
            self.assertEqual(checkpoint["model_name"], "pvt_uafnet")
            self.assertEqual(checkpoint["model_kwargs"]["num_classes"], 1)
            self.assertIn("optimizer_state_dict", checkpoint)
            model = load_model(checkpoint).eval()
            batch = next(iter(trainer.val_loader))
            image = (batch["image"] - trainer.mean) / trainer.std
            image = image.masked_fill(~batch["valid_mask"].unsqueeze(1), 0)
            with torch.no_grad():
                logits = model(image)[-1][:, 0]
            label = batch["label"]
            valid = label != -1
            expected_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits[valid], label[valid].float()).item()
            self.assertAlmostEqual(expected_loss, trainer.phase_metrics["val"]["loss"], places=6)
            predicted = (logits >= 0).long()
            counts = torch.bincount(label[valid] * 2 + predicted[valid], minlength=4).tolist()
            self.assertEqual(counts, [trainer.phase_metrics["val"][k] for k in ("tn", "fp", "fn", "tp")])


if __name__ == "__main__":
    unittest.main()
