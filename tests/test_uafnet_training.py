"""验证水体单通道训练、忽略像素、深监督以及检查点恢复后的预测。"""
import sys
import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"),
                str(Path(__file__).resolve().parents[1] / "scripts")]
from supervised import train
from supervised.models.losses import WaterSegmentationLoss
from supervised.models.pvt import PVTUANet, UncertaintyRank
from supervised.models.checkpoint import load_model


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
        loss = sum(WaterSegmentationLoss("bce_dice")(
            output, torch.ones(output.shape[0], *output.shape[-2:], dtype=torch.long)) for output in outputs)
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        fore, back = UncertaintyRank(include_midpoint=True)(torch.zeros(1, 1, 1, 1))
        self.assertEqual((fore.item(), back.item()), (5, 5))

    def test_training_and_reload_agree_with_masked_bce(self):
        self.check_training_and_reload("bce")

    def test_training_and_reload_agree_with_bce_dice(self):
        self.check_training_and_reload("bce_dice")

    def test_hard_sampling_trains_without_validation_leakage(self):
        self.check_training_and_reload("bce", sampling="hard")

    def test_residual_with_hard_sampling_trains_and_reloads(self):
        self.check_training_and_reload("bce", sampling="hard", residual=True)

    def test_weighted_residual_hard_sampling_trains_and_reloads(self):
        self.check_training_and_reload("bce", sampling="hard", residual=True,
                                       head_weights=(1., .5, .5, 2.))

    def test_scale_mbdc_residual_hard_sampling_trains_and_reloads(self):
        self.check_training_and_reload("bce", sampling="hard", residual=True,
                                       mbdc_dilations=train.UAFNET_MBDC_DILATIONS)

    def test_residual_with_hybrid_sampling_trains_and_reloads(self):
        self.check_training_and_reload("bce", sampling="hybrid_hard", residual=True)

    def test_residual_with_stratified_sampling_trains_and_reloads(self):
        self.check_training_and_reload("bce", sampling="stratified_hard", residual=True)

    def check_training_and_reload(self, loss_name, sampling="random", residual=False,
                                  head_weights=(1., 1., 1., 1.), mbdc_dilations=None):
        with TemporaryDirectory() as folder:
            with patch.object(train, "METHOD", "uafnet"), patch.object(train, "UAFNET_BACKBONE", "pvt_v2_b2.in1k"), patch.object(train, "UAFNET_REFINEMENT_DECODER", "uafm"), patch.object(train, "UAFNET_HIGH_RESOLUTION_REFINEMENT", False), \
                 patch.object(train, "UAFNET_MBDC_DILATIONS", mbdc_dilations), \
                 patch.object(train, "LOSS_NAME", loss_name), \
                 patch.object(train, "HEAD_LOSS_WEIGHTS", head_weights), \
                 patch.object(train, "UAFNET_FINAL_RESIDUAL", residual), \
                 patch.object(train, "SAMPLING", sampling), patch.object(train, "HARD_WARMUP_EPOCHS", 1), \
                 patch.object(train, "IMAGE_DIR", Path(folder) / "images"), \
                     patch.object(train, "LABEL_DIR", Path(folder) / "labels"), \
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
            self.assertEqual(checkpoint["model_kwargs"]["final_residual"], residual)
            self.assertEqual(checkpoint["model_kwargs"].get("mbdc_dilations"), mbdc_dilations)
            self.assertEqual(checkpoint["loss_name"], loss_name)
            self.assertEqual(tuple(checkpoint["head_loss_weights"]), head_weights)
            self.assertEqual(checkpoint["loss_kwargs"], trainer.loss_kwargs)
            self.assertEqual(checkpoint["sampling_config"]["mode"], sampling)
            if sampling == "hybrid_hard":
                self.assertEqual(checkpoint["sampling_config"]["low_water_pool_ratio"], .2)
                self.assertEqual(checkpoint["sampling_config"]["low_water_group"], 1)
            if sampling != "random":
                self.assertEqual(len(checkpoint["sampling_plan"]), len(trainer.train_set))
                self.assertTrue(torch.isfinite(checkpoint["sampling_losses"]).all())
                with (trainer.run_dir / "sampling.csv").open(encoding="utf-8-sig") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), 2 * len(trainer.train_set))
                self.assertEqual(set(r["name"] for r in rows), set(trainer.train_files))
                self.assertFalse(set(r["name"] for r in rows) & set(trainer.val_files))
                for epoch in (1, 2):
                    epoch_rows = [r for r in rows if int(r["epoch"]) == epoch]
                    self.assertEqual(sum(int(r["selected_count"]) for r in epoch_rows), len(trainer.train_set))
                before = trainer.sample_losses.clone()
                trainer.run_epoch(2, "val", trainer.val_loader)
                torch.testing.assert_close(before, trainer.sample_losses)
                if sampling in ("stratified_hard", "hybrid_hard"):
                    self.assertEqual(checkpoint["sampling_groups"].tolist(), trainer.sample_groups.tolist())
                    torch.testing.assert_close(checkpoint["sampling_water_ratios"], trainer.sample_water_ratios)
                    self.assertEqual(len(checkpoint["sampling_groups"]), len(trainer.train_set))
                    with (trainer.run_dir / "sampling_groups.csv").open(encoding="utf-8-sig") as stream:
                        group_rows = list(csv.DictReader(stream))
                    for epoch in (1, 2):
                        selected = [r for r in group_rows if int(r["epoch"]) == epoch]
                        self.assertEqual(sum(int(r["selected_count"]) for r in selected), len(trainer.train_set))
                        self.assertEqual(sum(int(r["hard_pool_samples"]) for r in selected),
                                         0 if epoch == 1 else len(trainer.hard_pool))
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
            if loss_name == "bce_dice":
                probability = logits.sigmoid()[valid]
                target = label[valid].float()
                dice = 1 - (2 * (probability * target).sum() + 1) / (probability.sum() + target.sum() + 1)
                expected_loss = train.BCE_WEIGHT * expected_loss + train.DICE_WEIGHT * dice.item()
            self.assertAlmostEqual(expected_loss, trainer.phase_metrics["val"]["loss"], places=6)
            predicted = (logits >= 0).long()
            counts = torch.bincount(label[valid] * 2 + predicted[valid], minlength=4).tolist()
            self.assertEqual(counts, [trainer.phase_metrics["val"][k] for k in ("tn", "fp", "fn", "tp")])


if __name__ == "__main__":
    unittest.main()
