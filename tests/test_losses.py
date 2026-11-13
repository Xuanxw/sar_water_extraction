"""验证损失公式、忽略像素梯度以及训练入口的实验分组。"""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn import functional as F

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"),
                str(Path(__file__).resolve().parents[1] / "scripts")]
from supervised.models.losses import WaterSegmentationLoss
from supervised import train

# 测试显式加载样例配置，正式脚本仅使用命令行提供的文件。
train.configure(Path(__file__).resolve().parents[1] / 'configs/train.yaml')
class LossTests(unittest.TestCase):
    def test_bce_matches_original_and_ignored_pixels_have_no_gradient(self):
        score = torch.tensor([[[[0., 2., -3.]]]], requires_grad=True)
        label = torch.tensor([[[1, 0, -1]]])
        actual = WaterSegmentationLoss("bce")(score, label)
        expected = F.binary_cross_entropy_with_logits(score[:, 0], label.clamp_min(0).float(),
                                                       reduction="none")[label != -1].mean()
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertEqual(score.grad[0, 0, 0, 2].item(), 0)

    def test_dice_formula_is_per_image_and_excludes_all_ignored_samples(self):
        score = torch.zeros(3, 1, 1, 3, requires_grad=True)
        label = torch.tensor([[[1, 0, -1]], [[0, 0, 0]], [[-1, -1, -1]]])
        # 第一幅 Dice=1-(1+1)/(1+1+1)=1/3；第二幅无水 Dice=1-1/(1.5+1)=0.6。
        bce = WaterSegmentationLoss("bce")(score, label)
        actual = WaterSegmentationLoss("bce_dice", bce_weight=1, dice_weight=1)(score, label)
        torch.testing.assert_close(actual, bce + torch.tensor((1 / 3 + .6) / 2))
        actual.backward()
        self.assertTrue(torch.isfinite(score.grad).all())
        self.assertEqual(score.grad[0, 0, 0, 2].item(), 0)
        self.assertEqual(score.grad[2].abs().sum().item(), 0)

    def test_all_ignored_and_empty_water_stay_finite(self):
        for name, channels in (("bce", 1), ("bce_dice", 1), ("ce", 2)):
            score = torch.zeros(1, channels, 2, 2, requires_grad=True)
            loss = WaterSegmentationLoss(name)(score, torch.full((1, 2, 2), -1))
            self.assertEqual(loss.item(), 0)
            loss.backward()
            self.assertEqual(score.grad.abs().sum().item(), 0)
        # 无水图上的明显假阳性必须比正确背景付出更大的损失。
        label = torch.zeros(1, 2, 2, dtype=torch.long)
        criterion = WaterSegmentationLoss("bce_dice")
        self.assertGreater(criterion(torch.full((1, 1, 2, 2), 5.), label).item(),
                           criterion(torch.full((1, 1, 2, 2), -5.), label).item())

    def test_ce_matches_original(self):
        score = torch.tensor([[[[2., -1., 0.]], [[-2., 1., 0.]]]])
        label = torch.tensor([[[0, 1, -1]]])
        torch.testing.assert_close(WaterSegmentationLoss("ce")(score, label),
                                   F.cross_entropy(score, label, ignore_index=-1))

    def test_experiment_selection_changes_only_loss_and_output_group(self):
        choices = {}
        for name in ("bce", "bce_dice"):
            with patch.multiple(train, METHOD="uafnet", DATA_LAYOUT="patch", CLASS_WEIGHT_MODE="none", POLARIZATIONS=("VV", "VH"), POLARIZATION_BANDS={"VV":1,"VH":2}), patch.object(train, "UAFNET_BACKBONE", "pvt_v2_b2.in1k"), patch.object(train, "UAFNET_REFINEMENT_DECODER", "uafm"), patch.object(train, "UAFNET_HIGH_RESOLUTION_REFINEMENT", False), patch.object(train, "UAFNET_MBDC_DILATIONS", None), patch.object(train, "LOSS_NAME", name), \
                 patch.object(train, "SAMPLING", "random"), \
                 patch.object(train, "HEAD_LOSS_WEIGHTS", (1., 1., 1., 1.)), \
                 patch.object(train, "UAFNET_FINAL_RESIDUAL", False), \
                 patch.object(train, "OUTPUT_DIR", None), \
                 patch.object(train, "WaterSegmentationTrainer") as factory:
                train.create_trainer()
                choices[name] = factory.call_args.kwargs
        baseline, combined = choices["bce"], choices["bce_dice"]
        for key in ("model_class", "model_kwargs", "bands", "batch_size", "epochs", "learning_rate",
                    "val_ratio", "seed", "optimizer_name", "weight_decay", "cosine_schedule", "train_augmentation"):
            self.assertEqual(baseline[key], combined[key], key)
        self.assertEqual(baseline["output_dir"].name, "pvt_uafnet")
        self.assertEqual(combined["output_dir"].name, "pvt_uafnet_bce_dice")
        self.assertEqual(combined["loss_kwargs"], dict(bce_weight=1., dice_weight=1.))


if __name__ == "__main__":
    unittest.main()
