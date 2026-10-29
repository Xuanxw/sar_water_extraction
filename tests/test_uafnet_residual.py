"""最终头残差对照：保留粗预测、接通梯度、兼容旧模型并隔离实验目录。"""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn import functional as F

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"),
                str(Path(__file__).resolve().parents[1] / "scripts")]
from supervised import train
from supervised.models.pvt import PVTUANet
from supervised.models.checkpoint import load_model


class ResidualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_initial_final_prediction_equals_coarse_without_changing_other_heads(self):
        torch.manual_seed(42)
        original = PVTUANet(pretrained=False, num_classes=1, stage="uafm").eval()
        torch.manual_seed(42)
        residual = PVTUANet(pretrained=False, num_classes=1, stage="uafm", final_residual=True).eval()
        image = torch.randn(2, 2, 64, 64)
        with torch.no_grad():
            old, new = original(image), residual(image)
        for a, b in zip(old[:3], new[:3]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(new[-1], new[0], rtol=0, atol=0)
        self.assertEqual(torch.count_nonzero(residual.ufm_layer2.seg_out.weight).item(), 0)
        self.assertEqual(torch.count_nonzero(residual.ufm_layer2.seg_out.bias).item(), 0)

    def test_final_loss_reaches_coarse_decoder_and_correction_learns_after_first_step(self):
        model = PVTUANet(pretrained=False, num_classes=1, stage="uafm", final_residual=True).eval()
        # 梯度检查不需要训练骨干；保持被检查的 FPN、粗解码器和细化分支可训练。
        for name in ("backbone", "conv_2", "conv_3", "conv_4", "conv_5"):
            getattr(model, name).requires_grad_(False)
        image = torch.randn(2, 2, 64, 64)
        target = (image[:, :1] > 0).float()
        optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=.01)
        F.binary_cross_entropy_with_logits(model(image)[-1], target).backward()
        for part in (model.neck, model.decoder, model.ufm_layer2.seg_out):
            grads = [p.grad for p in part.parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(g.abs().sum().item() for g in grads), 0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        outputs = model(image)
        self.assertFalse(torch.equal(outputs[0], outputs[-1]))
        F.binary_cross_entropy_with_logits(outputs[-1], target).backward()
        # 零初始化只让最后修正卷积之前的特征层在首步梯度为零，更新后应能学习。
        grad = model.ufm_layer2.conv_fusion.weight.grad
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.abs().sum().item(), 0)

    def test_old_checkpoint_without_residual_flag_preserves_original_behavior(self):
        options = dict(in_channels=2, num_classes=1, decoder_channels=32, pretrained=False, stage="uafm")
        old = PVTUANet(**options).eval()
        restored = load_model(dict(model_name="pvt_uafnet", model_kwargs=options,
                                   model_state_dict=old.state_dict())).eval()
        self.assertFalse(restored.final_residual)
        image = torch.randn(2, 2, 64, 64)
        with torch.no_grad():
            torch.testing.assert_close(old(image)[-1], restored(image)[-1], rtol=0, atol=0)

    def test_factory_keeps_training_settings_and_separates_residual_directory(self):
        options = {}
        for enabled in (False, True):
            with patch.object(train, "METHOD", "uafnet"), patch.object(train, "UAFNET_BACKBONE", "pvt_v2_b2.in1k"), patch.object(train, "UAFNET_REFINEMENT_DECODER", "uafm"), patch.object(train, "UAFNET_HIGH_RESOLUTION_REFINEMENT", False), patch.object(train, "UAFNET_MBDC_DILATIONS", None), patch.object(train, "LOSS_NAME", "bce"), \
                 patch.object(train, "SAMPLING", "hard"), patch.object(train, "OUTPUT_DIR", None), \
                 patch.object(train, "HEAD_LOSS_WEIGHTS", (1., 1., 1., 1.)), \
                 patch.object(train, "UAFNET_FINAL_RESIDUAL", enabled), \
                 patch.object(train, "WaterSegmentationTrainer") as factory:
                train.create_trainer()
                options[enabled] = factory.call_args.kwargs
        for key in options[False]:
            if key not in ("model_kwargs", "model_info", "output_dir"):
                self.assertEqual(options[False][key], options[True][key], key)
        a, b = dict(options[False]["model_kwargs"]), dict(options[True]["model_kwargs"])
        self.assertFalse(a.pop("final_residual"))
        self.assertTrue(b.pop("final_residual"))
        self.assertEqual(a, b)
        self.assertEqual(options[False]["output_dir"].name, "pvt_uafnet_hard_sampling")
        self.assertEqual(options[True]["output_dir"].name, "pvt_uafnet_hard_sampling_residual")


if __name__ == "__main__":
    unittest.main()
