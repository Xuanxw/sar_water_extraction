"""检查高分辨率细化的恒等初始化、梯度、尺寸和检查点兼容性；仅使用 CPU。"""
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from models.pvt import HighResolutionRefinement, PVTUANet
from models.checkpoint import load_model


class HighResolutionRefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_initial_outputs_and_existing_weights_are_unchanged(self):
        options = dict(in_channels=2, num_classes=1, pretrained=False,
                       stage="uafm", final_residual=True)
        torch.manual_seed(42)
        baseline = PVTUANet(**options).eval()
        torch.manual_seed(42)
        refined = PVTUANet(**options, high_resolution_refinement=True).eval()
        for name, value in baseline.state_dict().items():
            torch.testing.assert_close(value, refined.state_dict()[name], rtol=0, atol=0)
        self.assertFalse(any(name.startswith("highres_refinement.")
                             for name in baseline.state_dict()))
        self.assertEqual(sum(p.numel() for p in refined.parameters())
                         - sum(p.numel() for p in baseline.parameters()), 5105)
        image = torch.randn(1, 2, 64, 64)
        with torch.no_grad():
            old_outputs, new_outputs = baseline(image), refined(image)
        self.assertEqual(len(new_outputs), 4)
        for old, new in zip(old_outputs, new_outputs):
            torch.testing.assert_close(old, new, rtol=0, atol=0)
        # 分支学到非零修正后，前三个监督头仍不受它直接改写，最后头才发生变化。
        with torch.no_grad():
            refined.highres_refinement.delta.bias.fill_(0.2)
            changed = refined(image)
        for old, new in zip(old_outputs[:3], changed[:3]):
            torch.testing.assert_close(old, new, rtol=0, atol=0)
        torch.testing.assert_close(changed[-1], old_outputs[-1] + 0.2)

    def test_zero_initialization_gradients_and_odd_rectangular_sizes(self):
        for in_channels in (1, 2):
            with self.subTest(in_channels=in_channels):
                torch.manual_seed(42)
                branch = HighResolutionRefinement(in_channels)
                image = torch.randn(2, in_channels, 65, 71, requires_grad=True)
                logits = torch.randn(2, 1, 17, 18, requires_grad=True)
                correction = branch(image, logits)
                self.assertEqual(tuple(correction.shape), (2, 1, 65, 71))
                self.assertEqual(torch.count_nonzero(correction).item(), 0)
                optimizer = torch.optim.SGD(branch.parameters(), lr=0.1)
                F.binary_cross_entropy_with_logits(correction, torch.ones_like(correction)).backward()
                self.assertGreater(branch.delta.weight.grad.abs().sum().item(), 0)
                self.assertGreater(branch.delta.bias.grad.abs().sum().item(), 0)
                self.assertEqual(branch.shallow[0].weight.grad.abs().sum().item(), 0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                image.grad = None
                logits.grad = None
                correction = branch(image, logits)
                F.binary_cross_entropy_with_logits(correction, torch.ones_like(correction)).backward()
                for name, parameter in branch.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
                self.assertGreater(image.grad.abs().sum().item(), 0)
                self.assertGreater(logits.grad.abs().sum().item(), 0)

    def test_old_and_new_checkpoints_reload_strictly_for_both_polarizations(self):
        with TemporaryDirectory() as folder:
            for in_channels, enabled in ((2, False), (1, True), (2, True)):
                with self.subTest(in_channels=in_channels, enabled=enabled):
                    options = dict(in_channels=in_channels, num_classes=1, pretrained=False,
                                   stage="uafm", final_residual=True)
                    # 旧检查点没有新参数；新检查点通过 model_kwargs 恢复分支。
                    if enabled:
                        options["high_resolution_refinement"] = True
                    model = PVTUANet(**options).eval()
                    if enabled:
                        with torch.no_grad():
                            model.highres_refinement.delta.bias.fill_(0.25)
                    image = torch.randn(1, in_channels, 64, 64)
                    with torch.no_grad():
                        expected = model(image)
                    path = Path(folder) / "model.pth"
                    torch.save(dict(model_name="pvt_uafnet", model_kwargs=options,
                                    model_state_dict=model.state_dict()), path)
                    del model
                    restored = load_model(torch.load(path, weights_only=True)).eval()
                    self.assertEqual(restored.high_resolution_refinement, enabled)
                    with torch.no_grad():
                        outputs = restored(image)
                    for old, new in zip(expected, outputs):
                        torch.testing.assert_close(old, new, rtol=0, atol=0)
                    del restored

    def test_refinement_requires_single_output_uafnet(self):
        for options in (dict(num_classes=2, stage="uafm"),
                        dict(num_classes=2, stage="full"),
                        dict(num_classes=1, stage="fpn")):
            with self.subTest(options=options), self.assertRaises(ValueError):
                PVTUANet(pretrained=False, high_resolution_refinement=True, **options)


if __name__ == "__main__":
    unittest.main()
