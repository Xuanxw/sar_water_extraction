"""检验分阶段结构、多输出监督、等级权重和新旧模型加载接口。"""
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"),
                str(Path(__file__).resolve().parents[1])]
from supervised.models.pvt import PVTUANet
from supervised.models.pvt import UncertaintyRank
from supervised.models.checkpoint import load_model


class UANetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_rank_semantics_and_detach(self):
        probabilities = torch.tensor([.05, .15, .25, .35, .45, .5, .55, .65, .75, .85, .95])
        logits = torch.logit(probabilities).reshape(1, 1, 1, -1).requires_grad_()
        fore, back = UncertaintyRank()(logits)
        self.assertEqual(fore.flatten().tolist(), [0, 0, 0, 0, 0, 0, 5, 4, 3, 2, 1])
        self.assertEqual(back.flatten().tolist(), [1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0])
        self.assertFalse(fore.requires_grad or back.requires_grad)

    def test_all_stages_and_strict_reload(self):
        image = torch.randn(1, 2, 65, 97)
        for stage in ("mbdc", "fpn", "uafm", "full"):
            with self.subTest(stage=stage):
                model = PVTUANet(pretrained=False, stage=stage).eval()
                with torch.no_grad():
                    result = model(image)
                outputs = result if isinstance(result, tuple) else (result,)
                self.assertEqual(len(outputs), 1 if stage in ("mbdc", "fpn") else 4)
                for output in outputs:
                    self.assertEqual(tuple(output.shape), (1, 2, 65, 97))
                    self.assertTrue(torch.isfinite(output).all())
                checkpoint = dict(model_name="pvt_uanet", model_state_dict=model.state_dict(),
                                  model_kwargs=dict(in_channels=2, num_classes=2, decoder_channels=32,
                                                    pretrained=True, stage=stage))
                restored = load_model(checkpoint).eval()
                with torch.no_grad():
                    actual = restored(image)
                actual_outputs = actual if isinstance(actual, tuple) else (actual,)
                for x, y in zip(outputs, actual_outputs):
                    torch.testing.assert_close(x, y, rtol=0, atol=0)

    def test_deep_supervision_reaches_all_used_parameters(self):
        model = PVTUANet(pretrained=False, stage="full").train()
        image = torch.randn(2, 2, 64, 64)
        label = torch.randint(0, 2, (2, 64, 64))
        label[:, :4] = -1
        outputs = model(image)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        loss = sum(criterion(output, label) for output in outputs)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_training_and_calibration_use_final_prediction(self):
        from patch_training_fixture import train_patch_model

        with TemporaryDirectory() as folder:
            calibrator, validation = train_patch_model(Path(folder), "uanet", "pvt_v2_b2.in1k")
            self.assertEqual(calibrator.checkpoint["model_kwargs"]["stage"], "full")
            metrics = calibrator.evaluate()
            self.assertEqual(metrics["valid_pixels"], 2 * 62 * 64)
            for key in ("tn", "fp", "fn", "tp"):
                self.assertEqual(metrics[key], validation[key])
            self.assertAlmostEqual(metrics["loss"], validation["loss"], places=6)


if __name__ == "__main__":
    unittest.main()
