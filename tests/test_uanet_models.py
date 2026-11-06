"""检验分阶段结构、多输出监督、等级权重和新旧模型加载接口。"""
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"),
                str(Path(__file__).resolve().parents[1] / "scripts")]
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
        from supervised import train
        from calibrate_bn import BNCalibrator

        class TinyDataset(torch.utils.data.Dataset):
            def __init__(self, image_dir, label_dir=None, bands=None):
                self.image_dir = Path(image_dir)
                self.image_paths = [self.image_dir / f"{i}.tif" for i in range(4)]
            def __len__(self):
                return len(self.image_paths)
            def __getitem__(self, i):
                image = torch.randn(2, 64, 64, generator=torch.Generator().manual_seed(i))
                label = (image[0] > 0).long()
                label[:2] = -1
                return dict(image=image, label=label, name=f"{i}.tif",
                            valid_mask=torch.ones(64, 64, dtype=torch.bool))

        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            with patch.multiple(train, METHOD="uanet", DATA_LAYOUT="patch", POLARIZATIONS=("VV","VH"), POLARIZATION_BANDS={"VV":1,"VH":2}), patch.object(train, "STAGE", "full"), \
                 patch.object(train, "LOSS_NAME", "ce"), \
                 patch.object(train, "SAMPLING", "random"), \
                 patch.object(train, "IMAGE_DIR", Path(folder) / "images"), \
                 patch.object(train, "LABEL_DIR", Path(folder) / "labels"), \
                 patch.object(train, "OUTPUT_DIR", Path(folder)), \
                 patch.object(train, "PRETRAINED", False), patch.object(train, "EPOCHS", 1), \
                 patch("datasets.dfc_sample.s1_dataset.WaterDataset", TinyDataset), \
                 patch.object(torch.cuda, "is_available", return_value=False):
                trainer = train.create_trainer()
                trainer.train()
            checkpoint = torch.load(trainer.run_dir / "last.pth", weights_only=True)
            self.assertEqual(checkpoint["model_kwargs"]["stage"], "full")
            model = load_model(checkpoint).eval()
            calibrator = BNCalibrator.__new__(BNCalibrator)
            calibrator.model, calibrator.device = model, torch.device("cpu")
            calibrator.mean, calibrator.std = trainer.mean, trainer.std
            calibrator.val_loader = trainer.val_loader
            calibrator.ignore_index = -1
            metrics = calibrator.evaluate()
            for key in ("tn", "fp", "fn", "tp"):
                self.assertEqual(metrics[key], trainer.phase_metrics["val"][key])
            self.assertAlmostEqual(metrics["loss"], trainer.phase_metrics["val"]["loss"], places=6)


if __name__ == "__main__":
    unittest.main()
