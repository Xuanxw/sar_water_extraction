"""检查极化选择、输入通道和检查点恢复；不运行真实数据训练。"""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"),
                str(Path(__file__).resolve().parents[1] / "scripts")]
from supervised import train
from supervised.models.pvt import PVTUANet
from supervised.models.checkpoint import load_model


class PolarizationTests(unittest.TestCase):
    def test_selection_controls_bands_channels_metadata_and_directory(self):
        for names, bands, suffix in [(("VV", "VH"), (1, 2), ""),
                                      (("VV",), (1,), "_vv"),
                                      (("VH",), (2,), "_vh"),
                                      (("VH", "VV"), (2, 1), "_vh_vv")]:
            with self.subTest(names=names), \
                 patch.object(train, "METHOD", "uafnet"), patch.object(train, "UAFNET_BACKBONE", "pvt_v2_b2.in1k"), patch.object(train, "UAFNET_REFINEMENT_DECODER", "uafm"), patch.object(train, "UAFNET_HIGH_RESOLUTION_REFINEMENT", False), patch.object(train, "UAFNET_MBDC_DILATIONS", None), \
                 patch.object(train, "LOSS_NAME", "bce"), \
                 patch.object(train, "SAMPLING", "hard"), \
                 patch.object(train, "HEAD_LOSS_WEIGHTS", (1., 1., 1., 1.)), \
                 patch.object(train, "UAFNET_FINAL_RESIDUAL", True), \
                 patch.object(train, "POLARIZATIONS", names), \
                 patch.object(train, "POLARIZATION_BANDS", {"VV": 1, "VH": 2}), \
                 patch.object(train, "OUTPUT_DIR", None), \
                 patch.object(train, "WaterSegmentationTrainer") as factory:
                train.create_trainer()
                options = factory.call_args.kwargs
                self.assertEqual(options["bands"], bands)
                self.assertEqual(options["model_kwargs"]["in_channels"], len(names))
                self.assertTrue(options["model_kwargs"]["final_residual"])
                self.assertEqual(options["model_info"]["polarizations"], names)
                self.assertEqual(options["model_info"]["polarization_bands"], dict(zip(names, bands)))
                self.assertEqual(options["output_dir"].name,
                                 "pvt_uafnet_hard_sampling_residual" + suffix)

    def test_single_channel_residual_model_forward_and_checkpoint_reload(self):
        torch.set_num_threads(2)
        options = dict(in_channels=1, num_classes=1, pretrained=False,
                       stage="uafm", final_residual=True)
        model = PVTUANet(**options).eval()
        restored = load_model(dict(model_name="pvt_uafnet", model_kwargs=options,
                                   model_state_dict=model.state_dict())).eval()
        image = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            predictions = model(image)
            self.assertEqual(len(predictions), 4)
            for output in predictions:
                self.assertEqual(tuple(output.shape), (1, 1, 64, 64))
            torch.testing.assert_close(predictions[-1], predictions[0], rtol=0, atol=0)
            torch.testing.assert_close(predictions[-1], restored(image)[-1], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
