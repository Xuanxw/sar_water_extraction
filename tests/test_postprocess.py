"""检验概率集成、土地覆盖修正以及输出掩膜与标签的隔离。"""
import csv
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import rasterio
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluation import postprocess
import train


class PostprocessTests(unittest.TestCase):
    def test_terrain_gradient_uses_confirmed_scales_and_preserves_missing_data(self):
        # 每列增加 10 个高程值；10 米像元、1 米/值时为 45 度平面坡。
        dem = np.tile(np.arange(5, dtype=float) * 10, (5, 1))
        gradient, title = postprocess.terrain_gradient(dem)
        np.testing.assert_allclose(gradient, 10)
        self.assertIn("raw units/pixel", title)
        slope, title = postprocess.terrain_gradient(dem, (10, 10), 1)
        np.testing.assert_allclose(slope, 45)
        self.assertIn("degrees", title)
        dem[2, 2] = np.nan
        gradient, _ = postprocess.terrain_gradient(dem)
        self.assertTrue(np.isnan(gradient[2, 1]))
        self.assertTrue(np.isnan(postprocess.terrain_gradient(np.full((4, 4), np.nan))[0]).all())

    def test_merge_only_adds_valid_water(self):
        actual = postprocess.merge_landcover(
            torch.tensor([False, False, True, False]), torch.tensor([80, 80, 10, 80]),
            torch.tensor([True, False, True, True]), torch.tensor([True, True, True, False]))
        self.assertEqual(actual.tolist(), [True, False, True, False])

    def test_metrics_from_known_counts_and_empty_water(self):
        # FP 与 FN 故意不同，才能发现 Precision / Recall 被交换的问题。
        metrics = postprocess.metrics_from_counts((7, 2, 4, 5))
        self.assertAlmostEqual(metrics["iou"], 5 / 11)
        self.assertAlmostEqual(metrics["precision"], 5 / 7)
        self.assertAlmostEqual(metrics["recall"], 5 / 9)
        self.assertAlmostEqual(metrics["f1"], 10 / 16)
        self.assertEqual(metrics["valid_pixels"], 18)
        missed = postprocess.metrics_from_counts((7, 0, 4, 0))
        self.assertEqual([missed[k] for k in ("iou", "precision", "recall", "f1")], [0, 0, 0, 0])
        self.assertTrue(np.isnan(postprocess.metrics_from_counts((7, 2, 0, 0))["recall"]))
        self.assertTrue(np.isnan(postprocess.metrics_from_counts((4, 0, 0, 0))["f1"]))

    def test_probability_average_and_individual_normalization(self):
        class Model(torch.nn.Module):
            def __init__(self, binary):
                super().__init__()
                self.binary = binary

            def forward(self, image):
                logits = image[:, :1] if self.binary else torch.cat((-image[:, 1:2], image[:, 1:2]), 1)
                return torch.zeros_like(logits), logits

        checkpoints = [
            dict(model_name="a", epoch=1, bands=(1,), mean=torch.tensor([0.]), std=torch.tensor([1.]),
                 image_dir="images", label_dir="labels", val_files=["a.tif"]),
            dict(model_name="b", epoch=1, bands=(2, 1), mean=torch.tensor([2., 0.]), std=torch.tensor([2., 1.]),
                 image_dir="images", label_dir="labels", val_files=["a.tif"]),
        ]
        with patch.object(postprocess.torch, "load", side_effect=checkpoints), \
             patch.object(postprocess, "load_model", side_effect=[Model(True), Model(False)]), \
             patch.object(torch.cuda, "is_available", return_value=False):
            ensemble = postprocess.WaterEnsemble(["a.pth", "b.pth"])
        members, average = ensemble.predict(torch.tensor([[[[2.]], [[6.]]]]), torch.ones(1, 1, 1, dtype=torch.bool))
        expected = (torch.sigmoid(torch.tensor(2.)) + torch.sigmoid(torch.tensor(4.))) / 2
        self.assertAlmostEqual(average.item(), expected.item())
        self.assertNotEqual(members[0].item(), members[1].item())

    def test_real_raster_pipeline_and_saved_masks(self):
        class Ensemble:
            def __init__(self, paths, use_amp):
                self.device = torch.device("cpu")
                self.bands = (1, 2)
                self.members = [dict(image_dir=images, label_dir=labels, val_files=["a.tif"],
                                     path=Path("unused.pth"), ignore_index=-1)]

            def predict(self, image, valid):
                probability = torch.zeros(image.shape[0], *image.shape[-2:])
                return [probability], probability

        with TemporaryDirectory() as directory:
            folder = Path(directory)
            images, labels = folder / "images", folder / "labels"
            images.mkdir()
            labels.mkdir()
            data = np.ones((6, 4, 4), dtype=np.int16)
            data[4] = 10
            data[4, 1, :2] = 80
            data[:2, 0, 0] = -9999
            label = np.zeros((4, 4), dtype=np.uint8)
            label[1, 0] = 1
            with rasterio.open(images / "a.tif", "w", driver="GTiff", width=4, height=4,
                               count=6, dtype="int16", nodata=-9999) as target:
                target.write(data)
            with rasterio.open(labels / "a.tif", "w", driver="GTiff", width=4, height=4,
                               count=1, dtype="uint8") as target:
                target.write(label, 1)
            with patch.object(postprocess, "WaterEnsemble", Ensemble):
                metrics = postprocess.run_postprocess(["unused.pth"], output_dir=folder / "result")
            self.assertEqual(metrics["ensemble"]["f1"], 0)
            self.assertAlmostEqual(metrics["postprocessed"]["f1"], 2 / 3)
            self.assertTrue((folder / "result/figures/a.png").is_file())
            with (folder / "result/per_image.csv").open(encoding="utf-8-sig", newline="") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(float(row["raw_fn"]), 1)
            self.assertAlmostEqual(float(row["post_f1"]), 2 / 3)
            self.assertTrue((folder / "result/metrics.csv").is_file())
            with rasterio.open(folder / "result/postprocessed/a.tif") as source:
                result = source.read(1)
                self.assertEqual(source.nodata, 255)
                self.assertEqual(result[0, 0], 255)
                self.assertEqual(result[1, 0], 1)
                self.assertEqual(result[1, 1], 1)
            # 无标签推理只输出预测占比和可视化，不构造 F1 或错误计数。
            with patch.object(postprocess, "WaterEnsemble", Ensemble):
                metrics = postprocess.run_postprocess(
                    ["unused.pth"], with_labels=False, use_saved_val=False, save_masks=False,
                    output_dir=folder / "unlabeled")
            self.assertEqual(metrics, {})
            self.assertFalse((folder / "unlabeled/metrics.csv").exists())
            self.assertTrue((folder / "unlabeled/figures/a.png").is_file())
            with (folder / "unlabeled/per_image.csv").open(encoding="utf-8-sig", newline="") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["name"], "a.tif")
            self.assertFalse(any("f1" in key or key.endswith(("_fp", "_fn")) for key in row))

    def test_uafnet_backbone_selection_keeps_default_and_separate_directories(self):
        for variant in ("b2", "b3", "b4", "b5"):
            with patch.object(train, "METHOD", "uafnet"), patch.object(train, "UAFNET_MBDC_DILATIONS", None), \
                 patch.object(train, "LOSS_NAME", "bce"), \
                 patch.object(train, "SAMPLING", "random"), \
                 patch.object(train, "HEAD_LOSS_WEIGHTS", (1., 1., 1., 1.)), \
                 patch.object(train, "UAFNET_FINAL_RESIDUAL", False), \
                 patch.object(train, "UAFNET_BACKBONE", f"pvt_v2_{variant}.in1k"), \
                 patch.object(train, "OUTPUT_DIR", None), patch.object(train, "WaterSegmentationTrainer") as factory:
                train.create_trainer()
                options = factory.call_args.kwargs
                self.assertEqual(options["model_kwargs"]["backbone_name"], f"pvt_v2_{variant}.in1k")
                self.assertEqual(options["output_dir"].name, "pvt_uafnet" if variant == "b2" else f"pvt_uafnet_{variant}")


if __name__ == "__main__":
    unittest.main()
