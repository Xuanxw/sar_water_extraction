"""检验概率集成、土地覆盖修正以及输出掩膜与标签的隔离。"""
import csv
import json
from io import BytesIO
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import rasterio
import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"),
                str(Path(__file__).resolve().parents[1] / "scripts")]
from supervised.evaluation import postprocess
from supervised import train


class RasterResponse(BytesIO):
    def __init__(self, body):
        super().__init__(body)
        self.headers = {"Content-Type": "image/tiff", "Content-Length": str(len(body))}


class PostprocessTests(unittest.TestCase):
    def test_downloaded_auxiliary_analysis_and_explicit_water_addition(self):
        """真实栅格/对齐/分析，只有HTTP和模型替换：下载不应隐式补水或改变验证名单。"""
        class Ensemble:
            def __init__(self, paths, use_amp):
                self.device, self.bands = torch.device("cpu"), (1,)
                self.members = [dict(image_dir=images, label_dir=None, val_files=["a.tif"],
                                     path=Path("unused.pth"), ignore_index=-1)]

            def predict(self, image, valid):
                probability = image[:, 0].clamp(0, 1)
                return [probability], probability

        with TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            transform = rasterio.transform.from_origin(117.01, 30.02, .001, .001)

            def raster(path, data, nodata):
                with rasterio.open(path, "w", driver="GTiff", width=7, height=7, count=1,
                                   dtype=data.dtype, nodata=nodata, crs="EPSG:4326", transform=transform) as target:
                    target.write(data, 1)
                return path.read_bytes()

            sar = np.zeros((7, 7), dtype="float32")
            sar[2:5, 2:5] = 1
            sar[0, 0] = -9999
            raster(images / "a.tif", sar, -9999)
            # 未选中的影像没有CRS；若错误地下载整个目录，这里应使测试失败。
            with rasterio.open(images / "b.tif", "w", driver="GTiff", count=1,
                               width=7, height=7, dtype="uint8") as target:
                target.write(np.zeros((7, 7), dtype="uint8"), 1)
            cover = np.full((7, 7), 40, dtype="uint8")
            cover[0, 0] = cover[1, 1] = cover[3, 3] = 80
            cover[3, 4] = 0  # WorldCover未知，不能变成“陆地”或修正依据。
            terrain = np.full((7, 7), 5, dtype="float32")
            terrain[2:5, 2:5] = 2
            cover_body = raster(root / "cover.tif", cover, 255)
            terrain_body = raster(root / "terrain.tif", terrain, -9999)
            options = dict(with_labels=False, use_landcover=False, batch_size=1,
                           auxiliary_cache_dir=root / "cache",
                           auxiliary_options=dict(download_history=True, download_elevation=True, buffer_m=0),
                           save_visuals=True)

            def respond(url, **kwargs):
                return RasterResponse(cover_body if "worldcover" in url.full_url.lower() else terrain_body)

            with patch.object(postprocess, "WaterEnsemble", Ensemble), \
                 patch("urllib.request.urlopen", side_effect=respond) as request:
                postprocess.run_postprocess(["unused.pth"], output_dir=root / "baseline", **options)
            self.assertEqual(request.call_count, 2)
            baseline = root / "baseline"
            with rasterio.open(baseline / "raw/a.tif") as source:
                raw = source.read(1)
            with rasterio.open(baseline / "postprocessed/a.tif") as source:
                np.testing.assert_array_equal(raw, source.read(1))
            self.assertTrue((baseline / "figures/a.png").is_file())  # 单波段图也能输出。
            self.assertFalse((baseline / "raw/b.tif").exists())
            raw_summary = json.loads((baseline / "analysis/raw/a/summary.json").read_text())
            self.assertEqual(raw_summary["depth"]["elevation_kind"], "DSM")
            self.assertGreater(raw_summary["change"]["lost_water"]["m2"], 0)
            self.assertTrue((baseline / "analysis/raw/a/water_depth_m.tif").is_file())
            with rasterio.open(baseline / "analysis/raw/a/water_change.tif") as source:
                self.assertEqual(source.read(1)[3, 4], 255)

            # 相同辅助数据走缓存；只有显式开启补水才修正，并继续保留raw的退水证据。
            options["save_visuals"] = False
            with patch.object(postprocess, "WaterEnsemble", Ensemble), \
                 patch("urllib.request.urlopen", side_effect=AssertionError("应复用缓存")):
                postprocess.run_postprocess(["unused.pth"], use_external_water=True,
                                           output_dir=root / "corrected", **options)
            with rasterio.open(root / "corrected/postprocessed/a.tif") as source:
                corrected = source.read(1)
            expected = raw.copy()
            expected[1, 1] = 1
            np.testing.assert_array_equal(corrected, expected)
            self.assertEqual(corrected[0, 0], 255)
            post_summary = json.loads((root / "corrected/analysis/postprocessed/a/summary.json").read_text())
            self.assertEqual(post_summary["change"]["lost_water"]["m2"], 0)
            with (root / "corrected/per_image.csv").open(encoding="utf-8-sig") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(int(row["auxiliary_added_pixels"]), 1)
            self.assertLess(float(row["raw_water_area_m2"]), float(row["post_water_area_m2"]))
            # 已准备的逐景路径直接复用；不保存mask时分析仍生成，临时文件正常清理。
            prepared = json.loads((baseline / "postprocess.json").read_text(encoding="utf-8"))["scene_analysis_options"]
            with patch.object(postprocess, "WaterEnsemble", Ensemble), \
                 patch("urllib.request.urlopen", side_effect=AssertionError("本地参考不得联网")):
                postprocess.run_postprocess(["unused.pth"], with_labels=False, use_landcover=False,
                    save_masks=False, save_visuals=False, scene_analysis_options=prepared,
                    output_dir=root / "analysis_only")
            self.assertFalse((root / "analysis_only/raw").exists())
            self.assertFalse(list((root / "analysis_only").glob("water_analysis_*")))
            self.assertTrue((root / "analysis_only/analysis/raw/a/water_depth_m.tif").is_file())

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

    def test_small_negative_logits_are_not_rounded_into_water(self):
        # 来自DeepLab闭环测试：-6.437767e-8 的float32 sigmoid曾舍入为0.5。
        ensemble = postprocess.WaterEnsemble.__new__(postprocess.WaterEnsemble)
        ensemble.device, ensemble.use_amp, ensemble.bands = torch.device("cpu"), False, (1,)
        ensemble.members = [dict(model=torch.nn.Identity(), bands=(1,),
                                 mean=torch.zeros(1,1,1,1), std=torch.ones(1,1,1,1),
                                 speckle_filter=postprocess.SpeckleFilter())]
        logits = torch.tensor([[[[-2e-7, -6.437767e-8, -1e-8, 0., 1e-8, 2e-7]]]])
        members, average = ensemble.predict(logits, torch.ones(1,1,6,dtype=torch.bool))
        expected = logits[:,0] >= 0
        self.assertTrue(torch.equal(members[0] >= .5, expected))
        self.assertTrue(torch.equal(average >= .5, expected))

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
            with patch.object(train, "METHOD", "uafnet"), patch.object(train, "UAFNET_REFINEMENT_DECODER", "uafm"), patch.object(train, "UAFNET_HIGH_RESOLUTION_REFINEMENT", False), patch.object(train, "UAFNET_MBDC_DILATIONS", None), \
                 patch.object(train, "LOSS_NAME", "bce"), \
                 patch.object(train, "CLASS_WEIGHT_MODE", "none"), \
                 patch.object(train, "POLARIZATIONS", ("VV", "VH")), \
                 patch.object(train, "POLARIZATION_BANDS", {"VV": 1, "VH": 2}), \
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
