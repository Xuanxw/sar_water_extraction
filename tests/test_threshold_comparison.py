"""同一验证名单上的模型/阈值对照，检验标签隔离、共同有效域和边界计数。"""
import csv
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
import compare_threshold as compare_threshold
from supervised.evaluation.boundary_eval import boundary_regions, confusion_counts


class DarkWaterModel(torch.nn.Module):
    def forward(self, image):
        return -image[:, :1]


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


class ThresholdComparisonTests(unittest.TestCase):
    def test_saved_validation_common_pixels_and_label_independent_predictions(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            images, labels = root / "images", root / "labels"
            images.mkdir()
            labels.mkdir()
            truth = np.zeros((23, 23), dtype="int16")
            truth[6:17, 6:17] = 1
            image = np.full((2, 23, 23), 100, dtype="float32")
            image[:, truth == 1] = 1
            image[0, 2, 2] = -9999  # 只有 VV 无效，所有方法仍须共同排除。
            image[1, 3, 3] = -9999  # 只有 VH 无效，同样不能算进 VV 或网络。
            image[:, 8, 8] = 0  # 合法零值必须保留。
            truth[4, 4] = -1    # 标签孔洞不影响预测掩膜。
            truth[5, 5] = 7     # 非 0/1 的未知类别不参与任何方法的计量。
            for name in ("train.tif", "a.tif", "b.tif"):
                values = image + 10 if name == "train.tif" else image
                with rasterio.open(images / name, "w", driver="GTiff", width=23, height=23,
                                   count=2, dtype="float32", nodata=-9999) as target:
                    target.write(values)
                with rasterio.open(labels / name, "w", driver="GTiff", width=23, height=23,
                                   count=1, dtype="int16", nodata=-1) as target:
                    target.write(truth, 1)
            checkpoint = dict(model_name="synthetic", epoch=1, bands=(1, 2),
                              mean=torch.tensor([50., 50.]), std=torch.tensor([20., 20.]),
                              image_dir=str(images), label_dir=str(labels),
                              train_files=["train.tif"], val_files=["a.tif", "b.tif"], ignore_index=-1)
            checkpoint_path = root / "best_f1.pth"
            torch.save(checkpoint, checkpoint_path)
            options = dict(checkpoint_path=checkpoint_path, scene_csv=None, use_amp=False,
                           batch_size=2, threshold_options=dict(lee_window=1, water_percent=25),
                           boundary_radii=(1, 3), boundary_points=8, save_masks=True)
            # 实际 WaterEnsemble 负责训练统计标准化，只替换无关的重型模型构造。
            with patch("supervised.evaluation.postprocess.load_model", return_value=DarkWaterModel()), \
                 patch.object(torch.cuda, "is_available", return_value=False):
                first = compare_threshold.run_comparison(output_dir=root / "first",
                                                          max_visualizations=1, **options)
            self.assertEqual([r["name"] for r in read_rows(root / "first/samples.csv")], ["a.tif", "b.tif"])
            complete = {r["method"]: r for r in first if r["subset"] == "all"}
            self.assertEqual(set(complete), {"model", "threshold_VV", "threshold_VH"})
            self.assertTrue(all(r["valid_pixels"] == 2 * (529 - 4) for r in complete.values()))
            self.assertEqual(complete["model"]["f1"], 1)
            self.assertTrue((root / "first/figures/a.png").exists())
            self.assertTrue((root / "first/boundary/figures/a.png").exists())
            self.assertEqual(len(read_rows(root / "first/per_image.csv")), 6)
            valid = np.isin(truth, (0, 1))
            valid[2, 2] = valid[3, 3] = False
            regions = boundary_regions(truth, valid, (1, 3))
            boundary_rows = read_rows(root / "first/boundary/per_image.csv")
            for method in complete:
                with rasterio.open(root / "first/masks" / method / "a.tif") as source:
                    pred = source.read(1)
                self.assertEqual(pred[2, 2], 255)
                self.assertEqual(pred[3, 3], 255)
                self.assertNotEqual(pred[4, 4], 255)
                self.assertEqual(pred[8, 8], 1)
                counts = confusion_counts(truth, pred == 1, valid)
                np.testing.assert_array_equal(counts * 2, [complete[method][k] for k in ("tn", "fp", "fn", "tp")])
                for radius in regions:
                    selected = next(r for r in boundary_rows if r["name"] == "a.tif" and
                                    r["method"] == method and r["radius_px"] == str(radius) and
                                    r["region"] == "boundary" and r["scope"] == "all_pixels")
                    expected = confusion_counts(truth, pred == 1,
                                                regions[radius]["inner"] | regions[radius]["outer"])
                    np.testing.assert_array_equal(expected, [int(selected[k]) for k in ("tn", "fp", "fn", "tp")])
            point_rows = read_rows(root / "first/boundary/points.csv")
            self.assertGreater(len(point_rows), 0)
            for row in point_rows:
                r, c = int(row["row"]), int(row["col"])
                self.assertTrue(valid[r, c])
                # 一行固定点同时包含全部方法，阈值的强度不伪装成概率。
                self.assertIn("model_prediction", row)
                self.assertIn("threshold_VV_prediction", row)
                self.assertIn("threshold_VH_prediction", row)
                self.assertEqual(row["threshold_VV_probability"], "")
            # 已有模型掩膜模式完全不构造模型；整体与局部计数应与现场推理一致。
            with patch.object(compare_threshold, "WaterEnsemble", side_effect=AssertionError("不得构造模型")):
                cached = compare_threshold.run_comparison(
                    output_dir=root / "cached", model_mask_dir=root / "first/masks/model",
                    max_visualizations=0, **options)
            cached_all = {r["method"]: r for r in cached if r["subset"] == "all"}
            for method in complete:
                for key in ("tn", "fp", "fn", "tp", "iou", "f1"):
                    self.assertEqual(complete[method][key], cached_all[method][key])
            self.assertEqual(read_rows(root / "first/boundary/metrics.csv"),
                             read_rows(root / "cached/boundary/metrics.csv"))
            cached_points = read_rows(root / "cached/boundary/points.csv")
            self.assertTrue(all(r["model_probability"] == "" for r in cached_points))
            self.assertEqual(read_rows(root / "cached/protocol.csv")[0]["prediction_source"], "cached_masks")
            # 两侧均填缓存时既不构造网络，也不再计算Lee/百分位阈值。
            with patch.object(compare_threshold, "WaterEnsemble", side_effect=AssertionError("不得推理")), \
                 patch.object(compare_threshold, "ThresholdSegmenter", side_effect=AssertionError("不得重新分割")):
                reused = compare_threshold.run_comparison(
                    output_dir=root / "reused", model_mask_dir=root / "first/masks/model",
                    threshold_mask_dirs={band: root / f"first/masks/threshold_{band}" for band in ("VV", "VH")},
                    max_visualizations=0, **options)
            self.assertEqual(read_rows(root / "cached/metrics.csv"),
                             read_rows(root / "reused/metrics.csv"))
            self.assertEqual(read_rows(root / "first/boundary/metrics.csv"),
                             read_rows(root / "reused/boundary/metrics.csv"))
            threshold_rows = read_rows(root / "reused/thresholds.csv")
            self.assertTrue(all(r["prediction_source"] == "cached_masks" and r["intensity_threshold"] == ""
                                for r in threshold_rows))
            # 缓存缺少有效预测或发生网格偏移，必须拒绝，不能悄悄删掉这些评价像素。
            mask_path = root / "first/masks/model/a.tif"
            with rasterio.open(mask_path, "r+") as target:
                mask = target.read(1)
                mask[7, 7] = 255
                target.write(mask, 1)
            with patch.object(compare_threshold, "WaterEnsemble", side_effect=AssertionError("不得构造模型")), \
                 self.assertRaisesRegex(ValueError, "未覆盖共同影像有效域"):
                compare_threshold.run_comparison(output_dir=root / "missing",
                    model_mask_dir=root / "first/masks/model", max_visualizations=0, **options)
            with rasterio.open(mask_path, "r+") as target:
                mask[7, 7] = 1
                target.write(mask, 1)
                target.transform = rasterio.Affine.translation(1, 0)
            with patch.object(compare_threshold, "WaterEnsemble", side_effect=AssertionError("不得构造模型")), \
                 self.assertRaisesRegex(ValueError, "网格不一致"):
                compare_threshold.run_comparison(output_dir=root / "shifted",
                    model_mask_dir=root / "first/masks/model", max_visualizations=0, **options)
            with rasterio.open(mask_path, "r+") as target:
                target.transform = rasterio.Affine.identity()
            # 标签整体翻转以及新增孔洞都不得改变模型/传统方法的预测或强度阈值。
            for name in ("a.tif", "b.tif"):
                changed = np.where(np.isin(truth, (0, 1)), 1 - truth, truth).astype("int16")
                changed[9, 9] = -1
                with rasterio.open(labels / name, "r+") as target:
                    target.write(changed, 1)
            with patch("supervised.evaluation.postprocess.load_model", return_value=DarkWaterModel()), \
                 patch.object(torch.cuda, "is_available", return_value=False):
                second = compare_threshold.run_comparison(output_dir=root / "second",
                                                           max_visualizations=0, **options)
            self.assertEqual(next(r for r in second if r["method"] == "model" and r["subset"] == "all")["f1"], 0)
            for method in complete:
                for name in ("a.tif", "b.tif"):
                    with rasterio.open(root / "first/masks" / method / name) as a, \
                         rasterio.open(root / "second/masks" / method / name) as b:
                        np.testing.assert_array_equal(a.read(), b.read())
            self.assertEqual(read_rows(root / "first/thresholds.csv"), read_rows(root / "second/thresholds.csv"))

    def test_reversed_checkpoint_bands_match_standalone_ensemble(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            images, labels = root / "images", root / "labels"
            images.mkdir()
            labels.mkdir()
            truth = np.zeros((17, 17), dtype="int16")
            truth[5:12, 5:12] = 1
            data = np.full((2, 17, 17), 100, dtype="float32")
            data[1, truth == 1] = 1  # 只有 VH 显示水体，错换成 VV 会全部漏检。
            with rasterio.open(images / "a.tif", "w", driver="GTiff", width=17, height=17,
                               count=2, dtype="float32") as target:
                target.write(data)
            with rasterio.open(labels / "a.tif", "w", driver="GTiff", width=17, height=17,
                               count=1, dtype="int16") as target:
                target.write(truth, 1)
            checkpoint = dict(model_name="reversed", epoch=1, bands=(2, 1),
                              mean=torch.tensor([50., 50.]), std=torch.tensor([20., 20.]),
                              image_dir=str(images), label_dir=str(labels),
                              train_files=[], val_files=["a.tif"], ignore_index=-1)
            path = root / "reversed.pth"
            torch.save(checkpoint, path)
            with patch("supervised.evaluation.postprocess.load_model", return_value=DarkWaterModel()), \
                 patch.object(torch.cuda, "is_available", return_value=False):
                standalone = compare_threshold.WaterEnsemble([path], use_amp=False)
                _, expected = standalone.predict(torch.from_numpy(data[None]),
                                                   torch.ones((1, 17, 17), dtype=torch.bool))
                results = compare_threshold.run_comparison(checkpoint_path=path,
                    output_dir=root / "comparison", scene_csv=None, use_amp=False,
                    max_visualizations=0, save_masks=True, boundary_radii=(1,), boundary_points=8)
            with rasterio.open(root / "comparison/masks/model/a.tif") as source:
                prediction = source.read(1)
            np.testing.assert_array_equal(prediction, (expected[0].numpy() >= .5).astype("uint8"))
            np.testing.assert_array_equal(prediction, truth)
            self.assertEqual(next(row for row in results if row["method"] == "model" and
                                  row["subset"] == "all")["f1"], 1)

    def test_internal_validation_cannot_be_called_independent_test(self):
        with self.assertRaisesRegex(ValueError, "独立区域测试"):
            compare_threshold.run_comparison(independent_regions=True, region_csv="region.csv")

    def test_external_validation_requires_explicit_labeled_data(self):
        with self.assertRaisesRegex(ValueError, "真实标签"):
            compare_threshold.run_comparison(use_saved_val=False)


if __name__ == "__main__":
    unittest.main()
