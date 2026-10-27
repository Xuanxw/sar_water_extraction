"""独立阈值流程：标签不参与分割，支持无标签、单波段和文件有效掩膜。"""
import csv
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from thres_seg.threshold import ThresholdSegmenter, percentile_mask
from thres_seg.pipeline import run_segmentation


class ThresholdTests(unittest.TestCase):
    def test_percentile_direction_ties_and_invalid_values(self):
        score = np.array([0., 0., .2, .8, 1., np.nan])
        valid = np.isfinite(score)
        lower, threshold = percentile_mask(score, valid, 40, "bottom")
        np.testing.assert_array_equal(lower, [True, True, False, False, False, False])
        upper, threshold = percentile_mask(score, valid, 40, "top")
        np.testing.assert_array_equal(upper, [False, False, False, True, True, False])

    def test_valid_zero_kept_constant_and_empty_image(self):
        segmenter = ThresholdSegmenter(lee_window=1)
        image = np.zeros((8, 8))
        valid = np.ones((8, 8), dtype=bool)
        pred, score, _ = segmenter.predict(image, valid)
        self.assertTrue(np.isfinite(score).all())
        self.assertFalse(pred.any())
        pred, score, threshold = segmenter.predict(image, ~valid)
        self.assertFalse(pred.any())
        self.assertTrue(np.isnan(score).all())
        self.assertTrue(np.isnan(threshold))

    def test_labeled_and_unlabeled_raster_pipeline_needs_no_model(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            images, labels = root / "images", root / "labels"
            images.mkdir(); labels.mkdir()
            label = np.zeros((17, 17), dtype=np.int16)
            label[5:12, 5:12] = 1
            data = np.ones((6, 17, 17), dtype=np.int16) * 100
            data[:2, label == 1] = 1
            data[:2, 5, 5] = 0  # 合法的 SAR 零值仍是有效像素。
            data[:2, 0, 0] = -9999
            data[4] = 10
            label[2, 2] = -1  # 标签无效不应改变分割输入或输出掩膜。
            for name in ("a.tif", "b.tif"):
                with rasterio.open(images / name, "w", driver="GTiff", width=17, height=17,
                                   count=6, dtype="int16", nodata=-9999) as target:
                    target.write(data)
                with rasterio.open(labels / name, "w", driver="GTiff", width=17, height=17,
                                   count=1, dtype="int16", nodata=-1) as target:
                    target.write(label, 1)
            options = dict(image_dir=images, label_dir=labels, save_masks=True,
                           threshold_options=dict(lee_window=1, water_percent=25), max_visualizations=1)
            first = run_segmentation(output_dir=root / "first", **options)
            self.assertEqual(first["threshold_VV"]["f1"], 1)
            self.assertEqual(first["threshold_VV"]["valid_pixels"], 2 * (289 - 2))
            self.assertTrue((root / "first/boundary/figures/a.png").exists())
            for name in ("a.tif", "b.tif"):
                with rasterio.open(labels / name, "r+") as target:
                    target.write(np.where(label == -1, -1, 1 - label).astype(np.int16), 1)
            options["max_visualizations"] = 0
            second = run_segmentation(output_dir=root / "second", **options)
            self.assertEqual(second["threshold_VV"]["f1"], 0)
            with rasterio.open(root / "first/masks/threshold_VV/a.tif") as a, rasterio.open(root / "second/masks/threshold_VV/a.tif") as b:
                np.testing.assert_array_equal(a.read(), b.read())
                self.assertEqual(a.read(1)[0, 0], 255)
                self.assertEqual(a.read(1)[5, 5], 1)
                self.assertEqual(a.read(1)[2, 2], 0)
            with (root / "first/files.csv").open(encoding="utf-8-sig") as stream:
                self.assertEqual([r["name"] for r in csv.DictReader(stream)], ["a.tif", "b.tif"])
            with (root / "first/boundary/points.csv").open(encoding="utf-8-sig") as stream:
                self.assertEqual(next(csv.DictReader(stream))["threshold_VV_probability"], "")
            # 真实的单波段文件，没有标签目录，也没有辅助土地覆盖层。
            single = root / "single"
            single.mkdir()
            with rasterio.open(single / "s.tif", "w", driver="GTiff", width=17, height=17,
                               count=1, dtype="int16", nodata=-9999) as target:
                target.write(data[0], 1)
            metrics = run_segmentation(single, bands={"VV": 1}, output_dir=root / "unlabeled",
                                       threshold_options=dict(lee_window=1), use_landcover=False,
                                       max_visualizations=1)
            self.assertEqual(metrics, {})
            self.assertFalse((root / "unlabeled/metrics.csv").exists())
            self.assertFalse((root / "unlabeled/boundary").exists())
            self.assertTrue((root / "unlabeled/figures/s.png").exists())
            self.assertTrue((root / "unlabeled/masks/threshold_VV/s.tif").exists())
            with (root / "unlabeled/per_image.csv").open(encoding="utf-8-sig") as stream:
                row = next(csv.DictReader(stream))
            self.assertIn("pred_water_ratio", row)
            self.assertNotIn("f1", row)
            self.assertEqual(int(row["image_valid_pixels"]), 288)


if __name__ == "__main__":
    unittest.main()
