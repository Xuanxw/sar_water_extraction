"""验证固定模型入口、混淆计数聚合与重复样本隔离。"""
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
from supervised import validation as v
class ValidationTests(unittest.TestCase):
    def test_saved_validation_subdirectory_checks_training_sibling(self):
        # 物理 val 目录不包含 train_files，训练去重核查必须回到共同 images 根。
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "split_manifest.csv").write_text("fixed split", encoding="utf-8")
            for split, value in (("train", 0), ("val", 1)):
                for kind in ("images", "labels"):
                    directory = root / kind / split
                    directory.mkdir(parents=True)
                    with rasterio.open(directory / (split + ".tif"), "w", driver="GTiff",
                                       width=4, height=4, count=1, dtype="uint8") as dst:
                        dst.write(np.full((4, 4), value, dtype=np.uint8), 1)
            checkpoint = dict(bands=(1,), image_dir=str(root / "images"),
                              train_files=["train.tif"], val_files=["val.tif"])
            manifest = v.build_manifest(checkpoint, root / "images/val",
                                        root / "labels/val", True)
            self.assertEqual(len(manifest), 1)
            self.assertEqual(manifest[0]["name"], "val.tif")
            self.assertEqual(manifest[0]["duplicate_in_train"], 0)
            self.assertEqual(manifest[0]["unique_nontrain"], 1)

    def test_pooling_is_not_macro_and_empty_water_is_undefined(self):
        rows = [dict(name="a", **v.metrics_from_counts((80, 10, 5, 5))),
                dict(name="b", **v.metrics_from_counts((2, 0, 0, 8))),
                dict(name="dry", **v.metrics_from_counts((30, 0, 0, 0)))]
        result = v.summarize(rows)
        self.assertAlmostEqual(result["iou"], 13 / 28)
        self.assertAlmostEqual(result["macro_iou"], (0.25 + 1) / 2)
        self.assertEqual(result["defined_iou"], 2)
        self.assertEqual(result["worst_iou_name"], "a")

    def test_cpu_pipeline_normalization_duplicate_groups_and_boundary(self):
        class Model(torch.nn.Module):
            def forward(self, image):
                # 使用标准化后的第一通道；末头决定预测，首头故意相反。
                return -image[:, :1], image[:, :1]
        with TemporaryDirectory() as temp:
            root = Path(temp)
            images, labels = root / "images", root / "labels"
            images.mkdir(); labels.mkdir()
            truth = np.zeros((16, 16), dtype=np.uint8)
            truth[:, 8:] = 1
            for name, second in (("train", 0), ("a", 0), ("b", 1), ("c", 1)):
                data = np.stack((truth * 2, np.full_like(truth, second))).astype("float32")
                with rasterio.open(images / (name + ".tif"), "w", driver="GTiff",
                                   width=16, height=16, count=2, dtype="float32") as dst:
                    dst.write(data)
                with rasterio.open(labels / (name + ".tif"), "w", driver="GTiff",
                                   width=16, height=16, count=1, dtype="uint8") as dst:
                    dst.write(truth, 1)
            ckpt = root / "best.pth"
            torch.save(dict(model_name="test", epoch=2, bands=(1, 2),
                            mean=torch.tensor([1., 0.]), std=torch.tensor([2., 1.]),
                            image_dir=str(images), label_dir=str(labels), train_files=["train.tif"],
                            val_files=["a.tif", "b.tif", "c.tif"]), ckpt)
            with patch("supervised.evaluation.postprocess.load_model", return_value=Model()), \
                 patch.object(torch.cuda, "is_available", return_value=False):
                out = v.run_validation({"test": ckpt}, output_dir=root / "out", scene_csv=None,
                                       use_amp=False, save_masks=True, max_visualizations=1)
            summary = {r["subset"]: r for r in v.read_csv(out / "comparison.csv")}
            self.assertEqual(int(summary["all"]["tp"]), 3 * 128)
            self.assertEqual(float(summary["all"]["f1"]), 1.)
            self.assertEqual(int(summary["duplicate_in_train"]["samples"]), 1)
            self.assertEqual(int(summary["unique_nontrain"]["samples"]), 1)
            self.assertTrue((out / "test/boundary_groups.csv").exists())
            self.assertTrue((out / "test/figures/a.png").exists())
            self.assertFalse((out / "test/region_bootstrap.csv").exists())
            # 同一输入的两份标签发生冲突时，不能任意选其中一份纳入去重组。
            with rasterio.open(labels / "c.tif", "r+") as dst:
                changed = truth.copy()
                changed[0, 0] = 1
                dst.write(changed, 1)
            manifest = v.build_manifest(torch.load(ckpt, weights_only=True), images, labels, True)
            self.assertEqual(sum(r["conflicting_duplicate_labels"] for r in manifest), 2)
            self.assertEqual(sum(r["unique_nontrain"] for r in manifest), 0)
            with rasterio.open(out / "test/raw/b.tif") as src:
                np.testing.assert_array_equal(src.read(1), truth)

    def test_region_pooling_and_bootstrap_use_regions(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            rows = [dict(name=name, **v.metrics_from_counts(counts)) for name, counts in
                    (("a", (10, 1, 2, 4)), ("b", (20, 3, 1, 8)), ("c", (2, 0, 0, 3)))]
            manifest = [dict(name=r["name"], duplicate_in_train=0, validation_copies=1,
                             unique_nontrain=1, scene="unknown") for r in rows]
            region_csv = root / "regions.csv"
            v.write_csv(region_csv, [dict(name="a", region_id="R1"),
                                     dict(name="b", region_id="R1"), dict(name="c", region_id="R2")])
            v.save_reports(root, rows, manifest, region_csv, True, bootstrap_repeats=100)
            regions = v.read_csv(root / "per_region.csv")
            self.assertEqual(len(regions), 2)
            self.assertEqual(int(regions[0]["tp"]), 12)
            bootstrap = v.read_csv(root / "region_bootstrap.csv")[0]
            self.assertEqual(int(bootstrap["regions"]), 2)
            self.assertLessEqual(float(bootstrap["ci95_low"]), 15 / 22)
            self.assertGreaterEqual(float(bootstrap["ci95_high"]), 15 / 22)
            manifest[0]["duplicate_in_train"] = 1
            with self.assertRaisesRegex(ValueError, "整幅重复"):
                v.save_reports(root, rows, manifest, region_csv, True, bootstrap_repeats=100)

    def test_internal_validation_cannot_claim_independent_test(self):
        with self.assertRaisesRegex(ValueError, "内部验证集"):
            v.run_validation(independent_regions=True)


if __name__ == "__main__":
    unittest.main()
