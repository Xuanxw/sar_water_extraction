"""用可手算的岸线与无效孔洞检验边界语义，不以实现自身生成期望结果。"""
import csv
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluation.boundary_eval import BoundaryEvaluator, boundary_regions, confusion_counts, sample_region


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.label = np.zeros((15, 15), dtype=np.int64)
        self.label[5:10, 5:10] = 1
        self.valid = np.ones_like(self.label, dtype=bool)

    def test_square_inner_outer_core_counts_and_error_direction(self):
        areas = boundary_regions(self.label, self.valid, (1,))[1]
        self.assertEqual({k: int(v.sum()) for k, v in areas.items()}, dict(inner=16, outer=20, core=9))
        prediction = self.label.astype(bool)
        np.testing.assert_array_equal(confusion_counts(self.label, prediction, areas["inner"] | areas["outer"]), (20, 0, 0, 16))
        expanded = prediction | areas["outer"]
        np.testing.assert_array_equal(confusion_counts(self.label, expanded, areas["outer"]), (0, 20, 0, 0))
        eroded = areas["core"]
        np.testing.assert_array_equal(confusion_counts(self.label, eroded, areas["inner"]), (0, 0, 16, 0))

    def test_nodata_and_image_edge_never_become_shoreline(self):
        self.label[:] = 1
        self.valid[7, 7] = False
        areas = boundary_regions(self.label, self.valid, (1,))[1]
        self.assertFalse(areas["inner"].any())
        self.assertFalse(areas["outer"].any())
        self.assertFalse(areas["core"][0].any())
        self.assertFalse(areas["core"][7, 6])
        self.assertTrue(areas["core"][5, 7])
        empty = boundary_regions(np.zeros_like(self.label), self.valid)
        self.assertFalse(any(mask.any() for areas in empty.values() for mask in areas.values()))

    def test_points_fixed_and_full_counts_distinct_from_sample_counts(self):
        region = boundary_regions(self.label, self.valid, (1,))[1]["inner"]
        first = sample_region(region, "square.tif", 1, "inner", max_points=5)
        np.testing.assert_array_equal(first, sample_region(region, "square.tif", 1, "inner", max_points=5))
        self.assertEqual(len(np.unique(first, axis=0)), 5)
        with TemporaryDirectory() as folder:
            evaluator = BoundaryEvaluator(folder, radii=(1,), max_points=5)
            evaluator.update("square.tif", self.label, self.valid,
                             dict(perfect=self.label.astype(bool), missed=np.zeros_like(self.valid)))
            rows = evaluator.save()
            inner = next(r for r in rows if r["method"] == "missed" and r["scope"] == "all_pixels" and r["region"] == "inner")
            self.assertEqual(inner["fn"], 16)
            self.assertEqual(inner["recall"], 0)
            with (Path(folder) / "points.csv").open(encoding="utf-8-sig") as stream:
                points = list(csv.DictReader(stream))
            self.assertEqual(len(points), 15)
            self.assertEqual(points[0]["perfect_prediction"], points[0]["gt"])
            self.assertTrue(points[0]["label_sha256"])


if __name__ == "__main__":
    unittest.main()
