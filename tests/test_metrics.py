"""用非对称混淆矩阵及缺失类别检查共享像元指标。"""
import math
import sys
import unittest
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"),
                str(Path(__file__).resolve().parents[1] / "scripts")]
from supervised.evaluation.metrics import metrics_from_counts


class PixelMetricTests(unittest.TestCase):
    def test_asymmetric_confusion(self):
        m = metrics_from_counts((7, 2, 4, 5))
        expected = dict(accuracy=12/18, specificity=7/9, false_positive_rate=2/9,
                        false_negative_rate=4/9, false_discovery_rate=2/7,
                        background_iou=7/13, mean_iou=(5/11+7/13)/2,
                        balanced_accuracy=(5/9+7/9)/2, kappa=1/3,
                        mcc=27/math.sqrt(7*9*9*11))
        for key, value in expected.items():
            with self.subTest(key=key):
                self.assertAlmostEqual(m[key], value)
        self.assertEqual(m["valid_pixels"], 18)

    def test_perfect_and_inverted_predictions(self):
        m = metrics_from_counts((8, 0, 0, 2))
        for key in ("iou", "f1", "precision", "recall", "accuracy", "specificity",
                    "background_iou", "mean_iou", "balanced_accuracy", "kappa", "mcc"):
            self.assertEqual(m[key], 1)
        self.assertEqual(metrics_from_counts((0, 5, 5, 0))["mcc"], -1)

    def test_absent_classes_and_empty_valid_area(self):
        m = metrics_from_counts((10, 0, 0, 0))
        self.assertEqual(m["accuracy"], 1)
        self.assertEqual(m["precision"], 0)  # 保持既有零分母约定。
        for key in ("recall", "iou", "f1", "mean_iou", "balanced_accuracy", "kappa", "mcc", "false_discovery_rate"):
            self.assertTrue(math.isnan(m[key]), key)
        m = metrics_from_counts((0, 0, 0, 0))
        self.assertEqual(m["valid_pixels"], 0)
        self.assertTrue(math.isnan(m["accuracy"]))


if __name__ == "__main__":
    unittest.main()
