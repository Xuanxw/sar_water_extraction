"""用已知混淆矩阵检验独立评价：最后输出、固定划分、忽略像素及两种输出形式。"""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"),
                str(Path(__file__).resolve().parents[1] / "scripts")]
from supervised.evaluation import model_eval as evaluate


class EvaluationTests(unittest.TestCase):
    def test_histogram_scan_matches_direct_comparison_and_includes_equal_scores(self):
        score = torch.tensor([[-2., 0., 2., -4.], [2., 2., 0., 0.]])
        label = torch.tensor([[0, 1, 0, 1], [1, -1, 0, -1]])
        thresholds = [.2, .5, .8]
        actual = evaluate.threshold_counts(score, label, thresholds)
        for i, t in enumerate(thresholds):
            prediction = score >= torch.logit(torch.tensor(t))
            valid = label != -1
            expected = torch.bincount(label[valid] * 2 + prediction[valid].long(), minlength=4)
            torch.testing.assert_close(actual[i], expected)
        # 全忽略批次不凭空增加 TN/FP/FN/TP。
        self.assertEqual(evaluate.threshold_counts(score, torch.full_like(label, -1), thresholds).sum().item(), 0)

    def test_precision_requirement_uses_highest_recall_and_marks_unreachable(self):
        rows = [dict(threshold=t, precision=p, recall=r, f1=.7, fp=10, fn=20, tp=30)
                for t, p, r in ((.3, .79, .9), (.4, .86, .8), (.5, .85, .7), (.6, .91, .6))]
        # 没有预测像素不能作为满足精确率要求的解。
        rows.append(dict(threshold=.99, precision=1., recall=0., f1=0., fp=0, fn=50, tp=0))
        matches = evaluate.precision_matches(rows, [.85, .90, .95])
        self.assertEqual(matches[0]["threshold"], .4)
        self.assertEqual(matches[1]["threshold"], .6)
        self.assertEqual(matches[2]["status"], "unreachable")

    def test_known_confusion_matrix_for_binary_and_two_class_models(self):
        class Dataset(torch.utils.data.Dataset):
            def __init__(self, image_dir, label_dir, bands, speckle=None):
                self.image_paths = [Path(image_dir) / "a.tif", Path(image_dir) / "b.tif"]

            def select_files(self, names):
                paths = {p.name: p for p in self.image_paths}
                self.image_paths = [paths[name] for name in names]

            def __len__(self):
                return len(self.image_paths)

            def __getitem__(self, index):
                self_path = self.image_paths[index]
                # 验证确实只使用检查点记录的 b.tif，没有评价训练文件 a.tif。
                assert self_path.name == "b.tif"
                return dict(image=torch.tensor([[[2., -2., 2.], [-2., 2., 2.]]]),
                            label=torch.tensor([[1, 1, 0], [0, 1, -1]]),
                            valid_mask=torch.ones(2, 3, dtype=torch.bool), name="b.tif")

        class Model(torch.nn.Module):
            def __init__(self, channels):
                super().__init__()
                self.channels = channels

            def forward(self, image):
                final = image if self.channels == 1 else torch.cat((-image, image), dim=1)
                return -final, final  # 早期输出故意相反，确保只用最终预测。

        for channels in (1, 2):
            with self.subTest(channels=channels):
                checkpoint = dict(epoch=1, bands=(1,), image_dir="images", label_dir="labels",
                                  batch_size=1, amp_enabled=False, val_files=["b.tif"],
                                  mean=torch.zeros(1), std=torch.ones(1),
                                  model_kwargs=dict(num_classes=channels))
                with patch.object(evaluate.torch, "load", return_value=checkpoint), \
                     patch("datasets.dfc_sample.s1_dataset.WaterDataset", Dataset), \
                     patch.object(evaluate, "load_model", return_value=Model(channels)), \
                     patch.object(torch.cuda, "is_available", return_value=False):
                    metrics = evaluate.evaluate("unused.pth")
                    scanned = evaluate.evaluate("unused.pth", thresholds=[.2, .5, .8])
                self.assertEqual([metrics[k] for k in ("tn", "fp", "fn", "tp")], [1, 1, 1, 2])
                self.assertEqual(metrics["valid_pixels"], 5)
                self.assertEqual(metrics["ignored_pixels"], 1)
                self.assertAlmostEqual(metrics["f1"], 2 / 3)
                self.assertAlmostEqual(metrics["iou"], 0.5)
                self.assertEqual([scanned[1][k] for k in ("tn", "fp", "fn", "tp")], [1, 1, 1, 2])


if __name__ == "__main__":
    unittest.main()
