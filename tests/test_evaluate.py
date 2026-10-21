"""用已知混淆矩阵检验独立评价：最后输出、固定划分、忽略像素及两种输出形式。"""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import evaluate


class EvaluationTests(unittest.TestCase):
    def test_known_confusion_matrix_for_binary_and_two_class_models(self):
        class Dataset(torch.utils.data.Dataset):
            def __init__(self, image_dir, label_dir, bands):
                self.image_paths = [Path(image_dir) / "a.tif", Path(image_dir) / "b.tif"]

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
                     patch.object(evaluate, "WaterDataset", Dataset), \
                     patch.object(evaluate, "load_model", return_value=Model(channels)), \
                     patch.object(torch.cuda, "is_available", return_value=False):
                    metrics = evaluate.evaluate("unused.pth")
                self.assertEqual([metrics[k] for k in ("tn", "fp", "fn", "tp")], [1, 1, 1, 2])
                self.assertEqual(metrics["valid_pixels"], 5)
                self.assertEqual(metrics["ignored_pixels"], 1)
                self.assertAlmostEqual(metrics["f1"], 2 / 3)
                self.assertAlmostEqual(metrics["iou"], 0.5)


if __name__ == "__main__":
    unittest.main()
