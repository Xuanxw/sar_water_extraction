"""固定分组训练的合成验收：目录配对、停用副本、训练归一化、随机序列和检查点。"""
import csv
import hashlib
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_origin
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import train
from datasets.water_dataset import WaterDataset, resolve_sample_path


class TinyModel(nn.Module):
    """一层卷积足够检查训练数据流，extra_random 模拟新增模型层的随机初始化。"""
    def __init__(self, extra_random=0):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 1)
        if extra_random:
            torch.randn(extra_random)

    def forward(self, image):
        return self.conv(image)


class GroupedTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def make_data(self, folder):
        """实际写出小 TIFF/PNG；副本和冲突记录保留在物理目录，仅由清单停用。"""
        root = Path(folder) / "data"
        # 清单顺序故意不同于目录排序，以检查读取的是固定名单。
        records = [
            ("c", "train", 1, "g3", 10., 0),
            ("a", "train", 1, "g1", 2., 1),
            ("a_copy", "train", 0, "g1", 2., 1),
            ("b", "train", 1, "g2", 6., 0),
            ("conflict", "train", 0, "g_bad", 9000., 1),
            ("v", "val", 1, "g4", 100., 1),
            ("v_copy", "val", 0, "g4", 100., 1),
        ]
        rows = []
        for name, split, active, group, value, marker in records:
            image_dir, label_dir = root / "images" / split, root / "labels" / split
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            image = np.stack([np.full((8, 8), value, dtype=np.float32),
                              np.full((8, 8), value * 10, dtype=np.float32)])
            with rasterio.open(image_dir / f"{name}.tif", "w", driver="GTiff", width=8, height=8,
                               count=2, dtype="float32", transform=from_origin(10, 20, 1, 1)) as raster:
                raster.write(image)
            label = ((np.indices((8, 8)).sum(axis=0) + marker) % 2).astype(np.uint8)
            Image.fromarray(label).save(label_dir / f"{name}.png")
            rows.append(dict(name=f"{name}.tif", split=split, active=active, group_id=group))
        manifest = root / "split_manifest.csv"
        with manifest.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("name", "split", "active", "group_id"))
            writer.writeheader()
            writer.writerows(rows)
        return root

    def make_trainer(self, root, seed=42, extra_random=0):
        with patch.object(torch.cuda, "is_available", return_value=False), \
             patch.object(train, "cprint"), \
             patch.object(train, "random_split", side_effect=AssertionError("固定清单不得重新随机划分")):
            return train.WaterSegmentationTrainer(
                model_class=TinyModel, model_kwargs=dict(extra_random=extra_random),
                model_info=dict(model_name="tiny"), image_dir=root / "images", label_dir=root / "labels",
                output_dir=root.parent / "runs", bands=(1, 2), batch_size=2, epochs=1,
                loss_name="bce", seed=seed, val_ratio=0.5, sampling="random",
            )

    def test_physical_partitions_pair_labels_and_keep_old_logical_names(self):
        with TemporaryDirectory() as folder:
            root = self.make_data(folder)
            dataset = WaterDataset(root / "images", root / "labels", (1, 2))
            dataset.select_files(["v.tif", "a.tif", "b.tif"])
            self.assertEqual([dataset[i]["name"] for i in range(3)], ["v.tif", "a.tif", "b.tif"])
            for index, (value, marker) in enumerate(((100, 1), (2, 1), (6, 0))):
                sample = dataset[index]
                self.assertEqual(sample["image"][0, 0, 0].item(), value)
                self.assertEqual(sample["image"][1, 0, 0].item(), value * 10)
                self.assertEqual(sample["label"][0, 0].item(), marker)
            self.assertEqual(resolve_sample_path(root / "images", "v.tif"),
                             root / "images/val/v.tif")
            dataset.select_files(["train/a.tif", "val/v.tif"])
            self.assertEqual([dataset[i]["name"] for i in range(2)], ["a.tif", "v.tif"])

    def test_manifest_ignores_inactive_and_normalization_uses_only_training(self):
        with TemporaryDirectory() as folder:
            root = self.make_data(folder)
            for seed in (42, 17):
                trainer = self.make_trainer(root, seed=seed)
                self.assertEqual(trainer.train_files, ["c.tif", "a.tif", "b.tif"])
                self.assertEqual(trainer.val_files, ["v.tif"])
                self.assertEqual(len(trainer.dataset), 7)  # 没有删除物理副本。
                self.assertEqual((len(trainer.train_set), len(trainer.val_set)), (3, 1))
                torch.testing.assert_close(trainer.mean.flatten(), torch.tensor([6., 60.]))
                expected_std = torch.tensor([np.std([2., 6., 10.]), np.std([20., 60., 100.])],
                                            dtype=torch.float32)
                torch.testing.assert_close(trainer.std.flatten(), expected_std)
                self.assertEqual(trainer.output_dir.name, "runs_grouped")

    def test_checkpoint_records_logical_files_and_exact_manifest_snapshot(self):
        with TemporaryDirectory() as folder:
            root = self.make_data(folder)
            trainer = self.make_trainer(root)
            with patch.object(train, "cprint"):
                trainer.train()  # 只有 3 个 8×8 合成训练样本和 1 个合成验证样本。
            checkpoint = torch.load(trainer.run_dir / "last.pth", weights_only=True)
            self.assertEqual(checkpoint["train_files"], ["c.tif", "a.tif", "b.tif"])
            self.assertEqual(checkpoint["val_files"], ["v.tif"])
            self.assertFalse(set(checkpoint["train_files"]) & set(checkpoint["val_files"]))
            self.assertEqual(checkpoint["split_strategy"], "scene_stratified_grouped")
            source = (root / "split_manifest.csv").read_bytes()
            snapshot = (trainer.run_dir / "split_manifest.csv").read_bytes()
            self.assertEqual(snapshot, source)
            self.assertEqual(checkpoint["split_manifest_sha256"], hashlib.sha256(snapshot).hexdigest())
            self.assertEqual(checkpoint["split_manifest"], str(root / "split_manifest.csv"))
            torch.testing.assert_close(checkpoint["mean"], torch.tensor([6., 60.]))
            self.assertTrue(set(checkpoint["recent_train_files"]) <= set(checkpoint["train_files"]))

    def test_additional_model_initialization_keeps_sampling_random_sequence(self):
        with TemporaryDirectory() as folder:
            root = self.make_data(folder)
            states, orders, next_random = [], [], []
            for extra_random in (0, 10000):
                trainer = self.make_trainer(root, extra_random=extra_random)
                states.append(torch.get_rng_state().clone())
                orders.append([name for batch in trainer.train_loader for name in batch["name"]])
                next_random.append(torch.rand(8))
            torch.testing.assert_close(states[0], states[1], rtol=0, atol=0)
            self.assertEqual(orders[0], orders[1])
            self.assertEqual(set(orders[0]), {"a.tif", "b.tif", "c.tif"})
            torch.testing.assert_close(next_random[0], next_random[1], rtol=0, atol=0)

    def test_highres_flag_changes_model_kwargs_and_separates_experiment(self):
        choices = []
        for enabled in (False, True):
            with patch.object(train, "METHOD", "uafnet"), \
                 patch.object(train, "UAFNET_HIGH_RESOLUTION_REFINEMENT", enabled), \
                 patch.object(train, "UAFNET_MBDC_DILATIONS", None), \
                 patch.object(train, "UAFNET_FINAL_RESIDUAL", True), \
                 patch.object(train, "LOSS_NAME", "bce"), patch.object(train, "SAMPLING", "hard"), \
                 patch.object(train, "HEAD_LOSS_WEIGHTS", (1., 1., 1., 1.)), \
                 patch.object(train, "POLARIZATIONS", ("VV", "VH")), \
                 patch.object(train, "POLARIZATION_BANDS", {"VV": 1, "VH": 2}), \
                 patch.object(train, "UAFNET_BACKBONE", "pvt_v2_b2.in1k"), \
                 patch.object(train, "OUTPUT_DIR", None), patch.object(train, "cprint"), \
                 patch.object(train, "WaterSegmentationTrainer") as factory:
                train.create_trainer()
                choices.append(factory.call_args.kwargs)
        baseline, refined = choices
        self.assertNotIn("high_resolution_refinement", baseline["model_kwargs"])
        refined_options = dict(refined["model_kwargs"])
        self.assertTrue(refined_options.pop("high_resolution_refinement"))
        self.assertEqual(refined_options, baseline["model_kwargs"])
        self.assertEqual(baseline["output_dir"].name, "pvt_uafnet_hard_sampling_residual")
        self.assertEqual(refined["output_dir"].name, "pvt_uafnet_hard_sampling_residual_highres")
        for key in ("learning_rate", "epochs", "batch_size", "head_loss_weights", "sampling", "loss_name"):
            self.assertEqual(baseline[key], refined[key], key)

    def test_partitioned_data_without_manifest_does_not_fall_back_to_random(self):
        with TemporaryDirectory() as folder:
            root = self.make_data(folder)
            (root / "split_manifest.csv").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "split_manifest.csv"):
                self.make_trainer(root)


if __name__ == "__main__":
    unittest.main()
