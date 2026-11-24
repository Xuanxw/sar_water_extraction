"""独立 S1 BN 对照回归：统计域、Dropout、固定划分、阈值边界及中断续跑。"""

from copy import deepcopy
import csv
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from zipfile import ZipFile
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.transform import from_origin
import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from scripts.models import calibrate_s1_unet_bn as experiment


def small_model(checkpoint=None):
    model = nn.Sequential(nn.Conv2d(2, 3, 1), nn.BatchNorm2d(3), nn.ReLU(),
                          nn.Dropout(0.5), nn.Conv2d(3, 1, 1), nn.BatchNorm2d(1))
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
    return model


def write_tiff(path, values, nodata):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", count=len(values), height=values.shape[1],
                       width=values.shape[2], dtype=str(values.dtype), nodata=nodata,
                       crs="EPSG:32650", transform=from_origin(500000, 4000000, 10, 10)) as dst:
        dst.write(values)


class BNScanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_sequential_moments_dropout_and_layer_resume(self):
        torch.manual_seed(9)
        model = small_model()
        original = deepcopy(model.state_dict())
        samples = [dict(image=torch.randn(2, 8, 8) + index, valid_mask=torch.ones(8, 8, dtype=torch.bool),
                        name=str(index)) for index in range(3)]
        samples[0]["valid_mask"][0, 0] = False
        checkpoint = dict(mean=[0.3, 2.0], std=[0.2, 3.0], inputs_normalized=False)
        loader = DataLoader(samples, batch_size=2)
        expected = deepcopy(model).eval()
        # 独立参照逐层收集完整特征并直接求总体均值/方差，包含最后一个不足整批的样本。
        for layer in (item for item in expected.modules() if isinstance(item, nn.BatchNorm2d)):
            features = []
            handle = layer.register_forward_pre_hook(
                lambda module, inputs: features.append(inputs[0].detach().double().movedim(1, 0).flatten(1)))
            with torch.no_grad():
                for batch in loader:
                    expected(experiment.model_input(batch, checkpoint, "cpu"))
            handle.remove()
            values = torch.cat(features, dim=1)
            layer.running_mean.copy_(values.mean(dim=1))
            layer.running_var.copy_(values.var(dim=1, unbiased=False))

        with TemporaryDirectory() as tmp:
            calls = 0
            original_call = experiment.ChannelStatistics.__call__

            def interrupt_second_layer(statistics, module, inputs):
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise RuntimeError("模拟第二层中断")
                return original_call(statistics, module, inputs)

            with patch.object(experiment.ChannelStatistics, "__call__", interrupt_second_layer):
                with self.assertRaisesRegex(RuntimeError, "模拟第二层中断"):
                    experiment.calibrate_layers(model, loader, checkpoint, "cpu", Path(tmp), 5)
            saved = torch.load(Path(tmp) / "calibration_progress.pth", weights_only=True)
            self.assertEqual(saved["completed_layers"], ["1"])
            resumed = small_model()
            resumed.load_state_dict(original)
            experiment.calibrate_layers(resumed, loader, checkpoint, "cpu", Path(tmp), 5)
            for name, value in resumed.state_dict().items():
                torch.testing.assert_close(value, expected.state_dict()[name])
            for name, value in resumed.named_parameters():
                self.assertTrue(torch.equal(value, original[name]))
            self.assertTrue(all(not module.training for module in resumed.modules()))

    def test_threshold_edges_masks_and_interrupted_patch_resume(self):
        thresholds = [0.1, 0.5, 0.9]
        edges = torch.logit(torch.tensor(thresholds))
        score = torch.tensor([[-8., edges[0], edges[1]], [edges[2], 8., 0.]])
        label = torch.tensor([[0, 1, 1], [0, 1, -1]])
        actual = experiment.threshold_counts(score, label, thresholds)
        expected = torch.stack([torch.bincount(2 * label[label >= 0]
                                               + (score[label >= 0] >= edge), minlength=4) for edge in edges])
        torch.testing.assert_close(actual, expected)

        class Samples(torch.utils.data.Dataset):
            image_dir = Path(".")
            image_paths = [Path("a.tif"), Path("b.tif")]

            def __len__(self):
                return 2

            def __getitem__(self, index):
                return dict(image=score[None], label=label.clone(), valid_mask=label >= 0,
                            name=self.image_paths[index].name)

        checkpoint = dict(inputs_normalized=True, ignore_index=-1)
        with TemporaryDirectory() as tmp:
            with patch.object(experiment, "threshold_counts", side_effect=[actual, RuntimeError("中断")]):
                with self.assertRaisesRegex(RuntimeError, "中断"):
                    experiment.scan_validation(nn.Identity(), Samples(), checkpoint, thresholds,
                                               "cpu", Path(tmp), 1, 5)
            self.assertTrue((Path(tmp) / "00000.json").exists())
            self.assertFalse((Path(tmp) / "00001.json").exists())
            with patch.object(experiment, "threshold_counts", wraps=experiment.threshold_counts) as count:
                names, values = experiment.scan_validation(nn.Identity(), Samples(), checkpoint,
                                                          thresholds, "cpu", Path(tmp), 1, 5)
                self.assertEqual(count.call_count, 1)
            self.assertEqual(names, ["a.tif", "b.tif"])
            np.testing.assert_array_equal(values, np.stack([actual.numpy()] * 2))

    def prepare_run(self, root):
        torch.manual_seed(12)
        images, labels = root / "images", root / "labels"
        for number in range(5):
            values = np.arange(4 * 8 * 8, dtype="float32").reshape(4, 8, 8) + number * 10
            values[:, 0, 0] = np.nan
            write_tiff(images / f"{number}.tif", values, np.nan)
            if number >= 3:
                water = (np.indices((8, 8))[1] > 4).astype("uint8")[None]
                water[:, 0, 1] = 255
                write_tiff(labels / f"{number}.tif", water, 255)
        checkpoint = dict(model_name="sentinel1_water_unet", model_kwargs={}, model_state_dict=small_model().state_dict(),
                          image_dir=str(images), label_dir=str(labels), bands=(1, 4), input_channels=("VV", "DEM"),
                          patch_representation="vv_rank_dem_m", mean=torch.tensor([0.5, 210.]),
                          std=torch.tensor([0.3, 30.]), inputs_normalized=False, ignore_index=-1,
                          train_files=[f"{i}.tif" for i in range(3)], val_files=["3.tif", "4.tif"],
                          phase_metrics={"val": {"f1": 0.123}}, optimizer_state_dict={"state": "old"})
        paths = {}
        for epoch in (6, 20):
            checkpoint["epoch"] = epoch
            path = root / f"epoch_{epoch:03d}.pth"
            torch.save(checkpoint, path)
            paths[f"epoch_{epoch:03d}"] = str(path)
        config = dict(paths=dict(DATA_ROOT=str(root), CHECKPOINTS=paths, OUTPUT_DIR=str(root / "comparison")),
                      run=dict(EXPECTED_EPOCHS={"epoch_006": 6, "epoch_020": 20}, DEVICE="cpu",
                               CPU_THREADS=1, BATCH_SIZE=2, CALIBRATION_SAMPLES=None, SEED=42, LOG_EVERY=5,
                               THRESHOLDS=dict(start=0.1, stop=0.9, step=0.1)))
        path = root / "config.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return path, config

    def test_full_entrypoint_four_models_reports_and_resume(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path, config = self.prepare_run(root)
            source_hashes = {name: experiment.sha256(path) for name, path in config["paths"]["CHECKPOINTS"].items()}
            with patch.object(experiment, "load_model", side_effect=small_model), \
                    patch.object(torch.cuda, "is_available", side_effect=AssertionError("CPU 路径不能自动选择 CUDA")):
                output = experiment.main(config_path)
            protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
            self.assertEqual(protocol["calibration_files"], ["0.tif", "1.tif", "2.tif"])
            self.assertTrue(all(row["split"] == "val" for row in protocol["files"] if row["kind"] == "label"))
            with (output / "summary.csv").open(encoding="utf-8-sig") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 8)
            self.assertEqual({int(row["valid_pixels"]) for row in rows}, {124})
            self.assertEqual(len({row["model"] for row in rows}), 4)
            saved = torch.load(output / "epoch_006/calibrated.pth", weights_only=True)
            self.assertNotIn("optimizer_state_dict", saved)
            self.assertEqual(saved["resume_phase"], "bn_calibrated_inference_only")
            self.assertEqual(saved["phase_metrics"], {})
            self.assertTrue((output / "completed.json").exists())
            before_csv = (output / "summary.csv").read_bytes()
            with patch.object(experiment, "load_model", side_effect=small_model), \
                    patch.object(experiment, "calibrate_layers", side_effect=AssertionError("重复校准")), \
                    patch.object(experiment, "threshold_counts", side_effect=AssertionError("重复验证")):
                experiment.main(config_path)
            self.assertEqual(before_csv, (output / "summary.csv").read_bytes())
            for name, path in config["paths"]["CHECKPOINTS"].items():
                self.assertEqual(source_hashes[name], experiment.sha256(path))
            config["run"]["BATCH_SIZE"] = 1
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "计算协议已改变"):
                experiment.main(config_path)

    def test_reject_changed_epoch_and_train_val_overlap(self):
        with TemporaryDirectory() as tmp:
            path, config = self.prepare_run(Path(tmp))
            source = Path(config["paths"]["CHECKPOINTS"]["epoch_006"])
            checkpoint = torch.load(source, weights_only=True)
            checkpoint["epoch"] = 7
            torch.save(checkpoint, source)
            with self.assertRaisesRegex(ValueError, "实际为第 7 轮"):
                experiment.main(path)
            checkpoint["train_files"].append("3.tif")
            for alias, filename in config["paths"]["CHECKPOINTS"].items():
                checkpoint["epoch"] = config["run"]["EXPECTED_EPOCHS"][alias]
                torch.save(checkpoint, filename)
            with self.assertRaisesRegex(ValueError, "存在交集"):
                experiment.main(path)

    def test_forward_order_overrides_registration_and_rejects_shared_bn(self):
        class ReverseOrder(nn.Module):
            def __init__(self):
                super().__init__()
                self.last = nn.BatchNorm2d(2)
                self.first = nn.BatchNorm2d(2)

            def forward(self, image):
                return self.last(self.first(image) * 3 + 7)

        image = torch.arange(32, dtype=torch.float32).reshape(2, 4, 4)
        samples = [dict(image=image, valid_mask=torch.ones(4, 4, dtype=torch.bool), name="one")]
        model = ReverseOrder().eval()
        self.assertEqual([name for name, _ in experiment.batchnorm_forward_order(model, image[None])],
                         ["first", "last"])
        with TemporaryDirectory() as tmp:
            experiment.calibrate_layers(model, DataLoader(samples), dict(inputs_normalized=True),
                                        "cpu", Path(tmp), 5)
            torch.testing.assert_close(model.first.running_mean, image.mean((1, 2)))
            torch.testing.assert_close(model.last.running_mean, torch.full((2,), 7.0))
            expected_var = 9 * image.var((1, 2), unbiased=False) / (image.var((1, 2), unbiased=False) + 1e-5)
            torch.testing.assert_close(model.last.running_var, expected_var)
        shared = nn.BatchNorm2d(2)
        with self.assertRaisesRegex(ValueError, "重复调用"):
            experiment.batchnorm_forward_order(nn.Sequential(shared, shared), image[None])
        self.assertFalse(shared._forward_pre_hooks)

    def test_mixed_architectures_share_input_protocol(self):
        with TemporaryDirectory() as tmp:
            config_path, config = self.prepare_run(Path(tmp))
            path = Path(config["paths"]["CHECKPOINTS"]["epoch_020"])
            checkpoint = torch.load(path, weights_only=True)
            checkpoint["model_name"] = "pvt_deeplabv3plus"
            checkpoint["model_kwargs"] = {"in_channels": 2, "pretrained": False}
            torch.save(checkpoint, path)
            loaded = experiment.load_yaml(config_path)
            checkpoints, _, _, protocol = experiment.prepare_experiment(loaded)
            self.assertEqual(len(checkpoints), 2)
            self.assertEqual(protocol["models"]["epoch_020"]["model_name"], "pvt_deeplabv3plus")
            del checkpoints  # Windows 上先释放 mmap，再改写本测试的临时检查点。
            checkpoint["std"][0] *= 2
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "输入预处理协议不同"):
                experiment.prepare_experiment(loaded)

    def test_group_boundaries_and_paired_calibration_report(self):
        water = np.array([0, 1, 10, 50, 51])
        original = np.stack([100 - water, water * 0, water, water * 0], axis=-1)[:, None, :]
        calibrated = np.stack([100 - water, water * 0, water * 0, water], axis=-1)[:, None, :]
        names = [str(i) for i in range(5)]
        with TemporaryDirectory() as tmp:
            output = Path(tmp)
            experiment.write_tables(output, {"net_original": (names, original),
                                             "net_calibrated": (names, calibrated)}, [0.5])
            with (output / "water_groups.csv").open(encoding="utf-8-sig") as stream:
                rows = list(csv.DictReader(stream))
            selected = [row for row in rows if row["model"] == "net_calibrated"
                        and row["selection"] == "threshold_0.50"]
            self.assertEqual(len(selected), 5)
            self.assertEqual([int(row["patches"]) for row in selected], [1] * 5)
            self.assertEqual(sum(int(row["valid_pixels"]) for row in selected), 500)
            with (output / "bn_comparison.csv").open(encoding="utf-8-sig") as stream:
                paired = list(csv.DictReader(stream))
            self.assertEqual(len(paired), 2)
            self.assertAlmostEqual(float(paired[0]["delta_f1_pp"]), 100)
            self.assertAlmostEqual(float(paired[0]["delta_recall_pp"]), 100)

    def test_experiment_configs_hold_training_controls_fixed(self):
        archive = ZipFile(ROOT / "docs/archive/s1_rank_configs.zip")
        self.addCleanup(archive.close)
        original = yaml.safe_load(archive.read("configs/s1/train_pvt_deeplabv3plus.yaml"))
        combined = yaml.safe_load(archive.read("configs/s1/train_pvt_deeplabv3plus_bce_dice.yaml"))
        self.assertEqual(combined["training"]["LOSS_NAME"], "bce_dice")
        self.assertEqual(combined["training"]["BCE_WEIGHT"], 1.0)
        self.assertEqual(combined["training"]["DICE_WEIGHT"], 1.0)
        self.assertNotEqual(original["paths"]["OUTPUT_DIR"], combined["paths"]["OUTPUT_DIR"])
        combined["paths"]["OUTPUT_DIR"] = original["paths"]["OUTPUT_DIR"]
        combined["training"]["LOSS_NAME"] = original["training"]["LOSS_NAME"]
        self.assertEqual(original, combined)
        config = yaml.safe_load(archive.read("configs/s1/calibrate_baselines_bn.yaml"))
        self.assertEqual(len(config["paths"]["CHECKPOINTS"]), 6)
        self.assertEqual(sorted(config["run"]["EXPECTED_EPOCHS"].values()), [6, 9, 11, 20, 20, 20])
        self.assertIsNone(config["run"]["CALIBRATION_SAMPLES"])


if __name__ == "__main__":
    unittest.main()
