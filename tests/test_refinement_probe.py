"""冻结前端小实验的 CPU 合成测试；不触发正式模型或 GPU 推理。"""
import contextlib
import csv
import io
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluation import probe_refinement as probe
from models.losses import WaterSegmentationLoss


def make_cache(folder):
    """四张 64×64 标签、对应四级32通道特征；前两张训练，后两张验证。"""
    rng = np.random.default_rng(123)
    for key, size in zip(("f2", "f3", "f4", "f5"), (16, 8, 4, 2)):
        np.save(folder / f"{key}.npy", rng.normal(0, 1, (4, 32, size, size)).astype("float16"))
    coarse = rng.normal(0, 1, (4, 1, 16, 16)).astype("float16")
    np.save(folder / "coarse.npy", coarse)
    label = rng.integers(0, 2, (4, 64, 64), dtype=np.int8)
    label[:, :2, :] = -1
    np.save(folder / "label.npy", label)
    return probe.FeatureCache(folder, ["train_a", "train_b", "val_a", "val_b"], 2)


class RefinementProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_rank_flat_have_identical_trainable_state_for_same_seed(self):
        torch.manual_seed(42)
        rank = probe.RefinementProbe("rank")
        torch.manual_seed(42)
        flat = probe.RefinementProbe("flat")
        self.assertEqual(list(rank.state_dict()), list(flat.state_dict()))
        self.assertEqual(sum(p.numel() for p in rank.parameters()), sum(p.numel() for p in flat.parameters()))
        for name, value in rank.state_dict().items():
            torch.testing.assert_close(value, flat.state_dict()[name], rtol=0, atol=0)
        self.assertTrue(all(isinstance(layer.rank, probe.FlatRank) for layer in flat.layers))
        guide = torch.randn(2, 1, 4, 4)
        foreground, background = flat.layers[0].rank(guide)
        torch.testing.assert_close(foreground, torch.ones_like(guide))
        torch.testing.assert_close(background, torch.ones_like(guide))

    def test_all_modes_start_at_coarse_and_have_finite_two_step_gradients(self):
        torch.manual_seed(17)
        features = [torch.randn(2, 32, size, size) for size in (16, 8, 4, 2)]
        coarse = torch.randn(2, 1, 16, 16)
        label = torch.randint(0, 2, (2, 64, 64))
        label[:, :2] = -1
        expected = F.interpolate(coarse, size=(64, 64), mode="bilinear", align_corners=True)
        criterion = WaterSegmentationLoss("bce")
        for mode in probe.MODES:
            with self.subTest(mode=mode):
                torch.manual_seed(42)
                model = probe.RefinementProbe(mode).eval()
                with torch.no_grad():
                    outputs = model(features, coarse, (64, 64))
                self.assertEqual(len(outputs), 3)
                self.assertTrue(all(tuple(output.shape) == (2, 1, 64, 64) for output in outputs))
                torch.testing.assert_close(outputs[-1], expected, rtol=0, atol=0)
                model.train()
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
                for step in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    outputs = model(features, coarse, (64, 64))
                    loss = sum(criterion(output, label) for output in outputs)
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    gradients = [parameter.grad for parameter in model.parameters()]
                    self.assertTrue(all(grad is not None and torch.isfinite(grad).all() for grad in gradients))
                    self.assertTrue(any(torch.count_nonzero(grad) > 0 for grad in gradients))
                    if step == 1:
                        # 首步零初始化会阻断最终修正头前面的梯度；第二步这些层应开始学习。
                        name = "nested.x03" if mode == "nested" else "layers.2.conv_fusion"
                        final_path = [p.grad for key, p in model.named_parameters() if key.startswith(name)]
                        self.assertTrue(any(torch.count_nonzero(grad) > 0 for grad in final_path))
                    optimizer.step()
                self.assertTrue(all(torch.isfinite(parameter).all() for parameter in model.parameters()))

    def test_memmap_split_ignore_copy_and_cleanup(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            cache = make_cache(folder)
            mappings = [array._mmap for array in cache.arrays]
            self.assertEqual(cache.train_count, 2)
            self.assertEqual(cache.names[:cache.train_count], ["train_a", "train_b"])
            self.assertEqual(cache.names[cache.train_count:], ["val_a", "val_b"])
            training = DataLoader(Subset(cache, range(cache.train_count)), batch_size=2, shuffle=False)
            validation = DataLoader(Subset(cache, range(cache.train_count, len(cache))), batch_size=2, shuffle=False)
            train_batch, val_batch = next(iter(training)), next(iter(validation))
            torch.testing.assert_close(train_batch[0][0], cache[0][0])
            torch.testing.assert_close(val_batch[0][0], cache[2][0])
            self.assertEqual(int((val_batch[-1] == -1).sum()), 2 * 2 * 64)
            original = cache[0][0].clone()
            copied = cache[0][0]
            copied.fill_(999)
            torch.testing.assert_close(cache[0][0], original)
            model = probe.RefinementProbe("rank").eval()
            before = {key: value.clone() for key, value in model.state_dict().items()}
            metrics, rows = probe.evaluate_probe(model, validation, torch.device("cpu"), scan=True)
            self.assertEqual(metrics["valid_pixels"], 2 * (64 * 64 - 2 * 64))
            self.assertEqual(len(rows), len(probe.THRESHOLDS))
            self.assertTrue(all(row["valid_pixels"] == metrics["valid_pixels"] for row in rows))
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, before[key], rtol=0, atol=0)
            cache.close()
            self.assertEqual(cache.arrays, [])
            self.assertTrue(all(mapping.closed for mapping in mappings))
        self.assertFalse(folder.exists())

    def test_one_epoch_run_writes_fixed_final_and_actual_precision(self):
        metric_keys = list(probe.metrics_from_counts((1, 1, 1, 1)))
        histories, finals, sweeps = io.StringIO(), io.StringIO(), io.StringIO()
        history_writer = csv.DictWriter(histories, fieldnames=["source", "mode", "seed", "epoch",
            "train_loss", "amp_skipped_steps", "seconds", "val_loss", *metric_keys])
        final_writer = csv.DictWriter(finals, fieldnames=["source", "mode", "seed", "epoch",
            "parameters", "val_loss", *metric_keys, *[key for p in (80, 85, 90)
            for key in (f"actual_precision_at_p{p}", f"recall_at_p{p}", f"threshold_at_p{p}")]])
        scan_writer = csv.DictWriter(sweeps, fieldnames=["source", "mode", "seed", "threshold", *metric_keys])
        for writer in (history_writer, final_writer, scan_writer):
            writer.writeheader()
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            with TemporaryDirectory(prefix="cache_", dir=folder) as cache_directory:
                cache_path = Path(cache_directory)
                cache = make_cache(cache_path)
                try:
                    with patch.object(probe, "EPOCHS", 1), patch.object(probe, "BATCH_SIZE", 2), \
                         contextlib.redirect_stdout(io.StringIO()):
                        for mode in probe.MODES:
                            probe.run_probe(cache, "synthetic_front", mode, 42, folder, torch.device("cpu"),
                                            history_writer, final_writer, scan_writer)
                finally:
                    cache.close()
            self.assertFalse(cache_path.exists())
            final_rows = list(csv.DictReader(io.StringIO(finals.getvalue())))
            self.assertEqual(len(final_rows), 3)
            self.assertEqual({row["mode"] for row in final_rows}, set(probe.MODES))
            self.assertTrue(all(row["epoch"] == "1" for row in final_rows))
            for row in final_rows:
                for target in (80, 85, 90):
                    actual = row[f"actual_precision_at_p{target}"]
                    if actual:
                        self.assertGreaterEqual(float(actual), target / 100)
                        self.assertTrue(0 <= float(row[f"recall_at_p{target}"]) <= 1)
                    else:
                        self.assertEqual(row[f"recall_at_p{target}"], "")
                        self.assertEqual(row[f"threshold_at_p{target}"], "")
                saved = torch.load(folder / f"synthetic_front_{row['mode']}_seed42.pth", weights_only=True)
                self.assertEqual(saved["epoch"], 1)
                self.assertEqual(saved["mode"], row["mode"])
                self.assertTrue(all(value.device.type == "cpu" for value in saved["model_state_dict"].values()))
            history_rows = list(csv.DictReader(io.StringIO(histories.getvalue())))
            self.assertEqual(len(history_rows), 6)
            initial = [row for row in history_rows if row["epoch"] == "0"]
            self.assertEqual(len({(row["tn"], row["fp"], row["fn"], row["tp"]) for row in initial}), 1)
            self.assertTrue(all(row["amp_skipped_steps"] == "0" for row in history_rows))
            self.assertEqual(len(list(csv.DictReader(io.StringIO(sweeps.getvalue())))), 3 * len(probe.THRESHOLDS))


if __name__ == "__main__":
    unittest.main()
