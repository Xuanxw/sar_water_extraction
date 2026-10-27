"""困难采样保持每轮预算、索引边界与基线设置；不读取验证数据。"""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import train


class SamplingTests(unittest.TestCase):
    def test_plan_is_reproducible_and_oversamples_only_the_hard_pool(self):
        losses = torch.tensor([.1, .2, .3, .4, .5, .6, .7, .8])
        state = torch.get_rng_state().clone()
        plan, pool = train.build_hard_sample_plan(losses, .5, .25, 42)
        self.assertEqual((plan, pool), train.build_hard_sample_plan(losses, .5, .25, 42))
        torch.testing.assert_close(state, torch.get_rng_state())
        self.assertEqual(pool, [7, 6])
        self.assertEqual(len(plan), 8)
        self.assertTrue(all(0 <= i < 8 for i in plan))
        counts = torch.bincount(torch.tensor(plan), minlength=8)
        self.assertGreaterEqual(counts[pool].sum().item(), 4)
        self.assertTrue((counts[:6] <= 1).all())  # 普通部分不重复，困难部分只抽池内样本。

    def test_unknown_difficulties_fall_back_to_uniform_and_are_not_ranked(self):
        losses = torch.full((8,), float("nan"))
        plan, pool = train.build_hard_sample_plan(losses)
        self.assertEqual(sorted(plan), list(range(8)))
        self.assertEqual(pool, [])
        losses[3] = 10
        _, pool = train.build_hard_sample_plan(losses)
        self.assertEqual(pool, [3])

    def test_factory_preserves_baseline_settings_except_sampling_and_directory(self):
        options = {}
        for mode in ("random", "hard"):
            with patch.object(train, "METHOD", "uafnet"), patch.object(train, "UAFNET_MBDC_DILATIONS", None), patch.object(train, "LOSS_NAME", "bce"), \
                 patch.object(train, "SAMPLING", mode), patch.object(train, "OUTPUT_DIR", None), \
                 patch.object(train, "HEAD_LOSS_WEIGHTS", (1., 1., 1., 1.)), \
                 patch.object(train, "UAFNET_FINAL_RESIDUAL", False), \
                 patch.object(train, "WaterSegmentationTrainer") as factory:
                train.create_trainer()
                options[mode] = factory.call_args.kwargs
        for key in options["random"]:
            if key not in ("sampling", "output_dir"):
                self.assertEqual(options["random"][key], options["hard"][key], key)
        self.assertEqual(options["random"]["output_dir"].name, "pvt_uafnet")
        self.assertEqual(options["hard"]["output_dir"].name, "pvt_uafnet_hard_sampling")


if __name__ == "__main__":
    unittest.main()
