"""混合池保留全局难例、补充少水体样本，并保持预算与旧模式可比。"""
import unittest

import torch

from train import build_hard_sample_plan


class HybridSamplingTests(unittest.TestCase):
    def test_full_budget_and_global_core_are_preserved(self):
        groups = torch.repeat_interleave(torch.arange(5), torch.tensor([62, 598, 435, 165, 45]))
        losses = torch.arange(1305, dtype=torch.float32)
        state = torch.get_rng_state().clone()
        plan, pool = build_hard_sample_plan(losses, groups=groups, low_water_pool_ratio=.2)
        self.assertEqual((plan, pool), build_hard_sample_plan(losses, groups=groups, low_water_pool_ratio=.2))
        torch.testing.assert_close(state, torch.get_rng_state())
        self.assertEqual(len(plan), 1305)
        self.assertEqual(len(set(pool)), 326)
        self.assertEqual(pool[:261], list(range(1304, 1043, -1)))
        self.assertEqual(pool[261:], list(range(659, 594, -1)))
        counts = torch.bincount(torch.tensor(plan), minlength=1305)
        outside = torch.ones(1305, dtype=torch.bool)
        outside[pool] = False
        self.assertTrue((counts[outside] <= 1).all())
        self.assertGreaterEqual(int(counts[pool].sum()), 652)

    def test_overlap_and_shortage_fill_without_duplicates(self):
        losses = torch.arange(20, dtype=torch.float32)
        groups = torch.full((20,), 3)
        groups[[19, 18, 3]] = 1  # 全局前列已有少水体；池外只剩一个候选。
        _, pool = build_hard_sample_plan(losses, pool_ratio=.5, groups=groups, low_water_pool_ratio=.2)
        self.assertEqual(pool, [19, 18, 17, 16, 15, 14, 13, 12, 3, 11])
        self.assertEqual(len(set(pool)), 10)
        groups.fill_(0)  # 无水体切片不冒充少水体候选；缺额退回全局排名。
        _, pool = build_hard_sample_plan(losses, pool_ratio=.5, groups=groups, low_water_pool_ratio=.2)
        self.assertEqual(pool, list(range(19, 9, -1)))

    def test_zero_ratio_reproduces_global_plan_and_unknowns_are_excluded(self):
        losses = torch.arange(20, dtype=torch.float32)
        groups = torch.arange(20) % 5
        self.assertEqual(build_hard_sample_plan(losses),
                         build_hard_sample_plan(losses, groups=groups, low_water_pool_ratio=0))
        losses[19] = float('nan')
        groups[18] = -1
        plan, pool = build_hard_sample_plan(losses, pool_ratio=.5, groups=groups, low_water_pool_ratio=1)
        self.assertEqual(len(plan), 20)
        self.assertEqual(len(set(pool)), 9)
        self.assertNotIn(19, pool)
        self.assertNotIn(18, pool)
        self.assertTrue({1, 6, 11, 16}.issubset(pool))
        plan, pool = build_hard_sample_plan(torch.full((20,), float('nan')), groups=groups, low_water_pool_ratio=.2)
        self.assertEqual(sorted(plan), list(range(20)))
        self.assertEqual(pool, [])

    def test_invalid_hybrid_configuration_is_rejected(self):
        losses = torch.ones(8)
        with self.assertRaises(ValueError):
            build_hard_sample_plan(losses, low_water_pool_ratio=.2)
        with self.assertRaises(ValueError):
            build_hard_sample_plan(losses, groups=torch.ones(8), low_water_pool_ratio=1.1)


if __name__ == '__main__':
    unittest.main()
