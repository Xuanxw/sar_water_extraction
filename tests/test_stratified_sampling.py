"""验证分层名额、组内难度、有效标签比例及旧全局采样兼容性。"""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import train


class StratifiedSamplingTests(unittest.TestCase):
    def test_current_training_group_counts_keep_budget_and_reproducibility(self):
        groups = torch.repeat_interleave(torch.arange(5), torch.tensor([62, 598, 435, 165, 45]))
        losses = torch.arange(1305, dtype=torch.float32)
        state = torch.get_rng_state().clone()
        plan, pool = train.build_hard_sample_plan(losses, seed=42, groups=groups)
        self.assertEqual((plan, pool), train.build_hard_sample_plan(losses, seed=42, groups=groups))
        torch.testing.assert_close(state, torch.get_rng_state())
        self.assertEqual(len(plan), 1305)
        self.assertEqual(len(set(pool)), 326)
        self.assertEqual(torch.bincount(groups[pool], minlength=5).tolist(), [16, 149, 109, 41, 11])
        for group, quota in enumerate([16, 149, 109, 41, 11]):
            # 每组应选中该组损失最高的 quota 个样本，而非直接取全局最高分。
            candidates = torch.where(groups == group)[0].tolist()
            self.assertEqual(set(i for i in pool if groups[i] == group), set(candidates[-quota:]))
        counts = torch.bincount(torch.tensor(plan), minlength=1305)
        outside = torch.ones(1305, dtype=torch.bool);outside[pool] = False
        self.assertTrue((counts[outside] <= 1).all())
        self.assertGreaterEqual(int(counts[pool].sum()), 652)

    def test_empty_unknown_and_small_groups_do_not_invent_scores_or_quota(self):
        losses = torch.tensor([1., 2., 3., float('nan'), float('nan')])
        groups = torch.tensor([0, 1, 1, 2, -1])
        plan, pool = train.build_hard_sample_plan(losses, pool_ratio=.5, groups=groups)
        self.assertEqual(pool, [2])  # 已知人数 1:2，总池1个，余数大的组获得名额。
        self.assertEqual(len(plan), 5)
        _, pool = train.build_hard_sample_plan(losses, pool_ratio=1., groups=groups)
        self.assertEqual(set(pool), {0, 1, 2})
        plan, pool = train.build_hard_sample_plan(torch.full((5,), float('nan')), groups=groups)
        self.assertEqual(pool, [])
        self.assertEqual(sorted(plan), list(range(5)))

    def test_group_boundary_and_ignore_pixels_are_computed_from_training_labels(self):
        trainer = train.WaterSegmentationTrainer.__new__(train.WaterSegmentationTrainer)
        trainer.sampling = 'stratified_hard';trainer.bands = (1, 2);trainer.device = torch.device('cpu')
        samples = []
        for water in (0, 1, 10, 50, 51, 100):
            label = torch.zeros(10, 10, dtype=torch.long);label.view(-1)[:water] = 1
            samples.append(dict(image=torch.zeros(2, 10, 10), valid_mask=torch.ones(10, 10, dtype=torch.bool), label=label))
        # 99个忽略像素+1个水体，水体比例应为100%；全忽略保留组号-1。
        for water in (1, 0):
            label = torch.full((10, 10), -1, dtype=torch.long)
            if water:label[0, 0] = 1
            samples.append(dict(image=torch.zeros(2, 10, 10), valid_mask=torch.ones(10, 10, dtype=torch.bool), label=label))
        trainer.train_set = samples
        trainer.calculate_normalization()
        self.assertEqual(trainer.sample_groups.tolist(), [0, 1, 2, 3, 4, 4, 4, -1])
        self.assertEqual(trainer.sample_water_ratios[-2].item(), 1.)
        self.assertTrue(torch.isnan(trainer.sample_water_ratios[-1]))

    def test_factory_changes_only_sampling_mode_and_output_directory(self):
        options = {}
        for mode in ('hard', 'stratified_hard', 'hybrid_hard'):
            with patch.object(train, 'METHOD', 'uafnet'), patch.object(train, 'UAFNET_MBDC_DILATIONS', None), patch.object(train, 'LOSS_NAME', 'bce'), \
                 patch.object(train, 'SAMPLING', mode), patch.object(train, 'OUTPUT_DIR', None), \
                 patch.object(train, 'HEAD_LOSS_WEIGHTS', (1., 1., 1., 1.)), \
                 patch.object(train, 'UAFNET_FINAL_RESIDUAL', True), \
                 patch.object(train, 'WaterSegmentationTrainer') as factory:
                train.create_trainer();options[mode] = factory.call_args.kwargs
        for key in options['hard']:
            if key not in ('sampling', 'output_dir'):
                self.assertEqual(options['hard'][key], options['stratified_hard'][key], key)
                self.assertEqual(options['hard'][key], options['hybrid_hard'][key], key)
        self.assertEqual(options['hybrid_hard']['output_dir'].name,
                         'pvt_uafnet_hard_sampling_residual_hybrid')
        self.assertEqual(options['stratified_hard']['output_dir'].name,
                         'pvt_uafnet_hard_sampling_residual_stratified')


if __name__ == '__main__':
    unittest.main()
