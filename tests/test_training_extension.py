"""CPU 小模型核对延长阶段的学习率、状态保留、验证重放与中断续跑。"""
import csv
import hashlib
import math
from pathlib import Path
import random
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / 'src'), str(Path(__file__).resolve().parent)]
import test_resume_training as resume_helpers
from supervised import train


POLICY = dict(learning_rate=5e-4, min_lr=1e-5)


class TrainingExtensionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = patch.object(train, 'cprint')
        self.output.start()
        self.addCleanup(self.output.stop)
        self.helper = resume_helpers.ResumeTests()
        resume_helpers.fixtures.SceneDatasetTests().prepare(self.root)

    def make(self, checkpoint=None, **changes):
        trainer = self.helper.make_trainer(self.root, checkpoint, **changes)
        self.addCleanup(trainer.dataset.close)
        return trainer

    def source(self):
        trainer = self.make()
        trainer.train()
        return trainer.run_dir / 'last.pth'

    def extend(self, checkpoint, **changes):
        options = dict(epochs=5, training_extension=POLICY)
        options.update(changes)
        return self.make(checkpoint, **options)

    @staticmethod
    def hashes(directory):
        return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in directory.rglob('*') if path.is_file()}

    def assert_rng_equal(self, expected, actual):
        self.assertEqual(expected['python'], actual['python'])
        self.assertEqual(expected['numpy'], actual['numpy'])
        torch.testing.assert_close(expected['torch'], actual['torch'], rtol=0, atol=0)

    def rows(self, trainer):
        with trainer.metrics_path.open(encoding='utf-8-sig', newline='') as stream:
            return list(csv.DictReader(stream))

    def test_zero_lr_source_preserves_state_and_starts_independent_schedule(self):
        path = self.source()
        source = self.helper.load(path)
        source_hashes = self.hashes(path.parent)
        self.assertEqual(source['optimizer_state_dict']['param_groups'][0]['lr'], 0)
        extended = self.extend(path)
        initial = self.helper.load(extended.run_dir / 'last.pth')
        self.assertNotEqual(extended.run_dir, path.parent)
        self.assertEqual(initial['phase_metrics'], {})
        self.assertEqual(initial['best_val_f1'], float('-inf'))
        self.assertEqual(list(extended.run_dir.glob('best*.pth')), [])
        self.assertFalse((extended.run_dir / 'abnormal.pth').exists())
        self.assertEqual(self.rows(extended), [])
        self.assertEqual(extended.scheduler.T_max, 2)
        self.assertEqual(extended.scheduler.last_epoch, 0)
        self.assertEqual(extended.scheduler.base_lrs, [POLICY['learning_rate']])
        self.assertEqual(initial['scaler_state_dict'], source['scaler_state_dict'])
        self.assert_rng_equal(source['rng_state'], initial['rng_state'])
        for name, value in source['model_state_dict'].items():
            torch.testing.assert_close(initial['model_state_dict'][name], value, rtol=0, atol=0)
        for pid, state in source['optimizer_state_dict']['state'].items():
            for name, value in state.items():
                torch.testing.assert_close(initial['optimizer_state_dict']['state'][pid][name], value, rtol=0, atol=0)
        extended.train()
        final = self.helper.load(extended.run_dir / 'last.pth')
        rates = [float(row['learning_rate']) for row in self.rows(extended) if row['phase'] == 'train']
        expected = [POLICY['min_lr'] + (POLICY['learning_rate'] - POLICY['min_lr']) *
                    (1 + math.cos(math.pi * n / 2)) / 2 for n in range(2)]
        np.testing.assert_allclose(rates, expected, rtol=1e-12, atol=0)
        self.assertGreater(rates[0], rates[1])
        self.assertEqual(final['optimizer_state_dict']['param_groups'][0]['lr'], POLICY['min_lr'])
        self.assertEqual(final['scheduler_state_dict']['last_epoch'], 2)
        self.assertTrue(any(not torch.equal(value, final['model_state_dict'][name])
                            for name, value in source['model_state_dict'].items()))
        self.assertEqual(self.hashes(path.parent), source_hashes)
        self.assertEqual(final['training_extension']['source_epoch'], 3)
        self.assertEqual(final['training_extension']['source_learning_rates'], [0])

    def test_extended_train_and_validation_interruptions_restore_exact_state(self):
        path = self.source()
        continuous = self.extend(path)
        continuous.train()
        expected = self.helper.load(continuous.run_dir / 'last.pth')
        for epoch, phase, filename in [(4, 'train', 'last.pth'), (4, 'val', 'train_end.pth'),
                                       (5, 'train', 'last.pth')]:
            with self.subTest(epoch=epoch, phase=phase):
                stopped = self.extend(path)
                original = stopped.run_epoch

                def fail(number, current, loader):
                    if (number, current) == (epoch, phase):
                        # 模拟中断前已消耗随机数；恢复必须以保存的边界为准。
                        random.random()
                        np.random.random()
                        torch.rand(3)
                        raise RuntimeError('模拟阶段中断')
                    return original(number, current, loader)

                with patch.object(stopped, 'run_epoch', side_effect=fail), self.assertRaisesRegex(RuntimeError, '模拟阶段中断'):
                    stopped.train()
                before = self.helper.load(stopped.run_dir / filename)
                resumed = self.extend(stopped.run_dir / filename)
                self.assertEqual(resumed.run_dir, stopped.run_dir)
                self.assertEqual(resumed.scheduler.state_dict(), before['scheduler_state_dict'])
                resumed.train()
                actual = self.helper.load(resumed.run_dir / 'last.pth')
                self.helper.assert_same_result(expected, actual)
                self.assert_rng_equal(expected['rng_state'], actual['rng_state'])
                self.assertEqual([(int(r['epoch']), r['phase']) for r in self.rows(resumed)],
                                 [(e, p) for e in (4, 5) for p in ('train', 'val')])

    def test_source_train_end_replays_validation_and_activation_survives_interrupts(self):
        source = self.make()
        self.helper.interrupt(source, 1, 'val')
        path = source.run_dir / 'train_end.pth'
        source_hashes = self.hashes(source.run_dir)
        old = self.helper.load(path)
        control = self.extend(path)
        self.assertTrue(control.extension_schedule_pending)
        self.assertEqual(control.optimizer.param_groups[0]['lr'], old['optimizer_state_dict']['param_groups'][0]['lr'])
        self.assertEqual([(int(r['epoch']), r['phase']) for r in self.rows(control)], [(1, 'train')])
        control.train()
        expected = self.helper.load(control.run_dir / 'last.pth')
        stopped = self.extend(path)
        self.helper.interrupt(stopped, 1, 'val')
        branch_point = stopped.run_dir / 'train_end.pth'
        pending = self.extend(branch_point)
        self.assertEqual(pending.run_dir, stopped.run_dir)
        self.assertTrue(pending.extension_schedule_pending)
        self.helper.interrupt(pending, 2, 'train')
        activated = self.helper.load(pending.run_dir / 'last.pth')
        self.assertEqual(activated['epoch'], 1)
        self.assertEqual(activated['scheduler_state_dict']['T_max'], 4)
        self.assertEqual(activated['scheduler_state_dict']['last_epoch'], 0)
        self.assertEqual(activated['optimizer_state_dict']['param_groups'][0]['lr'], POLICY['learning_rate'])
        resumed = self.extend(pending.run_dir / 'last.pth')
        self.assertFalse(resumed.extension_schedule_pending)
        resumed.train()
        actual = self.helper.load(resumed.run_dir / 'last.pth')
        self.helper.assert_same_result(expected, actual)
        self.assert_rng_equal(expected['rng_state'], actual['rng_state'])
        rows = self.rows(resumed)
        self.assertEqual([(int(r['epoch']), r['phase']) for r in rows],
                         [(e, p) for e in range(1, 6) for p in ('train', 'val')])
        self.assertEqual(float(rows[1]['learning_rate']), old['optimizer_state_dict']['param_groups'][0]['lr'])
        self.assertEqual(float(rows[2]['learning_rate']), POLICY['learning_rate'])
        self.assertEqual(actual['scheduler_state_dict']['last_epoch'], 4)
        self.assertEqual(actual['optimizer_state_dict']['param_groups'][0]['lr'], POLICY['min_lr'])
        self.assertEqual(source_hashes, self.hashes(source.run_dir))

    def test_epoch_nine_source_uses_71_steps_and_first_new_lr_in_epoch_ten(self):
        source = self.make(epochs=20)
        self.helper.interrupt(source, 9, 'val')
        extension = self.extend(source.run_dir / 'train_end.pth', epochs=80,
                                training_extension=dict(learning_rate=5e-5, min_lr=1e-6))
        self.helper.interrupt(extension, 11, 'train')
        rows = self.rows(extension)
        self.assertEqual([(int(r['epoch']), r['phase']) for r in rows],
                         [(9, 'train'), (9, 'val'), (10, 'train'), (10, 'val')])
        self.assertEqual(float(rows[2]['learning_rate']), 5e-5)
        self.assertEqual(extension.scheduler.T_max, 71)
        self.assertEqual(extension.scheduler.last_epoch, 1)

    def test_strict_legacy_resume_and_extension_protocol_guards(self):
        path = self.source()
        with self.assertRaisesRegex(ValueError, 'epochs'):
            self.make(path, epochs=5)
        cases = [dict(epochs=3), dict(cosine_schedule=False), dict(learning_rate=2e-3),
                 dict(loss_kwargs={'class_weights': (1., 2.)}), dict(batch_size=1),
                 dict(validation_inference={'stride': 8, 'blend': 'gaussian'})]
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.extend(path, **changes)
        extended = self.extend(path)
        initial = extended.run_dir / 'last.pth'
        for changes in (dict(training_extension=None), dict(training_extension=dict(learning_rate=6e-4, min_lr=1e-5)),
                        dict(epochs=6)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.extend(initial, **changes)
        checkpoint = self.helper.load(path)
        checkpoint['resume_phase'] = 'bn_calibrated_inference_only'
        inference = self.root / 'calibrated.pth'
        torch.save(checkpoint, inference)
        with self.assertRaisesRegex(ValueError, 'BN校准'):
            self.extend(inference)

    def test_invalid_extension_policies_fail_before_creating_output(self):
        for config in ({}, {'learning_rate': 1e-4}, {'learning_rate': 0, 'min_lr': 0},
                       {'learning_rate': 1e-4, 'min_lr': -1e-6},
                       {'learning_rate': 1e-4, 'min_lr': 1e-3},
                       {'learning_rate': float('nan'), 'min_lr': 0},
                       {'learning_rate': 1e-4, 'min_lr': float('inf')},
                       {'learning_rate': True, 'min_lr': 0}, POLICY):
            with self.subTest(config=config), self.assertRaises(ValueError):
                self.make(training_extension=config)
        self.assertFalse((self.root / 'runs').exists())


if __name__ == '__main__':
    unittest.main()
