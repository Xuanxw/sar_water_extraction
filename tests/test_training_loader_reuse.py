"""真实 DataLoader 合成回归：采样、主进程增强/Dropout、RNG 与旧续跑边界。"""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from supervised import train


class DeterministicWindowDataset(Dataset):
    """与 SARSceneDataset 相同：worker 只读取确定性窗口，增强由主进程完成。"""

    def __len__(self):
        return 17

    def __getitem__(self, index):
        return dict(index=index, pid=os.getpid(),
                    image=torch.arange(16, dtype=torch.float32).reshape(1, 4, 4) + index)


def close_loader(loader):
    if isinstance(loader, train.ReusableSubsetLoader):
        loader = loader.loader
    iterator = getattr(loader, '_iterator', None)
    if iterator is not None:
        iterator._shutdown_workers()  # 测试及时回收进程，产品代码不依赖私有 iterator API。


class TrainingLoaderReuseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def make_trainer(self, workers, windows_per_epoch):
        trainer = train.WaterSegmentationTrainer.__new__(train.WaterSegmentationTrainer)
        trainer.train_set = Subset(DeterministicWindowDataset(), [14, 3, 1, 16, 8, 11, 4, 6, 2, 15, 0])
        trainer.sampling = 'random'
        trainer.scene_options = {}
        trainer.windows_per_epoch = windows_per_epoch
        trainer.batch_size = 3
        trainer.seed = 42
        trainer.loader_options = dict(num_workers=workers)
        if workers:
            trainer.loader_options.update(persistent_workers=True, prefetch_factor=2)
        trainer.train_loader = DataLoader(trainer.train_set, batch_size=trainer.batch_size,
                                          shuffle=True, **trainer.loader_options)
        return trainer

    def run_epochs(self, workers, windows_per_epoch, optimized, epochs=(1, 2), state=None):
        trainer = self.make_trainer(workers, windows_per_epoch)
        if state is None:
            torch.manual_seed(1977)
        else:
            torch.set_rng_state(state.clone())
        records, loaders, messages = [], [], []
        try:
            for epoch in epochs:
                before_prepare = torch.get_rng_state().clone()
                if optimized:
                    with patch.object(train, 'cprint') as output:
                        trainer.prepare_train_sampling(epoch)
                    messages.append(output.call_args.args[0])
                elif windows_per_epoch is not None:
                    # 优化前 prepare_train_sampling 的原始 random 定长子集分支。
                    plan = torch.randperm(len(trainer.train_set),
                        generator=torch.Generator().manual_seed(trainer.seed + epoch))[:windows_per_epoch].tolist()
                    close_loader(trainer.train_loader)
                    trainer.train_loader = DataLoader(trainer.train_set,
                        batch_size=trainer.batch_size, sampler=plan, **trainer.loader_options)
                self.assertTrue(torch.equal(before_prepare, torch.get_rng_state()))
                loaders.append(trainer.train_loader)
                indices, draws, transformed, dropped, pids = [], [], [], [], set()
                for batch in trainer.train_loader:
                    indices.extend(batch['index'].tolist())
                    pids.update(batch['pid'].tolist())
                    image = batch['image'].clone()
                    for i in range(len(image)):
                        # 与训练入口相同的主进程旋转和翻转随机调用。
                        turns = int(torch.randint(4, ()).item())
                        flip = bool(torch.rand(()).item() < 0.5)
                        image[i] = torch.rot90(image[i], turns, dims=(-2, -1))
                        if flip:
                            image[i] = torch.flip(image[i], dims=(-1,))
                        draws.append((turns, flip))
                    transformed.append(image.clone())
                    dropped.append(F.dropout2d(image, p=0.25, training=True))
                records.append(dict(indices=indices, draws=draws,
                    transformed=torch.cat(transformed), dropped=torch.cat(dropped),
                    rng=torch.get_rng_state().clone(), pids=pids))
            if optimized and len(loaders) == 2:
                self.assertIs(loaders[0], loaders[1])
                if workers:
                    self.assertEqual(records[0]['pids'], records[1]['pids'])
                    self.assertEqual(len(records[0]['pids']), workers)
            if optimized:
                count = min(windows_per_epoch, len(trainer.train_set)) if windows_per_epoch is not None else len(trainer.train_set)
                kind = '限量抽样' if count < len(trainer.train_set) else '完整遍历'
                self.assertTrue(all(kind in message and f'{count}/{len(trainer.train_set)}' in message
                                    for message in messages))
            return records
        finally:
            for loader in loaders + [trainer.train_loader]:
                close_loader(loader)

    def assert_records_equal(self, expected, actual):
        self.assertEqual(expected['indices'], actual['indices'])
        self.assertEqual(expected['draws'], actual['draws'])
        for key in ('transformed', 'dropped', 'rng'):
            self.assertTrue(torch.equal(expected[key], actual[key]), key)

    def test_two_epochs_and_resume_match_original_rng_and_sampling(self):
        for workers in (0, 2):
            for windows in (7, None):
                with self.subTest(workers=workers, windows_per_epoch=windows):
                    old = self.run_epochs(workers, windows, optimized=False)
                    new = self.run_epochs(workers, windows, optimized=True)
                    for expected, actual in zip(old, new):
                        self.assert_records_equal(expected, actual)
                        self.assertEqual(len(actual['indices']), 7 if windows else 11)
                        self.assertEqual(len(set(actual['indices'])), len(actual['indices']))
                    # 模拟 epoch_end 恢复；也覆盖 RNG 隔离验证后的 train_end 边界。
                    old_resume = self.run_epochs(workers, windows, False, epochs=(2,), state=old[0]['rng'])[0]
                    new_resume = self.run_epochs(workers, windows, True, epochs=(2,), state=old[0]['rng'])[0]
                    self.assert_records_equal(old_resume, new_resume)
                    if windows is not None or workers == 0:
                        self.assert_records_equal(old[1], new_resume)
                    # 完整遍历+多worker沿用旧的resume行为，本次不修复其额外base_seed消耗。

    def test_sampler_updates_indices_without_rebuilding_loader(self):
        trainer = self.make_trainer(0, 7)
        with patch.object(train, 'cprint'):
            trainer.prepare_train_sampling(1)
            loader = trainer.train_loader
            first = list(loader.sampler)
            trainer.prepare_train_sampling(2)
        self.assertIs(loader, trainer.train_loader)
        expected = torch.randperm(len(trainer.train_set),
            generator=torch.Generator().manual_seed(trainer.seed + 2))[:7].tolist()
        self.assertEqual(list(loader.sampler), expected)
        self.assertNotEqual(first, expected)
        self.assertEqual(len(loader), 3)


if __name__ == '__main__':
    unittest.main()
