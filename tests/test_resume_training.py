"""CPU小模型验证：按轮续跑、验证重放、日志去重、旧格式限制和配置保护。"""
import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / 'src'), str(Path(__file__).resolve().parent)]
from supervised import train
import test_sar_scene_dataset as fixtures

# 测试显式加载样例配置，正式脚本仅使用命令行提供的文件。
train.configure(Path(__file__).resolve().parents[1] / 'configs/train.yaml')


class TinyResumeNet(nn.Module):
    construction_pretrained = []

    def __init__(self, pretrained=True):
        super().__init__()
        self.construction_pretrained.append(pretrained)
        self.net = nn.Sequential(nn.Conv2d(1, 4, 1), nn.ReLU(), nn.Dropout2d(0.2), nn.Conv2d(4, 1, 1))

    def forward(self, x):
        return self.net(x)


class ResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def make_trainer(self, root, resume=None, **changes):
        options = dict(model_class=TinyResumeNet, model_kwargs={'pretrained': True}, model_info={'model_name':'tiny'},
                       image_dir=root/'images', label_dir=root/'labels', output_dir=root/'runs', bands=(1,),
                       epochs=3, learning_rate=1e-3, batch_size=2, loss_name='bce', optimizer_name='adamw',
                       cosine_schedule=True, train_augmentation=True, head_loss_weights=(1.,),
                       scene_options=dict(cache_dir=root/'cache', window_size=32, halo=8,
                                          normalization_windows=2, num_workers=0, windows_per_epoch=4),
                       resume_checkpoint=resume)
        options.update(changes)
        with patch.object(torch.cuda,'is_available',return_value=False):
            return train.WaterSegmentationTrainer(**options)

    def load(self, path):
        return torch.load(path,map_location='cpu',weights_only=True)

    def interrupt(self, trainer, epoch, phase):
        run = trainer.run_epoch
        def maybe_fail(number, current_phase, loader):
            if number == epoch and current_phase == phase:
                raise RuntimeError('模拟数据读取中断')
            return run(number,current_phase,loader)
        with patch.object(trainer,'run_epoch',side_effect=maybe_fail), self.assertRaisesRegex(RuntimeError,'模拟数据读取中断'):
            trainer.train()

    def assert_same_result(self, expected, actual):
        for key,tensor in expected['model_state_dict'].items():
            torch.testing.assert_close(tensor,actual['model_state_dict'][key],rtol=0,atol=0)
        self.assertEqual(expected['scheduler_state_dict'],actual['scheduler_state_dict'])
        self.assertEqual(expected['optimizer_state_dict']['param_groups'], actual['optimizer_state_dict']['param_groups'])
        for pid, state in expected['optimizer_state_dict']['state'].items():
            for key,value in state.items():
                torch.testing.assert_close(value,actual['optimizer_state_dict']['state'][pid][key],rtol=0,atol=0)

    def test_epoch_resume_equals_continuous_and_preserves_logs(self):
        with TemporaryDirectory() as temp, patch.object(train,'cprint'):
            root=Path(temp)
            fixtures.SceneDatasetTests().prepare(root)
            continuous=self.make_trainer(root)
            continuous.train()
            expected=self.load(continuous.run_dir/'last.pth')
            stopped=self.make_trainer(root)
            self.interrupt(stopped,2,'train')
            checkpoint=self.load(stopped.run_dir/'last.pth')
            self.assertEqual(checkpoint['epoch'],1)
            self.assertIn('rng_state',checkpoint)
            # 模拟检查点之后遗留的CSV尾行，续跑需备份后重写，不能重复追加。
            with stopped.metrics_path.open('a',encoding='utf-8',newline='') as stream:
                csv.DictWriter(stream,fieldnames=stopped.metric_fields).writerow(dict(epoch=2,phase='train'))
            resumed=self.make_trainer(root,stopped.run_dir/'last.pth')
            self.assertEqual(resumed.start_epoch,2)
            self.assertEqual(resumed.run_dir,stopped.run_dir)
            self.assertFalse(TinyResumeNet.construction_pretrained[-1])
            resumed.train()
            actual=self.load(resumed.run_dir/'last.pth')
            self.assert_same_result(expected,actual)
            with resumed.metrics_path.open(encoding='utf-8-sig') as stream:
                pairs=[(int(row['epoch']),row['phase']) for row in csv.DictReader(stream)]
            self.assertEqual(pairs,[(e,p) for e in (1,2,3) for p in ('train','val')])
            self.assertEqual(len(list(resumed.run_dir.glob('metrics.before_resume_*.csv'))),1)
            self.assertIn('模拟数据读取中断',(resumed.run_dir/'training_error.log').read_text(encoding='utf-8'))
            self.assertFalse(list(resumed.run_dir.glob('*.pth.partial')))

    def test_train_end_resume_only_replays_validation(self):
        with TemporaryDirectory() as temp, patch.object(train,'cprint'):
            root=Path(temp)
            fixtures.SceneDatasetTests().prepare(root)
            continuous=self.make_trainer(root)
            continuous.train()
            expected=self.load(continuous.run_dir/'last.pth')
            stopped=self.make_trainer(root)
            self.interrupt(stopped,1,'val')
            path=stopped.run_dir/'train_end.pth'
            self.assertEqual(self.load(path)['resume_phase'],'train_end')
            resumed=self.make_trainer(root,path)
            self.assertTrue(resumed.resume_validation)
            phases=[]
            run=resumed.run_epoch
            with patch.object(resumed,'run_epoch',side_effect=lambda e,p,l:(phases.append((e,p)),run(e,p,l))[1]):
                resumed.train()
            self.assertEqual(phases,[(1,'val'),(2,'train'),(2,'val'),(3,'train'),(3,'val')])
            self.assert_same_result(expected,self.load(resumed.run_dir/'last.pth'))

    def test_keyboard_interrupt_is_logged_without_fake_checkpoint(self):
        with TemporaryDirectory() as temp, patch.object(train, 'cprint'):
            root = Path(temp)
            fixtures.SceneDatasetTests().prepare(root)
            trainer = self.make_trainer(root)
            with patch.object(trainer, 'run_epoch', side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
                trainer.train()
            self.assertIn('KeyboardInterrupt', (trainer.run_dir / 'training_error.log').read_text(encoding='utf-8'))
            self.assertFalse((trainer.run_dir / 'last.pth').exists())
            self.assertFalse((trainer.run_dir / 'train_end.pth').exists())

    def test_legacy_rng_warning_and_incompatible_data_rejected(self):
        with TemporaryDirectory() as temp, patch.object(train,'cprint') as output:
            root=Path(temp)
            fixtures.SceneDatasetTests().prepare(root)
            stopped=self.make_trainer(root)
            self.interrupt(stopped,2,'train')
            path=stopped.run_dir/'last.pth'
            checkpoint=self.load(path)
            checkpoint.pop('rng_state')
            checkpoint.pop('resume_phase')
            torch.save(checkpoint,path)
            resumed=self.make_trainer(root,path)
            self.assertEqual(resumed.start_epoch,2)
            self.assertEqual(resumed.optimizer.param_groups[0]['lr'],checkpoint['optimizer_state_dict']['param_groups'][0]['lr'])
            self.assertTrue(any('没有随机数状态' in str(call) for call in output.call_args_list))
            resumed.dataset.close()
            checkpoint['split_manifest_sha256']='wrong'
            torch.save(checkpoint,path)
            with self.assertRaisesRegex(ValueError,'split_manifest_sha256'):
                self.make_trainer(root,path)
            checkpoint['split_manifest_sha256']=stopped.split_info['split_manifest_sha256']
            checkpoint.pop('optimizer_state_dict')
            torch.save(checkpoint,path)
            with self.assertRaisesRegex(ValueError,'缺少优化器'):
                self.make_trainer(root,path)


if __name__ == '__main__':
    unittest.main()
