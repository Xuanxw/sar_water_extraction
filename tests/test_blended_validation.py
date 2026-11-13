"""融合验证迁移：训练状态保留、评价口径隔离、RNG和质量图计数一致。"""
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import random
import numpy as np
import torch
from supervised import train
import test_sar_scene_dataset as fixtures
import test_resume_training

# 测试显式加载样例配置，正式脚本仅使用命令行提供的文件。
train.configure(Path(__file__).resolve().parents[1] / 'configs/train.yaml')


class BlendedValidationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.helper=test_resume_training.ResumeTests()
        self.options=dict(stride=8,blend='hann',num_workers=0)

    def test_protocol_migration_preserves_state_and_original_run(self):
        with TemporaryDirectory() as tmp, patch.object(train,'cprint'):
            root=Path(tmp);fixtures.SceneDatasetTests().prepare(root)
            stopped=self.helper.make_trainer(root)
            self.helper.interrupt(stopped,1,'val')
            old=stopped.run_dir;old_csv=(old/'metrics.csv').read_bytes()
            checkpoint=self.helper.load(old/'train_end.pth')
            resumed=self.helper.make_trainer(root,old/'train_end.pth',validation_inference=self.options,
                quality_options=dict(enabled=True,interval=1,max_scenes=None,max_side=20))
            self.assertNotEqual(old,resumed.run_dir)
            self.assertTrue(resumed.resume_validation)
            self.assertEqual(resumed.start_epoch,1)
            self.assertEqual(resumed.best_f1,float('-inf'))
            for k,v in checkpoint['model_state_dict'].items():
                torch.testing.assert_close(resumed.model.state_dict()[k].cpu(),v,rtol=0,atol=0)
            self.assertEqual(resumed.scheduler.state_dict(),checkpoint['scheduler_state_dict'])
            self.assertEqual(resumed.optimizer.state_dict()['param_groups'],checkpoint['optimizer_state_dict']['param_groups'])
            python_rng=random.getstate();numpy_rng=np.random.get_state();rng=torch.get_rng_state().clone()
            resumed.phase_metrics['val']=resumed.run_epoch(1,'val',resumed.val_loader)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertEqual(python_rng,random.getstate())
            np.testing.assert_array_equal(numpy_rng[1],np.random.get_state()[1])
            self.assertEqual((old/'metrics.csv').read_bytes(),old_csv)
            with (resumed.run_dir/'quality/epoch_001/per_scene.csv').open(encoding='utf-8-sig') as f: row=next(csv.DictReader(f))
            for key in ('tn','fp','fn','tp','valid_pixels'):
                self.assertEqual(int(row[key]),resumed.phase_metrics['val'][key])
            self.assertTrue((resumed.run_dir/'resume_source.json').exists())
            # 新口径第一轮val可以独立选best，不与源目录的旧分数混排。
            resumed.scheduler.step();resumed.save_checkpoint(1)
            self.assertEqual(self.helper.load(resumed.run_dir/'best_f1.pth')['validation_inference'],self.options)
            again=self.helper.make_trainer(root,resumed.run_dir/'last.pth',validation_inference=self.options)
            self.assertFalse(again.resume_protocol_changed)
            self.assertEqual(again.run_dir,resumed.run_dir)
            self.assertEqual(again.best_f1,resumed.best_f1)
            for t in (stopped,resumed,again):t.dataset.close()

    def test_fused_resume_matches_continuous_training(self):
        with TemporaryDirectory() as tmp, patch.object(train,'cprint'):
            root=Path(tmp);fixtures.SceneDatasetTests().prepare(root)
            continuous=self.helper.make_trainer(root,validation_inference=self.options)
            continuous.train();expected=self.helper.load(continuous.run_dir/'last.pth')
            stopped=self.helper.make_trainer(root,validation_inference=self.options)
            self.helper.interrupt(stopped,1,'val')
            resumed=self.helper.make_trainer(root,stopped.run_dir/'train_end.pth',validation_inference=self.options)
            phases=[];original=resumed.run_epoch
            with patch.object(resumed,'run_epoch',side_effect=lambda e,p,l:(phases.append((e,p)),original(e,p,l))[1]):
                resumed.train()
            self.assertEqual(phases[0],(1,'val'))
            self.assertNotIn((1,'train'),phases)
            self.helper.assert_same_result(expected,self.helper.load(resumed.run_dir/'last.pth'))

    def test_failed_validation_restores_rng(self):
        with TemporaryDirectory() as tmp, patch.object(train,'cprint'):
            root=Path(tmp);fixtures.SceneDatasetTests().prepare(root)
            trainer=self.helper.make_trainer(root,validation_inference=self.options)
            python_rng=random.getstate();numpy_rng=np.random.get_state();rng=torch.get_rng_state().clone()
            def fail(*args):
                torch.rand(1);np.random.rand();random.random()
                raise RuntimeError('test val interruption')
            with patch.object(trainer,'_run_epoch',side_effect=fail),self.assertRaisesRegex(RuntimeError,'interruption'):
                trainer.run_epoch(1,'val',trainer.val_loader)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertEqual(python_rng,random.getstate())
            np.testing.assert_array_equal(numpy_rng[1],np.random.get_state()[1])
            trainer.dataset.close()


if __name__=='__main__':unittest.main()
