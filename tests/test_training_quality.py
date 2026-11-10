"""训练图直接复用val；计数与原训练指标一致；后处理入口独立。"""
import csv
import gc
import weakref
import numpy as np
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import torch
from supervised import train, post_process
import test_sar_scene_dataset as fixtures
from supervised.evaluation import training_quality as quality


class TrainingQualityTests(unittest.TestCase):
    def test_shared_counts_and_sampled_copy_match_full_copy(self):
        # 不整除的尾块、未知像素和不落在概览采样点上的小块，都必须与旧输出相同。
        with TemporaryDirectory() as temp:
            root = Path(temp)
            old = quality.ValidationQualityWriter(object(), [], root/'old', 1, max_side=5)
            new = quality.ValidationQualityWriter(object(), [], root/'new', 1, max_side=5)
            values = torch.arange(37*53).reshape(37,53)
            labels = values % 3 - 1
            image = values.float()/100
            prediction = values % 2
            probability = image.sigmoid()
            valid = values % 11 != 0
            labels[~valid] = -1
            for top in range(0,37,7):
                for left in range(0,53,13):
                    region = np.s_[top:min(top+7,37),left:min(left+13,53)]
                    truth, pred = labels[region], prediction[region]
                    known = truth >= 0
                    counts = torch.bincount(2*truth[known]+pred[known],minlength=4).numpy()
                    args = ('scene',image[region],truth,probability[region],pred,valid[region])
                    options = dict(scene_window=(top,left,*truth.shape),scene_shape=(37,53))
                    old.add(*args,**options)
                    new.add(*args,**options,counts=counts)
            for key in ('image','label','probability','prediction','counts'):
                np.testing.assert_array_equal(getattr(old,key),getattr(new,key))

    def test_validation_figures_counts_and_curve(self):
        with TemporaryDirectory() as temp, patch.object(torch.cuda,'is_available',return_value=False):
            root=Path(temp);fixtures.SceneDatasetTests().prepare(root)
            trainer=train.WaterSegmentationTrainer(model_class=fixtures.TinySceneNet,model_kwargs={},
                model_info=dict(model_name='tiny'),image_dir=root/'images',label_dir=root/'labels',
                output_dir=root/'runs',bands=(1,),epochs=1,batch_size=2,loss_name='bce',
                scene_options=dict(cache_dir=root/'cache',window_size=32,halo=8,num_workers=0),
                quality_options=dict(enabled=True,interval=1,max_scenes=None,max_side=20))
            trainer.train()
            folder=trainer.run_dir/'quality/epoch_001'
            self.assertTrue((folder/'figures/val.png').is_file())
            self.assertTrue((trainer.run_dir/'quality/training_curves.png').is_file())
            self.assertTrue(json.loads((folder/'status.json').read_text())['complete'])
            with (folder/'per_scene.csv').open(encoding='utf-8-sig') as f: row=next(csv.DictReader(f))
            for key in ('tn','fp','fn','tp','valid_pixels'):
                self.assertEqual(int(row[key]),trainer.phase_metrics['val'][key])
            self.assertFalse((trainer.run_dir/'postprocess').exists())

    def test_repeated_figures_release_cycles(self):
        # 禁用自动GC，验证每张图主动清理，不能依赖偶然触发的垃圾回收。
        references = []
        original = quality.Figure
        def figure(*args, **kwargs):
            result = original(*args, **kwargs)
            references.append(weakref.ref(result))
            return result
        enabled = gc.isenabled()
        gc.disable()
        try:
            with TemporaryDirectory() as temp, patch.object(quality, 'Figure', side_effect=figure):
                image = np.zeros((768, 768), dtype=np.float32)
                label = np.zeros(image.shape, dtype=np.int8)
                for i in range(4):
                    quality.save_prediction_figure(image, label, image, label,
                        np.array([label.size, 0, 0, 0]), Path(temp)/f'{i}.png', 'memory regression')
                    self.assertTrue(all(ref() is None for ref in references))
        finally:
            if enabled:
                gc.enable()

    def test_compact_counts_keep_unknown_and_full_resolution(self):
        with TemporaryDirectory() as temp:
            writer = quality.ValidationQualityWriter(object(), [], Path(temp), 1, max_side=20)
            label = torch.arange(768*768).reshape(768, 768) % 3 - 1
            prediction = torch.arange(768*768).reshape(768, 768) % 2
            valid = label >= 0
            expected = torch.bincount(2*label[valid]+prediction[valid], minlength=4).numpy()
            image = torch.zeros((768, 768))
            writer.add('test', image, label, image, prediction, torch.ones_like(valid))
            np.testing.assert_array_equal(writer.counts, expected)
            with patch.object(quality, 'save_prediction_figure'):
                writer.finish()
            self.assertIsNone(writer.image)
            self.assertIsNone(writer.label)

    def test_postprocess_dispatch_has_own_output(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);checkpoint=root/'best_f1.pth';checkpoint.touch()
            paths=dict(post_process.PATHS,POSTPROCESS_CHECKPOINTS=[checkpoint],OUTPUT_DIR=root/'post')
            with patch.object(post_process,'PATHS',paths), \
                 patch.object(train,'create_trainer',side_effect=AssertionError('后处理不能训练')), \
                 patch('supervised.evaluation.scene_quality.run_scene_quality') as run:
                post_process.main()
            self.assertEqual(run.call_args.args[0],checkpoint)
            self.assertEqual(run.call_args.kwargs['output_dir'],root/'post/quality')
            self.assertTrue(list((root/'post').glob('post_process_config_*.yaml')))
            self.assertFalse((root/'quality').exists())
