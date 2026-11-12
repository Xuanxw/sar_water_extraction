"""BN对照只用训练窗口，支持单通道BCE，校准不改变可学习权重。"""
import hashlib
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import torch
from torch import nn
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'src'),str(Path(__file__).resolve().parents[1]/'scripts'),str(Path(__file__).resolve().parent)]
import calibrate_bn as bn
import test_sar_scene_dataset as fixture
from supervised.load.sar_scene_dataset import SARSceneDataset

class BNTests(unittest.TestCase):
    def test_scene_calibration_preserves_parameters_and_split(self):
        with TemporaryDirectory() as t, patch.object(torch.cuda,'is_available',return_value=False):
            root=Path(t);fixture.SceneDatasetTests().prepare(root)
            ds=SARSceneDataset(root,root/'cache',window_size=32,halo=8)
            model=nn.Sequential(nn.Conv2d(1,2,1),nn.BatchNorm2d(2),nn.ReLU(),nn.Conv2d(2,1,1))
            c=dict(model_state_dict=model.state_dict(),data_layout='full_scene',image_dir=str(root/'images'),
                label_dir=str(root/'labels'),bands=(1,),window_size=32,window_halo=8,
                train_files=[ds.names[i] for i in ds.train_ids],val_files=[ds.names[i] for i in ds.val_ids],
                split_manifest_sha256=hashlib.sha256((root/'split_manifest.csv').read_bytes()).hexdigest(),
                mean=torch.tensor([0.]),std=torch.tensor([1.]),batch_size=2,ignore_index=-1,val_loss=1.,epoch=20)
            ds.close();torch.save(c,root/'last.pth')
            params={k:v.clone() for k,v in model.named_parameters()}
            with patch.object(bn,'RUN_DIR',root),patch.object(bn,'ROOT',root),patch.object(bn,'load_model',return_value=model),patch.object(bn,'CHECKPOINT_NAME','last.pth'),patch.object(bn,'EXPECTED_EPOCH',20),patch.object(bn,'CACHE_DIR',root/'cache'),patch.object(bn,'RESUME_DIR',None):
                runner=bn.BNCalibrator()
                self.assertTrue(set(runner.calibration_files)<=set(c['train_files']))
                self.assertFalse(set(runner.calibration_files)&set(c['val_files']))
                before=runner.evaluate()
                # 校准对照必须能够恢复新Hann协议，不能继续使用旧的硬拼接计分。
                runner.checkpoint['val_scene_files']=[r['name'] for r in runner.dataset.scenes if r['split']=='val']
                runner.validation_inference=dict(stride=8,blend='hann',num_workers=0)
                runner.checkpoint['validation_inference']=runner.validation_inference
                fused=runner.evaluate()
                self.assertEqual(before['valid_pixels'],fused['valid_pixels'])
                self.assertTrue(runner.scene_metrics)
                before=fused
                runner.calibrate_layers();after=runner.evaluate()
                self.assertEqual(before['valid_pixels'],after['valid_pixels'])
                self.assertEqual(after['valid_pixels'],1147)
                for k,v in model.named_parameters():torch.testing.assert_close(v,params[k])
                runner.save_result(before,after)
                saved=torch.load(next((root/'bn_calibration').rglob('calibrated.pth')),weights_only=True)
                self.assertEqual(saved['resume_phase'],'bn_calibrated_inference_only')
                self.assertEqual(saved['bn_calibration']['validation_inference'],runner.validation_inference)
                runner.run()
                self.assertTrue((runner.output_dir/'progress.json').is_file())
                self.assertTrue((runner.output_dir/'before_per_scene.csv').is_file())
                self.assertTrue((runner.output_dir/'after_per_scene.csv').is_file())
                self.assertIn('model_state_dict',runner.checkpoint)
                self.assertNotIn('bn_calibration',runner.checkpoint)
                # 完成后再次启动不推理；只有after中断时只补该阶段，before不重写。
                torch.save(runner.checkpoint,root/'last.pth')
                before_path=runner.output_dir/'before_per_scene.csv'
                before_bytes=before_path.read_bytes()
                with patch.object(bn,'RESUME_DIR',runner.output_dir):
                    resumed=bn.BNCalibrator()
                    with patch.object(resumed,'calibrate_layers',side_effect=AssertionError('不能重新校准')), \
                         patch.object(resumed,'evaluate_blended',side_effect=AssertionError('不能重复评价')):
                        resumed.run()
                    self.assertEqual(before_path.read_bytes(),before_bytes)
                    after_path=runner.output_dir/'after_per_scene.csv'
                    after_path.write_text(after_path.read_text(encoding='utf-8-sig').splitlines()[0]+'\n',encoding='utf-8-sig')
                    resumed=bn.BNCalibrator()
                    with patch.object(resumed,'calibrate_layers',side_effect=AssertionError('不能重新校准')), \
                         patch.object(resumed,'evaluate_blended',wraps=resumed.evaluate_blended) as evaluate:
                        resumed.run()
                        self.assertEqual(evaluate.call_count,1)
                        self.assertEqual(resumed.phase,'after')
                    self.assertEqual(before_path.read_bytes(),before_bytes)


if __name__=='__main__':unittest.main()
