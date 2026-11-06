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
            with patch.object(bn,'RUN_DIR',root),patch.object(bn,'ROOT',root),patch.object(bn,'load_model',return_value=model):
                runner=bn.BNCalibrator()
                self.assertTrue(set(runner.calibration_files)<=set(c['train_files']))
                self.assertFalse(set(runner.calibration_files)&set(c['val_files']))
                before=runner.evaluate();runner.calibrate_layers();after=runner.evaluate()
                self.assertEqual(before['valid_pixels'],after['valid_pixels'])
                self.assertEqual(after['valid_pixels'],1147)
                for k,v in model.named_parameters():torch.testing.assert_close(v,params[k])
                runner.save_result(before,after)
                saved=torch.load(next((root/'bn_calibration').rglob('calibrated.pth')),weights_only=True)
                self.assertEqual(saved['resume_phase'],'bn_calibrated_inference_only')
                runner.train_loader.dataset.dataset.close()

if __name__=='__main__':unittest.main()
