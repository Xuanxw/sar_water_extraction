"""滤波数值、分块一致性、缓存隔离和训练/推理一致性。使用合成数据，不读正式训练集。"""
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.windows import Window
import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]/'src'), str(Path(__file__).resolve().parent)]
from datasets.speckle import SpeckleFilter, read_filtered_window
from datasets.sar_scene_dataset import SARSceneDataset
from supervised.train import WaterSegmentationTrainer
from supervised.evaluation import postprocess, model_eval
import test_sar_scene_dataset as fixtures

CONFIG = dict(method='lee', input_domain='amplitude', window_size=3, enl=1.0)


class TinyFilteredNet(torch.nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, num_classes, 1)

    def forward(self, image):
        return self.conv(image)


class SpeckleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_known_lee_value_masks_and_amplitude_conversion(self):
        power = np.ones((1, 3, 3), dtype='float32')
        power[0, 1, 1] = 9
        valid = np.ones((3, 3), dtype=bool)
        f = SpeckleFilter(dict(CONFIG, input_domain='intensity'))
        # 9个观测：均值17/9、样本方差64/9，按标准Lee公式手算中心结果。
        mean, variance = 17/9, 64/9
        expected = mean + (1 - mean**2/variance)*(9-mean)
        self.assertAlmostEqual(float(f(power, valid)[0,1,1]), expected, places=5)
        np.testing.assert_allclose(SpeckleFilter(CONFIG)(np.sqrt(power), valid)**2,
                                   f(power, valid), rtol=2e-6)
        valid[0,0] = False
        power[0,0,0] = np.nan
        result = f(power, valid)
        self.assertTrue(np.isfinite(result).all())
        self.assertEqual(result[0,0,0], 0)
        self.assertIs(SpeckleFilter()(power, valid), power)
        zeros = np.zeros((2,5,5), dtype='float32')
        np.testing.assert_array_equal(f(zeros, np.ones((5,5),bool)), zeros)
        np.testing.assert_array_equal(f(zeros, np.zeros((5,5),bool)), zeros)
        with self.assertRaises(ValueError):
            f(-np.ones((1,3,3)), np.ones((3,3),bool))

    def test_reduces_uniform_speckle_and_preserves_gain(self):
        data = np.random.default_rng(42).gamma(1, 3, (2,128,128)).astype('float32')
        valid = np.ones((128,128),bool)
        f = SpeckleFilter(dict(CONFIG,input_domain='intensity'))
        result = f(data, valid)
        self.assertLess(result.var(), data.var())
        self.assertLess(abs(result.mean()-data.mean())/data.mean(), .02)
        np.testing.assert_allclose(f(data*5,valid),result*5,rtol=2e-6,atol=1e-5)

    def test_spawn_worker_restores_dataset_created_before_filter_was_added(self):
        """复现旧主进程对象没有新属性、新spawn worker却导入新版读取方法的情形。"""
        import pickle
        from torch.utils.data import DataLoader

        with TemporaryDirectory() as temp:
            root=Path(temp);fixtures.SceneDatasetTests().prepare(root)
            ds=SARSceneDataset(root,root/'cache',window_size=32,halo=8)
            expected=ds[0]['image'].clone();ds.close()
            del ds.speckle_filter  # 模拟新增滤波之前已在内存中的数据集状态。
            restored=pickle.loads(pickle.dumps(ds))
            self.assertIsNone(restored.speckle_filter.config)
            batches=list(DataLoader(ds,batch_size=2,num_workers=1,multiprocessing_context='spawn'))
            torch.testing.assert_close(batches[0]['image'][0],expected,rtol=0,atol=0)
            ds.close();restored.close()
            enabled=SARSceneDataset(root,root/'cache',window_size=32,halo=8,speckle=CONFIG)
            self.assertEqual(pickle.loads(pickle.dumps(enabled)).speckle_filter.config,CONFIG)
            enabled.close()

    def test_window_matches_whole_scene_and_normalization_cache_is_separate(self):
        with TemporaryDirectory() as temp:
            root=Path(temp)
            arrays=fixtures.SceneDatasetTests().prepare(root)
            x=arrays['train'][0]
            f=SpeckleFilter(CONFIG)
            good=np.isfinite(x)
            whole=f(np.where(good,x,0)[None],good)
            with rasterio.open(root/'images/train/train.tif') as src:
                for x0,y0,w,h in [(0,0,16,16),(5,7,20,20),(31,20,16,15)]:
                    data,mask=read_filtered_window(src,(1,),Window(x0,y0,w,h),f)
                    np.testing.assert_allclose(data,whole[:,y0:y0+h,x0:x0+w],rtol=1e-6,atol=1e-6)
                    np.testing.assert_array_equal(mask,good[y0:y0+h,x0:x0+w])
            plain=SARSceneDataset(root,root/'cache',window_size=32,halo=8)
            filtered=SARSceneDataset(root,root/'cache',window_size=32,halo=8,speckle=CONFIG)
            self.assertEqual(plain.cache_key,filtered.cache_key)  # 标签索引不重复构建。
            label=plain[0]['label'];plain.close()
            np.testing.assert_array_equal(label,filtered[0]['label'])
            filtered.close()  # 两个GDAL Env按各自生命周期关闭，不交叉嵌套。
            a=plain.normalization(plain.train_ids)
            b=filtered.normalization(filtered.train_ids)
            self.assertEqual(len(list((root/'cache').glob('normalization_*'))),2)
            self.assertLess(b[0][0],3)  # 不使用val=1000的像素。
            self.assertEqual(a[2],b[2])  # 同一训练窗口抽样，控制变量。
            plain.close();filtered.close()

    def test_train_checkpoint_resume_and_inference_recover_filter(self):
        with TemporaryDirectory() as temp, patch.object(torch.cuda,'is_available',return_value=False):
            root=Path(temp);fixtures.SceneDatasetTests().prepare(root)
            kwargs=dict(model_class=TinyFilteredNet,model_kwargs={'num_classes':1},model_info=dict(model_name='tiny'),
                        image_dir=root/'images',label_dir=root/'labels',output_dir=root/'runs',
                        bands=(1,),batch_size=2,epochs=1,loss_name='bce',speckle=CONFIG,
                        scene_options=dict(cache_dir=root/'cache',window_size=32,halo=8,num_workers=0,windows_per_epoch=2))
            trainer=WaterSegmentationTrainer(**kwargs)
            trainer.train();trainer.dataset.close()
            path=trainer.run_dir/'last.pth'
            ckpt=torch.load(path,weights_only=True,map_location='cpu')
            self.assertEqual(ckpt['speckle'],CONFIG)
            resumed=WaterSegmentationTrainer(**kwargs,resume_checkpoint=path)
            self.assertEqual(resumed.start_epoch,2)
            resumed.dataset.close()
            with self.assertRaisesRegex(ValueError,'滤波配置'):
                WaterSegmentationTrainer(**dict(kwargs,speckle=None),resume_checkpoint=path)
            with patch.object(model_eval,'load_model',return_value=trainer.model), \
                 patch.object(model_eval,'SARSceneDataset',side_effect=lambda folder,cache,**opts:
                              SARSceneDataset(folder,root/'cache',**opts)):
                measured=model_eval.evaluate(path,use_amp=False)
            self.assertAlmostEqual(measured['loss'],trainer.phase_metrics['val']['loss'],places=6)

            # 质量输出/阈值对照通过WaterEnsemble读取原始像素，必须恢复相同滤波。
            with patch.object(postprocess,'load_model',return_value=torch.nn.Identity()):
                ensemble=postprocess.WaterEnsemble([path],use_amp=False)
            raw=torch.arange(25,dtype=torch.float32).reshape(1,1,5,5)
            mask=torch.ones((1,5,5),dtype=torch.bool)
            _,actual=ensemble.predict(raw,mask)
            expected=torch.from_numpy(SpeckleFilter(CONFIG)(raw[0].numpy(),mask[0].numpy()))[None]
            expected=((expected-ckpt['mean'].view(1,1,1,1))/ckpt['std'].view(1,1,1,1))[:,0].double().sigmoid()
            torch.testing.assert_close(actual,expected)


if __name__=='__main__':
    unittest.main()
