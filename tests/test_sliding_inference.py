"""融合推理验收：每像素一次、有效掩膜、尾块和边界连续性，全部在 CPU 上运行。"""
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.transform import from_origin
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from supervised.evaluation.sliding_inference import iter_scene_predictions, blending_weights
from supervised.load.scene_windows import SceneWindows
from supervised.evaluation.boundary_eval import boundary_regions


class PixelModel(torch.nn.Module):
    def forward(self, image):
        return image[:, :1]


class TwoClassModel(torch.nn.Module):
    def forward(self, image):
        return (image, torch.cat((-image[:, :1], image[:, :1]), dim=1))


class WindowPositionModel(torch.nn.Module):
    def forward(self, image):
        # 模拟模型在不同窗口坐标下产生的系统偏差，用于检查硬接缝是否被缓解。
        x = torch.arange(image.shape[-1], device=image.device, dtype=image.dtype)/image.shape[-1]
        return x[None, None, None, :].expand(image.shape[0], 1, image.shape[-2], -1)


def make_scene(folder, shape=(35, 47)):
    rows, cols = np.indices(shape)
    data = (rows*.01 + cols*.02).astype(np.float32)
    data[1, 1] = np.nan
    truth = (cols > shape[1]//2).astype(np.uint8)
    truth[2, 2] = 255
    profile = dict(driver='GTiff', height=shape[0], width=shape[1], count=1,
                   crs='EPSG:32650', transform=from_origin(100, 200, 3, 3))
    with rasterio.open(folder/'image.tif', 'w', dtype='float32', nodata=np.nan, **profile) as dst:
        dst.write(data, 1)
    with rasterio.open(folder/'label.tif', 'w', dtype='uint8', nodata=255, **profile) as dst:
        dst.write(truth, 1)
    return data, truth


class SlidingInferenceTests(unittest.TestCase):
    def test_skipping_input_labels_preserves_all_fused_outputs(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            data, _ = make_scene(root)
            ds = SceneWindows(root/'image.tif',root/'label.tif',window_size=32,halo=8,read_labels=False)
            item = ds[0]
            self.assertNotIn('label',item)
            self.assertIsNone(ds.label)
            ds.close()
            optimized = self.collect(root,PixelModel(),data.shape,blend='hann')
            def with_old_label_reads(*args,**kwargs):
                kwargs['read_labels'] = True
                return SceneWindows(*args,**kwargs)
            with patch('supervised.evaluation.sliding_inference.SceneWindows',side_effect=with_old_label_reads):
                original = self.collect(root,PixelModel(),data.shape,blend='hann')
            for before,after in zip(original,optimized):
                np.testing.assert_array_equal(before,after)

    def collect(self, folder, model, shape, *, stride=8, blend='gaussian', label=True,
                mean=(0.,), std=(1.,), normalized=True):
        outputs = np.zeros(shape, np.float32)
        image = np.zeros(shape, np.float32)
        truth = np.full(shape, -1, np.int64)
        valid = np.zeros(shape, bool)
        coverage = np.zeros(shape, np.uint8)
        boundary = np.zeros(shape, bool)
        for block in iter_scene_predictions(model, folder/'image.tif', folder/'label.tif' if label else None,
                window_size=32, halo=8, stride=stride, blend=blend, mean=mean, std=std,
                inputs_normalized=normalized, batch_size=3):
            top, left, height, width = block['window']
            target = np.s_[top:top+height, left:left+width]
            self.assertEqual(block['scores'].dtype, np.float32)
            self.assertEqual(block['scene_shape'], shape)
            outputs[target] = block['scores']
            image[target] = block['image'][0]
            truth[target] = block['label']
            valid[target] = block['valid_mask']
            coverage[target] += 1
            context = block['label_context']
            boundary[target] = boundary_regions(context, context>=0, (3,))[3]['inner'][block['core_slices']]
        return outputs, image, truth, valid, coverage, boundary

    def test_pointwise_preserved_partial_blocks_masks_and_boundary(self):
        for stride in (7, 8, 16):
            with self.subTest(stride=stride), TemporaryDirectory() as temp:
                root = Path(temp)
                data, truth = make_scene(root)
                got, image, labels, valid, coverage, boundary = self.collect(root, PixelModel(), data.shape, stride=stride)
                self.assertTrue((coverage == 1).all())
                np.testing.assert_array_equal(valid, np.isfinite(data))
                np.testing.assert_allclose(got[valid], data[valid], rtol=3e-7, atol=3e-7)
                np.testing.assert_allclose(image[valid], data[valid])
                np.testing.assert_array_equal(labels, np.where(valid & (truth!=255), truth.astype(np.int64), -1))
                expected = boundary_regions(labels, labels>=0, (3,))[3]['inner']
                np.testing.assert_array_equal(boundary, expected)

    def test_label_free_normalization_two_class_and_rng(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            data, _ = make_scene(root, (5, 9))
            model = TwoClassModel().train()
            before = torch.get_rng_state().clone()
            got, image, labels, valid, coverage, _ = self.collect(root, model, data.shape,
                label=False, normalized=False, mean=(.1,), std=(2.,))
            np.testing.assert_allclose(got[valid], (data[valid]-.1), atol=1e-7)
            np.testing.assert_allclose(image[valid], (data[valid]-.1)/2, atol=1e-7)
            self.assertTrue((labels==-1).all())
            self.assertTrue((coverage==1).all())
            self.assertTrue(model.training)
            self.assertTrue(torch.equal(torch.get_rng_state(), before))

    def test_overlap_reduces_window_position_jumps(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            data, _ = make_scene(root, (35, 67))
            hard = self.collect(root, WindowPositionModel(), data.shape, stride=16, blend='constant')[0]
            blended = self.collect(root, WindowPositionModel(), data.shape)[0]
            # 在相同硬拼接线比较，不把影像外缘或无效像素计入。
            seams = np.array([16, 32, 48])
            before = np.abs(hard[10, seams]-hard[10, seams-1]).mean()
            after = np.abs(blended[10, seams]-blended[10, seams-1]).mean()
            self.assertLess(after, before*.3)

    def test_rolling_fusion_matches_full_canvas_with_context_model(self):
        # 独立整幅累积作为参考，检查非局部预测、跨行batch、尾块和无效值。
        class ContextModel(torch.nn.Module):
            def forward(self, image):
                position = WindowPositionModel()(image)
                return image[:, :1] + 3*image.mean((1,2,3),keepdim=True) + position

        for stride in (7,8,16):
            with self.subTest(stride=stride), TemporaryDirectory() as temp:
                root=Path(temp)
                data,_=make_scene(root,(35,47))
                ds=SceneWindows(root/'image.tif',window_size=32,halo=8,stride=stride,read_labels=False)
                sums=np.zeros(data.shape,np.float64)
                totals=np.zeros(data.shape,np.float64)
                weight=blending_weights(16,'hann')
                model=ContextModel().eval()
                with torch.inference_mode():
                    for item in ds:
                        top,left,height,width=map(int,item['window'])
                        image=item['image'].clone()
                        image[:,~item['valid_mask']]=0
                        scores=model(image[None])[0,0,8:8+height,8:8+width].numpy()
                        w=weight[:height,:width]*item['valid_mask'][8:8+height,8:8+width].numpy()
                        region=np.s_[top:top+height,left:left+width]
                        sums[region]+=scores*w
                        totals[region]+=w
                ds.close()
                reference=np.divide(sums,totals,out=np.zeros_like(sums),where=totals>0)
                actual,_,_,valid,coverage,_=self.collect(root,model,data.shape,stride=stride,blend='hann')
                np.testing.assert_allclose(actual,reference,rtol=5e-7,atol=5e-7)
                np.testing.assert_array_equal(valid,np.isfinite(data))
                self.assertTrue((coverage==1).all())

    def test_legacy_constant_matches_direct_input(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            data, _ = make_scene(root)
            got, _, _, valid, coverage, _ = self.collect(root, PixelModel(), data.shape, stride=16, blend='constant')
            np.testing.assert_array_equal(got[valid], data[valid])
            self.assertTrue((coverage==1).all())

    def test_hann_positive_edges_and_partial_scene(self):
        self.assertTrue((blending_weights(16, 'hann') > 0).all())
        with TemporaryDirectory() as temp:
            root = Path(temp)
            data, _ = make_scene(root)
            got, _, _, valid, coverage, _ = self.collect(root, PixelModel(), data.shape, blend='hann')
            np.testing.assert_allclose(got[valid], data[valid], atol=3e-7)
            self.assertTrue((coverage==1).all())


if __name__ == '__main__':
    unittest.main()
