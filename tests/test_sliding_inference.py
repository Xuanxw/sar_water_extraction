"""融合推理验收：每像素一次、有效掩膜、尾块和边界连续性，全部在 CPU 上运行。"""
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from supervised.evaluation.sliding_inference import iter_scene_predictions, blending_weights
from supervised.load.scene_windows import SceneWindows
from supervised.evaluation.boundary_eval import boundary_regions
from datasets.speckle import read_filtered_window


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


def make_multiband_scene(folder, shape=(35, 47)):
    rows, cols = np.indices(shape)
    data = np.stack((rows*.07-cols*.03, rows*.11+cols*.05)).astype(np.float32)
    data[0, 1, 1], data[1, 2, 2] = np.nan, np.inf
    truth = ((rows+cols) % 3 == 0).astype(np.uint8)
    truth[3, 3], truth[4, 4] = 255, 2
    profile = dict(driver='GTiff', height=shape[0], width=shape[1],
                   crs='EPSG:32650', transform=from_origin(100, 200, 3, 3))
    with rasterio.open(folder/'image.tif', 'w', count=2, dtype='float32', nodata=np.nan, **profile) as dst:
        dst.write(data)
        mask = np.full(shape, 255, np.uint8)
        mask[0, -1] = 0
        dst.write_mask(mask)
    with rasterio.open(folder/'label.tif', 'w', count=1, dtype='uint8', nodata=255, **profile) as dst:
        dst.write(truth, 1)
    return data, truth


class MultibandContextModel(torch.nn.Module):
    def forward(self, image):
        value = image[:, :1]*.7-image[:, 1:2]*.3+image.mean((1, 2, 3), keepdim=True)
        value = value+WindowPositionModel()(image)
        return (image, torch.cat((-value, value), dim=1))


def legacy_reference(folder, model, *, stride, normalized, mean, std, label=True):
    """独立旧路径参考：完整输出复制 CPU、float32 整景累积、最终 halo 重读。

    仅供小合成图验收；保留旧 batch 顺序和 float32 加法顺序以要求逐位相等。
    """
    ds = SceneWindows(folder/'image.tif', folder/'label.tif' if label else None,
                      bands=(2, 1), window_size=32, halo=8, stride=stride, read_labels=False)
    means = torch.as_tensor(mean, dtype=torch.float32).reshape(1, 2, 1, 1)
    stds = torch.as_tensor(std, dtype=torch.float32).clamp_min(1e-6).reshape(1, 2, 1, 1)
    sums = np.zeros((ds.height, ds.width), np.float32)
    totals = np.zeros_like(sums)
    axis = np.hanning(16).astype(np.float32)
    weight = np.maximum(axis[:, None]*axis[None, :], 1e-3).astype(np.float32)
    try:
        loader = DataLoader(ds, batch_size=3, generator=torch.Generator().manual_seed(0))
        with torch.inference_mode():
            for batch in loader:
                image = batch['image']
                if not normalized:
                    image = (image-means)/stds
                image = image.masked_fill(~batch['valid_mask'][:, None], 0)
                prediction = model(image)
                logits = prediction[-1] if isinstance(prediction, (tuple, list)) else prediction
                scores = (logits[:, 1].float()-logits[:, 0].float()).cpu().numpy()
                for index, (top, left, height, width) in enumerate(batch['window'].tolist()):
                    core = np.s_[8:8+height, 8:8+width]
                    target = np.s_[top:top+height, left:left+width]
                    weights = weight[:height, :width]*batch['valid_mask'][index][core].numpy()
                    sums[target] += scores[index][core]*weights
                    totals[target] += weights
        blocks = []
        with rasterio.open(folder/'image.tif') as source:
            labels = rasterio.open(folder/'label.tif') if label else None
            try:
                for top in range(0, ds.height, stride):
                    height = min(stride, ds.height-top)
                    for left in range(0, ds.width, 16):
                        width = min(16, ds.width-left)
                        x0, y0 = max(0, left-8), max(0, top-8)
                        x1, y1 = min(ds.width, left+width+8), min(ds.height, top+height+8)
                        window = Window(x0, y0, x1-x0, y1-y0)
                        image, valid = read_filtered_window(source, (2, 1), window, ds.filter)
                        if not normalized:
                            image -= means.numpy()[0]
                            image /= stds.numpy()[0]
                        image[:, ~valid] = 0
                        truth = np.full(valid.shape, -1, np.int64)
                        if labels is not None:
                            values = labels.read(1, window=window, masked=True)
                            known = valid & ~np.ma.getmaskarray(values) & np.isin(values.data, (0, 1))
                            truth[known] = values.data[known]
                        core = np.s_[top-y0:top-y0+height, left-x0:left-x0+width]
                        target = np.s_[top:top+height, left:left+width]
                        score = np.divide(sums[target], totals[target],
                                          out=np.zeros((height, width), np.float32), where=totals[target]>0)
                        blocks.append(dict(window=(top, left, height, width), scores=score,
                            image=image[:, core[0], core[1]], label=truth[core],
                            valid_mask=valid[core] & (totals[target]>0), label_context=truth,
                            valid_context=valid, core_slices=core, scene_shape=(ds.height, ds.width)))
            finally:
                if labels is not None:
                    labels.close()
        return blocks
    finally:
        ds.close()


def confusion(blocks, threshold):
    result = np.zeros((2, 2), np.int64)
    for block in blocks:
        known = block['valid_mask'] & (block['label'] >= 0)
        encoded = 2*block['label'][known]+(block['scores'][known] > threshold)
        result += np.bincount(encoded, minlength=4).reshape(2, 2)
    return result


class SlidingInferenceTests(unittest.TestCase):
    def test_default_and_core_only_match_legacy_bitwise(self):
        for shape, stride, normalized, label in (((35, 47), 7, False, True),
                ((35, 47), 8, True, True), ((35, 47), 16, False, False),
                ((5, 9), 8, False, True)):
            with self.subTest(shape=shape, stride=stride, normalized=normalized), TemporaryDirectory() as temp:
                root = Path(temp)
                make_multiband_scene(root, shape)
                model = MultibandContextModel().eval()
                options = dict(stride=stride, normalized=normalized, mean=(.37, -1.13), std=(.013, 2.71), label=label)
                expected = legacy_reference(root, model, **options)
                for boundary_context in (True, False):
                    actual = list(iter_scene_predictions(model, root/'image.tif', root/'label.tif' if label else None,
                        bands=(2, 1), window_size=32, halo=8, stride=stride, blend='hann',
                        mean=options['mean'], std=options['std'], inputs_normalized=normalized,
                        batch_size=3, boundary_context=boundary_context))
                    self.assertEqual(len(actual), len(expected))
                    coverage = np.zeros(shape, np.uint8)
                    for before, after in zip(expected, actual):
                        self.assertEqual(before['window'], after['window'])
                        for key in ('scores', 'image', 'label', 'valid_mask'):
                            np.testing.assert_array_equal(before[key], after[key], err_msg=key)
                        top, left, height, width = after['window']
                        coverage[top:top+height, left:left+width] += 1
                        if boundary_context:
                            for key in ('label_context', 'valid_context'):
                                np.testing.assert_array_equal(before[key], after[key])
                            self.assertEqual(before['core_slices'], after['core_slices'])
                        else:
                            self.assertEqual(after['label_context'].shape, (height, width))
                            np.testing.assert_array_equal(after['label_context'], after['label'])
                            self.assertEqual(after['core_slices'], (slice(0, height), slice(0, width)))
                    self.assertTrue((coverage == 1).all())
                    for threshold in (-100., 0., 1., 100.):
                        np.testing.assert_array_equal(confusion(expected, threshold), confusion(actual, threshold))

    def test_core_only_avoids_final_sar_reads_and_bounds_cache(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            make_multiband_scene(root)
            allocations, transfers = [], []
            original_zeros, original_cpu = np.zeros, torch.Tensor.cpu
            def observed_zeros(shape, *args, **kwargs):
                allocations.append(shape)
                return original_zeros(shape, *args, **kwargs)
            def observed_cpu(tensor, *args, **kwargs):
                transfers.append(tuple(tensor.shape))
                return original_cpu(tensor, *args, **kwargs)
            with patch('supervised.evaluation.sliding_inference.read_filtered_window',
                       side_effect=AssertionError('core-only path must reuse input SAR')), \
                 patch('supervised.evaluation.sliding_inference.np.zeros', side_effect=observed_zeros), \
                 patch.object(torch.Tensor, 'cpu', observed_cpu):
                blocks = list(iter_scene_predictions(MultibandContextModel(), root/'image.tif', root/'label.tif',
                    bands=(2, 1), window_size=32, halo=8, stride=8, blend='hann', batch_size=3,
                    inputs_normalized=True, boundary_context=False))
            self.assertTrue(blocks)
            self.assertIn((2, 16, 47), allocations)
            self.assertNotIn((35, 47), allocations)
            self.assertNotIn((2, 35, 47), allocations)
            score_transfers = [shape for shape in transfers if len(shape) == 3]
            self.assertTrue(score_transfers)
            self.assertTrue(all(shape[-2:] == (16, 16) for shape in score_transfers))

    def test_unknown_window_dependent_filter_keeps_legacy_read_window(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            make_scene(root)
            class UnknownFilter:
                config = {'method': 'online'}
                snap = None
            def with_unknown_filter(*args, **kwargs):
                ds = SceneWindows(*args, **kwargs)
                ds.filter = UnknownFilter()
                return ds
            def window_dependent_read(source, bands, window, speckle):
                image, valid = read_filtered_window(source, bands, window, speckle)
                image += np.float32(window.height*.01+window.width*.03)
                return image, valid
            with patch('supervised.evaluation.sliding_inference.SceneWindows', side_effect=with_unknown_filter), \
                 patch('supervised.evaluation.sliding_inference.read_filtered_window', side_effect=window_dependent_read) as reader:
                options = dict(window_size=32, halo=8, stride=7, blend='hann', batch_size=3)
                before = list(iter_scene_predictions(PixelModel(), root/'image.tif', root/'label.tif', **options))
                after = list(iter_scene_predictions(PixelModel(), root/'image.tif', root/'label.tif', boundary_context=False, **options))
                self.assertEqual(reader.call_count, 2*len(before))
            for old, new in zip(before, after):
                for key in ('scores', 'image', 'label', 'valid_mask'):
                    np.testing.assert_array_equal(old[key], new[key])

    def test_nonfinite_halo_is_still_rejected(self):
        class InvalidHaloModel(PixelModel):
            def forward(self, image):
                output = super().forward(image).clone()
                output[:, :, 0, 0] = float('inf')
                return output
        with TemporaryDirectory() as temp:
            root = Path(temp)
            make_scene(root)
            for boundary_context in (True, False):
                model = InvalidHaloModel().train()
                with self.assertRaises(FloatingPointError):
                    list(iter_scene_predictions(model, root/'image.tif', window_size=32, halo=8,
                                               boundary_context=boundary_context))
                self.assertTrue(model.training)

    def test_early_close_releases_readers_workers_and_restores_model(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            make_scene(root)
            original_open = rasterio.open
            for boundary_context in (True, False):
                opened, shutdown = [], []
                def track_open(*args, **kwargs):
                    reader = original_open(*args, **kwargs)
                    opened.append(reader)
                    return reader
                def tracked_loader(*args, **kwargs):
                    iterator = iter(DataLoader(*args, **kwargs))
                    iterator._shutdown_workers = lambda: shutdown.append(True)
                    return iterator
                with patch('supervised.evaluation.sliding_inference.rasterio.open', side_effect=track_open), \
                     patch('supervised.evaluation.sliding_inference.DataLoader', side_effect=tracked_loader):
                    model = PixelModel().train()
                    stream = iter_scene_predictions(model, root/'image.tif', root/'label.tif',
                        window_size=32, halo=8, stride=8, batch_size=3, boundary_context=boundary_context)
                    block = next(stream)
                    image = block['image'].copy()
                    self.assertFalse(model.training)
                    stream.close()
                    self.assertTrue(model.training)
                    self.assertEqual(shutdown, [True])
                    self.assertTrue(all(reader.closed for reader in opened))
                    np.testing.assert_array_equal(block['image'], image)

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
