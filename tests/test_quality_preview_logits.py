"""质量图先抽样logits，与旧整块FP64概率保持一致。"""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import torch.nn.functional as F

from supervised.evaluation import training_quality as quality


def full_probability(logits):
    if logits.ndim == 2:
        return logits.detach().double().sigmoid().float()
    if logits.shape[0] == 1:
        return logits.detach()[None, :, :, :][:, 0].double().sigmoid().float()[0]
    return logits.detach()[None, :, :, :].double().softmax(dim=1)[:, 1].float()[0]


def counts_for(label, prediction):
    known = label >= 0
    return torch.bincount(2*label[known]+prediction[known], minlength=4).cpu().numpy()


class QualityPreviewLogitsTests(unittest.TestCase):
    def assert_same(self, expected, actual):
        for key in ('image', 'label', 'probability', 'prediction', 'counts'):
            np.testing.assert_array_equal(getattr(expected, key), getattr(actual, key), err_msg=key)

    def test_scene_windows_match_full_probability(self):
        devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
        for device in devices:
            for channels in (0, 1, 2):
                for dtype in (torch.float16, torch.float32, torch.float64):
                    for shared_counts in (False, True):
                        with self.subTest(device=device, channels=channels, dtype=dtype, counts=shared_counts), TemporaryDirectory() as temp:
                            root = Path(temp)
                            old = quality.ValidationQualityWriter(object(), [], root/'old', 1, max_side=5)
                            new = quality.ValidationQualityWriter(object(), [], root/'new', 1, max_side=5)
                            values = torch.arange(37*53, device=device).reshape(37, 53)
                            single = ((values.double() % 53 - 26) / 7).to(dtype)
                            logits = single if channels == 0 else single[None]
                            if channels == 2:
                                logits = torch.stack((single, -single*.7))
                            logits.requires_grad_(True)
                            original_logits = logits.detach().clone()
                            probability = full_probability(logits)
                            prediction = (single >= 0).long() if channels < 2 else logits.detach().argmax(dim=0)
                            label = values % 3 - 1
                            valid = values % 11 != 0
                            label[~valid] = -1
                            image = values.float()/100
                            for top in range(0, 37, 7):
                                for left in range(0, 53, 13):
                                    region = np.s_[top:min(top+7, 37), left:min(left+13, 53)]
                                    truth, pred = label[region], prediction[region]
                                    options = dict(scene_window=(top, left, *truth.shape), scene_shape=(37, 53))
                                    old.add('scene', image[region], truth, probability[region], pred, valid[region], **options)
                                    counts = counts_for(truth, pred) if shared_counts else None
                                    new.add('scene', image[region], truth, None, pred, valid[region],
                                            **options, counts=counts, logits=logits[(..., *region)])
                            self.assert_same(old, new)
                            np.testing.assert_array_equal(new.counts, counts_for(label, prediction))
                            torch.testing.assert_close(logits.detach(), original_logits, rtol=0, atol=0)
                            self.assertIsNone(logits.grad)

    def test_halo_windows_match_full_probability_and_counts(self):
        for channels in (1, 2):
            for shared_counts in (False, True):
                with self.subTest(channels=channels, counts=shared_counts), TemporaryDirectory() as temp:
                    root = Path(temp)
                    height, width, halo = 17, 23, 3
                    windows = [(0, top, left, min(6, height-top), min(7, width-left))
                               for top in range(0, height, 6) for left in range(0, width, 7)]
                    names = [f'window_{i}' for i in range(len(windows))]
                    dataset = SimpleNamespace(root=root, windows=windows, names=names, halo=halo,
                                              scenes=[dict(scene_id='scene', split='val', name='scene.tif')])
                    old = quality.ValidationQualityWriter(dataset, range(len(windows)), root/'old', 1, max_side=6)
                    new = quality.ValidationQualityWriter(dataset, range(len(windows)), root/'new', 1, max_side=6)
                    values = torch.arange(height*width).reshape(height, width)
                    image = values.float()/10
                    label = values % 3 - 1
                    valid = values % 13 != 0
                    label[~valid] = -1
                    single = (image - 15)/3
                    logits = single[None] if channels == 1 else torch.stack((single, -single*.3))
                    prediction = (single >= 0).long() if channels == 1 else logits.argmax(dim=0)
                    source = MagicMock()
                    source.__enter__.return_value.shape = (height, width)
                    with patch.object(quality.rasterio, 'open', return_value=source):
                        for name, (_, top, left, h, w) in zip(names, windows):
                            region = np.s_[top:top+h, left:left+w]
                            padding = (halo, halo, halo, halo)
                            chunk_image = F.pad(image[region], padding, value=-100)
                            chunk_label = F.pad(label[region], padding, value=-1)
                            chunk_valid = F.pad(valid[region], padding, value=False)
                            chunk_pred = F.pad(prediction[region], padding, value=1)
                            chunk_logits = F.pad(logits[(..., *region)], padding, value=500)
                            probability = full_probability(chunk_logits)
                            old.add(name, chunk_image, chunk_label, probability, chunk_pred, chunk_valid)
                            counts = counts_for(label[region], prediction[region]) if shared_counts else None
                            new.add(name, chunk_image, chunk_label, None, chunk_pred, chunk_valid,
                                    counts=counts, logits=chunk_logits)
                    self.assert_same(old, new)
                    np.testing.assert_array_equal(new.counts, counts_for(label, prediction))

    def test_empty_preview_keeps_counts_without_probability_work(self):
        for shared_counts in (False, True):
            with self.subTest(counts=shared_counts), TemporaryDirectory() as temp:
                writer = quality.ValidationQualityWriter(object(), [], Path(temp), 1, max_side=2)
                label = torch.ones((1, 1), dtype=torch.long)
                counts = np.array([0, 0, 0, 1]) if shared_counts else None
                writer.add('scene', object(), label, None, label, object(),
                           scene_window=(1, 1, 1, 1), scene_shape=(32, 32), counts=counts, logits=object())
                np.testing.assert_array_equal(writer.counts, [0, 0, 0, 1])
                self.assertTrue(np.isnan(writer.probability).all())

    def test_scene_limit_skips_logits(self):
        with TemporaryDirectory() as temp, patch.object(quality, 'save_prediction_figure'):
            writer = quality.ValidationQualityWriter(object(), [], Path(temp), 1, max_side=2, max_scenes=1)
            label = torch.zeros((3, 3), dtype=torch.long)
            image = label.float()
            valid = torch.ones_like(label, dtype=torch.bool)
            writer.add('first', image, label, None, label, valid, logits=image)
            for name in ('second', 'second', 'third'):
                writer.add(name, object(), label, None, object(), object(), logits=object())
            self.assertEqual(len(writer.rows), 1)
            self.assertFalse(writer.selected)

    def test_probability_stays_compatible_when_logits_are_supplied(self):
        with TemporaryDirectory() as temp:
            writer = quality.ValidationQualityWriter(object(), [], Path(temp), 1, max_side=2)
            label = torch.zeros((3, 3), dtype=torch.long)
            probability = torch.full((3, 3), .25)
            writer.add('scene', probability, label, probability, label, torch.ones_like(label, dtype=torch.bool),
                       logits=object())
            np.testing.assert_array_equal(writer.probability, np.full((2, 2), .25, dtype=np.float32))

    def test_fp64_conversion_only_receives_preview_pixels(self):
        for channels in (1, 2):
            with self.subTest(channels=channels), TemporaryDirectory() as temp:
                writer = quality.ValidationQualityWriter(object(), [], Path(temp), 1, max_side=5)
                label = torch.zeros((19, 31), dtype=torch.long)
                logits = torch.zeros((channels, 19, 31))
                converted = []
                original_double = torch.Tensor.double

                def record_double(tensor, *args, **kwargs):
                    converted.append(tensor.numel())
                    return original_double(tensor, *args, **kwargs)

                with patch.object(torch.Tensor, 'double', record_double):
                    writer.add('scene', label.float(), label, None, label, torch.ones_like(label, dtype=torch.bool),
                               logits=logits, counts=np.array([label.numel(), 0, 0, 0]))
                self.assertEqual(converted, [channels*len(writer.ys)*len(writer.xs)])
                self.assertLess(converted[0], logits.numel())


if __name__ == '__main__':
    unittest.main()
