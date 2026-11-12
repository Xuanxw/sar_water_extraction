"""效率改动集成验收：训练入口的快速验证必须保留指标和概览。"""
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / 'src'), str(Path(__file__).resolve().parent)]
import test_sar_scene_dataset as fixture
from supervised import train
from supervised.evaluation import sliding_inference, training_quality, model_eval


class EfficiencyTests(unittest.TestCase):
    def test_training_validation_fast_path_keeps_metrics_and_quality(self):
        with TemporaryDirectory() as temp, patch.object(torch.cuda, 'is_available', return_value=False):
            root = Path(temp)
            fixture.SceneDatasetTests().prepare(root)
            trainer = train.WaterSegmentationTrainer(fixture.TinySceneNet, {}, dict(model_name='tiny'),
                root/'images', root/'labels', root/'run', bands=(1,), batch_size=2, epochs=1, loss_name='bce',
                scene_options=dict(cache_dir=root/'cache', window_size=32, halo=8, num_workers=0),
                validation_inference=dict(stride=8, blend='hann', num_workers=0),
                quality_options=dict(enabled=True, interval=1, max_scenes=None, max_side=20))
            captured = []
            def capture(image, label, probability, prediction, counts, *args, **kwargs):
                captured.append([x.copy() for x in (image, label, probability, prediction, counts)])
            original = sliding_inference.iter_scene_predictions
            def old_context(*args, **kwargs):
                kwargs['boundary_context'] = True
                return original(*args, **kwargs)
            try:
                with patch.object(training_quality, 'save_prediction_figure', side_effect=capture):
                    with patch.object(sliding_inference, 'iter_scene_predictions', side_effect=old_context):
                        before = trainer.run_epoch(1, 'val', trainer.val_loader)
                    rng_before = torch.random.get_rng_state().clone()
                    after = trainer.run_epoch(1, 'val', trainer.val_loader)
                    torch.testing.assert_close(torch.random.get_rng_state(), rng_before)
                for key in ('tn', 'fp', 'fn', 'tp', 'valid_pixels', 'loss', 'f1', 'iou'):
                    self.assertEqual(before[key], after[key], key)
                self.assertEqual(len(captured), 2)
                for old, new in zip(*captured):
                    np.testing.assert_array_equal(old, new)
            finally:
                trainer.dataset.close()

    def test_scan_only_accepts_known_equivalent_upgrade(self):
        old = dict(protocol={'stride':384,'blend':'hann'}, thresholds=[.2,.5,.8], models={'six':'fixed'},
                   code_hash=model_eval.LEGACY_SCAN_HASHES[0],
                   inference_code_hash=model_eval.LEGACY_SCAN_HASHES[1])
        current = dict(old, code_hash='current', inference_code_hash=model_eval.OPTIMIZED_INFERENCE_HASH)
        self.assertTrue(model_eval.same_scan_protocol(old, current))
        self.assertTrue(model_eval.same_scan_protocol(current, current))
        self.assertFalse(model_eval.same_scan_protocol(old, dict(current, thresholds=[.3,.5,.8])))
        self.assertFalse(model_eval.same_scan_protocol(old, dict(current, models={'six':'changed'})))
        self.assertFalse(model_eval.same_scan_protocol(old, dict(current, inference_code_hash='unknown')))
        self.assertFalse(model_eval.same_scan_protocol(dict(old, code_hash='unknown'), current))


if __name__ == '__main__':
    unittest.main()
