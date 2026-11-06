"""用微型四头模型核对监督权重、验证口径和困难评分，不读取真实影像。"""
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from torch.nn import functional as F

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"),
                str(Path(__file__).resolve().parents[1] / "scripts")]
from supervised import train
from supervised.models.checkpoint import load_model


WEIGHTS = (1.0, 0.5, 0.5, 2.0)
LABEL = torch.tensor([[1, 0], [-1, 1]])


class TinyDataset(torch.utils.data.Dataset):
    """四张小图，固定标签使梯度可以直接手算。"""

    def __init__(self, image_dir, label_dir=None, bands=(1, 2)):
        self.image_paths = [Path(image_dir) / f"{i}.tif" for i in range(4)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2) + index
        return dict(image=image, label=LABEL.clone(),
                    valid_mask=LABEL != -1, name=f"{index}.tif")


class FourIndependentHeads(torch.nn.Module):
    """每个像素都是独立参数，便于检查被忽略像素与各头的梯度。"""

    def __init__(self, **kwargs):
        super().__init__()
        self.heads = torch.nn.ParameterList([
            torch.nn.Parameter(torch.tensor([[[[-1., .5], [3., 1.]]]]) + offset)
            for offset in (0., .2, -.3, .7)
        ])

    def forward(self, image):
        return tuple(head.expand(image.shape[0], -1, -1, -1) for head in self.heads)


class HeadSupervisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def make_trainer(self, folder, weights):
        with patch("datasets.dfc_sample.s1_dataset.WaterDataset", TinyDataset), \
             patch.object(torch.cuda, "is_available", return_value=False):
            trainer = train.WaterSegmentationTrainer(
                FourIndependentHeads, {}, {"model_name": "pvt_uafnet"},
                Path(folder) / "images", Path(folder) / "labels", Path(folder) / "outputs",
                batch_size=2, val_ratio=.5, epochs=1, learning_rate=0,
                loss_name="bce", sampling="hard", train_augmentation=False,
                head_loss_weights=weights)
        # lr=0 保留 logits，仍执行真实 backward/step，方便独立核对记录值。
        trainer.recent_train_files = []
        trainer.prepare_train_sampling(1)
        return trainer

    def test_actual_epoch_matches_manual_gradient_and_keeps_scores_unweighted(self):
        results = []
        for weights in (None, (1., 1., 1., 1.), WEIGHTS):
            with self.subTest(weights=weights), TemporaryDirectory() as folder:
                trainer = self.make_trainer(folder, weights)
                actual = trainer.run_epoch(1, "train", trainer.train_loader)
                head_losses = []
                valid = LABEL != -1
                for head, weight in zip(trainer.model.heads, weights or (1.,) * 4):
                    score = head.detach()[0, 0]
                    # softplus(z)-y*z 是 BCE 的独立表达；3 个有效像素，标签两图相同。
                    head_losses.append((F.softplus(score[valid]) -
                                        LABEL[valid] * score[valid]).mean())
                    expected_grad = torch.zeros_like(score)
                    expected_grad[valid] = weight * (score[valid].sigmoid() -
                                                    LABEL[valid]) / valid.sum()
                    torch.testing.assert_close(head.grad[0, 0], expected_grad)
                    self.assertEqual(head.grad[0, 0, 1, 0].item(), 0.)
                expected_loss = sum(w * value for w, value in zip(weights or (1.,) * 4, head_losses))
                self.assertAlmostEqual(actual["loss"], expected_loss.item(), places=6)
                torch.testing.assert_close(trainer.sample_losses,
                                           head_losses[-1].expand(len(trainer.train_set)))
                before = trainer.sample_losses.clone()
                validation = trainer.run_epoch(1, "val", trainer.val_loader)
                self.assertAlmostEqual(validation["loss"], head_losses[-1].item(), places=6)
                torch.testing.assert_close(before, trainer.sample_losses)
                results.append((validation["loss"], trainer.sample_losses.clone()))
        for loss, scores in results[1:]:
            self.assertAlmostEqual(loss, results[0][0], places=7)
            torch.testing.assert_close(scores, results[0][1])

    def test_checkpoint_records_weights_without_changing_model_reload(self):
        with TemporaryDirectory() as folder:
            trainer = self.make_trainer(folder, WEIGHTS)
            trainer.train()
            checkpoint = torch.load(trainer.run_dir / "last.pth", weights_only=True)
            self.assertEqual(tuple(checkpoint["head_loss_weights"]), WEIGHTS)
            self.assertEqual(checkpoint["sampling_config"]["mode"], "hard")
            image = torch.zeros(2, 2, 2, 2)
            # load_model 的模型工厂换成微型网络；实际执行严格 state_dict 加载。
            with patch("supervised.models.pvt.PVTUANet", FourIndependentHeads):
                restored = load_model(checkpoint)
                legacy = dict(checkpoint)
                legacy.pop("head_loss_weights")
                old_restored = load_model(legacy)
            for expected, current, old in zip(trainer.model(image), restored(image), old_restored(image)):
                torch.testing.assert_close(expected, current, rtol=0, atol=0)
                torch.testing.assert_close(expected, old, rtol=0, atol=0)

    def test_wrong_head_count_cannot_silently_drop_a_prediction(self):
        with TemporaryDirectory() as folder, self.assertRaises(ValueError):
            trainer = self.make_trainer(folder, (1., .5, 2.))
            trainer.run_epoch(1, "train", trainer.train_loader)

    def test_factory_changes_only_head_supervision_and_output_directory(self):
        options = []
        for weights in ((1., 1., 1., 1.), WEIGHTS):
            with patch.multiple(train, METHOD="uafnet", DATA_LAYOUT="patch", POLARIZATIONS=("VV","VH"), POLARIZATION_BANDS={"VV":1,"VH":2}), patch.object(train, "UAFNET_BACKBONE", "pvt_v2_b2.in1k"), patch.object(train, "UAFNET_REFINEMENT_DECODER", "uafm"), patch.object(train, "UAFNET_HIGH_RESOLUTION_REFINEMENT", False), patch.object(train, "UAFNET_MBDC_DILATIONS", None), patch.object(train, "LOSS_NAME", "bce"), \
                 patch.object(train, "SAMPLING", "hard"), patch.object(train, "OUTPUT_DIR", None), \
                 patch.object(train, "UAFNET_FINAL_RESIDUAL", True), \
                 patch.object(train, "HEAD_LOSS_WEIGHTS", weights), \
                 patch.object(train, "WaterSegmentationTrainer") as factory:
                train.create_trainer()
                options.append(factory.call_args.kwargs)
        baseline, weighted = options
        for key in baseline:
            if key not in ("head_loss_weights", "model_info", "output_dir"):
                self.assertEqual(baseline[key], weighted[key], key)
        before, after = dict(baseline["model_info"]), dict(weighted["model_info"])
        self.assertEqual(before.pop("supervision"), "sum_bce_all_train_heads; final_head_validation")
        self.assertEqual(after.pop("supervision"), "weighted_sum_bce_all_train_heads; final_head_validation")
        self.assertEqual(before, after)
        self.assertEqual(tuple(weighted["head_loss_weights"]), WEIGHTS)
        self.assertEqual(baseline["output_dir"].name, "pvt_uafnet_hard_sampling_residual")
        self.assertEqual(weighted["output_dir"].name, "pvt_uafnet_hard_sampling_residual_head_weights")

    def test_other_methods_keep_equal_supervision(self):
        for method in ("unet", "pvt", "resnet50", "uanet"):
            with self.subTest(method=method), patch.multiple(train, METHOD=method, DATA_LAYOUT="patch", POLARIZATIONS=("VV","VH"), POLARIZATION_BANDS={"VV":1,"VH":2}), \
                 patch.object(train, "LOSS_NAME", "ce"), patch.object(train, "SAMPLING", "random"), \
                 patch.object(train, "HEAD_LOSS_WEIGHTS", WEIGHTS), \
                 patch.object(train, "WaterSegmentationTrainer") as factory:
                train.create_trainer()
                self.assertIsNone(factory.call_args.kwargs["head_loss_weights"])
                self.assertNotIn("head_weights", factory.call_args.kwargs["output_dir"].name)


if __name__ == "__main__":
    unittest.main()
