"""类别权重验收：仅用train计数、未知像素零梯度、val保持原始BCE。"""
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from torch.nn import functional as F

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parent)]
from supervised import train
from supervised.models.losses import WaterSegmentationLoss
import test_sar_scene_dataset as scene_tests

# 测试显式加载样例配置，正式脚本仅使用命令行提供的文件。
train.configure(Path(__file__).resolve().parents[1] / 'configs/train.yaml')


class BCTrainingWeightTests(unittest.TestCase):
    def test_weighted_loss_balances_total_gradient_and_ignores_unknown(self):
        # 三个水体、一个非水体，逆频率权重使两类总梯度在零logit时相等。
        score = torch.zeros(1, 1, 1, 5, requires_grad=True)
        label = torch.tensor([[[1, 1, 1, 0, -1]]])
        loss = WaterSegmentationLoss("bce", class_weights=(2., 2./3))(score, label)
        torch.testing.assert_close(loss, torch.tensor(2.).log())
        loss.backward()
        torch.testing.assert_close(score.grad[..., :3].sum(), -score.grad[..., 3].sum())
        self.assertEqual(score.grad[..., 4].item(), 0)

    def test_train_counts_and_unweighted_validation(self):
        with TemporaryDirectory() as temp, patch.object(torch.cuda, "is_available", return_value=False):
            root = Path(temp)
            arrays = scene_tests.SceneDatasetTests().prepare(root)
            trainer = train.WaterSegmentationTrainer(
                scene_tests.TinySceneNet, {}, dict(model_name="tiny"), root/"images", root/"labels", root/"runs",
                bands=(1,), loss_name="bce", sampling="random", class_weight_mode="train_pixels",
                scene_options=dict(cache_dir=root/"cache", window_size=32, halo=8,
                                   normalization_windows=2, num_workers=0, windows_per_epoch=4))
            water = int((arrays["train"][1] == 1).sum())
            background = int((arrays["train"][1] == 0).sum())
            self.assertEqual(trainer.split_info["train_class_pixels"], dict(water=water, background=background))
            self.assertEqual(trainer.criterion.class_weights,
                             ((water+background)/(2*background), (water+background)/(2*water)))
            self.assertEqual(trainer.validation_criterion.class_weights, (1., 1.))
            score = torch.tensor([[[[2., -1., 9.]]]])
            label = torch.tensor([[[1, 0, -1]]])
            expected = F.binary_cross_entropy_with_logits(score[0,0,0,:2], label[0,0,:2].float())
            torch.testing.assert_close(trainer.validation_criterion(score, label), expected)
            trainer.dataset.close()

    def test_sqrt_weight_ratio_and_mean_scale(self):
        with TemporaryDirectory() as temp, patch.object(torch.cuda,'is_available',return_value=False):
            root=Path(temp);scene_tests.SceneDatasetTests().prepare(root)
            trainer=train.WaterSegmentationTrainer(scene_tests.TinySceneNet,{},dict(model_name='tiny'),
                root/'images',root/'labels',root/'runs',bands=(1,),loss_name='bce',sampling='random',
                class_weight_mode='train_pixels_sqrt',scene_options=dict(cache_dir=root/'cache',
                window_size=32,halo=8,normalization_windows=2,num_workers=0,windows_per_epoch=4))
            counts=trainer.split_info['train_class_pixels'];bg,water=counts['background'],counts['water']
            w0,w1=trainer.criterion.class_weights
            self.assertAlmostEqual(w1/w0,(bg/water)**.5)
            self.assertAlmostEqual((w0*bg+w1*water)/(bg+water),1.)
            self.assertEqual(trainer.validation_criterion.class_weights,(1.,1.))
            trainer.dataset.close()

    def test_formal_entry_keeps_single_head_and_fresh_initialization(self):
        with patch.object(train, "WaterSegmentationTrainer") as factory:
            train.create_trainer()
        config = factory.call_args.kwargs
        self.assertEqual(config["class_weight_mode"], "train_pixels_sqrt")
        self.assertEqual(config["sampling"], "random")
        self.assertEqual(config["head_loss_weights"], (1.,))
        self.assertTrue(config["model_kwargs"]["pretrained"])
        self.assertEqual(config["model_kwargs"]["in_channels"], 1)
        self.assertIn("sqrt_weight_bce", config["output_dir"].name)
        self.assertTrue(config["require_preprocessed"])
        self.assertTrue(config["output_dir_is_final"])
        self.assertEqual(config["output_dir"], train.OUTPUT_DIR)


if __name__ == "__main__":
    unittest.main()
