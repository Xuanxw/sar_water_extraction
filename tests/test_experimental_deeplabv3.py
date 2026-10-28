"""PVT-DeepLabv3+ 的梯度、重载、训练协议和 quality 闭环；仅用 CPU 合成数据。"""
import csv
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.transform import from_origin
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import experimental_model_deeplabv3 as experiment
import train
from models.checkpoint import load_model
from models.losses import WaterSegmentationLoss


class DeepLabExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_shapes_gradients_and_offline_reload(self):
        options = dict(in_channels=2, num_classes=1, pretrained=False,
                       decoder_channels=32, low_level_channels=16, atrous_rates=(2, 3, 4), dropout=.1)
        model = experiment.PVTDeepLabV3Plus(**options).train()
        image = torch.randn(2, 2, 64, 64)
        label = torch.randint(0, 2, (2, 64, 64))
        label[:, :4] = -1
        logits = model(image)
        self.assertEqual(tuple(logits.shape), (2, 1, 64, 64))
        WaterSegmentationLoss("bce")(logits, label).backward()
        for module in (model.backbone, model.aspp, model.low_projection,
                       model.fusion, model.segmentation_head):
            gradients = [p.grad for p in module.parameters() if p.requires_grad]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0)
        # 1×1池化支路无需两个样本估计BN；单样本训练应能前向/反向。
        model.zero_grad(set_to_none=True)
        model(image[:1]).mean().backward()
        checkpoint = dict(model_name="pvt_deeplabv3plus", model_kwargs={**options, "pretrained": True},
                          model_state_dict=model.state_dict())
        with patch.object(experiment, "build_pvt_backbone", wraps=experiment.build_pvt_backbone) as factory:
            restored = load_model(checkpoint).eval()
            self.assertFalse(factory.call_args.args[1])
        self.assertEqual([m.conv.dilation for m in restored.aspp.branches[1:]], [(2,2),(3,3),(4,4)])
        model.eval()
        with torch.no_grad():
            torch.testing.assert_close(restored(image), model(image), rtol=0, atol=0)
            self.assertEqual(tuple(restored(torch.randn(1, 2, 65, 97)).shape), (1, 1, 65, 97))
            single = experiment.PVTDeepLabV3Plus(in_channels=1, pretrained=False,
                                               decoder_channels=32, low_level_channels=16).eval()
            self.assertEqual(tuple(single(torch.randn(1, 1, 64, 64)).shape), (1, 1, 64, 64))

    def test_training_protocol_matches_control(self):
        with patch.object(train, "WaterSegmentationTrainer") as base:
            train.create_trainer()
            reference = base.call_args.kwargs
        with patch.object(experiment, "WaterSegmentationTrainer") as candidate:
            experiment.create_trainer()
            options = candidate.call_args.kwargs
        for key in ("image_dir", "label_dir", "bands", "batch_size", "epochs", "learning_rate",
                    "seed", "val_ratio", "max_samples", "loss_name", "optimizer_name", "weight_decay",
                    "cosine_schedule", "train_augmentation", "sampling", "hard_warmup_epochs",
                    "hard_sample_ratio", "hard_pool_ratio"):
            self.assertEqual(reference[key], options[key], key)
        self.assertEqual(options["head_loss_weights"], (1.,))
        self.assertEqual(options["model_info"]["output_stride"], 32)
        self.assertEqual(options["model_kwargs"]["atrous_rates"], (3,6,9))
        self.assertEqual(options["model_kwargs"]["decoder_channels"], 256)
        self.assertEqual(options["model_kwargs"]["low_level_channels"], 48)
        self.assertNotEqual(options["output_dir"], reference["output_dir"])

    def test_training_saved_split_and_quality(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            records = [("a", "train", 1), ("b", "train", 1), ("copy", "train", 0),
                       ("c", "val", 1), ("d", "val", 1)]
            rows = []
            for index, (name, split, active) in enumerate(records):
                images, labels = root / "data/images" / split, root / "data/labels" / split
                images.mkdir(parents=True, exist_ok=True)
                labels.mkdir(parents=True, exist_ok=True)
                rng = np.random.default_rng(index)
                image = rng.uniform(1, 10, (2, 64, 64)).astype("float32")
                label = (image[0] < 4).astype("uint8")
                label[:2] = 255  # 标签无效像素必须在损失和评价中排除。
                transform = from_origin(10, 20, 1, 1)
                with rasterio.open(images / f"{name}.tif", "w", driver="GTiff", width=64,
                                   height=64, count=2, dtype="float32", transform=transform) as dst:
                    dst.write(image)
                with rasterio.open(labels / f"{name}.tif", "w", driver="GTiff", width=64,
                                   height=64, count=1, dtype="uint8", nodata=255, transform=transform) as dst:
                    dst.write(label, 1)
                rows.append(dict(name=f"{name}.tif", split=split, active=active, group_id=name))
            manifest = root / "data/split_manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            with patch.object(experiment, "IMAGE_DIR", root / "data/images"), \
                 patch.object(experiment, "LABEL_DIR", root / "data/labels"), \
                 patch.object(experiment, "OUTPUT_DIR", root / "runs"), \
                 patch.object(experiment, "PRETRAINED", False), \
                 patch.object(experiment, "DECODER_CHANNELS", 32), \
                 patch.object(experiment, "LOW_LEVEL_CHANNELS", 16), \
                 patch.object(experiment, "EPOCHS", 3), \
                 patch.object(experiment, "RUN_POSTPROCESS", True), \
                 patch.object(experiment, "POSTPROCESS_OPTIONS", dict(threshold=.5, use_landcover=False,
                              save_masks=True, save_visuals=False)), \
                 patch.object(torch.cuda, "is_available", return_value=False):
                run = experiment.main()
            self.assertEqual(run.parent.name, "runs_grouped")
            checkpoint = torch.load(run / "best_f1.pth", weights_only=True)
            self.assertEqual(checkpoint["train_files"], ["a.tif", "b.tif"])
            self.assertEqual(checkpoint["val_files"], ["c.tif", "d.tif"])
            self.assertEqual(checkpoint["head_loss_weights"], (1.,))
            self.assertEqual(checkpoint["model_name"], "pvt_deeplabv3plus")
            self.assertEqual(checkpoint["model_kwargs"]["atrous_rates"], (3,6,9))
            self.assertEqual(checkpoint["optimizer_name"], "adamw")
            self.assertEqual(checkpoint["sampling_config"]["mode"], "hard")
            self.assertEqual((run / "split_manifest.csv").read_bytes(), manifest.read_bytes())
            with (run / "metrics.csv").open(encoding="utf-8-sig") as stream:
                metrics = list(csv.DictReader(stream))
            self.assertEqual(len(metrics), 6)
            self.assertTrue(all(float(r["ignored_pixels"]) > 0 for r in metrics))
            quality = next((run / "quality").iterdir())
            with (quality / "metrics.csv").open(encoding="utf-8-sig") as stream:
                raw = next(r for r in csv.DictReader(stream) if r["result"] == "ensemble")
            val = checkpoint["phase_metrics"]["val"]
            for key in ("tn", "fp", "fn", "tp"):
                self.assertEqual(int(raw[key]), val[key])
            self.assertTrue((quality / "boundary/metrics.csv").is_file())
            with (quality / "per_image.csv").open(encoding="utf-8-sig") as stream:
                self.assertEqual({r["name"] for r in csv.DictReader(stream)}, {"c.tif", "d.tif"})
            self.assertFalse((run / "features").exists())
            # 同一个检查点经过独立评价入口，混淆矩阵也应与训练val完全一致。
            from evaluation.model_eval import evaluate
            with patch.object(torch.cuda, "is_available", return_value=False):
                evaluated = evaluate(run / "best_f1.pth", use_saved_val=True, batch_size=2, use_amp=False)
            for key in ("tn", "fp", "fn", "tp"):
                self.assertEqual(evaluated[key], val[key])


if __name__ == "__main__":
    unittest.main()
