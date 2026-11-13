"""嵌套细化对照：共有权重、节点尺寸、梯度、离线重载及 grouped 训练/quality 闭环。"""
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

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"),
                str(Path(__file__).resolve().parents[1] / "scripts")]
from supervised import train
from supervised.models.pvt import PVTUANet
from supervised.models.checkpoint import load_model
from supervised.models.losses import WaterSegmentationLoss

# 测试显式加载样例配置，正式脚本仅使用命令行提供的文件。
train.configure(Path(__file__).resolve().parents[1] / 'configs/train.yaml')


class NestedDecoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_shared_initialization_shapes_and_offline_reload(self):
        options = dict(pretrained=False, num_classes=1, stage="uafm", final_residual=True)
        torch.manual_seed(42)
        control = PVTUANet(**options).eval()
        torch.manual_seed(42)
        nested = PVTUANet(**options, refinement_decoder="unetpp").eval()
        for key, value in control.state_dict().items():
            if not key.startswith("ufm_layer"):
                self.assertTrue(torch.equal(value, nested.state_dict()[key]), key)
        self.assertFalse(any(key.startswith("ufm_layer") for key in nested.state_dict()))
        image = torch.randn(1, 2, 512, 512)
        nodes = {}
        handles = [getattr(nested.nested_decoder, name).register_forward_hook(
            lambda module, args, out, name=name: nodes.update({name: (args[0].shape, out.shape)})
        ) for name in ("x01", "x11", "x21", "x02", "x12", "x03")]
        with torch.no_grad():
            old, new = control(image), nested(image)
        for handle in handles:
            handle.remove()
        for logits in new:
            self.assertEqual(tuple(logits.shape), (1, 1, 512, 512))
        torch.testing.assert_close(old[0], new[0], rtol=0, atol=0)
        torch.testing.assert_close(new[0], new[-1], rtol=0, atol=0)
        for name, channels, size in (("x01",64,128),("x11",64,64),("x21",64,32),
                                     ("x02",96,128),("x12",96,64),("x03",128,128)):
            self.assertEqual(tuple(nodes[name][0]), (1, channels, size, size))
            self.assertEqual(tuple(nodes[name][1]), (1, 32, size, size))
        checkpoint = dict(model_name="pvt_unetpp_refinement", model_state_dict=nested.state_dict(),
                          model_kwargs={**options, "pretrained": True, "refinement_decoder": "unetpp"})
        # pretrained=True 的训练元数据重载时必须被覆盖，确保不下载骨干权重。
        import supervised.models.pvt as pvt
        with patch.object(pvt, "build_pvt_backbone", wraps=pvt.build_pvt_backbone) as factory:
            restored = load_model(checkpoint).eval()
            self.assertFalse(factory.call_args.args[1])
        with torch.no_grad():
            odd = torch.randn(1, 2, 65, 97)
            for a, b in zip(nested(odd), restored(odd)):
                self.assertEqual(tuple(a.shape), (1, 1, 65, 97))
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            single = PVTUANet(**options, in_channels=1, refinement_decoder="unetpp").eval()
            self.assertEqual(single(torch.randn(1,1,64,64))[-1].shape, (1,1,64,64))

    def test_gradients_after_zero_initialized_head_updates(self):
        model = PVTUANet(pretrained=False, num_classes=1, stage="uafm", final_residual=True,
                         refinement_decoder="unetpp").eval()
        image = torch.randn(2, 2, 64, 64)
        label = (image[:, 0] > 0).long()
        label[:, :3] = -1
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        criterion = WaterSegmentationLoss("bce")
        # 第一轮零初始化头阻断其前方梯度，更新后六个节点都应收到有限、非零梯度。
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            outputs = model(image)
            sum(criterion(out, label) for out in outputs).backward()
            if step == 0:
                optimizer.step()
        for module in (model.backbone, model.neck, model.decoder, model.nested_decoder):
            grads = [p.grad for p in module.parameters() if p.requires_grad]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(g.abs().sum().item() for g in grads), 0)
        for name in ("x01", "x11", "x21", "x02", "x12", "x03"):
            self.assertGreater(sum(p.grad.abs().sum().item() for p in
                               getattr(model.nested_decoder, name).parameters()), 0)
        self.assertFalse(torch.equal(outputs[0], outputs[-1]))

    def test_factory_changes_only_refinement_configuration(self):
        choices = []
        for decoder in ("uafm", "unetpp"):
            with patch.multiple(train, OUTPUT_DIR=None, SAMPLING="hard",
                                POLARIZATIONS=("VV", "VH"), POLARIZATION_BANDS={"VV": 1, "VH": 2}), \
                 patch.object(train, "METHOD", "uafnet"), \
                 patch.object(train, "UAFNET_REFINEMENT_DECODER", decoder), \
                 patch.object(train, "WaterSegmentationTrainer") as factory:
                train.create_trainer()
                choices.append(factory.call_args.kwargs)
        old, new = choices
        for key in old:
            if key not in ("model_kwargs", "model_info", "output_dir"):
                self.assertEqual(old[key], new[key], key)
        options = dict(new["model_kwargs"])
        self.assertEqual(options.pop("refinement_decoder"), "unetpp")
        self.assertEqual(options, old["model_kwargs"])
        self.assertEqual(new["output_dir"].name, "pvt_unetpp_hard_sampling_residual")
        self.assertEqual(new["model_info"]["model_name"], "pvt_unetpp_refinement")

    def test_grouped_training_checkpoint_and_quality(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            rows = []
            for index, (name, split, active) in enumerate(
                (("a","train",1),("b","train",1),("copy","train",0),("c","val",1),("d","val",1))
            ):
                images, labels = root / "data/images" / split, root / "data/labels" / split
                images.mkdir(parents=True, exist_ok=True)
                labels.mkdir(parents=True, exist_ok=True)
                image = np.random.default_rng(index).uniform(1,10,(2,64,64)).astype("float32")
                label = (image[0] < 4).astype("uint8")
                label[:2] = 255
                for path, array in ((images / f"{name}.tif", image), (labels / f"{name}.tif", label[None])):
                    with rasterio.open(path, "w", driver="GTiff", width=64, height=64,
                                       count=len(array), dtype=array.dtype,
                                       nodata=255 if len(array) == 1 else None,
                                       transform=from_origin(10,20,1,1)) as dst:
                        dst.write(array)
                rows.append(dict(name=f"{name}.tif", split=split, active=active, group_id=name))
            manifest = root / "data/split_manifest.csv"
            with manifest.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            with patch.multiple(train, METHOD="uafnet", UAFNET_REFINEMENT_DECODER="unetpp", IMAGE_DIR=root / "data/images", LABEL_DIR=root / "data/labels",
                                OUTPUT_DIR=root / "runs", EPOCHS=3, PRETRAINED=False,
                                RESUME_CHECKPOINT=None, DATA_LAYOUT="patch", CLASS_WEIGHT_MODE="none",
                                POLARIZATIONS=("VV","VH"), POLARIZATION_BANDS={"VV":1,"VH":2},
                                VALIDATION_QUALITY_OPTIONS=dict(enabled=False)), \
                 patch.object(torch.cuda, "is_available", return_value=False):
                train.main()
            run = next((root / "runs").iterdir())
            checkpoint = torch.load(run / "best_f1.pth", weights_only=True)
            self.assertEqual(checkpoint["model_name"], "pvt_unetpp_refinement")
            self.assertEqual(checkpoint["model_kwargs"]["refinement_decoder"], "unetpp")
            self.assertEqual(checkpoint["train_files"], ["a.tif", "b.tif"])
            self.assertEqual(checkpoint["val_files"], ["c.tif", "d.tif"])
            self.assertEqual(checkpoint["head_loss_weights"], (1.,1.,1.,1.))
            self.assertEqual((run / "split_manifest.csv").read_bytes(), manifest.read_bytes())
            with (run / "metrics.csv").open(encoding="utf-8-sig") as stream:
                metrics = list(csv.DictReader(stream))
            self.assertEqual(len(metrics), 6)
            self.assertTrue(all(float(row["ignored_pixels"]) > 0 for row in metrics))
            # 训练入口不隐式后处理；使用已保存最佳模型显式复评。
            from supervised.evaluation.postprocess import run_postprocess
            quality = run / "postprocess" / "test" / "quality"
            run_postprocess([run / "best_f1.pth"],output_dir=quality,threshold=.5,
                            use_landcover=False,save_masks=True,save_visuals=False)
            with (quality / "metrics.csv").open(encoding="utf-8-sig") as stream:
                raw = next(row for row in csv.DictReader(stream) if row["result"] == "ensemble")
            for key in ("tn","fp","fn","tp"):
                self.assertEqual(int(raw[key]), checkpoint["phase_metrics"]["val"][key])
            self.assertTrue((quality / "boundary/metrics.csv").is_file())
            with (quality / "per_image.csv").open(encoding="utf-8-sig") as stream:
                self.assertEqual({row["name"] for row in csv.DictReader(stream)}, {"c.tif", "d.tif"})
            self.assertFalse((run / "features").exists())


if __name__ == "__main__":
    unittest.main()
