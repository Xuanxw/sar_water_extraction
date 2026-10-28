"""CPU 合成检查：输入扰动、错误转移、rank 清理、固定样本及流式诊断输出。"""
import contextlib
import csv
import io
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.transform import from_origin
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluation import diagnose_decoder as diagnosis
from models.pvt import UncertaintyRank


class TinyRankModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(2)
        for name in ("ufm_layer4", "ufm_layer3", "ufm_layer2"):
            layer = nn.Module()
            layer.rank = UncertaintyRank(include_midpoint=True)
            setattr(self, name, layer)

    def forward(self, image):
        logits = self.bn(image)[:, :1]
        results = [logits]
        for name in ("ufm_layer4", "ufm_layer3", "ufm_layer2"):
            foreground, background = getattr(self, name).rank(logits)
            logits = logits + .05 * (foreground - background)
            results.append(logits)
        return tuple(results)


class DecoderDiagnosisTests(unittest.TestCase):
    def test_gain_and_resolution_preserve_mask_and_input(self):
        image = torch.tensor([[[[2., 4.], [0., 6.]], [[4., 8.], [0., 12.]]]])
        original = image.clone()
        valid = torch.tensor([[[True, True], [False, True]]])
        out = diagnosis.transform_input(image, valid, "storage_gain_0.5")
        torch.testing.assert_close(out, image * .5)
        out = diagnosis.transform_input(image, valid, "vv_gain_1.5")
        torch.testing.assert_close(out[:, 0], image[:, 0] * 1.5)
        torch.testing.assert_close(out[:, 1], image[:, 1])
        out = diagnosis.transform_input(image, valid, "resolution_x2")
        torch.testing.assert_close(out[0, 0], torch.tensor([[4., 4.], [0., 4.]]))
        torch.testing.assert_close(out[0, 1], torch.tensor([[8., 8.], [0., 8.]]))
        torch.testing.assert_close(image, original)
        odd = torch.ones(2, 2, 5, 7) * 3
        torch.testing.assert_close(diagnosis.transform_input(odd, torch.ones(2, 5, 7, dtype=torch.bool),
                                                            "resolution_x2"), odd)

    def test_transition_counts_conserve_pixels_and_errors(self):
        coarse = torch.logit(torch.tensor([[[.05, .2, .5, .8, .95, .95, .05, .5]]]))
        final = torch.logit(torch.tensor([[[.8, .8, .05, .05, .95, .1, .8, .8]]]))
        label = torch.tensor([[[1, 1, 1, 0, 0, 1, 0, -1]]])
        counts = diagnosis.transition_counts(coarse, final, label)
        total = dict(zip(diagnosis.TRANSITION_FIELDS, counts.sum(0).tolist()))
        self.assertEqual(total, dict(pixels=7, water_pixels=4, background_pixels=3,
            coarse_fn=2, coarse_fp=2, final_fn=2, final_fp=2, tp_recovered=2,
            tp_lost=2, fp_removed=1, fp_added=1))
        self.assertEqual(total["final_fn"], total["coarse_fn"] - total["tp_recovered"] + total["tp_lost"])
        self.assertEqual(total["final_fp"], total["coarse_fp"] - total["fp_removed"] + total["fp_added"])
        self.assertEqual(counts[:, 0].tolist(), [2, 1, 1, 1, 2])

    def test_rank_counterfactual_preserves_means_routes_and_histograms(self):
        model = TinyRankModel().eval()
        logits = torch.logit(torch.tensor([[[[.01, .2, .45, .5], [.6, .75, .9, .99]]]]))
        original = model.ufm_layer4.rank(logits)
        for mode in diagnosis.RANK_CONDITIONS:
            statistics = {}
            with diagnosis.rank_counterfactual(model, mode, statistics):
                altered = model.ufm_layer4.rank(logits)
            for before, after in zip(original, altered):
                torch.testing.assert_close(before.mean(), after.mean())
                if mode == "rank_flat":
                    self.assertEqual(after.unique().numel(), 1)
                elif mode == "rank_spatial_roll":
                    torch.testing.assert_close(before.flatten().sort().values, after.flatten().sort().values)
                else:
                    self.assertTrue(torch.equal(before > 0, after > 0))
                    self.assertEqual(after[after > 0].unique().numel(), 1)
            self.assertEqual(len(statistics), 2)
            self.assertEqual(len(model.ufm_layer4.rank._forward_hooks), 0)
        # 一侧为空时，route_only 必须保持空分支，而不是引入 NaN 或正权重。
        with diagnosis.rank_counterfactual(model, "rank_route_only"):
            fore, back = model.ufm_layer4.rank(torch.ones(1, 1, 4, 4) * 9)
        self.assertTrue(torch.isfinite(fore).all())
        self.assertEqual(back.count_nonzero().item(), 0)

    def test_hooks_removed_on_exception_and_model_state_unchanged(self):
        model = TinyRankModel().eval()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            with diagnosis.rank_counterfactual(model, "rank_flat"):
                with torch.inference_mode():
                    model(torch.randn(2, 2, 8, 8))
                raise RuntimeError("synthetic")
        self.assertTrue(all(len(getattr(model, n).rank._forward_hooks) == 0
                            for n in ("ufm_layer4", "ufm_layer3", "ufm_layer2")))
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_fixed_samples_and_synthetic_streaming_report(self):
        torch.set_num_threads(2)
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            images, labels = root / "images", root / "labels"
            for directory in (images, labels):
                directory.mkdir()
            for index, name in enumerate(("train.tif", "val_a.tif", "val_b.tif")):
                data = np.ones((2, 8, 8), dtype=np.float32) * (index + 1)
                data[:, :4] *= .2
                with rasterio.open(images / name, "w", driver="GTiff", height=8, width=8,
                                   count=2, dtype="float32", transform=from_origin(0, 8, 1, 1)) as dst:
                    dst.write(data)
                with rasterio.open(labels / name, "w", driver="GTiff", height=8, width=8,
                                   count=1, dtype="uint8", transform=from_origin(0, 8, 1, 1)) as dst:
                    dst.write((data[0] > 1).astype("uint8"), 1)
            manifest = root / "split_manifest.csv"
            manifest.write_text("name,scene,split,active\ntrain.tif,城市,train,1\n"
                                "val_a.tif,农田,val,1\nval_b.tif,山地,val,1\n", encoding="utf-8")
            checkpoint = dict(epoch=1, model_kwargs={"num_classes": 1}, image_dir=str(images),
                              label_dir=str(labels), bands=(1, 2), val_files=["val_b.tif", "val_a.tif"],
                              mean=torch.tensor([1., 1.]), std=torch.ones(2), amp_enabled=False,
                              split_manifest=str(manifest), split_manifest_sha256=diagnosis.file_sha256(manifest))
            dataset = diagnosis.prepare_dataset(checkpoint)
            self.assertEqual([dataset[i]["name"] for i in range(len(dataset))], checkpoint["val_files"])
            path = root / "model.pth"
            torch.save(checkpoint, path)
            with patch.object(diagnosis, "load_model", side_effect=lambda _: TinyRankModel()), \
                 patch.object(torch.cuda, "is_available", return_value=False), contextlib.redirect_stdout(io.StringIO()):
                directory = diagnosis.run_diagnosis({"pvt_uafm": path, "pvt_nested": path}, root / "report")
            with (directory / "head_metrics.csv").open(encoding="utf-8-sig") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), (len(diagnosis.CONDITIONS) * 2 + len(diagnosis.RANK_CONDITIONS)) * 4)
            self.assertTrue(all(int(row["valid_pixels"]) == 128 for row in rows))
            with (directory / "per_image_metrics.csv").open(encoding="utf-8-sig") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual({r["name"] for r in rows}, {"val_a.tif", "val_b.tif"})
            self.assertEqual({r["scene"] for r in rows}, {"农田", "山地"})
            with (directory / "coarse_to_final.csv").open(encoding="utf-8-sig") as stream:
                rows = list(csv.DictReader(stream))
            self.assertTrue(all(int(r["pixels"]) == 128 for r in rows if r["confidence"] == "all"))
            self.assertTrue((directory / "rank_statistics.csv").is_file())
            self.assertTrue((directory / "run_info.txt").is_file())


if __name__ == "__main__":
    unittest.main()
