"""核对 DFC 地形检查点的质量评价和导出掩膜使用相同有效域。"""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import rasterio
import torch
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from datasets.dfc_terrain import DFC_TERRAIN_CHANNELS, dfc_terrain_protocol
from datasets.relative_normalization import LOG_REPRESENTATION, relative_normalization_protocol, training_files_signature
from supervised.evaluation import postprocess


class TerrainEnsembleTests(unittest.TestCase):
    def test_ground_truth_and_both_exported_masks_drop_unsupported_neighbours(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images, labels = root / "images", root / "labels"
            images.mkdir()
            labels.mkdir()
            raw = np.full((4, 22, 22), 10, np.float32)
            raw[3] = np.arange(22, dtype=np.float32)[None, :]
            shape = dict(driver="GTiff", height=22, width=22, crs="EPSG:32650",
                         transform=from_origin(500000, 4000000, 10, 10))
            with rasterio.open(images / "image.tif", "w", count=4, dtype="float32", nodata=-9999, **shape) as ds:
                raw[3, 9, 7] = -9999
                raw[3, 9, 11] = -9999
                ds.write(raw)
                ds.scales = (1, 1, 1, 10)
            with rasterio.open(labels / "image.tif", "w", count=1, dtype="uint8", **shape) as ds:
                ds.write(np.ones((22, 22), dtype="uint8"), 1)
            fitted = relative_normalization_protocol(LOG_REPRESENTATION)
            fitted.update(p02=0., p98=30., fit_pixels=144, **training_files_signature(["train.tif"], (1, 4)))
            # 两个空洞之间的像元缺少连续 30 m 横向邻域，原始栅格中这些中心像元有值。
            valid = np.ones((22, 22), dtype=bool)
            valid[9, 7:12] = False
            checkpoint = dict(
                model_name="pvt_deeplabv3plus", model_kwargs={"in_channels": 3}, epoch=1, bands=(1, 4),
                input_channels=DFC_TERRAIN_CHANNELS, dfc_terrain=dfc_terrain_protocol("slope"),
                patch_representation=LOG_REPRESENTATION, relative_normalization=fitted,
                mean=torch.tensor([0, 0, 44.]), std=torch.ones(3), image_dir=str(images), label_dir=str(labels),
                val_files=["image.tif"])
            model = torch.nn.Conv2d(3, 1, 1, bias=False)
            with torch.no_grad():
                model.weight.zero_()
                model.weight[0, 2] = 1
            with patch.object(postprocess.torch, "load", return_value=checkpoint), \
                    patch.object(postprocess, "load_model", return_value=model), \
                    patch.object(torch.cuda, "is_available", return_value=False):
                result = postprocess.run_postprocess(
                    [root / "weights.pth"], output_dir=root / "out", use_landcover=False,
                    use_amp=False, save_visuals=False, evaluate_boundaries=False)
            for metrics in result.values():
                self.assertEqual(metrics["valid_pixels"], int(valid.sum()))
                self.assertEqual(metrics["tp"], int(valid.sum()))
            for kind in ("raw", "postprocessed"):
                with rasterio.open(root / "out" / kind / "image.tif") as result_file:
                    np.testing.assert_array_equal(result_file.read(1), np.where(valid, 1, 255))
