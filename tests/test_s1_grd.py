"""密云 S1 整景推理回归：通道次序、覆盖掩膜、米制 DEM、重采样与输出网格。"""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import rasterio
from rasterio.transform import from_origin
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from supervised.evaluation.s1_grd import predict_grd_scene


class RankAndHeight(torch.nn.Module):
    def forward(self, image):
        # 第一个通道已按训练均值 0.5 居中；第二个通道按训练均值 20 m 居中。
        return torch.minimum(image[:, :1], image[:, 1:2])


def write_raster(path, data, transform, descriptions, nodata=None):
    with rasterio.open(path, "w", driver="GTiff", count=len(data), dtype="float32",
                       height=data.shape[1], width=data.shape[2], crs="EPSG:32650",
                       transform=transform, nodata=nodata) as target:
        target.write(data.astype("float32"))
        target.descriptions = descriptions


class S1GrdTests(unittest.TestCase):
    def test_reversed_grd_bands_and_common_valid_mask(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows, cols = np.indices((39, 45))
            transform = from_origin(500000, 5000000, 10, 10)
            vv = np.where(cols % 2, -5., -15.).astype("float32")
            vh = np.where(cols % 2, -25., -10.).astype("float32")
            vv[12, 10] = 0.0  # 单 VV 为 0 dB 仍是亮像元。
            vh[13, 11] = 0.0  # 单 VH 为 0 dB 同样有效。
            vv[15:19, 11:16] = vh[15:19, 11:16] = 0.0
            vv[14, 12] = np.nan
            elevation = np.where(rows % 2, 30., 10.).astype("float32")
            elevation[12, 10] = 30.
            elevation[19:23, 13:19] = -9999
            src, dem, result = root / "grd.tif", root / "dem.tif", root / "water.tif"
            write_raster(src, np.stack((vh, vv)), transform, ("Sigma0_VH_db", "Sigma0_VV_db"))
            write_raster(dem, elevation[None], transform, ("elevation_m",), -9999)
            predict_grd_scene(RankAndHeight(), src, result, mean=[0.5, 20.], std=[0.25, 10.],
                              patch_representation="vv_rank_dem_m", bands=(1, 4), dem_path=dem,
                              window_size=32, halo=4, batch_size=2)
            expected = ((vv > -10) & (elevation > 20)).astype("uint8")
            expected[((vv == 0) & (vh == 0)) | ~np.isfinite(vv) | (elevation == -9999)] = 255
            with rasterio.open(result) as water:
                np.testing.assert_array_equal(water.read(1), expected)
                self.assertEqual(water.nodata, 255)
                self.assertEqual(water.crs, rasterio.crs.CRS.from_epsg(32650))
                self.assertEqual(water.transform, transform)

    def test_bilinear_dem_matches_pixel_centers(self):
        # 20 m DEM 采样一条线性高程斜坡；阈值的位置可区分双线性和最近邻插值。
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = torch.nn.Conv2d(2, 1, 1, bias=False)
            with torch.no_grad():
                model.weight.copy_(torch.tensor([[[[0.]], [[1.]]]]))
            src, dem, result = root / "vv.tif", root / "dem.tif", root / "mask.tif"
            write_raster(src, np.zeros((1, 32, 32)), from_origin(0, 320, 10, 10),
                         ("Sigma0_VV_db",), -9999)
            meters = np.tile(np.arange(10, 320, 20), (16, 1))
            write_raster(dem, meters[None], from_origin(0, 320, 20, 20), ("elevation_m",), -9999)
            predict_grd_scene(model, src, result, mean=[0.5, 128.], std=[1., 5.],
                              patch_representation="vv_rank_dem_m", bands=(1, 4), dem_path=dem,
                              window_size=32, halo=4)
            with rasterio.open(result) as water:
                expected = np.tile(np.arange(5, 320, 10) > 128, (32, 1)).astype("uint8")
                np.testing.assert_array_equal(water.read(1), expected)
