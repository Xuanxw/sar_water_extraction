"""用临时栅格验证读取、配对和控制台统计，不接触下载数据。"""
import contextlib
import io
from pathlib import Path
import tempfile
import sys
import unittest

import numpy as np
import rasterio
from rasterio.transform import from_origin

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"),
                str(Path(__file__).resolve().parents[1] / "scripts")]
from datasets.dfc_sample.check_data import check_data


class CheckDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.images = self.root / "images"
        self.labels = self.root / "labels"
        self.images.mkdir()
        self.labels.mkdir()

    def raster(self, path, array, **kwargs):
        with rasterio.open(path, "w", driver="GTiff", height=array.shape[1],
                           width=array.shape[2], count=array.shape[0], dtype=array.dtype,
                           transform=from_origin(500000, 3000000, 10, 10), **kwargs) as dst:
            dst.write(array)

    def test_actual_statistics_and_no_output_files(self):
        data = np.array([[[0, 2], [np.nan, -9999]]], dtype="float32")
        self.raster(self.images / "a.tif", data, nodata=-9999)
        self.raster(self.labels / "a.tif", np.array([[[0, 1], [1, 255]]], dtype="uint8"))
        before = set(self.root.rglob("*"))
        terminal = io.StringIO()
        with contextlib.redirect_stdout(terminal):
            check_data(self.images, self.labels)
        output = terminal.getvalue()
        self.assertIn("min=0.0，max=2.0，掩膜无效=1，NaN/Inf=1", output)
        self.assertIn("{0: 1, 1: 2, 255: 1}", output)
        self.assertIn("完整读取成功：1/1", output)
        self.assertEqual(set(self.root.rglob("*")), before)

    def test_corrupt_file_does_not_stop_later_files(self):
        (self.images / "a.tif").write_bytes(b"incomplete raster")
        self.raster(self.images / "b.tif", np.ones((3, 2, 2), dtype="uint8"))
        terminal = io.StringIO()
        with contextlib.redirect_stdout(terminal):
            check_data(self.images)
        self.assertIn("读取失败：", terminal.getvalue())
        self.assertIn("完整读取成功：1/2", terminal.getvalue())
        self.assertIn("波段数=3", terminal.getvalue())
        self.assertIn("读取失败文件数：1", terminal.getvalue())

    def test_missing_orphan_and_mismatched_labels(self):
        self.raster(self.images / "a.tif", np.ones((1, 2, 2), dtype="uint8"))
        self.raster(self.images / "missing.tif", np.ones((1, 2, 2), dtype="uint8"))
        self.raster(self.labels / "a.tif", np.zeros((1, 3, 2), dtype="uint8"))
        self.raster(self.labels / "orphan.tif", np.zeros((1, 2, 2), dtype="uint8"))
        terminal = io.StringIO()
        with contextlib.redirect_stdout(terminal):
            check_data(self.images, self.labels)
        self.assertIn("缺少标签：missing", terminal.getvalue())
        self.assertIn("没有对应影像：orphan", terminal.getvalue())
        self.assertIn("尺寸不一致：1 对", terminal.getvalue())


if __name__ == "__main__":
    unittest.main()
