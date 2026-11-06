"""检查无滤波窗口读取及删除旧Lee后的配置行为。"""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import rasterio
from rasterio.windows import Window
from datasets.speckle import SpeckleFilter, read_filtered_window
import test_sar_scene_dataset as fixtures

class SpeckleTests(unittest.TestCase):
    def test_old_lee_is_rejected(self):
        with self.assertRaises(ValueError):
            SpeckleFilter(dict(method='lee', input_domain='amplitude'))

    def test_plain_window_preserves_zero_and_mask(self):
        with TemporaryDirectory() as temp:
            root=Path(temp)
            fixtures.SceneDatasetTests().prepare(root)
            with rasterio.open(root/'images/train/train.tif') as src:
                window=Window(0,0,16,16)
                original=src.read((1,),window=window,masked=True)
                actual,valid=read_filtered_window(src,(1,),window,SpeckleFilter())
                expected=~np.ma.getmaskarray(original).any(axis=0)&np.isfinite(original.data).all(axis=0)
                np.testing.assert_array_equal(valid,expected)
                np.testing.assert_array_equal(actual,np.where(expected[None],original.data,0))
