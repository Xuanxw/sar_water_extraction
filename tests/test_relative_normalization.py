"""固定训练分位数：保留块间强度、无验证/标签泄漏、掩膜和续跑协议一致。"""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from datasets.dfc_sample.s1_dataset import WaterDataset, relative_backscatter_ranks
from datasets.relative_normalization import (
    INPUT_DOMAIN, LOG_REPRESENTATION, REPRESENTATION, apply_vv_train_p02_p98_dem_m,
    relative_log10_power, training_files_signature, validate_relative_normalization,
)


def write_pair(path, vv, dem=None, mask=None, dem_scale=1.0, dem_offset=0.0):
    vv = np.asarray(vv, dtype=np.float32)
    dem = np.full(vv.shape, 20.0, dtype=np.float32) if dem is None else np.asarray(dem, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", count=2, height=vv.shape[0], width=vv.shape[1],
                       dtype="float32", crs="EPSG:32650", transform=from_origin(500000, 5000000, 10, 10)) as dst:
        dst.write(np.stack([vv, dem]))
        dst.scales = (1.0, dem_scale)
        dst.offsets = (0.0, dem_offset)
        if mask is not None:
            dst.write_mask(np.asarray(mask, dtype=np.uint8) * 255)


def parameters(names=("a.tif", "b.tif"), bands=(1, 2)):
    return dict(version=1, method="train_p02_p98", representation=REPRESENTATION,
                input_domain=INPUT_DOMAIN, fit_scope="all_valid_training_pixels",
                percentiles=[2, 98], range=[0.0, 1.0], p02=0.0, p98=20.0,
                fit_pixels=12, **training_files_signature(names, bands))


class RelativeNormalizationTests(unittest.TestCase):
    def test_log_protocol_quantiles_transform_before_interpolation_and_restore(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_pair(root / "train.tif", [[0, 1, 10, 100, -5]], dem=[[0, 1, 2, 3, 4]],
                       mask=[[1, 1, 1, 1, 0]], dem_scale=3., dem_offset=-2.)
            write_pair(root / "val.tif", [[1e8]])
            data = WaterDataset(root, bands=(1, 2), patch_representation=LOG_REPRESENTATION)
            fitted = data.fit_relative_normalization([0])
            logs = np.array([-60., 0., 10., 20.])
            limits = np.percentile(logs, [2, 98])
            np.testing.assert_allclose([fitted["p02"], fitted["p98"]], limits, atol=1e-12, rtol=0)
            self.assertEqual(fitted["fit_pixels"], 4)
            self.assertEqual(fitted["input_domain"], INPUT_DOMAIN)
            self.assertEqual(fitted["floor"], 1e-6)
            self.assertEqual(fitted["reference"], 1.)
            expected = np.clip((logs - limits[0]) / (limits[1] - limits[0]), 0, 1)
            sample = data[0]
            np.testing.assert_allclose(sample["image"][0, 0, :4], expected)
            np.testing.assert_equal(sample["image"][1].numpy(), [[-2, 1, 4, 7, 0]])
            np.testing.assert_equal(sample["valid_mask"].numpy(), [[True, True, True, True, False]])
            restored = WaterDataset(root, bands=(1, 2), patch_representation=LOG_REPRESENTATION,
                                    relative_normalization=json.loads(json.dumps(fitted)))
            np.testing.assert_array_equal(restored[0]["image"], sample["image"])
            self.assertEqual(float(restored[1]["image"][0, 0, 0]), 1.)
            with self.assertRaisesRegex(ValueError, "划分"):
                restored.set_relative_normalization(fitted, [1])
            with self.assertRaisesRegex(ValueError, "表示"):
                WaterDataset(root, bands=(1, 2), patch_representation=REPRESENTATION, relative_normalization=fitted)
            with self.assertRaisesRegex(ValueError, "表示"):
                restored.set_relative_normalization(parameters())
            for updates in ({"floor": 1.}, {"reference": 10.}, {"p02": -70.}, {"input_domain": "sigma0_db"}):
                with self.subTest(updates=updates), self.assertRaises(ValueError):
                    validate_relative_normalization(dict(fitted, **updates), LOG_REPRESENTATION)

    def test_log_rejects_invalid_inputs_and_preserves_zero(self):
        np.testing.assert_allclose(relative_log10_power([0., 1e-7, 1., 10.]), [-60., -60., 0., 10.])
        for bad in (-1., np.nan, np.inf):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "有限非负"):
                relative_log10_power([bad])
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_pair(root / "negative.tif", [[-1, 1, 2, 3]])
            data = WaterDataset(root, bands=(1, 2), patch_representation=LOG_REPRESENTATION)
            with self.assertRaisesRegex(ValueError, "非负"):
                data.fit_relative_normalization([0])

    def test_shared_brightness_window_invariance_and_mask(self):
        water = np.linspace(1, 2, 16, dtype=np.float32).reshape(4, 4)
        land = water * 10
        dem = np.arange(16, dtype=np.float32).reshape(4, 4)
        valid = np.ones((4, 4), bool)
        # 原逐块秩不能区分这两块；固定参数必须保留其亮暗关系。
        np.testing.assert_array_equal(relative_backscatter_ranks(water[None], valid),
                                      relative_backscatter_ranks(land[None], valid))
        image = np.stack([water, dem])
        original = image.copy()
        low = apply_vv_train_p02_p98_dem_m(image, valid, parameters())
        high = apply_vv_train_p02_p98_dem_m(np.stack([land, dem]), valid, parameters())
        self.assertLess(low[0].max(), high[0].min())
        np.testing.assert_array_equal(low[1], dem)
        crop = apply_vv_train_p02_p98_dem_m(image[:, 1:3, 1:3], valid[1:3, 1:3], parameters())
        np.testing.assert_array_equal(crop, low[:, 1:3, 1:3])
        np.testing.assert_array_equal(image, original)
        image[:, 0, 0] = np.nan
        valid[0, 0] = False
        masked = apply_vv_train_p02_p98_dem_m(image, valid, parameters())
        np.testing.assert_array_equal(masked[:, 0, 0], 0)
        self.assertEqual(masked.dtype, np.float32)

    def test_fit_reads_only_selected_training_images_and_all_valid_pixels(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_pair(root / "images/a.tif", [[0, 2, 4], [6, 8, 10]],
                       dem=[[0, 1, np.nan], [1, 1, 1]], mask=[[1, 1, 1], [1, 0, 1]])
            write_pair(root / "images/b.tif", [[20, 30]])
            write_pair(root / "images/val.tif", [[-1e9, 1e9]])
            (root / "labels").mkdir()
            (root / "labels/a.tif").write_text("label must never be opened while fitting", encoding="utf-8")
            dataset = WaterDataset(root / "images", root / "labels", bands=(1, 2), patch_representation=REPRESENTATION)
            with patch.object(dataset, "read_image", wraps=dataset.read_image) as reader:
                fitted = dataset.fit_relative_normalization([0, 1])
            self.assertEqual([call.args[0] for call in reader.call_args_list], [0, 1])
            expected = np.array([0, 2, 6, 10, 20, 30], dtype=np.float32)
            np.testing.assert_allclose([fitted["p02"], fitted["p98"]], np.percentile(expected, [2, 98]), rtol=0, atol=1e-12)
            self.assertEqual(fitted["fit_pixels"], 6)
            self.assertEqual(fitted["training_samples"], 2)
            self.assertEqual(dataset.relative_normalization, fitted)

    def test_common_positive_gain_cancels_when_statistics_are_fitted_together(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, factor in [("a", 1.0), ("b", 7.0)]:
                write_pair(root / name / "dark.tif", np.arange(1, 10).reshape(3, 3) * factor)
                write_pair(root / name / "bright.tif", np.arange(10, 19).reshape(3, 3) * factor)
            one = WaterDataset(root / "a", bands=(1, 2), patch_representation=REPRESENTATION)
            two = WaterDataset(root / "b", bands=(1, 2), patch_representation=REPRESENTATION)
            one.fit_relative_normalization([0, 1])
            two.fit_relative_normalization([0, 1])
            for index in range(2):
                np.testing.assert_allclose(one[index]["image"].numpy(), two[index]["image"].numpy(), rtol=0, atol=6e-8)

    def test_dem_meters_zero_validity_and_json_restore(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_pair(root / "a.tif", [[0, 5], [10, 20]], dem=[[0, 2], [3, 4]],
                       mask=[[1, 1], [1, 0]], dem_scale=2.0, dem_offset=-4.0)
            data = WaterDataset(root, bands=(1, 2), patch_representation=REPRESENTATION)
            fitted = data.fit_relative_normalization([0])
            restored = WaterDataset(root, bands=(1, 2), patch_representation=REPRESENTATION,
                                    relative_normalization=json.loads(json.dumps(fitted)))
            restored.set_relative_normalization(restored.relative_normalization, [0])
            sample = restored[0]
            np.testing.assert_array_equal(sample["valid_mask"], [[True, True], [True, False]])
            np.testing.assert_array_equal(sample["image"][1], [[-4, 0], [2, 0]])
            self.assertTrue(sample["valid_mask"][0, 0])
            self.assertEqual(float(sample["image"][0, 0, 0]), 0)
            np.testing.assert_array_equal(sample["image"], data[0]["image"])

    def test_training_signature_order_and_root_independence_but_split_and_bands_checked(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for folder in ["original", "moved"]:
                write_pair(root / folder / "a.tif", [[1, 2]])
                write_pair(root / folder / "b.tif", [[3, 4]])
            first = WaterDataset(root / "original", bands=(1, 2), patch_representation=REPRESENTATION)
            moved = WaterDataset(root / "moved", bands=(1, 2), patch_representation=REPRESENTATION)
            fitted = first.fit_relative_normalization([0, 1])
            self.assertEqual(fitted, moved.set_relative_normalization(fitted, [1, 0]))
            with self.assertRaisesRegex(ValueError, "划分"):
                moved.set_relative_normalization(fitted, [0])
            swapped = WaterDataset(root / "moved", bands=(2, 1), patch_representation=REPRESENTATION)
            with self.assertRaisesRegex(ValueError, "波段顺序"):
                swapped.set_relative_normalization(fitted, [0, 1])
            with self.assertRaisesRegex(ValueError, "重复"):
                first.fit_relative_normalization([0, 0])

    def test_direct_parameters_always_check_band_order_without_training_indices(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_pair(root / "a.tif", [[1, 2]])
            supplied = parameters(names=("a.tif",), bands=(1, 2))
            valid = WaterDataset(root, bands=(1, 2), patch_representation=REPRESENTATION,
                                 relative_normalization=supplied)
            self.assertEqual(valid.relative_normalization, supplied)
            with self.assertRaisesRegex(ValueError, "波段顺序"):
                WaterDataset(root, bands=(2, 1), patch_representation=REPRESENTATION,
                             relative_normalization=supplied)
            swapped = WaterDataset(root, bands=(2, 1), patch_representation=REPRESENTATION)
            with self.assertRaisesRegex(ValueError, "波段顺序"):
                swapped.set_relative_normalization(supplied)
            self.assertIsNone(swapped.relative_normalization)

    def test_invalid_protocol_and_unfitted_inputs_fail_explicitly(self):
        mutations = ({"p02": float("nan")}, {"p98": 0.0}, {"input_domain": "sigma0_db"},
                     {"fit_pixels": 0}, {"training_samples": True}, {"bands": [1, 1]},
                     {"train_files_sha256": "missing"}, {"unexpected": 1})
        for update in mutations:
            with self.subTest(update=update), self.assertRaises(ValueError):
                validate_relative_normalization(dict(parameters(), **update))
        image = np.zeros((2, 2, 2), dtype=np.float32)
        valid = np.ones((2, 2), bool)
        with self.assertRaises(ValueError):
            apply_vv_train_p02_p98_dem_m(image, valid, None)
        with self.assertRaises(ValueError):
            apply_vv_train_p02_p98_dem_m(image[:1], valid, parameters())
        image[1, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "有限"):
            apply_vv_train_p02_p98_dem_m(image, valid, parameters())

    def test_empty_or_degenerate_training_pixels_fail(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_pair(root / "constant/a.tif", [[3, 3], [3, 3]])
            constant = WaterDataset(root / "constant", bands=(1, 2), patch_representation=REPRESENTATION)
            with self.assertRaisesRegex(ValueError, "P98"):
                constant.fit_relative_normalization([0])
            write_pair(root / "empty/a.tif", [[1, 2]], dem=[[np.nan, np.nan]])
            empty = WaterDataset(root / "empty", bands=(1, 2), patch_representation=REPRESENTATION)
            with self.assertRaisesRegex(ValueError, "没有共同有效"):
                empty.fit_relative_normalization([0])

    def test_legacy_rank_and_stored_datasets_keep_existing_values(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            vv = np.array([[1, 2], [3, 4]], dtype=np.float32)
            dem = np.array([[2, 3], [4, 5]], dtype=np.float32)
            write_pair(root / "a.tif", vv, dem=dem, dem_scale=2.0, dem_offset=1.0)
            old = WaterDataset(root, bands=(1, 2), patch_representation="vv_rank_dem_m")[0]
            np.testing.assert_array_equal(old["image"][0], [[0.125, 0.375], [0.625, 0.875]])
            np.testing.assert_array_equal(old["image"][1], 2 * dem + 1)
            stored = WaterDataset(root, bands=(1, 2), patch_representation="stored")[0]
            np.testing.assert_array_equal(stored["image"], np.stack([vv, dem]))
            with self.assertRaisesRegex(ValueError, "只有"):
                WaterDataset(root, bands=(1, 2), relative_normalization=parameters())


if __name__ == "__main__":
    unittest.main()
