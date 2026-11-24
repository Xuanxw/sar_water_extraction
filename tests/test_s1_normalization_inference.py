"""固定训练分位数的评价协议、四基线选模和 dB 输入单位边界。"""
from copy import deepcopy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.transform import from_origin
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from scripts.models import calibrate_s1_unet_bn as calibration
from scripts.models import predict_best_s1 as prediction
from supervised import predict
from supervised.evaluation import postprocess
from datasets.dfc_sample.s1_dataset import WaterDataset
from datasets.relative_normalization import LOG_REPRESENTATION, REPRESENTATION, training_files_signature
from supervised.evaluation.s1_grd import predict_grd_scene


def normalization_parameters(names=("a.tif", "b.tif")):
    return dict(version=1, method="train_p02_p98", representation=REPRESENTATION,
                input_domain="dfc_stored_values_unknown_physical_scale",
                fit_scope="all_valid_training_pixels", percentiles=[2, 98], range=[0.0, 1.0],
                p02=0.0, p98=100.0, fit_pixels=32, **training_files_signature(names, (1, 4)))


def write_tiff(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", count=len(values), dtype=str(values.dtype),
                       height=values.shape[1], width=values.shape[2], crs="EPSG:32650",
                       transform=from_origin(500000, 4000000, 10, 10)) as dst:
        dst.write(values)


class S1NormalizationInferenceTests(unittest.TestCase):
    def make_calibration(self, root):
        images, labels = root / "images", root / "labels"
        for name, value in (("a.tif", 10), ("b.tif", 80), ("v.tif", 9000)):
            image = np.full((4, 4, 4), value, dtype="float32")
            image[3] = 15
            write_tiff(images / name, image)
        # 训练标签故意缺失，BN 统计和固定 VV 参数复用不得读取它们。
        write_tiff(labels / "v.tif", np.ones((1, 4, 4), dtype="uint8"))
        checkpoint = dict(model_name="sentinel1_water_unet", model_kwargs={}, epoch=80,
                          image_dir=str(images), label_dir=str(labels), bands=(1, 4),
                          input_channels=("VV", "DEM"), patch_representation=REPRESENTATION,
                          relative_normalization=normalization_parameters(), mean=[0.4, 10.0],
                          std=[0.3, 5.0], inputs_normalized=False, ignore_index=-1,
                          train_files=["a.tif", "b.tif"], val_files=["v.tif"],
                          split_manifest_sha256="split", dem_source="Copernicus_GLO30", dem_unit="m")
        paths = {}
        for name in ("first", "second"):
            path = root / f"{name}.pth"
            torch.save(checkpoint, path)
            paths[name] = path
        config = dict(paths=dict(CHECKPOINTS=paths), run=dict(EXPECTED_EPOCHS={name: 80 for name in paths},
                      CALIBRATION_SAMPLES=1, SEED=42, DEVICE="cpu", BATCH_SIZE=1, CPU_THREADS=1))
        return checkpoint, config

    def test_bn_reuses_complete_training_protocol_for_subset_and_validation(self):
        with TemporaryDirectory() as tmp:
            checkpoint, config = self.make_calibration(Path(tmp))
            with patch.object(WaterDataset, "fit_relative_normalization", side_effect=AssertionError("不得重新拟合")):
                checkpoints, train, val, metadata = calibration.prepare_experiment(config)
                self.assertEqual(len(train), 1)
                self.assertEqual(len(val), 1)
                self.assertEqual(train.relative_normalization, checkpoint["relative_normalization"])
                self.assertEqual(val.relative_normalization, checkpoint["relative_normalization"])
                self.assertEqual(metadata["protocol"]["relative_normalization"], checkpoint["relative_normalization"])
                np.testing.assert_array_equal(val[0]["image"][0].numpy(), np.ones((4, 4)))
                np.testing.assert_array_equal(val[0]["image"][1].numpy(), np.full((4, 4), 15))
                self.assertTrue(all(item["split"] == "val" for item in metadata["files"] if item["kind"] == "label"))
                self.assertIn("src/datasets/relative_normalization.py", metadata["code"])
                self.assertEqual(len(checkpoints), 2)

    def test_bn_rejects_missing_parameters_changed_fit_files_and_mixed_protocol(self):
        with TemporaryDirectory() as tmp:
            checkpoint, config = self.make_calibration(Path(tmp))
            missing = deepcopy(checkpoint)
            missing.pop("relative_normalization")
            with self.assertRaisesRegex(ValueError, "已拟合"):
                calibration.checkpoint_protocol(missing)
            altered = deepcopy(checkpoint)
            altered["train_files"] = ["a.tif", "v.tif"]
            with self.assertRaisesRegex(ValueError, "签名"):
                calibration.checkpoint_protocol(altered)
            altered = deepcopy(checkpoint)
            altered["relative_normalization"]["p98"] = 200.0
            torch.save(altered, config["paths"]["CHECKPOINTS"]["second"])
            with self.assertRaisesRegex(ValueError, "输入预处理协议不同"):
                calibration.prepare_experiment(config)

    def test_dfc_log_review_and_inference_restore_the_checkpoint_transform(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint, config = self.make_calibration(root)
            ds = WaterDataset(root / "images", bands=(1, 4), patch_representation=LOG_REPRESENTATION)
            ds.select_files(checkpoint["train_files"])
            fitted = ds.fit_relative_normalization(range(len(ds)))
            checkpoint.update(patch_representation=LOG_REPRESENTATION, relative_normalization=fitted)
            for path in config["paths"]["CHECKPOINTS"].values():
                torch.save(checkpoint, path)
            with patch.object(WaterDataset, "fit_relative_normalization", side_effect=AssertionError("must not refit")):
                _, training, val, meta = calibration.prepare_experiment(config)
            self.assertEqual(meta["protocol"]["relative_normalization"], fitted)
            self.assertEqual(training.relative_normalization, val.relative_normalization)
            np.testing.assert_array_equal(val[0]["image"][0], np.ones((4, 4)))

            # 10 和 80 的几何平均在这两个对数端点的中间。
            raw = np.array([[[np.sqrt(800)]], [[15.0]]], np.float32)
            valid = np.ones((1, 1), dtype=bool)
            inference_config = {"input": {"INPUTS_NORMALIZED": False}, "inference": {"BATCH_SIZE": 1}}
            settings, transform = predict.prediction_settings(checkpoint, inference_config)
            image, mask = transform(raw, valid)
            np.testing.assert_allclose(image[:, 0, 0], [0.5, 15.0], atol=1e-6)
            np.testing.assert_array_equal(mask, valid)
            generic_config = dict(inference_config, output={}, paths={}, postprocess={})
            generic_config['inference'] = dict(inference_config['inference'], THRESHOLD=0.5, USE_AMP=False)
            db_image = root / "images" / "v.tif"
            with rasterio.open(db_image, "r+") as dst:
                dst.set_band_unit(1, "dB")
            with self.assertRaisesRegex(ValueError, "DFC 存储值.*dB"):
                predict.predict_scene(torch.nn.Identity(), db_image, root / "out", settings, generic_config,
                                      checkpoint_info={'sha256': 'test'}, window_transform=transform)
            self.assertFalse((root / "out").exists())
            with self.assertRaisesRegex(ValueError, "DFC 存储值.*Sigma0 dB"):
                predict_grd_scene(torch.nn.Identity(), root / "missing_db.tif", root / "mask.tif",
                                  mean=[0, 0], std=[1, 1], patch_representation=LOG_REPRESENTATION,
                                  bands=(1, 4), dem_path=root / "dem")

            model = torch.nn.Conv2d(2, 1, 1, bias=False)
            model.weight.data.fill_(1.0)
            checkpoint["mean"] = torch.zeros(2)
            checkpoint["std"] = torch.ones(2)
            with patch.object(postprocess.torch, "load", return_value=checkpoint), \
                    patch.object(postprocess, "load_model", return_value=model), \
                    patch.object(torch.cuda, "is_available", return_value=False):
                ensemble = postprocess.WaterEnsemble(["mock.pth"])
            _, probabilities = ensemble.predict(torch.from_numpy(raw[None]), torch.from_numpy(valid[None]),
                                               dem_affines={4: [(0.1, -1.0)]})
            self.assertAlmostEqual(probabilities.item(), torch.sigmoid(torch.tensor(1.0)).item(), places=6)

            checkpoint["patch_representation"] = REPRESENTATION
            with self.assertRaises(ValueError):
                calibration.checkpoint_protocol(checkpoint)
            with self.assertRaises(ValueError):
                predict.prediction_settings(checkpoint, inference_config)

    def test_cli_accepts_three_four_or_more_configs_without_fixed_offsets(self):
        for count in (2, 3, 4, 5):
            configs = [f"model_{number}.yaml" for number in range(count)]
            self.assertEqual(prediction.parse_arguments(configs + ["scene_db.tif"]),
                             (configs, "scene_db.tif", None))
            self.assertEqual(prediction.parse_arguments(configs + ["scene_db.tif", "dem.tif"]),
                             (configs, "scene_db.tif", "dem.tif"))
        for arguments in (["one.yaml", "scene.tif"], ["one.yaml", "two.yaml"],
                          ["one.yaml", "two.yaml", "image", "dem", "extra"]):
            with self.assertRaises(ValueError):
                prediction.parse_arguments(arguments)

    def old_protocol(self):
        return dict(input_channels=("VV", "DEM"), patch_representation="vv_rank_dem_m",
                    inputs_normalized=False, speckle=None, relative_normalization=None,
                    bands=(1, 4), polarizations=("VV",), dem_source="Copernicus_GLO30", dem_unit="m",
                    split_manifest_sha256="same")

    def test_three_and_four_models_select_best_validation_f1(self):
        for count in (3, 4):
            with self.subTest(count=count), TemporaryDirectory() as tmp:
                root = Path(tmp)
                image, dem = root / "scene_db.tif", root / "dem.tif"
                image.touch()
                dem.touch()
                protocol = self.old_protocol()
                candidates = [(dict(model=f"network_{index}", f1=0.2 + index * 0.1, epoch=70,
                                    checkpoint=str(root / f"network_{index}/best_f1.pth")), deepcopy(protocol))
                              for index in range(count)]
                with patch.object(prediction, "candidate", side_effect=candidates), \
                        patch.object(torch.cuda, "is_available", return_value=False), \
                        patch.object(prediction.torch, "load", return_value=dict(mean=[0.5, 0], std=[0.3, 1])), \
                        patch.object(prediction, "load_model", return_value=torch.nn.Identity()), \
                        patch.object(prediction, "predict_grd_scene") as infer:
                    folder = prediction.main([f"model_{i}.yaml" for i in range(count)], image, dem)
                report = json.loads((folder / "selection.json").read_text(encoding="utf-8"))
                self.assertEqual(report["winner"], f"network_{count - 1}")
                self.assertEqual(len(report["candidates"]), count)
                infer.assert_called_once()

    def test_dfc_fixed_values_reject_db_before_model_dem_or_output(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol = self.old_protocol()
            protocol.update(patch_representation=REPRESENTATION, relative_normalization=normalization_parameters())
            candidates = [(dict(model=f"network_{index}", f1=0.4, checkpoint=str(root / f"{index}/best.pth")),
                           deepcopy(protocol)) for index in range(4)]
            with patch.object(prediction, "candidate", side_effect=candidates), \
                    patch.object(prediction, "load_model", side_effect=AssertionError("不得构造模型")), \
                    patch.object(torch.cuda, "is_available", side_effect=AssertionError("不得访问GPU")):
                with self.assertRaisesRegex(ValueError, "DFC 存储值.*Sigma0 dB"):
                    prediction.main([f"{i}.yaml" for i in range(4)], root / "missing_db.tif", root / "missing_dem")
            with patch("supervised.evaluation.s1_grd.aligned_dem_tiles", side_effect=AssertionError("不得准备DEM")):
                with self.assertRaisesRegex(ValueError, "DFC 存储值.*Sigma0 dB"):
                    predict_grd_scene(torch.nn.Identity(), root / "missing_db.tif", root / "output/mask.tif",
                                      mean=[0, 0], std=[1, 1], patch_representation=REPRESENTATION,
                                      bands=(1, 4), dem_path=root / "missing_dem")
            self.assertEqual(list(root.iterdir()), [])

    def test_mixed_fitted_parameters_reject_before_inference(self):
        protocols = [self.old_protocol(), self.old_protocol()]
        for protocol in protocols:
            protocol.update(patch_representation=REPRESENTATION, relative_normalization=normalization_parameters())
        protocols[1]["relative_normalization"]["p98"] = 101.0
        candidates = [(dict(model=f"network_{i}"), protocol) for i, protocol in enumerate(protocols)]
        with patch.object(prediction, "candidate", side_effect=candidates):
            with self.assertRaisesRegex(ValueError, "输入或验证划分不同"):
                prediction.main(["one.yaml", "two.yaml"], "missing_db.tif")


if __name__ == "__main__":
    unittest.main()
