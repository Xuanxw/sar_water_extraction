"""从发布目录准备小型整景，再走实际训练和融合验证入口。"""

import csv
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
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from datasets import s1s2_water
from scripts.data import prepare_s1s2_water
from supervised import train


def write_raster(path, data, transform):
    """生成带地图网格的单波段掩膜、米值高程或双波段 dB×100 SAR。"""
    data = np.atleast_3d(data).transpose(2, 0, 1)
    with rasterio.open(path, "w", driver="GTiff", crs="EPSG:32650", transform=transform,
                       dtype=data.dtype, count=data.shape[0], height=data.shape[1], width=data.shape[2]) as dst:
        dst.write(data)


class S1S2WaterPreparationTests(unittest.TestCase):
    def test_official_split_mask_resampling_and_full_scene_validation(self):
        with TemporaryDirectory() as scratch, patch.dict(vars(train)):
            root = Path(scratch)
            target = root / "prepared_vv_dem"
            links = []
            grid = from_origin(450000, 4400000, 10, 10)
            height_grid = from_origin(450000, 4400000, 30, 30)
            for scene, split, z in (("1", "train", 10), ("2", "val", 100), ("3", "test", 900),
                                    ("31", "test", None), ("59", "train", None)):
                folder = root / scene
                folder.mkdir()
                meta = folder / f"sentinel12_{scene}_meta.json"
                meta.write_text(json.dumps({"properties": {"split": split, "s1_srcids": ["S1A_1SDV_TEST"]}}))
                links.append(dict(rel="item", href=meta.relative_to(root).as_posix()))
                if z is None:
                    continue
                stem = folder / f"sentinel12_s1_{scene}"
                sar = np.full((68, 68, 2), -1750, dtype="int16")
                label = np.zeros(sar.shape[:2], dtype="uint8")
                label[:, 34:] = 1
                label[0, 3] = 255
                valid = np.ones_like(label)
                valid[0, 0] = valid[0, 34] = 0  # 非水和水都按官方掩膜排除。
                write_raster(stem.with_name(stem.name + "_img.tif"), sar, grid)
                write_raster(stem.with_name(stem.name + "_msk.tif"), label, grid)
                write_raster(stem.with_name(stem.name + "_valid.tif"), valid, grid)
                write_raster(folder / f"sentinel12_copdem30_{scene}_elevation.tif",
                             np.full((24, 24), z, dtype="int16"), height_grid)
            (root / "catalog.json").write_text(json.dumps(dict(links=links)))
            config = yaml.safe_load((ROOT / "configs/s1/train_s1s2_water_vv_dem.yaml").read_text(encoding="utf-8"))
            config["paths"].update(DATA_ROOT=str(root), OUTPUT_ROOT="runs", OUTPUT_DIR="runs/experiment")
            config["data"]["SCENE_OPTIONS"].update(window_size=64, halo=8, num_workers=0)
            config["validation_inference"]["stride"] = 24
            config["training"].update(EPOCHS=1, BATCH_SIZE=2, CPU_THREADS=1)
            config["quality"]["SAVE_VALIDATION_FIGURES"] = False
            config_path = root / "train.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

            with patch.object(s1s2_water, "OFFICIAL_COUNTS", dict(train=2, val=1, test=2)):
                prepare_s1s2_water.main(config_path)
                ready = target / "dem/train/s1s2water_1.tif"
                timestamp = ready.stat().st_mtime_ns
                prepare_s1s2_water.main(config_path)
                self.assertEqual(ready.stat().st_mtime_ns, timestamp)

            for filename, expected in (("split_manifest.csv", {"1": "train", "2": "val"}),
                                       ("test_manifest.csv", {"3": "test"})):
                with (target / filename).open(encoding="utf-8-sig", newline="") as stream:
                    actual = {row["official_scene"]: row["split"] for row in csv.DictReader(stream)}
                self.assertEqual(actual, expected)
            with rasterio.open(target / "images/train/s1s2water_1.vrt") as pair:
                self.assertEqual(pair.transform, grid)
                np.testing.assert_array_equal(pair.read(1), np.full((68, 68), .5))
                np.testing.assert_array_equal(pair.read(2), np.full((68, 68), 10))
            with rasterio.open(target / "labels/val/s1s2water_2.tif") as label_out:
                self.assertEqual(np.count_nonzero(label_out.read(1) == 255), 3)

            with patch.object(torch.cuda, "is_available", return_value=False):
                train.configure(config_path)
                trainer = train.create_trainer()
                try:
                    np.testing.assert_allclose(trainer.mean.cpu().flatten().numpy(), [0., 10.])
                    trainer.train()
                    self.assertEqual(trainer.phase_metrics["train"]["valid_pixels"], 68 * 68 - 3)
                    self.assertEqual(trainer.phase_metrics["val"]["valid_pixels"], 68 * 68 - 3)
                    self.assertEqual([row["split"] for row in trainer.dataset.scenes], ["train", "val"])
                finally:
                    trainer.dataset.close()
