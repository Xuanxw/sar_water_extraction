"""验证 TYSAR 坡度跨分块、掩膜和整景训练入口。"""

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.transform import from_origin
import torch
import yaml

from datasets import image_process, scene_slope
from datasets.dfc_terrain import dfc_terrain_from_checkpoint
from datasets.scene_dem import _write_input_vrt
from supervised import train
from utils.raster import windows


class SceneSlopeTests(unittest.TestCase):
    def test_slope_mode_and_fused_validation(self):
        with TemporaryDirectory() as tmp, patch.dict(vars(image_process)), patch.dict(vars(train)):
            root = Path(tmp)
            upstream = root/"normalized_relative_db_p02_p98_vv_dem"
            target = root/"normalized_relative_db_p02_p98_vv_slope"
            transform = from_origin(450000., 4500000., 3., 3.)
            profile = dict(driver="GTiff", width=68, height=68, count=1, crs="EPSG:32650", transform=transform)
            _, x = np.indices((68, 68))
            rows = []
            for split, gradient in (("train", .2), ("val", .5)):
                vv = root/"vv"/(split+".tif")
                dem = upstream/"dem"/split/(split+".tif")
                gt = upstream/"labels"/split/(split+".tif")
                vrt = upstream/"images"/split/(split+".vrt")
                height = (x * 3 * gradient).astype("float32")
                height[25:34, 29:36] = np.nan
                for path, values, nodata in ((vv, np.full(x.shape, .5, "float32"), np.nan),
                                              (dem, height, np.nan), (gt, (x > 32).astype("uint8"), 255)):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with rasterio.open(path, "w", **profile, dtype=values.dtype, nodata=nodata) as out:
                        out.write(values, 1)
                vrt.parent.mkdir(parents=True)
                with rasterio.open(vv) as sar:
                    _write_input_vrt(vrt, vrt, vv, dem, sar)
                rows.append(dict(name=vrt.name, split=split, label_name=gt.name, active=1,
                                 scene_id=split, group_id=split, dem_path=dem.relative_to(upstream).as_posix()))
            with (upstream/"split_manifest.csv").open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            (upstream/"preprocessing.json").write_text(json.dumps(dict(stage="vv_dem", complete=True,
                  bc_org_domain="intensity", input_domain="intensity", dem_unit="m",
                  output_domain="scene_relative_db_p02_p98")), encoding="utf-8")
            prep = root/"prep.yaml"
            prep.write_text(yaml.safe_dump(dict(paths=dict(DATA_ROOT=str(root), NORMALIZED_ROOT="normalized",
                      WINDOW_CACHE_DIR="cache"), run=dict(MODE="slope"),
                      processing=dict(NORMALIZATION_METHOD="scene_relative_db_p02_p98", WINDOW_SIZE=64, HALO=8))),
                      encoding="utf-8")
            with patch.object(scene_slope, "windows", side_effect=lambda ds, _: windows(ds, 19)):
                image_process.main(prep)
                file = target/"slope"/"val"/"val.tif"
                mtime = file.stat().st_mtime_ns
                image_process.main(prep)
                self.assertEqual(file.stat().st_mtime_ns, mtime)
            with rasterio.open(file) as degree, rasterio.open(target/"images"/"val"/"val.vrt") as pair, \
                    rasterio.open(target/"labels"/"val"/"val.tif") as gt:
                slope, normalized, truth = degree.read(1), pair.read(2), gt.read(1)
                expected = np.degrees(np.arctan(.5))
                np.testing.assert_allclose(slope[np.isfinite(slope)], expected, atol=1e-4)
                np.testing.assert_allclose(normalized[np.isfinite(slope)],
                                           expected - np.degrees(np.arctan(.2)), atol=1e-4)
                np.testing.assert_array_equal(~np.isfinite(slope), ~np.isfinite(normalized))
                np.testing.assert_array_equal(truth == 255, ~np.isfinite(slope))
                self.assertEqual(pair.transform, transform)
                self.assertTrue(np.all(pair.read(1) == .5))
                valid = int((truth != 255).sum())
                self.assertLess(valid, 68 * 68)

            cfg = yaml.safe_load(Path("configs/train.yaml").read_text(encoding="utf-8"))
            cfg["paths"].update(DATA_ROOT=str(root), IMAGE_DIR=str(target/"images"), LABEL_DIR=str(target/"labels"),
                                MANIFEST_PATHS=[str(target/"split_manifest.csv")], WINDOW_CACHE_DIR="cache",
                                OUTPUT_ROOT="runs", OUTPUT_DIR="runs/slope", RESUME_CHECKPOINT=None)
            cfg["data"].update(SLOPE_BAND=2)
            cfg["data"]["SCENE_OPTIONS"].update(window_size=64, halo=8, num_workers=0)
            cfg["model"]["PRETRAINED"] = False
            cfg["training"].update(EPOCHS=1, CPU_THREADS=1)
            cfg["quality"]["SAVE_VALIDATION_FIGURES"] = False
            cfg["validation_inference"]["stride"] = 24
            conf = root/"train.yaml"
            conf.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            with patch.object(torch.cuda, "is_available", return_value=False):
                train.configure(conf)
                trainer = train.create_trainer()
                try:
                    self.assertEqual(trainer.model_info["input_channels"], ("VV", "SLOPE"))
                    trainer.train()
                    self.assertEqual(trainer.phase_metrics["val"]["valid_pixels"], valid)
                    checkpoint = torch.load(trainer.run_dir/"last.pth", weights_only=True, map_location="cpu")
                    self.assertIsNone(dfc_terrain_from_checkpoint(checkpoint))
                finally:
                    trainer.dataset.close()
