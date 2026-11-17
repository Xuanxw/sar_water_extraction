"""TYSAR VV＋DEM 数据、续跑及训练/融合验证共用网格的回归。"""
import csv
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.transform import from_origin
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from datasets import image_process
from datasets.manifest_paths import source_path
from datasets.scene_dem import prepare_scene_dems
from supervised import train
from supervised.load.sar_scene_dataset import SARSceneDataset, merge_scene_manifests


def raster(path, data, grid, nodata):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=data.shape[0], width=data.shape[1],
                       count=1, dtype=data.dtype, crs="EPSG:4326", transform=grid, nodata=nodata) as f:
        f.write(data, 1)


class SceneDEMTests(unittest.TestCase):
    def make_inputs(self, base):
        vv_root = base / "normalized_p02_p98"
        grids = [from_origin(116 + i, 41, .01, .01) for i in range(2)]
        rows = []
        heights = np.indices((68, 68))[1].astype("float32") - 3.
        heights[8, 8] = -9999
        vv = np.full(heights.shape, .25, dtype="float32")
        vv[1, 1] = np.nan
        truth = (heights > 10).astype("uint8")
        for i, split in enumerate(("train", "val")):
            name = split + ".tif"
            raster(vv_root/"images"/split/name, vv, grids[i], np.nan)
            raster(vv_root/"labels"/split/name, truth, grids[i], 255)
            tile = base/"tiles"/f"Copernicus_DSM_COG_10_N40_00_E{116+i}_00_DEM.tif"
            raster(tile, np.where(heights == -9999, -9999, heights + 1000*i).astype("float32"), grids[i], -9999)
            rows.append(dict(split=split, active=1, scene_id=split, group_id=split, name=name, label_name=name,
                             valid_pixels=vv.size-1, water_pixels=int(truth.sum()), background_pixels=int((truth==0).sum())))
        with (vv_root/"split_manifest.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, rows[0])
            writer.writeheader()
            writer.writerows(rows)
        (vv_root/"preprocessing.json").write_text(json.dumps(dict(stage="normalized", complete=True,
                input_domain="intensity", bc_org_domain="intensity", output_domain="scene_p02_p98")), encoding="utf-8")
        return vv_root, heights, vv

    def test_prepare_train_validate_and_resume(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            vv_root, heights, vv = self.make_inputs(base)
            upstream = (vv_root/"split_manifest.csv").read_bytes()
            cache = base/"cache"
            cfg = dict(paths=dict(DATA_ROOT=str(base), NORMALIZED_ROOT="normalized", DEM_TILE_DIR="tiles",
                                  WINDOW_CACHE_DIR="cache"), run=dict(MODE="dem"),
                       processing=dict(NORMALIZATION_METHOD="scene_p02_p98", WINDOW_SIZE=64, HALO=8))
            config = base/"dem.yaml"
            config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            image_process.main(config)
            derived = vv_root.with_name(vv_root.name + "_vv_dem")
            good = np.isfinite(vv) & (heights != -9999)
            ds = SARSceneDataset(derived, cache, bands=(1, 2), window_size=64, halo=8, require_prepared=True)
            mean, std, _ = ds.normalization(ds.train_ids)
            np.testing.assert_allclose(mean, [0, heights[good].mean()], rtol=1e-6)
            np.testing.assert_allclose(std, [1, heights[good].std()], rtol=1e-6)
            for row in ds.scenes:
                self.assertFalse(Path(row["dem_path"]).is_absolute())
                dem = source_path(row["dem_path"], ds.manifest_path)
                with rasterio.open(dem) as elev, rasterio.open(derived/"images"/row["split"]/row["name"]) as pair:
                    self.assertEqual((elev.shape, elev.transform, elev.crs), (pair.shape, pair.transform, pair.crs))
                    np.testing.assert_array_equal(pair.read(2), elev.read(1))
                    np.testing.assert_array_equal(np.isfinite(elev.read(1)), good)
                    np.testing.assert_array_equal(pair.read(1), vv)
            ds.close()
            result = derived/"dem/train/train.tif"
            stamp = result.stat().st_mtime_ns
            image_process.main(config)
            self.assertEqual(result.stat().st_mtime_ns, stamp)
            self.assertEqual(upstream, (vv_root/"split_manifest.csv").read_bytes())
            # 经正式配置工厂构建 2 通道 PVT，只用微型本地样本与随机初始化，验证一轮和检查点。
            cfg = yaml.safe_load((REPO/"configs/168/train_tysar_vv_dem.yaml").read_text(encoding="utf-8"))
            cfg["paths"].update(DATA_ROOT=str(base), IMAGE_DIR=str(derived/"images"), LABEL_DIR=str(derived/"labels"),
                                MANIFEST_PATHS=[str(ds.manifest_path)], OUTPUT_DIR=str(base/"run"),
                                OUTPUT_ROOT=str(base/"runs"), WINDOW_CACHE_DIR=str(cache))
            cfg["model"]["PRETRAINED"] = False
            cfg["data"]["SCENE_OPTIONS"].update(num_workers=0, window_size=64, halo=8)
            cfg["validation_inference"].update(stride=24, num_workers=0)
            cfg["training"].update(EPOCHS=1, CPU_THREADS=1)
            cfg["quality"].update(SAVE_VALIDATION_FIGURES=False, SAVE_FEATURE_MAPS=False)
            config = base/"train.yaml"
            config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            previous = train.CONFIG_PATH or REPO/"configs/train.yaml"
            self.addCleanup(train.configure, previous)
            with patch.object(torch.cuda, "is_available", return_value=False):
                train.configure(config)
                trainer = train.create_trainer()
                try:
                    trainer.train()
                    self.assertEqual(trainer.phase_metrics["val"]["valid_pixels"], int(good.sum()))
                    ckpt = torch.load(trainer.run_dir/"last.pth", weights_only=True, map_location="cpu")
                    self.assertEqual(tuple(ckpt["input_channels"]), ("VV", "DEM"))
                    self.assertEqual(ckpt["dem_unit"], "m")
                    np.testing.assert_allclose(ckpt["mean"], mean, rtol=1e-6)
                    backup = trainer.run_dir/"split_manifest.csv"
                    with backup.open(encoding="utf-8-sig", newline="") as f:
                        saved = next(csv.DictReader(f))
                    self.assertEqual(source_path(saved["dem_path"], backup), result)
                finally:
                    trainer.dataset.close()

    def test_two_batches_share_train_dem_scale(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            manifests, expected = [], []
            cache = base/"cache"
            for i, batch in enumerate(("a", "b")):
                parent = base/batch
                source, heights, vv = self.make_inputs(parent)
                manifest = source/"split_manifest.csv"
                with manifest.open(encoding="utf-8-sig", newline="") as stream:
                    rows = list(csv.DictReader(stream))
                for row in rows:
                    row["scene_id"] = row["group_id"] = batch + "_" + row["split"]
                with manifest.open("w", encoding="utf-8-sig", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)
                if i:
                    heights[heights != -9999] += 100
                    heights[:10] = -9999
                    tile = next((parent/"tiles").glob("*E116*"))
                    with rasterio.open(tile, "r+") as dest:
                        dest.write(heights, 1)
                expected.append(heights[(heights != -9999) & np.isfinite(vv)])
                derived = prepare_scene_dems(source, parent/"tiles", cache, 64, 8)
                manifests.append(derived/"split_manifest.csv")
            options = dict(cache_dir=cache, bands=(1, 2), window_size=64, halo=8, require_prepared=True)
            merged = merge_scene_manifests(manifests, **options)
            ds = SARSceneDataset(merged, **options)
            try:
                mean, std, _ = ds.normalization(ds.train_ids)
                heights = np.concatenate(expected).astype("float64")
                np.testing.assert_allclose(mean, [0, heights.mean()])
                np.testing.assert_allclose(std, [1, heights.std()])
                self.assertEqual(len(ds.scenes), 4)
                for row in ds.scenes:
                    self.assertTrue(source_path(row["dem_path"], ds.manifest_path).is_file())
            finally:
                ds.close()

    def test_missing_tiles_leave_completed_scenes_for_retry(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            root, _, _ = self.make_inputs(base)
            tile = next((base/"tiles").glob("*E117*"))
            tile.rename(tile.with_suffix(".bak"))
            target = root.with_name(root.name+"_vv_dem")
            with self.assertRaisesRegex(ValueError, "未覆盖"):
                prepare_scene_dems(root, base/"tiles", base/"cache", 64, 8)
            self.assertTrue((target/".preprocessing_incomplete").is_file())
            self.assertFalse((target/"split_manifest.csv").exists())
            first = target/"dem/train/train.tif"
            stamp = first.stat().st_mtime_ns
            tile.with_suffix(".bak").rename(tile)
            prepare_scene_dems(root, base/"tiles", base/"cache", 64, 8)
            self.assertEqual(stamp, first.stat().st_mtime_ns)
            self.assertFalse((target/".preprocessing_incomplete").exists())
