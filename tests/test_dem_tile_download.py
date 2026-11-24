import csv
import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from datasets import dem_tile_download


def raster(path, west):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", crs="EPSG:4326",
                       width=4, height=4, count=1, dtype="float32",
                       transform=from_origin(west, 40.5, .001, .001)) as dst:
        dst.write(np.ones((4, 4), dtype="float32"), 1)


class Cop30DownloadTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.vv = Path(temp.name)
        self.tiles = self.vv / "tiles"
        self.method = "scene_relative_db_p02_p98"
        self.receipt = self.vv / "preprocessing.json"
        self.receipt.write_text(json.dumps(dict(stage="normalized", complete=True,
                                                output_domain=self.method)), encoding="utf-8")
        with (self.vv / "split_manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
            out = csv.DictWriter(f, fieldnames=["active", "split", "name", "scene_id"])
            out.writeheader()
            for scene, split, west in (("a", "train", 116), ("b", "val", 116), ("c", "val", 117)):
                name = scene + ".tif"
                out.writerow(dict(active=1, split=split, name=name, scene_id=scene))
                raster(self.vv / "images" / split / name, west + .1)
            out.writerow(dict(active=0, split="unused", name="absent.tif", scene_id="excluded"))

    def test_inspect_and_reuse_external_cache(self):
        with patch.object(dem_tile_download, "download_tile") as download, redirect_stdout(io.StringIO()) as log:
            dem_tile_download.ensure_dem_tiles(self.vv, self.tiles, self.method, list_only=True)
            self.assertFalse(self.tiles.exists())
            self.assertIn("训练 1 景，验证 2 景", log.getvalue())
            names = [line for line in log.getvalue().splitlines() if line.endswith("_DEM.tif")]
            self.assertEqual(len(names), 2)
            for name in names:
                raster(self.tiles / name, 116.1 if "E116_" in name else 117.1)
            dem_tile_download.ensure_dem_tiles(self.vv, self.tiles, self.method)
            download.assert_not_called()
            self.assertCountEqual((p.name for p in self.tiles.iterdir()), names)

    def test_download_failure_reports_scene_and_resumes(self):
        def first_run(url, path):
            if "E117_" in url:
                raise RuntimeError("HTTP 404")
            raster(path, 116.1)

        with patch.object(dem_tile_download, "download_tile", side_effect=first_run) as fetch:
            with self.assertRaisesRegex(RuntimeError, r"E117_.*val/c.*404"):
                dem_tile_download.ensure_dem_tiles(self.vv, self.tiles, self.method)
            self.assertEqual(fetch.call_count, 2)
        with patch.object(dem_tile_download, "download_tile", side_effect=lambda url, path: raster(path, 117.1)) as fetch:
            dem_tile_download.ensure_dem_tiles(self.vv, self.tiles, self.method)
            self.assertEqual(fetch.call_count, 1)
            self.assertIn("E117_", str(fetch.call_args.args[1]))

    def test_unfinished_or_wrong_domain_never_downloads(self):
        with patch.object(dem_tile_download, "download_tile") as fetch:
            with self.assertRaisesRegex(ValueError, "归一化方法"):
                dem_tile_download.ensure_dem_tiles(self.vv, self.tiles, "scene_p02_p98")
            (self.vv / ".preprocessing_incomplete").touch()
            with self.assertRaisesRegex(ValueError, "先完成 VV 归一化"):
                dem_tile_download.ensure_dem_tiles(self.vv, self.tiles, self.method)
            fetch.assert_not_called()
            self.assertFalse(self.tiles.exists())

    def test_pixel_center_on_west_and_north_tile_edges(self):
        def names(bounds):
            return [spec["tile"] for spec in dem_tile_download._tiles_for_grid(bounds)]

        west = names((115.99990, 40.25, 115.99995, 40.26))
        north = names((116.25, 41.00004, 116.26, 41.00008))
        self.assertEqual(west, ["Copernicus_DSM_COG_10_N40_00_E116_00_DEM"])
        self.assertEqual(north, west)
