"""用既有 TYSAR 对齐 DEM 生成整景坡度及 VV＋坡度训练成果。"""

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET

from filelock import FileLock
import numpy as np
import rasterio
from rasterio.windows import Window

from datasets.dfc_terrain import slope_degrees
from datasets.manifest_paths import relative_source, source_path
from datasets.scene_dem import _file_state, _write_input_vrt
from supervised.load.sar_scene_dataset import SARSceneDataset
from utils.files import sha256, write_json
from utils.raster import windows


SLOPE_RADIUS_M = 30.
SLOPE_PROTOCOL = dict(version=1, unit="degree", radius_m=SLOPE_RADIUS_M,
                      method="central_or_one_sided_finite_difference",
                      valid_support="finite_contiguous_dem_on_both_axes",
                      normalization="train_slope_pixels_zscore")


def _prepare_scene(source, target, row):
    """对单景正交地图网格分块计算角度，写入坡度和相应监督掩膜。"""
    split, scene = row["split"], row["scene_id"]
    if split not in ("train", "val") or not scene or Path(scene).name != scene:
        raise ValueError(f"无效划分或景号：{split}/{scene}")
    vrt = source/"images"/split/row["name"]
    vv_node = ET.parse(vrt).find("./VRTRasterBand[@band='1']/SimpleSource/SourceFilename")
    if vv_node is None or not vv_node.text:
        raise ValueError(f"缺少原归一化 VV 引用：{vrt}")
    vv = source_path(vv_node.text, vrt)
    dem = source_path(row["dem_path"], source/"split_manifest.csv")
    label = source/"labels"/split/row["label_name"]
    slope = target/"slope"/split/(scene+".tif")
    label_out = target/"labels"/split/(scene+".tif")
    receipt = slope.with_suffix(".json")
    inputs = [vv, dem, label]
    inputs += [sidecar for p in (vv, dem, label) for suffix in (".msk", ".aux.xml")
               if (sidecar := Path(str(p)+suffix)).is_file()]
    signature = dict(protocol=SLOPE_PROTOCOL, files=[[relative_source(p, receipt), *_file_state(p)] for p in inputs])
    if receipt.is_file() and slope.is_file() and label_out.is_file():
        old = json.loads(receipt.read_text(encoding="utf-8"))
        if old["inputs"] == signature and old["outputs"] == [_file_state(p) for p in (slope, label_out)]:
            return vv, slope, old["row"], True

    for p in (slope, label_out):
        p.parent.mkdir(parents=True, exist_ok=True)
    n = water = land = 0
    total = squares = 0.
    with TemporaryDirectory(dir=target) as tmp, rasterio.open(vv) as sar, \
            rasterio.open(dem) as terrain, rasterio.open(label) as gt:
        grid = sar.crs, sar.transform, sar.shape
        if not sar.crs or not sar.crs.is_projected or sar.transform.is_degenerate:
            raise ValueError(f"坡度需要带米制距离的投影网格：{vv}")
        if any(ds.count != 1 or (ds.crs, ds.transform, ds.shape) != grid for ds in (sar, terrain, gt)):
            raise ValueError(f"VV、DEM、标签必须同网格：{scene}")
        affine = sar.transform
        unit = sar.crs.linear_units_factor[1]
        col = np.array([affine.a, affine.d]) * unit
        line = np.array([affine.b, affine.e]) * unit
        spacings = np.linalg.norm(line), np.linalg.norm(col)
        if not np.isclose(np.dot(col, line) / np.prod(spacings), 0., atol=1e-6):
            raise ValueError(f"坡度要求相互正交的地图像元轴：{vv}")
        ry, rx = (max(1, round(SLOPE_RADIUS_M / step)) for step in spacings)
        slope_tmp, label_tmp = Path(tmp)/"slope.tif", Path(tmp)/"label.tif"
        profile = dict(driver="GTiff", count=1, width=sar.width, height=sar.height, crs=sar.crs,
                       transform=affine, compress="deflate", tiled=True, BIGTIFF="IF_SAFER")
        with rasterio.open(slope_tmp, "w", **profile, dtype="float32", nodata=np.nan, predictor=3) as dst, \
                rasterio.open(label_tmp, "w", **profile, dtype="uint8", nodata=255) as out:
            dst.set_band_unit(1, "degree")
            dst.update_tags(difference_radius_y_m=ry * spacings[0], difference_radius_x_m=rx * spacings[1])
            for win in windows(sar, 1024):
                x, y, w, h = (int(v) for v in (win.col_off, win.row_off, win.width, win.height))
                left, top = max(0, x-rx), max(0, y-ry)
                full = Window(left, top, min(sar.width, x+w+rx)-left, min(sar.height, y+h+ry)-top)
                dem_values = terrain.read(1, window=full, masked=True).filled(np.nan)
                degrees = slope_degrees(dem_values, spacings, (ry, rx))[y-top:y-top+h, x-left:x-left+w]
                vv_values = sar.read(1, window=win, masked=True).filled(np.nan)
                valid = np.isfinite(vv_values) & np.isfinite(degrees)
                target_labels = gt.read(1, window=win, masked=True)
                known = valid & ~np.ma.getmaskarray(target_labels) & np.isin(target_labels.data, (0, 1))
                labels = np.where(known, target_labels.data, 255).astype("uint8")
                dst.write(np.where(valid, degrees, np.nan), 1, window=win)
                out.write(labels, 1, window=win)
                n += int(valid.sum())
                water += int((labels == 1).sum())
                land += int((labels == 0).sum())
                if split == "train":
                    z = degrees[valid].astype(np.float64)
                    total += float(z.sum())
                    squares += float(np.square(z).sum())
        if not water + land:
            raise ValueError(f"第 {scene} 景没有 VV、坡度和标签共同有效像元")
        slope_tmp.replace(slope)
        label_tmp.replace(label_out)
        result = dict(valid_pixels=n, water_pixels=water, background_pixels=land,
                      ignored_pixels=sar.width*sar.height-water-land, slope_sum_deg=total,
                      slope_sum_sq_deg2=squares)
    write_json(receipt, dict(inputs=signature, outputs=[_file_state(p) for p in (slope, label_out)], row=result))
    return vv, slope, result, False


def prepare_scene_slopes(source, target, cache, window_size, halo):
    """按原 DEM 划分续跑，拟合训练景坡度标准化并发布两波段 VRT。"""
    source, target = Path(source), Path(target)
    manifest = source/"split_manifest.csv"
    metadata = source/"preprocessing.json"
    upstream = json.loads(metadata.read_text(encoding="utf-8")) if metadata.is_file() else {}
    if ((upstream.get("stage"), upstream.get("complete"), upstream.get("dem_unit")) != ("vv_dem", True, "m")
            or (source/".preprocessing_incomplete").exists()):
        raise ValueError(f"请先完成 MODE=dem：{source}")
    upstream_hash = sha256(manifest)
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if int(row["active"]) == 1]
    if {row["split"] for row in rows} != {"train", "val"} or len({row["scene_id"] for row in rows}) != len(rows):
        raise ValueError("原 DEM 清单缺少 train/val 或景号重复")
    target.mkdir(parents=True, exist_ok=True)
    pairs = []
    with FileLock(str(target/".slope.lock"), timeout=0):
        marker = target/".preprocessing_incomplete"
        marker.touch()
        for index, row in enumerate(rows, 1):
            vv, slope, stats, reused = _prepare_scene(source, target, row)
            row.update(stats)
            pairs.append((vv, slope))
            print(f"坡度 {index}/{len(rows)} [{row['split']}] {row['scene_id']} "
                  f"{'复用' if reused else '完成'}，有效标签 {row['water_pixels']+row['background_pixels']:,}", flush=True)
        train = [row for row in rows if row["split"] == "train"]
        count = sum(row["valid_pixels"] for row in train)
        mean = sum(row["slope_sum_deg"] for row in train) / count
        variance = sum(row["slope_sum_sq_deg2"] for row in train) / count - mean**2
        std = max(variance, 0.)**.5
        if std < 1e-6:
            std = 1.
        if (sha256(manifest) != upstream_hash or (source/".preprocessing_incomplete").exists()
                or json.loads(metadata.read_text(encoding="utf-8")) != upstream):
            raise ValueError("DEM 成果在坡度处理期间发生变化，请重跑")

        dest = target/"split_manifest.csv"
        for row, (vv, slope) in zip(rows, pairs):
            vrt = target/"images"/row["split"]/(row["scene_id"]+".vrt")
            vrt.parent.mkdir(parents=True, exist_ok=True)
            pending_vrt = vrt.with_suffix(".vrt.partial")
            with rasterio.open(vv) as sar:
                _write_input_vrt(pending_vrt, vrt, vv, slope, sar,
                                 aux_description="Terrain slope (training z-score)",
                                 aux_transform=(1/std, -mean/std))
            if not vrt.is_file() or vrt.read_bytes() != pending_vrt.read_bytes():
                pending_vrt.replace(vrt)
            else:
                pending_vrt.unlink()
            row.update(name=vrt.name, label_name=slope.name, slope_path=relative_source(slope, dest))
            for key in ("source_image", "source_label", "dem_path"):
                if row.get(key):
                    row[key] = relative_source(source_path(row[key], manifest), dest)
        pending = dest.with_suffix(".csv.partial")
        with pending.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        pending.replace(dest)
        # 坡度标准化由 VRT 完成，VV 沿用既有归一化值，训练入口直接读取两通道。
        write_json(target/"preprocessing.json", dict(
            upstream, stage="normalized", bands=[1, 2], input_channels=["VV", "SLOPE"],
            slope=dict(SLOPE_PROTOCOL, mean_deg=mean, std_deg=std, train_pixels=count),
            dem_manifest_sha256=upstream_hash))
        dataset = SARSceneDataset(target, cache, bands=(1, 2), window_size=window_size, halo=halo)
        dataset.close()
        marker.unlink()
    print(f"坡度准备完成：{dest}", flush=True)
