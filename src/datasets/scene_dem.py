"""按最终 VV 地图网格准备天仪整景训练所用的 Copernicus DEM 和两通道 VRT。"""
import csv
import json
import time
import xml.etree.ElementTree as ET
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

from filelock import FileLock
import numpy as np
import rasterio

from datasets.copernicus_dem import DemTileCatalog, aligned_dem_tiles, read_elevation
from datasets.manifest_paths import relative_source, source_path
from supervised.load.sar_scene_dataset import SARSceneDataset
from utils.files import sha256, write_json
from utils.raster import windows


def _file_state(path):
    """归档相对文件名和便宜的变更标识，数据目录整体迁移后仍可复用。"""
    stat = path.stat()
    return [path.name, stat.st_size, stat.st_mtime_ns]


def _write_input_vrt(pending, destination, vv, auxiliary, grid, *,
                     aux_description="Copernicus GLO-30 elevation (m)", aux_transform=None):
    """引用原 VV 与地形栅格；可在读取第二波段时应用线性缩放，NaN 保持无效。"""
    xml = ET.Element("VRTDataset", rasterXSize=str(grid.width), rasterYSize=str(grid.height))
    ET.SubElement(xml, "SRS").text = grid.crs.to_wkt()
    ET.SubElement(xml, "GeoTransform").text = ",".join(map(str, grid.transform.to_gdal()))
    for band, path, description in ((1, vv, "VV"), (2, auxiliary, aux_description)):
        channel = ET.SubElement(xml, "VRTRasterBand", dataType="Float32", band=str(band))
        ET.SubElement(channel, "Description").text = description
        ET.SubElement(channel, "NoDataValue").text = "nan"
        linear = aux_transform if band == 2 else None
        src = ET.SubElement(channel, "ComplexSource" if linear is not None else "SimpleSource")
        if linear is not None:
            ratio, offset = linear
            ET.SubElement(src, "NODATA").text = "nan"
            ET.SubElement(src, "ScaleRatio").text = str(ratio)
            ET.SubElement(src, "ScaleOffset").text = str(offset)
        name = relative_source(path, destination)
        ET.SubElement(src, "SourceFilename", relativeToVRT="0" if Path(name).is_absolute() else "1").text = name
        ET.SubElement(src, "SourceBand").text = "1"
    ET.ElementTree(xml).write(pending, encoding="utf-8", xml_declaration=True)


def _prepare_pair(vv, label, catalog, target, row):
    """对齐一景 DSM；在 SAR 与 DEM 有效交集内统计米制高程，标签未知写 255。"""
    split, scene = row["split"], row["scene_id"]
    dem_path = target / "dem" / split / (scene + ".tif")
    label_out = target / "labels" / split / (scene + ".tif")
    vrt = target / "images" / split / (scene + ".vrt")
    products = (dem_path, label_out, vrt)
    for path in products:
        path.parent.mkdir(parents=True, exist_ok=True)
    receipt = dem_path.with_suffix(".json")
    with ExitStack() as stack:
        sar = stack.enter_context(rasterio.open(vv))
        gt = stack.enter_context(rasterio.open(label))
        if sar.count != 1 or gt.count != 1 or (sar.shape, sar.crs, sar.transform) != (gt.shape, gt.crs, gt.transform):
            raise ValueError(f"{scene} 需要最终单 VV 影像和同网格标签")
        paths = catalog.covering(sar)
        inputs = dict(version=1, split=split, vv_path=relative_source(vv, vrt),
                      files=[_file_state(p) for p in (vv, label, *paths)])
        # 外部掩膜/附加栅格元数据变化也会改变共同有效域。
        for p in (vv, label, *paths):
            for suffix in (".msk", ".aux.xml"):
                sidecar = Path(str(p) + suffix)
                if sidecar.is_file():
                    inputs["files"].append(_file_state(sidecar))
        if receipt.is_file() and all(p.is_file() for p in products):
            record = json.loads(receipt.read_text(encoding="utf-8"))
            if record["inputs"] == inputs and record["outputs"] == [_file_state(p) for p in products]:
                return record["row"], True
        tiles = aligned_dem_tiles(paths, sar, stack)
        count = water = land = 0
        total = square_total = 0.0
        profile = dict(driver="GTiff", height=sar.height, width=sar.width, count=1, crs=sar.crs,
                       transform=sar.transform, tiled=True, compress="deflate", BIGTIFF="IF_SAFER")
        with TemporaryDirectory(dir=target) as temp:
            pending = Path(temp)
            with rasterio.open(pending/"dem.tif", "w", **profile, dtype="float32", nodata=np.nan) as dem, \
                    rasterio.open(pending/"label.tif", "w", **profile, dtype="uint8", nodata=255) as dst:
                dem.set_band_unit(1, "m")
                dem.update_tags(product="Copernicus GLO-30 DSM", vertical_datum="EGM2008")
                for win in windows(sar, 1024):
                    vv_values = sar.read(1, window=win, masked=True)
                    heights = read_elevation(tiles, win)
                    valid = (~np.ma.getmaskarray(vv_values) & np.isfinite(vv_values.data) & np.isfinite(heights))
                    heights[~valid] = np.nan
                    dem.write(heights, 1, window=win)
                    labels = gt.read(1, window=win, masked=True)
                    known = valid & ~np.ma.getmaskarray(labels) & np.isin(labels.data, (0, 1))
                    values = np.where(known, labels.data, 255).astype("uint8")
                    dst.write(values, 1, window=win)
                    count += int(np.count_nonzero(valid))
                    water += int(np.count_nonzero(values == 1))
                    land += int(np.count_nonzero(values == 0))
                    if split == "train":
                        meters = heights[valid].astype("float64")
                        total += meters.sum()
                        square_total += np.square(meters).sum()
            if not water + land:
                raise ValueError(f"{scene} 的 VV、DEM 和有效标签无共同像元")
            _write_input_vrt(pending/"image.vrt", vrt, vv, dem_path, sar)
            (pending/"dem.tif").replace(dem_path)
            (pending/"label.tif").replace(label_out)
            (pending/"image.vrt").replace(vrt)
        updated = dict(name=vrt.name, label_name=label_out.name,
                       dem_path=relative_source(dem_path, target/"split_manifest.csv"),
                       valid_pixels=count, water_pixels=water, background_pixels=land,
                       ignored_pixels=sar.width*sar.height-land-water,
                       dem_sum_m=total if split == "train" else "",
                       dem_sum_sq_m2=square_total if split == "train" else "")
        write_json(receipt, dict(inputs=inputs, outputs=[_file_state(p) for p in products], row=updated))
        return updated, False


def prepare_scene_dems(vv_root, tile_dir, cache_dir, window_size, halo):
    """使用完整归一化 VV 成果构建独立训练目录；逐景可续跑，最后发布清单和窗口索引。"""
    source = Path(vv_root)
    target = source.with_name(source.name + "_vv_dem")
    manifest = source/"split_manifest.csv"
    upstream = source/"preprocessing.json"
    record = json.loads(upstream.read_text(encoding="utf-8"))
    if (record.get("stage") != "normalized" or not record.get("complete")
            or record.get("bc_org_domain") != "intensity" or record.get("input_domain") != "intensity"
            or record.get("output_domain") not in ("standardized", "scene_p02_p98", "scene_relative_db_p02_p98")
            or (source/".preprocessing_incomplete").exists()):
        raise ValueError(f"请先完成单 VV 功率域归一化：{source}")
    manifest_hash = sha256(manifest)
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fields = list(dict.fromkeys([*reader.fieldnames, "dem_path", "valid_pixels", "water_pixels",
                                     "background_pixels", "ignored_pixels", "dem_sum_m", "dem_sum_sq_m2"]))
    catalog = DemTileCatalog(tile_dir)
    target.mkdir(parents=True, exist_ok=True)
    with FileLock(str(target/".dem.lock"), timeout=0):
        marker = target/".preprocessing_incomplete"
        marker.touch()
        active = sum(int(r["active"]) == 1 for r in rows)
        done = 0
        seen = set()
        for row in rows:
            if int(row["active"]) == 1:
                split, scene = row["split"], row["scene_id"]
                if split not in ("train", "val") or not scene or Path(scene).name != scene or scene in seen:
                    raise ValueError(f"无效划分或重复景号：{split} / {scene}")
                seen.add(scene)
                vv = source/"images"/split/row["name"]
                label = source/"labels"/split/row["label_name"]
                done += 1
                start = time.monotonic()
                print(f"DEM {done}/{active} [{split}] {scene}", flush=True)
                updated, reused = _prepare_pair(vv, label, catalog, target, row)
                row.update(updated)
                print(f"  {'复用' if reused else '完成'}，有效 {row['valid_pixels']} 像元，{time.monotonic()-start:.1f} 秒", flush=True)
            for field in ("source_image", "source_label"):
                if row.get(field):
                    row[field] = relative_source(source_path(row[field], manifest), target/manifest.name)
        if not active or manifest_hash != sha256(manifest) or (source/".preprocessing_incomplete").exists() or record != json.loads(upstream.read_text(encoding="utf-8")):
            raise ValueError("归一化数据为空或在 DEM 准备期间改变，请待上游完成后重跑")
        pending = target/"split_manifest.csv.partial"
        with pending.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        pending.replace(target/manifest.name)
        metadata = dict(record, stage="vv_dem", bands=[1, 2], input_channels=["VV", "DEM"],
                        dem_source="Copernicus GLO-30 DSM", dem_unit="m", vertical_datum="EGM2008",
                        vv_manifest_sha256=manifest_hash)
        write_json(target/"preprocessing.json", metadata)
        dataset = SARSceneDataset(target, cache_dir, bands=(1, 2), window_size=window_size, halo=halo)
        dataset.close()
        marker.unlink()
        print(f"DEM 准备完成：{target/manifest.name}", flush=True)
    return target
