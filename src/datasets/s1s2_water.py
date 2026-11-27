"""准备 S1S2-Water v1.0.1 官方划分的单景 VV、米制 DEM 和监督掩膜。"""

from collections import Counter
from contextlib import ExitStack
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from filelock import FileLock
import numpy as np
import rasterio

from datasets.copernicus_dem import aligned_dem_tiles
from datasets.manifest_paths import relative_source
from datasets.scene_dem import _write_input_vrt
from supervised.load.sar_scene_dataset import SARSceneDataset
from utils.files import write_json
from utils.raster import windows


VV_DB_RANGE = (-35., 0.)
EXCLUDED_SCENES = {"31": "test", "59": "train"}
OFFICIAL_COUNTS = {"train": 33, "val": 16, "test": 16}
VV_DEM_PROTOCOL = dict(
    dataset="s1s2_water", version="1.0.1", stage="vv_dem", bands=[1, 2],
    input_channels=["VV", "DEM"], input_domain="s1_int16_db_x100",
    output_domain="fixed_s1_db_range_0_1", vv_db_range=list(VV_DB_RANGE),
    dem_source="Copernicus GLO-30 DSM", dem_unit="m", vertical_datum="EGM2008",
    label_validity="s1_valid=1 & known_s1_water_label & valid_vv_dem", split_protocol="official",
)


def official_scenes(root):
    """从发布目录读取按景划分，以完整的 65 景目录核对数据来源与异常编号。"""
    catalog = json.loads((root / "catalog.json").read_text(encoding="utf-8-sig"))
    records = {}
    for link in catalog["links"]:
        if link["rel"] != "item":
            continue
        meta = (root / link["href"]).resolve()
        scene = meta.parent.name
        if (not scene.isascii() or not scene.isdecimal() or scene in records
                or meta != root / scene / f"sentinel12_{scene}_meta.json"):
            raise ValueError(f"发布目录包含无效或重复的景号：{meta}")
        records[scene] = json.loads(meta.read_text(encoding="utf-8-sig"))["properties"]
    counts = Counter(props["split"] for props in records.values())
    if counts != OFFICIAL_COUNTS or any(records.get(s, {}).get("split") != v for s, v in EXCLUDED_SCENES.items()):
        raise ValueError(f"v1.0.1 官方划分应为 {OFFICIAL_COUNTS}，实际为 {dict(counts)}")

    selected = []
    for scene in sorted(records, key=int):
        if scene in EXCLUDED_SCENES:
            continue
        props = records[scene]
        if not props.get("s1_srcids") or any("_1SDV_" not in x for x in props["s1_srcids"]):
            raise ValueError(f"第 {scene} 景发布元数据未确认 S1 VV/VH")
        folder = root / scene
        stem = f"sentinel12_s1_{scene}"
        files = tuple(folder / f"{stem}_{suffix}.tif" for suffix in ("img", "msk", "valid"))
        files += (folder / f"sentinel12_copdem30_{scene}_elevation.tif",)
        for path in files:
            if not path.is_file():
                raise FileNotFoundError(f"第 {scene} 景缺少文件：{path}")
        selected.append((scene, props["split"], files))
    return selected


def _prepare_scene(scene, split, files, target):
    """分块解码 VV 和双线性对齐 DEM，以官方有效掩膜写出 0/1/255 标签和统计。"""
    image, label, valid, dem = files
    name = f"s1s2water_{scene}"
    raster_name = name + ".tif"
    vv_out = target / "vv" / split / raster_name
    dem_out = target / "dem" / split / raster_name
    label_out = target / "labels" / split / raster_name
    pair = target / "images" / split / (name + ".vrt")
    outputs = (vv_out, dem_out, label_out, pair)
    receipt = target / "receipts" / (name + ".json")

    def stamps(paths):
        return [[relative_source(p, receipt), p.stat().st_size, p.stat().st_mtime_ns] for p in paths]

    inputs = list(files)
    for path in files:
        inputs.extend(p for suffix in (".msk", ".aux.xml") if (p := Path(str(path) + suffix)).is_file())
    source_state = dict(protocol=VV_DEM_PROTOCOL, split=split, files=stamps(inputs))
    if receipt.is_file() and all(path.is_file() for path in outputs):
        old = json.loads(receipt.read_text(encoding="utf-8"))
        if old["inputs"] == source_state and old["outputs"] == stamps(outputs):
            return old["row"], True
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    manifest = target / "split_manifest.csv"
    water = land = 0
    total = squares = 0.
    with TemporaryDirectory(dir=target) as scratch, ExitStack() as stack:
        sar, gt, quality = (stack.enter_context(rasterio.open(p)) for p in (image, label, valid))
        geometry = (sar.crs, sar.transform, sar.shape)
        if sar.dtypes != ("int16", "int16") or not sar.crs or sar.scales != (1., 1.) or sar.offsets != (0., 0.):
            raise ValueError(f"第 {scene} 景不符合已核实的 S1 Int16 dB×100 双波段编码")
        for mask in (gt, quality):
            if mask.count != 1 or (mask.crs, mask.transform, mask.shape) != geometry:
                raise ValueError(f"掩膜与 SAR 网格不一致：{mask.name}")
        elevation, = aligned_dem_tiles([dem], sar, stack)
        profile = dict(driver="GTiff", crs=sar.crs, transform=sar.transform,
                       width=sar.width, height=sar.height, count=1, tiled=True,
                       compress="deflate", BIGTIFF="IF_SAFER")
        pending_vrt = Path(scratch) / "image.vrt"
        rasters = [Path(scratch) / f"{key}.tif" for key in ("vv", "dem", "label")]
        db_min, db_max = VV_DB_RANGE
        with rasterio.open(rasters[0], "w", **profile, dtype="float32", nodata=np.nan, predictor=3) as vv_dst, \
                rasterio.open(rasters[1], "w", **profile, dtype="float32", nodata=np.nan, predictor=3) as dem_dst, \
                rasterio.open(rasters[2], "w", **profile, dtype="uint8", nodata=255) as label_dst:
            vv_dst.update_tags(s1_storage="dB x 100", vv_db_min=db_min, vv_db_max=db_max)
            dem_dst.set_band_unit(1, "m")
            dem_dst.update_tags(product="Copernicus GLO-30 DSM", vertical_datum="EGM2008")
            for window in windows(sar, 512):
                vv = sar.read(1, window=window, masked=True, out_dtype="float32").filled(np.nan)
                heights = elevation.read(1, window=window, masked=True).filled(np.nan)
                observed = np.isfinite(vv) & np.isfinite(heights)
                truth = gt.read(1, window=window, masked=True)
                known = (observed & (quality.read(1, window=window, masked=True).filled(0) == 1)
                         & ~np.ma.getmaskarray(truth) & np.isin(truth.data, (0, 1)))
                values = np.where(known, truth.data, 255).astype("uint8")
                # 只解码已有的 dB 存储值。各景使用同一显示和训练区间，无须重取对数。
                scaled = np.clip((vv / 100. - db_min) / (db_max - db_min), 0., 1.)
                vv_dst.write(np.where(observed, scaled, np.nan), 1, window=window)
                dem_dst.write(np.where(observed, heights, np.nan), 1, window=window)
                label_dst.write(values, 1, window=window)
                water += int(np.count_nonzero(values == 1))
                land += int(np.count_nonzero(values == 0))
                if split == "train":
                    meters = heights[known].astype(np.float64)
                    total += meters.sum()
                    squares += np.square(meters).sum()
        if not water + land:
            raise ValueError(f"第 {scene} 景没有官方掩膜内的共同有效标签像元")
        _write_input_vrt(pending_vrt, pair, vv_out, dem_out, sar)
        for src, dst in zip((*rasters, pending_vrt), outputs):
            src.replace(dst)
        row = dict(name=pair.name, label_name=label_out.name, split=split, active=1,
                   scene_id=name, group_id=name, official_scene=scene,
                   source_image=relative_source(image, manifest), source_label=relative_source(label, manifest),
                   dem_path=relative_source(dem_out, manifest), valid_pixels=water + land,
                   water_pixels=water, background_pixels=land, ignored_pixels=sar.width*sar.height-water-land,
                   dem_sum_m=total if split == "train" else "", dem_sum_sq_m2=squares if split == "train" else "")
    write_json(receipt, dict(inputs=source_state, outputs=stamps(outputs), row=row))
    return row, False


def prepare_dataset(root, target, cache, window_size, halo):
    """准备所有保留场景并建立训练/验证窗口索引，测试清单单独发布。"""
    root, target = Path(root).resolve(), Path(target).resolve()
    selected = official_scenes(root)
    if target == root or any(target == paths[0].parent for _, _, paths in selected):
        raise ValueError("请为准备成果使用独立目录")
    target.mkdir(parents=True, exist_ok=True)
    with FileLock(str(target / ".prepare.lock"), timeout=0):
        marker = target / ".preprocessing_incomplete"
        marker.touch()
        rows = []
        for index, (scene, split, files) in enumerate(selected, 1):
            print(f"准备 {index}/{len(selected)} [{split}] S1S2-Water {scene}", flush=True)
            row, reused = _prepare_scene(scene, split, files, target)
            rows.append(row)
            print(f"  {'复用' if reused else '完成'}，有效标签 {row['valid_pixels']:,} 像元", flush=True)
        for filename, subset in (
            ("split_manifest.csv", [row for row in rows if row["split"] in ("train", "val")]),
            ("test_manifest.csv", [row for row in rows if row["split"] == "test"]),
        ):
            path = target / filename
            pending = path.with_suffix(".csv.partial")
            with pending.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(subset)
            pending.replace(path)
        counts = dict(Counter(row["split"] for row in rows))
        write_json(target / "preprocessing.json", dict(VV_DEM_PROTOCOL, complete=True, counts=counts,
                                                       excluded_scenes=EXCLUDED_SCENES))
        # require_prepared=True 只用于消费已发布的数据。此处按清单构建正式训练的窗口索引。
        dataset = SARSceneDataset(target, cache, bands=(1, 2), window_size=window_size, halo=halo)
        dataset.close()
        marker.unlink()
    print(f"S1S2-Water 准备完成 {counts}，清单在 {target}", flush=True)
