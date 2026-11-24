"""按归一化 VV 清单准备 train、val 共用的 Copernicus GLO-30 原生瓦片。"""
import csv
import json
import math
from collections import Counter
from pathlib import Path

from filelock import FileLock
import rasterio
from rasterio.warp import transform_bounds

from thres_segmentation.auxiliary_data import download_tile, tile_specs, validate_bounds, validate_raster


def _tiles_for_grid(bounds):
    """AWS COG 使用像元中心定位。东、西偏移随纬度带的原生采样间距变化。"""
    west, south, east, north = validate_bounds(bounds)
    half_y = .5 / 3600
    for lat in range(max(-90, math.floor(south - half_y)), min(90, math.ceil(north - half_y))):
        widths = ((50, 3600), (60, 2400), (70, 1800), (80, 1200), (85, 720), (90, 360))
        width = next(width for limit, width in widths if abs(lat + .5) < limit)
        half_x = .5 / width
        for longitude in range(math.floor(west + half_x), math.ceil(east + half_x)):
            lon = (longitude + 180) % 360 - 180
            yield from tile_specs("copernicus", (lon, lat, lon + 1, lat + 1))


def _tile_plan(source, method):
    receipt = source / "preprocessing.json"
    if not receipt.is_file():
        raise ValueError(f"找不到 VV 归一化完成记录 {receipt}，请先完成 normalize")
    record = json.loads(receipt.read_text(encoding="utf-8"))
    if (record.get("stage") != "normalized" or record.get("complete") is not True
            or (source / ".preprocessing_incomplete").exists()):
        raise ValueError(f"请先完成 VV 归一化：{source}")
    expected_domain = "standardized" if method == "zscore" else method
    if record.get("output_domain") != expected_domain:
        raise ValueError(f"VV 成果的归一化方法与配置不符：{record.get('output_domain')}，需要 {expected_domain}")
    specs, scenes = {}, {}
    counts = Counter()
    with (source / "split_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if int(row["active"]) != 1:
                continue
            split, name = row["split"], row["name"]
            if split not in ("train", "val"):
                raise ValueError(f"未知划分：{split}")
            image = source / "images" / split / name
            with rasterio.open(image) as src:
                if not src.crs or src.transform == rasterio.Affine.identity():
                    raise ValueError(f"VV 缺少地图网格：{image}")
                bounds = transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)
            for spec in _tiles_for_grid(bounds):
                filename = spec["filename"]
                specs[filename] = spec
                scenes.setdefault(filename, []).append(f"{split}/{row.get('scene_id') or name}")
            counts[split] += 1
    if not counts:
        raise ValueError(f"清单没有有效景：{source}")
    return specs, scenes, counts


def _usable(path):
    if not path.is_file():
        return False
    try:
        validate_raster(path)
    except (ValueError, rasterio.errors.RasterioError) as error:
        print(f"缓存瓦片需重新下载：{path}，{error}", flush=True)
        return False
    return True


def ensure_dem_tiles(vv_root, tile_dir, normalization_method, *, list_only=False):
    """去重并补齐所需瓦片；list_only 只打印缺片名，不创建目录或联网。"""
    source = Path(vv_root)
    if tile_dir is None or Path(tile_dir).is_file():
        raise ValueError("自动下载需将 DEM_TILE_DIR 设为原生 Copernicus 瓦片目录")
    target = Path(tile_dir)
    specs, scenes, counts = _tile_plan(source, normalization_method)
    missing = [name for name in sorted(specs) if not _usable(target / name)]
    print(f"VV：{source}\nGLO-30：{target}\n训练 {counts['train']} 景，验证 {counts['val']} 景；"
          f"需 {len(specs)} 个不同瓦片，现有可用 {len(specs)-len(missing)}，待补 {len(missing)}", flush=True)
    if list_only:
        for name in missing:
            print(name)
        return
    for i, name in enumerate(missing, 1):
        path = target / name
        target.mkdir(parents=True, exist_ok=True)
        with FileLock(str(path) + ".lock"):
            if _usable(path):
                continue
            print(f"[{i}/{len(missing)}] {name}", flush=True)
            try:
                download_tile(specs[name]["url"], path)
            except (OSError, ValueError, RuntimeError) as error:
                examples = ", ".join(scenes[name][:3])
                raise RuntimeError(f"GLO-30 未就绪：{name}，涉及 {len(scenes[name])} 景"
                                   f"（{examples}）；{error}") from error
    print("所需 GLO-30 原生瓦片已就绪。", flush=True)
