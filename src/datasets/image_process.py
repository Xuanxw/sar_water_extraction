"""整景SAR数据准备与检查：MODE选择首次导入、标签重建、清单检查、辐射统计或快视图。
不生成训练切片；滤波、归一化与阶段快视图均由本入口按MODE执行。
"""
from collections import Counter
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile
import ast
import csv
import hashlib
import json
import os
import re
import shutil
import time
import sys
import subprocess
import zlib
import xml.etree.ElementTree as ET
from rasterio.shutil import copy as raster_copy

import fiona
from filelock import FileLock, Timeout
import numpy as np
import rasterio
from rasterio.enums import MaskFlags
from rasterio.features import rasterize
from rasterio.warp import transform_geom
from rasterio.windows import Window, bounds
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import box, mapping, shape
from shapely.strtree import STRtree
from shapely.validation import explain_validity
import torch

# 支持按文件路径直接执行。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets import snap_preprocess
from datasets.speckle import SpeckleFilter
from datasets.manifest_paths import source_path, relative_source, manifest_fingerprint, write_relative_manifest
from supervised.load.sar_scene_dataset import SARSceneDataset

# 参数统一在YAML维护；以下只将配置绑定到现有函数使用的变量。
from configuration import load_yaml, config_argument
CONFIG_PATH = CONFIG = None


def configure(config_path):
    """显式加载本次YAML；导入模块时不读取文件或命令行参数。"""
    global CONFIG_PATH, CONFIG, MODE, SPLIT_CSV, PREPARE_BATCH, IMAGE_ROOT, LABEL_ROOT, OUTPUT_DIR
    global CACHE_DIR, BANDS, READ_WINDOW_SIZE, INSPECT_SAMPLES, MAX_SCENES, INPUT_NODATA_VALUE, VALID_MASK_DIR, ALLOW_ALL_VALID_SOURCE
    global CLASS_FIELD, CLASS_MAP, OUTSIDE_LABEL, SCENE_CLASS_RULES, ACCEPT_REVIEWED_GEOMETRY, SOURCE_OVERRIDES, AUDIT_OUTPUT_ROOT, BAND
    global RADIOMETRY_WINDOW_SIZE, GRID_SIZE, QUICKLOOK_IMAGE_ROOT, SPLITS, Z_DISPLAY_RANGE, QUICKLOOK_LOOKS, QUICKLOOK_MAX_SIDE, STRIPE_ROWS
    global PNG_COMPRESSION, LABEL_RULE, CALIBRATED_ROOT, FILTERED_ROOT, NORMALIZED_ROOT, NORMALIZATION_REFERENCE, WINDOW_CACHE_DIR, SNAP_GPT_PATH
    global SNAP_CACHE_DIR, CLEAN_SNAP_CACHE_ON_SUCCESS, SNAP_WORK_DIR, NORMALIZE_WORK_DIR, NORMALIZE_READ_SIZE, NORMALIZE_THREADS, NORMALIZE_COMPRESSION_LEVEL, ENABLE_CALIBRATION
    global ENABLE_FILTER, INPUT_DOMAIN, FILTER_WINDOW, ENL, FILTER_INPUT_STAGE, NORMALIZE_INPUT_STAGE, WINDOW_SIZE, HALO
    global NORMALIZATION_WINDOWS, SEED, CALIBRATION_QUANTITY, CALIBRATION_OUTPUT_DOMAIN, CALIBRATION_REFERENCE, CALIBRATION_PRODUCTS_DIR, CALIBRATION_COMMAND, QUICKLOOK_BACKSCATTER_DB_RANGE
    global SNAP_THREADS, SNAP_TILE_CACHE
    CONFIG_PATH = Path(config_path).expanduser().resolve()
    CONFIG = load_yaml(CONFIG_PATH)

    MODE = CONFIG["run"]["MODE"]
    SPLIT_CSV = CONFIG["paths"]["SPLIT_CSV"]
    PREPARE_BATCH = CONFIG["run"]["PREPARE_BATCH"]
    IMAGE_ROOT = CONFIG["paths"]["IMAGE_ROOT"]
    LABEL_ROOT = CONFIG["paths"]["LABEL_ROOT"]
    OUTPUT_DIR = CONFIG["paths"]["OUTPUT_DIR"]
    CACHE_DIR = CONFIG["paths"]["CACHE_DIR"]
    BANDS = tuple(CONFIG["input"]["BANDS"])
    READ_WINDOW_SIZE = CONFIG["input"]["READ_WINDOW_SIZE"]
    INSPECT_SAMPLES = CONFIG["input"]["INSPECT_SAMPLES"]
    MAX_SCENES = CONFIG["input"]["MAX_SCENES"]
    INPUT_NODATA_VALUE = CONFIG["input"]["INPUT_NODATA_VALUE"]
    VALID_MASK_DIR = CONFIG["paths"]["VALID_MASK_DIR"]
    ALLOW_ALL_VALID_SOURCE = CONFIG["input"]["ALLOW_ALL_VALID_SOURCE"]
    CLASS_FIELD = CONFIG["labels"]["CLASS_FIELD"]
    CLASS_MAP = CONFIG["labels"]["CLASS_MAP"]
    OUTSIDE_LABEL = CONFIG["labels"]["OUTSIDE_LABEL"]
    SCENE_CLASS_RULES = CONFIG["labels"]["SCENE_CLASS_RULES"]
    ACCEPT_REVIEWED_GEOMETRY = CONFIG["labels"]["ACCEPT_REVIEWED_GEOMETRY"]
    SOURCE_OVERRIDES = CONFIG["paths"]["SOURCE_OVERRIDES"]
    AUDIT_OUTPUT_ROOT = CONFIG["paths"]["AUDIT_OUTPUT_ROOT"]
    BAND = CONFIG["inspection"]["BAND"]
    RADIOMETRY_WINDOW_SIZE = CONFIG["inspection"]["RADIOMETRY_WINDOW_SIZE"]
    GRID_SIZE = CONFIG["inspection"]["GRID_SIZE"]
    QUICKLOOK_IMAGE_ROOT = CONFIG["paths"]["QUICKLOOK_IMAGE_ROOT"]
    SPLITS = tuple(CONFIG["inspection"]["SPLITS"])
    Z_DISPLAY_RANGE = tuple(CONFIG["inspection"]["Z_DISPLAY_RANGE"])
    QUICKLOOK_LOOKS = CONFIG["inspection"]["QUICKLOOK_LOOKS"]
    QUICKLOOK_MAX_SIDE = CONFIG["inspection"]["QUICKLOOK_MAX_SIDE"]
    STRIPE_ROWS = CONFIG["inspection"]["STRIPE_ROWS"]
    PNG_COMPRESSION = CONFIG["inspection"]["PNG_COMPRESSION"]
    LABEL_RULE = CONFIG["labels"]["LABEL_RULE"]
    CALIBRATED_ROOT = CONFIG["paths"]["CALIBRATED_ROOT"]
    FILTERED_ROOT = CONFIG["paths"]["FILTERED_ROOT"]
    NORMALIZED_ROOT = CONFIG["paths"]["NORMALIZED_ROOT"]
    NORMALIZATION_REFERENCE = CONFIG["paths"].get("NORMALIZATION_REFERENCE")
    WINDOW_CACHE_DIR = CONFIG["paths"]["WINDOW_CACHE_DIR"]
    SNAP_GPT_PATH = CONFIG["paths"]["SNAP_GPT_PATH"]
    SNAP_CACHE_DIR = CONFIG["paths"]["SNAP_CACHE_DIR"]
    CLEAN_SNAP_CACHE_ON_SUCCESS = CONFIG["snap"]["CLEAN_SNAP_CACHE_ON_SUCCESS"]
    SNAP_WORK_DIR = CONFIG["paths"]["SNAP_WORK_DIR"]
    NORMALIZE_WORK_DIR = CONFIG["paths"]["NORMALIZE_WORK_DIR"]
    NORMALIZE_READ_SIZE = CONFIG["processing"]["NORMALIZE_READ_SIZE"]
    NORMALIZE_THREADS = CONFIG["processing"]["NORMALIZE_THREADS"]
    NORMALIZE_COMPRESSION_LEVEL = CONFIG["processing"]["NORMALIZE_COMPRESSION_LEVEL"]
    ENABLE_CALIBRATION = CONFIG["processing"]["ENABLE_CALIBRATION"]
    ENABLE_FILTER = CONFIG["processing"]["ENABLE_FILTER"]
    INPUT_DOMAIN = CONFIG["processing"]["INPUT_DOMAIN"]
    FILTER_WINDOW = CONFIG["processing"]["FILTER_WINDOW"]
    ENL = CONFIG["processing"]["ENL"]
    FILTER_INPUT_STAGE = CONFIG["processing"]["FILTER_INPUT_STAGE"]
    NORMALIZE_INPUT_STAGE = CONFIG["processing"]["NORMALIZE_INPUT_STAGE"]
    WINDOW_SIZE = CONFIG["processing"]["WINDOW_SIZE"]
    HALO = CONFIG["processing"]["HALO"]
    NORMALIZATION_WINDOWS = CONFIG["processing"]["NORMALIZATION_WINDOWS"]
    SEED = CONFIG["processing"]["SEED"]
    CALIBRATION_QUANTITY = CONFIG["processing"]["CALIBRATION_QUANTITY"]
    CALIBRATION_OUTPUT_DOMAIN = CONFIG["processing"]["CALIBRATION_OUTPUT_DOMAIN"]
    CALIBRATION_REFERENCE = CONFIG["processing"]["CALIBRATION_REFERENCE"]
    CALIBRATION_PRODUCTS_DIR = CONFIG["paths"]["CALIBRATION_PRODUCTS_DIR"]
    CALIBRATION_COMMAND = CONFIG["processing"]["CALIBRATION_COMMAND"]
    QUICKLOOK_BACKSCATTER_DB_RANGE = tuple(CONFIG["inspection"]["QUICKLOOK_BACKSCATTER_DB_RANGE"])
    SNAP_THREADS = CONFIG["snap"]["SNAP_THREADS"]
    SNAP_TILE_CACHE = CONFIG["snap"]["SNAP_TILE_CACHE"]


SCENE_PATTERN = re.compile(r"(BC\d+)[_-]([A-Z]+)[_-](ORG)[_-](?:2S)?(VV|VH|HH|HV)[_-](\d{8}T\d{6})[_-](\d+)", re.I)
MANIFEST_FIELDS = ("name", "label_name", "split", "active", "group_id", "scene_id", "scene",
                   "source_image", "source_label", "row_off", "col_off", "source_height", "source_width",
                   "valid_pixels", "water_pixels", "background_pixels", "ignored_pixels", "pixel_x", "pixel_y")


def scene_id(name):
    """统一2SVV包装名与VV景号；非BC数据使用不带扩展名的文件名。"""
    match = SCENE_PATTERN.search(str(name))
    return "-".join(match.groups()).upper() if match else Path(name).stem.upper()


def index_sources(image_root, label_root):
    """只索引路径，不将整库ZIP解压；同景的点文件在选标签时排除。"""
    images, labels = {}, {}
    for root, suffixes, target in ((image_root, {".tif", ".tiff", ".zip"}, images),
                                    (label_root, {".shp"}, labels)):
        for path in Path(root).rglob("*"):
            if path.is_file() and path.suffix.lower() in suffixes:
                target.setdefault(scene_id(path.name), []).append(path)
    return images, labels


def select_sources(identifier, images, labels, overrides):
    """按景号配对；核验ZIP与解压副本，真正不同的版本由SOURCE_OVERRIDES指定。"""
    override = overrides.get(identifier, {})
    image_paths = [Path(override["image"])] if "image" in override else images.get(identifier, [])
    label_paths = [Path(override["label"])] if "label" in override else labels.get(identifier, [])
    # 原库可能同时保留ZIP和解压TIFF。只合并经大小和CRC32核验的副本，
    # 继续读取ZIP保持原导入口径，避免解压目录的aux.xml改变读取结果。
    if len(image_paths) == 2:
        archives = [p for p in image_paths if p.suffix.lower() == ".zip"]
        tiffs = [p for p in image_paths if p.suffix.lower() in (".tif", ".tiff")]
        if len(archives) == 1 and len(tiffs) == 1:
            with ZipFile(archives[0]) as archive:
                members = [m for m in archive.infolist()
                           if Path(m.filename).suffix.lower() in (".tif", ".tiff")]
                if len(members) == 1 and members[0].file_size == tiffs[0].stat().st_size:
                    checksum = 0
                    with tiffs[0].open("rb") as stream:
                        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                            checksum = zlib.crc32(block, checksum)
                    if checksum == members[0].CRC:
                        print(f"  ZIP与解压TIFF大小/CRC32一致，使用原压缩包：{archives[0]}", flush=True)
                        image_paths = archives
    polygons = []
    for path in label_paths:
        with fiona.open(path) as source:
            if "Polygon" in source.schema["geometry"]:
                polygons.append(path)
    if len(image_paths) != 1 or len(polygons) != 1:
        raise ValueError(f"{identifier}：需要唯一影像和面标签，实际{len(image_paths)}/{len(polygons)}；请核对或配置SOURCE_OVERRIDES\n影像候选：{image_paths}\n面标签候选：{polygons}")
    return image_paths[0], polygons[0]


def image_path(path, cache=None):
    """检查时通过GDAL直接读ZIP头；准备时展开一景，避免反复解压压缩流。"""
    path = Path(path)
    if path.suffix.lower() in (".tif", ".tiff"):
        return str(path)
    if path.suffix.lower() != ".zip":
        raise ValueError("影像支持.tif/.tiff GeoTIFF或ZIP；裸binary需要先提供头文件并转换")
    with ZipFile(path) as archive:
        members = [m for m in archive.infolist() if Path(m.filename).suffix.lower() in (".tif", ".tiff")]
        if len(members) != 1:
            raise ValueError(f"ZIP内需有且只有一个GeoTIFF：{path}")
        member = members[0]
        if cache is None:
            return "/vsizip/" + path.as_posix() + "/" + member.filename
        # 固定临时文件名，不使用ZIP成员路径创建目录。
        target = Path(cache) / "image.tif"
        with archive.open(member) as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
        return str(target)


def read_polygons(path, target_crs, field, class_map):
    """读取标签面并转到影像CRS；仅清除退化内环，其余无效面仍按原规则处理。"""
    if not class_map or any(v not in (0, 1) for v in class_map.values()):
        raise ValueError("先确认标签类别，再配置CLASS_MAP或SCENE_CLASS_RULES；映射结果只能为0/1")
    geometries, classes, skipped = [], [], Counter()
    with fiona.open(path) as source:
        if not source.crs:
            raise ValueError(f"标签缺少坐标系：{path}")
        if field is not None and field not in source.schema["properties"]:
            raise ValueError(f"标签没有字段{field}，实际为{list(source.schema['properties'])}：{path}")
        same_crs = rasterio.crs.CRS.from_user_input(source.crs) == rasterio.crs.CRS.from_user_input(target_crs)
        for feature in source:
            value = feature["properties"][field] if field is not None else None
            if value not in class_map:
                skipped[str(value)] += 1
                continue
            if not feature["geometry"]:
                raise ValueError(f"标签要素{feature.id}为空：{path}")
            raw_geometry = feature["geometry"]
            removed_holes = 0
            if raw_geometry["type"] in ("Polygon", "MultiPolygon"):
                polygons = ([raw_geometry["coordinates"]] if raw_geometry["type"] == "Polygon"
                            else raw_geometry["coordinates"])
                cleaned = []
                for rings in polygons:
                    kept = list(rings[:1])  # 外环原样保留，不能为绕过报错而丢掉整个水体面。
                    for ring in rings[1:]:
                        # 不足三个不同平面坐标的内环没有面积，无法构成水体中的孔洞。
                        # 仅在内存清除；不改源SHP，也不按面积正负删除自相交环。
                        if len({tuple(point[:2]) for point in ring}) < 3:
                            removed_holes += 1
                        else:
                            kept.append(ring)
                    cleaned.append(kept)
                raw_geometry = dict(type=raw_geometry["type"], coordinates=(
                    cleaned[0] if raw_geometry["type"] == "Polygon" else cleaned))
            try:
                geometry = shape(raw_geometry)
            except ValueError as error:
                raise ValueError(f"标签要素{feature.id}无法构造面：{error}；{path}") from error
            if geometry.is_empty or (not geometry.is_valid and not ACCEPT_REVIEWED_GEOMETRY):
                raise ValueError(f"标签要素{feature.id}几何待修复：{explain_validity(geometry)}；{path}")
            if not same_crs:
                geometry = shape(transform_geom(source.crs_wkt, target_crs, mapping(geometry)))
            if geometry.geom_type not in ("Polygon", "MultiPolygon") or geometry.is_empty or (not geometry.is_valid and not ACCEPT_REVIEWED_GEOMETRY):
                raise ValueError(f"标签要素{feature.id}转换后不是有效面：{path}")
            if not geometry.is_valid:
                print(f"  已人工复核，原样栅格化：{path.name} 要素{feature.id}；{explain_validity(geometry)}", flush=True)
            if removed_holes:
                print(f"  读取时移除{removed_holes}个退化内环：{path.name} 要素{feature.id}；保留外环，原SHP未修改", flush=True)
            geometries.append(geometry)
            classes.append(class_map[value])
    if skipped:
        print(f"  未映射类别保持未知：{sum(skipped.values())}个面、{len(skipped)}种值；前5项：{dict(skipped.most_common(5))}", flush=True)
    if not geometries:
        raise ValueError(f"没有已确认类别的有效面：{path}")
    return geometries, np.asarray(classes, dtype=np.uint8)


def prepare_scene(row, image_file, label_file, output, cache_root, *, bands, read_window_size,
                  field, class_map, input_nodata=None, valid_mask=None, allow_all_valid=False, outside_label=255):
    """按窗口写入同一对整景GeoTIFF；保留完整范围，不缩放、不补齐影像尺寸。"""
    if outside_label not in (0, 255):
        raise ValueError("outside_label只支持已确认的非水体0或未知255")
    cache_root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="scene_", dir=cache_root) as cache, ExitStack() as stack:
        source = stack.enter_context(rasterio.open(image_path(image_file, cache)))
        if not source.crs or source.transform == rasterio.Affine.identity():
            raise ValueError(f"影像需要地理参考，不能仅凭.tiff后缀判断：{image_file}")
        if source.transform.b or source.transform.d:
            raise ValueError("当前整景入口处理无旋转的地理编码网格；旋转网格请先重投影")
        if not bands or min(bands) < 1 or max(bands) > source.count:
            raise ValueError(f"输入波段{bands}超出实际{source.count}个波段")
        if any(np.issubdtype(np.dtype(source.dtypes[b-1]), np.complexfloating) for b in bands):
            raise ValueError("复数SLC不能直接转float32，需先形成已确认物理含义的实数SAR影像")
        mask_source = stack.enter_context(rasterio.open(valid_mask)) if valid_mask else None
        if mask_source and (mask_source.shape != source.shape or mask_source.crs != source.crs or
                            mask_source.transform != source.transform or mask_source.count != 1):
            raise ValueError("有效覆盖掩膜必须与影像同尺寸、同CRS、同地理变换，且为单波段")
        source_all_valid = all(MaskFlags.all_valid in source.mask_flag_enums[b-1] for b in bands)
        if source_all_valid and input_nodata is None and mask_source is None and not allow_all_valid:
            raise ValueError("影像没有声明无效域；请提供VALID_MASK_DIR/已确认的INPUT_NODATA_VALUE，或确认整景有效后设置ALLOW_ALL_VALID_SOURCE=True")
        geometries, classes = read_polygons(label_file, source.crs, field, class_map)
        tree = STRtree(geometries)
        name = f"{row['scene_id']}.tif"
        image_target = output/'images'/row['split']/name
        label_target = output/'labels'/row['split']/name
        # 临时文件完整关闭后再改名；中断时不会把半幅影像登记为已完成。
        image_pending = image_target.with_suffix('.tif.partial')
        label_pending = label_target.with_suffix('.tif.partial')
        common = dict(driver='GTiff', height=source.height, width=source.width,
                      crs=source.crs, transform=source.transform, compress='deflate',
                      tiled=True, blockxsize=256, blockysize=256, BIGTIFF='IF_SAFER')
        totals = Counter()
        with rasterio.open(image_pending, 'w', **common, count=len(bands), dtype='float32', nodata=np.nan) as image_dst, \
             rasterio.open(label_pending, 'w', **common, count=1, dtype='uint8', nodata=255) as label_dst:
            label_dst.write_colormap(1, {0: (255,255,255,255), 1: (0,0,0,255), 255: (128,128,128,0)})
            label_dst.update_tags(class_0='non_water', class_1='water', class_255='invalid_or_unknown')
            for channel, band in enumerate(bands, 1):
                image_dst.set_band_description(channel, source.descriptions[band-1] or f"source_band_{band}")
            for top in range(0, source.height, read_window_size):
                for left in range(0, source.width, read_window_size):
                    height = min(read_window_size, source.height-top)
                    width = min(read_window_size, source.width-left)
                    window = Window(left, top, width, height)
                    data = source.read(bands, window=window, masked=True, out_dtype='float32')
                    valid = ~np.ma.getmaskarray(data).any(axis=0) & np.isfinite(data.data).all(axis=0)
                    if input_nodata is not None:
                        valid &= (data.data != input_nodata).all(axis=0)
                    if mask_source:
                        coverage = mask_source.read(1, window=window, masked=True)
                        valid &= ~np.ma.getmaskarray(coverage) & (coverage.data == 1)
                    selected = tree.query(box(*bounds(window, source.transform)))
                    masks = []
                    for category in (0, 1):
                        shapes = [(geometries[int(i)], 1) for i in selected if classes[i] == category]
                        masks.append(rasterize(shapes, out_shape=(height, width),
                                     transform=source.window_transform(window), fill=0,
                                     all_touched=False, dtype='uint8').astype(bool)
                                     if shapes else np.zeros((height, width), dtype=bool))
                    label = np.full((height, width), 255, dtype=np.uint8)
                    label[valid & ~masks[0] & ~masks[1]] = outside_label
                    label[valid & masks[0] & ~masks[1]] = 0
                    label[valid & masks[1] & ~masks[0]] = 1
                    # 面外按明确配置处理；类别冲突和无效像素仍为255，SAR保留完整范围。
                    image_dst.write(np.where(valid[None], data.data, np.nan), window=window)
                    label_dst.write(label, 1, window=window)
                    totals['valid_pixels'] += int(valid.sum())
                    totals['water_pixels'] += int((label == 1).sum())
                    totals['background_pixels'] += int((label == 0).sum())
                    totals['ignored_pixels'] += int((label == 255).sum())
                print(f"  整景写入：{min(top+read_window_size, source.height)}/{source.height}行", flush=True)
        image_pending.replace(image_target)
        label_pending.replace(label_target)
        return [dict(name=name, label_name=name, split=row['split'],
                     active=int(totals['water_pixels'] + totals['background_pixels'] > 0),
                     group_id=row['group_id'], scene_id=row['scene_id'], scene=row['scene'],
                     source_image=relative_source(image_file, output/'split_manifest.csv'),
                     source_label=relative_source(label_file, output/'split_manifest.csv'), row_off=0, col_off=0,
                     source_height=source.height, source_width=source.width, **totals,
                     pixel_x=source.res[0], pixel_y=source.res[1])]


def read_split_rows(path):
    """CSV顶部可放多行说明；从“影像编号”表头开始读取场景记录。"""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        for fields in reader:
            if fields and fields[0].strip() == "影像编号":
                break
        else:
            raise ValueError(f"分类CSV缺少影像编号表头：{path}")
        return [dict(zip(fields, row)) for row in reader if row and row[0].strip()]


def prepare_dataset(mode=None, reuse_complete=False):
    """检查或准备；只在全部选定场景成功后发布训练清单，失败时保留现场供检查。"""
    mode = MODE if mode is None else mode
    if mode not in ("split", "inspect", "prepare") or READ_WINDOW_SIZE < 1:
        raise ValueError("MODE选labels/split/inspect/prepare，READ_WINDOW_SIZE应大于0")
    rows = read_split_rows(SPLIT_CSV)
    print(f"运行模式：{mode}；输出目录：{OUTPUT_DIR}", flush=True)
    print(f"景级状态：{dict(Counter(r['是否采用'] for r in rows))}")
    selected, group_splits, names = [], {}, set()
    for row in rows:
        identifier = scene_id(row['影像编号'])
        split = row['数据划分']
        match = re.search(r"\bG_[0-9a-f]+\b", row['场景'], re.I)
        group = row.get('group_id') or (match.group() if match else '')
        if not group or split not in ('train','val') or identifier in names:
            raise ValueError(f"景号重复、分组缺失或划分错误：{identifier}")
        names.add(identifier)
        if group in group_splits and group_splits[group] != split:
            raise ValueError(f"地理组跨train/val：{group}")
        group_splits[group] = split
        # 已确认表示按用户要求纳入，不等于地物类型或标签几何已验收。
        allowed = ('是', '已确认', '待确认') if mode == 'inspect' else ('是', '已确认')
        in_batch = PREPARE_BATCH is None or row.get('准备批次') == PREPARE_BATCH
        if in_batch and row['是否采用'] in allowed:
            selected.append(dict(scene_id=identifier,split=split,group_id=group,scene=row['场景']))
    if not selected:
        message = "没有可处理场景。prepare只接受是否采用=是或已确认；当前待确认记录不会自动启用。"
        if mode == 'prepare':
            raise ValueError(message)
        print(message)
        return
    if mode == 'split':
        print(f"采用场景的划分：{dict(Counter(row['split'] for row in selected))}")
        print(f"采用场景地理组数：{len({row['group_id'] for row in selected})}")
        print('分类沿用清单中的地理组、行政区、时间与传感器；不进行地物分类、几何分析或切块。')
        return
    selected = selected[:INSPECT_SAMPLES] if mode == 'inspect' else selected[:MAX_SCENES]
    images, labels = index_sources(IMAGE_ROOT, LABEL_ROOT)
    if mode == 'inspect':
        for row in selected:
            image, label = select_sources(row['scene_id'],images,labels,SOURCE_OVERRIDES)
            with rasterio.open(image_path(image)) as source, fiona.open(label) as vector:
                print(f"\n{row['scene_id']} [{row['split']}]\n  影像：{image}\n  标签：{label}")
                print(f"  {source.driver}；尺寸={source.width}×{source.height}；通道={source.count}；类型={source.dtypes}")
                print(f"  CRS={source.crs}；网格间距={source.res}；nodata={source.nodata}；掩膜={source.mask_flag_enums}")
                print(f"  scale/offset={source.scales}/{source.offsets}（准备时保留原始存储值，不应用换算）")
                print(f"  标签CRS={vector.crs}；字段={vector.schema['properties']}；面数={len(vector)}")
                rule = SCENE_CLASS_RULES.get(row['scene_id'], dict(field=CLASS_FIELD, map=CLASS_MAP))
                field = rule['field']
                if field is not None and field in vector.schema['properties']:
                    counts = Counter(str(feature['properties'][field]) for feature in vector)
                    print(f"  字段{field}的值/面数：{dict(counts)}（数值不等于已确认类别含义）")
                print("  未确认标签映射和有效覆盖前，不进入prepare；不自动把零值或面外区域当背景。")
        return
    # 在展开大影像之前确认每景的类别配置。
    for row in selected:
        rule = SCENE_CLASS_RULES.get(row['scene_id'], dict(field=CLASS_FIELD, map=CLASS_MAP))
        if not rule['map'] or any(value not in (0, 1) for value in rule['map'].values()):
            raise ValueError(f"{row['scene_id']}尚未配置标签映射。MODE=prepare已启用，但CLASS_MAP仍为空；需明确面属性的水体/非水体含义，不能用采用状态代替像素标签。")
    # 读取已完成景的记录。每景完整写入后才登记；不与旧切片清单混用。
    partial = OUTPUT_DIR/'split_manifest.csv.partial'
    complete_manifest = OUTPUT_DIR/'split_manifest.csv'
    existing = []
    if OUTPUT_DIR.exists():
        if (complete_manifest.exists() and not reuse_complete) or (not complete_manifest.exists() and not partial.exists()):
            raise ValueError('prepare用于首次导入；此目录已完成或缺少续跑清单。仅重建已有标签请设置MODE="labels"')
        with (complete_manifest if complete_manifest.exists() else partial).open(encoding='utf-8-sig', newline='') as stream:
            existing = list(csv.DictReader(stream))
        completed_ids = {tile['scene_id'] for tile in existing}
        current_selection = {r['scene_id']: r for r in selected}
        for tile in existing:
            current = current_selection.get(tile['scene_id'])
            if current is None or tile['split'] != current['split'] or any(
                    tile.get(k) and tile[k] != current[k] for k in ('group_id', 'scene')):
                raise ValueError(f"已完成场景的采用状态或划分已改变：{tile['scene_id']}，请使用新输出目录")
        # 允许排除尚未完成的坏景；已完成成果及其划分不能随总表修改而失配。
        # 备注和其他批次更新不参与比较；不需要手动覆盖旧scene_split.csv来绕过检查。
        old_rows = read_split_rows(OUTPUT_DIR/'scene_split.csv')
        keys = ('数据划分', '场景', '准备批次', 'group_id')
        old_selection = {scene_id(r['影像编号']): r for r in old_rows
                         if PREPARE_BATCH is None or r.get('准备批次') == PREPARE_BATCH}
        new_selection = {scene_id(r['影像编号']): r for r in rows
                         if PREPARE_BATCH is None or r.get('准备批次') == PREPARE_BATCH}
        excluded = []
        for identifier in sorted(old_selection.keys() | new_selection.keys()):
            old, new = old_selection.get(identifier), new_selection.get(identifier)
            same_fields = old is not None and new is not None and all(old.get(k, '') == new.get(k, '') for k in keys)
            if same_fields and old['是否采用'] == new['是否采用']:
                continue
            if (identifier not in completed_ids and old is not None and old['是否采用'] in ('是', '已确认')
                    and ((new is None and identifier not in names) or (same_fields and new['是否采用'] == '否'))):
                excluded.append(identifier)
                continue
            raise ValueError(f"本批划分清单已改变：{identifier}；仅允许排除未完成场景，其他变更请使用新输出目录")
        if excluded:
            print(f"续跑清单更新：排除{len(excluded)}景尚未完成的数据：{', '.join(excluded)}", flush=True)
        if any(tile['name'] != tile['scene_id'] + '.tif' for tile in existing):
            raise ValueError('这是旧切片目录，请使用新的整景输出目录')
        for tile in existing:
            for kind, key in (('images', 'name'), ('labels', 'label_name')):
                if not (OUTPUT_DIR/kind/tile['split']/tile[key]).is_file():
                    raise ValueError(f"续跑记录缺少文件：{tile[key]}")
    completed = {tile['scene_id'] for tile in existing}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for kind in ('images','labels'):
        for split in ('train','val'):
            (OUTPUT_DIR/kind/split).mkdir(parents=True, exist_ok=True)
    # 保存本次实际配置，避免之后修改脚本后无法追溯标签映射及有效域规则。
    settings = dict(layout="full_scene", bands=BANDS, read_window_size=READ_WINDOW_SIZE, class_field=CLASS_FIELD, class_map=CLASS_MAP,
                    scene_class_rules=SCENE_CLASS_RULES, input_nodata=INPUT_NODATA_VALUE,
                    valid_mask_dir=VALID_MASK_DIR, allow_all_valid_source=ALLOW_ALL_VALID_SOURCE,
                    source_overrides=SOURCE_OVERRIDES, values='raw stored values; no scale/offset or dB',
                    labels='0 background / 1 water / 255 unknown; conflicting classes ignored',
                    outside_label=OUTSIDE_LABEL)
    if ACCEPT_REVIEWED_GEOMETRY:
        settings["geometry_policy"] = "user_reviewed_original_polygons_no_repair"
    settings_path = OUTPUT_DIR/'preparation_settings.txt'
    if existing:
        previous = dict(line.split(' = ', 1) for line in settings_path.read_text(encoding='utf-8').splitlines())
        for key, value in settings.items():
            old = ast.literal_eval(previous[key]) if key == 'scene_class_rules' else None
            if key == 'scene_class_rules':
                same = all(old.get(identifier) == value.get(identifier) for identifier in completed)
            else:
                same = previous.get(key) == repr(value)
            if not same:
                raise ValueError(f"已完成场景的配置{key}已改变，请使用新输出目录")
    if complete_manifest.exists():
        if completed != {r['scene_id'] for r in selected}:
            raise ValueError('已发布manifest与本批选择不一致，不能直接复用')
        print(f'prepare已完成：复用{len(completed)}景整景影像和标签。', flush=True)
        return
    shutil.copy2(SPLIT_CSV, OUTPUT_DIR/'scene_split.csv')
    settings_path.write_text(
        '\n'.join(f"{key} = {value!r}" for key,value in settings.items()), encoding='utf-8')
    totals = Counter()
    for tile in existing:
        totals[tile['split']] += int(tile['active'])
        totals['water_pixels'] += int(tile['water_pixels'])
        totals['background_pixels'] += int(tile['background_pixels'])
    print(f"续跑：保留{len(completed)}景、{len(existing)}对整景影像/标签", flush=True)
    skipped = []
    # 完成前仅写.partial；若中断，不要将这个不完整目录交给训练器。
    partial = OUTPUT_DIR/'split_manifest.csv.partial'
    with partial.open('a' if existing else 'w',encoding='utf-8-sig',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=MANIFEST_FIELDS)
        if not existing:
            writer.writeheader()
        for scene_number, row in enumerate(selected, 1):
            if row['scene_id'] in completed:
                continue
            print(f"开始场景 {scene_number}/{len(selected)}：{row['scene_id']} [{row['split']}]", flush=True)
            image, label = select_sources(row['scene_id'],images,labels,SOURCE_OVERRIDES)
            rule = SCENE_CLASS_RULES.get(row['scene_id'],dict(field=CLASS_FIELD,map=CLASS_MAP))
            # 在解压影像前检查类别：字段存在也不一定存放0/1标签。
            reason = None
            with fiona.open(label) as vector:
                field = rule['field']
                if field is not None and field not in vector.schema['properties']:
                    reason = f"缺少字段{field}；实际字段：{list(vector.schema['properties'])}"
                else:
                    values = Counter(feature['properties'][field] if field is not None else None
                                     for feature in vector)
                    matched = sum(count for value, count in values.items() if value in rule['map'])
                    if not matched:
                        reason = (f"字段{field}没有可映射类别；共{sum(values.values())}个面；"
                                  f"当前映射{rule['map']}；前5项：{values.most_common(5)}")
            if reason:
                skipped.append(dict(scene_id=row['scene_id'], split=row['split'], label=str(label), reason=reason))
                with (OUTPUT_DIR/'skipped_scenes.csv').open('w', encoding='utf-8-sig', newline='') as report:
                    report_writer = csv.DictWriter(report, fieldnames=('scene_id', 'split', 'label', 'reason'))
                    report_writer.writeheader()
                    report_writer.writerows(skipped)
                print(f"暂跳过：{reason}", flush=True)
                continue
            valid_mask = None
            if VALID_MASK_DIR:
                candidates = [Path(VALID_MASK_DIR)/(row['scene_id']+suffix) for suffix in ('.tif','.tiff')]
                found = [path for path in candidates if path.is_file()]
                if len(found) != 1:
                    raise ValueError(f"缺少唯一的有效覆盖掩膜：{row['scene_id']}")
                valid_mask = found[0]
            records = prepare_scene(row,image,label,OUTPUT_DIR,CACHE_DIR,bands=BANDS,read_window_size=READ_WINDOW_SIZE,
                                  field=rule['field'],class_map=rule['map'],input_nodata=INPUT_NODATA_VALUE,
                                  valid_mask=valid_mask,allow_all_valid=ALLOW_ALL_VALID_SOURCE,outside_label=OUTSIDE_LABEL)
            writer.writerows(records)
            stream.flush()
            totals[row['split']] += sum(record['active'] for record in records)
            for tile in records:
                totals['water_pixels'] += tile['water_pixels']
                totals['background_pixels'] += tile['background_pixels']
            print(f"{row['scene_id']} [{row['split']}]：输出整景影像及同尺寸标签",flush=True)
    if skipped:
        print(f"本次暂跳过{len(skipped)}景：见skipped_scenes.csv；其余结果已保留。补充SCENE_CLASS_RULES后可续跑。", flush=True)
        print("清单仍为.partial，尚未发布为完整训练数据集。", flush=True)
        return
    if (OUTPUT_DIR/'skipped_scenes.csv').exists():
        (OUTPUT_DIR/'skipped_scenes.csv').write_text('scene_id,split,label,reason\n', encoding='utf-8-sig')
    if not totals['train'] or not totals['val'] or not totals['water_pixels'] or not totals['background_pixels']:
        raise ValueError(f"当前准备结果不完整，尚未发布训练清单（需train/val、已确认水体/非水体）：{dict(totals)}")
    partial.replace(OUTPUT_DIR/'split_manifest.csv')
    print(f"准备完成：{OUTPUT_DIR}\n{dict(totals)}\n整景数据准备完成；训练需动态窗口读取，不能直接用现有整图加载器批量读取大景。单VV需关闭土地覆盖第5波段修正。")


def rebuild_scene(row, data_dir, read_window_size=1024):
    """沿SAR网格生成水体1/背景0；保留原影像路径。"""
    image_path = data_dir / 'images' / row['split'] / row['name']
    label_path = data_dir / 'labels' / row['split'] / row['label_name']
    label_path.parent.mkdir(parents=True, exist_ok=True)
    pending = label_path.with_suffix('.tif.partial')
    with rasterio.open(image_path) as src:
        # class=0和class=1的面均烧为1，其他字段值也不决定水陆。
        geometries, _ = read_polygons(source_path(row['source_label'], data_dir/'split_manifest.csv'), src.crs, None, {None: 1})
        tree = STRtree(geometries)
        total_pixels = src.width * src.height
        # 旧准备清单已经逐像素统计过有效性。整景全有效时无需再读数百MB的SAR。
        all_valid = int(row['valid_pixels']) == total_pixels
        water = background = ignored = 0
        profile = dict(driver='GTiff', width=src.width, height=src.height, count=1,
                       dtype='uint8', crs=src.crs, transform=src.transform, nodata=255,
                       tiled=True, blockxsize=512, blockysize=512, compress='deflate',
                       zlevel=1, num_threads='2', BIGTIFF='IF_SAFER')
        with rasterio.open(pending, 'w', **profile) as dst:
            # 颜色只负责显示；训练仍严格使用0=非水体、1=水体、255=无效。
            dst.write_colormap(1, {0: (255, 255, 255, 255), 1: (0, 0, 0, 255),
                                  255: (128, 128, 128, 0)})
            dst.update_tags(label_version='water_1_background_0', label_rule=LABEL_RULE,
                            class_0='non_water', class_1='water', class_255='invalid')
            for top in range(0, src.height, read_window_size):
                h = min(read_window_size, src.height - top)
                for left in range(0, src.width, read_window_size):
                    w = min(read_window_size, src.width - left)
                    window = Window(left, top, w, h)
                    ids = tree.query(box(*bounds(window, src.transform)))
                    shapes = [(geometries[int(i)], 1) for i in ids]
                    label = (rasterize(shapes, out_shape=(h, w), transform=src.window_transform(window),
                                       fill=0, all_touched=False, dtype='uint8')
                             if shapes else np.zeros((h, w), dtype='uint8'))
                    if not all_valid:
                        image = src.read(window=window, masked=True, out_dtype='float32')
                        valid = ~np.ma.getmaskarray(image).any(axis=0) & np.isfinite(image.data).all(axis=0)
                        label[~valid] = 255
                    water += int((label == 1).sum())
                    ignored += int((label == 255).sum())
                    dst.write(label, 1, window=window)
            background = total_pixels - water - ignored
        if total_pixels - ignored != int(row['valid_pixels']):
            raise ValueError(f"原清单与影像有效像素数不一致：{row['scene_id']}")
    pending.replace(label_path)
    updated = dict(row)
    updated.update(active=int(water + background > 0), water_pixels=water,
                   background_pixels=background, ignored_pixels=ignored)
    return updated


def prepare_labels(data_dir=None, read_window_size=None):
    """逐景原位替换标签，最后同步清单计数；中断后重新运行即可。"""
    data_dir = Path(OUTPUT_DIR if data_dir is None else data_dir)
    read_window_size = READ_WINDOW_SIZE if read_window_size is None else read_window_size
    manifest = data_dir / 'split_manifest.csv'
    with manifest.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    if not rows:
        raise ValueError('清单为空，没有可重建的标签')

    # 防止生成中断后，训练误读“部分新标签+部分旧标签”的混合数据。
    marker = data_dir / '.labels_rebuilding'
    marker.write_text('标签正在重建。请重新运行image_process.py（MODE=labels）完成后再训练。', encoding='utf-8')
    print(f'原位重建标签：{data_dir / "labels"}；共{len(rows)}景', flush=True)
    print('实际像素：水体=1，非水体=0，无效=255；显示：0白、1黑。', flush=True)
    totals = {split: dict(scenes=0, water=0, background=0, ignored=0)
              for split in ('train', 'val')}
    pending = manifest.with_name('split_manifest.labels.partial')
    with rasterio.Env(GDAL_CACHEMAX=64 * 1024 * 1024), \
            pending.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for number, row in enumerate(rows, 1):
            # 从源SHP重栅格化，不反转旧栅格，也不依赖查看软件的黑白显示。
            # 每幅临时标签写完并关闭后再替换原文件；原SAR文件及路径不动。
            result = rebuild_scene(row, data_dir, read_window_size)
            writer.writerow(result)
            stream.flush()
            total = totals[result['split']]
            total['scenes'] += 1
            for key, field in (('water', 'water_pixels'), ('background', 'background_pixels'),
                               ('ignored', 'ignored_pixels')):
                total[key] += int(result[field])
            print(f"{number}/{len(rows)} {row['scene_id']} [{row['split']}]："
                  f"1(水体)={result['water_pixels']:,}，0(非水体)={result['background_pixels']:,}，"
                  f"255(无效)={result['ignored_pixels']:,}", flush=True)

    # 清单计数和规则一并更新，训练会按标签文件变化自动重建窗口缓存与类别权重。
    pending.replace(manifest)
    (data_dir / 'label_statistics.json').write_text(
        json.dumps(totals, ensure_ascii=False, indent=2), encoding='utf-8')
    (data_dir / 'revision_settings.json').write_text(json.dumps(dict(
        label_rule=LABEL_RULE, labels='0 non_water / 1 water / 255 invalid',
        source='split_manifest.csv: source_label', storage='original SAR and in-place GeoTIFF labels',
        palette='0 white / 1 black', read_window_size=read_window_size),
        ensure_ascii=False, indent=2), encoding='utf-8')
    (data_dir / '标签说明.txt').write_text(
        LABEL_RULE + '\n有效区域：水体1、非水体0；255仅为无效像素。\n'
        '色带0白、1黑，颜色不等于像素数值。原SAR、原SHP和地理分组不变。\n', encoding='utf-8')
    marker.unlink()
    print(f'全部标签替换完成：{totals}\n请先运行src/datasets/image_process.py <配置.yaml>，再运行src/supervised/train.py <配置.yaml>。', flush=True)
    return data_dir


def inspect_scene(image_path, label_path, band=1, window_size=256, grid_size=3):
    """保留有效0值，统计原始DN与标签两类分布；只读有限数值。"""
    values, labels = [], []
    sampled = invalid = 0
    with rasterio.open(image_path) as src, rasterio.open(label_path) as gt:
        if (src.shape,src.crs,src.transform)!=(gt.shape,gt.crs,gt.transform):
            raise ValueError('标签与SAR网格不一致')
        h,w=min(window_size,src.height),min(window_size,src.width)
        # 分层网格单元的中心，而非外接矩形四角；仍保留读到的合法0值。
        ys=((np.arange(grid_size)+.5)/grid_size*max(0,src.height-h)).astype(int)
        xs=((np.arange(grid_size)+.5)/grid_size*max(0,src.width-w)).astype(int)
        coords=sorted({(int(y),int(x)) for y in ys for x in xs})
        for y,x in coords:
            win=Window(x,y,w,h)
            a=src.read(band,window=win,masked=True,out_dtype='float32')
            b=gt.read(1,window=win,masked=True)
            good=~np.ma.getmaskarray(a)&np.isfinite(a.data)
            labelled=~np.ma.getmaskarray(b)&np.isin(b.data,(0,1))
            sampled+=good.size;invalid+=int((~good).sum())
            values.append(a.data[good].astype(np.float64))
            labels.append(np.where(labelled,b.data.astype(np.int64),-1)[good])
        x=np.concatenate(values); y=np.concatenate(labels)
        if not len(x):
            raise ValueError('抽样位置没有有效SAR像素')
        q=np.percentile(x,(1,5,50,95,99))
        positive=x>0
        pq=np.percentile(x[positive],(1,50,95,99)) if positive.any() else [float('nan')]*4
        water=x[(y==1)&positive];land=x[(y==0)&positive]
        result=dict(width=src.width,height=src.height,dtype=src.dtypes[band-1],
                    pixel_x=abs(src.transform.a),pixel_y=abs(src.transform.e),
                    sampled_pixels=sampled,valid_samples=len(x),invalid_ratio=invalid/sampled,
                    minimum=float(x.min()),maximum=float(x.max()),mean=float(x.mean()),std=float(x.std()),
                    p01=q[0],p05=q[1],p50=q[2],p95=q[3],p99=q[4],
                    zero_ratio=float(np.mean(x==0)),negative_ratio=float(np.mean(x<0)),
                    water_samples=int(sum(y==1)),land_samples=int(sum(y==0)),
                    water_median=float(np.median(x[y==1])) if (y==1).any() else float('nan'),
                    land_median=float(np.median(x[y==0])) if (y==0).any() else float('nan'),
                    storage_scale=src.scales[band-1],storage_offset=src.offsets[band-1],
                    declared_unit=str(src.units[band-1]),physical_domain='unconfirmed',
                    calibration_status='not_verified_from_DN_statistics',
                    positive_samples=int(positive.sum()),positive_p01=pq[0],positive_p50=pq[1],
                    positive_p95=pq[2],positive_p99=pq[3],
                    water_positive_samples=len(water),land_positive_samples=len(land),
                    water_positive_median=float(np.median(water)) if len(water) else float('nan'),
                    land_positive_median=float(np.median(land)) if len(land) else float('nan'))
    return result


def audit(root=None, output_dir=None):
    """输出全部采用景，独立列出待读数/尺度审查项；不修改采用清单。"""
    root=Path(OUTPUT_DIR if root is None else root)
    output=Path(output_dir) if output_dir else AUDIT_OUTPUT_ROOT/datetime.now().strftime('%Y%m%d_%H%M%S')
    output.mkdir(parents=True,exist_ok=False)
    with (root/'split_manifest.csv').open(encoding='utf-8-sig',newline='') as f:
        scenes=[r for r in csv.DictReader(f) if int(r['active'])==1]
    rows=[]
    for i,r in enumerate(scenes,1):
        row=dict(scene_id=r['scene_id'],split=r['split'],group_id=r['group_id'],
                 scene=r.get('scene',''),sensor=r['scene_id'].split('-')[0],status='ok',error='')
        try:
            row.update(inspect_scene(root/'images'/r['split']/r['name'],
                                     root/'labels'/r['split']/r['label_name'],BAND,RADIOMETRY_WINDOW_SIZE,GRID_SIZE))
        except Exception as e:
            row.update(status='read_error',error=str(e))
            print(f'读取异常：{r["scene_id"]}：{e}',flush=True)
        rows.append(row)
        if i%25==0 or i==len(scenes):
            print(f'幅度分布审查：{i}/{len(scenes)}',flush=True)
            save_csv(output/'scene_statistics.partial.csv',rows)
    train=[r for r in rows if r['split']=='train' and r['status']=='ok']
    for row in rows:
        row.update(review_reason='',reference_count=0,reference_p50_low='',reference_p50_high='',
                   recommended_sample_weight='not_assigned')
        if row['status']!='ok':
            row['review_reason']='read_error';continue
        pool=[r for r in train if r['sensor']==row['sensor'] and np.isclose(r['pixel_x'],row['pixel_x']) and np.isfinite(r['positive_p50'])]
        scope='same_sensor_and_pixel_size'
        if len(pool)<5:
            pool=[r for r in train if np.isfinite(r['positive_p50'])];scope='all_train_fallback'
        row['reference_scope']=scope
        if not pool:
            row['review_reason']='no_training_reference';continue
        low,high=np.percentile([r['positive_p50'] for r in pool],(1,99))
        row.update(reference_count=len(pool),reference_p50_low=low,reference_p50_high=high)
        flags=[]
        if row['positive_p50']<low or row['positive_p50']>high:flags.append('positive_median_outside_train_p01_p99')
        if row['negative_ratio']>0:flags.append('negative_values_check_domain')
        if row['zero_ratio']>.2:flags.append('many_zeros_check_coverage_not_auto_nodata')
        if row['invalid_ratio']>.2:flags.append('large_invalid_sample_fraction')
        if row['std']==0:flags.append('constant_sample')
        if row['water_positive_samples']>=512 and row['land_positive_samples']>=512 and row['water_positive_median']>=row['land_positive_median']:
            flags.append('water_not_darker_check_scene_or_labels')
        row['review_reason']=';'.join(flags)
    save_csv(output/'scene_statistics.csv',rows)
    save_csv(output/'review_scenes.csv',[r for r in rows if r['review_reason']])
    (output/'scene_statistics.partial.csv').unlink(missing_ok=True)
    (output/'README.txt').write_text(
        '3x3 interior cell-center 256px windows; sampled estimates, not exact full-scene histograms.\n'
        'All-valid and conditional-positive distributions are both reported; positives do NOT redefine validity.\n'
        'Reference quantiles use train conditional-positive medians, preferably same sensor/pixel size.\n'
        'Flags are review candidates, not evidence of corruption or wrong labels. Zero remains valid.\n'
        'No automatic scene weights or radiometric calibration. DN moments cannot identify amplitude vs power.\n',encoding='utf-8')
    print(f'审查完成：{output}；待复核{sum(bool(r["review_reason"]) for r in rows)}/{len(rows)}景',flush=True)
    return output


def save_csv(path, rows):
    if rows:
        fields=list(dict.fromkeys(k for r in rows for k in r))
        with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def display_bytes(values, valid, low, high):
    """有效数值映射到1..255，0仅用于显示无效区；有效原始0值仍然保留。"""
    scaled=np.clip((values-low)/max(high-low,1e-6),0,1)
    return np.where(valid,1+254*np.nan_to_num(scaled,nan=0,posinf=1,neginf=0),0).astype('uint8')


def read_overview(src):
    """分块平均幅度生成概览；忽略无效/非有限值，合法0参与平均，边缘保留。"""
    looks = max(QUICKLOOK_LOOKS, (max(src.width, src.height) + QUICKLOOK_MAX_SIDE - 1) // QUICKLOOK_MAX_SIDE)
    height = (src.height + looks - 1) // looks
    width = (src.width + looks - 1) // looks
    overview = np.zeros((height, width), dtype='float32')
    counts = np.zeros((height, width), dtype='int64')
    columns = np.arange(0, src.width, looks)
    # 每批读取整数个平均窗口，避免分条接缝改变平均结果。
    step = max(1, STRIPE_ROWS // looks)
    for top in range(0, height, step):
        rows = min(step, height - top)
        start = top * looks
        raw = src.read(BAND, window=Window(0, start, src.width, min(rows * looks, src.height - start)),
                       masked=True, out_dtype='float32')
        valid = ~np.ma.getmaskarray(raw) & np.isfinite(raw.data)
        values = np.where(valid, raw.data, 0).astype('float64')
        row_starts = np.arange(0, raw.shape[0], looks)
        sums = np.add.reduceat(np.add.reduceat(values, row_starts, axis=0), columns, axis=1)
        number = np.add.reduceat(np.add.reduceat(valid.astype('int64'), row_starts, axis=0), columns, axis=1)
        np.divide(sums, number, out=sums, where=number > 0)
        overview[top:top + rows] = sums
        counts[top:top + rows] = number
    return overview, counts, looks






def save_record(path, record):
    """临时写入后发布，只有完整成果才允许续跑复用。"""
    path = Path(path)
    pending = path.with_suffix(path.suffix + '.partial')
    pending.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    pending.replace(path)


def file_stamp(path):
    """VRT也登记实际引用影像，避免仅修改底层SAR后继续复用旧成果。"""
    path = Path(path).resolve()
    items = [(str(path), path.stat().st_size, path.stat().st_mtime_ns)]
    if path.suffix.lower() == '.vrt':
        for node in ET.parse(path).findall('.//SourceFilename'):
            linked = Path(node.text)
            if node.get('relativeToVRT') == '1':
                linked = path.parent / linked
            items.extend(file_stamp(linked))
    return items


def stage_root(stage):
    return {'org': OUTPUT_DIR, 'calibrated': CALIBRATED_ROOT,
            'filtered': FILTERED_ROOT, 'normalized': NORMALIZED_ROOT}[stage]


def stage_record(folder):
    path = Path(folder) / 'preprocessing.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def write_normalized_scene(source, pending, bands, mean, std):
    """分块标准化并在SSD完成压缩，再跨盘复制到目标临时文件；公式和有效域不变。"""
    if NORMALIZE_READ_SIZE < 1 or NORMALIZE_THREADS < 1:
        raise ValueError('归一化读取块和压缩线程数需为正整数')
    NORMALIZE_WORK_DIR.mkdir(parents=True, exist_ok=True)
    timings = dict(read_seconds=0., calculate_seconds=0., write_compress_seconds=0.)
    # 均值/标准差使用原统计精度，完成运算后才转float32，与旧实现一致。
    mean, std = np.asarray(mean)[:,None,None], np.asarray(std)[:,None,None]
    with TemporaryDirectory(prefix='normalize_', dir=NORMALIZE_WORK_DIR) as temporary:
        local = Path(temporary)/'normalized.tif'
        with rasterio.open(source) as src:
            profile = src.profile.copy()
            profile.update(driver='GTiff', count=len(bands), dtype='float32', nodata=float('nan'),
                           tiled=True, blockxsize=256, blockysize=256, compress='deflate', predictor=3,
                           zlevel=NORMALIZE_COMPRESSION_LEVEL, num_threads=str(NORMALIZE_THREADS), BIGTIFF='IF_SAFER')
            with rasterio.open(local,'w',**profile) as dst:
                for top in range(0,src.height,NORMALIZE_READ_SIZE):
                    for left in range(0,src.width,NORMALIZE_READ_SIZE):
                        win=Window(left,top,min(NORMALIZE_READ_SIZE,src.width-left),min(NORMALIZE_READ_SIZE,src.height-top))
                        started=time.perf_counter()
                        raw=src.read(bands,window=win,masked=True,out_dtype='float32')
                        timings['read_seconds']+=time.perf_counter()-started
                        started=time.perf_counter()
                        valid=~np.ma.getmaskarray(raw).any(axis=0)&np.isfinite(raw.data).all(axis=0)
                        values=(raw.data-mean)/std
                        values=np.where(valid[None],values,np.nan).astype('float32')
                        timings['calculate_seconds']+=time.perf_counter()-started
                        started=time.perf_counter()
                        dst.write(values,window=win)
                        timings['write_compress_seconds']+=time.perf_counter()-started
                dst.update_tags(input_stage='normalized', statistics='train_only', source=str(source))
                started=time.perf_counter()
            # 多线程压缩剩余任务在关闭文件时完成，也计入写入时间。
            timings['write_compress_seconds']+=time.perf_counter()-started
        started=time.perf_counter()
        shutil.copyfile(local,pending)
        timings['publish_copy_seconds']=time.perf_counter()-started
    return {key:round(value,3) for key,value in timings.items()}


def cleanup_snap_cache(filtering, rows, source_root, source_bands, target):
    """只清理全批成功后可丢弃的缓存TIFF；保留其他批次、日志和锁文件。"""
    status = stage_record(target)
    if not status.get('complete') or (target/'.preprocessing_incomplete').exists():
        return
    cache_root = SNAP_CACHE_DIR.resolve()
    removed, freed = 0, 0
    for row in rows:
        source = source_root/'images'/row['split']/row['name']
        try:
            # create=False不会启动GPT；支持续跑时直接复用正式成果的场景。
            cached = filtering.snap.prepare_file(source, source_bands, create=False).resolve()
            if cached.parent.parent != cache_root or cached.name != 'filtered.tif':
                raise ValueError(f'拒绝清理非指定缓存路径：{cached}')
            # 缓存必须独立于源影像和正式成果，不能因路径配置错误删掉数据。
            output = target/'images'/row['split']/row['name']
            if cached == source.resolve() or cached == output.resolve() or not output.is_file():
                raise ValueError(f'缓存不是可删除的独立副本：{cached}')
            # 遇到其他进程正在准备该缓存时跳过，不阻塞或停止其他处理。
            with FileLock(str(cached.parent/'prepare.lock'), timeout=0):
                size = cached.stat().st_size
                cached.unlink()
                (cached.parent/'processing.json').unlink(missing_ok=True)
                removed += 1
                freed += size
        except FileNotFoundError:
            continue  # 已清理的缓存无需重新生成。
        except (OSError, ValueError, Timeout) as error:
            print(f'SNAP缓存保留：{row["scene_id"]}；{error}', flush=True)
    filtering.snap._prepared.clear()
    save_record(target/'cache_cleanup.json', dict(removed_images=removed, freed_bytes=freed,
                cache_root=str(cache_root), scope='current_batch_completed_cache_images_only'))
    print(f'SNAP缓存清理：删除{removed}个缓存影像，释放{freed/1024**3:.2f} GiB；日志和其他任务缓存保留。', flush=True)


def process_stage(stage):
    """校正/滤波/归一化分别落盘；每景参数和源文件一致时复用，支持中断重跑。"""
    if stage == 'calibrated':
        source_root = OUTPUT_DIR
    elif stage == 'filtered':
        source_root = stage_root(FILTER_INPUT_STAGE)
    else:
        source_root = stage_root(NORMALIZE_INPUT_STAGE)
    target = stage_root(stage)
    if Path(source_root).resolve() == Path(target).resolve():
        raise ValueError('源目录和处理目录不能相同')
    if (OUTPUT_DIR / '.labels_rebuilding').exists():
        raise RuntimeError('标签仍在重建，请完成后再处理')
    manifest = OUTPUT_DIR / 'split_manifest.csv'
    with manifest.open(encoding='utf-8-sig', newline='') as f:
        all_rows = list(csv.DictReader(f))
    rows = [r for r in all_rows if int(r['active']) == 1]
    source_bands = BANDS if source_root == OUTPUT_DIR else tuple(range(1,len(BANDS)+1))
    parent = stage_record(source_root)
    label_stamps = [file_stamp(OUTPUT_DIR/'labels'/r['split']/r['label_name']) for r in rows]
    label_fingerprint = hashlib.sha256(json.dumps(label_stamps,sort_keys=True).encode()).hexdigest()
    if source_root != OUTPUT_DIR and parent.get('source_label_fingerprint') != label_fingerprint:
        raise RuntimeError('原始标签已变化，请先重新运行上游处理阶段同步标签副本')
    if source_root != OUTPUT_DIR and (not parent.get('complete') or (source_root / '.preprocessing_incomplete').exists()):
        raise RuntimeError(f'输入阶段未完成：{source_root}')
    if source_root != OUTPUT_DIR and parent.get('source_manifest_sha256') != manifest_fingerprint(manifest):
        raise RuntimeError('输入阶段的标签/划分清单已变化，请先重新生成该阶段')
    domain = parent.get('output_domain', INPUT_DOMAIN)
    quantity = parent.get('quantity', 'uncalibrated_DN')
    filtering = None
    if stage == 'calibrated':
        if CALIBRATION_QUANTITY not in ('sigma0', 'beta0', 'gamma0') or not CALIBRATION_REFERENCE:
            raise ValueError('辐射校正尚未配置：需厂商系数/LUT或已校正成果，明确CALIBRATION_QUANTITY和CALIBRATION_REFERENCE；幅度平方不是辐射校正')
        if CALIBRATION_COMMAND is None and CALIBRATION_PRODUCTS_DIR is None:
            raise ValueError('请配置CALIBRATION_COMMAND参数列表，或CALIBRATION_PRODUCTS_DIR已有校正成果')
        if CALIBRATION_OUTPUT_DOMAIN not in ('amplitude', 'intensity'):
            raise ValueError('校正输出域需明确为amplitude或intensity')
        domain, quantity = CALIBRATION_OUTPUT_DOMAIN, CALIBRATION_QUANTITY
    if stage == 'filtered':
        snap_preprocess.configure_paths(SNAP_GPT_PATH, SNAP_CACHE_DIR, SNAP_WORK_DIR,
                                        threads=SNAP_THREADS, tile_cache=SNAP_TILE_CACHE)
        filtering = SpeckleFilter(dict(method='snap_lee', input_domain=domain, window_size=FILTER_WINDOW, enl=ENL))
    parameters = dict(stage=stage, bands=list(BANDS), input_domain=parent.get('output_domain', INPUT_DOMAIN),
                      output_domain='standardized' if stage == 'normalized' else domain, quantity=quantity,
                      source_root=str(source_root), source_manifest_sha256=manifest_fingerprint(manifest),
                      filter=filtering.config if filtering else None,
                      calibration_reference=CALIBRATION_REFERENCE if stage == 'calibrated' else parent.get('calibration_reference',''),
                      calibration_command=CALIBRATION_COMMAND if stage == 'calibrated' else None)
    if stage == 'normalized':
        if NORMALIZATION_REFERENCE is not None:
            reference_path = Path(NORMALIZATION_REFERENCE)
            if reference_path.parent.resolve() == target.resolve():
                raise ValueError('归一化参考不能指向本次输出目录')
            reference = json.loads(reference_path.read_text(encoding='utf-8'))
            if not reference.get('complete') or reference.get('stage') != 'normalized' or (reference_path.parent/'.preprocessing_incomplete').exists():
                raise ValueError('参考归一化阶段尚未完整完成')
            # 仅复用已完成训练统计；不同滤波/物理量不能混用同一标准化尺度。
            keys = ('stage','bands','input_domain','output_domain','quantity','filter','calibration_reference')
            reference_parent = reference.get('upstream',{})
            if any(reference_parent.get(k) != parent.get(k) for k in keys) or reference.get('bands') != list(BANDS):
                raise ValueError('参考归一化的上游处理或波段与本批不一致')
            mean, std = np.asarray(reference['mean']), np.asarray(reference['std'])
            if mean.shape != (len(BANDS),) or std.shape != mean.shape or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
                raise ValueError('参考均值/标准差无效')
            indices = []
            parameters.update(statistics_protocol='reference_train_statistics',
                              statistics_reference=str(reference_path),
                              statistics_reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest())
            print(f'复用训练归一化统计：{reference_path}；mean={mean}；std={std}', flush=True)
        else:
            ds = SARSceneDataset(source_root, WINDOW_CACHE_DIR, bands=source_bands, window_size=WINDOW_SIZE,
                                 halo=HALO, normalization_windows=NORMALIZATION_WINDOWS, seed=SEED)
            try:
                mean, std, indices = ds.normalization(ds.train_ids)
            finally:
                ds.close()
            parameters['statistics_protocol'] = 'fixed_train_windows_only'
        parameters.update(mean=mean.tolist(), std=std.tolist(),
                          statistics_window_indices=indices, window_size=WINDOW_SIZE, halo=HALO,
                          seed=SEED, normalization_windows=NORMALIZATION_WINDOWS, upstream=parent)
    target.mkdir(parents=True, exist_ok=True)
    marker = target / '.preprocessing_incomplete'
    marker.write_text('处理中；请完成image_process.py后再训练', encoding='utf-8')
    save_record(target / 'preprocessing.json', dict(parameters, complete=False))
    # 标签只是引用同一监督依据；使用独立副本，避免后续操作修改源标签。
    for split in ('train','val'):
        for kind in ('images','labels'):
            (target / kind / split).mkdir(parents=True, exist_ok=True)
    reused = generated = 0
    try:
        for number, row in enumerate(rows, 1):
            source = source_root / 'images' / row['split'] / row['name']
            output = target / 'images' / row['split'] / row['name']
            label = OUTPUT_DIR / 'labels' / row['split'] / row['label_name']
            receipt = output.with_suffix(output.suffix + '.processing.json')
            stamps = file_stamp(source)
            imported = None
            if stage == 'calibrated' and CALIBRATION_PRODUCTS_DIR is not None:
                imported = Path(CALIBRATION_PRODUCTS_DIR) / row['split'] / row['name']
                stamps += file_stamp(imported)
            signature = dict(parameters, sources=stamps)
            match = False
            if output.exists() and receipt.exists():
                match = json.dumps(json.loads(receipt.read_text(encoding='utf-8')).get('signature'),sort_keys=True) == json.dumps(signature,sort_keys=True)
                if match:
                    with rasterio.open(source) as src, rasterio.open(output) as dst:
                        match = src.shape == dst.shape and src.crs == dst.crs and src.transform == dst.transform and dst.count == len(BANDS)
            print(f'{stage} {number}/{len(rows)} [{row["split"]}] {row["scene_id"]}：{"复用" if match else "处理"}', flush=True)
            if match:
                reused += 1
            else:
                pending = output.with_suffix('.tif.partial')
                if stage == 'filtered':
                    cached = filtering.snap.prepare_file(source, source_bands, create=True)
                    # 使用GeoTIFF副本确保后续成果不依赖缓存被清理。
                    started = time.perf_counter()
                    # 缓存已是符合要求的压缩GeoTIFF，直接复制，保持文件字节一致。
                    shutil.copyfile(cached, pending)
                    copy_seconds = round(time.perf_counter()-started, 3)
                    save_record(output.with_suffix('.copy_timing.json'), dict(cache_copy_seconds=copy_seconds))
                    print(f'  缓存直接复制：{copy_seconds}秒', flush=True)
                elif stage == 'calibrated':
                    if imported is not None:
                        raster_copy(str(imported), str(pending), driver='GTiff', compress='deflate', tiled=True, BIGTIFF='IF_SAFER')
                    else:
                        if not isinstance(CALIBRATION_COMMAND, (list, tuple)):
                            raise ValueError('CALIBRATION_COMMAND需为参数列表，不经shell拼接')
                        def wsl_path(path):
                            text = Path(path).resolve().as_posix()
                            return '/mnt/' + text[0].lower() + text[2:] if len(text)>1 and text[1]==':' else text
                        fields = dict(input=str(source), output=str(pending), scene_id=row['scene_id'],
                                      input_wsl=wsl_path(source), output_wsl=wsl_path(pending))
                        command = [str(arg).format(**fields) for arg in CALIBRATION_COMMAND]
                        with output.with_suffix('.calibration.log').open('w',encoding='utf-8') as log:
                            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                                           creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                else:
                    timings = write_normalized_scene(source, pending, source_bands, mean, std)
                    save_record(output.with_suffix('.normalize_timing.json'), timings)
                    print(f'  归一化耗时（秒）：{timings}', flush=True)
                with rasterio.open(source) as src, rasterio.open(pending) as dst:
                    if src.shape != dst.shape or src.crs != dst.crs or src.transform != dst.transform or dst.count != len(BANDS):
                        raise ValueError(f'处理结果网格/波段改变，不能沿用原标签：{row["scene_id"]}')
                if stage == 'calibrated':
                    with rasterio.open(source) as raw, rasterio.open(pending) as calibrated:
                        for _, win in raw.block_windows(1):
                            original = raw.read(BANDS, window=win, masked=True)
                            valid = ~np.ma.getmaskarray(original).any(axis=0) & np.isfinite(original.data).all(axis=0)
                            values = calibrated.read(window=win, masked=True)
                            usable = ~np.ma.getmaskarray(values).any(axis=0) & np.isfinite(values.data).all(axis=0)
                            if (valid & ~usable).any() or (values.data[:,valid] < 0).any():
                                raise ValueError('校正结果丢失原有效像素或出现负值；请检查有效域和输出物理域')
                pending.replace(output)
                save_record(receipt, dict(signature=signature))
                generated += 1
            copied_label = target / 'labels' / row['split'] / row['label_name']
            if not copied_label.exists() or copied_label.stat().st_mtime_ns != label.stat().st_mtime_ns or copied_label.stat().st_size != label.stat().st_size:
                shutil.copy2(label, copied_label)
            save_record(target/'processing_status.json', dict(stage=stage, complete=False, completed=number,
                        expected=len(rows), reused=reused, generated=generated, last_scene=row['scene_id']))
        write_relative_manifest(manifest, target/'split_manifest.csv')
        if stage == 'normalized':
            # 在发布完成标记前预建窗口索引；训练只读取，不再估计/应用归一化。
            ds = SARSceneDataset(target, WINDOW_CACHE_DIR, bands=tuple(range(1,len(BANDS)+1)), window_size=WINDOW_SIZE,
                                 halo=HALO, normalization_windows=NORMALIZATION_WINDOWS, seed=SEED)
            ds.close()
        save_record(target/'preprocessing.json', dict(parameters, complete=True, source_data_root=str(OUTPUT_DIR), source_label_fingerprint=label_fingerprint))
        save_record(target/'processing_status.json', dict(stage=stage, complete=True, completed=len(rows),
                    expected=len(rows), reused=reused, generated=generated))
        marker.unlink()
        print(f'{stage}全部完成：{target}', flush=True)
        if stage == 'filtered' and CLEAN_SNAP_CACHE_ON_SUCCESS:
            cleanup_snap_cache(filtering, rows, source_root, source_bands, target)
        return stage_record(target)
    except (Exception, KeyboardInterrupt) as error:
        save_record(target/'processing_error.json', dict(error=str(error) or '手动中断；重新运行可续接'))
        raise


def export_scene(path):
    """按已有成果输出原图/校正/滤波/归一化概览；缺失阶段明确标注。"""
    output = path.parent / 'quicklooks'
    output.mkdir(exist_ok=True)
    figure = output / (path.stem + '.quick_overview.png')
    metadata = figure.with_suffix('.json')
    sources = [('org',path,{}), *[(stage, stage_root(stage)/'images'/path.parent.name/path.name, stage_record(stage_root(stage)))
                                for stage in ('calibrated','filtered','normalized')]]
    available = []
    for stage, candidate, record in sources:
        ready = stage == 'org' or (record.get('complete') and not (stage_root(stage)/'.preprocessing_incomplete').exists())
        available.append(candidate.exists() and ready)
    signature = dict(version=5, looks=QUICKLOOK_LOOKS, max_side=QUICKLOOK_MAX_SIDE,
                     band=BAND, z_range=list(Z_DISPLAY_RANGE), db_range=list(QUICKLOOK_BACKSCATTER_DB_RANGE),
                     sources=[dict(stage=stage, files=file_stamp(candidate) if ok else [], record=record)
                              for (stage,candidate,record),ok in zip(sources,available)])
    if figure.exists() and metadata.exists():
        saved = json.loads(metadata.read_text(encoding='utf-8'))
        if json.dumps(saved.get('configuration'),sort_keys=True) == json.dumps(signature,sort_keys=True):
            return dict(name=path.name,status='reused',**saved['statistics'])
    panels = []
    for (stage,candidate,record),ok in zip(sources,available):
        if not ok:
            panels.append(None)
            continue
        with rasterio.open(candidate) as src:
            values,counts,looks = read_overview(src)
            width,height = src.width,src.height
        valid = counts > 0
        if stage == 'normalized':
            gray = display_bytes(values, valid, *Z_DISPLAY_RANGE)
            display = f'fixed z range {Z_DISPLAY_RANGE}'
        elif record.get('quantity') in ('sigma0','beta0','gamma0'):
            positive = valid & (values > 0)
            db = (20 if record['output_domain']=='amplitude' else 10) * np.log10(np.maximum(values,1e-30))
            gray = display_bytes(db, positive, *QUICKLOOK_BACKSCATTER_DB_RANGE)
            display = f'{record["quantity"]}; fixed dB {QUICKLOOK_BACKSCATTER_DB_RANGE}; zero black'
        else:
            gray = np.where(valid,np.clip(values,0,255),0).astype('uint8')
            display = 'amplitude mean; direct gray [0,255], no stretch'
        panels.append((gray,display))
    shape = panels[0][0].shape
    col,row_height = max(shape[1],620), shape[0]+72
    canvas = Image.new('L',(2*col+24,2*row_height+54),255)
    draw = ImageDraw.Draw(canvas)
    # 使用Pillow自带字体，不依赖Windows或Linux的字体安装路径。
    font = ImageFont.load_default(size=15)
    draw.text((8,4),path.stem,fill=0,font=font)
    draw.text((8,26),'Display overviews only; source pixels and training files are not modified.',fill=0,font=font)
    rectangles = {}
    for i,((stage,_,_),panel) in enumerate(zip(sources,panels)):
        x,y=(i%2)*(col+24),54+(i//2)*row_height
        draw.text((x+8,y+4),stage.upper(),fill=0,font=font)
        if panel is None:
            draw.text((x+8,y+28),'Not generated / stage incomplete',fill=0,font=font)
        else:
            gray,description=panel
            draw.text((x+8,y+28),description,fill=0,font=font)
            canvas.paste(Image.fromarray(gray),(x,y+65))
            rectangles[stage]=[x,y+65,gray.shape[1],gray.shape[0]]
    pending=figure.with_suffix('.png.partial')
    canvas.save(pending,format='PNG',compress_level=PNG_COMPRESSION)
    pending.replace(figure)
    statistics=dict(width=width,height=height,figure_width=canvas.width,figure_height=canvas.height,
                    overview_width=shape[1],overview_height=shape[0],available_stages=[r[0] for r,ok in zip(sources,available) if ok])
    save_record(metadata,dict(configuration=signature,statistics=statistics,panels=rectangles))
    return dict(name=path.name,status='written',**statistics)


def export_quicklooks():
    """生成四阶段概览，不启动校正、滤波或归一化计算。"""
    if QUICKLOOK_LOOKS < 1 or QUICKLOOK_MAX_SIDE < 1:
        raise ValueError('概览平均倍数和最长边需为正整数')
    files = [p for split in SPLITS for p in sorted((QUICKLOOK_IMAGE_ROOT/split).iterdir())
             if p.is_file() and p.suffix.lower() in ('.tif','.tiff','.vrt')]
    errors = []
    for i,path in enumerate(files,1):
        try:
            row=export_scene(path)
            print(f'快视图 {i}/{len(files)}：{path.stem} {row["status"]}；{row["available_stages"]}',flush=True)
        except Exception as error:
            errors.append(dict(image=str(path),error=str(error)))
            print(f'快视图失败 {path.name}：{error}',flush=True)
    for split in SPLITS:
        folder=QUICKLOOK_IMAGE_ROOT/split/'quicklooks'
        folder.mkdir(exist_ok=True)
        save_record(folder/'overview_status.json',dict(complete=not errors,expected=len(files),errors=errors))
    print(f'快视图完成：成功{len(files)-len(errors)}/{len(files)}',flush=True)


def main(config_path=None):
    """按显式YAML执行；all串联准备、预处理和快视图。"""
    if config_path is not None:
        configure(config_path)
    if CONFIG is None:
        raise ValueError("请传入YAML路径，或先调用configure(config_path)")
    print(f"影像处理配置：{CONFIG_PATH}；模式：{MODE}", flush=True)
    if MODE == 'labels':
        prepare_labels(OUTPUT_DIR, READ_WINDOW_SIZE)
    elif MODE in ('prepare','split','inspect'):
        prepare_dataset()
    elif MODE == 'radiometry':
        audit(OUTPUT_DIR)
    elif MODE == 'calibrate':
        process_stage('calibrated')
    elif MODE == 'filter':
        process_stage('filtered')
    elif MODE == 'normalize':
        process_stage('normalized')
    elif MODE in ('all','preprocess'):
        expected_filter = 'calibrated' if ENABLE_CALIBRATION else 'org'
        expected_normalize = 'filtered' if ENABLE_FILTER else expected_filter
        if ENABLE_FILTER and FILTER_INPUT_STAGE != expected_filter:
            raise ValueError('FILTER_INPUT_STAGE与校正开关不一致')
        if NORMALIZE_INPUT_STAGE != expected_normalize:
            raise ValueError('NORMALIZE_INPUT_STAGE与校正/滤波开关不一致')
        if MODE == 'all':
            prepare_dataset(mode='prepare', reuse_complete=True)
            if not (OUTPUT_DIR/'split_manifest.csv').is_file():
                raise RuntimeError('prepare尚未全部完成，保留续跑记录；本次不执行预处理和快视图')
        if ENABLE_CALIBRATION:
            process_stage('calibrated')
        if ENABLE_FILTER:
            process_stage('filtered')
        process_stage('normalized')
        if MODE == 'all':
            export_quicklooks()
    elif MODE == 'quicklook':
        export_quicklooks()
    else:
        raise ValueError('MODE选all/prepare/labels/split/inspect/radiometry/calibrate/filter/normalize/preprocess/quicklook')


if __name__ == '__main__':
    main(config_argument())
