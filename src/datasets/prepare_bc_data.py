"""新SAR数据准备：修改顶部变量，运行 python src/datasets/prepare_bc_data.py。

输入：单景 GeoTIFF（.tif/.tiff），或仅含一个GeoTIFF的ZIP；标签为SHP面。
输出：float32 SAR切片 + uint8标签切片 + 当前训练器使用的split_manifest.csv。
只转换组织形式，保留存储数值/原生网格，不推断辐射单位，不自动修复标签。
"""
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile
import csv
import re
import shutil

import fiona
import numpy as np
import rasterio
from rasterio.enums import MaskFlags
from rasterio.features import rasterize
from rasterio.warp import transform_geom
from rasterio.windows import Window, bounds as window_bounds
from shapely.geometry import box, mapping, shape
from shapely.strtree import STRtree
from shapely.validation import explain_validity

# ---------- 入口和绝对路径 ----------
MODE = "split"  # split：只显示清单分类；inspect：检查格式；prepare：正式切块。
SPLIT_CSV = Path(r"I:\水体数据\水体数据_train_val划分_已确认.csv")
IMAGE_ROOT = Path(r"I:\水体数据\SAR影像_卫星环境应用中心")
LABEL_ROOT = Path(r"I:\水体数据\水体提取结果_卫星环境应用中心")
OUTPUT_DIR = Path(r"I:\水体数据\prepared")  # 必须是新的目录，防止混入上次切片。
CACHE_DIR = Path(r"I:\水体数据\prepare_cache")  # ZIP逐景展开，单景结束即清理临时影像。

# ---------- 切片和输入有效域 ----------
BANDS = (1,)  # 新数据单VV，文件波段编号从1开始。
TILE_SIZE = 512  # 步长同尺寸，不重叠；右/下边缘补无效值。
INSPECT_SAMPLES = 3
MAX_SCENES = None  # prepare试跑时可设小整数；正式运行用None。
INPUT_NODATA_VALUE = None  # 只填写已确认的无效编码；不默认把0当作无效。
VALID_MASK_DIR = None  # 可选：绝对目录，文件名为完整景号.tif/.tiff，1有效、0无效。
ALLOW_ALL_VALID_SOURCE = False  # 无nodata/内部mask时，只有确认整幅有效才设True。

# ---------- 标签类别：先确认含义，再填写，禁止自动猜测 ----------
CLASS_FIELD = "class"
CLASS_MAP = {}  # 仅确认0非水体/1水体后填 {0: 0, 1: 1}；未列值保持未知。
# 各景属性含义不同，可单独覆盖，例如："完整景号": {"field": "Value", "map": {0: 0, 255: 1}}
# field=None、map={None: 1} 仅适用于已确认所有面均为水体的标签；面外仍未知。
SCENE_CLASS_RULES = {}
# 同景有多个源文件时显式指定；不靠文件遍历顺序挑选版本。
SOURCE_OVERRIDES = {}  # "完整景号": {"image": Path(r"绝对影像路径"), "label": Path(r"绝对SHP路径")}

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
    """按完整景号配对，排除评价点；版本不唯一时由SOURCE_OVERRIDES指定。"""
    override = overrides.get(identifier, {})
    image_paths = [Path(override["image"])] if "image" in override else images.get(identifier, [])
    label_paths = [Path(override["label"])] if "label" in override else labels.get(identifier, [])
    polygons = []
    for path in label_paths:
        with fiona.open(path) as source:
            if "Polygon" in source.schema["geometry"]:
                polygons.append(path)
    if len(image_paths) != 1 or len(polygons) != 1:
        raise ValueError(f"{identifier}：需要唯一影像和面标签，实际{len(image_paths)}/{len(polygons)}；请核对或配置SOURCE_OVERRIDES")
    return image_paths[0], polygons[0]


def image_path(path, cache=None):
    """检查时通过GDAL直接读ZIP头；切块时展开一景，避免反复解压压缩流。"""
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
    """按明确映射读取面并转到影像CRS；未知类别不烧为背景，无效面不自动修复。"""
    if not class_map or any(v not in (0, 1) for v in class_map.values()):
        raise ValueError("先确认标签类别，再配置CLASS_MAP或SCENE_CLASS_RULES；映射结果只能为0/1")
    geometries, classes, skipped = [], [], Counter()
    with fiona.open(path) as source:
        if not source.crs:
            raise ValueError(f"标签缺少坐标系：{path}")
        if field is not None and field not in source.schema["properties"]:
            raise ValueError(f"标签没有字段{field}，实际为{list(source.schema['properties'])}：{path}")
        for feature in source:
            value = feature["properties"][field] if field is not None else None
            if value not in class_map:
                skipped[str(value)] += 1
                continue
            if not feature["geometry"]:
                raise ValueError(f"标签要素{feature.id}为空：{path}")
            geometry = shape(feature["geometry"])
            if not geometry.is_valid or geometry.is_empty:
                raise ValueError(f"标签要素{feature.id}几何待修复：{explain_validity(geometry)}；{path}")
            geometry = shape(transform_geom(source.crs_wkt, target_crs, mapping(geometry)))
            if geometry.geom_type not in ("Polygon", "MultiPolygon") or not geometry.is_valid:
                raise ValueError(f"标签要素{feature.id}转换后不是有效面：{path}")
            geometries.append(geometry)
            classes.append(class_map[value])
    if skipped:
        print(f"  未映射类别保持未知：{dict(skipped)}", flush=True)
    if not geometries:
        raise ValueError(f"没有已确认类别的有效面：{path}")
    return geometries, np.asarray(classes, dtype=np.uint8)


def prepare_scene(row, image_file, label_file, output, cache_root, *, bands, tile_size,
                  field, class_map, input_nodata=None, valid_mask=None, allow_all_valid=False):
    """顺序窗口读取一景，输出配对GeoTIFF；返回切片记录，场景归属始终不变。"""
    cache_root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="scene_", dir=cache_root) as cache, ExitStack() as stack:
        source = stack.enter_context(rasterio.open(image_path(image_file, cache)))
        if not source.crs or source.transform == rasterio.Affine.identity():
            raise ValueError(f"影像需要地理参考，不能仅凭.tiff后缀判断：{image_file}")
        if source.transform.b or source.transform.d:
            raise ValueError("当前切块入口处理无旋转的地理编码网格；旋转网格请先重投影")
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
        records = []
        for top in range(0, source.height, tile_size):
            for left in range(0, source.width, tile_size):
                height, width = min(tile_size, source.height-top), min(tile_size, source.width-left)
                window = Window(left, top, width, height)
                selected = tree.query(box(*window_bounds(window, source.transform)))
                if not len(selected):
                    continue  # 全部未标注的区域不生成训练样本。
                data = source.read(bands, window=window, masked=True, out_dtype="float32")
                valid = ~np.ma.getmaskarray(data).any(axis=0) & np.isfinite(data.data).all(axis=0)
                if input_nodata is not None:
                    valid &= (data.data != input_nodata).all(axis=0)
                if mask_source:
                    coverage = mask_source.read(1, window=window, masked=True)
                    valid &= ~np.ma.getmaskarray(coverage) & (coverage.data == 1)
                transform = source.window_transform(window)
                masks = []
                for category in (0, 1):
                    shapes = [(geometries[int(i)], 1) for i in selected if classes[i] == category]
                    masks.append(rasterize(shapes, out_shape=(height,width), transform=transform,
                                           fill=0, all_touched=False, dtype="uint8").astype(bool)
                                 if shapes else np.zeros((height,width), dtype=bool))
                # 正负标签重叠属于冲突，保持255，不能按矢量先后顺序覆盖。
                label = np.full((tile_size,tile_size), 255, dtype=np.uint8)
                actual = label[:height,:width]
                actual[valid & masks[0] & ~masks[1]] = 0
                actual[valid & masks[1] & ~masks[0]] = 1
                if not np.any(label != 255):
                    continue
                image = np.full((len(bands),tile_size,tile_size), np.nan, dtype=np.float32)
                image[:,:height,:width] = np.where(valid[None], data.data, np.nan)
                name = f"{row['scene_id']}_r{top:06d}_c{left:06d}.tif"
                common = dict(driver="GTiff",height=tile_size,width=tile_size,crs=source.crs,
                              transform=transform,compress="deflate")
                with rasterio.open(output/'images'/row['split']/name, 'w', **common,
                                   count=len(bands),dtype='float32',nodata=np.nan) as target:
                    target.write(image)
                    for channel, band in enumerate(bands, 1):
                        target.set_band_description(channel, source.descriptions[band-1] or f"source_band_{band}")
                with rasterio.open(output/'labels'/row['split']/name, 'w', **common,
                                   count=1,dtype='uint8',nodata=255) as target:
                    target.write(label,1)
                records.append(dict(name=name,label_name=name,split=row['split'],active=1,group_id=row['group_id'],
                                    scene_id=row['scene_id'],scene=row['scene'],source_image=str(image_file),
                                    source_label=str(label_file),row_off=top,col_off=left,source_height=height,source_width=width,
                                    valid_pixels=int(valid.sum()),water_pixels=int((label==1).sum()),
                                    background_pixels=int((label==0).sum()),ignored_pixels=int((label==255).sum()),
                                    pixel_x=source.res[0],pixel_y=source.res[1]))
        return records


def main():
    """检查或准备；只在全部选定场景成功后发布训练清单，失败时保留现场供检查。"""
    if MODE not in ("split", "inspect", "prepare") or TILE_SIZE < 1:
        raise ValueError("MODE选split/inspect/prepare，TILE_SIZE应大于0")
    with SPLIT_CSV.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
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
        allowed = ('是', '已确认', '待确认') if MODE == 'inspect' else ('是', '已确认')
        if row['是否采用'] in allowed:
            selected.append(dict(scene_id=identifier,split=split,group_id=group,scene=row['场景']))
    if not selected:
        message = "没有可处理场景。prepare只接受是否采用=是或已确认；当前待确认记录不会自动启用。"
        if MODE == 'prepare':
            raise ValueError(message)
        print(message)
        return
    if MODE == 'split':
        print(f"采用场景的划分：{dict(Counter(row['split'] for row in selected))}")
        print(f"采用场景地理组数：{len({row['group_id'] for row in selected})}")
        print('分类沿用清单中的地理组、行政区、时间与传感器；不进行地物分类、几何分析或切块。')
        return
    selected = selected[:INSPECT_SAMPLES] if MODE == 'inspect' else selected[:MAX_SCENES]
    images, labels = index_sources(IMAGE_ROOT, LABEL_ROOT)
    if MODE == 'inspect':
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
            raise ValueError(f"{row['scene_id']}尚未配置有效的0/1类别映射")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    for kind in ('images','labels'):
        for split in ('train','val'):
            (OUTPUT_DIR/kind/split).mkdir(parents=True)
    shutil.copy2(SPLIT_CSV, OUTPUT_DIR/'scene_split.csv')
    # 保存本次实际配置，避免之后修改脚本后无法追溯标签映射及有效域规则。
    settings = dict(bands=BANDS, tile_size=TILE_SIZE, class_field=CLASS_FIELD, class_map=CLASS_MAP,
                    scene_class_rules=SCENE_CLASS_RULES, input_nodata=INPUT_NODATA_VALUE,
                    valid_mask_dir=VALID_MASK_DIR, allow_all_valid_source=ALLOW_ALL_VALID_SOURCE,
                    source_overrides=SOURCE_OVERRIDES, values='raw stored values; no scale/offset or dB',
                    labels='0 background / 1 water / 255 unknown; conflicting classes ignored')
    (OUTPUT_DIR/'preparation_settings.txt').write_text(
        '\n'.join(f"{key} = {value!r}" for key,value in settings.items()), encoding='utf-8')
    totals = Counter()
    # 完成前仅写.partial；若中断，不要将这个不完整目录交给训练器。
    partial = OUTPUT_DIR/'split_manifest.csv.partial'
    with partial.open('w',encoding='utf-8-sig',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in selected:
            image, label = select_sources(row['scene_id'],images,labels,SOURCE_OVERRIDES)
            rule = SCENE_CLASS_RULES.get(row['scene_id'],dict(field=CLASS_FIELD,map=CLASS_MAP))
            valid_mask = None
            if VALID_MASK_DIR:
                candidates = [Path(VALID_MASK_DIR)/(row['scene_id']+suffix) for suffix in ('.tif','.tiff')]
                found = [path for path in candidates if path.is_file()]
                if len(found) != 1:
                    raise ValueError(f"缺少唯一的有效覆盖掩膜：{row['scene_id']}")
                valid_mask = found[0]
            tiles = prepare_scene(row,image,label,OUTPUT_DIR,CACHE_DIR,bands=BANDS,tile_size=TILE_SIZE,
                                  field=rule['field'],class_map=rule['map'],input_nodata=INPUT_NODATA_VALUE,
                                  valid_mask=valid_mask,allow_all_valid=ALLOW_ALL_VALID_SOURCE)
            writer.writerows(tiles)
            stream.flush()
            totals[row['split']] += len(tiles)
            for tile in tiles:
                totals['water_pixels'] += tile['water_pixels']
                totals['background_pixels'] += tile['background_pixels']
            print(f"{row['scene_id']} [{row['split']}]：输出{len(tiles)}个切片",flush=True)
    if not totals['train'] or not totals['val'] or not totals['water_pixels'] or not totals['background_pixels']:
        raise ValueError(f"当前准备结果不完整，尚未发布训练清单（需train/val、已确认水体/非水体）：{dict(totals)}")
    partial.replace(OUTPUT_DIR/'split_manifest.csv')
    print(f"准备完成：{OUTPUT_DIR}\n{dict(totals)}\n训练仍需选择单VV，并关闭土地覆盖第5波段修正。")


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(f"数据准备未完成：{error}",flush=True)
        raise SystemExit(1)
