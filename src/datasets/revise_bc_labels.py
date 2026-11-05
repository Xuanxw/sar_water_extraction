"""按已确认的SHP语义重建标签：面内水体、面外非水体，保留无效像素。

运行 python src/datasets/revise_bc_labels.py。
读取已准备好的整景影像，不重新解压ORG、不复制大幅SAR、不改旧标签。
训练划分沿用旧清单；本次只修订已准备的场景，不自动增加其他场景。
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import sys
import xml.etree.ElementTree as ET

import numpy as np
import rasterio
from rasterio.shutil import copy as copy_raster
from rasterio.features import rasterize
from rasterio.windows import Window, bounds
from shapely.geometry import box
from shapely.strtree import STRtree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.prepare_bc_data import MANIFEST_FIELDS, read_polygons

SOURCE_DIR = Path(r"I:\水体数据\prepared_scenes")
OUTPUT_DIR = Path(r"I:\水体数据\prepared_scenes_v2")
CACHE_DIR = Path(r"E:\Code_projects\sar_water_extraction\data_cache\bc_scenes_v2")
READ_WINDOW_SIZE = 1024
SCENE_WORKERS = 4  # 不同景独立写入，主线程按清单顺序登记，内存有界。
BUILD_TRAIN_CACHE = True
WINDOW_SIZE = 1024
HALO = 128
NORMALIZATION_WINDOWS = 2
SEED = 42
LABEL_RULE = "用户确认：所有SHP面为水体，面外为非水体；不使用class字段区分水陆"


def rebuild_scene(row, source_dir, output_dir, read_window_size=1024, *, create_vrt=True):
    """沿SAR网格生成水体1/背景0；create_vrt=False时保留原影像路径。"""
    image_path = source_dir / 'images' / row['split'] / row['name']
    label_path = output_dir / 'labels' / row['split'] / row['label_name']
    vrt_path = output_dir / 'images' / row['split'] / (row['scene_id'] + '.vrt')
    label_path.parent.mkdir(parents=True, exist_ok=True)
    pending = label_path.with_suffix('.tif.partial')
    with rasterio.open(image_path) as src:
        # class=0和class=1的面均烧为1，其他字段值也不决定水陆。
        geometries, _ = read_polygons(row['source_label'], src.crs, None, {None: 1})
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
    if create_vrt:
        # 旧的独立版本入口可创建VRT；prepare_bc3原位重建不进入此分支。
        vrt_path.unlink(missing_ok=True)
        copy_raster(str(image_path), str(vrt_path), driver='VRT')
        document = ET.parse(vrt_path)
        for node in document.findall('.//SourceFilename'):
            node.text = str(image_path.resolve())
            node.set('relativeToVRT', '0')
        document.write(vrt_path, encoding='utf-8', xml_declaration=True)
        updated['name'] = vrt_path.name
    updated.update(active=int(water + background > 0), water_pixels=water,
                   background_pixels=background, ignored_pixels=ignored)
    return updated


def prepare_labels(source_dir=SOURCE_DIR, output_dir=OUTPUT_DIR, read_window_size=READ_WINDOW_SIZE, workers=1):
    """建立独立版本，逐景保存续跑进度，全部成功才发布训练清单。"""
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    if source_dir.resolve() == output_dir.resolve():
        raise ValueError('新版目录不能覆盖旧数据目录')
    manifest = source_dir / 'split_manifest.csv'
    with manifest.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    groups = {split: {r['group_id'] for r in rows if r['split'] == split and int(r['active'])}
              for split in ('train', 'val')}
    if not groups['train'] or not groups['val'] or groups['train'] & groups['val']:
        raise ValueError('原train/val地理组不完整或有交叉')
    if any(int(r['active']) != 1 for r in rows):
        raise ValueError('本入口只修订当前有效场景；不自动启用停用样本')
    signature = dict(source_dir=str(source_dir.resolve()), label_rule=LABEL_RULE,
                     source_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
    source_stamps = []
    for row in rows:
        label = Path(row['source_label'])
        paths = [source_dir / 'images' / row['split'] / row['name']]
        paths += [label.with_suffix(ext) for ext in ('.shp', '.shx', '.dbf', '.prj', '.cpg')
                  if label.with_suffix(ext).exists()]
        for path in paths:
            stat = path.stat()
            source_stamps.append([str(path), stat.st_size, stat.st_mtime_ns])
    signature['source_stamps'] = source_stamps
    settings = output_dir / 'revision_settings.json'
    if output_dir.exists():
        if not settings.exists() or json.loads(settings.read_text(encoding='utf-8')) != signature:
            raise ValueError('新版目录的来源或标签规则不一致，请使用新目录')
        if (output_dir / 'split_manifest.csv').exists():
            print(f'新版标签已完成，保留现有结果：{output_dir}', flush=True)
            return output_dir
    else:
        for kind in ('images', 'labels'):
            for split in ('train', 'val'):
                (output_dir / kind / split).mkdir(parents=True, exist_ok=True)
        settings.write_text(json.dumps(signature, ensure_ascii=False, indent=2), encoding='utf-8')
    partial = output_dir / 'split_manifest.csv.partial'
    existing = []
    if partial.exists():
        with partial.open(encoding='utf-8-sig', newline='') as stream:
            existing = list(csv.DictReader(stream))
    completed = {r['scene_id'] for r in existing}
    records = list(existing)
    with partial.open('a' if partial.exists() else 'w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        if stream.tell() == 0:
            writer.writeheader()
        pending_rows = [(n, row) for n, row in enumerate(rows, 1) if row['scene_id'] not in completed]
        pool = ThreadPoolExecutor(max_workers=workers)
        try:
            results = pool.map(lambda pair: rebuild_scene(pair[1], source_dir, output_dir, read_window_size), pending_rows)
            for (n, row), result in zip(pending_rows, results):
                records.append(result)
                writer.writerow(result)
                stream.flush()
                print(f"标签 {n}/{len(rows)}：{row['scene_id']} [{row['split']}] "
                      f"水体={result['water_pixels']:,}，非水体={result['background_pixels']:,}", flush=True)
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
    totals = {split: dict(scenes=0, water=0, background=0, ignored=0) for split in ('train', 'val')}
    for r in records:
        total = totals[r['split']]
        total['scenes'] += 1
        for key, field in [('water', 'water_pixels'), ('background', 'background_pixels'), ('ignored', 'ignored_pixels')]:
            total[key] += int(r[field])
    if len(records) != len(rows) or any(not t['water'] or not t['background'] for t in totals.values()):
        raise ValueError('新版标签尚不具备完整的train/val水体和非水体监督')
    (output_dir / 'label_statistics.json').write_text(json.dumps(totals, ensure_ascii=False, indent=2), encoding='utf-8')
    (output_dir / '标签说明.txt').write_text(
        LABEL_RULE + '\n0=非水体（白），1=水体（黑），255=无效（nodata）。\n'
        '保留原地理分组，图像为VRT引用，不能删除旧prepared_scenes/images目录。\n'
        '面外作为非水体是用户2026-11-02确认的标注规则，不代表已逐景人工验收。\n', encoding='utf-8')
    partial.replace(output_dir / 'split_manifest.csv')
    print(f'新版标签完成：{output_dir}\n{totals}', flush=True)
    return output_dir


def main():
    with rasterio.Env(GDAL_CACHEMAX=64 * 1024 * 1024):
        output = prepare_labels(workers=SCENE_WORKERS)
    if BUILD_TRAIN_CACHE:
        from datasets.sar_scene_dataset import SARSceneDataset
        dataset = SARSceneDataset(output, CACHE_DIR, window_size=WINDOW_SIZE, halo=HALO,
                                  normalization_windows=NORMALIZATION_WINDOWS, seed=SEED)
        try:
            mean, std, _ = dataset.normalization(dataset.train_ids)
            water, background = dataset.windows[dataset.train_ids, 5:7].sum(axis=0).tolist()
            weights = ((water + background) / (2 * background), (water + background) / (2 * water))
            print(f'训练/验证窗口：{len(dataset.train_ids)}/{len(dataset.val_ids)}', flush=True)
            print(f'训练均值={mean}，标准差={std}；BCE权重（非水体，水体）={weights}', flush=True)
        finally:
            dataset.close()


if __name__ == '__main__':
    main()
