"""按整景有效SAR范围统计地物占比；独立于模型，不改变标签与划分。

python src/datasets/classify_scenes.py
原始WorldCover保留11类，不能用后处理的二值历史水体图代替。
"""
from contextlib import ExitStack
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from thres_segmentation.auxiliary_data import tile_specs, download_tile, validate_raster

# 小工具直接在这里修改参数，不读取独立YAML；运行结果仍保存参数快照便于追溯。
PATHS = dict(
    MANIFESTS=[Path(r'I:\水体数据\prepared_scenes\split_manifest.csv')],
    # batch 2准备完成后，可在MANIFESTS追加其split_manifest.csv。
    AUXILIARY_CACHE_DIR=Path(r'E:\Code_projects\sar_water_extraction\data_cache\auxiliary'),
    OUTPUT_DIR=Path(r'E:\Code_projects\sar_water_extraction\outputs\scene_classification'),
)
OPTIONS = dict(
    DOWNLOAD_MISSING=True,
    INCLUDE_TERRAIN=True,
    WORLDCOVER_YEAR=2021,
    RESOLUTION_M=30,  # UTM网格抽样估计占比，不冒充原生10米精确面积。
    BLOCK_SIZE=256,
    MIN_COVERAGE=0.95,
    FEATURE_FRACTION=0.20,  # 占比达到20%标记“较多”，允许多个标签。
    DOMINANT_FRACTION=0.50,
    STEEP_SLOPE_DEG=15,
    STEEP_FRACTION=0.20,
    MAX_GRID_PIXELS=25000000,
    MAX_SCENES=None,  # None处理全部；可设3先检查结果。
    TIMEOUT=60,
    RETRIES=3,
)
CLASSES = {10:'树木', 20:'灌木', 30:'草地', 40:'农田', 50:'建成区',
           60:'裸地或稀疏植被', 70:'冰雪', 80:'永久水体', 90:'草本湿地', 95:'红树林', 100:'苔藓地衣'}
VERSION = 1


def read_scenes(manifests):
    """多清单按景号去重；保留原始split/group，不依据地物重新划分。"""
    scenes = {}
    for manifest in manifests:
        with Path(manifest).open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream):
                scene_id = row['scene_id']
                item = dict(row, manifest=str(manifest), image=str(
                    Path(manifest).parent/'images'/row['split']/row['name']))
                if scene_id in scenes:
                    old = scenes[scene_id]
                    if any(old.get(k, '') != row.get(k, '') for k in ('split','group_id','active')):
                        raise ValueError(f'同景清单划分冲突：{scene_id}')
                    continue
                scenes[scene_id] = item
    return list(scenes.values())


def get_tiles(product, bounds, cache, options):
    """复用后处理相同tiles缓存；下载中断由原下载器处理，不接受残缺文件。"""
    paths = []
    for spec in tile_specs(product, bounds, options['WORLDCOVER_YEAR']):
        path = cache/'tiles'/product/spec['filename']
        if path.is_file():
            validate_raster(path)
        elif options['DOWNLOAD_MISSING']:
            download_tile(spec['url'], path, timeout=options['TIMEOUT'], retries=options['RETRIES'])
        else:
            raise FileNotFoundError(f'未缓存：{path}；可开启DOWNLOAD_MISSING')
        paths.append(path)
    return paths


def read_mosaic(vrts, window, classes=False):
    """一次仅读一个小块；瓦片外和nodata保持未知，不能当作非农田/平地。"""
    shape = (int(window.height), int(window.width))
    result = np.full(shape, np.nan, dtype=np.float32)
    for vrt in vrts:
        block = vrt.read(1, window=window, masked=True)
        values = block.filled(np.nan)
        valid = ~np.ma.getmaskarray(block) & np.isfinite(values)
        if classes:
            valid &= np.isin(values, list(CLASSES))
        take = valid & ~np.isfinite(result)
        result[take] = values[take]
    return result


def summarize(counts, sar_count, terrain_count, steep_count, slope_sum, options):
    """覆盖不足时保留统计值，但不给确定标签；阈值是项目筛查规则。"""
    known = int(counts.sum())
    cover = known / sar_count if sar_count else 0.
    terrain_cover = terrain_count / sar_count if sar_count else 0.
    output = {'SAR抽样有效像素':sar_count, '土地覆盖有效像素':known,
              '土地覆盖覆盖率':cover, '地形有效像素':terrain_count, '地形覆盖率':terrain_cover}
    for i, name in enumerate(CLASSES.values()):
        output[name+'占比'] = float(counts[i]/known) if known else ''
    if known and cover >= options['MIN_COVERAGE']:
        best = int(np.argmax(counts))
        output['主导地物'] = list(CLASSES.values())[best] if counts[best]/known >= options['DOMINANT_FRACTION'] else '混合地物'
        output['较多地物'] = '；'.join(name for i, name in enumerate(CLASSES.values())
                                    if counts[i]/known >= options['FEATURE_FRACTION']) or '分散混合'
    else:
        output['主导地物'] = output['较多地物'] = '待确认：覆盖不足'
    output['平均坡度_度'] = slope_sum/terrain_count if terrain_count else ''
    output['陡坡占比'] = steep_count/terrain_count if terrain_count else ''
    output['地形标签'] = '待确认：地形未启用或覆盖不足'
    if terrain_count and terrain_cover >= options['MIN_COVERAGE']:
        output['地形标签'] = ('山地/陡坡候选' if steep_count/terrain_count >= options['STEEP_FRACTION']
                            else '陡坡占比较低')
    return output


def classify_scene(image_path, cache, options):
    """UTM规则网格近似面积占比；SAR只读抽样有效域，内存按块有界。"""
    resolution = float(options['RESOLUTION_M'])
    if resolution <= 0 or int(options['BLOCK_SIZE']) < 1:
        raise ValueError('RESOLUTION_M和BLOCK_SIZE必须为正')
    with rasterio.open(image_path) as source:
        if not source.crs:
            raise ValueError('影像没有CRS，不能匹配地物数据')
        bounds = transform_bounds(source.crs, 'EPSG:4326', *source.bounds, densify_pts=21)
        west, south, east, north = bounds
        if east-west > 6 or not (-80 <= south < north <= 84):
            raise ValueError('场景超出本脚本局部UTM统计适用范围，需要单独投影处理')
        zone = min(60, max(1, math.floor(((west+east)/2+180)/6)+1))
        crs = f'EPSG:{(32600 if (south+north)/2 >= 0 else 32700)+zone}'
        left, bottom, right, top = transform_bounds(source.crs, crs, *source.bounds, densify_pts=21)
        width, height = math.ceil((right-left)/resolution), math.ceil((top-bottom)/resolution)
        if width*height > options['MAX_GRID_PIXELS']:
            raise ValueError('统计网格过大：调大RESOLUTION_M或核查场景范围')
        transform = from_origin(left, top, resolution, resolution)
    notes = []
    products = {}
    # 每个产品单独失败记录，缺DEM不会伪造地形，也不抹去已有地类统计。
    for product in ('worldcover', 'copernicus'):
        if product == 'copernicus' and not options['INCLUDE_TERRAIN']:
            products[product] = []
            continue
        try:
            # DEM外扩约一个统计网格，减少场景边缘坡度缺值。
            padding = resolution*2/111000
            product_bounds = (west-padding/max(math.cos(math.radians((south+north)/2)),.1),
                              south-padding,east+padding/max(math.cos(math.radians((south+north)/2)),.1),north+padding) if product == 'copernicus' else bounds
            products[product] = get_tiles(product, product_bounds, cache, options)
        except Exception as error:
            notes.append(f'{product}: {error}')
            products[product] = []
    counts = np.zeros(len(CLASSES), dtype=np.int64)
    sar_count = terrain_count = steep_count = 0
    slope_sum = 0.
    with rasterio.Env(GDAL_CACHEMAX=64*1024*1024, GDAL_NUM_THREADS='1'), ExitStack() as stack:
        grid = dict(crs=crs, transform=transform, width=width, height=height, dtype='float32', nodata=np.nan)
        source = stack.enter_context(rasterio.open(image_path))
        sar = stack.enter_context(WarpedVRT(source, **grid, resampling=Resampling.nearest))
        vrts = {}
        for product, paths in products.items():
            vrts[product] = []
            for path in paths:
                src = stack.enter_context(rasterio.open(path))
                vrt = stack.enter_context(WarpedVRT(src, **grid,
                    resampling=Resampling.nearest if product == 'worldcover' else Resampling.bilinear))
                vrts[product].append(vrt)
        block_size = int(options['BLOCK_SIZE'])
        for y in range(0, height, block_size):
            for x in range(0, width, block_size):
                h, w = min(block_size,height-y), min(block_size,width-x)
                window = Window(x,y,w,h)
                raw = sar.read(1,window=window,masked=True)
                good = ~np.ma.getmaskarray(raw) & np.isfinite(raw.filled(np.nan))
                sar_count += int(good.sum())
                cover = read_mosaic(vrts['worldcover'],window,classes=True)
                for i, code in enumerate(CLASSES):
                    counts[i] += np.count_nonzero(good & (cover == code))
                if vrts['copernicus']:
                    # 块外扩1像素后中心差分；不在块边界产生假坡度。
                    x0,y0 = max(0,x-1), max(0,y-1)
                    x1,y1 = min(width,x+w+1), min(height,y+h+1)
                    dem = read_mosaic(vrts['copernicus'],Window(x0,y0,x1-x0,y1-y0))
                    if min(dem.shape) < 3:
                        continue
                    slope = np.full(dem.shape,np.nan,dtype=np.float32)
                    dx=(dem[1:-1,2:]-dem[1:-1,:-2])/(2*resolution)
                    dy=(dem[2:,1:-1]-dem[:-2,1:-1])/(2*resolution)
                    slope[1:-1,1:-1]=np.degrees(np.arctan(np.hypot(dx,dy)))
                    slope[~np.isfinite(dem)] = np.nan
                    slope=slope[y-y0:y-y0+h,x-x0:x-x0+w]
                    valid=good & np.isfinite(slope)
                    terrain_count += int(valid.sum())
                    steep_count += int(np.count_nonzero(valid & (slope >= options['STEEP_SLOPE_DEG'])))
                    slope_sum += float(slope[valid].sum(dtype=np.float64))
    result = summarize(counts,sar_count,terrain_count,steep_count,slope_sum,options)
    result.update(状态='完成' if not notes and sar_count else '待补充', 备注='；'.join(notes),
                  统计分辨率_m=resolution, 统计CRS=crs,
                  WorldCover来源=f'ESA WorldCover {options["WORLDCOVER_YEAR"]}',
                  地形来源='Copernicus GLO-30 DSM' if options['INCLUDE_TERRAIN'] else '未启用',
                  土地覆盖年份=options['WORLDCOVER_YEAR'],
                  瓦片路径='；'.join(str(p) for paths in products.values() for p in paths))
    if result['土地覆盖覆盖率'] < options['MIN_COVERAGE'] or (options['INCLUDE_TERRAIN'] and result['地形覆盖率'] < options['MIN_COVERAGE']):
        result['状态'] = '待补充'
    return result


def write_table(path, rows):
    """每完成一景更新UTF-8 BOM清单；临时文件替换，避免中断写坏旧表。"""
    fields=list(dict.fromkeys(k for row in rows for k in row))
    pending=path.with_suffix('.partial.csv')
    with pending.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    pending.replace(path)


def main():
    paths,options=PATHS,OPTIONS
    config=dict(paths=paths,classification=options)
    scenes=read_scenes(paths['MANIFESTS'])
    if options['MAX_SCENES'] is not None:
        scenes=scenes[:int(options['MAX_SCENES'])]
    output=paths['OUTPUT_DIR'];output.mkdir(parents=True,exist_ok=True)
    records=output/'records';records.mkdir(exist_ok=True)
    table=output/'场景地物分类.csv'
    (output/'classification_config.json').write_text(json.dumps(config,default=str,ensure_ascii=False,indent=2),encoding='utf-8')
    rows=[]
    for index,scene in enumerate(scenes,1):
        print(f'场景分类 {index}/{len(scenes)}：{scene["scene_id"]}',flush=True)
        row=dict(影像编号=scene['scene_id'],数据划分=scene['split'],地理组=scene.get('group_id',''),
                 原场景说明=scene.get('scene',''),影像路径=scene['image'],源清单=scene['manifest'])
        try:
            if scene.get('active','1') != '1':
                row.update(状态='未采用',备注='原manifest active不为1；保留原划分')
            else:
                stat=Path(scene['image']).stat()
                signature=hashlib.sha256(json.dumps(dict(version=VERSION,scene=scene,options=options,
                    cache=str(paths['AUXILIARY_CACHE_DIR']),size=stat.st_size,mtime=stat.st_mtime_ns),sort_keys=True).encode()).hexdigest()
                receipt=records/(hashlib.sha256(scene['scene_id'].encode()).hexdigest()[:20]+'.json')
                saved=json.loads(receipt.read_text('utf-8')) if receipt.is_file() else {}
                if saved.get('signature') == signature and saved['result']['状态'] == '完成':
                    result=saved['result'];print('  复用已完成统计',flush=True)
                else:
                    result=classify_scene(scene['image'],paths['AUXILIARY_CACHE_DIR'],options)
                    pending=receipt.with_suffix('.partial')
                    pending.write_text(json.dumps(dict(signature=signature,result=result),ensure_ascii=False,indent=2),encoding='utf-8')
                    pending.replace(receipt)
                row.update(result)
        except Exception as error:
            row.update(状态='失败',备注=str(error))
        rows.append(row);write_table(table,rows)
        print(f'  {row["状态"]}；{row.get("较多地物","")}；{row.get("地形标签","")}',flush=True)
    print(f'分类表：{table}；原始清单、影像、标签及train/val未修改',flush=True)


if __name__ == '__main__':
    main()
