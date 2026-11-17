"""OSM共享路网与水体冲突核验；不修改输入水体或提供道路硬删除功能。"""
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import ssl
import time
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import HTTPSHandler, Request, build_opener

import fiona
from affine import Affine
from filelock import FileLock
import numpy as np
from pyproj import CRS, Transformer
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize, shapes
from rasterio.vrt import WarpedVRT
from rasterio.windows import bounds as window_bounds
from shapely import STRtree, transform as transform_geometry
from shapely.geometry import LineString, box, mapping, shape

from thres_segmentation.water_analysis import pixel_areas, area_sum
from utils.files import sha256, signature, write_json
from utils.raster import windows


def download_osm_road_tile(bounds, settings, cache, opener, log=print):
    """下载WGS84固定网格内的highway线，完整响应原子落盘后才可复用。

    bounds为西/南/东/北经纬度。共享缓存键不包含机器路径或服务镜像；
    超时remark、HTML及不完整JSON不得被当作空路网。中止后只重下未完成瓦片。
    """
    classes = sorted(settings['highway_classes'])
    identity = dict(bounds=list(bounds), highway_classes=classes, query_version=1)
    path = Path(cache) / 'tiles' / (signature(identity) + '.json')
    if path.exists():
        saved = json.loads(path.read_text(encoding='utf-8'))
        if saved.get('complete') and saved.get('query') == identity:
            log(f'复用OSM道路瓦片：{path.name}')
            return path
    west, south, east, north = bounds
    pattern = '|'.join(re.escape(value) for value in classes)
    query = (f'[out:json][timeout:{settings["query_timeout_seconds"]}];'
             f'way["highway"~"^({pattern})$"]({south},{west},{north},{east});out tags geom;')
    last_error = None
    for attempt in range(settings['retries']):
        for endpoint in settings['endpoints']:
            try:
                log(f'请求OSM道路瓦片 {list(bounds)}；尝试{attempt + 1}；{endpoint}')
                request = Request(endpoint, data=urlencode({'data': query}).encode('utf-8'),
                                  headers={'Content-Type': 'application/x-www-form-urlencoded'})
                with opener.open(request, timeout=settings['http_timeout_seconds']) as response:
                    data = json.load(response)
                if data.get('remark') or not isinstance(data.get('elements'), list):
                    raise ValueError(f'Overpass响应不完整：{data.get("remark", "缺少elements")}')
                for element in data['elements']:
                    if element.get('type') != 'way' or len(element.get('geometry', [])) < 2:
                        raise ValueError('Overpass道路缺少完整线几何')
                write_json(path, dict(
                    complete=True, query=identity, endpoint=endpoint,
                    downloaded_at=datetime.now(timezone.utc).isoformat(),
                    osm_base_timestamp=data.get('osm3s', {}).get('timestamp_osm_base'),
                    attribution='© OpenStreetMap contributors', license='ODbL-1.0',
                    elements=data['elements'],
                ))
                log(f'OSM道路瓦片完成：{len(data["elements"])}条；{path.name}')
                return path
            except (URLError, OSError, ValueError) as exc:
                last_error = exc
                log(f'OSM请求未完成：{exc}')
        if attempt + 1 < settings['retries']:
            time.sleep(settings['retry_delay_seconds'])
    raise RuntimeError(f'OSM道路下载未完成；保留已完成瓦片供续跑。最后错误：{last_error}')


def prepare_osm_road_network(aoi, crs, cache, settings, log=print):
    """按行政区一次准备米制道路中心线和缓冲面，供所有日期/景复用。

    返回缓冲几何/属性列表、米制CRS、共享GPKG路径及来源说明。width是道路总宽米，
    缓冲半径=总宽/2+额外配准容差；缺失/无法解释的width使用YAML道路等级宽度。
    桥梁、隧道均保留。OSM无道路记录不代表已确认非道路。
    """
    metric_crs = CRS.from_user_input(crs)
    if not metric_crs.is_projected or any(axis.unit_conversion_factor != 1 for axis in metric_crs.axis_info[:2]):
        raise ValueError('OSM道路缓冲需要以米为单位的投影坐标系')
    step = settings['tile_degrees']
    if step <= 0 or not settings['highway_classes'] or not settings['endpoints']:
        raise ValueError('OSM瓦片间隔、道路类别及服务地址不能为空')
    if (settings['context_buffer_m'] < 0 or settings['extra_buffer_m'] < 0
            or not 0 < settings['min_width_m'] <= settings['max_width_m']
            or settings['retries'] < 1):
        raise ValueError('OSM缓冲/宽度/重试参数无效')
    region = aoi.buffer(settings['context_buffer_m'])
    to_ll = Transformer.from_crs(metric_crs, 4326, always_xy=True)
    region_ll = transform_geometry(region, to_ll.transform, interleaved=False)
    west, south, east, north = region_ll.bounds
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    tile_paths = []
    elements = {}
    # 公共服务串行请求，避免每景重复下载或并发冲击Overpass；锁仅保护同一共享缓存。
    with FileLock(str(cache / 'osm_roads.lock')):
        opener = build_opener(HTTPSHandler(context=ssl.create_default_context()))
        opener.addheaders = [('User-Agent', settings['user_agent'])]
        for lat in range(math.floor(south / step), math.ceil(north / step)):
            for lon in range(math.floor(west / step), math.ceil(east / step)):
                bounds = tuple(round(value, 10) for value in (lon * step, lat * step, (lon + 1) * step, (lat + 1) * step))
                if not region_ll.intersects(box(*bounds)):
                    continue
                path = download_osm_road_tile(bounds, settings, cache, opener, log)
                tile_paths.append(path)
                for element in json.loads(path.read_text(encoding='utf-8'))['elements']:
                    elements[element['id']] = element
        # 原始瓦片与宽度策略分开缓存：改缓冲宽度不重下OSM，同一路段跨瓦片按osm_id去重。
        recipe = dict(
            version=1, aoi_wkb=region.wkb_hex, crs=metric_crs.to_string(),
            tiles=[dict(name=path.name, sha256=sha256(path)) for path in tile_paths],
            widths={key: settings[key] for key in ('width_by_class_m', 'default_width_m', 'min_width_m',
                                                  'max_width_m', 'extra_buffer_m')},
        )
        network_path = cache / ('network_' + signature(recipe) + '.gpkg')
        receipt = network_path.with_suffix('.json')
        if network_path.exists() and receipt.exists():
            info = json.loads(receipt.read_text(encoding='utf-8'))
            if info.get('complete'):
                with fiona.open(network_path, layer='road_buffers') as layer:
                    records = list(layer)
                buffers = [shape(row['geometry']) for row in records]
                attributes = [dict(row['properties']) for row in records]
                log(f'复用OSM共享路网：{len(buffers)}条；{network_path}')
                return dict(buffers=buffers, attributes=attributes, crs=metric_crs,
                            path=network_path, metadata=info)
        rows = []
        to_metric = Transformer.from_crs(4326, metric_crs, always_xy=True)
        for identifier, element in sorted(elements.items()):
            tags = element.get('tags', {})
            if tags.get('highway') not in settings['highway_classes']:
                continue
            line = LineString([(point['lon'], point['lat']) for point in element['geometry']])
            line = transform_geometry(line, to_metric.transform, interleaved=False)
            if not line.intersects(region):
                continue
            width_tag = str(tags.get('width', ''))
            # 只解释明确的米制数字；例如"3;5"、英尺或文字宽度不能猜成米。
            matched = re.fullmatch(r'\s*(\d+(?:\.\d+)?)\s*(?:m|metres?|meters?)?\s*', width_tag, re.I)
            width = float(matched[1]) if matched else 0.0
            from_tag = settings['min_width_m'] <= width <= settings['max_width_m']
            if not from_tag:
                width = settings['width_by_class_m'].get(tags.get('highway'), settings['default_width_m'])
            if not settings['min_width_m'] <= width <= settings['max_width_m']:
                raise ValueError(f'道路等级宽度超出设置范围：{tags.get("highway")}={width}')
            rows.append(dict(osm_id=str(identifier), highway=tags.get('highway'), name=tags.get('name', ''),
                             width_m=width, width_tag=width_tag, width_source='osm_width' if from_tag else 'class_fallback',
                             bridge=str(tags.get('bridge', '')), tunnel=str(tags.get('tunnel', '')),
                             surface=str(tags.get('surface', '')), geometry=line))
        properties = dict(osm_id='str', highway='str', name='str', width_m='float', width_tag='str',
                          width_source='str', bridge='str', tunnel='str', surface='str')
        buffers = [row['geometry'].buffer(row['width_m'] / 2 + settings['extra_buffer_m']) for row in rows]
        partial = network_path.with_suffix('.partial.gpkg')
        partial.unlink(missing_ok=True)
        for layer, geometries, geometry_type in (('road_centerlines', [row['geometry'] for row in rows], 'LineString'),
                                                  ('road_buffers', buffers, 'Polygon')):
            with fiona.open(partial, 'w', driver='GPKG', layer=layer, crs_wkt=metric_crs.to_wkt(),
                            schema=dict(geometry=geometry_type, properties=properties)) as dst:
                dst.writerecords(dict(geometry=mapping(geometry), properties={key: row[key] for key in properties})
                                 for row, geometry in zip(rows, geometries))
        partial.replace(network_path)
        info = dict(complete=True, policy='review_only_no_deletion', features=len(rows), crs=metric_crs.to_string(),
                    tile_count=len(tile_paths), width_from_tag=sum(row['width_source'] == 'osm_width' for row in rows),
                    width_from_class=sum(row['width_source'] == 'class_fallback' for row in rows),
                    attribution='© OpenStreetMap contributors', license='ODbL-1.0',
                    license_url='https://www.openstreetmap.org/copyright',
                    source_note='固定瓦片下载快照，非道路完整性保证；桥梁和隧道不排除。', recipe=recipe)
        write_json(receipt, info)
        log(f'OSM共享路网完成：{len(rows)}条；{network_path}')
        return dict(buffers=buffers, attributes=[{key: row[key] for key in properties} for row in rows],
                    crs=metric_crs, path=network_path, metadata=info)


def export_road_water_conflicts(filtered, raw, corrected, final, network, output, *, block_size=512, preview_size=1200, log=print):
    """只读三个水体阶段，分块输出路网冲突栅格、核验面、面积JSON及对比图。

    road_conflict.tif三波段依次为原阈值候选、辅助修正后、最终成面冲突；
    0无已记录路网冲突、1有冲突、255共同有效域外。缓冲命中不等于确认误检。
    GPKG核验面按块输出，不做耗时几何并集，不以面数量当作错误对象数。
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    corridor_path = output / 'road_corridor.tif'
    conflict_path = output / 'road_conflict.tif'
    vectors_path = output / 'road_conflicts.gpkg'
    names = ('raw_threshold', 'after_auxiliary', 'final_polygon')
    areas = {name: 0.0 for name in names}
    counts = {name: 0 for name in names}
    common_area = road_area = 0.0
    with rasterio.Env(GDAL_CACHEMAX=64 * 1024 * 1024), ExitStack() as stack:
        readers = [stack.enter_context(rasterio.open(path)) for path in (raw, corrected, final)]
        src = readers[0]
        if any((reader.crs, reader.transform, reader.shape) != (src.crs, src.transform, src.shape) for reader in readers[1:]):
            raise ValueError('路网冲突核验需要三个水体阶段使用同一原生网格')
        to_raster = Transformer.from_crs(network['crs'], src.crs, always_xy=True)
        roads = [transform_geometry(geometry, to_raster.transform, interleaved=False) for geometry in network['buffers']]
        index = STRtree(roads)
        profile = dict(src.profile, driver='GTiff', count=1, dtype='uint8', nodata=255,
                       compress='lzw', BIGTIFF='IF_SAFER')
        corridor = stack.enter_context(rasterio.open(corridor_path.with_suffix('.partial.tif'), 'w', **profile))
        conflict = stack.enter_context(rasterio.open(conflict_path.with_suffix('.partial.tif'), 'w', **dict(profile, count=3)))
        for band, name in enumerate(names, 1):
            conflict.set_band_description(band, name)
        pending_vectors = vectors_path.with_suffix('.partial.gpkg')
        pending_vectors.unlink(missing_ok=True)
        layers = {name: stack.enter_context(fiona.open(
            pending_vectors, 'w', driver='GPKG', layer=name + '_conflicts', crs_wkt=src.crs.to_wkt(),
            schema=dict(geometry='Polygon', properties=dict(area_m2='float')),
        )) for name in (names[0], names[2])}
        for number, win in enumerate(windows(src, block_size), 1):
            data = [reader.read(1, window=win) for reader in readers]
            valid = np.logical_and.reduce([np.isin(value, (0, 1)) for value in data])
            transform = src.window_transform(win)
            bounds = window_bounds(win, src.transform)
            # 只栅格化与当前窗口相交的道路，不在每块遍历整区路网；半径已按米制计算。
            selected = index.query(box(min(bounds[0], bounds[2]), min(bounds[1], bounds[3]),
                                       max(bounds[0], bounds[2]), max(bounds[1], bounds[3])), predicate='intersects')
            if len(selected):
                road = rasterize(((roads[i], 1) for i in selected), out_shape=valid.shape,
                                 transform=transform, fill=0, dtype='uint8', all_touched=False) == 1
            else:
                road = np.zeros(valid.shape, dtype=bool)
            pixel_area = pixel_areas(src.crs, transform, valid.shape)
            common_area += area_sum(valid, pixel_area)
            road_area += area_sum(road & valid, pixel_area)
            corridor.write(np.where(valid, road, 255).astype('uint8'), 1, window=win)
            for band, (name, value) in enumerate(zip(names, data), 1):
                overlap = road & valid & (value == 1)
                counts[name] += int(overlap.sum())
                areas[name] += area_sum(overlap, pixel_area)
                conflict.write(np.where(valid, overlap, 255).astype('uint8'), band, window=win)
                if name in layers:
                    records = []
                    for polygon, _ in shapes(overlap.astype('uint8'), mask=overlap, transform=transform):
                        geometry = shape(polygon)
                        # 面积字段采用与栅格统计相同的测地/投影口径，不能用经纬度平方当平方米。
                        if src.crs.is_geographic:
                            area = abs(CRS.from_user_input(src.crs).get_geod().geometry_area_perimeter(geometry)[0])
                        else:
                            area = geometry.area * float(pixel_area)
                            area /= abs(transform.a * transform.e - transform.b * transform.d)
                        records.append(dict(geometry=polygon, properties=dict(area_m2=area)))
                    # 一块一次事务，避免每个道路小面单独提交SQLite导致长时间停顿。
                    layers[name].writerecords(records)
            if number == 1 or number % 50 == 0:
                log(f'道路/水体冲突核验：{number}块；最终冲突{areas["final_polygon"] / 1e6:.4f} km²')
    corridor_path.with_suffix('.partial.tif').replace(corridor_path)
    conflict_path.with_suffix('.partial.tif').replace(conflict_path)
    pending_vectors.replace(vectors_path)
    report = dict(
        policy='review_only_no_deletion', water_modified=False, network_path=str(network['path']),
        attribution=network['metadata']['attribution'], license=network['metadata']['license'],
        common_valid_area_m2=common_area, mapped_road_corridor_area_m2=road_area,
        conflict_area_m2=areas, conflict_pixels=counts, bands={str(i): name for i, name in enumerate(names, 1)},
        artifacts=dict(corridor=corridor_path.name, conflict=conflict_path.name, vectors=vectors_path.name,
                       preview='road_conflicts.png'),
        interpretation='冲突仅表示水体候选与OSM缓冲重叠，不是真值误检。桥梁、淹水道路需人工核验；无路网记录不等于非道路。',
        vector_note='核验面按栅格块输出，块边界不是独立错误对象边界。',
    )
    with rasterio.Env(GDAL_CACHEMAX=64 * 1024 * 1024):
        save_road_conflict_preview(filtered, raw, final, corridor_path, conflict_path,
                                   output / 'road_conflicts.png', preview_size)
    write_json(output / 'road_review.json', report)
    log(f'道路冲突核验完成；原阈值候选{areas["raw_threshold"] / 1e6:.4f} km²，'
        f'最终保留{areas["final_polygon"] / 1e6:.4f} km²；未删除水体')
    return [corridor_path, conflict_path, vectors_path, output / 'road_conflicts.png', output / 'road_review.json']


def save_road_conflict_preview(filtered, raw, final, corridor, conflict, output, size):
    """只对显示做P2–P98拉伸，冲突用max聚合保留细路，不改变原生面积统计。"""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    with rasterio.open(raw) as reference:
        scale = min(1.0, size / max(reference.shape))
        shape = tuple(max(1, round(length * scale)) for length in reference.shape)
        a = reference.transform
        sx, sy = reference.width / shape[1], reference.height / shape[0]
        target = dict(crs=reference.crs, transform=Affine(a.a * sx, a.b * sy, a.c, a.d * sx, a.e * sy, a.f),
                      width=shape[1], height=shape[0], resampling=Resampling.max, nodata=255)
    with rasterio.open(filtered) as src:
        image = src.read(1, out_shape=shape, out_dtype='float32', masked=True, resampling=Resampling.average).filled(np.nan)
    good = np.isfinite(image)
    low, high = np.percentile(image[good], (2, 98)) if good.any() else (0, 1)
    image = np.where(good, np.clip((image - low) / max(high - low, 1e-6), 0, 1), np.nan)
    # max聚合属于GDAL重投影算法，不能直接用于RasterIO的out_shape读取。
    with rasterio.open(corridor) as src, WarpedVRT(src, **target) as vrt:
        road = vrt.read(1, masked=True).filled(0) == 1
    figure = Figure(figsize=(15, 6))
    FigureCanvasAgg(figure)
    for i, (title, mask_path, band) in enumerate((('OSM road corridors (orange)', None, None),
                                               ('Raw candidates / road overlap (magenta)', raw, 1),
                                               ('Final water / road overlap (magenta)', final, 3)), 1):
        ax = figure.add_subplot(1, 3, i)
        ax.imshow(image, cmap='gray', vmin=0, vmax=1)
        overlay = np.zeros((*shape, 4), np.uint8)
        if mask_path is None:
            overlay[road] = (255, 170, 0, 210)
        else:
            with rasterio.open(mask_path) as src, WarpedVRT(src, **target) as vrt:
                water = vrt.read(1, masked=True).filled(0) == 1
            with rasterio.open(conflict) as src, WarpedVRT(src, **target) as vrt:
                overlap = vrt.read(band, masked=True).filled(0) == 1
            overlay[water] = (40, 160, 245, 160)
            overlay[overlap] = (255, 0, 180, 240)
        ax.imshow(overlay)
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    figure.suptitle('Review only: no water removed | SAR display P2-P98 | © OpenStreetMap contributors, ODbL')
    figure.tight_layout()
    figure.savefig(output, dpi=110)
    figure.clear()
