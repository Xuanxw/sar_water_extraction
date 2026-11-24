"""根据案例 JSON 制作洪涝专题图和设施潜在暴露 GeoPackage。"""

import argparse
from glob import glob
import json
import re
from pathlib import Path
from datetime import datetime, timedelta

import fiona
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager, patheffects
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.colors import LightSource
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, PathPatch, Rectangle
from matplotlib.path import Path as PlotPath
from matplotlib.ticker import MaxNLocator
import numpy as np
from pyproj import CRS, Transformer
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.vrt import WarpedVRT
from shapely import transform, union_all
from shapely.geometry import LineString, MultiLineString, MultiPolygon, box, mapping, shape
from shapely.geometry.polygon import orient
from shapely.strtree import STRtree

from map_boundaries import collect_outlines, export_outlines, parts
from place_labels import label_map, read_labels
from print_map import fit_print_map


INK = '#182E3D'
RULE = '#A2ACB1'
GAIN = '#F29D27'
STABLE = '#5AA9D0'
LOSS = '#7067A2'
MISSING = '#DBDFDF'
ROAD = '#AA2835'
BRIDGE = '#6B1823'
BUILDINGS = ('#AE144F', '#D36883', '#116964')
WATER = [('stable_water', STABLE, '稳定水面'), ('water_gain', GAIN, '新增水面'),
         ('water_loss', LOSS, '减少水面')]


def load_vector(spec, crs):
    """读取面或线，并转换到制图使用的米制坐标系。"""
    with fiona.open(spec['path'], layer=spec.get('layer')) as source:
        project = Transformer.from_crs(source.crs, crs, always_xy=True).transform
        return [(transform(shape(f.geometry), project, interleaved=False), dict(f.properties))
                for f in source if f.geometry]


def exposure(specs, gain, common, crs, destination):
    """用新增水面与共同有效区筛选设施，线输出相交段，面输出建筑完整轮廓。"""
    index = STRtree(gain)
    outputs = {}
    for spec in specs:
        road = spec['kind'] == 'road'
        kind = 'LineString' if road else 'Polygon'
        constructor = MultiLineString if road else MultiPolygon
        schema = {'geometry': f'Multi{kind}', 'properties': {
            'source': 'str', 'source_id': 'str', 'name': 'str', 'overlap_m': 'float',
            'overlap_m2': 'float', 'overlap_pct': 'float', 'bridge_tunnel': 'int'}}
        rows = []
        seen = set()
        for geometry, props in load_vector(spec, crs):
            source_id = str(props[spec['id']])
            if source_id in seen:
                raise ValueError(f'{spec["layer"]} 中存在重复设施编号 {source_id}')
            seen.add(source_id)
            candidates = index.query(geometry, predicate='intersects')
            if len(candidates) == 0:
                continue
            intersection = geometry.intersection(union_all([gain[i] for i in candidates])).intersection(common)
            if road and intersection.length > 0:
                selected = list(parts(intersection, kind))
                if not selected:
                    continue
                measurement = sum(p.length for p in selected)
            elif not road and intersection.area > 0:
                selected = list(parts(geometry, kind))
                measurement = intersection.area
            else:
                continue
            flag = any(str(props.get(field)).lower() in ('t', 'true', '1', 'yes') for field in ('bridge', 'tunnel'))
            rows.append((constructor(selected), {'source': spec['label'], 'source_id': source_id,
                         'name': props.get('name'), 'overlap_m': measurement if road else None,
                         'overlap_m2': None if road else measurement,
                         'overlap_pct': None if road else 100 * measurement / geometry.area,
                         'bridge_tunnel': int(flag) if road else None}))
        with fiona.open(destination, 'w', driver='GPKG', layer=spec['key'], crs_wkt=crs.to_wkt(), schema=schema) as dst:
            dst.writerecords({'geometry': mapping(g), 'properties': p} for g, p in rows)
        outputs[spec['key']] = (spec, rows)
    return outputs


def fill(ax, geometries, color, *, edge='none', width=0.35, alpha=1, z=2):
    patches = []
    for geometry in geometries:
        for polygon in parts(geometry, 'Polygon'):
            polygon = orient(polygon)
            rings = [PlotPath(np.asarray(r.coords), closed=True) for r in (polygon.exterior, *polygon.interiors)]
            patches.append(PathPatch(PlotPath.make_compound_path(*rings)))
    ax.add_collection(PatchCollection(patches, facecolor=color, edgecolor=edge, linewidth=width,
                                     alpha=alpha, zorder=z))


def lines(ax, geometries, color, width, z, style='solid'):
    paths = [np.asarray(p.coords) for g in geometries for p in parts(g, 'LineString')]
    ax.add_collection(LineCollection(paths, colors=color, linewidths=width, linestyles=style, zorder=z))


def frame(ax, bounds, north=True, small=False, crs=None):
    x0, y0, x1, y1 = bounds
    ax.set(xlim=(x0, x1), ylim=(y0, y1), xticks=[], yticks=[], facecolor='#F5F4EF')
    ax.set_aspect('equal')
    for edge in ax.spines.values():
        edge.set_color(RULE)
        edge.set_linewidth(.55)
    dx, dy = x1 - x0, y1 - y0
    target = dx * .19
    step = 10 ** np.floor(np.log10(target))
    length = max(a * step for a in (.2, .5, 1, 2, 5) if a * step <= target)
    start, base = x0 + dx * .055, y0 + dy * .055
    ax.plot([start, start + length], [base, base], color='white', lw=5, zorder=16, solid_capstyle='butt')
    ax.plot([start, start + length], [base, base], color=INK, lw=1.8, zorder=17, solid_capstyle='butt')
    text = f'{length/1000:g} km' if length >= 1000 else f'{length:g} m'
    ax.text(start, base + dy * .015, f'0        {text}', fontsize=8 if small else 11, zorder=17,
            color=INK, path_effects=[patheffects.withStroke(linewidth=2.6, foreground='white')])
    if north:
        ax.annotate('N', xy=(.954, .895), xytext=(.954, .97), xycoords='axes fraction',
                    ha='center', va='top', size=13, weight='bold', color='white',
                    bbox={'boxstyle': 'square,pad=.19', 'fc': INK, 'ec': 'none'},
                    arrowprops={'arrowstyle': '<|-', 'color': 'white', 'lw': 1.4}, zorder=17)
    if crs and not small:
        # 将经纬网投影到地图坐标系，并用网线和图框的真实交点标注。
        to_geo = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
        to_map = Transformer.from_crs('EPSG:4326', crs, always_xy=True)
        west, south, east, north_bound = to_geo.transform_bounds(*bounds)
        boundary = box(*bounds).boundary
        tick_locator = MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10])
        for longitude in (True, False):
            low, high = (west, east) if longitude else (south, north_bound)
            for value in tick_locator.tick_values(low, high):
                other = np.linspace(south-.001, north_bound+.001, 40) if longitude else np.linspace(west-.001, east+.001, 40)
                constant = np.full_like(other, value)
                x, y = to_map.transform(constant, other) if longitude else to_map.transform(other, constant)
                line = LineString(zip(x, y))
                for hit in parts(line.intersection(boundary), 'Point'):
                    if longitude and abs(hit.y-y0) < 0.1:
                        ax.annotate(f'{abs(value):g}°{"E" if value >= 0 else "W"}', (hit.x, y0),
                                    xytext=(0,-4), textcoords='offset points', ha='center', va='top', fontsize=9, color=INK)
                    elif not longitude and abs(hit.x-x0) < 0.1:
                        ax.annotate(f'{abs(value):g}°{"N" if value >= 0 else "S"}', (x0, hit.y),
                                    xytext=(-4,0), textcoords='offset points', ha='right', va='center', fontsize=9, color=INK)
                ax.plot(x, y, color='white', linewidth=.55, alpha=.36, zorder=2)


def bbox(bounds, factor=1.1):
    x0, y0, x1, y1 = bounds
    x, y = (x0+x1)/2, (y0+y1)/2
    dx, dy = (x1-x0)*factor/2, (y1-y0)*factor/2
    return x-dx, y-dy, x+dx, y+dy


def display_grid(bounds, width):
    x0, y0, x1, y1 = bounds
    height = max(1, round(width * (y1-y0)/(x1-x0)))
    return dict(transform=from_bounds(x0, y0, x1, y1, width, height), width=width, height=height)


def raster_view(paths, crs, bounds, *, size=800, band=1, units='power', stretch=None):
    """按原始值的物理口径渲染 SAR。dB 值可为负，正线性功率才计算 10 log10。"""
    grid = display_grid(bounds, size)
    height, width = grid['height'], grid['width']
    mosaic = np.full((height, width), np.nan, dtype='float32')
    for path in paths:
        with rasterio.open(path) as src, WarpedVRT(
            src, crs=crs, **grid, dtype='float32', nodata=np.nan, resampling=Resampling.nearest,
        ) as aligned:
            signal = aligned.read(band)
        valid = np.isfinite(signal)
        if units in ('power', 'amplitude'):
            valid &= signal > 0
            signal[valid] = (10 if units == 'power' else 20) * np.log10(signal[valid])
        if not valid.any():
            continue
        low, high = stretch if stretch is not None else np.percentile(signal[valid], [2, 98])
        signal = np.clip((signal-low)/max(high-low, 1e-6), 0, 1)
        take = valid & ~np.isfinite(mosaic)
        mosaic[take] = signal[take]
    return mosaic


def optical_view(tiles, crs, bounds, size=800):
    """同日 TCI 真彩色拼接。用同瓦片 SCL 只保留清晰陆地和水面。"""
    grid = display_grid(bounds, size)
    rgba = np.zeros((grid['height'], grid['width'], 4), dtype='uint8')
    for rgb_path, scl_path in tiles:
        with rasterio.open(scl_path) as src, WarpedVRT(src, crs=crs, **grid, nodata=0,
                                                      resampling=Resampling.nearest) as aligned:
            clear = np.isin(aligned.read(1), (4, 5, 6))
        take = clear & (rgba[..., 3] == 0)
        if not take.any():
            continue
        with rasterio.open(rgb_path) as src, WarpedVRT(src, crs=crs, **grid, nodata=0,
                                                      resampling=Resampling.bilinear) as aligned:
            rgb = aligned.read((1, 2, 3)).transpose(1, 2, 0)
        take &= np.any(rgb != 0, axis=-1)
        rgba[take, :3] = rgb[take]
        rgba[take, 3] = 255
    return rgba


def backdrop(ax, image, bounds, *, z=0):
    if image is not None:
        ax.imshow(image, cmap='gray', vmin=0, vmax=1, zorder=z,
                  extent=(bounds[0], bounds[2], bounds[1], bounds[3]), interpolation='nearest')


def terrain_view(paths, crs, bounds, size=800):
    """制作浅灰晕渲底图，沿用原制图的 DEM 高程与照明参数。"""
    grid = display_grid(bounds, size)
    dem = np.full((grid['height'], grid['width']), np.nan, dtype='float32')
    for path in paths:
        with rasterio.open(path) as src, WarpedVRT(src, crs=crs, **grid, dtype='float32', nodata=np.nan,
                                                  resampling=Resampling.bilinear) as aligned:
            values = aligned.read(1)
        take = np.isfinite(values) & ~np.isfinite(dem)
        dem[take] = values[take]
    relief = LightSource(315, 50).hillshade(np.ma.masked_invalid(dem),
                                           dx=grid['transform'].a, dy=-grid['transform'].e)
    return .66 + .34 * relief


def image_label(ax, message):
    ax.text(.018,.979,message,transform=ax.transAxes,va='top',size=12,color='white',zorder=18,
            bbox={'facecolor': INK, 'edgecolor': 'none', 'pad': 4, 'alpha': .94})


def administrative(ax, roi, towns, labels=(), outlines=None):
    if outlines:
        boundaries = union_all([g.boundary for g, _ in outlines['towns']])
        inner = boundaries.difference(roi.boundary.buffer(1.0))
        lines(ax, [inner], '#FFFFFF', 1.7, 7.3)
        lines(ax, [inner], '#856348', .85, 7.4, '--')
        lines(ax, [roi.boundary], 'white', 2.35, 7.5)
        lines(ax, [roi.boundary], INK, 1.36, 7.6)
    else:
        fill(ax, [g for g, _ in towns], 'none', edge='#A6ACAA', width=.55, z=3)
        fill(ax, [roi], 'none', edge=INK, width=1.2, z=8)
    for geometry, props in towns:
        if props.get('label') not in labels:
            continue
        p = geometry.representative_point()
        ax.text(p.x, p.y, props['label'], ha='center', fontsize=11, color=INK, zorder=10,
                path_effects=[patheffects.withStroke(linewidth=2.4, foreground='white')], clip_on=True)


def reference_outlines(ax, outlines, bounds):
    """按图幅筛选居住用地面与参考水系，描边保留原要素轮廓。"""
    extent = box(*bounds)
    min_area = ((bounds[2] - bounds[0]) * .0025) ** 2
    settlements = [g.boundary for g, _ in outlines.get('settlements', []) if g.area >= min_area and g.intersects(extent)]
    water = [g.boundary for g, _ in outlines.get('reference_water', []) if g.area >= min_area and g.intersects(extent)]
    rivers = [g for g, _ in outlines.get('river_axis', []) if g.intersects(extent)]
    lines(ax, settlements, '#FDF2D8', 1.4, 5.1)
    lines(ax, settlements, '#997338', .65, 5.2)
    lines(ax, water, '#306B8B', .65, 5.15, ':')
    lines(ax, rivers, '#4C7EA0', .55, 3.1)


def sheet(config, index, title, dates, *, portrait=False):
    fig = plt.figure(figsize=(11.69, 16.54) if portrait else (16.54, 11.69), facecolor='white')
    if config.get('basemap_mode') == 'terrain':
        heading_y, detail_y, rule_y = (.976, .946, .921) if portrait else (.965, .909, .867)
        fig.text(.026, heading_y, config['place'], color=INK, size=31, va='top', weight='bold')
        fig.text(.026, detail_y, title, color=INK, size=21, va='top')
        fig.text(.974, heading_y-.005, f'{config["product_id"]}  |  {index:02d}', ha='right', va='top', size=15, color=INK, weight='bold')
        fig.text(.974, detail_y, dates, ha='right', va='top', size=14, color=INK)
        fig.add_artist(Line2D([.026,.974],[rule_y,rule_y],transform=fig.transFigure,color=INK,lw=1))
        return fig
    base_y = .920 if portrait else .881
    fig.add_artist(Rectangle((.019,base_y),.962,.055 if portrait else .094,
                             transform=fig.transFigure,facecolor=INK,edgecolor='none'))
    fig.text(.032, .962 if portrait else .952, f'{config["place"]}  {title}', color='white', size=25, va='top', weight='bold')
    fig.text(.966, .965 if portrait else .953, f'{config["product_id"]}  |  {index:02d}', ha='right', va='top', size=13, color='white', weight='bold')
    fig.text(.966, .942 if portrait else .914, dates, ha='right', va='top', size=13, color='#DBE7EE')
    return fig


def sym(color, title, kind='area'):
    if kind == 'area':
        return Patch(facecolor=color, edgecolor='none', label=title)
    if kind == 'point':
        return Line2D([], [], color=color, marker='s', ls='', markersize=8, label=title)
    return Line2D([], [], color=color, lw=2, ls='--' if kind == 'dash' else '-', label=title)


def sidebar(fig, roi, focus, bounds, optical, optical_label, numbers, legend, caveat, *, portrait=False):
    locator = fig.add_axes([.685,.751,.278,.147] if portrait else [.773,.654,.193,.173])
    backdrop(locator, optical, bounds)
    fill(locator, [roi], 'none' if optical is not None else '#E6EBE9',
         edge='white' if optical is not None else RULE, width=.55, z=3)
    if focus is not None:
        selected = box(*focus) if isinstance(focus, tuple) else focus
        fill(locator, [selected], 'none', edge='#FFBE3A' if optical is not None else '#B13422', width=1.25, z=6)
    locator.set(xlim=(bounds[0], bounds[2]), ylim=(bounds[1], bounds[3]),
                xticks=[], yticks=[], aspect='equal', facecolor='#DFE1DE')
    for spine in locator.spines.values():
        spine.set_color(RULE)
    x = .68 if portrait else .767
    heading_y, metric_y = (.908, .709) if portrait else (.835, .622)
    metric_step = .044 if portrait else .054
    fig.text(x,heading_y,optical_label, size=13.5, weight='bold', color=INK)
    for i, (value, name) in enumerate(numbers):
        y = metric_y-i*metric_step
        fig.text(x,y,value,size=20,weight='bold',color=INK,va='center')
        fig.text(.80 if portrait else .889,y,name,size=13,color=INK,va='center')
    legend_y = metric_y-.011-len(numbers)*metric_step
    fig.text(x,legend_y,'图例',size=15,weight='bold',color=INK,va='top')
    compact = len(legend) >= 11
    fig.legend(handles=legend, loc='upper left', bbox_to_anchor=(x-.007,legend_y-.024),
               frameon=False, fontsize=11.0 if compact and not portrait else 12.5, labelcolor=INK, handlelength=2,
               labelspacing=.32 if compact and not portrait else .54)
    fig.text(x,.119 if portrait else .095,caveat,size=11.5,color=INK,linespacing=1.44,va='bottom')


def save(fig, out, name, source, crs, attribution=None):
    fig.add_artist(Line2D([.026,.974],[.073,.073],transform=fig.transFigure,color=RULE,lw=.65))
    fig.text(.027,.046,source,color=INK,size=11,va='center')
    fig.text(.973,.046,crs,ha='right',color=INK,size=11,va='center')
    if attribution:
        fig.text(.027,.025,attribution,color=INK,size=9,va='center')
    fig.savefig(out / f'{name}.png', dpi=220)
    fig.savefig(out / f'{name}.pdf')
    plt.close(fig)


def thematic(ax, changes, imagery=True):
    for kind, color, _ in WATER:
        geometries = [g for g, p in changes if p['class'] == kind]
        fill(ax, geometries, color, alpha=(.34 if kind == 'stable_water' else .72) if imagery else 1, z=4)
        if imagery and kind != 'stable_water':
            fill(ax, geometries, 'none', edge=color, width=.23, z=4.1)


def infrastructure(ax, exposure_layers, detail=False):
    building_index = 0
    for spec, records in exposure_layers.values():
        if spec['kind'] == 'road':
            for flag, color in [(0, ROAD), (1, BRIDGE)]:
                paths = [g for g, p in records if p['bridge_tunnel'] == flag]
                lines(ax, paths, 'white', 3.0, 6)
                lines(ax, paths, color, 2.1, 7, '--' if flag else '-')
            continue
        groups = [(lambda p: p['overlap_pct'] >= 50, BUILDINGS[0]),
                  (lambda p: p['overlap_pct'] < 50, BUILDINGS[1])] if building_index == 0 else [(lambda p: True, BUILDINGS[2])]
        for select, color in groups:
            geometries = [g for g, p in records if select(p)]
            if detail:
                fill(ax, geometries, color, edge='white', width=.31, z=9)
            else:
                points = [g.representative_point() for g in geometries]
                ax.scatter([p.x for p in points], [p.y for p in points], c=color, marker='s', s=15,
                           edgecolors='white', linewidths=.42, zorder=9)
        building_index += 1


def draw_case(config, out):
    crs = CRS.from_user_input(config['crs'])
    if not crs.is_projected or crs.axis_info[0].unit_conversion_factor != 1:
        raise ValueError('制图及长度统计需要米制投影坐标系。')
    font = config.get('font_file')
    if font:
        font_manager.fontManager.addfont(font)
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=font).get_name()
    plt.rcParams.update({'pdf.fonttype': 42, 'axes.unicode_minus': False})
    out.mkdir(parents=True, exist_ok=True)
    source = config['vectors']
    roi = union_all([g for g, _ in load_vector(source['roi'], crs)])
    towns = load_vector(source['towns'], crs) if source.get('towns') else []
    common = union_all([g for g, _ in load_vector(source['common'], crs)])
    changes = load_vector(source['changes'], crs)
    gain = [g for g, p in changes if p['class'] == 'water_gain']
    streets = config['facilities']
    exposure_layers = exposure(streets, gain, common, crs, out/'facility_exposure.gpkg')
    roads = [(spec, records) for spec, records in exposure_layers.values() if spec['kind'] == 'road']
    buildings = [(spec, records) for spec, records in exposure_layers.values() if spec['kind'] == 'building']
    if len(buildings) > 2 or len(roads) != 1:
        raise ValueError('模板支持一套道路和至多两套建筑底图。')
    _, road_records = roads[0]
    road_km = sum(p['overlap_m'] for _, p in road_records)/1000
    bridges = [p for _, p in road_records if p['bridge_tunnel']]
    summary = {'road_features': len(road_records), 'road_overlap_km': road_km,
               'bridge_tunnel_features': len(bridges), 'bridge_tunnel_km': sum(p['overlap_m'] for p in bridges)/1000,
               'buildings': {spec['label']: {'features': len(records),
                             'at_least_half': sum(p['overlap_pct'] >= 50 for _, p in records)} for spec, records in buildings}}
    (out/'exposure_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

    full = bbox(roi.bounds, 1.10)
    imagery = config.get('basemap_mode', 'imagery') == 'imagery'
    optical = config.get('optical')
    optical_full = optical_view(optical['context']['tiles'], crs, full, size=1900) if imagery else None
    terrain = None if imagery else terrain_view(config['dem_files'], crs, full)
    locator_label = optical['context']['label'] + '定位图' if imagery else '分析位置'
    label_settings = config.get('labels', {})
    catalog = read_labels(label_settings.get('sources', []), crs, full)
    boundary_settings = config.get('boundaries', {})
    outlines = collect_outlines(boundary_settings['sources'], crs, roi, towns, catalog) if boundary_settings else None
    attribution = boundary_settings.get('attribution')
    if outlines:
        export_outlines(out/'base_outlines.gpkg', crs, outlines)
    outline_legend = ([sym('#856348','乡镇界','dash'), sym('#997338','居住用地轮廓','line'),
                       sym('#306B8B','参考水系／岸线','dash')] if outlines else [])
    label_audit = {}
    observations = config['observations']
    if len(observations) != 2 or any(not 1 <= len(day['files']) <= 4 for day in observations):
        raise ValueError('快视图要求两个时相，每期一至四景。')
    sar_units = config.get('observation_units', 'power')
    late_files = observations[1]['files']
    late_view = raster_view(late_files, crs, full, size=2200, units=sar_units) if imagery else None
    late_coverage = load_vector(source['post_coverage'], crs) if imagery and source.get('post_coverage') else []
    selected_labels = [] if catalog else config.get('town_labels', [])
    date_main = config['main_dates']
    captions = config['sources']
    water_legend = [sym(c, title) for _, c, title in WATER]

    fig = sheet(config, 1, '两期 SAR 水面变化', date_main)
    ax = fig.add_axes([.068,.137,.674,.702] if imagery else [.026,.137,.716,.702])
    backdrop(ax, optical_full if imagery else terrain, full)
    backdrop(ax, late_view, full, z=1)
    fill(ax, [roi.difference(common)], MISSING, alpha=.33)
    administrative(ax, roi, towns, selected_labels, outlines)
    thematic(ax, changes, imagery)
    if outlines:
        reference_outlines(ax, outlines, full)
    lines(ax, [g.boundary for g, _ in late_coverage], '#F3F7F8', .94, 8, '--')
    frame(ax, full, crs=crs if imagery else None)
    if imagery:
        image_label(ax, config['image_labels']['main'])
    label_audit['01_water_change'] = label_map(ax, catalog, full, 31, priority=label_settings.get('overview', []))
    areas = {key: sum(g.area for g, p in changes if p['class'] == key)/1e6 for key, _, _ in WATER}
    sidebar(fig, roi, common.simplify(450), full, optical_full, locator_label, [(f'{common.area/1e6:,.2f}', '分析区 km²'),
            (f'{areas["water_gain"]:.2f}', '新增 km²'), (f'{areas["water_loss"]:.2f}', '减少 km²')],
            water_legend + [sym(MISSING,'未共同观测'), sym(INK,'区界','line')] + outline_legend +
            ([sym('#819AA4','后期 SAR 覆盖边界','dash')] if imagery else []),
            '新增包括库岸扩张和陆域积水候选。\n遥感解译结果供现场核查。')
    save(fig, out, '01_water_change', captions['water'], crs.name, attribution)

    fig = sheet(config, 2, 'SAR 影像覆盖与分析范围', date_main)
    for row, day in enumerate(observations):
        frames = [[filename] for filename in day['files']] + [day['files']]
        for column, files in enumerate(frames):
            y = .486 - row*.335
            width = .706/len(frames)
            ax = fig.add_axes([.028 + column*width, y,width-.004,.276])
            data = raster_view(files, crs, full, size=480, units=sar_units)
            ax.imshow(data, cmap='gray', vmin=0, vmax=1, extent=(full[0],full[2],full[1],full[3]))
            merged = column == len(frames)-1
            administrative(ax, roi, [] if not merged else towns, outlines=outlines if merged else None)
            if outlines and merged:
                reference_outlines(ax, outlines, full)
            frame(ax, full, north=column == len(frames)-1, small=True)
            scene_label = f'分景 {column+1}'
            if column == len(frames)-1:
                scene_label = '同日快视拼接'
                label_audit[f'02_{day["date"]}'] = label_map(ax, catalog, full, 5, small=True,
                                                             priority=label_settings.get('coverage', []))
            elif day.get('scene_times'):
                scene_label = day['scene_times'][column]
            ax.set_title(scene_label, fontsize=12, color=INK, pad=4)
        fig.text(.029,.816-row*.335,day['date'],color=INK,weight='bold',size=16)
    sidebar(fig, roi, common.simplify(450), full, optical_full, locator_label, [(str(len(day['files'])), f'{day["date"][5:]} 影像景数') for day in observations] +
            [(f'{common.area/1e6:,.2f}', '共同有效 km²')],
            [sym('#636363','SAR 灰度快视'), sym('#F5F4EF','未显示有效影像'), sym(INK,'区界','line')] + outline_legend,
            ('SAR 灰度逐景拉伸。\n侧栏黄线标出共同有效区。' if imagery else
             'SAR 灰度逐景拉伸。\n侧栏红线标出共同有效区。') + '\n同日拼接用于展示观测位置。')
    save(fig, out, '02_observation', captions['coverage'], crs.name, attribution)

    auxiliary = load_vector(source['aux_changes'], crs)
    patch = load_vector(source['aux_review'], crs) if source.get('aux_review') else []
    aux_extent = bbox(union_all([g for g, _ in auxiliary]).bounds, 1.22)
    fig = sheet(config, 3, config['aux_title'], config['aux_dates'])
    ax = fig.add_axes([.065,.138,.677,.700] if imagery else [.026,.138,.716,.700])
    if imagery:
        sar = config['aux_sar']
        bg = raster_view([sar['path']], crs, aux_extent, size=1950, band=sar['band'],
                         units=sar['units'], stretch=sar.get('stretch'))
    else:
        bg = terrain_view(config['dem_files'], crs, aux_extent)
    backdrop(ax, bg, aux_extent)
    administrative(ax, roi, towns, selected_labels, outlines)
    thematic(ax, auxiliary, imagery)
    if outlines:
        reference_outlines(ax, outlines, aux_extent)
    component = [g for g, p in auxiliary if p['class'] == 'component_change_only']
    if component:
        fill(ax, component, '#526675', z=5)
    if patch:
        fill(ax, [g for g, _ in patch], 'none', edge='#FF5A47', width=1.9, z=11)
    frame(ax, aux_extent, crs=crs if imagery else None)
    label_audit['03_auxiliary'] = label_map(ax, catalog, aux_extent, 28, priority=label_settings.get('aux', []))
    if imagery:
        image_label(ax, sar['label'])
        review_extent = bbox(union_all([g for g, _ in patch]).bounds, 2.9) if patch else aux_extent
        clear_review = optical_view(optical['review']['tiles'], crs, review_extent, size=570)
        sidebar_args = (union_all([g for g, _ in patch]) if patch else None, review_extent, clear_review,
                        optical['review']['label'] + '复核')
    else:
        sidebar_args = (aux_extent, full, None, locator_label)
    sidebar(fig, roi, *sidebar_args, config['aux_numbers'], water_legend + outline_legend +
            ([sym('#526675','主库连通归属改变')] if component else []) +
            ([sym('#FF5A47','人工复核区边界','line')] if patch else []), config['aux_note'])
    save(fig, out, '03_auxiliary', captions['aux'], crs.name, attribution)

    facility_print = config.get('facility_print')
    portrait = bool(facility_print and facility_print['height_mm'] > facility_print['width_mm'])
    fig = sheet(config, 4, '道路与建筑潜在暴露', date_main, portrait=portrait)
    if portrait:
        overview = fig.add_axes([.033,.504,.605,.395])
        detail = fig.add_axes([.033,.103,.605,.355] if not imagery else [.033,.195,.294,.237])
    else:
        overview = fig.add_axes([.064,.14,.423,.700] if imagery else [.026,.14,.46,.700])
        detail = fig.add_axes([.508,.491,.235,.344] if imagery else [.505,.31,.235,.415])
    focus = tuple(config['exposure_detail_bounds'])
    detail_source = boundary_settings.get('detail_optical')
    if imagery:
        sar_detail = fig.add_axes([.344,.195,.294,.237] if portrait else [.508,.119,.235,.344])
        optical_focus = optical_view(optical['context']['tiles'], crs, focus, size=650)
        sar_focus = raster_view(late_files, crs, focus, size=650, units=sar_units)
        panels = [(overview, full, optical_full), (detail, focus, optical_focus), (sar_detail, focus, sar_focus)]
    else:
        detail_image = optical_view(detail_source['tiles'], crs, focus, size=710) if detail_source else None
        panels = [(overview, full, terrain), (detail, focus, detail_image)]
    reference_roads = [g for g, _ in load_vector(roads[0][0], crs)]
    for ax, bounds, raster in panels:
        backdrop(ax, raster, bounds)
        fill(ax, [roi.difference(common)], MISSING, alpha=.34)
        administrative(ax, roi, towns, selected_labels if ax is overview else (), outlines)
        thematic(ax, changes, imagery or (ax is detail and detail_source is not None))
        if outlines:
            reference_outlines(ax, outlines, bounds)
        lines(ax, reference_roads, '#BCC3BB', .28 if ax is overview else .7, 5)
        infrastructure(ax, exposure_layers, detail=ax is not overview)
        frame(ax, bounds, north=ax is overview, small=ax is not overview, crs=crs if imagery else None)
    label_audit['04_overview'] = label_map(overview, catalog, full, 18, priority=label_settings.get('overview', []))
    label_audit['04_detail'] = label_map(detail, catalog, focus, 9, priority=label_settings.get('detail', []))
    local_buildings = [g for _, records in buildings for g, _ in records if g.intersects(box(*focus))]
    if imagery and local_buildings:
        position = union_all(local_buildings).centroid
        detail.annotate('候选建筑', xy=(position.x, position.y), xytext=(.97,.75),
                                textcoords='axes fraction', ha='right', va='center', size=10,
                                color='white', zorder=19, arrowprops={'arrowstyle': '->', 'color': 'white'},
                                bbox={'facecolor': INK, 'edgecolor': 'none', 'pad': 3})
    fill(overview, [box(*focus)], 'none', edge=INK, width=1.1, z=11)
    overview.annotate('B', xy=(focus[2], focus[3]), xytext=(7, 8), textcoords='offset points',
                      size=12, weight='bold', color=INK, zorder=12,
                      path_effects=[patheffects.withStroke(linewidth=2.5, foreground='white')])
    if imagery:
        image_label(overview, 'A  ' + optical['context']['label'] + '全区总览')
        image_label(detail, 'B  ' + optical['context']['label'] + '局部')
        image_label(sar_detail, 'C  ' + config['image_labels']['late_sar'] + '同范围')
    else:
        overview.set_title('A  分析区内设施候选', color=INK, fontsize=15, pad=8, loc='left')
        detail.set_title('B  道路相交段与建筑轮廓', color=INK, fontsize=15, pad=8, loc='left')
        if detail_source:
            detail.set_xlabel(detail_source['label'], color=INK, fontsize=11, labelpad=7)
    figures = [(f'{road_km:.2f}', '道路相交 km')] + [(str(len(rows)), f'{spec["label"]} 建筑轮廓') for spec, rows in buildings]
    facility_legend = [sym(ROAD,'道路与新增水面相交段','line'), sym(BRIDGE,'桥隧标记路段','dash')]
    if buildings:
        name = buildings[0][0]['label']
        facility_legend += [sym(BUILDINGS[0], f'{name} 建筑 ≥50%','point'), sym(BUILDINGS[1], f'{name} 建筑 <50%','point')]
    if len(buildings) > 1:
        facility_legend.append(sym(BUILDINGS[2], f'{buildings[1][0]["label"]} 建筑相交','point'))
    facility_legend.extend([sym('#747A79','参考道路','line'), sym(GAIN,'新增水面'), sym(STABLE,'稳定水面'), sym(LOSS,'减少水面'), sym(MISSING,'未共同观测')])
    facility_legend.extend(outline_legend)
    sidebar(fig, roi, focus, full, optical_full, locator_label, figures, facility_legend,
            '按空间重叠筛查设施。\n桥隧、受淹和通行情况需核查。', portrait=portrait)
    save(fig, out, '04_facility_exposure', captions['facilities'], crs.name, attribution)
    if facility_print:
        sheet_path = out / '04_facility_exposure'
        fit_print_map(sheet_path, sheet_path, **facility_print)
    if catalog:
        (out/'map_labels.json').write_text(json.dumps(label_audit, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary


def load_case(path):
    def resolve(value):
        return str((path.parent / value).resolve())

    config = json.loads(path.read_text(encoding='utf-8'))
    for spec in [*config['vectors'].values(), *config['facilities']]:
        spec['path'] = resolve(spec['path'])
    for day in config['observations']:
        pattern = Path(resolve(day['glob'])) if day.get('glob') else None
        day['files'] = ([str(p) for p in sorted(pattern.parent.parent.glob(f'{pattern.parent.name}/{pattern.name}'))]
                        if pattern else [resolve(p) for p in day['files']])
        if day.get('utc_offset') is not None:
            timecodes = [re.search(r'\d{8}T\d{6}', Path(f).name).group() for f in day['files']]
            day['scene_times'] = [(datetime.strptime(stamp,'%Y%m%dT%H%M%S')+timedelta(hours=day['utc_offset'])).strftime('%H:%M:%S')
                                  for stamp in timecodes]
    for spec in config.get('labels', {}).get('sources', []):
        spec['path'] = resolve(spec['path'])
    if config.get('basemap_mode') == 'terrain':
        config['dem_files'] = sorted(Path(resolve(config['dem_folder'])).glob('*.tif'))
        if not config['dem_files']:
            raise ValueError('地形模式缺少 DEM 栅格。')
    boundary_settings = config.get('boundaries', {})
    for spec in boundary_settings.get('sources', {}).values():
        spec['path'] = resolve(spec['path'])
    optical_specs = list(config.get('optical', {}).values())
    if boundary_settings.get('detail_optical'):
        optical_specs.append(boundary_settings['detail_optical'])
    for optical in optical_specs:
        if optical.get('glob'):
            tci = sorted(Path(file) for file in glob(resolve(optical['glob'])))
            optical['tiles'] = [(str(file), str(file.with_name('SCL.tif'))) for file in tci]
        else:
            optical['tiles'] = [(resolve(rgb), resolve(scl)) for rgb, scl in optical['tiles']]
        if not optical['tiles'] or any(not Path(scl).is_file() for _, scl in optical['tiles']):
            raise ValueError(f'{optical["label"]} 缺少 S2 真彩色瓦片或配套 SCL 分类。')
    if config.get('aux_sar'):
        config['aux_sar']['path'] = resolve(config['aux_sar']['path'])
    if config.get('font_file'):
        config['font_file'] = resolve(config['font_file'])
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    print(json.dumps(draw_case(load_case(args.case.resolve()), args.output), ensure_ascii=False, indent=2))
