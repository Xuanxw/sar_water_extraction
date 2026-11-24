"""读取本地地名和线要素，并按比例尺和可用空间显示注记。"""

from collections import defaultdict
import re

import fiona
from matplotlib import patheffects
from matplotlib.font_manager import FontProperties
from matplotlib.transforms import Bbox
from pyproj import Transformer
from shapely import transform
from shapely.geometry import box, shape
from shapely.ops import linemerge


def read_labels(specs, crs, bounds):
    """用数据源实际几何定位注记，同名线段汇总后选择图框内最长的一段。"""
    catalog = []
    for spec in specs:
        data = defaultdict(list)
        with fiona.open(spec['path'], layer=spec['layer']) as features:
            geo_bounds = Transformer.from_crs(crs, features.crs, always_xy=True).transform_bounds(*bounds)
            project = Transformer.from_crs(features.crs, crs, always_xy=True).transform
            for item in features.filter(bbox=geo_bounds):
                attrs = item.properties
                cls = attrs.get('fclass')
                kind = spec['kind']
                if not item.geometry or (spec.get('classes') and cls not in spec['classes']):
                    continue
                road_name = (attrs.get('name') or '').strip()
                name = road_name
                ref = (attrs.get('ref') or '').split(';')[0].strip()
                if kind == 'road' and cls in ('motorway', 'trunk', 'primary') and re.fullmatch(r'[GSXY]\d{2,4}', ref):
                    name = ref
                elif not re.search(r'[\u3400-\u9fff]', name):
                    name = ref if kind == 'road' and re.fullmatch(r'[GSXY]\d{2,4}', ref) else ''
                if not name:
                    continue
                geom = transform(shape(item.geometry), project, interleaved=False)
                if kind == 'place':
                    category = 'major' if cls in ('city', 'town', 'suburb') else 'village'
                    catalog.append((name, category, geom))
                elif kind == 'road' or geom.geom_type in ('LineString', 'MultiLineString'):
                    category = 'route' if re.fullmatch(r'[GSXY]\d{2,4}', name) else kind
                    for line in geom.geoms if geom.geom_type == 'MultiLineString' else (geom,):
                        if line.geom_type == 'LineString' and not line.is_empty:
                            data[(name, category)].append(line)
                            if category == 'route' and re.search(r'[\u3400-\u9fff]', road_name):
                                data[(f'{road_name}（{name}）', 'road_alias')].append(line)
                else:
                    catalog.append((name, kind, geom))
        for (name, category), geoms in data.items():
            # OSM 道路已分段，最长的可见路段优先。
            catalog.append((name, category, linemerge(geoms)))
    return catalog


STYLES = {
    'major': (0, 14, '#243A4D', 'bold'), 'water': (1, 13, '#125D89', 'bold'),
    'route': (2, 11, '#77491A', 'bold'), 'village': (3, 11, '#2D3C48', 'normal'),
    'road': (4, 10, '#714A2A', 'normal'),
    'road_alias': (4, 11, '#714A2A', 'bold'),
}


def label_map(ax, catalog, bounds, limit, *, small=False, priority=()):
    """图内候选按类别及优先名称排序，在屏幕坐标检查碰撞和裁边。"""
    extent = box(*bounds)
    ax.apply_aspect()
    renderer = ax.figure.canvas.get_renderer()
    frame = ax.get_window_extent()
    occupied = [Bbox.from_extents(frame.x0, frame.y0, frame.x0+frame.width*.27, frame.y0+frame.height*.093),
                Bbox.from_extents(frame.x0+frame.width*.88, frame.y0+frame.height*.88, frame.x1, frame.y1)]
    priority_rank = {text: index - len(priority) for index, text in enumerate(priority)}
    candidates = []
    for text, kind, geometry in catalog:
        if kind == 'road_alias' and text not in priority:
            continue
        if not geometry.intersects(extent):
            continue
        clipped = geometry.intersection(extent)
        if clipped.is_empty:
            continue
        order, size, color, weight = STYLES[kind]
        if clipped.geom_type in ('LineString', 'MultiLineString'):
            line = max(clipped.geoms, key=lambda g: g.length) if clipped.geom_type == 'MultiLineString' else clipped
            if line.length < (bounds[2]-bounds[0])*.04 and text not in priority:
                continue
            point = line.interpolate(.5, normalized=True)
        else:
            point = clipped.representative_point()
        x, y = ax.transData.transform((point.x, point.y))
        rank = (priority_rank.get(text, order), point.distance(extent.centroid))
        candidates.append((rank, text, kind, point, x, y, size, color, weight))

    names = set()
    for _, text, kind, point, x, y, size, color, weight in sorted(candidates, key=lambda row: row[0]):
        if text in names or (kind == 'route' and any(s.endswith(f'（{text}）') for s in names)):
            continue
        size = size * (.78 if small else 1)
        font = FontProperties(size=size, weight=weight)
        width, height, _ = renderer.get_text_width_height_descent(text, font, ismath=False)
        for dx, dy in ((0,4), (0,-10), (12,3), (-12,3), (0,17), (0,-20)):
            px, py = renderer.points_to_pixels(dx), renderer.points_to_pixels(dy)
            bounds_px = Bbox.from_bounds(x+px-width/2-4, y+py-height/2-4, width+8, height+8)
            if (not frame.contains(*bounds_px.min) or not frame.contains(*bounds_px.max)
                    or any(bounds_px.overlaps(other) for other in occupied)):
                continue
            ax.annotate(text, (point.x, point.y), xytext=(dx,dy), textcoords='offset points',
                        ha='center', va='center', color=color, fontsize=size, weight=weight,
                        path_effects=[patheffects.withStroke(linewidth=2.6, foreground='white')], zorder=19)
            if kind in ('major','village'):
                ax.scatter([point.x], [point.y], s=9 if kind == 'major' else 5, c=color,
                           edgecolors='white', linewidths=.4, zorder=18)
            occupied.append(bounds_px)
            names.add(text)
            break
        if len(names) == limit:
            break
    return sorted(names)
