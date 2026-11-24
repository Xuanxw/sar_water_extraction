"""读取专题地图的行政界与参考要素，输出用于制图的独立矢量图层。"""

from collections import defaultdict

import fiona
from pyproj import Transformer
from shapely import make_valid, transform, union_all
from shapely.geometry import MultiLineString, MultiPolygon, box, mapping, shape
from shapely.strtree import STRtree


def parts(geometry, kind):
    """从多部件或几何集合中保留所需的点、线或面。"""
    if geometry.geom_type == kind and not geometry.is_empty:
        yield geometry
    elif hasattr(geometry, 'geoms'):
        for item in geometry.geoms:
            yield from parts(item, kind)


def read_context(spec, crs, bounds):
    """按米制地图范围查找参考要素，并投影到制图坐标系。"""
    with fiona.open(spec['path'], layer=spec['layer']) as source:
        filter_bounds = Transformer.from_crs(crs, source.crs, always_xy=True).transform_bounds(*bounds)
        project = Transformer.from_crs(source.crs, crs, always_xy=True).transform
        for f in source.filter(bbox=filter_bounds):
            attrs = f.properties
            if not f.geometry or (spec.get('classes') and attrs.get('fclass') not in spec['classes']):
                continue
            geom = transform(shape(f.geometry), project, interleaved=False)
            yield geom, {'name': attrs.get('name') or '', 'source_id': str(attrs.get('osm_id') or f.id),
                         'source': spec['source']}


def collect_outlines(specs, crs, roi, towns, labels):
    """整理区界、乡镇范围和参考线面。居民点只在唯一包含关系成立时关联居住用地。"""
    town_parts = defaultdict(list)
    for geometry, attrs in towns:
        town_parts[attrs.get('label') or ''].append(geometry)
    outputs = {
        'district': [(roi, {'name': '', 'source_id': '', 'source': '研究区行政界'})],
        'towns': [(union_all(geoms), {'name': name, 'source_id': '', 'source': '项目乡镇界'})
                  for name, geoms in sorted(town_parts.items())],
    }
    village_names = [(name, geometry) for name, kind, geometry in labels if kind == 'village']
    village_index = STRtree([geom for _, geom in village_names])
    for key, spec in specs.items():
        selected = []
        kind = 'LineString' if key == 'river_axis' else 'Polygon'
        constructor = MultiLineString if kind == 'LineString' else MultiPolygon
        for geometry, attrs in read_context(spec, crs, roi.bounds):
            if not geometry.is_valid:
                geometry = make_valid(geometry)
            if not geometry.intersects(roi):
                continue
            clipped = list(parts(geometry.intersection(roi), kind))
            if not clipped:
                continue
            if key == 'settlements' and not attrs['name']:
                matches = {village_names[index][0] for index in village_index.query(geometry, predicate='covers')}
                if len(matches) == 1:
                    attrs['name'] = matches.pop()
            selected.append((constructor(clipped), attrs))
        outputs[key] = selected
    return outputs


def export_outlines(destination, crs, outlines):
    """写出独立的区界、乡镇面、居民用地面、参考水面和河流线，继承米制投影。"""
    for name, rows in outlines.items():
        kind = 'LineString' if name == 'river_axis' else 'Polygon'
        constructor = MultiLineString if kind == 'LineString' else MultiPolygon
        schema = {'geometry': f'Multi{kind}', 'properties': {'name': 'str', 'source_id': 'str', 'source': 'str'}}
        with fiona.open(destination, 'w', driver='GPKG', layer=name, schema=schema, crs_wkt=crs.to_wkt()) as writer:
            writer.writerecords({'geometry': mapping(constructor(list(parts(g, kind)))), 'properties': props}
                                for g, props in rows)
