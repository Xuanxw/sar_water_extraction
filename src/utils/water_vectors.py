"""将地理参考水体栅格面化；直接读取GDAL波段，不建立整景NumPy数组。"""

import fiona
import rasterio
from rasterio.features import shapes
from shapely import make_valid
from shapely.geometry import shape, mapping


def polygon_parts(geometry):
    """递归保留有效Polygon部分，忽略修复后可能出现的线和点。"""
    if geometry.geom_type == 'Polygon':
        if not geometry.is_empty:
            yield geometry
    elif hasattr(geometry, 'geoms'):
        for part in geometry.geoms:
            yield from polygon_parts(part)


def write_water_polygons(mask_path, destination, log=print, *, band_index=1, connectivity=8,
                         selected_value=1):
    """将指定分类值输出为SHP面，保持像元边界、孔洞和源CRS，不平滑。

    GDAL按完整波段追踪连通轮廓，避免逐窗口面化造成拼缝；每500面写入一次。
    band_index 从 1 开始，默认提取类别值 1。自接触轮廓经 make_valid 修复后只保留面。
    """
    count = 0
    with rasterio.open(mask_path) as source:
        schema = dict(geometry='Polygon', properties={'id': 'int', 'water': 'int'})
        with fiona.open(destination, 'w', driver='ESRI Shapefile', schema=schema,
                        crs_wkt=source.crs.to_wkt(), encoding='UTF-8') as target:
            batch = []
            for geometry, value in shapes(rasterio.band(source, band_index), connectivity=connectivity,
                                          transform=source.transform):
                if value != selected_value:
                    continue
                polygon = shape(geometry)
                if not polygon.is_valid:
                    polygon = make_valid(polygon)
                for part in polygon_parts(polygon):
                    count += 1
                    batch.append(dict(geometry=mapping(part), properties=dict(id=count, water=1)))
                if len(batch) >= 500:
                    target.writerecords(batch)
                    batch.clear()
                    log(f'已输出{count}个水体面')
            if batch:
                target.writerecords(batch)
    log(f'水体面化完成：{count}个面')
    return count
