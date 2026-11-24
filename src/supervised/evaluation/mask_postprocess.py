"""水体掩膜的局部开闭运算；标签不参与处理，NoData保持未知。"""

from contextlib import ExitStack

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from scipy import ndimage

from thres_segmentation.water_analysis import pixel_areas
from utils.raster import windows


def clean_mask(raw, valid, opening_radius=0, closing_radius=0):
    """圆形开闭运算；半径单位像元，邻域含无效像元时保留raw，不跨NoData补水。"""
    result = raw.copy()
    for radius, operation in ((opening_radius, ndimage.binary_opening),
                              (closing_radius, ndimage.binary_closing)):
        if radius:
            y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
            footprint = x*x+y*y <= radius*radius
            changed = operation(result, structure=footprint)
            safe = ndimage.binary_erosion(valid, structure=footprint, iterations=2, border_value=0)
            result[safe] = changed[safe]
    return result & valid


def postprocess_water_raster(source_path, destination, *, opening_radius=0, closing_radius=0,
                             exclusion_masks=(), block_size=1024):
    """带halo分块修正拼接后的0/1/255掩膜，并按原生像元计数、计算平方米面积。

    可选排除图中1=明确排除，0=不排除，255/NoData=未知。以最近邻对齐，不将
    未知当排除，也不自动下载辅助数据。形态学后再次施加排除，避免将其填回。
    """
    if any(not isinstance(r, int) or r < 0 for r in (opening_radius, closing_radius)):
        raise ValueError('开闭运算半径应为非负整数像元')
    margin = 2 * (opening_radius + closing_radius)
    report = dict(valid_pixels=0, nodata_pixels=0, raw_water_pixels=0, water_pixels=0,
                  added_pixels=0, removed_pixels=0, valid_area_m2=0.0, water_area_m2=0.0)
    with ExitStack() as stack:
        source = stack.enter_context(rasterio.open(source_path))
        target = stack.enter_context(rasterio.open(destination, 'w', **source.profile))
        readers = []
        for path in exclusion_masks:
            auxiliary = stack.enter_context(rasterio.open(path))
            if auxiliary.crs is None:
                raise ValueError(f'排除图缺少坐标系：{path}')
            readers.append(stack.enter_context(WarpedVRT(
                auxiliary, crs=source.crs, transform=source.transform,
                width=source.width, height=source.height, dtype='float32',
                nodata=np.nan, resampling=Resampling.nearest)))
        for window in windows(source, block_size):
            left, top = int(window.col_off), int(window.row_off)
            width, height = int(window.width), int(window.height)
            x0, y0 = max(0, left-margin), max(0, top-margin)
            x1, y1 = min(source.width, left+width+margin), min(source.height, top+height+margin)
            extended = Window(x0, y0, x1-x0, y1-y0)
            raw = source.read(1, window=extended)
            valid = raw != 255
            excluded = np.zeros(raw.shape, bool)
            for reader in readers:
                values = reader.read(1, window=extended, masked=True)
                known = ~np.ma.getmaskarray(values) & np.isfinite(values.data)
                if not np.isin(values.data[known], (0, 1, 255)).all():
                    raise ValueError('排除掩膜应为0/1/255，不能直接输入DEM或土地覆盖类别')
                excluded |= known & (values.data == 1)
            corrected = clean_mask((raw == 1) & ~excluded, valid, opening_radius, closing_radius)
            corrected &= ~excluded
            core = np.s_[top-y0:top-y0+height, left-x0:left-x0+width]
            raw, valid, corrected = raw[core], valid[core], corrected[core]
            target.write(np.where(valid, corrected, 255).astype('uint8'), 1, window=window)
            area = pixel_areas(source.crs, source.window_transform(window), valid.shape)
            report['valid_pixels'] += int(valid.sum())
            report['nodata_pixels'] += int((~valid).sum())
            report['raw_water_pixels'] += int((raw == 1).sum())
            report['water_pixels'] += int(corrected.sum())
            report['added_pixels'] += int((corrected & (raw == 0)).sum())
            report['removed_pixels'] += int(((raw == 1) & ~corrected).sum())
            report['valid_area_m2'] += float(np.sum(valid * area))
            report['water_area_m2'] += float(np.sum(corrected * area))
        target.update_tags(**source.tags(), postprocessing='opening_closing_then_explicit_exclusion')
    return report
