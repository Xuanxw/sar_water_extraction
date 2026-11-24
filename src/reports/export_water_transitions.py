"""导出两期水面变化的原网格分类图、WGS84 显示图及分类面。"""

from contextlib import ExitStack
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from xml.etree.ElementTree import Element, ElementTree, SubElement, indent

import fiona
import numpy as np
from pyproj import CRS
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from shapely.geometry import shape

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.water_vectors import write_water_polygons


NODATA = 255
STABLE_DRY = 3
WATER_CLASSES = (
    (2, 'stable_water', '稳定水体'),
    (1, 'water_gain', '水面新增'),
    (4, 'water_loss', '水面减少'),
)
# 类别值与 RGB 颜色沿用密云 S1 AI4G 后处理的 water_transitions.tif。
PALETTE = {
    1: (13, 150, 75, 255),
    2: (230, 55, 140, 255),
    STABLE_DRY: (236, 146, 27, 255),
    4: (112, 102, 158, 255),
    NODATA: (0, 0, 0, 0),
}
CLASS_LABELS = {1: '非水面 → 水面', 2: '水面 → 水面', 3: '非水面 → 非水面', 4: '水面 → 非水面'}


def write_transition_raster(destination, transition_path, pre_mask, post_mask):
    """写出带色表的单波段分类栅格，保留原变化网格或将两期掩膜对齐到后期网格。"""
    with ExitStack() as stack:
        current = stack.enter_context(rasterio.open(transition_path or post_mask))
        if not current.crs:
            raise ValueError(f'输入缺少坐标系：{current.name}')

        if transition_path:
            # 用波段名称定位类别，因此稳定非水位于第几波段不会影响输出。
            names = [description.split('|', 1)[0].strip() if description else ''
                     for description in current.descriptions]
            if any(names.count(key) != 1 for _, key, _ in WATER_CLASSES):
                raise ValueError('输入变化栅格须包含 stable_water、water_gain 和 water_loss 波段。')
            source_bands = [names.index(key) + 1 for _, key, _ in WATER_CLASSES]
        else:
            previous = stack.enter_context(rasterio.open(pre_mask))
            if not previous.crs:
                raise ValueError(f'输入缺少坐标系：{previous.name}')
            if previous.nodata in (0, 1) or current.nodata in (0, 1):
                raise ValueError('两期水体掩膜须以 0 为非水、1 为水体，NoData 使用独立值。')
            # 离散分类用最近邻。前期影像范围以外按 255 填充，不进入变化类别。
            previous_aligned = stack.enter_context(WarpedVRT(
                previous, crs=current.crs, transform=current.transform,
                width=current.width, height=current.height, dtype='float32', nodata=NODATA,
                resampling=Resampling.nearest,
            ))

        with rasterio.open(
            destination, 'w', driver='GTiff', width=current.width, height=current.height,
            count=1, dtype='uint8', crs=current.crs, transform=current.transform,
            nodata=NODATA, compress='deflate',
            tiled=True, blockxsize=256, blockysize=256,
        ) as target:
            target.write_colormap(1, PALETTE)
            target.set_band_description(1, '水面变化类别')
            target.update_tags(classes='1=0to1,2=1to1,3=0to0,4=1to0,255=unknown or outside selection')

            for _, window in target.block_windows(1):
                if transition_path:
                    values = current.read(source_bands, window=window, masked=True)
                    common = np.all(((values == 0) | (values == 1)).filled(False), axis=0)
                    values = (values == 1).filled(False)
                    if np.any(common & (np.count_nonzero(values, axis=0) > 1)):
                        raise ValueError('输入变化栅格的分类波段在共同有效区内存在重叠。')
                else:
                    pre = previous_aligned.read(1, window=window, masked=True)
                    post = current.read(1, window=window, masked=True)
                    pre_dry, pre_water = (pre == 0).filled(False), (pre == 1).filled(False)
                    post_dry, post_water = (post == 0).filled(False), (post == 1).filled(False)
                    common = (pre_dry | pre_water) & (post_dry | post_water)
                    values = np.stack((pre_water & post_water, pre_dry & post_water, pre_water & post_dry))
                # 共同有效区内没有命中三个水面类别的像元，两期均为非水。
                classified = np.where(common, STABLE_DRY, NODATA).astype(np.uint8)
                for (code, _, _), selected in zip(WATER_CLASSES, values):
                    classified[common & selected] = code
                target.write(classified, 1, window=window)


def write_qgis_palette(destination):
    """输出四类固定颜色的 QML，恢复正常混合和无灰度滤镜的显示。"""
    qgis = Element('qgis', version='3.40', styleCategories='Symbology')
    pipe = SubElement(qgis, 'pipe')
    renderer = SubElement(pipe, 'rasterrenderer', type='paletted', band='1', alphaBand='-1', opacity='1')
    palette = SubElement(renderer, 'colorPalette')
    for code, label in CLASS_LABELS.items():
        red, green, blue, alpha = PALETTE[code]
        SubElement(palette, 'paletteEntry', value=str(code), label=label,
                   color=f'#{red:02x}{green:02x}{blue:02x}', alpha=str(alpha))
    SubElement(pipe, 'brightnesscontrast', contrast='0', brightness='0', gamma='1')
    SubElement(pipe, 'huesaturation', grayscaleMode='0', saturation='0', colorizeOn='0', invertColors='0')
    SubElement(qgis, 'blendMode').text = '0'
    indent(qgis)
    ElementTree(qgis).write(destination, encoding='utf-8', xml_declaration=True)


def write_transition_display(classified_path, output_dir):
    """将原网格分类图转为经纬度显示图及 QML。面积统计仍使用原网格。"""
    name = classified_path.stem.removesuffix('_palette') + '_display_4326.tif'
    display = output_dir / name
    with rasterio.open(classified_path) as source, WarpedVRT(
        source, crs='EPSG:4326', src_nodata=NODATA, nodata=NODATA, resampling=Resampling.nearest,
    ) as view:
        profile = source.profile.copy()
        profile.update(crs=view.crs, transform=view.transform, width=view.width, height=view.height)
        with rasterio.open(display, 'w', **profile) as target:
            # 按显示网格分块读取。最近邻保留原分类码，覆盖范围外填 255。
            for _, window in target.block_windows(1):
                target.write(view.read(1, window=window), 1, window=window)
            target.write_colormap(1, source.colormap(1))
            target.set_band_description(1, source.descriptions[0])
            target.update_tags(**source.tags())
            target.update_tags(purpose='WGS84 display; use the original grid for area statistics',
                               resampling='nearest', source_raster=classified_path.name)
    write_qgis_palette(display.with_suffix('.qml'))
    return display


def rebuild_transition_palette(transition_path, destination):
    """从变化波段重建分类图并导出显示图。传入 None 时直接使用已有分类图。"""
    destination = Path(destination)
    if transition_path is not None and Path(transition_path).resolve() == destination.resolve():
        raise ValueError('源变化栅格与目标分类栅格须为不同文件。')
    with TemporaryDirectory(dir=destination.parent) as scratch:
        stage = Path(scratch)
        classified = destination
        if transition_path is not None:
            classified = stage / destination.name
            write_transition_raster(classified, transition_path, None, None)
            write_qgis_palette(classified.with_suffix('.qml'))
        display = write_transition_display(classified, stage)
        if transition_path is not None and Path(transition_path).resolve() == (destination.parent / display.name).resolve():
            raise ValueError('源变化栅格与目标显示图须为不同文件。')
        for product in stage.iterdir():
            product.replace(destination.parent / product.name)
    print(f'原网格分类图（统计）：{destination}')
    print(f'QGIS 显示图（EPSG:4326）：{destination.parent / display.name}')


def write_classified_shp(destination, folder):
    """合并三个面图层，沿用密云 S1 变化图层的 class、label、area_m2 字段。"""
    with fiona.open(folder / f'{WATER_CLASSES[0][1]}.shp') as first:
        crs_wkt = first.crs_wkt
    crs = CRS.from_wkt(crs_wkt)
    metres_per_unit = crs.axis_info[0].unit_conversion_factor if crs.is_projected else None
    geod = crs.get_geod() if crs.is_geographic else None
    schema = {'geometry': 'Polygon', 'properties': {'class': 'str:20', 'label': 'str:32',
                                                     'area_m2': 'float:20.3'}}

    def records():
        for _, key, label in WATER_CLASSES:
            with fiona.open(folder / f'{key}.shp') as source:
                for feature in source:
                    geometry = shape(feature.geometry)
                    if geod is None:
                        area = geometry.area * metres_per_unit ** 2
                    else:
                        area = abs(geod.geometry_area_perimeter(geometry)[0])
                    yield {'geometry': feature.geometry,
                           'properties': {'class': key, 'label': label, 'area_m2': area}}

    with fiona.open(destination, 'w', driver='ESRI Shapefile', schema=schema,
                    crs_wkt=crs_wkt, encoding='UTF-8') as target:
        target.writerecords(records())


def export_water_transitions(output_dir, *, transition_path=None, pre_mask=None, post_mask=None):
    """用已有变化栅格或两期二值水体掩膜，输出分类 TIFF、三个分类面和一份合并面。"""
    if transition_path is not None and (pre_mask is not None or post_mask is not None):
        raise ValueError('请选择已有变化栅格或两期水体掩膜中的一种输入。')
    if transition_path is None and (pre_mask is None or post_mask is None):
        raise ValueError('分别指定早期和后期的水体掩膜。')
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f'请指定新的输出目录：{output_dir}')
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    # 先在同盘临时目录生成整套文件，再发布，避免留下不完整的 SHP 组件。
    with TemporaryDirectory(dir=output_dir.parent) as scratch:
        stage = Path(scratch) / 'products'
        stage.mkdir()
        tif = stage / 'water_transitions_palette.tif'
        write_transition_raster(tif, transition_path, pre_mask, post_mask)
        write_qgis_palette(tif.with_suffix('.qml'))
        write_transition_display(tif, stage)
        for code, key, label in WATER_CLASSES:
            print(f'生成类别 {code} 对应的{label}面')
            # 4 连通沿用双期分析口径，避免仅角点相接的像元合并成一个面。
            write_water_polygons(tif, stage / f'{key}.shp', selected_value=code, connectivity=4)
        write_classified_shp(stage / 'water_changes.shp', stage)
        stage.rename(output_dir)
    print(f'成果目录：{output_dir}')


if __name__ == '__main__':
    FOLDER = Path('E:/InSAR/水体提取/北京密云区/Report/两期水面变化_矢量对比/TYSAR_20250725_20250805')
    SOURCE = FOLDER / 'water_transitions.tif'
    DESTINATION = FOLDER / 'water_transitions_palette.tif'
    # 已归档的目录可能只保留单波段分类图，可以直接用它生成显示版。
    if not SOURCE.is_file() and not DESTINATION.is_file():
        raise FileNotFoundError(f'未找到变化波段文件 {SOURCE} 或已有分类图 {DESTINATION}')
    rebuild_transition_palette(SOURCE if SOURCE.is_file() else None, DESTINATION)
