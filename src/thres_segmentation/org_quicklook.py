"""ORG处理过程快视图：仅显示/统计已有产品，不修改水体、SHP或判别参数。"""
from contextlib import ExitStack
import gc
import math
from pathlib import Path
import textwrap

from affine import Affine
import fiona
import numpy as np
from PIL import Image
from pyproj import CRS
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_geom
from shapely.geometry import shape

from thres_segmentation.water_analysis import pixel_areas, area_sum
from utils.raster import windows

QUICKLOOK_VERSION = 'org_process_quicklook_1'


def quicklook_is_current(path, max_side):
    """只检查展示格式/缩略尺寸；旧PNG升级不使任何分割阶段失效。"""
    try:
        with Image.open(path) as image:
            return (image.info.get('Description') == QUICKLOOK_VERSION
                    and image.info.get('PreviewSize') == str(max_side))
    except OSError:
        return False


def read_quicklook_layer(path, target, resampling=Resampling.max):
    """有界读取共同显示网格；NoData转NaN，分类不采用双线性插值。

    二值层用max保留细水域/掩膜，SAR用average。显示聚合不能用于原生面积计算。
    缺失图层返回None，由面板明确说明，不补成全0掩膜。
    """
    if path is None or not Path(path).is_file():
        return None
    with rasterio.open(path) as source, WarpedVRT(
        source, **target, dtype='float32', nodata=np.nan, resampling=resampling,
    ) as vrt:
        return vrt.read(1, masked=True).filled(np.nan)


def summarize_water_stage_changes(paths, preview_shape, *, cover_path=None, excluded_classes=(), block_size=512):
    """先逐原生像元比较三阶段，再聚合变化位置，返回缩略图及真实删除/新增面积。

    paths依次为纯阈值、辅助修正后、清理后栅格。每对只比较双方0/1有效像元，
    未知不当作非水体。禁止直接相减两张独立下采样图，否则细线会产生伪变化。
    旧产品缺少土地覆盖掩膜时，可从同一缓存及排除类别重建显示，不写回原产品。
    """
    changes = {stage: {kind: np.zeros(preview_shape, bool) for kind in ('removed', 'added')}
               for stage in ('auxiliary', 'cleanup', 'total')}
    areas = {stage: dict(removed=0.0, added=0.0) for stage in changes}
    cover_known = np.zeros(preview_shape, bool)
    cover_excluded = np.zeros(preview_shape, bool)
    with rasterio.Env(GDAL_CACHEMAX=64 * 1024 * 1024), ExitStack() as stack:
        readers = [stack.enter_context(rasterio.open(path)) for path in paths]
        ref = readers[0]
        if any((src.crs, src.transform, src.shape) != (ref.crs, ref.transform, ref.shape) for src in readers[1:]):
            raise ValueError('变化对照要求三个水体栅格位于同一原生网格')
        cover = None
        if cover_path is not None:
            source = stack.enter_context(rasterio.open(cover_path))
            cover = stack.enter_context(WarpedVRT(source, crs=ref.crs, transform=ref.transform,
                width=ref.width, height=ref.height, resampling=Resampling.nearest, dtype='float32', nodata=np.nan))
        for win in windows(ref, block_size):
            data = [src.read(1, window=win) for src in readers]
            # 将原生像元中心归入共同缩略单元；同一单元可同时包含删除和新增，分别保留。
            rr = ((np.arange(int(win.height)) + int(win.row_off) + .5) * preview_shape[0] / ref.height).astype(int)
            cc = ((np.arange(int(win.width)) + int(win.col_off) + .5) * preview_shape[1] / ref.width).astype(int)
            cell_areas = pixel_areas(ref.crs, ref.window_transform(win), data[0].shape)
            for stage, before, after in (('auxiliary', data[0], data[1]), ('cleanup', data[1], data[2]),
                                         ('total', data[0], data[2])):
                valid = np.isin(before, (0, 1)) & np.isin(after, (0, 1))
                for kind, mask in (('removed', valid & (before == 1) & (after == 0)),
                                   ('added', valid & (before == 0) & (after == 1))):
                    y, x = np.nonzero(mask)
                    changes[stage][kind][rr[y], cc[x]] = True
                    areas[stage][kind] += area_sum(mask, cell_areas)
            if cover is not None:
                values = cover.read(1, window=win, masked=True).filled(np.nan)
                valid = np.isin(data[0], (0, 1)) & np.isin(values, (10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100))
                y, x = np.nonzero(valid)
                cover_known[rr[y], cc[x]] = True
                y, x = np.nonzero(valid & np.isin(values, excluded_classes))
                cover_excluded[rr[y], cc[x]] = True
    reconstructed_cover = np.where(cover_known, cover_excluded, np.nan).astype('float32') if cover_path else None
    return changes, areas, reconstructed_cover


def rasterize_water_shp_preview(path, target):
    """读取实际最终SHP并在显示网格栅格化；保留孔洞，仅临时简化显示几何。

    批量256个面写入缩略数组，避免把全部顶点交给绘图库。all_touched仅用于显示，
    不生成新水体成果，也不以此图量算面积；SHP平滑边界可能与清理后TIF略有不同。
    """
    if not Path(path).is_file():
        return None
    result = np.zeros((target['height'], target['width']), np.uint8)
    a = target['transform']
    tolerance = .25 * min(math.hypot(a.a, a.d), math.hypot(a.b, a.e))
    with fiona.open(path) as source:
        if not source.crs:
            raise ValueError('最终水体SHP缺少坐标系，不能猜测缩略位置')
        same_crs = CRS.from_user_input(source.crs) == CRS.from_user_input(target['crs'])
        batch = []
        for feature in source:
            if feature['geometry'] is None:
                continue
            geometry = feature['geometry'] if same_crs else transform_geom(source.crs, target['crs'], feature['geometry'])
            polygon = shape(geometry)
            if polygon.geom_type not in ('Polygon', 'MultiPolygon'):
                raise ValueError('最终水体SHP缩略图需要Polygon/MultiPolygon面')
            polygon = polygon.simplify(tolerance, preserve_topology=True)
            if not polygon.is_empty:
                batch.append((polygon, 1))
            if len(batch) >= 256:
                rasterize(batch, out=result, transform=a, all_touched=True)
                batch.clear()
        if batch:
            rasterize(batch, out=result, transform=a, all_touched=True)
    return result


def save_org_process_quicklook(image_path, folder, destination, max_side, config, auxiliary,
                              original_path=None, log=print):
    """输出输入、纯阈值、各掩膜、三阶段变化及真实SHP的过程对照PNG。

    所有面板采用共同范围；仅读取缩略数组，变化另作原生分块统计。掩膜可能重叠，
    不能把各掩膜覆盖面积相加当作实际删除。道路永远标作review，不暗示已删除。
    """
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    if max_side < 1:
        raise ValueError('过程快视图的最长边必须为正像元数')
    folder = Path(folder)
    destination = Path(destination)
    post = config['org']['postprocess']
    guard = config['threshold'].get('slope_guard', {})
    with rasterio.open(image_path) as ref:
        scale = min(1.0, max_side / max(ref.shape))
        height, width = max(1, round(ref.height * scale)), max(1, round(ref.width * scale))
        a = ref.transform
        sx, sy = ref.width / width, ref.height / height
        target = dict(crs=ref.crs, transform=Affine(a.a * sx, a.b * sy, a.c, a.d * sx, a.e * sy, a.f),
                      width=width, height=height)
    layers = {}
    paths = [folder / name for name in ('raw.tif', 'water.tif', 'water_polygon.tif')]
    with rasterio.Env(GDAL_CACHEMAX=64 * 1024 * 1024):
        filtered = read_quicklook_layer(image_path, target, Resampling.average)
        original = read_quicklook_layer(original_path, target, Resampling.average)
        for name, path in zip(('raw', 'corrected', 'final'), paths):
            layers[name] = read_quicklook_layer(path, target)
        reconstruct_cover = post['landcover_enabled'] and not (folder / 'landcover_exclusion.tif').exists()
        changes = areas = None
        if all(path.is_file() for path in paths):
            changes, areas, cover = summarize_water_stage_changes(
                paths, (height, width), cover_path=auxiliary.get('worldcover') if reconstruct_cover else None,
                excluded_classes=post['excluded_cover_classes'], block_size=config['processing']['block_size'],
            )
            if reconstruct_cover and cover is not None:
                layers['cover'] = cover
                log('土地覆盖掩膜未保存：仅从本配置及缓存重建显示，不重算水体')
        masks = []
        if post['landcover_enabled']:
            title = 'WorldCover exclusion' + (' (reconstructed)' if reconstruct_cover else '')
            masks.append(('cover', title, folder / 'landcover_exclusion.tif', 'mask'))
        if post['terrain_enabled']:
            if post.get('terrain_method', 'slope_only') in ('gui_geometry', 'radar_profiles'):
                masks.extend((('shadow', 'Radar shadow mask', folder / 'shadow_mask.tif', 'mask'),
                              ('layover', 'Radar layover mask', folder / 'layover_mask.tif', 'mask'),
                              ('terrain', 'Joint terrain exclusion', folder / 'terrain_exclusion.tif', 'mask')))
            else:
                slope = read_quicklook_layer(folder / 'slope.tif', target)
                layers['terrain'] = np.where(np.isfinite(slope), slope > post['max_slope_deg'], np.nan) if slope is not None else None
                masks.append(('terrain', f'Slope > {post["max_slope_deg"]:g} deg exclusion', None, 'mask'))
        if guard.get('enabled', False):
            masks.append(('slope_guard', f'Steep-slope threshold guard > {guard.get("slope_deg", 15):g} deg',
                          folder / 'slope_threshold_rejection.tif', 'mask'))
        if post['history_enabled']:
            masks.append(('history', 'Historical water prior (not a mask)', folder / 'history.tif', 'water'))
        if config.get('road_review', {}).get('enabled', False):
            masks.append(('road', 'OSM corridors - REVIEW ONLY', folder / 'road_review/road_corridor.tif', 'road'))
        for name, _, path, _ in masks:
            if name not in layers:
                layers[name] = read_quicklook_layer(path, target)
        layers['shp'] = rasterize_water_shp_preview(folder / 'water_polygon.shp', target)
    panels = [('original', 'SAR before Lee (cropped)', original, 'sar'),
              ('filtered', 'Filtered SAR / threshold input', filtered, 'sar'),
              ('raw', 'Raw threshold candidates - NO postprocessing', layers['raw'], 'water'),
              ('corrected', 'After auxiliary + segment morphology', layers['corrected'], 'water')]
    panels.extend((name, title, layers.get(name), kind) for name, title, _, kind in masks)
    panels.extend((('final', 'Final cleaned water TIF', layers['final'], 'water'),
                   ('shp', 'Actual final SHP (display rasterized)', layers['shp'], 'water')))
    if changes is not None:
        for stage, title in (('auxiliary', 'Raw to auxiliary corrections'), ('cleanup', 'Auxiliary to cleaned TIF'),
                             ('total', 'Total: raw to cleaned TIF')):
            title += f' | -{areas[stage]["removed"] / 1e6:.6g} / +{areas[stage]["added"] / 1e6:.6g} km2'
            panels.append((stage, title, changes[stage], 'change'))
    values = filtered[np.isfinite(filtered)]
    low, high = np.percentile(values, (2, 98)) if values.size else (0, 1)
    gray = np.where(np.isfinite(filtered), np.clip((filtered - low) / max(high - low, 1e-6), 0, 1), .25)
    background = np.repeat((gray * 255).astype('uint8')[..., None], 3, axis=2)
    columns = 4
    rows = math.ceil(len(panels) / columns)
    figure = Figure(figsize=(20, rows * 4.5 + 1.0), facecolor='white')
    FigureCanvasAgg(figure)
    axes = ax = None
    try:
        axes = figure.subplots(rows, columns, squeeze=False)
        for number, ((_, title, data, kind), ax) in enumerate(zip(panels, axes.ravel()), 1):
            ax.imshow(background)
            if data is None:
                ax.text(.5, .5, 'Not available / not saved', transform=ax.transAxes, ha='center', color='orange')
            elif kind == 'sar':
                v = data[np.isfinite(data)]
                lo, hi = np.percentile(v, (2, 98)) if v.size else (0, 1)
                ax.imshow(np.ma.masked_invalid(data), cmap='gray', vmin=lo, vmax=max(hi, lo + 1e-6))
            else:
                rgba = np.zeros((height, width, 4), np.uint8)
                if kind == 'change':
                    rgba[data['removed']] = (255, 45, 45, 235)
                    rgba[data['added']] = (40, 230, 70, 235)
                    rgba[data['removed'] & data['added']] = (255, 220, 0, 245)
                elif kind == 'water':
                    rgba[data == 1] = (35, 135, 245, 205)
                elif kind == 'road':
                    rgba[data == 1] = (255, 170, 0, 220)
                else:
                    rgba[~np.isfinite(data)] = (255, 210, 70, 85)
                    rgba[data == 1] = (255, 45, 45, 170)
                ax.imshow(rgba)
            ax.set_title(textwrap.fill(f'{number:02d} {title}', 43), fontsize=10)
            ax.set_axis_off()
        for ax in axes.ravel()[len(panels):]:
            ax.set_visible(False)
        figure.suptitle(folder.name + ' | Processing overview - no ground truth', fontsize=13, y=.985)
        note = ('Blue: water | Mask panels: red=exclusion domain, pale yellow=unknown | Orange: roads (review only)\n'
                'Change panels: red=removed, green=added, yellow=both within overview cell | Areas: native pixels\n'
                'P2-P98 display only; binary max aggregation; SHP display simplification. Mask effects overlap; changes use TIF stages.')
        if config.get('road_review', {}).get('enabled', False):
            note += '\nOSM: © OpenStreetMap contributors, ODbL'
        figure.text(.5, .015, note, ha='center', va='bottom', fontsize=9)
        figure.subplots_adjust(left=.01, right=.99, top=.95, bottom=.095, hspace=.22, wspace=.03)
        destination.parent.mkdir(parents=True, exist_ok=True)
        pending = destination.with_suffix('.partial.png')
        figure.savefig(pending, dpi=100, metadata=dict(Description=QUICKLOOK_VERSION, PreviewSize=str(max_side),
                       PanelTitles='\n'.join(title for _, title, _, _ in panels)))
        pending.replace(destination)
    finally:
        figure.clear()
        del axes, ax
        del figure
        gc.collect()
    log(f'处理过程快视图完成：{len(panels)}面板；{destination}')
    return destination
