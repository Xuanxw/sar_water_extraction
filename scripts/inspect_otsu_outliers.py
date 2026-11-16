"""核查Otsu极亮值并比较直方图范围：原始/滤波局部图、坐标、原生精确统计。

只读已有ORG裁剪缓存与SNAP成果，不重新滤波，不改原数据，不输出生产水体面。
所有范围使用同一AOI有效像元、同一Q30回退及接受判据；临时值缓冲结束后删除。
"""
import heapq
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import Window
from pyproj import Transformer
from shapely import transform as transform_geometry
from shapely.geometry import mapping
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from configuration import config_argument, load_yaml
from thres_segmentation.org_io import read_aoi, local_source, crop_org
from thres_segmentation.org_pipeline import SceneProgress
from thres_segmentation.metadata import read_source_metadata
from thres_segmentation.pipeline import write_csv
from thres_segmentation.scene_pipeline import disk_percentiles
from thres_segmentation.threshold import select_otsu_or_percentile
from thres_segmentation.run_log import RunLog
from utils.files import write_json
from utils.raster import windows


def export_bright_pixel_patch(raw, filtered, row, col, destination, size, input_domain='amplitude'):
    """输出同网格局部对照；完整数值范围和P2/P98显示分开标注，原始数组不拉伸落盘。"""
    half = size // 2
    x0, y0 = max(0, col - half), max(0, row - half)
    win = Window(x0, y0, min(size, raw.width - x0), min(size, raw.height - y0))
    a, b = raw.read(1, window=win, masked=True), filtered.read(1, window=win, masked=True)
    y, x = row - y0, col - x0
    low, high = np.percentile(a.compressed(), [2, 98])
    quantity = 'power' if input_domain=='intensity' else 'amplitude'
    fig = Figure(figsize=(15, 9))
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 3)
    for ax, data, title in [(axes[0, 0], a, f'ORG {quantity}, full local range'),
                             (axes[0, 1], b, f'Lee {quantity}, full local range')]:
        im = ax.imshow(data, cmap='gray', vmin=0, vmax=float(data.max()))
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.7)
    for ax, data, title in [(axes[1, 0], a, 'ORG display P2-P98'),
                             (axes[1, 1], b, 'Lee display: same ORG P2-P98')]:
        ax.imshow(data, cmap='gray', vmin=low, vmax=high)
        ax.set_title(title)
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]):
        ax.plot(x, y, '+', color='red', markersize=12)
        ax.axis('off')
    core = np.s_[max(0, y-2):y+3, max(0, x-2):x+3]
    formatter = {'float_kind': lambda value: f'{value:8.1e}'}
    text = f'ORG 5x5 {quantity}\n' + np.array2string(a[core].filled(np.nan), formatter=formatter)
    text += f'\n\nLee 5x5 {quantity}\n' + np.array2string(b[core].filled(np.nan), formatter=formatter)
    axes[0, 2].text(0, 1, text, va='top', family='monospace', fontsize=8)
    axes[0, 2].axis('off')
    axes[1, 2].plot(a[y], label='ORG')
    axes[1, 2].plot(b[y], label='Lee', alpha=0.8)
    axes[1, 2].axvline(x, color='red', linestyle='--')
    axes[1, 2].set_yscale('symlog', linthresh=1)
    axes[1, 2].set_ylim(0, max(float(a[y].max()), float(b[y].max())) * 1.2)
    axes[1, 2].set_title(f'Center row: unchanged {quantity}, log axis')
    axes[1, 2].legend()
    fig.suptitle(f'Native pixel row={row}, col={col} | ORG={float(a[y,x]):.7g}, Lee={float(b[y,x]):.7g}')
    fig.tight_layout()
    fig.savefig(destination, dpi=130)
    fig.clear()
    cells = []
    for dy in range(-5, 6):
        for dx in range(-5, 6):
            iy, ix = y + dy, x + dx
            if 0 <= iy < a.shape[0] and 0 <= ix < a.shape[1]:
                cells.append(dict(row=row+dy, col=col+dx, input_domain=input_domain,
                                  raw_value=float(a[iy,ix]) if not np.ma.is_masked(a[iy,ix]) else None,
                                  filtered_value=float(b[iy,ix]) if not np.ma.is_masked(b[iy,ix]) else None))
    write_csv(destination.with_suffix('.csv'), cells)
    return dict(raw_at_peak=None if np.ma.is_masked(a[y,x]) else float(a[y,x]),
                filtered_at_peak=float(b[y,x]), raw_local_max=float(a.max()),
                filtered_local_max=float(b.max()), local_display_percentiles=[float(low), float(high)])


def save_outlier_contact_sheet(raw, filtered, points, destination, size):
    """集中展示各极亮位置的ORG/Lee局部；每行共用原图P2/P98显示范围。"""
    fig = Figure(figsize=(12, 4 * len(points)))
    FigureCanvasAgg(fig)
    axes = fig.subplots(len(points), 2, squeeze=False)
    for index, point in enumerate(points):
        row, col = point['row'], point['col']
        x0, y0 = max(0, col - size//2), max(0, row - size//2)
        win = Window(x0, y0, min(size, raw.width-x0), min(size, raw.height-y0))
        a, b = raw.read(1, window=win, masked=True), filtered.read(1, window=win, masked=True)
        for ax, data, title in [(axes[index, 0], a, 'ORG'), (axes[index, 1], b, 'Lee')]:
            ax.imshow(data, cmap='gray', vmin=point['local_display_percentiles'][0],
                      vmax=point['local_display_percentiles'][1])
            ax.plot(col-x0, row-y0, '+', color='red', markersize=10)
            ax.set_title(f'#{index+1} {title} | {point["longitude"]:.6f}E, {point["latitude"]:.6f}N')
            ax.axis('off')
    fig.suptitle('Bright-pixel locations | display stretch only; original pixel values unchanged')
    fig.tight_layout()
    fig.savefig(destination, dpi=130)
    fig.clear()


def save_otsu_range_panels(values, image, valid, trials, destination, bins):
    """直方图计数为全部有效像元；分类图是缩略原始候选，尚未作地形/矢量后处理。"""
    fig = Figure(figsize=(6 * len(trials), 12))
    FigureCanvasAgg(fig)
    axes = fig.subplots(3, len(trials), squeeze=False)
    for column, trial in enumerate(trials):
        report = trial['report']
        low, high = report['histogram_min'], report['histogram_max']
        axis_low, axis_high = report['histogram_axis_min'], report['histogram_axis_max']
        counts = np.zeros(bins, dtype=np.int64)
        for start in range(0, values.size, 1024 * 1024):
            part = np.clip(values[start:start+1024*1024], low, high)
            if report['log_coefficient'] is not None:
                part = report['log_coefficient'] * np.log10(np.maximum(part, report['otsu_log_floor']))
            counts += np.histogram(part, bins=bins, range=(axis_low, axis_high))[0]
        centers = axis_low + (np.arange(bins) + 0.5) * (axis_high-axis_low) / bins
        axes[0, column].plot(centers, counts)
        axes[0, column].set_yscale('symlog', linthresh=1)
        if report['otsu_threshold_axis'] is not None:
            axes[0, column].axvline(report['otsu_threshold_axis'], color='red', label='Otsu candidate')
        selected_axis = trial['selected_threshold']
        if report['log_coefficient'] is not None:
            selected_axis = report['log_coefficient'] * np.log10(max(selected_axis, report['otsu_log_floor']))
        axes[0, column].axvline(selected_axis, color='green', linestyle='--', label='Actually used')
        axes[0, column].set_title(trial['name'] + '\n' + report['method_used'])
        axis_label = {'log_amplitude':'20log10(A), arbitrary reference',
                      'log_intensity':'10log10(I), arbitrary reference'}.get(report['otsu_domain'], 'Linear input value')
        axes[0, column].set_xlabel(axis_label)
        axes[0, column].legend()
        candidate = report['otsu_threshold']
        for row, threshold, title in [(1, candidate, 'Otsu candidate (before acceptance)'),
                                       (2, trial['selected_threshold'], 'Actually used: no auxiliary correction')]:
            mask = image < threshold if threshold is not None else np.zeros(image.shape, bool)
            axes[row, column].imshow(np.ma.array(mask, mask=~valid), cmap='gray_r', vmin=0, vmax=1)
            axes[row, column].set_title(title)
            axes[row, column].axis('off')
    fig.tight_layout()
    fig.savefig(destination, dpi=120)
    fig.clear()


def prepare_otsu_scene_cache(baseline, cache_root, scene_id, config, aoi, crs):
    """显式重建单景诊断输入；不修改原成果，保持所声明的输入物理量与Lee参数。"""
    from datasets.snap_preprocess import configure_paths
    from datasets.speckle import SpeckleFilter

    record = next(r for r in json.loads((baseline / 'inventory.json').read_text(encoding='utf-8'))
                  if r['scene_id'] == scene_id)
    original = baseline / record['date'] / scene_id
    snap_root = cache_root / 'snap'
    progress = SceneProgress(cache_root, scene_id, original, snap_root, resume=True)
    if not progress.completed('crop'):
        source = local_source(record, cache_root)
        metadata, _, _ = read_source_metadata(record['path'])
        metadata_path = progress.folder / 'metadata.json'
        write_json(metadata_path, metadata)
        filename = 'intensity.tif' if config['org']['filter']['input_domain']=='intensity' else 'amplitude.tif'
        cropped = crop_org(source, progress.folder / filename, aoi, crs,
                           next(iter(config['input']['bands'].values())), config['org']['crop_buffer_m'],
                           config['org']['input_nodata_value'], config['processing']['block_size'])
        if cropped is None:
            raise ValueError('所选景与行政区没有相交')
        progress.mark('crop', [cropped, metadata_path])
    options = config['org']
    configure_paths(config['paths']['SNAP_GPT_PATH'], snap_root, cache_root / 'snap_work',
                    threads=options['snap_threads'], tile_cache=options['snap_tile_cache'])
    speckle = SpeckleFilter(options['filter'])
    raw = progress.files('crop')[0]
    # 由SNAP现有缓存键核对物理量/窗口/ENL/版本；不能仅因阶段完成就复用错误输入域。
    filtered = speckle.snap.prepare_file(raw, (1,)) if speckle.snap else raw
    progress.mark('filter', [filtered])


def inspect_otsu_outliers(config_path):
    """精确定位AOI内极亮像元、回查ORG局部，并测试相同像元上的稳健直方图范围。"""
    options = load_yaml(config_path)
    baseline, output = options['paths']['BASELINE_OUTPUT_DIR'], options['paths']['OUTPUT_DIR']
    settings = options['audit']
    config = json.loads((baseline / 'run.json').read_text(encoding='utf-8'))['config']
    if 'input_domain' in settings:
        if not settings.get('prepare_source', False):
            raise ValueError('改变输入物理量时必须使用prepare_source和独立SOURCE_CACHE_DIR重新滤波')
        config['org']['filter']['input_domain'] = settings['input_domain']
    aoi, crs = read_aoi(config['paths']['AOI_PATH'], config['org']['output_crs'])
    ranges = settings['histogram_ranges']
    domains = settings.get('otsu_domains', ['linear'])
    percentiles = sorted({0., 2., 30., 50., 98., 99., 99.9, 99.99, 100.}
                         | {float(q) for pair in ranges if pair is not None for q in pair})
    log = RunLog(output, console_color=True)
    with log.capture(), log.stage('00', '极亮值核查与直方图范围对照'):
        for scene_id in settings['scene_ids']:
            original = next(baseline.glob(f'*/{scene_id}'))
            cache_root = Path(options['paths'].get('SOURCE_CACHE_DIR', config['paths']['CACHE_DIR']))
            if settings.get('prepare_source', False):
                with log.stage('PREP', '重建/复用所选景裁剪及Lee缓存', scene_id):
                    prepare_otsu_scene_cache(baseline, cache_root, scene_id, config, aoi, crs)
            cache = cache_root / 'scenes' / scene_id
            state = json.loads((cache / 'progress.json').read_text(encoding='utf-8'))
            roots = {'snap': cache_root / 'snap' if settings.get('prepare_source', False)
                     else Path(config['paths']['SNAP_CACHE_DIR']), 'cache': cache, 'output': original}
            entries = {stage: state['stages'][stage]['files'][0] for stage in ('crop', 'filter')}
            raw_path, filtered_path = [roots[entries[stage]['root']] / entries[stage]['path'] for stage in ('crop', 'filter')]
            meta_entry = next(e for e in state['stages']['crop']['files'] if Path(e['path']).name == 'metadata.json')
            metadata = json.loads((roots[meta_entry['root']] / meta_entry['path']).read_text(encoding='utf-8'))
            stats_path = cache / 'statistics.json'
            old = json.loads(stats_path.read_text(encoding='utf-8')).get('threshold_selection', {}) if stats_path.exists() else {}
            outlier_threshold = old.get('otsu_threshold')
            folder = output / scene_id
            folder.mkdir(parents=True, exist_ok=True)
            with rasterio.open(raw_path) as raw, rasterio.open(filtered_path) as filtered:
                if (raw.shape, raw.crs, raw.transform) != (filtered.shape, filtered.crs, filtered.transform):
                    raise ValueError('ORG裁剪与滤波影像不在同一网格，不能按像元回查')
                scene_aoi = transform_geometry(aoi, Transformer.from_crs(crs, filtered.crs, always_xy=True).transform, interleaved=False)
                count, extreme_count = 0, 0 if outlier_threshold is not None else None
                peak_heap = []
                with TemporaryDirectory(prefix='values_', dir=folder) as scratch:
                    buffer = Path(scratch) / 'valid_values.bin'
                    with log.stage('01', '完整扫描AOI有效像元，定位极值', scene_id), buffer.open('wb') as stream:
                        for index, win in enumerate(windows(filtered, settings['block_size']), 1):
                            data = filtered.read(1, window=win, masked=True)
                            inside = geometry_mask([mapping(scene_aoi)], out_shape=data.shape,
                                                   transform=filtered.window_transform(win), invert=True)
                            good = inside & ~np.ma.getmaskarray(data) & np.isfinite(data.data)
                            indices = np.flatnonzero(good)
                            values = data.data.ravel()[indices].astype('float64')
                            values.tofile(stream)
                            count += values.size
                            if outlier_threshold is not None:
                                extreme_count += int(np.count_nonzero(values >= outlier_threshold))
                            if values.size:
                                picks = np.argpartition(values, -min(8, values.size))[-min(8, values.size):]
                                for pick in picks:
                                    flat = int(indices[pick])
                                    item = (float(values[pick]), int(win.row_off) + flat // data.shape[1],
                                            int(win.col_off) + flat % data.shape[1])
                                    if len(peak_heap) < 256:
                                        heapq.heappush(peak_heap, item)
                                    elif item[0] > peak_heap[0][0]:
                                        heapq.heapreplace(peak_heap, item)
                            if index % 50 == 0:
                                print(f'{scene_id}：扫描{index}块；有效像元{count:,}', flush=True)
                    with log.stage('02', '精确分位数与范围实验', scene_id):
                        quantiles = disk_percentiles(buffer, count, percentiles)[0]
                        lookup = dict(zip(map(str, percentiles), map(float, quantiles)))
                        values = np.memmap(buffer, dtype='float64', mode='r', shape=(count,))
                        try:
                            trials = []
                            for domain in domains:
                                for pair in ranges:
                                    bounds = [lookup[str(float(q))] for q in pair] if pair is not None else None
                                    threshold, report = select_otsu_or_percentile(values, lookup['30.0'], 'otsu',
                                        config['threshold'].get('otsu_bins',1024), config['threshold'].get('otsu_min_separability',0.5),
                                        config['threshold'].get('otsu_min_class_fraction',0.001), bounds, domain,
                                        settings.get('otsu_log_floor',1e-6))
                                    report['histogram_percentiles'] = pair
                                    name = domain + ': ' + ('full_min_max' if pair is None else f'P{pair[0]:g}_P{pair[1]:g}')
                                    trials.append(dict(name=name, selected_threshold=threshold, report=report))
                                    print(f'{name}：{report["method_used"]}；输入域阈值={threshold:.6g}；回退={report["fallback_reason"]}', flush=True)
                            if outlier_threshold is None:
                                # 旧缓存已清理时，使用本次线性全范围候选定位长尾，不伪造旧记录。
                                reference = next((t for t in trials if t['report']['otsu_domain']=='linear'
                                                  and t['report']['histogram_percentiles'] is None), None)
                                outlier_threshold = reference['report']['otsu_threshold'] if reference else None
                                if outlier_threshold is not None:
                                    extreme_count = sum(int(np.count_nonzero(values[start:start+1024*1024] >= outlier_threshold))
                                                        for start in range(0, count, 1024*1024))
                            scale = min(1, settings['preview_size'] / max(filtered.shape))
                            shape = (max(1, round(filtered.height * scale)), max(1, round(filtered.width * scale)))
                            preview = filtered.read(1, out_shape=shape, masked=True)
                            transform = filtered.transform * rasterio.Affine.scale(filtered.width/shape[1], filtered.height/shape[0])
                            valid = (geometry_mask([mapping(scene_aoi)], out_shape=shape, transform=transform, invert=True)
                                     & ~np.ma.getmaskarray(preview) & np.isfinite(preview.data))
                            save_otsu_range_panels(values, preview.data, valid, trials, folder / 'histogram_range_comparison.png',
                                                   config['threshold'].get('otsu_bins',1024))
                            if len(domains) > 1:
                                for domain in domains:
                                    group = [t for t in trials if t['report']['otsu_domain'] == domain]
                                    save_otsu_range_panels(values, preview.data, valid, group,
                                        folder / f'{domain}_comparison.png', config['threshold'].get('otsu_bins',1024))
                        finally:
                            # Windows绘图库可能延迟释放数组视图；计算/绘图结束后显式关闭映射。
                            values._mmap.close()
                            del values
                with log.stage('03', '极亮像元局部对照与位置图', scene_id):
                    selected = []
                    for value, row, col in sorted(peak_heap, reverse=True):
                        if all(max(abs(row-p['row']), abs(col-p['col'])) >= settings['patch_size'] for p in selected):
                            x, y = filtered.xy(row, col)
                            lon, lat = Transformer.from_crs(filtered.crs, 4326, always_xy=True).transform(x, y)
                            name = f'patch_{len(selected)+1:02d}.png'
                            detail = export_bright_pixel_patch(raw, filtered, row, col, folder / name, settings['patch_size'],
                                                               config['org']['filter']['input_domain'])
                            selected.append(dict(row=row, col=col, x=x, y=y, longitude=lon, latitude=lat,
                                                 figure=name, **detail))
                        if len(selected) >= settings['max_patches']:
                            break
                    fig = Figure(figsize=(9, 9))
                    FigureCanvasAgg(fig)
                    ax = fig.subplots()
                    ax.imshow(preview, cmap='gray', vmin=lookup['2.0'], vmax=lookup['98.0'])
                    for n, point in enumerate(selected, 1):
                        xx = point['col'] * shape[1] / filtered.width
                        yy = point['row'] * shape[0] / filtered.height
                        ax.plot(xx, yy, '+', color='red', markersize=10)
                        ax.text(xx+8, yy, str(n), color='red')
                    ax.set_title('Filtered SAR display P2-P98 | numbered local patches')
                    ax.axis('off')
                    fig.tight_layout()
                    fig.savefig(folder / 'outlier_locations.png', dpi=130)
                    fig.clear()
                    save_outlier_contact_sheet(raw, filtered, selected, folder / 'local_comparison.png', settings['patch_size'])
                    # 本地原始GeoTIFF仍在时直接回查；缺少源文件不妨碍审查已保存的未滤波缓存。
                    if Path(metadata['source_raster']).is_file():
                        with rasterio.open(metadata['source_raster']) as source:
                            to_source = Transformer.from_crs(filtered.crs, source.crs, always_xy=True)
                            for point in selected:
                                xy = to_source.transform(point['x'], point['y'])
                                value = next(source.sample([xy], masked=True))[0]
                                point['original_geotiff_value'] = None if np.ma.is_masked(value) else float(value)
                                point['matches_unfiltered_cache'] = point['original_geotiff_value'] == point['raw_at_peak']
                write_csv(folder / 'outlier_points.csv', selected)
                write_csv(folder / 'range_trials.csv', [dict(name=t['name'], selected_threshold=t['selected_threshold'],
                          **{k: v for k, v in t['report'].items() if k != 'histogram_percentiles'}) for t in trials])
                write_json(folder / 'audit.json', dict(scene_id=scene_id, raw_image=raw_path, filtered_image=filtered_path,
                    shape=filtered.shape, crs=str(filtered.crs), raw_dtype=raw.dtypes[0], filtered_dtype=filtered.dtypes[0],
                    raw_nodata=raw.nodata if raw.nodata is not None and np.isfinite(raw.nodata) else None,
                    filtered_nodata=filtered.nodata if filtered.nodata is not None and np.isfinite(filtered.nodata) else None,
                    valid_pixels=count, pixels_above_previous_otsu=extreme_count,
                    outlier_reference_threshold=outlier_threshold,
                    outlier_reference_source='previous_statistics' if old.get('otsu_threshold') is not None else 'current_linear_full_histogram',
                    source_raster=metadata['source_raster'], source_xml_pixel_value=metadata.get('pixelValue'),
                    configured_input_domain=config['org']['filter']['input_domain'],
                    quantiles_input=lookup, previous_threshold_selection=old, trials=trials, patches=selected,
                    notes=['范围试验使用全部原生AOI有效像元，分位数为精确值。',
                           '图中的候选分类未作土地覆盖、地形或面清理，不是最终水体产品。',
                           '未删除原像元或修改原数据；局部图的显示拉伸不参与计算。']), default=str)
        print(f'核查完成：{output}', flush=True)
    return output


if __name__ == '__main__':
    inspect_otsu_outliers(config_argument())
