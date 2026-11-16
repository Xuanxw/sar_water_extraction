"""固定原滤波和地形掩膜，扫描整景百分位基线；不运行Otsu、不改原成果。"""

import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from shapely import transform as transform_geometry

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from configuration import config_argument, load_yaml
from thres_segmentation.org_io import read_aoi, prepare_auxiliary, export_water_polygon
from thres_segmentation.org_pipeline import SceneProgress, statistics, segment, save_preview
from thres_segmentation.product_layout import scene_product
from thres_segmentation.run_log import RunLog
from thres_segmentation.pipeline import write_csv
from utils.files import write_json


def save_percentile_panels(filtered, masks, percentiles, destination, max_side):
    """同一视野显示SAR及各Q的最终水体，缩略读图；不把快视图当作原像元评价。"""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    with rasterio.open(filtered) as src:
        scale = min(1, max_side / max(src.shape))
        shape = (max(1, round(src.height * scale)), max(1, round(src.width * scale)))
        image = src.read(1, out_shape=shape, masked=True)
    values = image.compressed()
    low, high = np.percentile(values[np.isfinite(values)], [2, 98])
    columns = 4
    rows = (len(masks) + 1 + columns - 1) // columns
    fig = Figure(figsize=(16, 4 * rows))
    FigureCanvasAgg(fig)
    axes = fig.subplots(rows, columns, squeeze=False).ravel()
    axes[0].imshow(image, cmap='gray', vmin=low, vmax=high)
    axes[0].set_title('Filtered SAR: display P2-P98')
    for ax, mask, q in zip(axes[1:], masks, percentiles):
        with rasterio.open(mask) as src:
            water = src.read(1, out_shape=shape, masked=True)
        ax.imshow(image, cmap='gray', vmin=low, vmax=high)
        ax.imshow(np.ma.masked_where(water.filled(255) != 1, np.ones(shape)),
                  cmap='Blues', vmin=0, vmax=1, alpha=0.65)
        ax.set_title(f'Base Q{q:g}: final polygons raster')
    for ax in axes:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(destination, dpi=120)
    fig.clear()


def run_base_threshold_scan(config_path):
    """扫描所选日期/景的Q，输出面成果、质量记录及面积表，支持按阶段续跑。

    原run.json冻结科学参数；不使用标签挑选阈值，也不把水体面积增加视为精度提高。
    原滤波和地形文件只读。每景一次磁盘分位数统计，所有Q共用，不重复SNAP或几何计算。
    """
    options = load_yaml(config_path)
    paths, experiment = options['paths'], options['experiment']
    baseline, output, cache = paths['BASELINE_OUTPUT_DIR'], paths['OUTPUT_DIR'], paths['CACHE_DIR']
    if any(target.resolve().is_relative_to(baseline.resolve())
           or baseline.resolve().is_relative_to(target.resolve()) for target in (output, cache)):
        raise ValueError('扫描输出和缓存不能与原成果目录重叠')
    percentiles = sorted(set(float(q) for q in experiment['percentiles']))
    if not percentiles or not all(0 < q < 100 for q in percentiles):
        raise ValueError('扫描百分位需位于0～100之间，不含端点')
    config = json.loads((baseline / 'run.json').read_text(encoding='utf-8'))['config']
    config['paths'] = {key: [Path(p) for p in value] if isinstance(value, list)
                       else Path(value) if isinstance(value, str) else value
                       for key, value in config['paths'].items()}
    if config['org']['postprocess']['history_enabled']:
        raise ValueError('本扫描要求沿用关闭历史放宽的Q30基线，以单独检验Base Q')
    config['threshold'].update(method='percentile', water_percent=percentiles[0])
    config['org']['auxiliary']['download_missing'] = False
    records = []
    for path in sorted(baseline.glob('*/*/scene.json')):
        record = json.loads(path.read_text(encoding='utf-8'))
        if (record['status'] == 'complete' and record['date'] in experiment['dates']
                and (not experiment['scene_ids'] or record['scene_id'] in experiment['scene_ids'])):
            records.append((path.parent, record))
    if not records:
        raise ValueError('未找到所选日期/景的完整基线成果')
    # 同时求各Q的敏感性分位，避免后续各组继续使用原Q30的Q28/Q32。
    delta = config['org']['sensitivity_delta_percent']
    requested = sorted({p for q in percentiles for p in (q, max(0., q - delta), min(100., q + delta))})
    log = RunLog(output, experiment['console_color'])
    with log.capture(), log.stage('00', '百分位基线阈值扫描'):
        output.mkdir(parents=True, exist_ok=True)
        cache.mkdir(parents=True, exist_ok=True)
        write_json(output / 'experiment.json', options, default=str)
        aoi, crs = read_aoi(config['paths']['AOI_PATH'], config['org']['output_crs'])
        with log.stage('01', '复用辅助数据'):
            auxiliary = prepare_auxiliary(config, aoi, crs)
        summaries = []
        for index, (original, record) in enumerate(records, 1):
            sid, date = record['scene_id'], record['date']
            print(f'场景 {index}/{len(records)}：{sid}', flush=True)
            original_cache = config['paths']['CACHE_DIR'] / 'scenes' / sid
            state = json.loads((original_cache / 'progress.json').read_text(encoding='utf-8'))
            entry = state['stages']['filter']['files'][0]
            roots = {'snap': config['paths']['SNAP_CACHE_DIR'], 'cache': original_cache, 'output': original}
            filtered = roots[entry['root']] / entry['path']
            geometry_mode = (config['org']['postprocess']['terrain_enabled']
                             and config['org']['postprocess'].get('terrain_method') in ('gui_geometry', 'radar_profiles'))
            terrain = ({name: scene_product(original, name + '.tif')
                        for name in ('terrain_exclusion', 'shadow_mask', 'layover_mask')} if geometry_mode else None)
            with rasterio.open(filtered) as src:
                scene_crs = src.crs
                scene_aoi = transform_geometry(aoi, Transformer.from_crs(crs, scene_crs, always_xy=True).transform,
                                               interleaved=False)
            common = SceneProgress(cache / 'statistics', sid, output, config['paths']['SNAP_CACHE_DIR'], experiment['resume'])
            with log.stage('02', '一次计算全部扫描分位数', sid) as step:
                stats_path = common.folder / 'statistics.json'
                if common.completed('statistics'):
                    stats = json.loads(stats_path.read_text(encoding='utf-8'))
                    step['status'] = 'reused'
                else:
                    # 复用现有多分位接口；此处只取数值，不启用历史水体分支。
                    stats = statistics(filtered, scene_aoi, config, common.folder, history_percentiles=requested)
                    stats['scan_thresholds_amplitude'] = stats.pop('history_thresholds_amplitude')
                    write_json(stats_path, stats)
                    common.mark('statistics', [stats_path])
            masks = []
            for number, q in enumerate(percentiles, 1):
                folder = output / f'base_q{q:g}' / date / sid
                folder.mkdir(parents=True, exist_ok=True)
                progress = SceneProgress(cache / f'base_q{q:g}', sid, folder,
                                         config['paths']['SNAP_CACHE_DIR'], experiment['resume'])
                with log.stage('03', f'Q{q:g}（{number}/{len(percentiles)}）：分割及面成果', sid) as step:
                    if progress.completed('complete'):
                        step['status'] = 'reused'
                    else:
                        trial = deepcopy(config)
                        trial['threshold']['water_percent'] = q
                        trial_stats = deepcopy(stats)
                        lookup = trial_stats.pop('scan_thresholds_amplitude')
                        trial_stats['water_percent'] = q
                        trial_stats['thresholds_amplitude'] = [lookup[str(p)] for p in
                                                               (q, max(0., q - delta), min(100., q + delta), q)]
                        threshold = trial_stats['thresholds_amplitude'][0]
                        trial_stats['threshold_selection']['percentile_threshold'] = threshold
                        trial_stats['decision_threshold'] = ((threshold - stats['mean']) / max(stats['std'], 1e-6)
                                                             if stats['normalization'] == 'scene_zscore' else threshold)
                        write_json(folder / 'parameters.json', {'base_q': q, 'filtered_image': filtered,
                                   'source_run': baseline / 'run.json', 'threshold': trial['threshold'],
                                   'postprocess': trial['org']['postprocess'], 'frozen_terrain': terrain}, default=str)
                        if not progress.completed('mask'):
                            segment(filtered, scene_aoi, auxiliary, trial, folder, trial_stats,
                                    save_context=False, terrain_products=terrain)
                            progress.mark('mask', [folder / 'water.tif', folder / 'extraction.json'])
                        export_water_polygon(folder / 'water.tif', options=trial.get('polygon_output', {}),
                                             postprocess=trial['org']['postprocess'], cover_path=auxiliary.get('worldcover'),
                                             slope_path=auxiliary.get('slope'),
                                             terrain_path=terrain['terrain_exclusion'] if terrain else None,
                                             clip_geometry=scene_aoi, clip_crs=scene_crs)
                        save_preview(filtered, folder / 'water_polygon.tif', folder / 'quicklook.png',
                                     config['output']['preview_size'])
                        extraction = json.loads((folder / 'extraction.json').read_text(encoding='utf-8'))
                        polygon = json.loads((folder / 'water_polygon.json').read_text(encoding='utf-8'))
                        write_json(folder / 'quality.json', {'scene_id': sid, 'base_q': q, 'extraction': extraction,
                                   'polygon': polygon, 'source_mask': str(folder / 'water.tif'),
                                   'accuracy': {'status': 'not_evaluated_no_reference', 'f1': None, 'recall': None},
                                   'notes': ['面积增加不等于精度改善；需人工核查水库恢复及陆地误检。']})
                        write_json(folder / 'scene.json', {**record, 'status': 'complete',
                                   'water_shp': str(folder / 'water_polygon.shp'),
                                   'water_polygon_shp': str(folder / 'water_polygon.shp')})
                        progress.mark('complete', list(folder.glob('water_polygon.*')) +
                                      [folder / name for name in ('water.tif', 'extraction.json', 'parameters.json',
                                                                  'quality.json', 'scene.json', 'quicklook.png')])
                    quality = json.loads((folder / 'quality.json').read_text(encoding='utf-8'))
                    extraction, polygon = quality['extraction'], quality['polygon']
                    summaries.append(dict(scene_id=sid, date=date, base_q=q,
                        threshold_amplitude=extraction['statistics']['thresholds_amplitude'][0],
                        valid_km2=extraction['valid_area_m2'] / 1e6, raw_candidate_km2=extraction['raw_area_m2'] / 1e6,
                        corrected_water_km2=extraction['final_area_m2'] / 1e6,
                        cleaned_water_pixels=polygon['cleaned_water_pixels'],
                        polygon_count=polygon['polygon_count'], water_shp=str(folder / 'water_polygon.shp')))
                    write_csv(output / 'comparison.csv', summaries)
                    masks.append(folder / 'water_polygon.tif')
            with log.stage('04', '输出各阈值并排图', sid):
                figures = output / 'figures'
                figures.mkdir(exist_ok=True)
                save_percentile_panels(filtered, masks, percentiles, figures / f'{sid}.png',
                                       config['output']['preview_size'])
        print(f'扫描完成：{output}；面积不是精度，原基线和Otsu成果未改变。', flush=True)
    return output


if __name__ == '__main__':
    run_base_threshold_scan(config_argument())
