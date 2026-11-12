"""历史水体分位阈值对照；复用已有滤波影像，人工启动，不改原成果。"""
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path

import fiona
import numpy as np
import rasterio
from pyproj import Transformer
from shapely import transform as transform_geometry

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from configuration import load_yaml
from thres_segmentation.org_pipeline import SceneProgress, statistics, segment
from thres_segmentation.product_layout import scene_product
from thres_segmentation.org_io import read_aoi, prepare_auxiliary, export_water_polygon, windows, write_json
from thres_segmentation.run_log import RunLog

CONFIG_PATH = Path(r'E:\Code_projects\sar_water_extraction\configs\北京密云区\history_threshold_comparison.yaml')


def write_csv(path, rows):
    """临时文件写完再替换，便于中断后续跑。"""
    temp = path.with_suffix('.partial.csv')
    with temp.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def compare_masks(current, baseline, block_size):
    """同一网格逐像元比较；15%组应与原基线完全一致。"""
    counts = {'different_pixels': 0, 'added_water_pixels': 0, 'removed_water_pixels': 0}
    with rasterio.open(current) as new, rasterio.open(baseline) as old:
        if (new.shape, new.crs, new.transform) != (old.shape, old.crs, old.transform):
            raise ValueError('对照结果与原基线网格不一致')
        for win in windows(new, block_size):
            a, b = new.read(1, window=win), old.read(1, window=win)
            counts['different_pixels'] += int(np.count_nonzero(a != b))
            counts['added_water_pixels'] += int(np.count_nonzero((a == 1) & (b == 0)))
            counts['removed_water_pixels'] += int(np.count_nonzero((a == 0) & (b == 1)))
    return counts


def sample_fixed_points(source_gpkg, mask, output, scene_id, percentile):
    """对原150个随机点取新预测，不重新抽样；分层及权重仍来自原基线。"""
    with fiona.open(source_gpkg, layer='verification_points') as points, rasterio.open(mask) as src:
        rows = []
        for feature in points:
            p = dict(feature['properties'])
            x, y = src.transform * (p['col'] + .5, p['row'] + .5)
            pred = int(next(src.sample([(x, y)]))[0])
            rows.append({'point_id': f"{scene_id}_r{p['row']}_c{p['col']}",
                         'scene_id': scene_id, 'history_percentile': percentile,
                         'stratum': p['stratum'], 'population': p['population'],
                         'sample_n': p['sample_n'], 'weight': p['weight'],
                         'baseline_pred': p['pred'], 'trial_pred': pred})
    if rows:write_csv(output, rows)


def save_comparison(filtered, masks, percentiles, destination, max_side):
    """同一幅度显示范围和视野，输出SAR与三组结果并排图；不加载整景数组。"""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    with rasterio.open(filtered) as src:
        ratio = min(1, max_side / max(src.shape))
        size = (max(1, round(src.height * ratio)), max(1, round(src.width * ratio)))
        image = src.read(1, out_shape=size, masked=True)
    values = image.compressed()
    low, high = np.percentile(values[np.isfinite(values)], [2, 98])
    fig = Figure(figsize=(16, 5)); FigureCanvasAgg(fig)
    axes = fig.subplots(1, len(masks) + 1)
    axes[0].imshow(image, cmap='gray', vmin=low, vmax=high)
    axes[0].set_title('Filtered SAR (display P2-P98)')
    for ax, path, percentile in zip(axes[1:], masks, percentiles):
        with rasterio.open(path) as src:water = src.read(1, out_shape=size, masked=True)
        ax.imshow(water, cmap='Blues', vmin=0, vmax=1)
        ax.set_title(f'Base Q10 | History Q{percentile:g}')
    for ax in axes:ax.axis('off')
    fig.tight_layout();fig.savefig(destination, dpi=120)
    fig.clear()


def main(config_path=None):
    options = load_yaml(config_path or CONFIG_PATH)
    paths, experiment = options['paths'], options['experiment']
    baseline, output, cache = paths['BASELINE_OUTPUT_DIR'], paths['OUTPUT_DIR'], paths['CACHE_DIR']
    # 只保护源成果目录，避免配置笔误导致覆盖；不增加路径哈希等续跑限制。
    sources = (baseline, paths['BASELINE_CACHE_DIR'], paths['SNAP_CACHE_DIR'], paths['AUXILIARY_CACHE_DIR'])
    for target in (output, cache):
        if any(target.resolve().is_relative_to(s.resolve()) or s.resolve().is_relative_to(target.resolve()) for s in sources):
            raise ValueError('实验输出/缓存须与原成果及原缓存分开存放')
    # 使用原成果保存的参数，避免当前生产YAML后来修改导致对照条件改变。
    config = json.loads((baseline / 'run.json').read_text(encoding='utf-8'))['config']
    config['paths'] = {k: Path(v) if isinstance(v, str) else v for k, v in config['paths'].items()}
    if config['threshold'] != {'water_percent': 10, 'mode': 'bottom'}:
        raise ValueError('本实验要求原基线为bottom、第10百分位')
    post = config['org']['postprocess']
    if not post['history_enabled'] or post['history_relax_percent'] != 5:
        raise ValueError('本实验要求原基线开启历史水体，第15百分位')
    percentiles = [float(p) for p in experiment['history_percentiles']]
    if percentiles != [15., 20., 25.]:
        raise ValueError('本轮对照固定为15/20/25，基础阈值固定10')
    config['paths']['AUXILIARY_CACHE_DIR'] = paths['AUXILIARY_CACHE_DIR']
    # 辅助数据直接复用已有缓存；缺失时明确报错，避免实验中另行下载。
    config['org']['auxiliary']['download_missing'] = False
    records = json.loads((baseline / 'results.json').read_text(encoding='utf-8'))
    log = RunLog(output, experiment['console_color'])
    with log.capture(), log.stage('00', '历史水体阈值对照'):
        output.mkdir(parents=True, exist_ok=True); cache.mkdir(parents=True, exist_ok=True)
        write_json(output / 'experiment.json', options)
        with log.stage('01', '复用行政区及辅助数据'):
            aoi, crs = read_aoi(config['paths']['AOI_PATH'], config['org']['output_crs'])
            auxiliary = prepare_auxiliary(config, aoi, crs)
        summary = []
        for index, record in enumerate(records, 1):
            if record['status'] != 'complete':continue
            sid, date = record['scene_id'], record['date']
            original = baseline / date / sid
            common_cache = cache / 'scenes' / sid; common_cache.mkdir(parents=True, exist_ok=True)
            state = json.loads((paths['BASELINE_CACHE_DIR'] / 'scenes' / sid / 'progress.json').read_text(encoding='utf-8'))
            entry = state['stages']['filter']['files'][0]
            roots = {'snap': paths['SNAP_CACHE_DIR'], 'cache': paths['BASELINE_CACHE_DIR'] / 'scenes' / sid, 'output': original}
            filtered = roots[entry['root']] / entry['path']
            with rasterio.open(filtered) as src:
                scene_aoi = transform_geometry(aoi, Transformer.from_crs(crs, src.crs, always_xy=True).transform, interleaved=False, include_z=None)
            print(f'场景 {index}/{len(records)}：{sid}；复用滤波影像：{filtered}', flush=True)
            with log.stage('02', '一次计算Q10及历史Q15/Q20/Q25', sid):
                stats_file = common_cache / 'statistics.json'
                if experiment['resume'] and stats_file.exists():
                    stats = json.loads(stats_file.read_text(encoding='utf-8'))
                else:
                    stats = statistics(filtered, scene_aoi, config, common_cache, percentiles)
                    write_json(stats_file, stats)
            masks = []
            for percentile in percentiles:
                name = f'history_{percentile:g}'
                folder = output / name / date / sid;folder.mkdir(parents=True, exist_ok=True)
                if scene_product(original,'metadata.json').exists():write_json(folder/'metadata.json',json.loads(scene_product(original,'metadata.json').read_text(encoding='utf-8')))
                progress = SceneProgress(cache / name, sid, folder, paths['SNAP_CACHE_DIR'], experiment['resume'])
                with log.stage('03', f'{name}分割/矢量/质量记录', sid) as step:
                    if progress.completed('complete') and (folder/'water_polygon.shp').exists():
                        step['status'] = 'reused'
                    else:
                        trial = deepcopy(config)
                        trial['org']['postprocess']['history_relax_percent'] = percentile - 10
                        trial_stats = deepcopy(stats)
                        trial_stats['thresholds_amplitude'][3] = stats['history_thresholds_amplitude'][str(percentile)]
                        trial_stats['history_percentile'] = percentile
                        write_json(folder / 'parameters.json', {'source_run': str(baseline / 'run.json'),
                                   'base_percentile': 10, 'history_percentile': percentile,
                                   'postprocess': trial['org']['postprocess'], 'filtered_image': str(filtered)})
                        products = segment(filtered, scene_aoi, auxiliary, trial, folder, trial_stats, save_context=False)
                        differences = compare_masks(folder / 'water.tif', scene_product(original,'water.tif'), config['processing']['block_size'])
                        if percentile == 15 and differences['different_pixels']:
                            raise ValueError(f'{sid}：15%组与原结果有{differences["different_pixels"]}像元不同，请检查源缓存/辅助数据')
                        export_water_polygon(folder / 'water.tif', options=config.get('polygon_output',{}),
                                postprocess=trial['org']['postprocess'], cover_path=auxiliary.get('worldcover'),
                                slope_path=auxiliary.get('slope'), clip_geometry=scene_aoi)
                        sample_fixed_points(original / 'review/review.gpkg', folder / 'water.tif',
                                            folder / 'fixed_points_predictions.csv', sid, percentile)
                        extraction = json.loads((folder / 'extraction.json').read_text(encoding='utf-8'))
                        write_json(folder / 'quality.json', {'scene_id': sid, 'history_percentile': percentile,
                                   'accuracy': {'status': 'not_evaluated_no_reference', 'f1': None, 'iou': None},
                                   'extraction': extraction, 'comparison_to_original': differences,
                                   'sampling': {'source': str(original / 'review/review.gpkg'),
                                                'predictions': 'fixed_points_predictions.csv', 'resampled': False},
                                   'notes': ['新增/减少像元相对于Q15实验基线，不是相对历史水体的洪水变化。',
                                             '沿用原核验点；未生成新的风险图层，原风险面不代表本组已确认错误。']})
                        progress.mark('complete', products + list(folder.glob('water_polygon.*')) +
                                      [folder / 'quality.json', folder / 'fixed_points_predictions.csv', folder / 'parameters.json'])
                    q = json.loads((folder / 'quality.json').read_text(encoding='utf-8'))
                    e = q['extraction']
                    summary.append({'scene_id': sid, 'date': date, 'history_percentile': percentile,
                                    'base_threshold_amplitude': e['statistics']['thresholds_amplitude'][0],
                                    'history_threshold_amplitude': e['statistics']['thresholds_amplitude'][3],
                                    'water_km2': e['final_area_m2'] / 1e6,
                                    **q['comparison_to_original'], 'water_shp': str(folder / 'water_polygon.shp')})
                    masks.append(folder / 'water.tif')
                    write_csv(output / 'comparison.csv', summary)
            with log.stage('04', '输出三组并排快视图', sid):
                figures = output / 'figures';figures.mkdir(exist_ok=True)
                picture = figures / f'{sid}.png'
                if not experiment['resume'] or not picture.exists():
                    save_comparison(filtered, masks, percentiles, picture, config['output']['preview_size'])
        print(f'对照完成：{output}；面积/变化像元不是精度评价。', flush=True)
    return output


if __name__ == '__main__':
    main()
