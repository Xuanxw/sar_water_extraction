"""坡度20/15/10度对照：固定基础Q10和历史Q20，复用生产缓存，人工启动。"""
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from shapely import transform as transform_geometry

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from configuration import load_yaml, config_argument
from thres_segmentation.org_pipeline import SceneProgress, statistics, segment
from thres_segmentation.product_layout import scene_product
from thres_segmentation.org_io import read_aoi, prepare_auxiliary, export_water_polygon, write_json
from thres_segmentation.run_log import RunLog
# 复用纯工具函数；导入不会启动历史阈值实验，也不会读取它的结果。
from history_threshold_comparison import write_csv, compare_masks, sample_fixed_points



def save_comparison(filtered, masks, slopes, destination, max_side):
    """同一视野输出SAR及三组水体；仅读取缩略图，避免载入整景。"""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    with rasterio.open(filtered) as src:
        ratio = min(1, max_side / max(src.shape))
        size = (max(1, round(src.height * ratio)), max(1, round(src.width * ratio)))
        image = src.read(1, out_shape=size, masked=True)
    values = image.compressed()
    values = values[np.isfinite(values)]
    low, high = np.percentile(values, [2, 98]) if values.size else (0, 1)
    fig = Figure(figsize=(16, 5)); FigureCanvasAgg(fig)
    axes = fig.subplots(1, len(masks) + 1)
    axes[0].imshow(image, cmap='gray', vmin=low, vmax=high)
    axes[0].set_title('Filtered SAR (display P2-P98)')
    for ax, path, slope in zip(axes[1:], masks, slopes):
        with rasterio.open(path) as src:
            water = src.read(1, out_shape=size, masked=True)
        ax.imshow(water, cmap='Blues', vmin=0, vmax=1)
        ax.set_title(f'Base Q10 | History Q20 | Slope <= {slope:g} deg')
    for ax in axes:
        ax.axis('off')
    fig.tight_layout(); fig.savefig(destination, dpi=120); fig.clear()


def main(config_path):
    options = load_yaml(config_path)
    paths, experiment = options['paths'], options['experiment']
    baseline, output, cache = paths['BASELINE_OUTPUT_DIR'], paths['OUTPUT_DIR'], paths['CACHE_DIR']
    # 防止实验覆盖原成果/缓存；续跑仍只依据本实验data_cache中的完成记录。
    sources = (baseline, paths['BASELINE_CACHE_DIR'], paths['SNAP_CACHE_DIR'], paths['AUXILIARY_CACHE_DIR'])
    for target in (output, cache):
        if any(target.resolve().is_relative_to(s.resolve()) or s.resolve().is_relative_to(target.resolve()) for s in sources):
            raise ValueError('实验输出/缓存须与原成果及原缓存分开存放')
    if output.resolve().is_relative_to(cache.resolve()) or cache.resolve().is_relative_to(output.resolve()):
        raise ValueError('实验输出和实验缓存须使用独立目录')
    slopes = [float(v) for v in experiment['max_slope_degrees']]
    if (experiment['base_percentile'], experiment['history_percentile'], slopes) != (10, 20, [20., 15., 10.]):
        raise ValueError('本轮固定基础Q10、历史Q20，依次比较坡度20/15/10度')

    # 其余科学参数继承原生产快照，不依赖正在运行的历史阈值实验。
    config = json.loads((baseline / 'run.json').read_text(encoding='utf-8'))['config']
    config['paths'] = {k: Path(v) if isinstance(v, str) else v for k, v in config['paths'].items()}
    if config['threshold'] != {'water_percent': 10, 'mode': 'bottom'}:
        raise ValueError('原生产结果须为bottom、基础Q10')
    post = config['org']['postprocess']
    if not post['history_enabled'] or not post['terrain_enabled']:
        raise ValueError('本对照需要原流程已开启历史水体和坡度筛除')
    post['history_relax_percent'] = 10  # 基础10 + 放宽10 = 历史Q20。
    config['paths']['AUXILIARY_CACHE_DIR'] = paths['AUXILIARY_CACHE_DIR']
    config['org']['auxiliary']['download_missing'] = False
    records = json.loads((baseline / 'results.json').read_text(encoding='utf-8'))
    records = [r for r in records if r['status'] == 'complete']
    log = RunLog(output, experiment['console_color'])
    with log.capture(), log.stage('00', '固定Q10/Q20的坡度对照'):
        output.mkdir(parents=True, exist_ok=True); cache.mkdir(parents=True, exist_ok=True)
        write_json(output / 'experiment.json', options)
        with log.stage('01', '复用行政区及辅助数据'):
            aoi, crs = read_aoi(config['paths']['AOI_PATH'], config['org']['output_crs'])
            auxiliary = prepare_auxiliary(config, aoi, crs)
        summary = []
        for index, record in enumerate(records, 1):
            sid, date = record['scene_id'], record['date']
            original = baseline / date / sid
            common_cache = cache / 'scenes' / sid; common_cache.mkdir(parents=True, exist_ok=True)
            state = json.loads((paths['BASELINE_CACHE_DIR'] / 'scenes' / sid / 'progress.json').read_text(encoding='utf-8'))
            entry = state['stages']['filter']['files'][0]
            roots = {'snap': paths['SNAP_CACHE_DIR'], 'cache': paths['BASELINE_CACHE_DIR'] / 'scenes' / sid, 'output': original}
            filtered = roots[entry['root']] / entry['path']
            with rasterio.open(filtered) as src:
                scene_aoi = transform_geometry(aoi, Transformer.from_crs(crs, src.crs, always_xy=True).transform,
                                               interleaved=False, include_z=None)
            print(f'场景 {index}/{len(records)}：{sid}；复用滤波影像：{filtered}', flush=True)
            with log.stage('02', '一次计算Q10/Q20，三组共享幅度门限', sid):
                stats_file = common_cache / 'statistics.json'
                if experiment['resume'] and stats_file.exists():
                    stats = json.loads(stats_file.read_text(encoding='utf-8'))
                else:
                    stats = statistics(filtered, scene_aoi, config, common_cache)
                    stats['history_percentile'] = 20
                    write_json(stats_file, stats)
            reference = output / 'slope_20' / date / sid / 'water.tif'
            masks = []
            for slope in slopes:
                name = f'slope_{slope:g}'
                folder = output / name / date / sid; folder.mkdir(parents=True, exist_ok=True)
                if scene_product(original,'metadata.json').exists():write_json(folder/'metadata.json',json.loads(scene_product(original,'metadata.json').read_text(encoding='utf-8')))
                progress = SceneProgress(cache / name, sid, folder, paths['SNAP_CACHE_DIR'], experiment['resume'])
                with log.stage('03', f'{name}分割/矢量/质量记录', sid) as step:
                    if progress.completed('complete') and (folder/'water_polygon.shp').exists():
                        step['status'] = 'reused'
                    else:
                        trial = deepcopy(config)
                        trial['org']['postprocess']['max_slope_deg'] = slope
                        write_json(folder / 'parameters.json', {'source_run': str(baseline / 'run.json'),
                                   'base_percentile': 10, 'history_percentile': 20, 'max_slope_deg': slope,
                                   'postprocess': trial['org']['postprocess'], 'filtered_image': str(filtered)})
                        if not progress.completed('segment'):
                            products = segment(filtered, scene_aoi, auxiliary, trial, folder, stats, save_context=False)
                            progress.mark('segment', products)
                        # 对照是本实验的20度组，不是原先历史Q15的生产结果。
                        differences = compare_masks(folder / 'water.tif', reference, config['processing']['block_size'])
                        if not (progress.completed('vector') and (folder/'water_polygon.shp').exists()):
                            export_water_polygon(folder / 'water.tif', options=config.get('polygon_output',{}),
                                postprocess=trial['org']['postprocess'], cover_path=auxiliary.get('worldcover'),
                                slope_path=auxiliary.get('slope'), clip_geometry=scene_aoi)
                            progress.mark('vector', list(folder.glob('water_polygon.*')))
                        points_file = folder / 'fixed_points_predictions.csv'
                        sample_fixed_points(original / 'review/review.gpkg', folder / 'water.tif', points_file, sid, 20)
                        with points_file.open(encoding='utf-8-sig', newline='') as stream:
                            points = list(csv.DictReader(stream))
                        for point in points:
                            # 原核验点的pred来自Q15生产结果，明确命名避免误当作本轮20度组。
                            point['source_production_pred'] = point.pop('baseline_pred')
                            point['max_slope_deg'] = slope
                        write_csv(points_file, points)
                        extraction = json.loads((folder / 'extraction.json').read_text(encoding='utf-8'))
                        write_json(folder / 'quality.json', {'scene_id': sid, 'base_percentile': 10,
                                   'history_percentile': 20, 'max_slope_deg': slope,
                                   'accuracy': {'status': 'not_evaluated_no_reference', 'f1': None, 'iou': None},
                                   'extraction': extraction, 'comparison_to_slope20': differences,
                                   'sampling': {'source': str(original / 'review/review.gpkg'),
                                                'predictions': 'fixed_points_predictions.csv', 'resampled': False},
                                   'notes': ['增减像元相对于本实验Q10/Q20/坡度20度组，不是洪水变化或精度。',
                                             '原核验点及分层权重不变；source_production_pred来自原Q15生产结果。',
                                             '坡度筛除不是真正的雷达阴影/叠掩判定；未生成新的风险面。']})
                        progress.mark('complete', [folder / 'water.tif', folder / 'extraction.json'] +
                                      list(folder.glob('water_polygon.*')) + [folder / 'quality.json', points_file, folder / 'parameters.json'])
                    q = json.loads((folder / 'quality.json').read_text(encoding='utf-8'))
                    e = q['extraction']
                    summary.append({'scene_id': sid, 'date': date, 'base_percentile': 10, 'history_percentile': 20,
                                    'max_slope_deg': slope, 'reference_max_slope_deg': 20,
                                    'base_threshold_amplitude': e['statistics']['thresholds_amplitude'][0],
                                    'history_threshold_amplitude': e['statistics']['thresholds_amplitude'][3],
                                    'water_km2': e['final_area_m2'] / 1e6,
                                    **q['comparison_to_slope20'], 'water_shp': str(folder / 'water_polygon.shp')})
                    masks.append(folder / 'water.tif')
                    write_csv(output / 'comparison.csv', summary)
            with log.stage('04', '输出坡度三组并排快视图', sid):
                figures = output / 'figures'; figures.mkdir(exist_ok=True)
                picture = figures / f'{sid}.png'
                if not experiment['resume'] or not picture.exists():
                    save_comparison(filtered, masks, slopes, picture, config['output']['preview_size'])
        print(f'对照完成：{output}；面积减少不等于误检减少，仍需人工核验。', flush=True)
    return output


if __name__ == '__main__':
    main(config_argument())
