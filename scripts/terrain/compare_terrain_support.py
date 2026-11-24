"""单景比较DEM支持域和高程包络；只输出候选地形，不重做或覆盖水体。

四组固定同一轨道、雷达网格、30米输出和7.5米剖面采样。区域包络先检查
全局高程界限下潜在同斜距地形的数值包围域，再扫描完整DEM及大地水准面。
所有结果仍受原始成像约定与DEM分辨率限制，未知减少不等于真值精度提高。
"""
from copy import deepcopy
from pathlib import Path
import json
import shutil
import sys

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from configuration import config_argument, load_yaml
from sar_geometry.metadata import read_bc_metadata
from sar_geometry.terrain_support import bound_same_range_terrain, scan_regional_dem_envelope
from sar_geometry.scene import run_zero_doppler_terrain_visibility
from sar_geometry.scene_validation import ResourceMonitor
from utils.files import sha256, write_json


def compare_masks(baseline, candidate, water_path):
    """在冻结的30米网格及现有水体有效范围统计类别转移；不计算真值精度。"""
    with rasterio.open(baseline/'radar_coverage.tif') as ref:
        with rasterio.open(water_path) as src, WarpedVRT(src, crs=ref.crs, transform=ref.transform,
                width=ref.width, height=ref.height, nodata=255, resampling=Resampling.nearest) as vrt:
            water = vrt.read(1)
        valid = (ref.read(1)==1) & np.isin(water, (0, 1))
        result = dict(evaluated_pixels=int(valid.sum()), masks={})
        for name in ('shadow_mask', 'layover_mask', 'terrain_exclusion'):
            with rasterio.open(baseline/(name+'.tif')) as a, rasterio.open(candidate/(name+'.tif')) as b:
                if (a.crs, a.transform, a.shape) != (b.crs, b.transform, b.shape):
                    raise ValueError('对照网格不同，不能直接比较')
                before, after = a.read(1), b.read(1)
            result['masks'][name] = dict(unknown_pixels=int(np.count_nonzero(valid & (after==255))),
                excluded_pixels=int(np.count_nonzero(valid & (after==1))),
                unknown_to_clear=int(np.count_nonzero(valid & (before==255) & (after==0))),
                unknown_to_excluded=int(np.count_nonzero(valid & (before==255) & (after==1))),
                known_to_unknown=int(np.count_nonzero(valid & (before<2) & (after==255))),
                known_class_changed=int(np.count_nonzero(valid & (before<2) & (after<2) & (before!=after))))
            if name == 'terrain_exclusion':
                result['existing_water_newly_excluded_pixels'] = int(np.count_nonzero(valid & (water==1) & (before!=1) & (after==1)))
    return result


def main(config_path):
    """扫描依据→四组单景计算→差异统计；几何内核和水体参数保持不变。"""
    cfg = load_yaml(config_path)
    paths, options = cfg['paths'], cfg['experiment']
    output, baseline = paths['OUTPUT_DIR'], paths['BASELINE_GEOMETRY_DIR']
    output.mkdir(parents=True, exist_ok=True)
    def log(message):
        print(message, flush=True)
        with (output/'processing.log').open('a', encoding='utf-8') as f:
            f.write(message+'\n')
    frozen = [baseline/(name+'.tif') for name in ('shadow_mask', 'layover_mask', 'terrain_exclusion')]
    frozen += [paths['SCENE_DIR']/'water_polygon.tif', paths['SCENE_DIR']/'water_polygon.shp']
    hashes = {str(p): sha256(p) for p in frozen}
    source = load_yaml(paths['BASE_CONFIG'])
    products = Path(json.loads((paths['SCENE_DIR']/'quality.json').read_text(encoding='utf-8'))['source_mask']).parent
    metadata_record = json.loads((products/'metadata.json').read_text(encoding='utf-8'))
    member = Path(metadata_record['source_image']).stem+'.xml'
    metadata = read_bc_metadata(products/'source.xml', xml_member=member)
    # 直接复用本次功率Lee缓存，不重做滤波。progress只用于定位输入文件。
    progress = json.loads((products.parent/'progress.json').read_text(encoding='utf-8'))
    entry = progress['stages']['filter']['files'][0]
    filtered = source['paths']['SNAP_CACHE_DIR']/entry['path']
    profile = deepcopy(source['org']['profile_geometry'])
    domain_options = dict(options['coverage_survey'], output_crs=profile['scene']['output_crs'])
    executable = source['paths']['GEOMETRY_CORE_PATH']
    log('[01] 扫描潜在同斜距支持域及完整DEM/大地水准面极值')
    domain = bound_same_range_terrain(metadata, executable, options['global_height_bounds_m'], domain_options)
    envelope = scan_regional_dem_envelope(source['paths']['PROFILE_DEM_TILES'], profile['geoid_url'], domain,
                options['height_margin_m'], options['height_rounding_m'])
    write_json(output/'regional_height_envelope.json', envelope)
    log('[01] 区域椭球高包络：'+str(envelope['height_bounds_m']))
    base_config = dict(mode='visibility', xml_member=member,
        paths=dict(XML_PATH=products/'source.xml', ORG_PATH=filtered,
                   DEM_TILES=source['paths']['PROFILE_DEM_TILES'], GEOID_PATH=profile['geoid_url'],
                   CORE_PATH=executable, OUTPUT_DIR=None),
        model=profile['model'], scene=profile['scene'], execution=profile['execution'])
    results = []
    inputs_by_buffer = {options['baseline_buffer_m']: baseline/'data_cache'}
    variants = [
        ('baseline', options['baseline_buffer_m'], options['global_height_bounds_m']),
        ('expanded_global', options['expanded_buffer_m'], options['global_height_bounds_m']),
        ('regional', options['baseline_buffer_m'], envelope['height_bounds_m']),
        ('expanded_regional', options['expanded_buffer_m'], envelope['height_bounds_m']),
    ]
    for name, buffer, bounds in variants:
        if name == 'baseline':
            report = json.loads((baseline/'geometry_quality.json').read_text(encoding='utf-8'))
            directory = baseline
        else:
            directory = output/name
            case = deepcopy(base_config)
            case['paths']['OUTPUT_DIR'] = directory
            case['scene'].update(support_buffer_m=buffer, height_bounds_m=bounds)
            cache = directory/'data_cache'
            cache.mkdir(parents=True, exist_ok=True)
            if buffer in inputs_by_buffer:
                for filename in ('dem_ellipsoid.tif', 'reference.tif', 'inputs.json'):
                    target = cache/filename
                    if not target.exists():
                        shutil.copyfile(inputs_by_buffer[buffer]/filename, target)
            write_json(directory/'config.json', case, default=str)
            log(f'[02] {name}：支持域{buffer/1000:g} km；高程{bounds}')
            with ResourceMonitor() as usage, rasterio.Env(GDAL_CACHEMAX=64*1024*1024, GDAL_HTTP_TIMEOUT=30):
                report = run_zero_doppler_terrain_visibility(case, metadata, log)
            report['memory_monitor'] = usage.result()
            write_json(directory/'geometry_quality.json', report)
            inputs_by_buffer[buffer] = cache
        comparison = compare_masks(baseline, directory, paths['SCENE_DIR']/'water_polygon.tif')
        row = dict(variant=name, support_buffer_m=buffer, height_bounds_m=bounds, output_dir=str(directory),
                   geometry_seconds=report['current_run_geometry_seconds'],
                   total_seconds=report['current_run_total_seconds'], reused=report.get('cache_reused_this_run',False),
                   geometry_time_budget_met=report['geometry_time_budget_met'], memory_monitor=report.get('memory_monitor'),
                   **comparison)
        results.append(row)
        write_json(output/'comparison.json', results)
        log('[03] '+name+'：'+json.dumps(comparison['masks']['terrain_exclusion'],ensure_ascii=False))
    unchanged = all(sha256(Path(p))==value for p,value in hashes.items())
    write_json(output/'baseline_integrity.json',dict(before_sha256=hashes,unchanged=unchanged))
    if not unchanged:
        raise RuntimeError('冻结基线内容发生变化，请停止使用本次对照结论')
    shutil.copyfile(config_path, output/'experiment.yaml')
    log('[04] 单景对照完成，冻结水体和地形成果哈希未变化。')
    return results


if __name__ == '__main__':
    main(config_argument())
