"""阈值结果人工核验：汇总已有抽样点，人工填写后按分层权重估计精度。

直接修改文件末尾 MODE 和路径。prepare 不重复抽样、不覆盖已填写的核验图层；
evaluate 只读取人工标签，不运行阈值分割，也不把辅助数据当作真值。
"""
import csv
import json
from collections import Counter
from pathlib import Path

import fiona
import rasterio
from fiona.transform import transform_geom
from pyproj import CRS


def write_csv(path, records):
    """Excel 可直接打开的 UTF-8 CSV；核验标签本身在 GPKG 中编辑。"""
    if records:
        with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)


def pixel_area(source, row, col):
    """计算抽样像元面积：地理坐标用椭球面积，投影坐标考虑单位。"""
    crs = CRS.from_user_input(source.crs)
    affine = source.transform
    if crs.is_projected:
        factors = [axis.unit_conversion_factor for axis in crs.axis_info[:2]]
        return abs(affine.a * affine.e - affine.b * affine.d) * factors[0] * factors[1]
    corners = [affine * (col + x, row + y) for x, y in ((0, 0), (1, 0), (1, 1), (0, 1))]
    x, y = zip(*corners)
    return abs(crs.get_geod().polygon_area_perimeter(x, y)[0])


def prepare(results_dir, review_dir):
    """合并原核验点为 WGS84 图层，并保存不可随人工编辑改变的抽样底表。"""
    results_dir, review_dir = Path(results_dir), Path(review_dir)
    review_dir.mkdir(parents=True, exist_ok=True)
    gpkg = review_dir / 'manual_review.gpkg'
    frame_path = review_dir / 'sample_frame.json'
    if gpkg.exists() and frame_path.exists():
        print(f'复用已有核验材料，保留人工填写内容：{gpkg}')
        return
    if gpkg.exists() or frame_path.exists():
        raise FileExistsError('核验材料不完整，请检查现有文件；不自动覆盖可能已填写的标签。')
    results = json.loads((results_dir / 'results.json').read_text(encoding='utf-8'))
    schema = {'geometry': 'Point', 'properties': {
        'point_id': 'str', 'scene_id': 'str', 'date': 'str', 'sensor': 'str',
        'stratum': 'str', 'row': 'int', 'col': 'int', 'pred': 'int',
        'ref_class': 'int', 'confidence': 'str', 'note': 'str',
        'reviewer': 'str', 'reference': 'str', 'review_date': 'str',
        'population': 'int64', 'sample_n': 'int', 'incl_prob': 'float',
        'weight': 'float', 'pixel_m2': 'float'}}
    frame = {'scope': 'scene_observations_not_unique_geographic_area', 'scenes': [], 'points': []}
    index = []
    # 先写临时图层；成功后再发布，避免中断时留下看似完成的文件。
    temp = review_dir / 'manual_review_building.gpkg'
    if temp.exists():
        raise FileExistsError(f'存在上次未完成的临时图层，请先检查：{temp}')
    with fiona.open(temp, 'w', driver='GPKG', layer='verification_points',
                    crs='EPSG:4326', schema=schema) as target:
        for result in results:
            if result['status'] != 'complete':
                raise ValueError(f"场景尚未完成：{result['scene_id']}")
            scene_id, date = result['scene_id'], result['date']
            folder = results_dir / date / scene_id
            quality = json.loads((folder / 'quality.json').read_text(encoding='utf-8'))
            original = folder / quality['artifacts']['vectors']
            points = []
            with rasterio.open(folder / 'water.tif') as mask, fiona.open(
                    original, layer='verification_points') as source:
                for feature in source:
                    p = dict(feature['properties'])
                    row, col = p['row'], p['col']
                    point_id = f'{scene_id}_r{row}_c{col}'
                    area = pixel_area(mask, row, col)
                    # 保留原始抽样位置/预测/权重；新增字段只为人工判读提供索引。
                    p.update(point_id=point_id, date=date, sensor=result['sensor'],
                             reviewer='', reference='', review_date='', pixel_m2=area)
                    target.write({'geometry': transform_geom(source.crs, 'EPSG:4326', feature['geometry']),
                                  'properties': p})
                    points.append({key: p[key] for key in (
                        'point_id', 'scene_id', 'stratum', 'row', 'col', 'pred',
                        'population', 'sample_n', 'incl_prob', 'weight', 'pixel_m2')})
            for name, info in quality['sampling']['strata'].items():
                group = [p for p in points if p['stratum'] == name]
                if len(group) != info['sample_n'] or (info['population'] > 0 and not group):
                    raise ValueError(f'{scene_id}/{name}：样本数量与原抽样记录不一致')
                for p in group:
                    if p['population'] != info['population'] or p['sample_n'] != len(group):
                        raise ValueError(f'{scene_id}/{name}：抽样权重元数据不一致')
            if sum(i['population'] for i in quality['sampling']['strata'].values()) != quality['coverage']['valid_pixels']:
                raise ValueError(f'{scene_id}：抽样分层未覆盖全部有效像元')
            frame['scenes'].append({'scene_id': scene_id, 'date': date, 'sensor': result['sensor'],
                                    'strata': quality['sampling']['strata']})
            frame['points'].extend(points)
            index.append({'影像编号': scene_id, '日期': date, '传感器': result['sensor'],
                          '核验点数': len(points), '原始ORG': result['path'],
                          'ZIP内影像': result.get('member', ''),
                          '预测水体': str(folder / 'water.shp'), '预测栅格': str(folder / 'water.tif'),
                          '结果快视图': str(folder / 'quicklook.png'), '疑似区域图层': str(original),
                          '原质量JSON': str(folder / 'quality.json')})
            print(f'整理 {len(frame["scenes"])}/{len(results)} 景：{scene_id}，{len(points)} 点')
    ids = [p['point_id'] for p in frame['points']]
    if len(ids) != len(set(ids)):
        raise ValueError('原抽样结果存在重复点位编号')
    frame_path.write_text(json.dumps(frame, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.rename(gpkg)
    write_csv(review_dir / 'scene_index.csv', index)
    print(f'人工编辑图层：{gpkg}')


def weighted_metrics(points, references, area_weight=False):
    """分层扩展估计：每点代表 N_h/n_h 个像元；面积估计再乘像元面积。

    返回比率估计，不是把每层等量抽取的点直接混合计数。
    只能在全部指定点具有确定参考标签时调用。
    """
    counts = dict.fromkeys(('TN', 'FP', 'FN', 'TP'), 0.0)
    for p in points:
        ref = references[p['point_id']]
        if ref not in (0, 1):
            raise ValueError('有未核验或无法判读点，不能估计完整总体精度')
        weight = p['population'] / p['sample_n']
        if area_weight:
            weight *= p['pixel_m2']
        key = ('TN', 'FP', 'FN', 'TP')[2 * ref + p['pred']]
        counts[key] += weight
    tn, fp, fn, tp = (counts[key] for key in ('TN', 'FP', 'FN', 'TP'))
    def ratio(a, b):
        return a / b if b else None
    return {**counts, 'Precision': ratio(tp, tp + fp), 'Recall': ratio(tp, tp + fn),
            'F1': ratio(2 * tp, 2 * tp + fp + fn), 'IoU': ratio(tp, tp + fp + fn),
            'OA': ratio(tp + tn, tp + tn + fp + fn)}


def evaluate(review_dir):
    """统计核验进度；完整核验的景才报告精度，所有景完成后才汇总总体。"""
    review_dir = Path(review_dir)
    frame = json.loads((review_dir / 'sample_frame.json').read_text(encoding='utf-8'))
    references = {}
    with fiona.open(review_dir / 'manual_review.gpkg', layer='verification_points') as source:
        for feature in source:
            p = feature['properties']
            key, value = p['point_id'], p['ref_class']
            if key in references or value not in (None, -1, 0, 1):
                raise ValueError(f'点编号重复或参考类别不合法：{key}')
            references[key] = value
    expected = {p['point_id'] for p in frame['points']}
    if set(references) != expected:
        raise ValueError('核验点有删除、增加或编号被修改，请恢复原抽样点；不能删掉难判点。')
    progress, metrics = [], []
    complete_scenes = 0
    for scene in frame['scenes']:
        points = [p for p in frame['points'] if p['scene_id'] == scene['scene_id']]
        for name in scene['strata']:
            group = [p for p in points if p['stratum'] == name]
            counts = Counter(references[p['point_id']] for p in group)
            progress.append({'影像编号': scene['scene_id'], '日期': scene['date'], '分层': name,
                             '点数': len(group), '已判非水体': counts[0], '已判水体': counts[1],
                             '无法判读': counts[-1], '未核验': counts[None]})
        complete = bool(points) and all(references[p['point_id']] in (0, 1) for p in points)
        complete_scenes += int(complete)
        for scope in ('all_valid', 'shore_buffer'):
            subset = points if scope == 'all_valid' else [p for p in points if p['stratum'] == scope]
            for area_weight in (False, True):
                values = weighted_metrics(subset, references, area_weight) if complete else dict.fromkeys(
                    ('TN', 'FP', 'FN', 'TP', 'Precision', 'Recall', 'F1', 'IoU', 'OA'))
                metrics.append({'scene_id': scene['scene_id'], 'date': scene['date'], 'scope': scope,
                                'weight_unit': 'm2' if area_weight else 'pixels',
                                'status': 'estimated' if complete else 'awaiting_reference', **values})
    ready = bool(frame['scenes']) and complete_scenes == len(frame['scenes'])
    overall = []
    if ready:
        for scope in ('all_valid', 'shore_buffer'):
            subset = frame['points'] if scope == 'all_valid' else [p for p in frame['points'] if p['stratum'] == scope]
            for area_weight in (False, True):
                overall.append({'scope': scope, 'weight_unit': 'm2' if area_weight else 'pixels',
                                **weighted_metrics(subset, references, area_weight)})
    summary = {'status': 'estimated' if ready else 'awaiting_reference',
               'completed_scenes': complete_scenes, 'total_scenes': len(frame['scenes']),
               'points': len(references), 'reviewed': sum(v in (0, 1) for v in references.values()),
               'uncertain': sum(v == -1 for v in references.values()),
               'not_reviewed': sum(v is None for v in references.values()),
               'population': frame['scope'], 'overall': overall,
               'notes': ['重叠景和多日期按观测次数计入，不是去重后的密云区面积精度。',
                         'shore_buffer 是预测岸线附近分类精度，不是容差边界F1或边界距离。',
                         '当前输出为抽样点估计，不含置信区间；少量样本未观测到错误不代表没有错误。',
                         '只用SAR判读得到的是人工一致性评估；独立同期参考才能更可靠地支持真实精度结论。']}
    write_csv(review_dir / 'review_progress.csv', progress)
    write_csv(review_dir / 'accuracy_by_scene.csv', metrics)
    (review_dir / 'evaluation.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                                                         allow_nan=False), encoding='utf-8')
    print(f'确定标签 {summary["reviewed"]}/{summary["points"]}，无法判读 {summary["uncertain"]}；'
          f'完整核验 {complete_scenes}/{len(frame["scenes"])} 景')
    print('精度已估计。' if ready else '等待人工核验；未完成范围不输出精度值。')
    return summary


# 执行设置：首次 prepare；在 QGIS 保存人工标签并关闭图层编辑后改为 evaluate。
MODE = 'prepare'  # prepare / evaluate
RESULTS_DIR = Path(r'E:\InSAR\水体提取\北京密云区\outputs\thres_seg')
REVIEW_DIR = Path(r'E:\InSAR\水体提取\北京密云区\outputs\manual_review')

if __name__ == '__main__':
    if MODE == 'prepare':
        prepare(RESULTS_DIR, REVIEW_DIR)
        evaluate(REVIEW_DIR)
    elif MODE == 'evaluate':
        evaluate(REVIEW_DIR)
    else:
        raise ValueError('MODE 只能为 prepare 或 evaluate')
