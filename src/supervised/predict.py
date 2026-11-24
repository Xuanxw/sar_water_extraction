"""无标签整景推理：python src/supervised/predict.py <配置.yaml>。

复用模型重建、重叠logit融合和掩膜后处理。只读取已准备的GeoTIFF，不训练、
不读取标签/划分清单、不启动SNAP，也不自动转换幅度、功率或dB。
"""

from contextlib import ExitStack, closing
from pathlib import Path
import json
import math
import shutil
import sys
from time import perf_counter

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.special import expit
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configuration import config_argument, load_yaml
from datasets.relative_normalization import (
    RELATIVE_REPRESENTATIONS, apply_vv_train_p02_p98_dem_m, validate_relative_normalization,
)
from supervised.models.checkpoint import load_model
from supervised.evaluation.sliding_inference import iter_scene_predictions
from supervised.evaluation.mask_postprocess import postprocess_water_raster
from thres_segmentation.run_log import RunLog
from utils.files import sha256, signature, write_json
from utils.water_presentation import save_raster_water_presentation
from utils.water_vectors import write_water_polygons


def prediction_settings(checkpoint, config):
    """按检查点恢复通道、标准化与窗口表示；YAML仅调整显式允许的推理参数。"""
    inputs, inference = config['input'], config['inference']
    bands = list(inputs.get('BANDS') or checkpoint['bands'])
    if len(bands) != len(checkpoint['bands']):
        raise ValueError('输入波段数量必须与检查点一致；BANDS只用于对应通道的文件波段重映射')
    normalized = inputs['INPUTS_NORMALIZED']
    if not isinstance(normalized, bool) or normalized != checkpoint.get('inputs_normalized', False):
        raise ValueError(f'INPUTS_NORMALIZED须为{checkpoint.get("inputs_normalized", False)}，'
                         '与检查点一致；不能漏做或重复做训练均值/标准差标准化')
    if checkpoint.get('speckle') and not inputs.get('FILTER_ALREADY_APPLIED', False):
        raise ValueError('该检查点需要滤波输入，请先完成匹配的离线滤波，并设置FILTER_ALREADY_APPLIED=true')
    mean = torch.as_tensor(checkpoint['mean']).flatten().tolist()
    std = torch.as_tensor(checkpoint['std']).flatten().tolist()
    if (len(mean) != len(bands) or len(std) != len(bands)
            or not np.isfinite([mean, std]).all() or min(std) <= 0):
        raise ValueError('检查点缺少匹配通道的有限均值和正标准差')
    size = inference.get('WINDOW_SIZE') or checkpoint.get('window_size', 512)
    halo = inference.get('HALO')
    halo = checkpoint.get('window_halo', 64) if halo is None else halo
    saved = checkpoint.get('validation_inference') or {}
    stride = inference.get('STRIDE') or saved.get('stride') or (size-2*halo)//2
    blend = inference.get('BLEND') or saved.get('blend', 'gaussian')
    representation = checkpoint.get('patch_representation', 'stored')
    transform = None
    if representation in ('relative_backscatter_ranks', 'vv_rank_dem_m'):
        from datasets.dfc_sample.s1_dataset import relative_backscatter_ranks, vv_rank_dem_m
        operation = vv_rank_dem_m if representation == 'vv_rank_dem_m' else relative_backscatter_ranks
        transform = lambda image, valid: (operation(image, valid), valid)
    elif representation in RELATIVE_REPRESENTATIONS:
        parameters = validate_relative_normalization(checkpoint.get('relative_normalization'), representation)
        if list(checkpoint['bands']) != parameters['bands']:
            raise ValueError('检查点波段与拟合 VV/DEM 时的顺序不一致')
        transform = lambda image, valid: (apply_vv_train_p02_p98_dem_m(image, valid, parameters), valid)
    elif representation != 'stored':
        raise ValueError(f'尚未适配的检查点输入表示：{representation}')
    if transform is not None and normalized:
        raise ValueError('窗级相对表示不能再次应用于已经标准化的影像')
    settings = dict(bands=bands, window_size=size, halo=halo, stride=stride, blend=blend,
                    mean=mean, std=std, inputs_normalized=normalized, patch_representation=representation,
                    relative_normalization=checkpoint.get('relative_normalization'),
                    batch_size=inference['BATCH_SIZE'], num_workers=inference.get('NUM_WORKERS', 0))
    return settings, transform


def predict_scene(model, image_path, destination, settings, config, *, checkpoint_info,
                  device='cpu', window_transform=None, log=print):
    """单景分块推理并发布成果；quality.json最后写入，只有完整景才允许续跑复用。

    输出保持输入尺寸、CRS和transform。水体1、非水体0、未知255；真实0值按
    GDAL有效掩膜读取，不自动视为无效。质量统计来自全景原生像元，不含真值精度。
    """
    image_path, destination = Path(image_path), Path(destination)
    options = config['output']
    threshold = config['inference']['THRESHOLD']
    if not 0 < threshold < 1:
        raise ValueError('水体概率阈值必须在(0,1)内')
    if settings.get('patch_representation', 'stored') != 'stored' and window_transform is None:
        raise ValueError('检查点要求逐窗复现训练输入，缺少 window_transform')
    post = config['postprocess']
    exclusions = config['paths'].get('EXCLUSION_MASKS') or []
    source_stat = image_path.stat()
    identity = dict(version=1, checkpoint=checkpoint_info['sha256'],
                    source=dict(size=source_stat.st_size, mtime_ns=source_stat.st_mtime_ns),
                    settings=settings, threshold=threshold, postprocess=post, output=options,
                    amp=config['inference']['USE_AMP'],
                    exclusions=[dict(size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for p in exclusions])
    representation = settings.get('patch_representation', 'stored')
    if representation in ('vv_rank_dem_m', *RELATIVE_REPRESENTATIONS):
        with rasterio.open(image_path) as source:
            if not all(b in source.indexes for b in settings['bands']):
                raise ValueError(f'输入波段超出影像范围：{image_path}')
            vv, dem = [band - 1 for band in settings['bands']]
            unit = (source.units[vv] or '').lower()
            description = (source.descriptions[vv] or '').lower().replace('_', ' ').replace('-', ' ').split()
            if representation in RELATIVE_REPRESENTATIONS and (unit in ('db', 'decibel', 'decibels') or 'db' in description):
                raise ValueError('DFC 存储值模型与已声明的 dB 产品不匹配，请先核实单位映射')
            scale, offset = source.scales[dem], source.offsets[dem]
        if scale != 1.0 or offset != 0.0:
            # 只在实际需要单位转换时改变复用签名；已存为米的旧结果可继续复用。
            identity['dem_scale_offset'] = [scale, offset]
            operation = window_transform

            def window_transform(raw, valid):
                raw[1] = raw[1] * scale + offset
                return operation(raw, valid)

    key = signature(identity)
    marker = destination / 'quality.json'
    required = ['water_raw.tif', 'water.tif', 'water_polygon.shp', 'water_polygon.shx',
                'water_polygon.dbf', 'water_polygon.prj', 'water_polygon.cpg', 'quicklook.png']
    if options['SAVE_PROBABILITY']:
        required.append('probability.tif')
    protected = {image_path.resolve(), *(Path(path).resolve() for path in exclusions)}
    if checkpoint_info.get('checkpoint'):
        protected.add(Path(checkpoint_info['checkpoint']).resolve())
    if any((destination / name).resolve() in protected for name in required):
        raise ValueError('景级输出会覆盖输入影像、检查点或排除图，请更换输出目录')
    if marker.exists():
        previous = json.loads(marker.read_text(encoding='utf-8'))
        if previous.get('complete'):
            if previous.get('key') != key:
                raise ValueError(f'{destination}已有不同参数的完整成果，请设置新的OUTPUT_DIR')
            if config['run']['RESUME'] and all((destination / name).is_file() for name in required):
                log(f'复用完整景：{image_path.name}')
                return previous
            # 完整但缺少成果时重建该景；不因不完整PNG或SHP判为已完成。
    destination.mkdir(parents=True, exist_ok=True)
    if marker.exists():
        marker.unlink()
    started = perf_counter()
    raw_pending = destination / 'water_raw.partial.tif'
    probability_pending = destination / 'probability.partial.tif'
    use_amp = bool(config['inference']['USE_AMP'] and torch.device(device).type == 'cuda')
    report = dict(complete=False, key=key, image=str(image_path), model=checkpoint_info,
                  inference={**settings, 'threshold': threshold, 'amp_used': use_amp},
                  input_description=config['input'].get('DESCRIPTION', ''), postprocess=post,
                  exclusion_masks=[str(path) for path in exclusions],
                  accuracy=dict(available=False, reason='未提供真值，不能计算F1、IoU、Precision或Recall'))
    try:
        with rasterio.open(image_path) as source:
            if source.crs is None:
                raise ValueError(f'输入影像缺少地理参考：{image_path}')
            if not all(isinstance(b, int) and 1 <= b <= source.count for b in settings['bands']):
                raise ValueError(f'输入波段超出影像范围：{image_path}')
            if any(np.issubdtype(np.dtype(source.dtypes[b-1]), np.complexfloating) for b in settings['bands']):
                raise ValueError('输入必须是已地理编码的实数SAR，不接收复数SLC')
            profile = dict(driver='GTiff', width=source.width, height=source.height, count=1,
                           crs=source.crs, transform=source.transform, dtype='uint8', nodata=255,
                           tiled=True, blockxsize=256, blockysize=256, compress='deflate', BIGTIFF='IF_SAFER')
            report['grid'] = dict(crs=str(source.crs), shape=list(source.shape), transform=list(source.transform))
            # 概览只读缩略图；推理使用滚动融合缓冲，不创建整景张量。
            call_options = {key: value for key, value in settings.items()
                            if key not in ('patch_representation', 'relative_normalization')}
            blocks = iter_scene_predictions(model, image_path, device=device, use_amp=use_amp,
                boundary_context=False, return_labels=False, window_transform=window_transform, **call_options)
            logit_threshold = math.log(threshold / (1-threshold))
            probability_sum = 0.0
            predicted_pixels = 0
            next_progress = 10
            with ExitStack() as stack:
                stack.enter_context(closing(blocks))
                raw = stack.enter_context(rasterio.open(raw_pending, 'w', **profile))
                probability = None
                if options['SAVE_PROBABILITY']:
                    probability = stack.enter_context(rasterio.open(probability_pending, 'w',
                        **{**profile, 'dtype': 'float32', 'nodata': -1.0, 'predictor': 3}))
                for block in blocks:
                    top, left, height, width = block['window']
                    window = Window(left, top, width, height)
                    valid = block['valid_mask']
                    # 直接比较logit，避免float32概率在0.5附近舍入后改变类别。
                    water = block['scores'] >= logit_threshold
                    raw.write(np.where(valid, water, 255).astype('uint8'), 1, window=window)
                    scores = expit(block['scores'].astype('float64'))
                    probability_sum += float(scores[valid].sum())
                    predicted_pixels += int(valid.sum())
                    if probability is not None:
                        probability.write(np.where(valid, scores, -1).astype('float32'), 1, window=window)
                    progress = 100 * (top+height) // source.height
                    if left+width == source.width and progress >= next_progress:
                        log(f'{image_path.name}：融合预测 {top+height}/{source.height}行（{progress}%）')
                        next_progress = 10 * (1+progress//10)
                raw.update_tags(water='1', nonwater='0', unknown='255', threshold=threshold,
                                checkpoint_sha256=checkpoint_info['sha256'], inference='overlap_logit_blend')
        raw_pending.replace(destination / 'water_raw.tif')
        if probability is not None:
            probability_pending.replace(destination / 'probability.tif')
        log('后处理：开闭运算与显式排除掩膜')
        pending = destination / 'water.partial.tif'
        counts = postprocess_water_raster(destination / 'water_raw.tif', pending,
            opening_radius=post['OPENING_RADIUS'], closing_radius=post['CLOSING_RADIUS'],
            exclusion_masks=exclusions, block_size=post['BLOCK_SIZE'])
        pending.replace(destination / 'water.tif')
        report['counts'] = counts
        report['water_fraction'] = counts['water_pixels'] / counts['valid_pixels'] if counts['valid_pixels'] else None
        report['water_area_km2'] = counts['water_area_m2'] / 1e6
        report['mean_water_probability'] = probability_sum / predicted_pixels if predicted_pixels else None
        report['probability_scope'] = 'raw_model_prediction_before_postprocessing'
        log('面化：保留原生像元边界与孔洞，不做平滑')
        report['polygon_count'] = write_water_polygons(destination / 'water.tif', destination / 'water_polygon.shp', log)
        log('生成水体快视图')
        save_raster_water_presentation(image_path, destination / 'water.tif', destination / 'quicklook.png',
            max_side=options['QUICKLOOK_SIZE'], band=settings['bands'][0],
            input_caption=f'SAR file band {settings["bands"][0]} (P2-P98 display)',
            title=f'{image_path.stem} | Predicted water | threshold={threshold:g} | No ground truth')
        report.update(complete=True, outputs=required, elapsed_seconds=perf_counter()-started,
                      note='面积按栅格原生网格计算；概率未必经过校准，快视图仅用于显示。不自动计算DEM/土地覆盖/历史水体修正，仅应用所列显式排除图。')
        write_json(marker, report)
        return report
    except BaseException as error:
        report.update(error=str(error), elapsed_seconds=perf_counter()-started)
        write_json(marker, report)
        raise


def main(config_path):
    """加载一次检查点，逐景执行；新影像不需要标签、manifest或训练时的文件名单。"""
    config = load_yaml(config_path)
    paths = config['paths']
    checkpoint_path, image_path, output = paths['CHECKPOINT_PATH'], paths['IMAGE_PATH'], paths['OUTPUT_DIR']
    if not all(path is not None for path in (checkpoint_path, image_path, output)):
        raise ValueError('请在YAML中填写CHECKPOINT_PATH、IMAGE_PATH和OUTPUT_DIR')
    if output.resolve() == image_path.resolve():
        raise ValueError('输出目录不能与输入目录相同')
    if image_path.is_file():
        images = [image_path]
    else:
        candidates = image_path.rglob('*') if config['run']['RECURSIVE'] else image_path.glob('*')
        images = sorted(path for path in candidates if path.is_file() and path.suffix.lower() in ('.tif', '.tiff')
                        and not path.resolve().is_relative_to(output.resolve()))
    if not images or any(path.suffix.lower() not in ('.tif', '.tiff') for path in images):
        raise ValueError('IMAGE_PATH应为GeoTIFF文件或含GeoTIFF的目录')
    if len({path.stem for path in images}) != len(images):
        raise ValueError('不同目录存在同名影像，请分别执行，避免景级成果混用')
    output.mkdir(parents=True, exist_ok=True)
    log = RunLog(output, console_color=config['run'].get('CONSOLE_COLOR', True))
    with log.capture(), rasterio.Env(GDAL_CACHEMAX=64*1024*1024):
        with log.stage('01', '加载模型与输入配置'):
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True, mmap=True)
            settings, transform = prediction_settings(checkpoint, config)
            device = config['run']['DEVICE']
            device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
            model = load_model(checkpoint).to(device).eval()
            info = dict(name=checkpoint.get('model_name', 'unet'), epoch=checkpoint.get('epoch'),
                        checkpoint=str(checkpoint_path), sha256=sha256(checkpoint_path))
            del checkpoint
            shutil.copyfile(config_path, output / f'predict_config_{log.run_id}.yaml')
            log.message(f'设备：{device}；{len(images)}景；输出：{output}')
            log.message(f'输入波段：{settings["bands"]}；已标准化：{settings["inputs_normalized"]}；标签：不读取')
        reports = []
        for index, image in enumerate(images, 1):
            with log.stage('02', f'整景推理及成果 {index}/{len(images)}', scene=image.stem):
                report = predict_scene(model, image, output / image.stem, settings, config,
                    checkpoint_info=info, device=device, window_transform=transform, log=log.message)
                reports.append(report)
        log.message('全部完成；每景quality.json为完整成果标记，processing.log与stage_timings.csv记录本次过程')
        return reports


if __name__ == '__main__':
    main(config_argument())
