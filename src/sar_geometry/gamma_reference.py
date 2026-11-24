"""GAMMA ls_map编码与同网格逐像元比较；不重采样、不改动输入成果。"""

from pathlib import Path
from contextlib import ExitStack

import numpy as np
import rasterio

from utils.files import sha256, write_json
from utils.raster import windows

GAMMA_VERSION = 'gc_map2 v5.7 28-Jul-2023'
GAMMA_BINARY_SHA256 = '6de93a4c12591753545aae9e9787f8041ba147e046ae03321737eb83c4262676'


def decode_gc_map2_ls_map(values, ls_scaling=0):
    """UCHAR分类→(shadow, layover)，分别返回0/1/255；255代表本接口的未知。

    ls_scaling=0按位4/16读取，1使用显示码表。显示码255是叠掩，不能按常见
    GeoTIFF NoData习惯丢弃。未解释的位组合保持未知，调用者应记录原始码直方图。
    """
    values = np.asarray(values, dtype='uint8')
    shadow = np.full(values.shape, 255, dtype='uint8')
    layover = np.full(values.shape, 255, dtype='uint8')
    if ls_scaling == 0:
        known = np.isin(values, [1, 4, 5, 16, 17, 20, 21])
        shadow[known] = (values[known] & 16) != 0
        layover[known] = (values[known] & 4) != 0
    elif ls_scaling == 1:
        for code, pair in {127: (0, 0), 255: (0, 1), 31: (1, 0), 63: (1, 1)}.items():
            shadow[values == code], layover[values == code] = pair
    else:
        raise ValueError('ls_scaling只能为0（处理码）或1（显示码）')
    return shadow, layover


def encode_gc_map2_ls_map(shadow, layover, ls_scaling=0):
    """项目的独立0/1/255掩膜→GAMMA兼容码；返回合成码及双项已知标记。

    任一项未知时合成码为0，保留原始独立掩膜才能表达部分已知状态。处理码采用
    1/4/16/20的规范代表值，不强行给叠掩/阴影附加tested位，不承诺最终产品字节相等。
    """
    shadow, layover = np.broadcast_arrays(shadow, layover)
    if not np.isin(shadow, [0, 1, 255]).all() or not np.isin(layover, [0, 1, 255]).all():
        raise ValueError('独立阴影/叠掩掩膜只能包含0/1/255')
    known = (shadow != 255) & (layover != 255)
    values = np.zeros(shadow.shape, dtype='uint8')
    if ls_scaling == 0:
        values[known] = shadow[known] * 16 + layover[known] * 4
        values[known & (values == 0)] = 1
    elif ls_scaling == 1:
        for pair, code in {(0, 0): 127, (0, 1): 255, (1, 0): 31, (1, 1): 63}.items():
            values[known & (shadow == pair[0]) & (layover == pair[1])] = code
    else:
        raise ValueError('ls_scaling只能为0或1')
    return values, known


def compare_gc_map2_masks(config, log=print):
    """按原生像元比较地图坐标ls_map与独立阴影/叠掩GeoTIFF，分块计数。

    GAMMA输入可为UCHAR原始文件，但必须另给经GAMMA导出的同网格GeoTIFF，
    不猜测DEM_par角点约定。禁止将ORG地图网格与ls_map_rdc雷达网格直接比较。
    输出差异图中0=双方未知、1=已知一致、2=本实现多排除、3=本实现少排除、
    4=本实现未知而GAMMA已知、5=GAMMA未知而本实现已知。
    """
    paths = config['paths']
    options = config['reference']
    if options['version'] != GAMMA_VERSION:
        raise ValueError(f'本对照固定为{GAMMA_VERSION}，不混用其他版本')
    if options['grid_domain'] != 'map':
        raise ValueError('本接口比较地图网格ls_map；ls_map_rdc需要独立雷达网格对照')
    if not paths.get('GAMMA_LS_MAP'):
        raise ValueError('请先提供v5.7参考ls_map；当前尚无同输入参考成果')
    output = Path(paths['OUTPUT_DIR'])
    sources = [Path(paths[key]).resolve() for key in ('SHADOW_PATH', 'LAYOVER_PATH', 'GAMMA_LS_MAP')]
    # 对照输出必须独立，不能覆写输入目录里的掩膜或水体成果。
    output_files = [output / 'shadow_difference.tif', output / 'layover_difference.tif', output / 'gamma_comparison.json']
    if any(target.resolve() in sources for target in output_files):
        raise ValueError('对照输出路径与输入冲突')
    counts = {'shadow': np.zeros(6, dtype='int64'), 'layover': np.zeros(6, dtype='int64')}
    confusion = {key: np.zeros(4, dtype='int64') for key in counts}
    histogram = np.zeros(256, dtype='int64')
    reference_path = Path(paths['GAMMA_LS_MAP'])
    raw = None
    with ExitStack() as stack:
        shadow_source = stack.enter_context(rasterio.open(paths['SHADOW_PATH']))
        layover_source = stack.enter_context(rasterio.open(paths['LAYOVER_PATH']))
        if options['format'] == 'uchar':
            grid = stack.enter_context(rasterio.open(paths['GAMMA_GRID_TIF']))
            if reference_path.stat().st_size != grid.width * grid.height:
                raise ValueError('UCHAR文件字节数与参考网格尺寸不符')
            raw = np.memmap(reference_path, mode='r', dtype='uint8', shape=(grid.height, grid.width))
        elif options['format'] == 'geotiff':
            grid = stack.enter_context(rasterio.open(reference_path))
            if grid.dtypes[0] != 'uint8':
                raise ValueError('GAMMA ls_map应为UCHAR/uint8；不自动转换其他产品')
        else:
            raise ValueError('reference.format只能为uchar或geotiff')
        for source in (shadow_source, layover_source):
            if (source.crs is None or source.crs != grid.crs or source.transform != grid.transform
                    or source.shape != grid.shape or source.count != 1):
                raise ValueError('参考与项目掩膜必须是完全相同的地图网格；禁止静默重采样')
        profile = grid.profile.copy()
        profile.update(driver='GTiff', count=1, dtype='uint8', nodata=None, compress='deflate')
        output.mkdir(parents=True, exist_ok=True)
        destinations = {
            key: stack.enter_context(rasterio.open(output / f'{key}_difference.tif', 'w', **profile))
            for key in counts
        }
        total = 0
        for window in windows(grid, config.get('block_size', 1024)):
            rows, cols = window.toslices()
            codes = raw[rows, cols] if raw is not None else grid.read(1, window=window)
            histogram += np.bincount(codes.ravel(), minlength=256)
            reference = decode_gc_map2_ls_map(codes, options['ls_scaling'])
            for key, source, expected in zip(counts, (shadow_source, layover_source), reference):
                actual = source.read(1, window=window)
                if not np.isin(actual, [0, 1, 255]).all():
                    raise ValueError(f'{key}掩膜有0/1/255以外的类别')
                actual = actual.astype('uint8', copy=False)
                actual_known = actual != 255
                reference_known = expected != 255
                common = actual_known & reference_known
                difference = np.zeros(actual.shape, dtype='uint8')
                difference[common & (actual == expected)] = 1
                difference[common & (actual > expected)] = 2
                difference[common & (actual < expected)] = 3
                difference[~actual_known & reference_known] = 4
                difference[actual_known & ~reference_known] = 5
                counts[key] += np.bincount(difference.ravel(), minlength=6)
                # tn/fp/fn/tp仅表示相对于GAMMA的分类关系，不是相对于地物真值。
                confusion[key] += np.bincount(2 * expected[common] + actual[common], minlength=4)
                destinations[key].write(difference, 1, window=window)
            total += int(window.width * window.height)
        log(f'完成{total:,}个原生像元对照；未知范围单独计数')
        grid_info = dict(crs=str(grid.crs), transform=list(grid.transform), shape=list(grid.shape))
    if raw is not None:
        del raw
    metrics = {}
    for key in counts:
        tn, fp, fn, tp = map(int, confusion[key])
        common = tn + fp + fn + tp
        metrics[key] = dict(
            both_unknown=int(counts[key][0]), known_agree=int(counts[key][1]),
            more_exclusion=int(counts[key][2]), less_exclusion=int(counts[key][3]),
            ours_unknown=int(counts[key][4]), gamma_unknown=int(counts[key][5]),
            common_known=common, confusion_against_gamma=dict(tn=tn, fp=fp, fn=fn, tp=tp),
            known_agreement=(tn + tp) / common if common else None,
            exclusion_iou=tp / (tp + fp + fn) if tp + fp + fn else None,
        )
    input_paths = dict(paths)
    report = dict(
        target_version=GAMMA_VERSION, target_binary_sha256=GAMMA_BINARY_SHA256,
        reference_provenance='user supplied; original v5.7 executable not available for re-execution',
        same_inputs_confirmed=bool(options.get('same_inputs_confirmed', False)),
        numerical_equivalence_verified=False,
        interpretation='classification agreement with GAMMA, not ground-truth accuracy or full-chain equivalence',
        grid=grid_info, metrics=metrics,
        gamma_code_counts={str(i): int(n) for i, n in enumerate(histogram) if n},
        inputs={key: dict(path=str(value), sha256=sha256(value)) for key, value in input_paths.items()
                if key != 'OUTPUT_DIR' and value},
    )
    write_json(output / 'gamma_comparison.json', report)
    return report
