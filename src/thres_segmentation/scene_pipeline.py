"""BC整景阈值：分块读写、整景统一百分位、原始元数据与稀疏标签评价。"""
import csv
import json
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
from scipy.ndimage import uniform_filter

from thres_segmentation.metadata import read_source_metadata
from thres_segmentation.pipeline import write_csv, save_figure
from thres_segmentation.threshold import ThresholdSegmenter, read_input_preprocessing
from supervised.evaluation.boundary_eval import boundary_regions, confusion_counts
from supervised.evaluation.metrics import metrics_from_counts


def windows(source, size):
    """整景核心窗口互不重叠；分块只控制I/O，不改变像元网格。"""
    for top in range(0, source.height, size):
        for left in range(0, source.width, size):
            yield Window(left, top, min(size, source.width-left), min(size, source.height-top))


def read_window(source, band, core, halo=0):
    """读取核心及邻域，返回数组、有效掩膜和核心切片；有限0值保留。"""
    left, top = max(0, int(core.col_off)-halo), max(0, int(core.row_off)-halo)
    right = min(source.width, int(core.col_off+core.width)+halo)
    bottom = min(source.height, int(core.row_off+core.height)+halo)
    region = Window(left, top, right-left, bottom-top)
    data = source.read(band, window=region, masked=True, out_dtype="float64")
    valid = ~np.ma.getmaskarray(data) & np.isfinite(data.data)
    crop = (slice(int(core.row_off)-top, int(core.row_off+core.height)-top),
            slice(int(core.col_off)-left, int(core.col_off+core.width)-left))
    return data.filled(0), valid, crop, region


def disk_percentiles(path, count, percentages):
    """在磁盘映射的一维有效值缓冲上原地选择分位点，与numpy线性百分位规则一致。"""
    if count == 0:
        raise ValueError("整景没有有效SAR像素")
    ranks = (count-1)*np.asarray(percentages, dtype="float64")/100
    lower, upper = np.floor(ranks).astype('int64'), np.ceil(ranks).astype('int64')
    values = np.memmap(path, dtype="float64", mode="r+", shape=(count,))
    try:
        values.partition(np.unique(np.concatenate((lower, upper))))
        lo, hi = np.array(values[lower]), np.array(values[upper])
        return lo+(hi-lo)*(ranks-lower), lo, hi, ranks-lower
    finally:
        values.flush()
        del values


def filtered_windows(source, band, size, segmenter, median, variance):
    """halo消除块边界滤波接缝，噪声方差和无效填充值使用全景统计量。"""
    # 离线模式LEE_WINDOW=1，直接读取z-score，不进行任何数值变换。
    halo = segmenter.lee_window//2
    for core in windows(source, size):
        data, valid, crop, _ = read_window(source, band, core, halo)
        data[~valid] = median
        if segmenter.lee_window > 1:
            mean = uniform_filter(data, segmenter.lee_window, mode="mirror")
            local_var = np.maximum(uniform_filter(data*data, segmenter.lee_window, mode="mirror")-mean*mean, 0)
            weight = local_var/(local_var+variance*segmenter.noise_ratio+1e-10)
            data = mean+weight*(data-mean)
        data, valid = data[crop], valid[crop]
        if segmenter.power_to_db:
            if np.any(data[valid] < 0):
                raise ValueError("负值不能按线性功率执行10log10；先确认数据的物理含义")
            data = 10*np.log10(data+1e-6)
        yield core, data, valid


def scene_threshold(source, band, size, segmenter, scratch):
    """两遍扫描：原始全景统计→Lee后全景分位数；临时值缓冲逐阶段复用。"""
    buffer = Path(scratch)/"valid_values.bin"
    if segmenter.preprocessing == 'offline_normalized':
        # 一遍写入有效z-score；全景只计算决策百分位，免去旧Lee的两遍统计。
        count = 0
        with buffer.open('wb') as stream:
            for core in windows(source, size):
                data, valid, _, _ = read_window(source, band, core)
                values = data[valid]
                values.tofile(stream)
                count += len(values)
        quantile = segmenter.water_percent if segmenter.mode == 'bottom' else 100-segmenter.water_percent
        threshold = float(disk_percentiles(buffer, count, [quantile])[0][0])
        buffer.unlink()
        print(f"  有效SAR像素={count}；复用离线滤波/z-score，仅计算全景阈值", flush=True)
        return 0., 0., None, None, threshold
    count, mean, m2 = 0, 0., 0.
    with buffer.open('wb') as stream:
        for core in windows(source, size):
            data, valid, _, _ = read_window(source, band, core)
            values = data[valid]
            values.tofile(stream)
            n = len(values)
            if n:
                block_mean = float(values.mean())
                delta = block_mean-mean
                m2 += float(np.square(values-block_mean).sum()) + delta*delta*count*n/(count+n)
                mean += delta*n/(count+n)
                count += n
    median = float(disk_percentiles(buffer, count, [50])[0][0])
    variance = m2/count
    print(f"  有效SAR像素={count}；全景Lee统计完成", flush=True)
    with buffer.open('wb') as stream:
        for _, data, valid in filtered_windows(source, band, size, segmenter, median, variance):
            data[valid].tofile(stream)
    quantile = segmenter.water_percent if segmenter.mode == "bottom" else 100-segmenter.water_percent
    percentile, lo, hi, fraction = disk_percentiles(buffer, count, [*segmenter.stretch, quantile])
    low, high = percentile[:2]
    # 在归一化后的两个相邻秩上插值，保持裁剪后的百分位定义，包括并列值和端点。
    scale = lambda x: np.clip((x-low)/(high-low+1e-6), 0, 1)
    threshold = float(scale(lo[2])+(scale(hi[2])-scale(lo[2]))*fraction[2])
    buffer.unlink()
    return median, variance, float(low), float(high), threshold


def run_scene_segmentation(image_dir, label_dir, split_manifest, output_dir, bands=None,
                           threshold_options=None, block_size=1024, cache_dir=None,
                           max_visualizations=None, preview_size=1200, save_masks=True,
                           scene_prefix=None, boundary_radii=(1,3,5), analysis_options=None,
                           scene_analysis_options=None):
    """仅使用清单中的active val；原始路径仅提供meta，标签不参与分割阈值。"""
    # 1. 数据准备：锁定active val、原生网格和逐景辅助路径；下载由main统一完成。
    image_dir, output_dir, split_manifest = Path(image_dir), Path(output_dir), Path(split_manifest)
    label_dir = Path(label_dir) if label_dir is not None else None
    bands = bands or {"VV":1}
    if image_dir.resolve() != (split_manifest.parent/'images/val').resolve():
        raise ValueError("整景评价应输入清单对应的images/val")
    if label_dir is not None and label_dir.resolve() != (split_manifest.parent/'labels/val').resolve():
        raise ValueError("整景评价应输入清单对应的labels/val")
    with split_manifest.open(encoding='utf-8-sig',newline='') as stream:
        records = [r for r in csv.DictReader(stream) if r['split']=='val' and int(r['active'])==1
                   and (scene_prefix is None or r['name'].startswith(scene_prefix))]
    if not records:
        raise ValueError("没有符合清单与scene_prefix的验证影像")
    segmenter = ThresholdSegmenter(**(threshold_options or {}))
    if segmenter.lee_window < 1 or segmenter.lee_window % 2 != 1 or block_size < 1 or preview_size < 1:
        raise ValueError("Lee窗口需为正奇数，读取窗口和预览边长需为正数")
    if segmenter.mode not in ('bottom','top') or not 0 <= segmenter.water_percent <= 100:
        raise ValueError("阈值模式为bottom/top，百分位比例为0～100")
    if segmenter.stretch is not None and (len(segmenter.stretch)!=2 or not 0 <= segmenter.stretch[0] < segmenter.stretch[1] <= 100):
        raise ValueError("stretch应为0～100内递增的两个百分位")
    if any(radius <= 0 for radius in boundary_radii):
        raise ValueError("边界半径需要为正数")
    preprocessing = read_input_preprocessing(split_manifest.parent, segmenter, list(bands.values()))
    output_dir.mkdir(parents=True,exist_ok=False)
    if preprocessing:
        (output_dir/"input_preprocessing.json").write_text(json.dumps(preprocessing, ensure_ascii=False, indent=2), encoding="utf-8")
    cache_dir = Path(cache_dir) if cache_dir else output_dir
    cache_dir.mkdir(parents=True,exist_ok=True)
    write_csv(output_dir/'files.csv',records)
    write_csv(output_dir/'methods.csv',[dict(method='threshold_'+name,band=band,parameters=str(vars(segmenter)),
        input_domain='z_score' if preprocessing else 'legacy_stretched_0_1', threshold_scope='whole_scene_exact_percentile', block_size=block_size,
        boundary_radii=str(boundary_radii), preview_size=preview_size, scene_prefix=scene_prefix) for name,band in bands.items()])
    all_counts, all_boundary, rows, boundary_rows, metadata_rows = {}, {}, [], [], []
    print(f"BC整景阈值：{len(records)}景；{bands}；meta沿source_image读取；不使用checkpoint",flush=True)
    with rasterio.Env(GDAL_CACHEMAX=64*1024*1024, GDAL_NUM_THREADS='1'):
        for index, record in enumerate(records):
            name = record['name']
            scene_options = scene_analysis_options[name] if scene_analysis_options is not None else (analysis_options or {})
            print(f"开始 {index+1}/{len(records)}：{name}",flush=True)
            meta, original, xml = read_source_metadata(record['source_image'])
            metadata_rows.append(dict(name=name,**meta))
            write_csv(output_dir/'metadata.csv',metadata_rows)
            if xml:
                path=output_dir/'metadata'/Path(name).with_suffix('.xml'); path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(xml)
            else:
                print("  原始同名XML缺失；纯SAR阈值可继续，不生成虚构的成像参数",flush=True)
            with ExitStack() as stack:
                source = stack.enter_context(rasterio.open(image_dir/name))
                for key in ('height','width','crs','transform'):
                    if source.profile[key] != original[key]:
                        raise ValueError(f"{name}：准备影像与原始影像的{key}不一致，不能直接套用meta")
                truth = stack.enter_context(rasterio.open(label_dir/record.get('label_name',name))) if label_dir else None
                if truth and (truth.shape!=source.shape or truth.crs!=source.crs or truth.transform!=source.transform):
                    raise ValueError(f"{name}：标签网格与影像不一致")
                if min(bands.values())<1 or max(bands.values())>source.count:
                    raise ValueError(f"{name}：输入仅{source.count}波段，所选波段{bands}不存在")
                shape=(max(1,round(source.height*min(1,preview_size/max(source.shape)))),
                       max(1,round(source.width*min(1,preview_size/max(source.shape)))))
                show = max_visualizations is None or index < max_visualizations
                preview_predictions, image_rows = {}, {}
                for band_name, band in bands.items():
                    # 2. 基础预测：BC直接对离线z-score求全景阈值；legacy才在块内进行旧滤波。
                    method='threshold_'+band_name
                    # 不复制原始整景，只临时保存一列有效像素值供精确百分位选择；完成后自动清理。
                    scratch=stack.enter_context(TemporaryDirectory(prefix='bc_threshold_',dir=cache_dir))
                    median, variance, low, high, threshold=scene_threshold(source,band,block_size,segmenter,scratch)
                    profile=original.copy();profile.update(driver='GTiff',count=1,dtype='uint8',nodata=255,
                        compress='deflate',tiled=True,blockxsize=256,blockysize=256,BIGTIFF='IF_SAFER')
                    # 仅预览时也先分块写临时mask，再按有界尺寸读取，不把整景预测留在内存。
                    mask_path=(output_dir/'masks'/method/Path(name).with_suffix('.tif') if save_masks else Path(scratch)/'mask.tif')
                    mask_path.parent.mkdir(parents=True,exist_ok=True)
                    counts=np.zeros(4,dtype='int64'); image_count=water_count=0; scene_boundary={}
                    with rasterio.open(mask_path,'w',**profile) as target:
                        target.update_tags(source_image=meta['source_image'], source_xml=meta['source_xml'])
                        for core,data,valid in filtered_windows(source,band,block_size,segmenter,median,variance):
                            score = data if segmenter.preprocessing == "offline_normalized" else np.clip((data-low)/(high-low+1e-6),0,1)
                            prediction=(score<threshold if segmenter.mode=='bottom' else score>=threshold)&valid
                            target.write(np.where(valid,prediction,255).astype('uint8'),1,window=core)
                            image_count+=int(valid.sum());water_count+=int(prediction.sum())
                            if truth:
                                # 边界距离在带halo的标签上计算，只统计核心，块边界不会变成假岸线。
                                halo=int(np.ceil(max(boundary_radii,default=0)))+1
                                _, image_valid, crop, expanded=read_window(source,band,core,halo)
                                labels=truth.read(1,window=expanded,masked=True)
                                valid_labels=image_valid & ~np.ma.getmaskarray(labels) & np.isin(labels.data,(0,1))
                                label=labels.data[crop]; evaluation_valid=valid_labels[crop]
                                counts+=confusion_counts(label,prediction,evaluation_valid)
                                regions=boundary_regions(labels.data,valid_labels,boundary_radii)
                                for radius, areas in regions.items():
                                    parts={region:confusion_counts(label,prediction,mask[crop]) for region,mask in areas.items()}
                                    parts['boundary']=parts['inner']+parts['outer']
                                    for region,value in parts.items():
                                        key=(radius,region);scene_boundary[key]=scene_boundary.get(key,np.zeros(4,dtype='int64'))+value
                    # 3. 辅助分析：外部历史水体只对比，不强制补入；DEM/DSM估计几何水深。
                    from thres_segmentation.water_analysis import analyze_water
                    analysis = analyze_water(mask_path, output_dir/'analysis'/method/Path(name).with_suffix(''),
                                             **scene_options)
                    # 4. 评价与输出：统计使用原生像元；阶段2同步累计标签计数以避免重复整景读取。
                    row=dict(name=name,method=method,decision_threshold=threshold,threshold_domain="z_score" if preprocessing else "legacy_stretched_0_1",stretch_low=low,stretch_high=high,
                             image_valid_pixels=image_count,predicted_water_pixels=water_count)
                    row['water_area_m2'] = (analysis['current_water']['m2'] if analysis['current_water'] else None)
                    row['area_status'] = analysis['status']
                    if truth:
                        row.update(metrics_from_counts(counts));all_counts[method]=all_counts.get(method,np.zeros(4,dtype='int64'))+counts
                        for (radius,region),value in scene_boundary.items():
                            boundary_rows.append(dict(name=name,method=method,radius_px=radius,region=region,scope='all_pixels',**metrics_from_counts(value)))
                            key=(method,radius,region);all_boundary[key]=all_boundary.get(key,np.zeros(4,dtype='int64'))+value
                    else:
                        row['pred_water_ratio']=water_count/image_count
                    rows.append(row);image_rows[method]=row
                    if show:
                        with rasterio.open(mask_path) as mask:
                            preview_predictions[method]=mask.read(1,out_shape=shape,resampling=Resampling.nearest)==1
                    print(f"  {method}：全景阈值={threshold:.6g}；预测水体像素={water_count}",flush=True)
                if show:
                    data=source.read(list(bands.values()),out_shape=(len(bands),*shape),out_dtype='float32',masked=True,resampling=Resampling.nearest)
                    image_valid=~np.ma.getmaskarray(data).any(axis=0)&np.isfinite(data.data).all(axis=0)
                    label, valid=None,image_valid
                    if truth:
                        label_data=truth.read(1,out_shape=shape,masked=True,resampling=Resampling.nearest)
                        valid=image_valid & ~np.ma.getmaskarray(label_data)&np.isin(label_data.data,(0,1))
                        label=label_data.data
                    save_figure(data.filled(0),[f"{b} z-score" if preprocessing else b for b in bands],label,valid,preview_predictions,image_rows,
                                output_dir/'figures'/Path(name).with_suffix('.png'),image_valid=image_valid)
            write_csv(output_dir/'per_image.csv',rows)
            write_csv(output_dir/'metrics.csv',[dict(method=m,**metrics_from_counts(c)) for m,c in all_counts.items()])
            if boundary_rows:
                (output_dir/'boundary').mkdir(exist_ok=True)
                write_csv(output_dir/'boundary/per_image.csv',boundary_rows)
                write_csv(output_dir/'boundary/metrics.csv',[dict(method=m,radius_px=r,region=g,scope='all_pixels',**metrics_from_counts(c)) for (m,r,g),c in all_boundary.items()])
    print(f"结果：{output_dir}；PNG为降采样预览，精度按原生像元统计。",flush=True)
    return {method:metrics_from_counts(counts) for method,counts in all_counts.items()}
