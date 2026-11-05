"""只读SAR逐景数值审查：固定网格抽样，不假定幅度/功率，不做辐射校正或改写。

偏离参考分布只代表需要审查，不自动排除样本、改变标签或生成训练权重。
参考范围仅由train估计；val只能用于诊断，不反向参与参考范围计算。
"""
from pathlib import Path
from datetime import datetime
import csv
import numpy as np
import rasterio
from rasterio.windows import Window

ROOT = Path(r"I:\水体数据\prepared_scenes")
OUTPUT_ROOT = Path(r"E:\Code_projects\sar_water_extraction\codex_analysis\radiometry_audit")
BAND = 1
WINDOW_SIZE = 256
GRID_SIZE = 3  # 每景固定3×3位置；这是抽样估计，不是整景精确直方图。


def inspect_scene(image_path, label_path, band=1, window_size=256, grid_size=3):
    """保留有效0值，统计原始DN与标签两类分布；只读有限数值。"""
    values, labels = [], []
    sampled = invalid = 0
    with rasterio.open(image_path) as src, rasterio.open(label_path) as gt:
        if (src.shape,src.crs,src.transform)!=(gt.shape,gt.crs,gt.transform):
            raise ValueError('标签与SAR网格不一致')
        h,w=min(window_size,src.height),min(window_size,src.width)
        # 分层网格单元的中心，而非外接矩形四角；仍保留读到的合法0值。
        ys=((np.arange(grid_size)+.5)/grid_size*max(0,src.height-h)).astype(int)
        xs=((np.arange(grid_size)+.5)/grid_size*max(0,src.width-w)).astype(int)
        coords=sorted({(int(y),int(x)) for y in ys for x in xs})
        for y,x in coords:
            win=Window(x,y,w,h)
            a=src.read(band,window=win,masked=True,out_dtype='float32')
            b=gt.read(1,window=win,masked=True)
            good=~np.ma.getmaskarray(a)&np.isfinite(a.data)
            labelled=~np.ma.getmaskarray(b)&np.isin(b.data,(0,1))
            sampled+=good.size;invalid+=int((~good).sum())
            values.append(a.data[good].astype(np.float64))
            labels.append(np.where(labelled,b.data.astype(np.int64),-1)[good])
        x=np.concatenate(values); y=np.concatenate(labels)
        if not len(x):
            raise ValueError('抽样位置没有有效SAR像素')
        q=np.percentile(x,(1,5,50,95,99))
        positive=x>0
        pq=np.percentile(x[positive],(1,50,95,99)) if positive.any() else [float('nan')]*4
        water=x[(y==1)&positive];land=x[(y==0)&positive]
        result=dict(width=src.width,height=src.height,dtype=src.dtypes[band-1],
                    pixel_x=abs(src.transform.a),pixel_y=abs(src.transform.e),
                    sampled_pixels=sampled,valid_samples=len(x),invalid_ratio=invalid/sampled,
                    minimum=float(x.min()),maximum=float(x.max()),mean=float(x.mean()),std=float(x.std()),
                    p01=q[0],p05=q[1],p50=q[2],p95=q[3],p99=q[4],
                    zero_ratio=float(np.mean(x==0)),negative_ratio=float(np.mean(x<0)),
                    water_samples=int(sum(y==1)),land_samples=int(sum(y==0)),
                    water_median=float(np.median(x[y==1])) if (y==1).any() else float('nan'),
                    land_median=float(np.median(x[y==0])) if (y==0).any() else float('nan'),
                    storage_scale=src.scales[band-1],storage_offset=src.offsets[band-1],
                    declared_unit=str(src.units[band-1]),physical_domain='unconfirmed',
                    calibration_status='not_verified_from_DN_statistics',
                    positive_samples=int(positive.sum()),positive_p01=pq[0],positive_p50=pq[1],
                    positive_p95=pq[2],positive_p99=pq[3],
                    water_positive_samples=len(water),land_positive_samples=len(land),
                    water_positive_median=float(np.median(water)) if len(water) else float('nan'),
                    land_positive_median=float(np.median(land)) if len(land) else float('nan'))
    return result


def audit(root=ROOT, output_dir=None):
    """输出全部采用景，独立列出待读数/尺度审查项；不修改采用清单。"""
    root=Path(root)
    output=Path(output_dir) if output_dir else OUTPUT_ROOT/datetime.now().strftime('%Y%m%d_%H%M%S')
    output.mkdir(parents=True,exist_ok=False)
    with (root/'split_manifest.csv').open(encoding='utf-8-sig',newline='') as f:
        scenes=[r for r in csv.DictReader(f) if int(r['active'])==1]
    rows=[]
    for i,r in enumerate(scenes,1):
        row=dict(scene_id=r['scene_id'],split=r['split'],group_id=r['group_id'],
                 scene=r.get('scene',''),sensor=r['scene_id'].split('-')[0],status='ok',error='')
        try:
            row.update(inspect_scene(root/'images'/r['split']/r['name'],
                                     root/'labels'/r['split']/r['label_name'],BAND,WINDOW_SIZE,GRID_SIZE))
        except Exception as e:
            row.update(status='read_error',error=str(e))
            print(f'读取异常：{r["scene_id"]}：{e}',flush=True)
        rows.append(row)
        if i%25==0 or i==len(scenes):
            print(f'幅度分布审查：{i}/{len(scenes)}',flush=True)
            save_csv(output/'scene_statistics.partial.csv',rows)
    train=[r for r in rows if r['split']=='train' and r['status']=='ok']
    for row in rows:
        row.update(review_reason='',reference_count=0,reference_p50_low='',reference_p50_high='',
                   recommended_sample_weight='not_assigned')
        if row['status']!='ok':
            row['review_reason']='read_error';continue
        pool=[r for r in train if r['sensor']==row['sensor'] and np.isclose(r['pixel_x'],row['pixel_x']) and np.isfinite(r['positive_p50'])]
        scope='same_sensor_and_pixel_size'
        if len(pool)<5:
            pool=[r for r in train if np.isfinite(r['positive_p50'])];scope='all_train_fallback'
        row['reference_scope']=scope
        if not pool:
            row['review_reason']='no_training_reference';continue
        low,high=np.percentile([r['positive_p50'] for r in pool],(1,99))
        row.update(reference_count=len(pool),reference_p50_low=low,reference_p50_high=high)
        flags=[]
        if row['positive_p50']<low or row['positive_p50']>high:flags.append('positive_median_outside_train_p01_p99')
        if row['negative_ratio']>0:flags.append('negative_values_check_domain')
        if row['zero_ratio']>.2:flags.append('many_zeros_check_coverage_not_auto_nodata')
        if row['invalid_ratio']>.2:flags.append('large_invalid_sample_fraction')
        if row['std']==0:flags.append('constant_sample')
        if row['water_positive_samples']>=512 and row['land_positive_samples']>=512 and row['water_positive_median']>=row['land_positive_median']:
            flags.append('water_not_darker_check_scene_or_labels')
        row['review_reason']=';'.join(flags)
    save_csv(output/'scene_statistics.csv',rows)
    save_csv(output/'review_scenes.csv',[r for r in rows if r['review_reason']])
    (output/'scene_statistics.partial.csv').unlink(missing_ok=True)
    (output/'README.txt').write_text(
        '3x3 interior cell-center 256px windows; sampled estimates, not exact full-scene histograms.\n'
        'All-valid and conditional-positive distributions are both reported; positives do NOT redefine validity.\n'
        'Reference quantiles use train conditional-positive medians, preferably same sensor/pixel size.\n'
        'Flags are review candidates, not evidence of corruption or wrong labels. Zero remains valid.\n'
        'No automatic scene weights or radiometric calibration. DN moments cannot identify amplitude vs power.\n',encoding='utf-8')
    print(f'审查完成：{output}；待复核{sum(bool(r["review_reason"]) for r in rows)}/{len(rows)}景',flush=True)
    return output


def save_csv(path, rows):
    if rows:
        fields=list(dict.fromkeys(k for r in rows for k in r))
        with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


if __name__=='__main__':
    audit()
