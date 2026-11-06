"""整景质量输出：单次推理、多阈值统计、核心区拼接和可选局部形态学修正。

不将整幅影像送入网络或内存。标签只用于计量，不参与修正或阈值生成。
每景完成后保存计数，可使用同一 output_dir 续跑；已完成景不会重新推理。
"""
from supervised.load.sar_scene_dataset import check_preprocessed_input

import csv
import hashlib
import json
import shutil
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
from scipy import ndimage
import torch
from torch.utils.data import DataLoader

from supervised.load.scene_windows import SceneWindows
from supervised.models.checkpoint import load_model
from supervised.evaluation.boundary_eval import boundary_regions, confusion_counts, BoundaryPointReservoir
from supervised.evaluation.metrics import metrics_from_counts
from supervised.evaluation.model_eval import threshold_counts, precision_matches


def write_rows(path, rows):
    """小型统计表统一写UTF-8 BOM；计算过程中每景落盘，异常时可定位进度。"""
    if rows:
        with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def clean_mask(raw, valid, opening_radius=0, closing_radius=0):
    """可选圆形开闭运算；邻域含无效像素的位置保持raw，不跨NoData补水。"""
    result = raw.copy()
    for radius, operation in ((opening_radius, ndimage.binary_opening),
                              (closing_radius, ndimage.binary_closing)):
        if radius:
            y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
            footprint = x*x+y*y <= radius*radius
            changed = operation(result, structure=footprint)
            safe = ndimage.binary_erosion(valid, structure=footprint, iterations=2, border_value=0)
            result[safe] = changed[safe]
    return result & valid


def apply_scene_morphology(raw_path, label_path, post_path, windows, radii, opening_radius, closing_radius):
    """对已拼接raw做第二遍分块修正，避免用相邻窗口不同的预测上下文替代raw。"""
    counts=np.zeros(4,np.int64)
    boundaries=np.zeros((len(radii),2,4),np.int64)
    margin=max((2*(opening_radius+closing_radius),*radii))
    with rasterio.open(raw_path) as raw, rasterio.open(label_path) as labels:
        with rasterio.open(post_path,'w',**raw.profile) as dst:
            dst.update_tags(**raw.tags(),correction='opening_then_closing_on_stitched_raw')
            for top,left,h,w in windows:
                x0,y0=max(0,left-margin),max(0,top-margin)
                x1,y1=min(raw.width,left+w+margin),min(raw.height,top+h+margin)
                extended=Window(x0,y0,x1-x0,y1-y0)
                a=raw.read(1,window=extended)
                b=labels.read(1,window=extended,masked=True)
                valid=a!=255
                label=np.where(valid & ~np.ma.getmaskarray(b) & np.isin(b.data,(0,1)),b.data.astype(np.int64),-1)
                corrected=clean_mask(a==1,valid,opening_radius,closing_radius)
                ys,xs=slice(top-y0,top-y0+h),slice(left-x0,left-x0+w)
                dst.write(np.where(valid[ys,xs],corrected[ys,xs],255).astype('uint8'),1,window=Window(left,top,w,h))
                counts+=confusion_counts(label[ys,xs],corrected[ys,xs],label[ys,xs]>=0)
                regions=boundary_regions(label,label>=0,radii)
                for i,radius in enumerate(radii):
                    for j,region in enumerate(('inner','outer')):
                        boundaries[i,j]+=confusion_counts(label[ys,xs],corrected[ys,xs],regions[radius][region][ys,xs])
    return counts,boundaries


def save_scene_figure(image_path, label_path, output_dir, scene, metrics, threshold):
    """只为显示降采样；标题中的指标来自原分辨率全景计数。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with rasterio.open(image_path) as src:
        scale = min(1., 1200 / max(src.height, src.width))
        shape = (max(1, round(src.height*scale)), max(1, round(src.width*scale)))
        sar = src.read(1, out_shape=shape, masked=True, resampling=Resampling.nearest).filled(np.nan)
    with rasterio.open(label_path) as src:
        gt = src.read(1, out_shape=shape, masked=True, resampling=Resampling.nearest).filled(255)
    masks = []
    for kind in ('raw', 'postprocessed'):
        with rasterio.open(output_dir/kind/(scene+'.tif')) as src:
            masks.append(src.read(1, out_shape=shape, resampling=Resampling.nearest))
    valid = np.isin(gt, (0, 1)) & (masks[0] != 255)
    error = np.zeros((*shape, 3), np.uint8)
    error[valid & (gt == 1) & (masks[1] == 1)] = (40, 140, 255)
    error[valid & (gt == 0) & (masks[1] == 1)] = (255, 60, 60)
    error[valid & (gt == 1) & (masks[1] == 0)] = (255, 220, 40)
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    finite = sar[np.isfinite(sar)]
    lo, hi = np.percentile(finite, (2, 98)) if finite.size else (0, 1)
    axes[0].imshow(sar, cmap='gray', vmin=lo, vmax=max(hi,lo+1e-6))
    for ax, arr in zip(axes[1:4], (gt, *masks)):
        ax.imshow(np.ma.masked_where(arr == 255, arr), cmap='gray_r', vmin=0, vmax=1)
    axes[4].imshow(error)
    for ax, title in zip(axes, ('SAR display p2-p98', 'GT: water=1 black', 'Raw', 'Post', 'Blue TP / Red FP / Yellow FN')):
        ax.set_title(title); ax.axis('off')
    fig.suptitle(f'{scene} | threshold={threshold:g} | native-grid Post F1={metrics["f1"]:.3f} '
                 f'P={metrics["precision"]:.3f} R={metrics["recall"]:.3f} | preview only')
    fig.tight_layout()
    fig.savefig(output_dir/'figures'/(scene+'.png'), dpi=110)
    plt.close(fig)


def summarize(output, scenes, thresholds, radii, precision_targets, recall_targets):
    """汇总已完成景，分母来自混淆计数；未完成时不把部分结果称为全val。"""
    totals = np.zeros((len(thresholds), 4), np.int64)
    boundaries = np.zeros((len(radii), 2, len(thresholds), 4), np.int64)
    groups, scene_rows, fixed_rows = {}, [], []
    post_boundaries = np.zeros((len(radii),2,4),np.int64)
    point_totals=np.zeros_like(boundaries)
    post_point_totals=np.zeros_like(post_boundaries)
    completed = 0
    for scene in scenes:
        path = output/'counts'/(scene['scene_id']+'.npz')
        if not path.exists():
            continue
        completed += 1
        with np.load(path) as d:
            counts, bounds, post = d['counts'], d['boundary'], d['post']
            post_boundaries += d['post_boundary']
            point_totals += d['point_counts']
            post_point_totals += d['post_point_counts']
            totals += counts; boundaries += bounds
            group = scene['group_id']
            groups.setdefault(group, np.zeros_like(totals))[:] += counts
            for t, c in zip(thresholds, counts):
                scene_rows.append(dict(scene_id=scene['scene_id'], group_id=group,
                                       threshold=float(t), **metrics_from_counts(c)))
            fixed_rows.append(dict(scene_id=scene['scene_id'], group_id=group, method='post',
                                   **metrics_from_counts(post)))
    complete = completed == len(scenes)
    overall = [dict(threshold=float(t), completed_scenes=completed, expected_scenes=len(scenes),
                    complete=complete, **metrics_from_counts(c)) for t, c in zip(thresholds, totals)]
    for row in overall:
        values = [r['f1'] for r in scene_rows if r['threshold'] == row['threshold'] and np.isfinite(r['f1'])]
        row['scene_macro_f1'] = float(np.mean(values)) if values else float('nan')
    group_rows = [dict(group_id=g, threshold=float(t), **metrics_from_counts(c))
                  for g, counts in groups.items() for t, c in zip(thresholds, counts)]
    boundary_rows = []
    for i, radius in enumerate(radii):
        for name, counts in (('inner', boundaries[i,0]), ('outer', boundaries[i,1]),
                             ('boundary', boundaries[i].sum(0))):
            for t, c in zip(thresholds, counts):
                boundary_rows.append(dict(radius_px=radius, region=name, threshold=float(t),
                                         **metrics_from_counts(c)))
    write_rows(output/'threshold_overall.csv', overall)
    write_rows(output/'threshold_scenes.csv', scene_rows)
    write_rows(output/'threshold_groups.csv', group_rows)
    write_rows(output/'threshold_boundary.csv', boundary_rows)
    write_rows(output/'post_boundary.csv', [dict(radius_px=r,region=name,**metrics_from_counts(counts))
        for i,r in enumerate(radii) for name,counts in (('inner',post_boundaries[i,0]),
        ('outer',post_boundaries[i,1]),('boundary',post_boundaries[i].sum(0)))])
    write_rows(output/'threshold_boundary_points.csv',[dict(radius_px=r,region=name,threshold=float(t),**metrics_from_counts(c))
        for i,r in enumerate(radii) for name,counts in (('inner',point_totals[i,0]),('outer',point_totals[i,1]),
        ('boundary',point_totals[i].sum(0))) for t,c in zip(thresholds,counts)])
    write_rows(output/'post_boundary_points.csv',[dict(radius_px=r,region=name,**metrics_from_counts(c))
        for i,r in enumerate(radii) for name,c in (('inner',post_point_totals[i,0]),('outer',post_point_totals[i,1]),
        ('boundary',post_point_totals[i].sum(0)))])
    write_rows(output/'post_scenes.csv', fixed_rows)
    if fixed_rows:
        post_counts = [sum(r[k] for r in fixed_rows) for k in ('tn','fp','fn','tp')]
        write_rows(output/'post_overall.csv', [dict(completed_scenes=completed, complete=complete,
                                                   **metrics_from_counts(post_counts))])
    write_rows(output/'precision_targets.csv', precision_matches(overall, precision_targets))
    matches = []
    for target in recall_targets:
        eligible = [r for r in overall if r['recall'] >= target and r['tp']+r['fp'] > 0]
        best = max(eligible, key=lambda r:(r['precision'],r['recall'])) if eligible else None
        matches.append(dict(target_recall=target, status='reachable' if best else 'unreachable',
                            **{k:best[k] if best else '' for k in ('threshold','precision','recall','f1')}))
    write_rows(output/'recall_targets.csv', matches)
    status = dict(complete=complete, completed_scenes=completed, expected_scenes=len(scenes),
                  best_validation_f1=max(overall,key=lambda r:r['f1']) if completed else None,
                  note='Validation threshold selection, not independent test accuracy; boundaries are strict inner/outer bands, not tolerance Boundary-F1.')
    (output/'status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    return overall


def run_scene_quality(checkpoint_path, image_dir=None, label_dir=None, output_dir=None,
                      thresholds=None, threshold=0.5, boundary_radii=(1,3,5),
                      save_probabilities=True, save_visuals=True, batch_size=None,
                      use_amp=None, opening_radius=0, closing_radius=0,
                      precision_targets=(.5,.7,.8,.9), recall_targets=(.8,.85,.9),
                      scene_limit=None, num_workers=0, boundary_points=256):
    """使用保存的val景；扫描只改变判别阈值，模型、划分和归一化完全固定。"""
    checkpoint_path = Path(checkpoint_path)
    c = torch.load(checkpoint_path,map_location='cpu',weights_only=True,mmap=True)
    if c.get('data_layout') != 'full_scene':
        raise ValueError('此入口只接受整景检查点；旧DFC切片请使用原postprocess')
    image_dir = Path(image_dir or c['image_dir'])
    label_dir = Path(label_dir or c['label_dir'])
    check_preprocessed_input(image_dir,c)
    manifest = image_dir.parent/'split_manifest.csv'
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != c['split_manifest_sha256']:
        raise ValueError('划分清单与训练不一致')
    revision = image_dir.parent/'revision_settings.json'
    if c.get('label_revision_sha256') and hashlib.sha256(revision.read_bytes()).hexdigest() != c['label_revision_sha256']:
        raise ValueError('标签版本与训练不一致')
    with manifest.open(encoding='utf-8-sig',newline='') as f:
        scenes = [r for r in csv.DictReader(f) if r['split']=='val' and int(r['active'])==1]
    if set(r['name'] for r in scenes) != set(c['val_scene_files']):
        raise ValueError('验证景名单不一致')
    # 当前正式协议只支持完整验证；试跑也显式记录为subset，不与正式指标混淆。
    if c.get('evaluation_scope') == 'smoke_window_subset':
        raise ValueError('试跑窗口检查点不能冒充整景验证')
    if scene_limit is not None:
        scenes = scenes[:scene_limit]
    thresholds = sorted(set([float(threshold), *(thresholds if thresholds is not None else np.arange(.01,1.,.01))]))
    if any(not 0<t<1 for t in thresholds):
        raise ValueError('概率阈值必须在0和1之间')
    radii = tuple(boundary_radii)
    halo, size = c['window_halo'], c['window_size']
    if any(r<=0 or r>halo for r in radii) or min(opening_radius,closing_radius)<0 or 2*(opening_radius+closing_radius)>halo:
        raise ValueError('边界半径/形态学邻域不能超过上下文宽度')
    output = Path(output_dir) if output_dir else checkpoint_path.parent/'quality'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True,exist_ok=True)
    # 文件大小和修改时间用于续跑保护；不对数十TB数据反复计算内容hash。
    sources=[]
    for r in scenes:
        for folder,key in ((image_dir,'name'),(label_dir,'label_name')):
            p=folder/'val'/r[key]; st=p.stat(); sources.append((str(p),st.st_size,st.st_mtime_ns))
    with checkpoint_path.open('rb') as f:
        checkpoint_hash=hashlib.file_digest(f,'sha256').hexdigest()
    config=dict(checkpoint=str(checkpoint_path), checkpoint_sha256=checkpoint_hash,
                sources=sources, thresholds=thresholds, threshold=threshold, boundary_radii=radii,
                opening_radius=opening_radius,closing_radius=closing_radius,save_probabilities=save_probabilities,
                save_visuals=save_visuals,batch_size=batch_size or c['batch_size'],
                use_amp=c.get('amp_enabled',False) if use_amp is None else use_amp,scene_limit=scene_limit,
                window_size=size,halo=halo,boundary_points=boundary_points, version=2)
    encoded=json.dumps(config,ensure_ascii=False,indent=2)
    config_path=output/'configuration.json'
    if config_path.exists() and config_path.read_text(encoding='utf-8') != encoded:
        raise ValueError('续跑配置/源文件改变；请使用新的quality目录')
    config_path.write_text(encoded,encoding='utf-8')
    for name in ('counts','raw','postprocessed','probability','figures','boundary_points'):
        (output/name).mkdir(exist_ok=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=load_model(c).to(device).eval()
    mean=c['mean'].to(device).view(1,-1,1,1)
    std=c['std'].to(device).clamp_min(1e-6).view(1,-1,1,1)
    amp=config['use_amp'] and device.type=='cuda'
    batch_size=config['batch_size']
    # 记录原验证计数，0.5的复评应与原轮次一致；差异不隐藏。
    expected=c.get('phase_metrics',{}).get('val',{})
    for number,r in enumerate(scenes,1):
        name=r['scene_id']; marker=output/'counts'/(name+'.npz')
        if marker.exists():
            print(f'复用完成景 {number}/{len(scenes)}：{name}',flush=True);continue
        print(f'整景质量 {number}/{len(scenes)}：{name}',flush=True)
        ds=SceneWindows(image_dir/'val'/r['name'],label_dir/'val'/r['label_name'],
                        c['bands'],size,halo,c.get('speckle'))
        # 保守按未压缩的单景新增量检查，完成的其他景可原样保留并续跑。
        needed=ds.width*ds.height*(6 if save_probabilities else 2)+2*1024**3
        if shutil.disk_usage(output).free < needed:
            raise OSError(f'磁盘剩余空间不足以安全输出下一景：{name}')
        counts=torch.zeros((len(thresholds),4),dtype=torch.int64,device=device)
        bounds=torch.zeros((len(radii),2,len(thresholds),4),dtype=torch.int64,device=device)
        post_counts=np.zeros(4,np.int64)
        post_bounds=np.zeros((len(radii),2,4),np.int64)
        reservoirs={(i,j):BoundaryPointReservoir(name,radius,region,boundary_points)
            for i,radius in enumerate(radii) for j,region in enumerate(('inner','outer'))}
        loader=DataLoader(ds,batch_size=batch_size,shuffle=False,num_workers=num_workers,pin_memory=amp)
        profile=ds.profile.copy()
        profile.update(driver='GTiff',count=1,tiled=True,blockxsize=256,blockysize=256,
                       compress='deflate',BIGTIFF='IF_SAFER')
        try:
            with ExitStack() as stack, torch.inference_mode():
                writers={}
                for kind in ('raw','postprocessed',*(['probability'] if save_probabilities else [])):
                    prob=kind=='probability'
                    p={**profile,'dtype':'float32' if prob else 'uint8','nodata':-1. if prob else 255,
                       'predictor':3 if prob else 2}
                    writers[kind]=stack.enter_context(rasterio.open(output/kind/(name+'.tif'),'w',**p))
                    writers[kind].update_tags(water_value=1,non_water_value=0,threshold=threshold,
                                              source_checkpoint=str(checkpoint_path),epoch=c['epoch'])
                for step,batch in enumerate(loader,1):
                    valid=batch['valid_mask'].to(device)
                    image=batch['image'].to(device)
                    if not c.get('inputs_normalized',False):
                        image=(image-mean)/std
                    image=image.masked_fill(~valid[:,None],0)
                    with torch.autocast(device_type=device.type,enabled=amp):
                        prediction=model(image)
                        logits=prediction[-1] if isinstance(prediction,(list,tuple)) else prediction
                    scores=logits[:,0].float() if logits.shape[1]==1 else logits[:,1].float()-logits[:,0].float()
                    if not torch.isfinite(scores).all():
                        raise FloatingPointError(f'非有限模型输出：{name}，batch={step}')
                    for j,coords in enumerate(batch['window'].tolist()):
                        top,left,h,w=coords
                        ys,xs=slice(halo,halo+h),slice(halo,halo+w)
                        y=batch['label'][j].numpy()
                        good=batch['valid_mask'][j].numpy()
                        label=torch.as_tensor(y[ys,xs].copy(),device=device)
                        core_score=scores[j,ys,xs]
                        counts+=threshold_counts(core_score,label,thresholds)
                        score_array=scores[j].cpu().numpy()
                        raw_full=(score_array>=float(np.log(threshold/(1-threshold)))) & good
                        post_full=raw_full  # 需要形态学时，在raw全景写完后另做有邻域的分块修正。
                        post_counts+=confusion_counts(y[ys,xs],post_full[ys,xs],y[ys,xs]>=0)
                        # 纯单类别窗口不可能有水陆边界，跳过EDT与空GPU统计；整体指标仍完整累计。
                        regions=boundary_regions(y,y>=0,radii) if radii and (y==1).any() and (y==0).any() else {}
                        for ri,radius in (enumerate(radii) if regions else ()):
                            for si,region in enumerate(('inner','outer')):
                                mask=regions[radius][region][ys,xs]
                                if not mask.any():
                                    continue
                                target=label.masked_fill(~torch.as_tensor(mask,device=device),-1)
                                bounds[ri,si]+=threshold_counts(core_score,target,thresholds)
                                post_bounds[ri,si]+=confusion_counts(y[ys,xs],post_full[ys,xs],mask)
                                reservoirs[ri,si].update(mask,y[ys,xs],score_array[ys,xs],
                                    raw_full[ys,xs],post_full[ys,xs],top,left)
                        win=Window(left,top,w,h)
                        for kind,mask in (('raw',raw_full),('postprocessed',post_full)):
                            writers[kind].write(np.where(good[ys,xs],mask[ys,xs],255).astype('uint8'),1,window=win)
                        if save_probabilities:
                            prob=core_score.double().sigmoid().cpu().numpy().astype('float32')
                            # 保存值仅有float32精度；扫描直接用logit，不受概率文件量化影响。
                            writers['probability'].write(np.where(good[ys,xs],prob,-1).astype('float32'),1,window=win)
                    if step%50==0:
                        print(f'  {name}：{step}/{len(loader)}批',flush=True)
            if opening_radius or closing_radius:
                post_counts,post_bounds=apply_scene_morphology(output/'raw'/(name+'.tif'),
                    ds.label_path,output/'postprocessed'/(name+'.tif'),ds.windows,radii,opening_radius,closing_radius)
                with rasterio.open(output/'postprocessed'/(name+'.tif')) as post_src:
                    for reservoir in reservoirs.values():
                        for point in reservoir.points:
                            point[5]=post_src.read(1,window=Window(int(point[1]),int(point[0]),1,1))[0,0]
            if save_visuals:
                save_scene_figure(ds.image_path,ds.label_path,output,name,metrics_from_counts(post_counts),threshold)
            point_counts=np.zeros((len(radii),2,len(thresholds),4),np.int64)
            post_point_counts=np.zeros((len(radii),2,4),np.int64)
            point_rows=[]
            for (ri,si),reservoir in reservoirs.items():
                points=reservoir.points
                truth=torch.from_numpy(points[:,2].astype(np.int64))
                score=torch.from_numpy(points[:,3].astype(np.float32))
                point_counts[ri,si]=threshold_counts(score,truth,thresholds).numpy()
                post_point_counts[ri,si]=confusion_counts(points[:,2],points[:,5],np.ones(len(points),bool))
                for y,x,gt,z,raw,post in points:
                    point_rows.append(dict(scene_id=name,radius_px=radii[ri],region=('inner','outer')[si],
                        row=int(y),col=int(x),gt=int(gt),logit=z,raw=int(raw),post=int(post),seed=42))
            write_rows(output/'boundary_points'/(name+'.csv'),point_rows)
            pending=marker.with_suffix('.partial')
            with pending.open('wb') as f:
                np.savez_compressed(f,counts=counts.cpu().numpy(),boundary=bounds.cpu().numpy(),
                                    post=post_counts,post_boundary=post_bounds,point_counts=point_counts,post_point_counts=post_point_counts)
            pending.replace(marker)  # 只有所有栅格与图像写完才标记完成。
        finally:
            ds.close()
        overall=summarize(output,scenes,thresholds,radii,precision_targets,recall_targets)
        print(f'已保存 {name}；Post F1={metrics_from_counts(post_counts)["f1"]:.4f}',flush=True)
    overall=summarize(output,scenes,thresholds,radii,precision_targets,recall_targets)
    original=next((r for r in overall if r['threshold']==.5),None)
    if original and expected and scene_limit is None:
        checks=[dict(metric=k,training=int(expected[k]),quality=int(original[k]),
                     delta=int(original[k])-int(expected[k])) for k in ('tn','fp','fn','tp')]
        write_rows(output/'training_parity.csv',checks)
        print('与原验证0.5计数差：',{r['metric']:r['delta'] for r in checks},flush=True)
    print(f'整景质量完成：{output}',flush=True)
    return output
