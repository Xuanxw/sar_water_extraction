"""逐景ORG→AOI裁剪→SNAP Lee→阈值/辅助修正→SHP与质量JSON。日期仅作目录分组。"""
import json
import shutil
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from pyproj import Transformer
from shapely.geometry import mapping
from shapely import transform as transform_geometry
from scipy.ndimage import binary_opening, binary_closing
from filelock import FileLock

from thres_segmentation.org_io import (write_json,inventory,read_aoi,prepare_auxiliary,
    local_source,crop_org,export_water_polygon,windows)
from thres_segmentation.metadata import read_source_metadata
from thres_segmentation.scene_pipeline import disk_percentiles
from thres_segmentation.quality_review import export_quality_review
from thres_segmentation.water_analysis import pixel_areas,area_sum


class SceneProgress:
    """只在data_cache中保存一份逐景进度；相对路径支持同时迁移缓存和成果目录。"""
    def __init__(self,cache_root,scene_id,output_folder,snap_root,resume=True):
        self.folder=Path(cache_root)/'scenes'/scene_id
        self.folder.mkdir(parents=True,exist_ok=True)
        self.path=self.folder/'progress.json'
        self.roots={'cache':self.folder,'output':Path(output_folder),'snap':Path(snap_root)}
        self.resume=resume
        self.state={'scene_id':scene_id,'stages':{}}
        if resume and self.path.exists():
            self.state=json.loads(self.path.read_text(encoding='utf-8'))

    def files(self,name):
        return [self.roots[item['root']]/item['path']
                for item in self.state['stages'].get(name,{}).get('files',[])]

    def completed(self,name):
        """完成记录最后发布；只确认记录和文件存在，不比较路径哈希、mtime或配置。"""
        stage=self.state['stages'].get(name,{})
        return (self.resume and stage.get('complete',False) and bool(stage.get('files'))
                and all(p.is_file() and p.stat().st_size>0 for p in self.files(name)))

    def mark(self,name,files):
        records=[]
        for path in files:
            path=Path(path).resolve()
            for root_name in ('output','snap','cache'):
                try:relative=path.relative_to(self.roots[root_name].resolve())
                except ValueError:continue
                records.append({'root':root_name,'path':relative.as_posix()});break
            else:raise ValueError(f'阶段产物不在配置的缓存或成果目录内：{path}')
        self.state['stages'][name]={'complete':True,'files':records}
        write_json(self.path,self.state)


def statistics(filtered, aoi, config, folder, history_percentiles=None):
    """只在AOI有效观测内估计整景分位数；磁盘缓冲避免整景数组和重复滤波。"""
    settings=config['org'];base=float(config['threshold']['water_percent'])
    delta=settings['sensitivity_delta_percent'];relax=settings['postprocess']['history_relax_percent']
    percentages=[base,max(0,base-delta),min(100,base+delta),min(100,base+relax)]
    # 对照实验可一次计算多个历史水体分位；原生产入口仍使用原来的四个值。
    extra=[] if history_percentiles is None else list(history_percentiles)
    percentages.extend(extra)
    if config['threshold']['mode']=='top':percentages=[100-p for p in percentages]
    count=0;total=square=0.
    with TemporaryDirectory(prefix='quantile_',dir=folder) as temp:
        buffer=Path(temp)/'values.bin'
        with rasterio.open(filtered) as src,buffer.open('wb') as stream:
            for win in windows(src,config['processing']['block_size']):
                data=src.read(1,window=win,masked=True)
                inside=geometry_mask([mapping(aoi)],out_shape=data.shape,transform=src.window_transform(win),invert=True)
                good=inside&~np.ma.getmaskarray(data)&np.isfinite(data.data)
                values=data.data[good].astype('float64')
                values.tofile(stream);count+=values.size;total+=float(values.sum());square+=float(np.square(values).sum())
        if not count:raise ValueError('行政区相交范围没有有效SAR像元')
        thresholds=disk_percentiles(buffer,count,percentages)[0].tolist()
    mean=total/count;std=max(0.,square/count-mean**2)**.5
    result={'thresholds_amplitude':thresholds[:4],'water_percent':base,'valid_pixels':count,
            'mean':mean,'std':std,'normalization':settings['normalization'],
            'decision_threshold':(thresholds[0]-mean)/max(std,1e-6) if settings['normalization']=='scene_zscore' else thresholds[0]}
    if extra:result['history_thresholds_amplitude']=dict(zip(map(str,extra),thresholds[4:]))
    return result


def segment(filtered, aoi, auxiliary, config, folder, stats, save_context=True):
    """分块加halo完成形态学；辅助缺测处不排除、不强制恢复，原始未知保持255。"""
    post=config['org']['postprocess'];size=config['processing']['block_size']
    radius=post['opening_radius_px']+post['closing_radius_px'];halo=2*radius
    totals={'raw_area_m2':0.,'final_area_m2':0.,'valid_area_m2':0.,'landcover_removed_m2':0.,
            'terrain_removed_m2':0.,'auxiliary_unknown_m2':0.,'sensitivity_area_m2':[0.,0.,0.]}
    files=[]
    geometry_mode=post['terrain_enabled'] and post.get('terrain_method','slope_only')=='gui_geometry'
    if geometry_mode:
        from thres_segmentation.terrain_geometry import generate_terrain_mask
        meta=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
        terrain=generate_terrain_mask(filtered,auxiliary['slope'],auxiliary['aspect'],meta,folder,post['max_slope_deg'])
        auxiliary={**auxiliary,'terrain_exclusion':terrain['terrain_exclusion']}
        terrain_files=[*terrain.values(),folder/'terrain_geometry.json']
    with ExitStack() as stack:
        src=stack.enter_context(rasterio.open(filtered));profile=src.profile.copy()
        profile.update(dtype='uint8',nodata=255,count=1)
        readers={}
        for name,path in auxiliary.items():
            if name in ('copernicus','aspect'):continue
            original=stack.enter_context(rasterio.open(path))
            readers[name]=stack.enter_context(WarpedVRT(original,crs=src.crs,transform=src.transform,
                width=src.width,height=src.height,dtype='float32',nodata=np.nan,
                resampling=Resampling.bilinear if name=='slope' else Resampling.nearest))
        writers={}
        # 参数对照只输出变化的water；原生产入口仍保留全部上下文栅格。
        for name in (('raw','water','coverage') if save_context else ('water',)):
            path=folder/f'{name}.tif';files.append(path)
            writers[name]=stack.enter_context(rasterio.open(path.with_suffix('.partial.tif'),'w',**profile))
        for name in (('history','slope') if save_context else ()):
            if name=='history' and not post['history_enabled']:continue
            if name=='slope' and not post['terrain_enabled']:continue
            path=folder/f'{name}.tif';files.append(path)
            p=profile.copy()
            if name=='slope':p.update(dtype='float32',nodata=np.nan)
            writers[name]=stack.enter_context(rasterio.open(path.with_suffix('.partial.tif'),'w',**p))
        for core in windows(src,size):
            x0=max(0,int(core.col_off)-halo);y0=max(0,int(core.row_off)-halo)
            x1=min(src.width,int(core.col_off+core.width)+halo);y1=min(src.height,int(core.row_off+core.height)+halo)
            win=Window(x0,y0,x1-x0,y1-y0)
            crop=np.s_[int(core.row_off)-y0:int(core.row_off+core.height)-y0,
                       int(core.col_off)-x0:int(core.col_off+core.width)-x0]
            data=src.read(1,window=win,masked=True)
            inside=geometry_mask([mapping(aoi)],out_shape=data.shape,transform=src.window_transform(win),invert=True)
            valid=inside&~np.ma.getmaskarray(data)&np.isfinite(data.data)
            compare=(lambda t:data.data<t) if config['threshold']['mode']=='bottom' else (lambda t:data.data>=t)
            raw=compare(stats['thresholds_amplitude'][0])&valid
            excluded_cover=np.zeros(valid.shape,bool);excluded_slope=excluded_cover.copy()
            history=np.full(valid.shape,255,np.uint8);slope=np.full(valid.shape,np.nan,np.float32)
            aux_missing=np.zeros(valid.shape,bool)
            if 'worldcover' in readers:
                cover=readers['worldcover'].read(1,window=win,masked=True)
                known=~np.ma.getmaskarray(cover)&np.isin(cover.data,(10,20,30,40,50,60,70,80,90,95,100))
                if post['landcover_enabled']:
                    excluded_cover=known&np.isin(cover.data,post['excluded_cover_classes']);aux_missing|=~known
                if post['history_enabled'] and 'history' not in readers:
                    history[known]=(cover.data[known]==80).astype('uint8');aux_missing|=~known
            if 'history' in readers and post['history_enabled']:
                values=readers['history'].read(1,window=win,masked=True)
                known=~np.ma.getmaskarray(values)&np.isin(values.data,post['history_water_values']+post['history_dry_values'])
                history[known]=np.isin(values.data[known],post['history_water_values']);aux_missing|=~known
            if 'slope' in readers:
                values=readers['slope'].read(1,window=win,masked=True)
                known=~np.ma.getmaskarray(values)&np.isfinite(values.data)
                slope[known]=values.data[known]
                excluded_slope=known&(slope>post['max_slope_deg']);aux_missing|=~known
            if geometry_mode:
                terrain=readers['terrain_exclusion'].read(1,window=win,masked=True)
                excluded_slope=(~np.ma.getmaskarray(terrain))&(terrain.data==1)
            def correct(base):
                value=compare(base)&valid
                if post['history_enabled']:
                    # 历史水体只能恢复仍满足较宽SAR阈值的像元，不能填入未覆盖区域。
                    value |= (history==1)&compare(stats['thresholds_amplitude'][3])&valid
                value &= ~(excluded_cover|excluded_slope)
                if post['opening_radius_px']:
                    r=post['opening_radius_px'];yy,xx=np.ogrid[-r:r+1,-r:r+1]
                    value=binary_opening(value,structure=xx*xx+yy*yy<=r*r)
                if post['closing_radius_px']:
                    r=post['closing_radius_px'];yy,xx=np.ogrid[-r:r+1,-r:r+1]
                    value=binary_closing(value,structure=xx*xx+yy*yy<=r*r)
                return value&valid&~(excluded_cover|excluded_slope)
            final=correct(stats['thresholds_amplitude'][0]);good=valid[crop]
            for name,mask in (('raw',raw[crop]),('water',final[crop]),('coverage',good)):
                if name in writers:
                    writers[name].write(np.where(good,mask,255).astype('uint8'),1,window=core)
            if 'history' in writers:writers['history'].write(np.where(valid,history,255)[crop].astype('uint8'),1,window=core)
            if 'slope' in writers:writers['slope'].write(np.where(valid,slope,np.nan)[crop],1,window=core)
            areas=pixel_areas(src.crs,src.window_transform(core),good.shape)
            for key,mask in (('raw_area_m2',raw[crop]),('final_area_m2',final[crop]),('valid_area_m2',good),
                ('landcover_removed_m2',(raw&excluded_cover)[crop]),('terrain_removed_m2',(raw&excluded_slope)[crop]),
                ('auxiliary_unknown_m2',(valid&aux_missing)[crop])):
                totals[key]+=area_sum(mask,areas)
            for i,t in enumerate(stats['thresholds_amplitude'][:3]):
                value=final if i==0 else correct(t)
                totals['sensitivity_area_m2'][i]+=area_sum(value[crop],areas)
    for path in files:path.with_suffix('.partial.tif').replace(path)
    write_json(folder/'extraction.json',{'statistics':stats,**totals})
    return files+[folder/'extraction.json']+(terrain_files if geometry_mode else [])


def save_preview(image_path,mask_path,destination,max_side):
    """只读取缩略图，不为出图载入整景；统计另按原始像元计算。"""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    with rasterio.open(image_path) as src,rasterio.open(mask_path) as mask:
        ratio=min(1,max_side/max(src.shape));size=(max(1,round(src.height*ratio)),max(1,round(src.width*ratio)))
        image=src.read(1,out_shape=size,masked=True);water=mask.read(1,out_shape=size,masked=True)
    valid=image.compressed();low,high=np.percentile(valid,[2,98]) if len(valid) else (0,1)
    fig=Figure(figsize=(10,5));FigureCanvasAgg(fig);ax=fig.subplots(1,2)
    ax[0].imshow(image,cmap='gray',vmin=low,vmax=high);ax[0].set_title('Filtered SAR (display P2-P98)')
    ax[1].imshow(water,cmap='Blues',vmin=0,vmax=1);ax[1].set_title('Final water (1=water; masked=unknown)')
    for a in ax:a.axis('off')
    fig.tight_layout();fig.savefig(destination,dpi=110)


def run_org_scenes(config, config_path=None):
    """建立同步日志后进入处理；不因日志开关变化重新计算影像。"""
    from thres_segmentation.run_log import RunLog
    paths=config['paths']
    log_root=paths['OUTPUT_DIR'] or paths['OUTPUT_ROOT']/'logs'
    log=RunLog(log_root,config.get('logging',{}).get('console_color',True))
    with log.capture(),log.stage('00','本次运行总计'):
        if config_path is not None:log.message(f'阈值分割配置：{config_path}')
        return _run_org_scenes(config,log)


def _run_org_scenes(config,log):
    """按景独立处理，日期只用于目录组织。完成标记在全部产物发布后最后写入。"""
    from datasets.snap_preprocess import configure_paths
    from datasets.speckle import SpeckleFilter
    with log.stage('01','扫描ORG及读取行政区'):
        paths=config['paths'];options=config['org'];size=config['processing']['block_size']
        if config['threshold']['mode'] not in ('bottom','top') or not 0<=config['threshold']['water_percent']<=100:
            raise ValueError('ORG阈值mode为bottom/top，water_percent为0～100')
        if options['normalization'] not in ('none','scene_zscore'):raise ValueError('normalization为none/scene_zscore')
        if options['filter']['method'] not in ('none','snap_lee'):raise ValueError('滤波为none/snap_lee')
        if options['filter']['input_domain']!='amplitude':raise ValueError('本ORG流程要求线性幅度输入')
        post=options['postprocess'];aux=options['auxiliary']
        if size<1 or options['crop_buffer_m']<0 or aux['buffer_m']<0:
            raise ValueError('分块大小需为正，缓冲距离不能为负')
        if min(aux['landcover_resolution_m'],aux['dem_resolution_m'],aux['elevation_scale'])<=0:
            raise ValueError('辅助分辨率和高程缩放需为正')
        if min(options['sensitivity_delta_percent'],post['history_relax_percent'])<0:
            raise ValueError('分位数变化范围不能为负')
        if any(not isinstance(post[n],int) or post[n]<0 for n in ('opening_radius_px','closing_radius_px')):
            raise ValueError('形态学半径需为非负整数')
        if set(post['history_water_values'])&set(post['history_dry_values']):
            raise ValueError('历史水体和非水体编码不能重叠')
        if len(config['input']['bands'])!=1:raise ValueError('ORG生产本轮使用一个极化波段')
        records=inventory(paths['ORG_ROOT'],options)
        aoi,crs=read_aoi(paths['AOI_PATH'],options['output_crs'])
        root=paths['OUTPUT_DIR'] or paths['OUTPUT_ROOT']/'org_scenes'
        root.mkdir(parents=True,exist_ok=True)
    # 防止两个进程同时改写同一任务；不会停止用户的其他训练/下载。
    with FileLock(str(root/'processing.lock'),timeout=0):
        from thres_segmentation.product_layout import arrange_intermediates
        archive=options.get('archive_intermediates',False)
        if archive:arrange_intermediates(config,restore=True)
        # run.json仅记录当前参数，不再承担续跑validation。
        write_json(root/'run.json',{'config':config,'aoi_area_m2':aoi.area,'merge_by_date':False})
        write_json(root/'inventory.json',records)
        print(f'ORG逐景处理：{len(records)}景，{len({r["date"] for r in records})}日期；输出：{root}',flush=True)
        with log.stage('02','辅助数据下载/复用与空间对齐'):
            auxiliary=prepare_auxiliary(config,aoi,crs)
        polygon_options=dict(options=config.get('polygon_output',{}),postprocess=options['postprocess'],
                             cover_path=auxiliary.get('worldcover'),slope_path=auxiliary.get('slope'),
                             clip_geometry=aoi,clip_crs=crs)
        with log.stage('03','初始化SNAP运行环境'):
            filter_config=dict(options['filter'])
            snap=None
            if filter_config['method']=='snap_lee':
                configure_paths(paths['SNAP_GPT_PATH'],paths['SNAP_CACHE_DIR'],paths['SNAP_WORK_DIR'],
                     threads=options['snap_threads'],tile_cache=options['snap_tile_cache'])
                snap=SpeckleFilter(filter_config).snap
        band=next(iter(config['input']['bands'].values()))
        rows=[]
        for index,record in enumerate(records,1):
            with log.stage('SCENE','逐景处理总计',record['scene_id']) as scene_stage:
                folder=root/record['date']/record['scene_id'];folder.mkdir(parents=True,exist_ok=True)
                progress=SceneProgress(paths['CACHE_DIR'],record['scene_id'],folder,paths['SNAP_CACHE_DIR'],options['resume'])
                print(f'开始 {index}/{len(records)}：{record["scene_id"]}',flush=True)
                # 旧版只有坡度筛查；补齐原GUI地形分类时复用滤波/阈值缓存，仅重做下游。
                if options['postprocess'].get('terrain_method')=='gui_geometry' and not (folder/'terrain_geometry.json').exists():
                    for stage in ('mask','vector','quality','preview','complete'):progress.state['stages'].pop(stage,None)
                if progress.completed('complete'):
                    scene_stage['status']='reused'
                    log.message('复用已完成的本景成果/覆盖检查','REUSED')
                    summary=json.loads((folder/'scene.json').read_text(encoding='utf-8'))
                    if summary['status']=='complete':
                        polygon=export_water_polygon(folder/'water.tif',**polygon_options)
                        had_polygon=(summary.get('water_polygon_shp')==str(polygon.resolve()) and summary.get('water_shp')==str(polygon.resolve()))
                        summary.update(water_shp=str(polygon.resolve()),water_polygon_shp=str(polygon.resolve()),quality_json=str((folder/'quality.json').resolve()))
                        if not had_polygon:write_json(folder/'scene.json',summary)
                    rows.append(summary);continue
                write_json(folder/'input.json',record)
                crop=progress.folder/'amplitude.tif'
                with log.stage('04','原始元数据/解压与行政区裁剪',record['scene_id']) as step:
                    if progress.completed('crop'):
                        step['status']='reused'
                    else:
                        print('  裁剪原始幅度及追溯元数据',flush=True)
                        meta,_,xml=read_source_metadata(record['path'])
                        write_json(folder/'metadata.json',meta)
                        if xml:(folder/'source.xml').write_bytes(xml)
                        source=local_source(record,paths['CACHE_DIR'])
                        cropped=crop_org(source,crop,aoi,crs,band,options['crop_buffer_m'],options['input_nodata_value'],size)
                        if cropped is None:
                            step['status']='skipped';scene_stage['status']='skipped'
                            log.message('与行政区外扩范围无相交，跳过','SKIPPED')
                            summary={**record,'status':'outside_aoi'}
                            write_json(folder/'scene.json',summary)
                            progress.mark('complete',[folder/'scene.json'])
                            rows.append(summary);continue
                        progress.mark('crop',[crop,folder/'metadata.json'])
                with log.stage('05','SNAP Lee滤波',record['scene_id']) as step:
                    if progress.completed('filter'):
                        filtered=progress.files('filter')[0];step['status']='reused'
                    else:
                        filtered=snap.prepare_file(crop,(1,)) if snap else crop
                        if not snap:step['status']='skipped'
                        progress.mark('filter',[filtered])
                    if snap:
                        log.message(f'SNAP详细日志：{filtered.parent / "gpt.log"}')
                        log.message(f'SNAP内部计时：{filtered.parent / "timings.json"}')
                with rasterio.open(crop) as source:
                    scene_aoi=transform_geometry(aoi,Transformer.from_crs(crs,source.crs,always_xy=True).transform,interleaved=False,include_z=None)
                with log.stage('06','AOI内整景分位阈值统计',record['scene_id']) as step:
                    stats_file=progress.folder/'statistics.json'
                    if progress.completed('statistics'):
                        stats=json.loads(stats_file.read_text(encoding='utf-8'));step['status']='reused'
                    else:
                        stats=statistics(filtered,scene_aoi,config,progress.folder)
                        write_json(stats_file,stats);progress.mark('statistics',[stats_file])
                with log.stage('07','阈值分割及辅助修正',record['scene_id']) as step:
                    if progress.completed('mask'):step['status']='reused'
                    else:
                        products=segment(filtered,scene_aoi,auxiliary,config,folder,stats)
                        (folder/'water_polygon.json').unlink(missing_ok=True)
                        progress.mark('mask',products)
                water=folder/'water.tif';shp=folder/'water_polygon.shp'
                with log.stage('08','完整水体面输出',record['scene_id']):
                    export_water_polygon(water,overwrite=not options['resume'],**polygon_options)
                    progress.mark('vector',list(folder.glob('water_polygon.*')))
                review=dict(config['quality_review']);enabled=review.pop('enabled')
                with log.stage('09','质量JSON/分层核验点/疑似区域',record['scene_id']) as step:
                    if progress.completed('quality'):
                        step['status']='reused'
                    else:
                        review_dir=folder/'review'
                        if review_dir.exists():
                            # 仅清除本任务固定review子目录中的中断产物，不触及输入和其它景。
                            resolved=review_dir.resolve()
                            if resolved.parent!=folder.resolve():raise ValueError('核查目录越界')
                            shutil.rmtree(resolved)
                        extraction=json.loads((folder/'extraction.json').read_text(encoding='utf-8'))
                        if enabled:
                            quality=export_quality_review(water,review_dir,scene_id=record['scene_id'],
                                history_path=folder/'history.tif' if (folder/'history.tif').exists() else None,
                                slope_path=folder/'slope.tif' if (folder/'slope.tif').exists() else None,
                                raw_mask_path=folder/'raw.tif',metadata={'date':record['date'],'sensor':record['sensor']},**review)
                        else:
                            quality={'scene_id':record['scene_id'],'assessment':'quality_only_without_ground_truth',
                                     'water':{'area_m2':extraction['final_area_m2']},'accuracy':None}
                        if enabled:quality['artifacts']['vectors']='review/review.gpkg'
                        quality['administrative_coverage']={'aoi_area_m2':aoi.area,'valid_area_m2':extraction['valid_area_m2'],
                            'fraction':extraction['valid_area_m2']/aoi.area,'method':'pixel_center_grid_area_approximation'}
                        quality['extraction']=extraction
                        quality['correction_policy']=options['postprocess']
                        quality['not_evaluated']=['overlap_consistency','temporal_change','ray_traced_radar_occlusion']
                        if (folder/'terrain_geometry.json').exists():quality['terrain_geometry']=json.loads((folder/'terrain_geometry.json').read_text(encoding='utf-8'))
                        write_json(folder/'quality.json',quality)
                        if enabled:
                            # 每景仅保留一份简洁质量JSON；矢量仍在review/下。
                            (review_dir/'quality.json').unlink()
                        files=[folder/'quality.json']+list(review_dir.glob('*'))
                        progress.mark('quality',files)
                with log.stage('10','输出快视图',record['scene_id']) as step:
                    if not options['save_quicklook']:step['status']='skipped'
                    elif progress.completed('preview'):step['status']='reused'
                    else:
                        # 展示正式交付的清理后栅格，与water_polygon.shp保持同一处理阶段。
                        save_preview(filtered,folder/'water_polygon.tif',folder/'quicklook.png',config['output']['preview_size'])
                        previews=[folder/'quicklook.png']
                        if (folder/'terrain_class.tif').exists():
                            from thres_segmentation.terrain_geometry import save_terrain_preview
                            save_terrain_preview(folder/'water_polygon.tif',folder/'terrain_class.tif',folder/'terrain_quicklook.png',config['output']['preview_size'])
                            previews.append(folder/'terrain_quicklook.png')
                        progress.mark('preview',previews)
                with log.stage('11','保存逐景完成记录',record['scene_id']):
                    summary={**record,'status':'complete','water_shp':str(shp.resolve()),'water_polygon_shp':str((folder/'water_polygon.shp').resolve()),'quality_json':str((folder/'quality.json').resolve())}
                    write_json(folder/'scene.json',summary)
                    files=[p for p in folder.rglob('*') if p.is_file() and '.partial' not in p.name and p.name!='complete.done.json']
                    progress.mark('complete',files)
                    rows.append(summary)
                    write_json(root/'progress.json',{'completed':len(rows),'total':len(records),'last_scene':record['scene_id']})
        write_json(root/'results.json',rows)
        if archive:arrange_intermediates(config)
        print(f'逐景处理完成：{root}；未按日期合并。',flush=True)
    return root
