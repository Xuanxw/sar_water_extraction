"""整景零Doppler地形可见性实验。所有派生结果留在独立输出目录。

DEM和ORG输入只读。按方位批次生成完整横向剖面，再分块映射回地理网格；
不依赖supervised或threshold运行入口，不调用它们覆盖现有水体。
"""
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter
import hashlib
import json
import os
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from rasterio.transform import from_origin
from rasterio.windows import Window
from pyproj import Transformer, CRS
from .core import solve_points, orbit_states
from .metadata import make_model
from .profiles import ElevationSampler, build_profiles, profile_coordinates, A
from .terrain_io import write_json, sha256

VERSION='zero_doppler_plane_profiles_0.2'


def grid(bounds, resolution, crs):
    """按米制固定网格构造输出；类别掩膜不会上采样为虚假的SAR原生地形精度。"""
    left,bottom,right,top=bounds
    left=np.floor(left/resolution)*resolution;top=np.ceil(top/resolution)*resolution
    return dict(driver='GTiff',count=1,width=int(np.ceil((right-left)/resolution)),
                height=int(np.ceil((top-bottom)/resolution)),transform=from_origin(left,top,resolution,resolution),
                crs=crs,dtype='float32',nodata=np.nan,tiled=True,compress='lzw',BIGTIFF='IF_SAFER')


def prepare_inputs(config, log):
    """从本地DEM瓦片生成椭球高支持域与ORG有效覆盖，复用完成缓存。

    Copernicus的EGM2008正高H按h=H+N转换。N来自显式GEOID_PATH，不使用经验常数。
    仅这一步可按配置读取公开远程格网；几何计算本身无网络请求。
    """
    paths=config['paths'];options=config['scene'];out=Path(paths['OUTPUT_DIR'])
    cache=out/'data_cache';cache.mkdir(parents=True,exist_ok=True)
    dem=cache/'dem_ellipsoid.tif';reference=cache/'reference.tif';record=cache/'inputs.json'
    files=[Path(paths['ORG_PATH']),*[Path(p) for p in paths['DEM_TILES']]]
    # 准备阶段快速检查文件大小/mtime，避免每次续跑重读数百MB ORG；几何阶段再散列小型派生输入。
    identity=dict(inputs=[dict(name=p.name,size=p.stat().st_size,mtime=p.stat().st_mtime_ns) for p in files],
                  geoid=str(paths['GEOID_PATH']),resolution=options['resolution_m'],buffer=options['support_buffer_m'],
                  crs=options['output_crs'],nodata=options['org_nodata'],height=config['model']['height_datum'])
    if config['execution']['resume'] and record.exists() and dem.exists() and reference.exists():
        old=json.loads(record.read_text(encoding='utf-8'))
        if old['identity']==identity:
            log('[02] 复用椭球高DEM和ORG覆盖缓存');return dem,reference
    started=perf_counter()
    if config['model']['height_datum'] not in ('EGM2008','WGS84_ellipsoid'):
        raise ValueError('支持域准备只接受明确的EGM2008或WGS84_ellipsoid高程')
    with rasterio.open(paths['ORG_PATH']) as org:
        bounds=transform_bounds(org.crs,options['output_crs'],*org.bounds,densify_pts=21)
        target=grid(bounds,options['resolution_m'],options['output_crs'])
        rprofile=dict(target,dtype='uint8',nodata=0)
        with (WarpedVRT(org,crs=target['crs'],transform=target['transform'],width=target['width'],height=target['height'],
                       src_nodata=options['org_nodata'],nodata=np.nan,dtype='float32',resampling=Resampling.nearest) as vrt,
             rasterio.open(reference.with_suffix('.partial.tif'),'w',**rprofile) as dst):
            for y in range(0,dst.height,128):
                win=Window(0,y,dst.width,min(128,dst.height-y))
                image=vrt.read(1,window=win,masked=True)
                dst.write((~np.ma.getmaskarray(image)&np.isfinite(image.data)).astype('uint8'),1,window=win)
        reference.with_suffix('.partial.tif').replace(reference)
    b=options['support_buffer_m'];expanded=(bounds[0]-b,bounds[1]-b,bounds[2]+b,bounds[3]+b)
    profile=grid(expanded,options['resolution_m'],options['output_crs'])
    to_ll=Transformer.from_crs(profile['crs'],4326,always_xy=True)
    geoid=None
    try:
        if config['model']['height_datum']=='EGM2008':
            if paths.get('GEOID_PATH') is None:raise ValueError('EGM2008需要提供大地水准面格网')
            geoid=ElevationSampler(paths['GEOID_PATH'])
        with ExitStack() as stack:
            readers=[]
            for path in paths['DEM_TILES']:
                src=stack.enter_context(rasterio.open(path))
                readers.append(stack.enter_context(WarpedVRT(src,crs=profile['crs'],transform=profile['transform'],
                    width=profile['width'],height=profile['height'],nodata=np.nan,dtype='float32',resampling=Resampling.bilinear)))
            dst=stack.enter_context(rasterio.open(dem.with_suffix('.partial.tif'),'w',**profile))
            valid_count=0;geoid_range=[None,None]
            for y in range(0,dst.height,128):
                win=Window(0,y,dst.width,min(128,dst.height-y));shape=(int(win.height),dst.width)
                heights=np.full(shape,np.nan,np.float32)
                for vrt in readers:
                    values=vrt.read(1,window=win,masked=True).filled(np.nan)
                    use=~np.isfinite(heights)&np.isfinite(values);heights[use]=values[use]
                if geoid is not None:
                    rr,cc=np.mgrid[y:y+shape[0],0:shape[1]];tr=profile['transform']
                    lon,lat=to_ll.transform(tr.c+(cc+.5)*tr.a,tr.f+(rr+.5)*tr.e)
                    n=geoid.sample(lon,lat);heights+=n.astype('float32')
                    if np.isfinite(n).any():
                        geoid_range=[min(geoid_range[0] if geoid_range[0] is not None else np.inf,float(np.nanmin(n))),
                                     max(geoid_range[1] if geoid_range[1] is not None else -np.inf,float(np.nanmax(n)))]
                valid_count+=int(np.isfinite(heights).sum());dst.write(heights,1,window=win)
        dem.with_suffix('.partial.tif').replace(dem)
        write_json(record,dict(identity=identity,complete=True,height_conversion='h=H+N' if geoid else 'none',
                              geoid_undulation_range_m=geoid_range,valid_dem_pixels=valid_count,
                              elapsed_seconds=perf_counter()-started))
    finally:
        if geoid is not None:geoid.close()
    log(f'[02] DEM支持域及高程转换完成：{profile["width"]}×{profile["height"]}；{perf_counter()-started:.2f}s')
    return dem,reference


def profile_grid(config, model, reference, exe):
    """根据输出四角及高程上下界确定方位时间/地表角范围；横向保留外部地形支持。"""
    options=config['scene'];hmin,hmax=options['height_bounds_m']
    with rasterio.open(reference) as src:
        left,bottom,right,top=src.bounds
        x=np.array([left,left,right,right]*2);y=np.array([bottom,top,bottom,top]*2)
        z=np.array([hmin]*4+[hmax]*4)
        p=np.stack(Transformer.from_crs(CRS.from_user_input(src.crs).to_3d(),4978,always_xy=True).transform(x,y,z),axis=-1)
    solved=solve_points(exe,model,p);t,theta,good=profile_coordinates(exe,model,p,solved)
    if not good.all():raise ValueError('输出四角几何求解失败，不能猜测观测剖面范围')
    center=orbit_states(exe,model['orbit'],[float(np.mean(t))])[0]
    ground_speed=np.linalg.norm(center[3:])*A/np.linalg.norm(center[:3])
    dt=options['azimuth_step_m']/ground_speed;dtheta=options['range_step_m']/A
    times=np.arange(np.floor(t.min()/dt)*dt-dt,np.ceil(t.max()/dt)*dt+2*dt,dt)
    margin=options['support_buffer_m']/A
    theta=np.arange(np.floor((theta.min()-margin)/dtheta)*dtheta,np.ceil((theta.max()+margin)/dtheta)*dtheta+dtheta,dtheta)
    if len(times)*len(theta)>options['max_profile_points']:
        raise ValueError('剖面网格超过内存/磁盘预算，请检查输出范围或采样间隔')
    return times,theta


def combine_masks(shadow, layover):
    """0/1/255联合：任一确认排除即1，两者均确认无影响才0，其余保持未知。"""
    return np.where((shadow==1)|(layover==1),1,np.where((shadow==0)&(layover==0),0,255)).astype('uint8')


def run_visibility(config, metadata, log=print):
    """可续跑单景计算并返回质量记录；无需运行原阈值分割，也不会更改水体成果。"""
    started=perf_counter();paths=config['paths'];options=config['scene'];execution=config['execution']
    exe=paths['CORE_PATH'];output=Path(paths['OUTPUT_DIR']);cache=output/'data_cache';cache.mkdir(parents=True,exist_ok=True)
    settings=dict(config['model'])
    if not settings['radar_grid_confirmed']:
        if not options.get('allow_unconfirmed_geometry',False) or not options.get('geometry_assumption_note'):
            raise ValueError('真实成像约定未确认；探索性对照必须明确启用并记录几何假设')
        log('[WARN] 以显式零Doppler假设做探索性对照；不作为已验证的生产几何。')
        settings['radar_grid_confirmed']=True
    model=make_model(metadata,settings)
    if model['doppler_mode']!=0:raise ValueError('当前整景剖面只支持零Doppler实验')
    os.environ['OMP_NUM_THREADS']=str(execution['threads'])
    dem,reference=prepare_inputs(config,log);prepared=perf_counter()
    times,theta=profile_grid(config,model,reference,exe)
    identity=dict(version=VERSION,xml=hashlib.sha256(metadata['xml_bytes']).hexdigest(),
                  dem=sha256(dem),reference=sha256(reference),exe=sha256(exe),model=config['model'],scene=options)
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    progress=cache/'scene_progress.json';state=json.loads(progress.read_text()) if progress.exists() else {}
    names=('shadow_mask','layover_mask','terrain_exclusion','geometry_validity','radar_coverage')
    products=[output/(name+'.tif') for name in names];report_path=output/'geometry_quality.json'
    if execution['resume'] and state.get('key')==key and state.get('complete') and all(p.exists() for p in products) and report_path.exists():
        log('[03] 复用完整整景几何结果')
        reused=json.loads(report_path.read_text(encoding='utf-8'));reused['cache_reused_this_run']=True
        return reused
    mask_path=cache/'profiles.npy';resuming=execution['resume'] and state.get('key')==key and mask_path.exists()
    if not resuming:
        state=dict(key=key,complete=False,next_profile=0,next_row=0,counts={},max_height_residual_m=0.,profile_seconds=0.,mapping_seconds=0.)
        masks=np.lib.format.open_memmap(mask_path,mode='w+',dtype='uint8',shape=(len(times),len(theta),2));masks[:]=255;masks.flush()
        write_json(progress,state)
    else:masks=np.lib.format.open_memmap(mask_path,mode='r+')
    log(f'[03] 观测支持网格：{len(times)}条×{len(theta)}点；OpenMP {execution["threads"]}线程')
    sampler=ElevationSampler(dem);phase=perf_counter()
    try:
        for start in range(state['next_profile'],len(times),execution['profile_batch_rows']):
            end=min(start+execution['profile_batch_rows'],len(times));tick=perf_counter()
            result,_,_,error=build_profiles(exe,model,sampler,times[start:end],theta,options['height_bounds_m'],options['height_tolerance_m'])
            masks[start:end]=result;masks.flush()
            state['next_profile']=end;state['profile_seconds']+=perf_counter()-tick
            state['max_height_residual_m']=max(state['max_height_residual_m'],error or 0)
            write_json(progress,state)
            if end==len(times) or start//execution['profile_batch_rows']%10==0:
                log(f'[03] 完整剖面 {end}/{len(times)}；本次阶段{perf_counter()-phase:.1f}s')
    finally:sampler.close()
    # 映射采用(t,theta)最近邻，不用有多值的(t,range)插值；不会抹掉叠掩分支。
    phase=perf_counter();partials={n:cache/(n+'.partial.tif') for n in names}
    with rasterio.open(reference) as ref,rasterio.open(dem) as elevation:
        profile=ref.profile.copy();profile.update(nodata=255,dtype='uint8')
        if not all(p.exists() for p in partials.values()):state['next_row']=0;state['counts']={};state['mapping_seconds']=0.
        with WarpedVRT(elevation,crs=ref.crs,transform=ref.transform,width=ref.width,height=ref.height,
                       dtype='float32',nodata=np.nan,resampling=Resampling.bilinear) as source:
            transform=Transformer.from_crs(CRS.from_user_input(ref.crs).to_3d(),4978,always_xy=True)
            with ExitStack() as stack:
                writers={n:stack.enter_context(rasterio.open(p,'w' if state['next_row']==0 else 'r+',**(profile if state['next_row']==0 else {}))) for n,p in partials.items()}
                # GDAL VRT不支持boundless读；输出中心坐标直接采样椭球高DEM，法向本阶段无需计算。
                for y in range(state['next_row'],ref.height,execution['map_block_rows']):
                    tick=perf_counter();win=Window(0,y,ref.width,min(execution['map_block_rows'],ref.height-y))
                    rr,cc=np.mgrid[y:y+int(win.height),0:ref.width];a=ref.transform
                    h=source.read(1,window=win,masked=True).filled(np.nan)
                    p=np.stack(transform.transform(a.c+(cc+.5)*a.a,a.f+(rr+.5)*a.e,h),axis=-1).reshape(-1,3)
                    solved=solve_points(exe,model,p);t,angle,good=profile_coordinates(exe,model,p,solved)
                    row=np.rint(np.where(good,(t-times[0])/(times[1]-times[0]),-1)).astype('int64')
                    col=np.rint(np.where(good,(angle-theta[0])/(theta[1]-theta[0]),-1)).astype('int64')
                    covered=(ref.read(1,window=win).ravel()==1)&(solved[:,5]==0)
                    sample=covered&good&(row>=0)&(row<len(times))&(col>=0)&(col<len(theta))
                    value=np.full((len(p),2),255,np.uint8);value[sample]=masks[row[sample],col[sample]]
                    shape=h.shape;shadow=value[:,0].reshape(shape);layover=value[:,1].reshape(shape)
                    arrays=dict(shadow_mask=shadow,layover_mask=layover,terrain_exclusion=combine_masks(shadow,layover),
                                geometry_validity=good.reshape(shape).astype('uint8'),radar_coverage=covered.reshape(shape).astype('uint8'))
                    for name,array in arrays.items():
                        writers[name].write(array,1,window=win)
                        if name in names[:3]:
                            counter=state['counts'].setdefault(name,{'0':0,'1':0,'255':0})
                            for val,count in zip(*np.unique(array[covered.reshape(shape)],return_counts=True)):counter[str(val)]+=int(count)
                    # 关闭并重开写句柄后才提交已完成行，避免GDAL缓存尚未落盘却被续跑跳过。
                    for writer in writers.values():writer.close()
                    writers={n:stack.enter_context(rasterio.open(p,'r+')) for n,p in partials.items()}
                    state['next_row']=y+int(win.height);state['mapping_seconds']+=perf_counter()-tick;write_json(progress,state)
                    log(f'[04] 地理网格 {state["next_row"]}/{ref.height}行；阶段{perf_counter()-phase:.1f}s')
    del masks
    for name,p in partials.items():p.replace(output/(name+'.tif'))
    report=dict(complete=True,cache_reused_this_run=False,algorithm=VERSION,production_ready=False,geometry_confirmed=config['model']['radar_grid_confirmed'],
                geometry_assumption=options.get('geometry_assumption_note'),doppler_mode='zero',height_datum='WGS84_ellipsoid',
                profile_shape=[len(times),len(theta)],profile_step_m=[options['azimuth_step_m'],options['range_step_m']],
                resolution_m=options['resolution_m'],height_bounds_m=options['height_bounds_m'],
                source=identity,counts=state['counts'],max_height_residual_m=state['max_height_residual_m'],
                profile_seconds=state['profile_seconds'],mapping_seconds=state['mapping_seconds'],
                current_run_preparation_seconds=prepared-started,current_run_total_seconds=perf_counter()-started,
                current_run_geometry_seconds=perf_counter()-prepared,
                time_budget_seconds=execution['time_budget_seconds'],
                note='掩膜是指定零Doppler/DEM/采样间隔下的候选；未知255。不得直接替换现有水体。')
    report['geometry_time_budget_met']=report['current_run_geometry_seconds']<=execution['time_budget_seconds']
    write_json(report_path,report);state['complete']=True;write_json(progress,state)
    log(f'[05] 整景掩膜完成：几何{report["current_run_geometry_seconds"]:.2f}s；含准备{report["current_run_total_seconds"]:.2f}s')
    return report
