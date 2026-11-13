"""单景非破坏对照与资源监测；没有真值时只报告差异，不计算虚假的精度。"""
from pathlib import Path
from time import perf_counter
from threading import Thread,Event
import json
import xml.etree.ElementTree as ET
import numpy as np
import os
import ctypes
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from pyproj import Transformer,Geod
from .terrain_io import sha256,write_json
from .metadata import read_bc_metadata,make_model
from .core import solve_points,orbit_states,ACTIVE_CHILDREN
from .profiles import ElevationSampler


def process_rss(pid):
    """读取Windows工作集或Linux驻留页，不为性能统计额外安装依赖。进程已退出时返回0。"""
    if os.name!='nt':
        try:return int(Path(f'/proc/{pid}/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
        except (OSError,IndexError):return 0
    class Counters(ctypes.Structure):
        _fields_=[('cb',ctypes.c_ulong),('faults',ctypes.c_ulong)]+[(n,ctypes.c_size_t) for n in
            ('peak','working','quota_peak_paged','quota_paged','quota_peak_nonpaged','quota_nonpaged','pagefile','peak_pagefile')]
    kernel=ctypes.WinDLL('kernel32',use_last_error=True);psapi=ctypes.WinDLL('psapi',use_last_error=True)
    kernel.OpenProcess.restype=ctypes.c_void_p;kernel.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong]
    kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    psapi.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.POINTER(Counters),ctypes.c_ulong]
    handle=kernel.OpenProcess(0x0400|0x0010,0,pid)
    if not handle:return 0
    try:
        values=Counters();values.cb=ctypes.sizeof(values)
        return int(values.working) if psapi.GetProcessMemoryInfo(handle,ctypes.byref(values),values.cb) else 0
    finally:kernel.CloseHandle(handle)


class ResourceMonitor:
    """每20ms采样父/子进程RSS；为采样峰值，非操作系统精确瞬时峰值。"""
    def __enter__(self):
        self.stop=Event();self.peak=0;self.started=perf_counter()
        def poll():
            while not self.stop.is_set():
                used=sum(process_rss(pid) for pid in [os.getpid(),*ACTIVE_CHILDREN.copy()])
                self.peak=max(self.peak,used);self.stop.wait(.02)
        self.thread=Thread(target=poll,daemon=True);self.thread.start();return self

    def __exit__(self,*args):
        self.stop.set();self.thread.join();self.elapsed=perf_counter()-self.started

    def result(self):
        return dict(sampled_peak_parent_children_rss_mib=self.peak/1024**2,
                    observed_seconds=self.elapsed,sample_interval_seconds=.02,
                    cpu_logical_count=os.cpu_count())


def compare_sampling(coarse_dir, fine_dir):
    """同一输出网格的采样敏感性；正类集合IoU表示结果稳定性，不是真值精度。"""
    coarse_dir,fine_dir=Path(coarse_dir),Path(fine_dir);result={}
    with rasterio.open(fine_dir/'radar_coverage.tif') as src:covered=src.read(1)==1
    for name in ('shadow_mask','layover_mask','terrain_exclusion'):
        with rasterio.open(coarse_dir/(name+'.tif')) as a,rasterio.open(fine_dir/(name+'.tif')) as b:
            if (a.shape,a.crs,a.transform)!=(b.shape,b.crs,b.transform):raise ValueError('采样对照必须使用同一输出网格')
            x,y=a.read(1),b.read(1)
        common=covered&(x<2)&(y<2);intersection=int((common&(x==1)&(y==1)).sum())
        union=int((common&((x==1)|(y==1))).sum())
        result[name]=dict(common_pixels=int(common.sum()),different_pixels=int((common&(x!=y)).sum()),
            agreement_fraction=float((x[common]==y[common]).mean()) if common.any() else None,
            positive_set_iou=intersection/union if union else None,
            coarse_only=int((common&(x==1)&(y==0)).sum()),fine_only=int((common&(x==0)&(y==1)).sum()),
            coarse_unknown_within_coverage=int((covered&(x==255)).sum()),fine_unknown_within_coverage=int((covered&(y==255)).sum()))
    write_json(fine_dir/'sampling_comparison.json',dict(coarse=str(coarse_dir),fine=str(fine_dir),results=result))
    return result


def warped_array(path, reference, dtype='float32', nodata=np.nan):
    """只在小型地形对照网格读取整图；类别最近邻，未覆盖保持NoData。"""
    with rasterio.open(path) as src, WarpedVRT(src,crs=reference.crs,transform=reference.transform,
            width=reference.width,height=reference.height,dtype=dtype,nodata=nodata,resampling=Resampling.nearest) as vrt:
        return vrt.read(1)


def compare_scene(config, destination=None):
    """同网格、同有效域比较新旧地形掩膜，同时检查XML粗略角点的一致性。

    输出位于新实验目录；现有水体/地形文件只读。面积变化不能解释成精度提升。
    """
    paths=config['paths'];out=Path(paths['OUTPUT_DIR']);baseline=Path(paths['BASELINE_DIR'])
    destination=Path(destination) if destination is not None else out
    destination.mkdir(parents=True,exist_ok=True)
    local_meta=json.loads((baseline/'terrain_geometry.json').read_text(encoding='utf-8'))
    local_label='Corrected local' if local_meta.get('method')=='gui_local_ground_to_sensor_v2' else 'Legacy GUI'
    original={n:sha256(baseline/(n+'.tif')) for n in ('shadow_mask','layover_mask','terrain_exclusion','water')}
    report={'comparison':f'zero-Doppler candidate versus {local_label}','local_method':local_meta.get('method'),'ground_truth_available':False,'original_sha256':original}
    with rasterio.open(out/'data_cache/reference.tif') as ref:
        valid=ref.read(1)==1
        new={n:warped_array(out/(n+'.tif'),ref,'uint8',255) for n in ('shadow_mask','layover_mask','terrain_exclusion','radar_coverage')}
        old={n:warped_array(baseline/(n+'.tif'),ref,'uint8',255) for n in ('shadow_mask','layover_mask','terrain_exclusion','water')}
        sar=warped_array(paths['ORG_PATH'],ref);area=abs(ref.transform.a*ref.transform.e-ref.transform.b*ref.transform.d)
        report['pixel_area_m2']=area;report['counts']={}
        for name in ('shadow_mask','layover_mask','terrain_exclusion'):
            common=valid&(new['radar_coverage']==1)&(new[name]<2)&(old[name]<2)
            a,b=old[name],new[name]
            counts={label:int((common&condition).sum()) for label,condition in {
                'both_clear':(a==0)&(b==0),'old_only':(a==1)&(b==0),'new_only':(a==0)&(b==1),'both_excluded':(a==1)&(b==1)}.items()}
            counts.update(common_pixels=int(common.sum()),new_unknown_within_old_coverage=int((valid&(old[name]<2)&(new[name]==255)).sum()))
            water=common&(old['water']==1)
            counts['existing_water_pixels']=int(water.sum());counts['new_mask_on_existing_water_pixels']=int((water&(b==1)).sum())
            report['counts'][name]=counts
        report['note']='旧terrain_exclusion含15度坡度规则，新联合掩膜仅含物理阴影/叠掩候选；应优先逐类比较。现有water不是人工真值。'
    # 诊断复用已修正的函数和实际航向；不再用heading+180模拟修正，避免重复反转。
    if paths.get('GUI_SLOPE_PATH') and paths.get('GUI_ASPECT_PATH'):
        from thres_segmentation.terrain_geometry import classify_terrain
        previous=json.loads((baseline/'terrain_geometry.json').read_text(encoding='utf-8'))
        with rasterio.open(out/'data_cache/reference.tif') as ref:
            slope=warped_array(paths['GUI_SLOPE_PATH'],ref);aspect=warped_array(paths['GUI_ASPECT_PATH'],ref)
            flipped=classify_terrain(slope,aspect,previous['heading_deg'],previous['incidence_deg'],previous['look_side'],previous['slope_threshold_deg'])
        report['direction_correction_diagnostic']={}
        for name,code in [('shadow_mask',1),('layover_mask',2)]:
            common=valid&(new['radar_coverage']==1)&(new[name]<2)&(flipped>0)&(old[name]<2)
            a,b=flipped==code,new[name]==1
            report['direction_correction_diagnostic'][name]=dict(common_pixels=int(common.sum()),
                original_overlap_pixels=int((common&(old[name]==1)&b).sum()),
                corrected_overlap_pixels=int((common&a&b).sum()),
                corrected_local_pixels=int((common&a).sum()),profile_pixels=int((common&b).sum()))
    # XML只有四个低精度角点且未给高程：用DEM高程核查，不作为独立测量控制点。
    meta=read_bc_metadata(paths['XML_PATH'],config.get('xml_member'));root=ET.fromstring(meta['xml_bytes'])
    for e in root.iter():e.tag=e.tag.split('}')[-1]
    corners=root.findall('geolocationGrid/geolocationGridPointList/geolocationGridPoint')
    if corners:
        ll=np.array([[float(p.findtext('longitude')),float(p.findtext('latitude'))] for p in corners])
        sampler=ElevationSampler(out/'data_cache/dem_ellipsoid.tif')
        try:h=sampler.sample(ll[:,0],ll[:,1])
        finally:sampler.close()
        xyz=np.stack(Transformer.from_crs(4979,4978,always_xy=True).transform(ll[:,0],ll[:,1],h),axis=-1)
        settings=dict(config['model'],radar_grid_confirmed=True);model=make_model(meta,settings)
        middle=model['time0']+(model['rows']-1)*model['line_time']/2
        state=orbit_states(paths['CORE_PATH'],model['orbit'],[middle])[0]
        sat_lon,sat_lat,_=Transformer.from_crs(4978,4979,always_xy=True).transform(*state[:3])
        ground_lon,ground_lat=ll.mean(axis=0)
        bearing=Geod(ellps='WGS84').inv(ground_lon,ground_lat,sat_lon,sat_lat)[0]%360
        report['ground_to_sensor_bearing_deg']=float(bearing)
        result=solve_points(paths['CORE_PATH'],model,xyz)
        report['xml_corner_check']=[]
        for p,r,height in zip(corners,result,h):
            report['xml_corner_check'].append(dict(line=int(p.findtext('line')),pixel=int(p.findtext('pixel')),
                elevation_m=float(height) if np.isfinite(height) else None,status=int(r[5]),
                range_difference_m=float((r[0]-float(p.findtext('pixel')))*model['range_spacing']) if np.isfinite(r[0]) else None,
                time_difference_seconds=float((r[1]-float(p.findtext('line')))*model['line_time']) if np.isfinite(r[1]) else None))
    # 面板固定类别和显示拉伸，突出空间差异；不通过更改水体阈值制造收益。
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import ListedColormap
    fig=Figure(figsize=(16,8));FigureCanvasAgg(fig);axes=fig.subplots(2,4)
    good=valid&np.isfinite(sar)&(sar>0);lo,hi=np.percentile(sar[good],[2,98])
    axes[0,0].imshow(np.ma.masked_where(~valid,sar),cmap='gray',vmin=lo,vmax=hi);axes[0,0].set_title('ORG amplitude P2-P98')
    panels=[(0,1,old['shadow_mask'],local_label+' shadow'),(0,2,new['shadow_mask'],'New profile shadow'),
            (1,0,old['layover_mask'],local_label+' layover'),(1,1,new['layover_mask'],'New profile layover'),
            (1,2,new['terrain_exclusion'],'New physical exclusion'),(0,3,old['terrain_exclusion'],local_label+' incl. slope15')]
    for row,col,a,title in panels:
        display=np.where(~valid| (a==255),2,a)
        axes[row,col].imshow(display,cmap=ListedColormap(['white','red','lightgray']),vmin=0,vmax=2);axes[row,col].set_title(title)
    common=valid&(old['terrain_exclusion']<2)&(new['terrain_exclusion']<2)
    diff=np.where(common,old['terrain_exclusion']+2*new['terrain_exclusion'],4)
    axes[1,3].imshow(diff,cmap=ListedColormap(['white','orange','blue','purple','lightgray']),vmin=0,vmax=4)
    axes[1,3].set_title('Orange old / blue new / purple both')
    for a in axes.ravel():a.axis('off')
    fig.suptitle('Miyun 2025-07-27 02:52:42 | zero-Doppler hypothesis | gray: unknown/outside')
    fig.tight_layout();fig.savefig(destination/'terrain_comparison.png',dpi=130);fig.clear()
    report['original_products_unchanged']=original=={n:sha256(baseline/(n+'.tif')) for n in original}
    write_json(destination/'comparison.json',report)
    return report
