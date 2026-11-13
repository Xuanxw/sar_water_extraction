"""GDAL坡度/坡向的局部地形分类，已修正照射方向；不包含远处山体遮挡搜索。"""
import json,os,subprocess
from pathlib import Path
import numpy as np
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.windows import Window

TERRAIN_METHOD_VERSION='gui_local_ground_to_sensor_v2'


def prepare_gui_layers(dem, executable):
    """与gis_worker一致：ZevenbergenThorne、computeEdges、平地坡向为0。"""
    dem=Path(dem);exe=Path(executable)
    env=os.environ.copy();env['PATH']=str(exe.parent)+os.pathsep+env.get('PATH','')
    result={}
    for kind in ('slope','aspect'):
        output=dem.with_name(dem.stem+'_gui_zt_'+kind+'.tif')
        if not output.exists():
            pending=output.with_suffix('.partial.tif')
            args=[str(exe),kind,str(dem),str(pending),'-alg','ZevenbergenThorne','-compute_edges','-co','COMPRESS=LZW','-co','TILED=YES']
            if kind=='aspect':args+=['-zero_for_flat']
            subprocess.run(args,env=env,check=True)
            pending.replace(output)
        result[kind]=output
    return result


def classify_terrain(slope,aspect,heading,incidence,look_side,slope_threshold):
    """GDAL坡向/入射角均为度：0未知、1局部阴影、2叠掩候选、3高坡、4正常。

    坡向是下坡朝向，亦为向上地表法向的水平分量方向。法向点积必须使用地面→卫星。
    此次只修正180度方向，保留总坡度阈值规则；它仍不是完整观测剖面模型。
    """
    if look_side not in ('left','right'):raise ValueError('look_side必须为left或right')
    sensor_to_ground=(heading+90)%360 if look_side=='right' else (heading-90)%360
    ground_to_sensor=(sensor_to_ground+180)%360
    phi=(aspect-ground_to_sensor+180)%360-180
    rs=np.deg2rad(slope);rt=np.deg2rad(incidence)
    cosine=np.cos(rt)*np.cos(rs)+np.sin(rt)*np.sin(rs)*np.cos(np.deg2rad(phi))
    valid=np.isfinite(slope)&np.isfinite(aspect)&(slope>=0)&(slope<=90)&(aspect>=0)&(aspect<=360)&np.isfinite(cosine)
    cls=np.zeros(slope.shape,np.uint8)
    cls[valid&(cosine<=0)]=1
    facing=(cosine>0)&(np.abs(phi)<=90)
    cls[valid&facing&(slope>incidence)]=2
    cls[valid&facing&(slope<=incidence)&(slope>slope_threshold)]=3
    cls[valid&(cls==0)]=4
    return cls


def geometry_parameters(metadata):
    """只读取本景XML元数据，缺失时停止，不用默认角度生成伪掩膜。"""
    heading=float(metadata['platformHeading']);theta=float(metadata['incidenceAngleMidSwath'])
    look=metadata['look_side'].strip().lower()
    if look not in ('left','right') or not (0<theta<90) or not np.isfinite(heading):raise ValueError('本景SAR几何参数无效')
    return heading,theta,look


def generate_terrain_mask(reference,slope_path,aspect_path,metadata,folder,threshold=15,block_size=512):
    """分块在SAR网格计算原GUI分类；坡向最近邻，坡度双线性并保留原DEM空值范围。"""
    folder=Path(folder);heading,theta,look=geometry_parameters(metadata)
    outputs={k:folder/(k+'.tif') for k in ('terrain_class','terrain_exclusion','layover_mask','shadow_mask')}
    counts=np.zeros(5,np.int64)
    from contextlib import ExitStack
    with ExitStack() as stack:
        src=stack.enter_context(rasterio.open(reference));readers={};nearest={}
        for name,path in (('slope',slope_path),('aspect',aspect_path)):
            base=stack.enter_context(rasterio.open(path))
            kw=dict(crs=src.crs,transform=src.transform,width=src.width,height=src.height,dtype='float32',nodata=np.nan)
            readers[name]=stack.enter_context(WarpedVRT(base,**kw,resampling=Resampling.bilinear if name=='slope' else Resampling.nearest))
            nearest[name]=stack.enter_context(WarpedVRT(base,**kw,resampling=Resampling.nearest))
        profile=src.profile.copy();profile.update(dtype='uint8',nodata=255,count=1,compress='lzw',BIGTIFF='IF_SAFER')
        writers={k:stack.enter_context(rasterio.open(p.with_suffix('.partial.tif'),'w',**profile)) for k,p in outputs.items()}
        for y in range(0,src.height,block_size):
            w=Window(0,y,src.width,min(block_size,src.height-y));values={}
            for name in readers:
                a=readers[name].read(1,window=w,masked=True).filled(np.nan)
                footprint=nearest[name].read(1,window=w,masked=True).filled(np.nan)
                maximum=90 if name=='slope' else 360
                a[~np.isfinite(footprint)|(footprint<0)|(footprint>maximum)]=np.nan
                values[name]=a
            cls=classify_terrain(values['slope'],values['aspect'],heading,theta,look,threshold)
            valid=src.read_masks(1,window=w)>0
            counts+=np.bincount(cls[valid],minlength=5)
            excluded=np.isin(cls,(1,2,3)).astype('uint8')
            writers['terrain_class'].write(np.where(valid,cls,255).astype('uint8'),1,window=w)
            writers['terrain_exclusion'].write(np.where(valid,excluded,255).astype('uint8'),1,window=w)
            writers['layover_mask'].write(np.where(valid,cls==2,255).astype('uint8'),1,window=w)
            writers['shadow_mask'].write(np.where(valid,cls==1,255).astype('uint8'),1,window=w)
    for p in outputs.values():p.with_suffix('.partial.tif').replace(p)
    sensor_to_ground=(heading+90)%360 if look=='right' else (heading-90)%360
    report=dict(method=TERRAIN_METHOD_VERSION,heading_deg=heading,incidence_deg=theta,look_side=look,slope_threshold_deg=threshold,
                sensor_to_ground_azimuth_deg=sensor_to_ground,ground_to_sensor_azimuth_deg=(sensor_to_ground+180)%360,
                class_pixels=dict(zip(('unknown','shadow','layover','foreshortening','normal'),map(int,counts))),
                note='已修正法向点积的180度方向错误；其余局部规则保留，没有沿视线搜索远处山体遮挡。')
    (folder/'terrain_geometry.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('  方向修正后的局部地形分类：'+str(report['class_pixels']),flush=True)
    return outputs


def save_terrain_preview(mask_path,class_path,destination,max_side=1200):
    """显示最终水体与源码地形分类；只读取缩略图，不改变原始分辨率成果。"""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import ListedColormap
    with rasterio.open(mask_path) as src,rasterio.open(class_path) as terrain:
        ratio=min(1,max_side/max(src.shape));size=tuple(max(1,round(v*ratio)) for v in src.shape)
        water=src.read(1,out_shape=size);classes=terrain.read(1,out_shape=size)
    unknown=water==255;excluded=np.isin(classes,(1,2,3))
    fig=Figure(figsize=(17,5));FigureCanvasAgg(fig);axes=fig.subplots(1,4)
    axes[0].imshow(np.ma.masked_where(unknown,classes),cmap=ListedColormap(['gray','purple','red','orange','white']),vmin=0,vmax=4)
    axes[0].set_title('Terrain: purple shadow / red layover\norange foreshortening / white normal')
    axes[1].imshow(np.ma.masked_where(unknown,classes==2),cmap='Reds',vmin=0,vmax=1);axes[1].set_title('Layover mask')
    axes[2].imshow(np.ma.masked_where(unknown,water),cmap='Blues',vmin=0,vmax=1);axes[2].set_title('Final water polygon raster')
    axes[3].imshow(np.ma.masked_where(unknown,excluded),cmap='Reds',vmin=0,vmax=1);axes[3].set_title('Combined terrain exclusion mask')
    for ax in axes:ax.axis('off')
    fig.suptitle(Path(mask_path).parent.name,fontsize=9);fig.tight_layout();fig.savefig(destination,dpi=110);fig.clear()
