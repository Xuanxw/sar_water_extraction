"""可选的水体SHP按日期合并；不由阈值提取主流程调用。"""
import json
import sys
from threading import Event, Thread
from time import perf_counter
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from thres_segmentation.run_log import RunLog

import fiona
from pyproj import CRS, Transformer
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
from shapely import transform as transform_geometry, from_wkb, to_wkb

# 独立执行设置。只读取已完成的逐景产品，不修改它们。
INPUT_ROOT=Path(r'E:\InSAR\水体提取\北京密云区\outputs\thres_seg_20261112')
OUTPUT_DIR=Path(r'E:\InSAR\水体提取\北京密云区\outputs\thres_seg_20261112\merged')
TARGET_CRS='EPSG:32650'
ROI_PATH=Path(r'E:\InSAR\水体提取\北京密云区\密云区行政划分.shp')  # 不需要ROI时设为None。
SAVE_QUICKLOOK=True
PREVIEW_SIZE=1600  # 快视图栅格的最长边，不影响SHP几何精度。
CONSOLE_COLOR=True
MERGE_BATCH_SIZE=2000  # 每个叶子批次独立合并，不反复合并之前的全部结果。
WRITE_BATCH_SIZE=1000
HEARTBEAT_SECONDS=10  # GEOS单次计算期间显示等待时长，不伪造完成百分比。
RESUME=True  # 同一输入下复用本输出目录.merge_cache；换输入/参数请换输出目录或设False。


@contextmanager
def show_activity(log,label):
    """长计算期间定时显示存活信息，退出或异常时停止提示线程。"""
    stop=Event();started=perf_counter()
    def heartbeat():
        while not stop.wait(HEARTBEAT_SECONDS):
            log.message(f'{label}：处理中，已耗时{perf_counter()-started:.0f}秒')
    thread=Thread(target=heartbeat,daemon=True);thread.start()
    try:
        yield
    finally:
        stop.set();thread.join()


def cached_union(geometries, destination, log, label):
    """一个树节点一次求并集；完整WKB原子发布，中断节点下次重算。"""
    if RESUME and destination.exists():
        log.message(f'{label}：复用完成的几何节点','REUSED')
        return
    started=perf_counter()
    log.message(f'{label}：开始几何并集')
    with show_activity(log,label):
        # 上层传入的是两个已合并节点；叶子传入固定数量的原始面。
        result=unary_union(geometries)
        pending=destination.with_suffix('.partial.wkb')
        pending.write_bytes(to_wkb(result));pending.replace(destination)
    log.message(f'{label}：完成，耗时{perf_counter()-started:.1f}秒','DONE')


def merge_date(records,target,cache,log):
    """磁盘上的平衡归并树：每个节点只参与下一层一次，叶子可断点复用。"""
    cache.mkdir(parents=True,exist_ok=True)
    nodes=[]
    for scene_index,record in enumerate(records,1):
        with log.stage('03',f'读取第{scene_index}/{len(records)}景',record['scene_id']), fiona.open(record['water_shp']) as src:
            if not src.crs:raise ValueError(f"缺少CRS：{record['water_shp']}")
            same_crs=CRS.from_user_input(src.crs)==target
            transform=None if same_crs else Transformer.from_crs(src.crs,target,always_xy=True).transform
            total=len(src);batch=[];last_node=None
            print(f'要素数：{total}；同坐标系直接读取：{same_crs}',flush=True)
            for index,feature in enumerate(src):
                node=cache/f'scene_{scene_index:03d}_batch_{index//MERGE_BATCH_SIZE:06d}.wkb'
                if node!=last_node:
                    nodes.append(node);last_node=node
                # 已完成批次只遍历Fiona记录定位，跳过几何构造和计算。
                if not (RESUME and node.exists()) and feature['geometry']:
                    geom=shape(feature['geometry'])
                    if transform is not None:geom=transform_geometry(geom,transform,interleaved=False,include_z=None)
                    batch.append(geom)
                if (index+1)%MERGE_BATCH_SIZE==0 or index+1==total:
                    cached_union(batch,node,log,f'叶子批次；本景{index+1}/{total}面 ({(index+1)/max(total,1):.1%})')
                    batch=[]
    level=0
    while len(nodes)>1:
        level+=1;parents=[];pairs=(len(nodes)+1)//2
        with log.stage('04',f'归并第{level}层：{len(nodes)}个节点',cache.name):
            for index in range(0,len(nodes),2):
                if index+1==len(nodes):parents.append(nodes[index]);continue
                parent=cache/f'level_{level:03d}_pair_{index//2:06d}.wkb'
                if RESUME and parent.exists():
                    log.message(f'归并第{level}层 {index//2+1}/{pairs}：复用','REUSED')
                else:
                    children=[from_wkb(p.read_bytes()) for p in nodes[index:index+2]]
                    cached_union(children,parent,log,f'归并第{level}层 {index//2+1}/{pairs}')
                parents.append(parent)
        nodes=parents
    return from_wkb(nodes[0].read_bytes()) if nodes else unary_union([])


def save_quicklook(water, roi, path, title, target, max_side):
    """仅在显示网格上栅格化，保留水体孔洞；ROI叠加显示，不裁剪合并结果。"""
    import numpy as np
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.patches import Patch
    geometries=[g for g in (water,roi) if g is not None and not g.is_empty]
    if geometries:
        bounds=[g.bounds for g in geometries]
        left,bottom=min(b[0] for b in bounds),min(b[1] for b in bounds)
        right,top=max(b[2] for b in bounds),max(b[3] for b in bounds)
    else:
        left,bottom,right,top=0,0,1,1
    margin=max(right-left,top-bottom)*.04
    left-=margin;right+=margin;bottom-=margin;top+=margin
    scale=max_side/max(right-left,top-bottom)
    width,height=max(1,round((right-left)*scale)),max(1,round((top-bottom)*scale))
    transform=from_bounds(left,bottom,right,top,width,height)
    rgb=np.full((height,width,3),255,np.uint8)
    if water is not None and not water.is_empty:
        mask=rasterize([(mapping(water),1)],out_shape=(height,width),transform=transform,dtype='uint8')
        rgb[mask==1]=(35,115,195)
    if roi is not None and not roi.is_empty:
        border=rasterize([(mapping(roi.boundary),1)],out_shape=(height,width),transform=transform,dtype='uint8')
        rgb[border==1]=(210,65,45)
    fig=Figure(figsize=(9,8));FigureCanvasAgg(fig)
    ax=fig.subplots();ax.imshow(rgb,extent=(left,right,bottom,top),origin='upper')
    ax.set_aspect('equal');ax.set_title(title);ax.set_xlabel(f'Easting (m) | {target.to_string()}');ax.set_ylabel('Northing (m)')
    legend=[]
    if water is not None:legend.append(Patch(facecolor='#2373c3',label='Merged water'))
    if roi is not None:legend.append(Patch(facecolor='none',edgecolor='#d2412d',label='ROI boundary'))
    if legend:ax.legend(handles=legend,loc='upper right')
    if water is not None and water.is_empty:ax.text(.5,.5,'No detected water',transform=ax.transAxes,ha='center')
    fig.text(.5,.01,'White background does not establish non-water or valid SAR coverage.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.03,1,1));fig.savefig(path,dpi=150);fig.clear()


def merge_results(input_root,output_dir,target_crs,roi_path=None,save_previews=True,preview_size=1600):
    """同步输出控制台、processing.log和stage_timings.csv；异常与中断也记入日志。"""
    log=RunLog(output_dir,CONSOLE_COLOR)
    with log.capture(),log.stage('00','按日期合并水体总计'):
        print(f'输入：{input_root}；输出：{output_dir}；ROI：{roi_path}',flush=True)
        return _merge_results(input_root,output_dir,target_crs,roi_path,save_previews,preview_size,log)


def _merge_results(input_root,output_dir,target_crs,roi_path,save_previews,preview_size,log):
    """同日期几何并集；可选导出ROI和快视图，不裁剪水体、不作时序判别。"""
    target=CRS.from_user_input(target_crs)
    if not target.is_projected or any(abs(a.unit_conversion_factor-1)>1e-9 for a in target.axis_info[:2]):
        raise ValueError('面积输出要求米制投影坐标系')
    grouped={}
    with log.stage('01','扫描已完成的逐景成果'):
        for path in sorted(Path(input_root).glob('*/*/scene.json')):
            record=json.loads(path.read_text(encoding='utf-8'))
            if record['status']!='complete':continue
            water=path.parent/'water.shp'
            if not water.is_file():raise FileNotFoundError(f'水体成果缺失：{water}')
            record['water_shp']=str(water)
            grouped.setdefault(record['date'],[]).append(record)
        print(f'发现{len(grouped)}个日期、{sum(map(len,grouped.values()))}景已完成成果',flush=True)
    if not grouped:raise ValueError('没有已完成的逐景水体SHP')
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    roi=None
    if roi_path is not None:
        print('[02] 读取/重投影ROI并导出ROI副本',flush=True)
        with fiona.open(roi_path) as src:
            if not src.crs:raise ValueError(f'ROI缺少CRS：{roi_path}')
            transform=Transformer.from_crs(src.crs,target,always_xy=True).transform
            roi=unary_union([transform_geometry(shape(f['geometry']),transform,interleaved=False,include_z=None)
                             for f in src if f['geometry']])
        if roi.is_empty or roi.geom_type not in ('Polygon','MultiPolygon'):
            raise ValueError('ROI必须包含非空面几何')
        # 输出重投影后的ROI副本；保留源文件，合并内部行政单元边界。
        with fiona.open(output_dir/'roi.shp','w',driver='ESRI Shapefile',
                        schema={'geometry':'Polygon','properties':{'id':'int'}},crs_wkt=target.to_wkt(),encoding='UTF-8') as dst:
            for i,geom in enumerate([roi] if roi.geom_type=='Polygon' else roi.geoms,1):
                dst.write({'geometry':mapping(geom),'properties':{'id':i}})
        if save_previews:
            with log.stage('02','生成ROI快视图'):
                save_quicklook(None,roi,output_dir/'roi.png','ROI',target,preview_size)
    for date_index,(date,records) in enumerate(grouped.items(),1):
        print(f'日期 {date_index}/{len(grouped)}：{date}；共{len(records)}景',flush=True)
        summary_file=output_dir/f'{date}_sources.json'
        required=[output_dir/f'{date}_water{suffix}' for suffix in ('.shp','.shx','.dbf','.prj')]
        if save_previews:required.append(output_dir/f'{date}_water.png')
        if RESUME and summary_file.exists() and all(p.exists() for p in required):
            previous=json.loads(summary_file.read_text(encoding='utf-8'))
            if previous.get('complete'):
                log.message(f'{date}：复用已完成的SHP和快视图','REUSED');continue
        # 先撤销本日期完成标记；中断后不能把部分写出的SHP当作成品。
        if summary_file.exists():summary_file.unlink()
        combined=merge_date(records,target,output_dir/'.merge_cache'/date,log)
        schema={'geometry':'Polygon','properties':{'date':'str:10','n_scenes':'int'}}
        with log.stage('06','写出合并SHP',date), show_activity(log,f'{date}写出SHP'), fiona.open(output_dir/f'{date}_water.shp','w',driver='ESRI Shapefile',schema=schema,
                        crs_wkt=target.to_wkt(),encoding='UTF-8') as dst:
            polygons=[combined] if combined.geom_type=='Polygon' else getattr(combined,'geoms',[])
            features=[]
            for polygon_index,geom in enumerate(polygons,1):
                if geom.geom_type=='Polygon' and not geom.is_empty:
                    features.append({'geometry':mapping(geom),'properties':{'date':date,'n_scenes':len(records)}})
                if len(features)>=WRITE_BATCH_SIZE:
                    dst.writerecords(features);features=[]
                    print(f'SHP写出：{polygon_index}/{len(polygons)}个面 ({polygon_index/max(len(polygons),1):.1%})',flush=True)
            if features:dst.writerecords(features)
        print(f'水体SHP已写出：{output_dir / (date+"_water.shp")}',flush=True)
        if save_previews:
            with log.stage('07','栅格化并保存日期快视图',date), show_activity(log,f'{date}生成快视图'):
                save_quicklook(combined,roi,output_dir/f'{date}_water.png',
                               f'{date} | {len(records)} scenes | water {combined.area/1e6:.3f} km2',target,preview_size)
        summary={'complete':True,'date':date,'scene_ids':[r['scene_id'] for r in records],'water_area_m2':combined.area,
                 'roi_source':str(roi_path) if roi_path is not None else None,'roi_role':'display_only_no_clipping',
                 'quicklook':f'{date}_water.png' if save_previews else None,
                 'method':'geometric_union','note':'仅对已有水体面求并集；未覆盖范围不代表非水体，不评定精度。'}
        pending=summary_file.with_suffix('.partial.json')
        pending.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8');pending.replace(summary_file)
        print(f'{date}：{len(records)}景，合并水体面积{combined.area/1e6:.3f}平方公里',flush=True)
    return output_dir


if __name__=='__main__':
    merge_results(INPUT_ROOT,OUTPUT_DIR,TARGET_CRS,ROI_PATH,SAVE_QUICKLOOK,PREVIEW_SIZE)
