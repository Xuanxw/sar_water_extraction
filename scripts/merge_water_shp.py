"""按日期拼接水体SHP；默认追加要素，可选几何并集消除重叠。"""
import json
import sys
from pathlib import Path
from threading import Event, Thread
from time import perf_counter
from contextlib import contextmanager
import fiona
from fiona.transform import transform_geom
from pyproj import CRS, Transformer
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
from shapely import transform as transform_geometry, from_wkb, to_wkb, get_parts, geos_version

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from thres_segmentation.run_log import RunLog

INPUT_ROOT=Path(r'E:\InSAR\水体提取\北京密云区\outputs\thres_seg_20261112')
OUTPUT_DIR=Path(r'E:\InSAR\水体提取\北京密云区\outputs\thres_seg_20261112\merged')
TARGET_CRS='EPSG:32650'
ROI_PATH=Path(r'E:\InSAR\水体提取\北京密云区\密云区行政划分.shp')  # 可设None。
SAVE_QUICKLOOK=True
PREVIEW_SIZE=1600
WRITE_BATCH_SIZE=5000
DISSOLVE=False  # False：快速追加，保留重叠；True：几何并集，消除重叠和内部边界。
MERGE_BATCH_SIZE=2000  # 仅并集模式使用：磁盘分批归并，完成节点可以续跑。
HEARTBEAT_SECONDS=10  # GEOS计算期间定时提示耗时。
UNION_METHOD='disjoint_subset'  # 支持时采用空间子集并集，否则回退unary_union。
CONSOLE_COLOR=True
RESUME=True  # 复用本脚本已完成的日期；更换输入/参数后另设输出目录或设False。


def merge_geometries(geometries):
    """拆开MultiPolygon，按真实相交关系归组；仍为精确矢量并集，不简化边界。"""
    import shapely
    if UNION_METHOD=='disjoint_subset' and geos_version>=(3,12,0) and hasattr(shapely,'disjoint_subset_union_all'):
        # 必须先拆分大MultiPolygon，否则它包含的离散小面无法充分参与空间分组。
        return shapely.disjoint_subset_union_all(get_parts(geometries))
    return unary_union(geometries)


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
        result=merge_geometries(geometries)
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


def save_quicklook(water_path,roi_path,destination,title,target,max_side):
    """流式栅格化仅用于显示，避免将所有水体组成一个巨大几何。"""
    import numpy as np
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.patches import Patch
    bounds=[]
    for path in (water_path,roi_path):
        if path is not None:
            with fiona.open(path) as src:
                if len(src):bounds.append(src.bounds)
    if bounds:
        left,bottom=min(b[0] for b in bounds),min(b[1] for b in bounds)
        right,top=max(b[2] for b in bounds),max(b[3] for b in bounds)
    else:left,bottom,right,top=0,0,1,1
    margin=max(right-left,top-bottom)*.04
    left-=margin;right+=margin;bottom-=margin;top+=margin
    scale=max_side/max(right-left,top-bottom)
    size=(max(1,round((top-bottom)*scale)),max(1,round((right-left)*scale)))
    grid=from_bounds(left,bottom,right,top,size[1],size[0])
    rgb=np.full((*size,3),255,np.uint8)
    mask=np.zeros(size,np.uint8)
    if water_path is not None:
        with fiona.open(water_path) as src:
            batch=[];total=len(src)
            for index,feature in enumerate(src,1):
                if feature['geometry']:batch.append((feature['geometry'],1))
                if index%WRITE_BATCH_SIZE==0 or index==total:
                    if batch:
                        layer=rasterize(batch,out_shape=size,transform=grid,dtype='uint8')
                        np.maximum(mask,layer,out=mask);batch=[]
                    print(f'快视图绘制：{index}/{total}面 ({index/max(total,1):.1%})',flush=True)
        rgb[mask==1]=(35,115,195)
    if roi_path is not None:
        lines=[]
        with fiona.open(roi_path) as src:
            for feature in src:
                g=feature['geometry']
                if not g:continue
                polygons=[g['coordinates']] if g['type']=='Polygon' else g['coordinates']
                for polygon in polygons:
                    for ring in polygon:lines.append(({'type':'LineString','coordinates':ring},1))
        if lines:
            border=rasterize(lines,out_shape=size,transform=grid,dtype='uint8')
            rgb[border==1]=(210,65,45)
    fig=Figure(figsize=(9,8));FigureCanvasAgg(fig)
    ax=fig.subplots();ax.imshow(rgb,extent=(left,right,bottom,top),origin='upper')
    ax.set_aspect('equal');ax.set_title(title)
    ax.set_xlabel(f'Easting (m) | {target.to_string()}');ax.set_ylabel('Northing (m)')
    legend=[]
    if water_path is not None:legend.append(Patch(facecolor='#2373c3',label='Water'))
    if roi_path is not None:legend.append(Patch(facecolor='none',edgecolor='#d2412d',label='ROI boundary'))
    if legend:ax.legend(handles=legend,loc='upper right')
    fig.text(.5,.01,'White does not establish non-water or SAR coverage.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.03,1,1));fig.savefig(destination,dpi=150);fig.clear()


def merge_results(input_root,output_dir,target_crs,roi_path=None,save_previews=True,preview_size=1600,dissolve=False):
    """默认追加；并集使用独立文件名和分批缓存，两种结果可以共存。"""
    output_dir=Path(output_dir);target=CRS.from_user_input(target_crs)
    method='geometric_union' if dissolve else 'feature_append'
    mode='几何并集（消除重叠）' if dissolve else '追加要素（保留重叠）'
    log=RunLog(output_dir,CONSOLE_COLOR)
    with log.capture(),log.stage('00',f'按日期合并SHP：{mode}'):
        print(f'输入：{input_root}；输出：{output_dir}；模式：{mode}',flush=True)
        grouped={}
        with log.stage('01','扫描已完成的逐景产品'):
            for path in sorted(Path(input_root).glob('*/*/scene.json')):
                record=json.loads(path.read_text(encoding='utf-8'))
                if record['status']!='complete':continue
                water=path.parent/'water.shp'
                if not water.is_file():raise FileNotFoundError(water)
                record['water_shp']=str(water)
                grouped.setdefault(record['date'],[]).append(record)
            print(f'{len(grouped)}日期，{sum(map(len,grouped.values()))}景',flush=True)
        if not grouped:raise ValueError('没有已完成的逐景水体SHP')
        roi_output=None
        if roi_path is not None:
            roi_output=output_dir/'roi.shp'
            with log.stage('02','导出ROI及快视图'),fiona.open(roi_path) as src:
                if not src.crs:raise ValueError(f'ROI缺少CRS：{roi_path}')
                same=CRS.from_user_input(src.crs)==target
                with fiona.open(roi_output,'w',driver='ESRI Shapefile',schema=src.schema,crs_wkt=target.to_wkt(),encoding='UTF-8') as dst:
                    for f in src:
                        if not f['geometry']:continue
                        geom=f['geometry'] if same else transform_geom(src.crs,target.to_wkt(),f['geometry'])
                        dst.write({'geometry':geom,'properties':dict(f['properties'])})
                if save_previews:save_quicklook(None,roi_output,output_dir/'roi.png','ROI',target,preview_size)
        schema={'geometry':'Polygon','properties':({'date':'str:10'} if dissolve else {'scene_id':'str:120','date':'str:10','sensor':'str:8'})}
        for date_index,(date,records) in enumerate(grouped.items(),1):
            print(f'日期 {date_index}/{len(grouped)}：{date}，{len(records)}景',flush=True)
            suffix='_union' if dissolve else ''
            summary_file=output_dir/f'{date}{suffix}_sources.json'
            water_output=output_dir/f'{date}_water{suffix}.shp'
            preview_output=water_output.with_suffix('.png')
            required=[water_output.with_suffix(s) for s in ('.shp','.shx','.dbf','.prj')]
            if save_previews:required.append(preview_output)
            if RESUME and summary_file.exists() and all(p.exists() for p in required):
                previous=json.loads(summary_file.read_text(encoding='utf-8'))
                if previous.get('complete') and previous.get('method')==method:
                    log.message(f'{date}：复用已完成拼接','REUSED');continue
            if summary_file.exists():summary_file.unlink()
            pending=water_output.with_suffix('.partial.shp')
            count=0
            if dissolve:
                # 并集后的面可能来自多景，逐面不再填写单一scene_id；来源写入JSON。
                water=merge_date(records,target,output_dir/'.merge_cache'/date,log)
                with log.stage('05','写出几何并集SHP',date),fiona.open(pending,'w',driver='ESRI Shapefile',schema=schema,crs_wkt=target.to_wkt(),encoding='UTF-8') as dst:
                    batch=[]
                    for geom in get_parts(water):
                        if geom.is_empty:continue
                        batch.append({'geometry':mapping(geom),'properties':{'date':date}})
                        if len(batch)>=WRITE_BATCH_SIZE:
                            dst.writerecords(batch);count+=len(batch);batch=[]
                            print(f'并集SHP已写出{count}面',flush=True)
                    if batch:dst.writerecords(batch);count+=len(batch)
                del water
            else:
                with log.stage('03','读取并批量写出SHP',date),fiona.open(pending,'w',driver='ESRI Shapefile',schema=schema,crs_wkt=target.to_wkt(),encoding='UTF-8') as dst:
                    for scene_index,record in enumerate(records,1):
                        with fiona.open(record['water_shp']) as src:
                            if not src.crs:raise ValueError(f"SHP缺少CRS：{record['water_shp']}")
                            same=CRS.from_user_input(src.crs)==target
                            total=len(src);batch=[]
                            print(f'景{scene_index}/{len(records)}：{record["scene_id"]}；{total}面；同CRS直接复制={same}',flush=True)
                            for index,feature in enumerate(src,1):
                                if feature['geometry']:
                                    geom=feature['geometry'] if same else transform_geom(src.crs,target.to_wkt(),feature['geometry'])
                                    batch.append({'geometry':geom,'properties':{'scene_id':record['scene_id'],'date':date,'sensor':record.get('sensor','')}})
                                if index%WRITE_BATCH_SIZE==0 or index==total:
                                    dst.writerecords(batch);count+=len(batch);batch=[]
                                    print(f'景{scene_index}/{len(records)} 已追加{index}/{total}面 ({index/max(total,1):.1%})；本日期累计{count}面',flush=True)
            for suffix in ('.shp','.shx','.dbf','.prj','.cpg'):
                source=pending.with_suffix(suffix)
                if source.exists():source.replace(water_output.with_suffix(suffix))
            print(f'SHP已完成：{water_output}；{count}面；{mode}',flush=True)
            if save_previews:
                with log.stage('04','生成日期快视图',date):
                    save_quicklook(water_output,roi_output,preview_output,f'{date} | {method} | {len(records)} scenes | {count} features',target,preview_size)
            summary={'complete':True,'method':method,'date':date,'scene_ids':[r['scene_id'] for r in records],
                     'feature_count':count,'roi_source':str(roi_path) if roi_path else None,'roi_role':'display_only_no_clipping',
                     'quicklook':preview_output.name if save_previews else None,
                     'note':('几何并集，重叠已消除；逐面不再保留单景来源，全部来源见scene_ids。' if dissolve else '仅拼接，不做溶解/去重；重叠面保留，不能直接累加面面积作为去重水体面积。')}
            temp=summary_file.with_suffix('.partial.json')
            temp.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(summary_file)
        return output_dir


if __name__=='__main__':
    merge_results(INPUT_ROOT,OUTPUT_DIR,TARGET_CRS,ROI_PATH,SAVE_QUICKLOOK,PREVIEW_SIZE,dissolve=DISSOLVE)