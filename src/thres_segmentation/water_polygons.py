"""沿用GUI的水体面成图步骤：填孔、开闭、重施排除、8连通筛选、面化和平滑。"""
import json
from pathlib import Path
from time import perf_counter
from contextlib import ExitStack
import numpy as np
import fiona
import rasterio
from rasterio.windows import Window
from rasterio.features import shapes
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from scipy import ndimage
from shapely.geometry import shape, mapping, Polygon
from shapely import make_valid, transform as transform_geometry
from pyproj import Transformer

DEFAULTS=dict(hole_max_pixels=200,morph_kernel_size=3,min_water_pixels=200,smooth_iterations=2)
BLOCK_ROWS=256


def component_sizes(ids,count):
    """分行统计整景连通域，不把整幅ID数组转换成int64。"""
    sizes=np.zeros(count+1,dtype=np.int64)
    for y in range(0,ids.shape[0],BLOCK_ROWS):
        sizes+=np.bincount(ids[y:y+BLOCK_ROWS].ravel(),minlength=count+1)
    return sizes


def clean_mask(label,excluded,options):
    """整景8连通避免分块边界切断水体；形态学按带halo的行块计算。"""
    height,width=label.shape
    structure=np.ones((3,3),bool)
    report={}
    if options['hole_max_pixels']>0:
        print('  [面1/4] 整景识别内部小孔洞',flush=True)
        ids,count=ndimage.label(label==0,structure=structure)
        sizes=component_sizes(ids,count)
        fill=(sizes<=options['hole_max_pixels']);fill[0]=False
        fill[np.unique(np.concatenate((ids[0],ids[-1],ids[:,0],ids[:,-1])))]=False
        for y in range(0,height,BLOCK_ROWS):
            a=max(0,y-1);b=min(height,y+BLOCK_ROWS+1)
            protected=(label[a:b]==255)|excluded[a:b]
            near=ndimage.binary_dilation(protected,structure=structure)[y-a:min(y+BLOCK_ROWS,height)-a]
            fill[np.unique(ids[y:y+BLOCK_ROWS][near])]=False
        report['filled_holes']=int(fill.sum());report['filled_pixels']=int(sizes[fill].sum())
        for y in range(0,height,BLOCK_ROWS):label[y:y+BLOCK_ROWS][fill[ids[y:y+BLOCK_ROWS]]]=1
        del ids,sizes,fill
    kernel=int(options['morph_kernel_size'])
    if kernel>1:
        print('  [面2/4] 开闭运算并恢复辅助排除区',flush=True)
        if kernel%2==0:raise ValueError('morph_kernel_size必须为奇数或0/1')
        radius=kernel//2;yy,xx=np.ogrid[-radius:radius+1,-radius:radius+1]
        kernel_mask=(xx*xx+yy*yy)<=radius*radius
        halo=4*radius
        original=label.copy()
        for y in range(0,height,BLOCK_ROWS):
            end=min(height,y+BLOCK_ROWS);a=max(0,y-halo);b=min(height,end+halo)
            water=original[a:b]==1
            water=ndimage.binary_opening(water,structure=kernel_mask)
            water=ndimage.binary_closing(water,structure=kernel_mask)
            label[y:end]=np.where(original[y:end]==255,255,water[y-a:end-a]).astype('uint8')
        del original
    for y in range(0,height,BLOCK_ROWS):
        block=label[y:y+BLOCK_ROWS];block[excluded[y:y+BLOCK_ROWS]&(block!=255)]=0
    print('  [面3/4] 整景8连通，按最小水体面积筛选',flush=True)
    ids,count=ndimage.label(label==1,structure=structure)
    sizes=component_sizes(ids,count);keep=sizes>=max(1,options['min_water_pixels']);keep[0]=False
    report.update(components_before=int(count),components_kept=int(keep.sum()),components_removed=int(count-keep.sum()),
                  removed_small_pixels=int(sizes[1:][~keep[1:]].sum()))
    for y in range(0,height,BLOCK_ROWS):
        block=label[y:y+BLOCK_ROWS];block[(block==1)&~keep[ids[y:y+BLOCK_ROWS]]]=0
    return label,report


def smooth_polygon(polygon,tolerance,iterations):
    """与GUI相同：半像元拓扑简化，再做Chaikin拐角切割；无效则回退。"""
    simple=polygon.simplify(tolerance,preserve_topology=True)
    if simple.is_empty or simple.geom_type!='Polygon':simple=polygon
    def ring(coords):
        a=np.asarray(coords,dtype='float64')[:-1,:2]
        for _ in range(iterations):
            b=np.roll(a,-1,axis=0);out=np.empty((2*len(a),2))
            out[0::2]=.75*a+.25*b;out[1::2]=.25*a+.75*b;a=out
        return a
    result=Polygon(ring(simple.exterior.coords),[ring(h.coords) for h in simple.interiors])
    if result.is_empty or not result.is_valid:return simple,True
    return result,False


def polygon_parts(geom):
    if geom.geom_type=='Polygon':
        if not geom.is_empty:yield geom
    elif hasattr(geom,'geoms'):
        for part in geom.geoms:yield from polygon_parts(part)


def build_water_polygons(mask_path,destination,*,options=None,cover_path=None,slope_path=None,terrain_path=None,
                         postprocess=None,clip_geometry=None,clip_crs=None,overwrite=False):
    """从最终阈值分类图生成清理后的水体面；原分类成果不改，新增结果独立记录。"""
    mask_path=Path(mask_path);destination=Path(destination)
    settings={**DEFAULTS,**(options or {})};post=postprocess or {}
    report_path=destination.with_suffix('.json');raster_path=destination.with_suffix('.tif')
    outputs=[destination.with_suffix(s) for s in ('.shp','.shx','.dbf','.prj','.cpg','.tif')]
    if not overwrite and report_path.exists() and all(p.exists() for p in outputs):
        info=json.loads(report_path.read_text(encoding='utf-8'))
        if info.get('complete') and info.get('algorithm')=='gui_polygon_cleanup_v1' and info.get('options')==settings:
            print(f'  复用已完成水体面：{destination}',flush=True);return destination
    started=perf_counter()
    if report_path.exists():report_path.unlink()
    with ExitStack() as stack:
        src=stack.enter_context(rasterio.open(mask_path));profile=src.profile.copy()
        if clip_geometry is not None and clip_crs is not None:
            clip_geometry=transform_geometry(clip_geometry,Transformer.from_crs(clip_crs,src.crs,always_xy=True).transform,interleaved=False)
        label=src.read(1);original_count=int(np.count_nonzero(label==1))
        excluded=np.zeros(src.shape,bool)
        # DEM缺测处不排除；土地覆盖只应用配置启用的类别。
        for path,kind in ((cover_path,'cover'),(terrain_path,'terrain') if terrain_path else (slope_path,'slope')):
            enabled=post.get('landcover_enabled',False) if kind=='cover' else post.get('terrain_enabled',False)
            if not enabled:continue
            if path is None:raise ValueError(f'清理水体面缺少启用的{kind}辅助数据，不能安全填孔')
            base=stack.enter_context(rasterio.open(path))
            if (base.shape,base.crs,base.transform)==(src.shape,src.crs,src.transform):
                reader=base  # 同网格直接读取，避免二次重采样改变阈值边缘。
            else:
                reader=stack.enter_context(WarpedVRT(base,crs=src.crs,transform=src.transform,width=src.width,height=src.height,
                       dtype='float32',nodata=np.nan,resampling=Resampling.nearest if kind in ('cover','terrain') else Resampling.bilinear))
            for y in range(0,src.height,BLOCK_ROWS):
                end=min(src.height,y+BLOCK_ROWS);win=Window(0,y,src.width,end-y)
                values=reader.read(1,window=win,masked=True)
                known=~np.ma.getmaskarray(values)&np.isfinite(values.data)
                bad=(np.isin(values.data,post.get('excluded_cover_classes',[50])) if kind=='cover' else
                     values.data==1 if kind=='terrain' else values.data>post.get('max_slope_deg',15.))
                excluded[y:end]|=known&bad
        nodata_count=int(np.count_nonzero(label==255))
        label,report=clean_mask(label,excluded,settings)
        del excluded
        profile.update(driver='GTiff',dtype='uint8',count=1,nodata=255,compress='lzw',BIGTIFF='IF_SAFER')
        pending_raster=raster_path.with_suffix('.partial.tif')
        with rasterio.open(pending_raster,'w',**profile) as dst:dst.write(label,1)
        final_count=int(np.count_nonzero(label==1))
        assert int(np.count_nonzero(label==255))==nodata_count
        print(f'  [面4/4] 栅格面化及边界平滑；连通域{report["components_before"]}→{report["components_kept"]}',flush=True)
        pending=destination.with_suffix('.partial.shp')
        meta_path=mask_path.parent/'scene.json'
        meta=json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else {}
        schema={'geometry':'Polygon','properties':{'water':'int','scene_id':'str:120','date':'str:10','sensor':'str:8'}}
        count=fallback=0
        tolerance=.25*(np.hypot(src.transform.a,src.transform.d)+np.hypot(src.transform.b,src.transform.e))
        with fiona.open(pending,'w',driver='ESRI Shapefile',schema=schema,crs_wkt=src.crs.to_wkt(),encoding='UTF-8') as dst:
            batch=[]
            for geom,value in shapes(label,mask=label==1,connectivity=8,transform=src.transform):
                polygon=shape(geom)
                # 对角相接的8连通轮廓可能自接触；修复为可编辑面，保留所有面部分。
                if not polygon.is_valid:polygon=make_valid(polygon)
                for part in polygon_parts(polygon):
                    if settings['smooth_iterations']>0:
                        part,failed=smooth_polygon(part,tolerance,int(settings['smooth_iterations']));fallback+=int(failed)
                    if clip_geometry is not None:part=part.intersection(clip_geometry)
                    for piece in polygon_parts(part):
                        batch.append({'geometry':mapping(piece),'properties':{'water':1,'scene_id':meta.get('scene_id',mask_path.parent.name),
                                    'date':meta.get('date',mask_path.parent.parent.name),'sensor':meta.get('sensor',mask_path.parent.name.split('-')[0])}})
                        count+=1
                    if len(batch)>=500:
                        dst.writerecords(batch);batch=[];print(f'    已输出{count}个水体面',flush=True)
            if batch:dst.writerecords(batch)
    for suffix in ('.shx','.dbf','.prj','.cpg','.shp'):pending.with_suffix(suffix).replace(destination.with_suffix(suffix))
    pending_raster.replace(raster_path)
    report.update(complete=True,algorithm='gui_polygon_cleanup_v1',options=settings,source=str(mask_path),
                  original_water_pixels=original_count,cleaned_water_pixels=final_count,unknown_pixels=nodata_count,
                  polygon_count=count,smoothing_fallbacks=fallback,elapsed_seconds=perf_counter()-started,
                  note='200像素规则会删除小水体；平滑面与清理后栅格边界略有差别。原quality.json仍描述原water.tif。')
    temp=report_path.with_suffix('.partial.json');temp.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(report_path)
    print(f'  完成：{count}面，耗时{report["elapsed_seconds"]:.1f}秒；{destination}',flush=True)
    return destination
