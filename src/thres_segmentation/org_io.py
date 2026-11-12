"""ORG逐景生产的输入、空间范围、辅助缓存与矢量输出；不依赖训练目录。"""
import hashlib
import json
import math
import re
import shutil
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile
from zoneinfo import ZoneInfo

import fiona
import numpy as np
import rasterio
from rasterio.features import geometry_mask, geometry_window, shapes
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from rasterio.windows import Window
from pyproj import CRS, Transformer
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
from shapely import transform as transform_geometry
from scipy.ndimage import uniform_filter
from filelock import FileLock

from thres_segmentation.auxiliary_data import tile_specs, download_tile


def write_json(path, content):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_suffix(path.suffix+'.partial')
    pending.write_text(json.dumps(content,ensure_ascii=False,indent=2,default=str,allow_nan=False),encoding='utf-8')
    pending.replace(path)


def stamp(path):
    path=Path(path);stat=path.stat()
    return {'path':str(path.resolve()),'size':stat.st_size,'mtime_ns':stat.st_mtime_ns}


def signature(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()[:20]


def windows(src, size):
    for top in range(0,src.height,size):
        for left in range(0,src.width,size):
            yield Window(left,top,min(size,src.width-left),min(size,src.height-top))


def inventory(root, options):
    """只选ORG；同景ZIP与解压TIFF并存时优先唯一ZIP，排除SLC和下载临时文件。"""
    records={}
    for path in sorted(Path(root).rglob('*')):
        if not path.is_file() or path.suffix.lower() not in ('.zip','.tif','.tiff') or '-ORG-' not in path.name.upper():
            continue
        member=None
        if path.suffix.lower()=='.zip':
            with ZipFile(path) as archive:
                members=[n for n in archive.namelist() if Path(n).suffix.lower() in ('.tif','.tiff') and '-ORG-' in Path(n).name.upper()]
                if len(members)!=1:raise ValueError(f'ORG压缩包必须有唯一影像：{path}')
                member=members[0]
        identifier=Path(member or path.name).stem.upper()
        match=re.search(r'(BC[2-5])-.*?-ORG-.*?-(\d{8}T\d{6})-',identifier)
        if not match:raise ValueError(f'无法识别BC2–BC5景号/时间：{identifier}')
        time=datetime.strptime(match[2],'%Y%m%dT%H%M%S').replace(tzinfo=ZoneInfo(options['filename_timezone']))
        record={'scene_id':identifier,'path':str(path.resolve()),'member':member,
             'sensor':match[1],'acquisition_time':time.isoformat(),
             'date':time.astimezone(ZoneInfo(options['date_timezone'])).date().isoformat(),'source_stamp':stamp(path)}
        records.setdefault(identifier, []).append(record)
    result=[]
    for identifier, candidates in records.items():
        archives=[r for r in candidates if r['member'] is not None]
        # 使用原始压缩包及其中XML，与此前生产来源保持一致；不删除解压文件。
        # 这是明确的输入优先级，不以同名推断文件内容完全相同。
        if len(archives)==1:
            selected=archives[0]
            if len(candidates)>1:
                print(f'同景输入去重：{identifier}；优先ZIP，忽略{len(candidates)-1}个解压TIFF：{selected["path"]}',flush=True)
        elif not archives and len(candidates)==1:
            selected=candidates[0]
        else:
            paths='；'.join(r['path'] for r in (archives or candidates))
            raise ValueError(f'同景存在多个版本，请保留一个输入：{paths}')
        result.append(selected)
    if not result:raise ValueError(f'没有已命名完成的ORG文件：{root}')
    return sorted(result,key=lambda r:(r['date'],r['scene_id']))


def read_aoi(path, target_crs):
    """合并行政区内部单元；目标投影必须以米为单位，便于面积和坡度计算。"""
    target=CRS.from_user_input(target_crs)
    if not target.is_projected or any(abs(a.unit_conversion_factor-1)>1e-9 for a in target.axis_info[:2]):
        raise ValueError('output_crs需使用米制投影坐标系')
    with fiona.open(path) as source:
        if not source.crs:raise ValueError('行政边界缺少CRS')
        geoms=[shape(f['geometry']) for f in source if f['geometry']]
        if not geoms or any(g.is_empty or not g.is_valid for g in geoms):raise ValueError('行政边界含空或无效几何')
        aoi=unary_union(geoms)
        aoi=transform_geometry(aoi,Transformer.from_crs(source.crs,target,always_xy=True).transform,interleaved=False,include_z=None)
    return aoi,target


def profile_for(bounds, crs, resolution):
    left,bottom,right,top=bounds
    left=math.floor(left/resolution)*resolution;top=math.ceil(top/resolution)*resolution
    return dict(driver='GTiff',crs=crs,transform=from_origin(left,top,resolution,resolution),
         width=math.ceil((right-left)/resolution),height=math.ceil((top-bottom)/resolution),
         count=1,dtype='float32',nodata=float('nan'),tiled=True,blockxsize=256,blockysize=256,
         compress='deflate',BIGTIFF='IF_SAFER')


def align_sources(paths, profile, destination, nearest=True, block_size=1024):
    """逐窗重投影；分类用最近邻，高程用双线性，不载入整幅辅助栅格。"""
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    pending=destination.with_suffix('.partial.tif')
    with ExitStack() as stack:
        readers=[]
        for path in paths:
            src=stack.enter_context(rasterio.open(path))
            readers.append(stack.enter_context(WarpedVRT(src,crs=profile['crs'],transform=profile['transform'],
                width=profile['width'],height=profile['height'],dtype='float32',nodata=float('nan'),
                resampling=Resampling.nearest if nearest else Resampling.bilinear)))
        dst=stack.enter_context(rasterio.open(pending,'w',**profile))
        for win in windows(dst,block_size):
            result=np.full((int(win.height),int(win.width)),np.nan,np.float32)
            for src in readers:
                data=src.read(1,window=win,masked=True)
                good=~np.ma.getmaskarray(data)&np.isfinite(data.data)&~np.isfinite(result)
                result[good]=data.data[good]
            dst.write(result,1,window=win)
    pending.replace(destination)
    return destination


def prepare_auxiliary(config, aoi, crs):
    """同一缓存中的下载和派生产品串行发布，多个任务不会同时下载同一批瓦片。"""
    cache=config['paths']['AUXILIARY_CACHE_DIR'];cache.mkdir(parents=True,exist_ok=True)
    with FileLock(str(cache/'org_auxiliary.lock')):
        return _prepare_auxiliary(config,aoi,crs)


def _prepare_auxiliary(config, aoi, crs):
    """按行政区一次下载公共瓦片；各景共享缓存，更新参数时只重建派生网格。"""
    paths=config['paths'];options=config['org']['auxiliary'];post=config['org']['postprocess']
    bounds=aoi.buffer(options['buffer_m']).bounds
    wgs=transform_bounds(crs,'EPSG:4326',*bounds,densify_pts=21)
    cache=paths['AUXILIARY_CACHE_DIR'];cache.mkdir(parents=True,exist_ok=True)
    # 本地辅助文件均可省略；缺省和YAML的null都表示按下载开关处理。
    landcover=paths.get('LANDCOVER_PATH');elevation=paths.get('ELEVATION_PATH')
    history=paths.get('ORG_HISTORY_PATH')
    products={}
    needed={'worldcover':post['landcover_enabled'] or (post['history_enabled'] and history is None),
            'copernicus':post['terrain_enabled']}
    for product,required in needed.items():
        if not required:continue
        local=landcover if product=='worldcover' else elevation
        if local is not None:
            tiles=[local]
        else:
            specs=tile_specs(product,wgs,options['worldcover_year'])
            if len(specs)>options['max_tiles']:raise ValueError('辅助下载瓦片数量超过配置上限')
            if options['download_missing']:
                tiles=[download_tile(s['url'],cache/'tiles'/product/s['filename'],timeout=options['timeout'],
                           retries=options['retries']) for s in specs]
            else:
                # 禁止新下载时仍允许复用已下载瓦片，供独立参数对照使用。
                tiles=[cache/'tiles'/product/s['filename'] for s in specs]
                if any(not p.is_file() or p.stat().st_size==0 for p in tiles):
                    raise FileNotFoundError(f'缺少{product}缓存瓦片且禁止下载')
        resolution=options['landcover_resolution_m'] if product=='worldcover' else options['dem_resolution_m']
        profile=profile_for(bounds,crs,resolution)
        if profile['width']*profile['height']>options['max_mosaic_pixels']:raise ValueError('辅助网格超过像元上限')
        key=signature({'sources':[stamp(p) for p in tiles],'profile':profile,'algorithm':'org_aux_v1'})
        destination=cache/'aligned'/f'{product}_{key}.tif'
        if not destination.exists():align_sources(tiles,profile,destination,nearest=product=='worldcover')
        products[product]=destination
    if 'copernicus' in products:
        dem=products['copernicus']
        slope_key=signature({'dem':stamp(dem),'elevation_scale':options['elevation_scale'],'algorithm':'gradient_v1'})
        slope=dem.with_name('slope_'+slope_key+'.tif')
        if not slope.exists():
            pending=slope.with_suffix('.partial.tif')
            with rasterio.open(dem) as src,rasterio.open(pending,'w',**src.profile) as dst:
                for win in windows(src,config['processing']['block_size']):
                    expanded=Window(win.col_off-1,win.row_off-1,win.width+2,win.height+2)
                    z=src.read(1,window=expanded,boundless=True,fill_value=np.nan)*options['elevation_scale']
                    dy,dx=np.gradient(z,abs(src.transform.e),abs(src.transform.a))
                    value=np.degrees(np.arctan(np.hypot(dx,dy)))
                    # 高程缺测邻域的坡度保持未知，不用零高程制造陡坡。
                    good=uniform_filter(np.isfinite(z).astype('float32'),3,mode='constant')>.999
                    dst.write(np.where(good,value,np.nan)[1:-1,1:-1].astype('float32'),1,window=win)
            pending.replace(slope)
        products['slope']=slope
    if post['history_enabled'] and history is not None:
        products['history']=history
    return products


def local_source(record, cache):
    """ZIP只流式提取目标TIFF，完整读至末尾检查成员CRC；同景缓存复用。"""
    path=Path(record['path'])
    if record['member'] is None:return path
    folder=Path(cache)/'extracted'/signature(record['source_stamp']);folder.mkdir(parents=True,exist_ok=True)
    target=folder/Path(record['member']).name
    if target.exists():return target
    pending=target.with_suffix('.partial')
    with ZipFile(path) as archive,archive.open(record['member']) as src,pending.open('wb') as dst:
        shutil.copyfileobj(src,dst,8*1024*1024)
    pending.replace(target)
    return target


def crop_org(source, destination, aoi, aoi_crs, band, buffer_m, nodata_value, block_size):
    """裁剪到AOI外扩包围框；保留原SAR网格，滤波后再严格施加行政边界。"""
    with rasterio.open(source) as src:
        if not src.crs:raise ValueError(f'ORG缺少地理坐标系：{source}')
        if band not in src.indexes:raise ValueError(f'影像不存在波段{band}：{source}')
        if np.dtype(src.dtypes[band-1]).kind=='c':raise ValueError('ORG应为实数幅度，不能输入复数SLC')
        region=transform_geometry(aoi.buffer(buffer_m),Transformer.from_crs(aoi_crs,src.crs,always_xy=True).transform,interleaved=False,include_z=None)
        try:win=geometry_window(src,[mapping(region)])
        except rasterio.errors.WindowError:return None
        profile=src.profile.copy();profile.update(driver='GTiff',count=1,dtype='float32',nodata=np.nan,
            width=int(win.width),height=int(win.height),transform=src.window_transform(win),
            tiled=True,blockxsize=256,blockysize=256,compress='deflate',BIGTIFF='IF_SAFER')
        pending=Path(destination).with_suffix('.partial.tif')
        with rasterio.open(pending,'w',**profile) as dst:
            for part in windows(dst,block_size):
                source_win=Window(win.col_off+part.col_off,win.row_off+part.row_off,part.width,part.height)
                values=src.read(band,window=source_win,masked=True,out_dtype='float32')
                valid=~np.ma.getmaskarray(values)&np.isfinite(values.data)
                if nodata_value is not None:valid &= values.data!=nodata_value
                if np.any(values.data[valid]<0):raise ValueError('幅度存在有效负值，请核对NoData/输入域')
                dst.write(np.where(valid,values.data,np.nan),1,window=part)
        pending.replace(destination)
    return Path(destination)


def vectorize(mask_path, destination, scene_id, date, sensor, clip_geometry=None):
    """直接流式矢量化最终0/1/255栅格，不拼接分块多边形、不平滑原岸线。"""
    destination=Path(destination)
    schema={'geometry':'Polygon','properties':{'scene_id':'str:120','date':'str:10','sensor':'str:8'}}
    with rasterio.open(mask_path) as src,fiona.open(destination,'w',driver='ESRI Shapefile',
                schema=schema,crs_wkt=src.crs.to_wkt(),encoding='UTF-8') as dst:
        for geom,value in shapes(rasterio.band(src,1),connectivity=4,transform=src.transform):
            if value!=1:continue
            polygon=shape(geom)
            if clip_geometry is not None:polygon=polygon.intersection(clip_geometry)
            parts=[polygon] if polygon.geom_type=='Polygon' else getattr(polygon,'geoms',[])
            for part in parts:
                if part.geom_type=='Polygon' and not part.is_empty:
                    dst.write({'geometry':mapping(part),'properties':{'scene_id':scene_id,'date':date,'sensor':sensor}})
    return destination
