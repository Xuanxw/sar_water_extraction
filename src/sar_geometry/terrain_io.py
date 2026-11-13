"""DEM原生网格上的名义几何实验：分块I/O；尚不生成整景地形可见性。"""
from pathlib import Path
import hashlib
import json
from time import perf_counter
import numpy as np
import rasterio
from rasterio.windows import Window
from pyproj import CRS, Transformer
from .core import solve_points, STATUS_NAMES

ALGORITHM_VERSION = 'nominal_hermite_rd_0.1'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    """小型完成记录原子替换；中断时不留下伪完整JSON。"""
    path = Path(path)
    tmp = path.with_suffix('.partial.json')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(path)


def surface_block(source, window, transformer):
    """在同一DEM网格上读一像元邻域；中心差分，边缘/缺邻点的法向保持未知。"""
    w = Window(window.col_off-1,window.row_off-1,window.width+2,window.height+2)
    h = source.read(1,window=w,boundless=True,masked=True,out_dtype='float64').filled(np.nan)
    rr,cc = np.mgrid[0:h.shape[0],0:h.shape[1]]
    cc = cc+w.col_off+0.5; rr = rr+w.row_off+0.5
    a = source.transform
    x,y = a.a*cc+a.b*rr+a.c, a.d*cc+a.e*rr+a.f
    xyz = np.stack(transformer.transform(x,y,h),axis=-1)
    center = xyz[1:-1,1:-1]
    dx = xyz[1:-1,2:]-xyz[1:-1,:-2]
    dy = xyz[2:,1:-1]-xyz[:-2,1:-1]
    normal = np.cross(dx,dy)
    # 南北向行序不同可能使叉积向内；统一朝向地球外部。
    normal *= np.where(np.sum(normal*center,axis=-1)<0,-1,1)[...,None]
    length = np.linalg.norm(normal,axis=-1)
    normal = np.divide(normal,length[...,None],out=np.full_like(normal,np.nan),where=length[...,None]>0)
    valid = np.isfinite(center).all(axis=-1)
    center[~valid] = np.nan
    return center,normal


def run_nominal(config, metadata, model, log=print):
    """可独立续跑整景名义LUT；失败行块重算，尚未接入生产水体排除。"""
    paths,options = config['paths'],config['execution']
    dem,output,exe = Path(paths['DEM_PATH']),Path(paths['OUTPUT_DIR']),Path(paths['CORE_PATH'])
    if config['model']['height_datum']!='WGS84_ellipsoid':
        raise ValueError('本阶段需要已转换为WGS84椭球高的DEM；不把正高标签直接改成椭球高')
    output.mkdir(parents=True,exist_ok=True)
    cache = output/'data_cache';cache.mkdir(exist_ok=True)
    # 一个阶段状态记录；身份只取内容和科学参数，不包含源/输出绝对路径。
    key_data = dict(algorithm=ALGORITHM_VERSION,xml=hashlib.sha256(metadata['xml_bytes']).hexdigest(),
                    dem=sha256(dem),core=sha256(exe),model=config['model'])
    key = hashlib.sha256(json.dumps(key_data,sort_keys=True).encode()).hexdigest()
    state_path = cache/'progress.json'
    state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {}
    names = ('lookup','geometry_validity','radar_coverage','geometry_status','local_incidence','doppler_residual')
    completed = output/'geometry_quality.json'
    if options['resume'] and state.get('key')==key and state.get('complete') and completed.exists() and all((output/f'{n}.tif').is_file() for n in names):
        log('[03] 复用完整名义几何成果');return json.loads(completed.read_text(encoding='utf-8'))
    start = perf_counter()
    with rasterio.open(dem) as src:
        if src.crs is None:
            raise ValueError('DEM缺少水平坐标系')
        crs = CRS.from_user_input(src.crs)
        if crs.geodetic_crs is None or crs.geodetic_crs.to_epsg()!=4326:
            raise ValueError('首版只支持WGS84水平地理/投影坐标系')
        if src.count!=1 or src.width<3 or src.height<3:
            raise ValueError('DEM应为单波段且至少3×3像元')
        # 只对显式WGS84椭球高执行坐标转换，不启用隐式垂直基准变换。
        transformer = Transformer.from_crs(crs.to_3d(),4978,always_xy=True)
        block_rows = int(options['block_rows'])
        if block_rows<1:
            raise ValueError('block_rows必须为正')
        partials = {n:cache/f'{n}.partial.tif' for n in names}
        continuing = (options['resume'] and state.get('key')==key and not state.get('complete')
                      and state.get('block_rows')==block_rows and all(p.is_file() for p in partials.values()))
        if not continuing:
            state = dict(key=key,complete=False,next_row=0,block_rows=block_rows,counts={},residual_abs_max_hz=None)
            profile = dict(driver='GTiff',width=src.width,height=src.height,crs=src.crs,transform=src.transform,
                           tiled=True,compress='lzw',BIGTIFF='IF_SAFER')
            for name,path in partials.items():
                floating = name in ('lookup','local_incidence','doppler_residual')
                with rasterio.open(path,'w',**profile,count=2 if name=='lookup' else 1,
                                   dtype='float64' if name=='lookup' else ('float32' if floating else 'uint8'),
                                   nodata=np.nan if floating else 255):
                    pass
            write_json(state_path,state)
        for y in range(state['next_row'],src.height,block_rows):
            window = Window(0,y,src.width,min(block_rows,src.height-y))
            points,normals = surface_block(src,window,transformer)
            result = solve_points(exe,model,points.reshape(-1,3),normals.reshape(-1,3))
            result = result.reshape(int(window.height),src.width,8)
            status = result[...,5].astype('uint8')
            solved = np.isin(status,(0,6))
            lookup = np.where(solved[...,None],result[...,:2],np.nan)
            arrays = dict(lookup=np.moveaxis(lookup,-1,0),geometry_validity=solved.astype('uint8')[None],
                          radar_coverage=(status==0).astype('uint8')[None],geometry_status=status[None],
                          local_incidence=np.where(solved,result[...,6],np.nan)[None],
                          doppler_residual=result[...,3][None])
            # 每块关闭栅格后再提交进度；意外中断最多重算一个块，不跳过未落盘数据。
            for name,array in arrays.items():
                with rasterio.open(partials[name],'r+') as dst:
                    dst.write(array.astype(dst.dtypes[0]),window=window)
            for value,count in zip(*np.unique(status,return_counts=True)):
                k=STATUS_NAMES[int(value)];state['counts'][k]=state['counts'].get(k,0)+int(count)
            residual = np.abs(result[...,3][solved])
            if residual.size:
                state['residual_abs_max_hz']=max(state['residual_abs_max_hz'] or 0,float(residual.max()))
            state['next_row']=y+int(window.height);write_json(state_path,state)
            log(f'[03] 名义几何：{state["next_row"]}/{src.height}行；耗时{perf_counter()-start:.1f}s')
        report = dict(stage='nominal_only',complete=True,algorithm=ALGORITHM_VERSION,source=key_data,
                      metadata=metadata['description'],width=src.width,height=src.height,crs=src.crs.to_string(),
                      transform=list(src.transform),counts=state['counts'],status_codes=STATUS_NAMES,
                      residual_abs_max_hz=state['residual_abs_max_hz'],visibility='not_computed',
                      refinement='not_computed',height_datum='WGS84_ellipsoid',
                      note='名义LUT实验，不可将geometry_validity作为无遮挡或水体排除掩膜。')
    for name,path in partials.items():
        path.replace(output/f'{name}.tif')
    write_json(completed,report);state['complete']=True;write_json(state_path,state)
    return report
