"""按当前视窗读取本地影像/天地图；统一Web Mercator，不修改原影像。"""
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen
import math

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.vrt import WarpedVRT
from PIL import Image, ImageDraw

WORLD = 20037508.342789244


def map_bounds(center, zoom, width, height):
    """天地图w矩阵与EPSG:3857使用一致的瓦片分辨率。"""
    resolution = 2 * WORLD / (256 * 2**zoom)
    x, y = center
    return (x-width*resolution/2, y-height*resolution/2,
            x+width*resolution/2, y+height*resolution/2)


def render_local(path, bounds, width, height, bands=(1,2,3), stretch=True):
    """虚拟重投影只读取当前视窗；支持不同CRS、单波段和RGB GeoTIFF。"""
    with rasterio.Env(GDAL_CACHEMAX=64*1024*1024), rasterio.open(path) as src:
        if src.crs is None:raise ValueError('影像缺少坐标系，无法自动叠加核验点')
        selected = (1,1,1) if src.count == 1 else tuple(bands)
        if len(selected)!=3 or min(selected)<1 or max(selected)>src.count:
            raise ValueError(f'影像共{src.count}个波段，请检查RGB波段编号')
        with WarpedVRT(src, crs='EPSG:3857', transform=from_bounds(*bounds,width,height),
                       width=width,height=height,resampling=Resampling.bilinear,
                       dtype='float32',nodata=np.nan) as vrt:
            data = vrt.read(selected,masked=True)
    values = data.filled(np.nan)
    valid = np.isfinite(values).all(axis=0)
    rgb = np.zeros((height,width,3),np.uint8)
    for c in range(3):
        plane=values[c]
        if stretch and valid.any():
            low,high=np.percentile(plane[valid],[2,98])
            scaled=(plane-low)/max(high-low,1e-6)*255
        else:scaled=plane
        rgb[:,:,c]=np.nan_to_num(np.clip(scaled,0,255),nan=0).astype('uint8')
    rgb[~valid]=(38,44,52)
    return Image.fromarray(rgb), ('当前视窗无有效影像覆盖' if not valid.any() else '本地影像；显示拉伸不修改源数据')


def fetch_tile(args):
    """只缓存有效图片；Key不写入缓存文件名、日志或错误消息。"""
    x,y,z,token,cache=args
    target=Path(cache)/'img_w'/str(z)/str(x)/f'{y}.png'
    if target.exists():
        try:
            with Image.open(target) as im:return x,y,im.convert('RGB'),None
        except OSError:pass
    url=('https://t0.tianditu.gov.cn/img_w/wmts?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0'
         f'&LAYER=img&STYLE=default&TILEMATRIXSET=w&FORMAT=tiles&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}&tk={token}')
    try:
        with urlopen(Request(url,headers={'User-Agent':'SARWaterPointReviewer/1.0'}),timeout=6) as response:
            payload=response.read(4*1024*1024)
        with Image.open(BytesIO(payload)) as im:tile=im.convert('RGB')
        if tile.size!=(256,256):raise ValueError('unexpected tile size')
        target.parent.mkdir(parents=True,exist_ok=True)
        # GUI单渲染线程保证同一瓦片不会同时写入。
        pending=target.with_suffix('.partial.png');tile.save(pending);pending.replace(target)
        return x,y,tile,None
    except Exception as exc:
        return x,y,None,type(exc).__name__


def render_tianditu(center,zoom,width,height,token,cache):
    if not token:raise ValueError('请填写天地图Key，或切换到本地光学影像')
    n=2**zoom;resolution=2*WORLD/(256*n)
    px=(center[0]+WORLD)/resolution;py=(WORLD-center[1])/resolution
    left,top=px-width/2,py-height/2
    jobs=[(x,y,zoom,token,cache) for y in range(max(0,math.floor(top/256)),min(n,math.ceil((top+height)/256)))
          for x in range(max(0,math.floor(left/256)),min(n,math.ceil((left+width)/256)))]
    image=Image.new('RGB',(width,height),(38,44,52));failed=0
    with ThreadPoolExecutor(max_workers=4) as pool:
        for x,y,tile,error in pool.map(fetch_tile,jobs):
            if error:failed+=1;continue
            image.paste(tile,(round(x*256-left),round(y*256-top)))
    if failed:
        ImageDraw.Draw(image).text((12,12),f'Tile requests failed: {failed}/{len(jobs)}',fill='white')
    message=f'天地图影像 | 未知拍摄日期 | © 天地图 | 瓦片失败 {failed}/{len(jobs)}'
    if failed:message+='；请检查Key类型、权限或网络，失败瓦片未缓存'
    return image,message
