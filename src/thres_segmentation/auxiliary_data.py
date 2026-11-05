"""Headless auxiliary-data acquisition; no Qt, GUI worker or osgeo dependency.

Sources: https://esa-worldcover.org/en/data-access
         https://registry.opendata.aws/copernicus-dem/
"""
from contextlib import ExitStack
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import ssl
import tempfile
import time
import urllib.error
import urllib.request

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from rasterio.windows import Window

WORLD_COVER_VERSIONS = {2020: 'v100', 2021: 'v200'}
WORLD_COVER_CLASSES = (10,20,30,40,50,60,70,80,90,95,100)


def validate_bounds(bounds):
    west,south,east,north = map(float,bounds)
    if not all(math.isfinite(x) for x in (west,south,east,north)) or not (
            -180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError(f'Invalid WGS84 bounds or antimeridian crossing: {bounds}')
    # A nearly global box often indicates a wrapped scene. Split it explicitly.
    if east-west > 180:
        raise ValueError('Split scenes spanning more than 180 degrees before downloading')
    return west,south,east,north


def tile_specs(product, bounds, year=2021):
    west,south,east,north = validate_bounds(bounds)
    if product == 'worldcover':
        if year not in WORLD_COVER_VERSIONS:
            raise ValueError('WorldCover year must be 2020 or 2021')
        step = 3
    elif product == 'copernicus':
        step = 1
    else:
        raise ValueError(f'Unknown auxiliary product: {product}')
    specs = []
    for lat in range(math.floor(south/step)*step,math.ceil(north/step)*step,step):
        for lon in range(math.floor(west/step)*step,math.ceil(east/step)*step,step):
            ns,ew = ('N' if lat >= 0 else 'S'),('E' if lon >= 0 else 'W')
            if product == 'worldcover':
                tile = f'{ns}{abs(lat):02d}{ew}{abs(lon):03d}'
                version = WORLD_COVER_VERSIONS[year]
                filename = f'ESA_WorldCover_10m_{year}_{version}_{tile}_Map.tif'
                url = f'https://esa-worldcover.s3.eu-central-1.amazonaws.com/{version}/{year}/map/{filename}'
            else:
                tile = f'Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM'
                filename = tile+'.tif'
                url = f'https://copernicus-dem-30m.s3.amazonaws.com/{tile}/{filename}'
            specs.append(dict(tile=tile,filename=filename,url=url))
    return specs


def _check_cancel(cancel):
    if cancel and cancel():
        raise InterruptedError('Auxiliary data preparation cancelled')


def validate_raster(path):
    """Reject HTML, missing CRS and incomplete corner blocks before accepting a cache."""
    with rasterio.open(path) as source:
        if source.driver != 'GTiff' or not source.crs or source.count != 1 or min(source.shape) < 1:
            raise ValueError(f'Not a georeferenced single-band GeoTIFF: {path}')
        source.read(1,window=Window(0,0,min(32,source.width),min(32,source.height)))
        source.read(1,window=Window(max(0,source.width-32),max(0,source.height-32),
                                   min(32,source.width),min(32,source.height)))


def _write_json(path, value):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name+'.',suffix='.part',dir=path.parent)
    try:
        with os.fdopen(descriptor,'w',encoding='utf-8') as stream:
            json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def download_tile(url, destination, timeout=60, retries=3, log=print, cancel=None):
    """Atomic download with bounded retries, TIFF validation and a source receipt.

    Cache hits require matching URL, byte size and modification time. SHA256 is
    recorded during download; it is not a vendor checksum or authenticity proof.
    """
    if timeout <= 0 or retries < 1:
        raise ValueError('timeout and retries must be positive')
    destination = Path(destination)
    receipt_path = destination.with_suffix('.source.json')
    _check_cancel(cancel)
    if destination.is_file() and receipt_path.is_file():
        try:
            receipt = json.loads(receipt_path.read_text('utf-8'))
            stat = destination.stat()
            if receipt['url'] == url and receipt['bytes'] == stat.st_size and receipt['mtime_ns'] == stat.st_mtime_ns:
                validate_raster(destination)
                log(f'复用瓦片：{destination.name}')
                return destination
        except (ValueError,KeyError,OSError,rasterio.errors.RasterioError):
            pass
    destination.parent.mkdir(parents=True,exist_ok=True)
    context = ssl.create_default_context()
    # Use the active environment's trusted CA bundle when available.
    try:
        import certifi
        context.load_verify_locations(certifi.where())
    except ImportError:
        pass
    for attempt in range(1,retries+1):
        _check_cancel(cancel)
        descriptor, temporary = tempfile.mkstemp(prefix=destination.name+'.',suffix='.part',dir=destination.parent)
        try:
            log(f'下载 {destination.name} ({attempt}/{retries})')
            request = urllib.request.Request(url,headers={'User-Agent':'SARWater-Auxiliary/1.0'})
            digest = hashlib.sha256(); length = 0
            with os.fdopen(descriptor,'wb') as output:
                with urllib.request.urlopen(request,timeout=timeout,context=context) as response:
                    content_type = response.headers.get('Content-Type','').lower()
                    if any(x in content_type for x in ('text/html','text/xml','application/xml','application/json')):
                        raise ValueError(f'Server returned {content_type}, expected GeoTIFF')
                    expected = response.headers.get('Content-Length')
                    while True:
                        _check_cancel(cancel)
                        chunk = response.read(1024*1024)
                        if not chunk:
                            break
                        if length == 0 and chunk[:4] not in (b'II*\x00',b'MM\x00*',b'II+\x00',b'MM\x00+'):
                            raise ValueError('Downloaded content has no TIFF signature')
                        output.write(chunk); digest.update(chunk); length += len(chunk)
                    if expected is not None and length != int(expected):
                        raise ValueError('Incomplete HTTP body (Content-Length mismatch)')
                    etag = response.headers.get('ETag')
            validate_raster(temporary)
            os.replace(temporary,destination)
            _write_json(receipt_path,dict(url=url,bytes=length,sha256=digest.hexdigest(),etag=etag,
                                        mtime_ns=destination.stat().st_mtime_ns))
            return destination
        except InterruptedError:
            raise
        except Exception as exc:
            if attempt == retries or isinstance(exc,urllib.error.HTTPError) and exc.code in (400,401,403,404):
                raise RuntimeError(f'Auxiliary tile unavailable: {url}: {exc}') from exc
            log(f'下载失败，重试：{exc}')
            time.sleep(min(attempt,3))
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def _signature(path):
    path = Path(path)
    stat = path.stat()
    return dict(path=str(path.resolve()),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)


def mosaic_tiles(paths, bounds, destination, water=False, max_pixels=100000000,
                 block_size=512, log=print, cancel=None):
    """Windowed WGS84 crop at the finest source angular grid spacing.

    Water output uses 1=water, 0=dry, 255=unknown. WorldCover class 0 and
    unrecognized values remain unknown. Elevation keeps valid zeros.
    """
    bounds = validate_bounds(bounds)
    if not paths or block_size < 1 or max_pixels < 1:
        raise ValueError('Mosaic requires source tiles and positive size limits')
    destination = Path(destination)
    receipt_path = destination.with_suffix('.manifest.json')
    request = dict(algorithm='auxiliary_crop_v1',bounds=list(bounds),water=water,
                   sources=[_signature(p) for p in paths])
    if destination.is_file() and receipt_path.is_file():
        try:
            previous = json.loads(receipt_path.read_text('utf-8'))
            if previous['request'] == request and previous['output'] == _signature(destination):
                validate_raster(destination)
                with rasterio.open(destination) as source:
                    if source.width*source.height > max_pixels:
                        raise ValueError('Cached auxiliary crop exceeds max_pixels')
                log(f'复用辅助裁剪：{destination.name}')
                return destination
        except (KeyError,OSError,rasterio.errors.RasterioError,json.JSONDecodeError):
            pass
    destination.parent.mkdir(parents=True,exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=destination.name+'.',suffix='.part.tif',dir=destination.parent)
    os.close(descriptor)
    try:
        with ExitStack() as stack:
            sources = [stack.enter_context(rasterio.open(p)) for p in paths]
            if any(s.crs != rasterio.crs.CRS.from_epsg(4326) or s.transform.b or s.transform.d or
                   s.transform.a <= 0 or s.transform.e >= 0 for s in sources):
                raise ValueError('Public auxiliary tiles must have north-up EPSG:4326 grids')
            rx,ry = min(s.res[0] for s in sources),min(s.res[1] for s in sources)
            west,south,east,north = bounds
            left = math.floor((west+180)/rx)*rx-180
            top = 90-math.floor((90-north)/ry)*ry
            width,height = math.ceil((east-left)/rx),math.ceil((top-south)/ry)
            if width*height > max_pixels:
                raise ValueError(f'Auxiliary crop has {width*height} pixels, exceeds max_pixels={max_pixels}')
            transform = from_origin(left,top,rx,ry)
            warped = [stack.enter_context(WarpedVRT(s,crs='EPSG:4326',transform=transform,
                       width=width,height=height,dtype='float32',nodata=float('nan'),
                       resampling=Resampling.nearest if water else Resampling.bilinear)) for s in sources]
            profile = dict(driver='GTiff',width=width,height=height,count=1,crs='EPSG:4326',
                transform=transform,dtype='uint8' if water else 'float32',nodata=255 if water else -9999.,
                compress='deflate',tiled=True,blockxsize=256,blockysize=256,BIGTIFF='IF_SAFER')
            with rasterio.open(temporary,'w',**profile) as output:
                output.update_tags(product='WorldCover_water' if water else 'Copernicus_GLO30_DSM',
                                   classes='0=dry,1=water,255=unknown' if water else 'elevation_m')
                for row in range(0,height,block_size):
                    for col in range(0,width,block_size):
                        _check_cancel(cancel)
                        window = Window(col,row,min(block_size,width-col),min(block_size,height-row))
                        shape = (int(window.height),int(window.width))
                        result = np.full(shape,profile['nodata'],dtype=profile['dtype'])
                        occupied = np.zeros(shape,bool)
                        for source in warped:
                            data = source.read(1,window=window,masked=True)
                            valid = ~np.ma.getmaskarray(data) & np.isfinite(data.data) & ~occupied
                            if water:
                                valid &= np.isin(data.data,WORLD_COVER_CLASSES)
                                result[valid] = (data.data[valid] == 80).astype('uint8')
                            else:
                                result[valid] = data.data[valid]
                            occupied |= valid
                        output.write(result,1,window=window)
        validate_raster(temporary)
        os.replace(temporary,destination)
        _write_json(receipt_path,dict(request=request,output=_signature(destination)))
        return destination
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def selected_images(image_dir, split_manifest=None, scene_prefix=None):
    """Use exactly active validation records, or the external-directory raster list."""
    image_dir = Path(image_dir)
    if split_manifest is not None:
        split_manifest = Path(split_manifest)
        if image_dir.resolve() != (split_manifest.parent/'images/val').resolve():
            raise ValueError('Auxiliary inputs must use manifest images/val')
        with split_manifest.open(encoding='utf-8-sig',newline='') as stream:
            names = [r['name'] for r in csv.DictReader(stream) if r['split']=='val' and int(r['active'])==1]
    else:
        names = sorted(p.relative_to(image_dir).as_posix() for p in image_dir.rglob('*')
                       if p.is_file() and p.suffix.lower() in ('.tif','.tiff','.img','.vrt','.png'))
    if scene_prefix is not None:
        names = [name for name in names if name.startswith(scene_prefix)]
    result = {}
    for name in names:
        path = image_dir/name
        if not path.resolve().is_relative_to(image_dir.resolve()) or not path.is_file():
            raise ValueError(f'Missing or out-of-directory scene: {name}')
        if name in result:
            raise ValueError(f'Duplicate scene: {name}')
        result[name] = path
    if not result:
        raise ValueError('No selected images for auxiliary preparation')
    return result


def prepare_auxiliary_inputs(images, cache_dir, analysis_options=None, download_history=False,
                             download_elevation=False, worldcover_year=2021, buffer_m=1000.,
                             max_tiles_per_scene=64, max_mosaic_pixels=100000000,
                             timeout=60, retries=3, require_depth_reference=True, log=print, cancel=None):
    """Prepare per-scene analysis kwargs; explicit local paths take precedence.

    Download failures abort; unknown/missing tiles are never silently treated as
    land or zero elevation. A versioned batch manifest is published on success.
    """
    base = dict(analysis_options or {})
    for key in ('reference_path','elevation_path','reference_coverage_path'):
        if base.get(key) and not Path(base[key]).is_file():
            raise FileNotFoundError(f'Configured local auxiliary file missing: {key}={base[key]}')
    if require_depth_reference and (download_elevation or base.get('elevation_path')) and base.get('depth_scope','new_water') == 'new_water' and not (base.get('reference_path') or download_history):
        raise ValueError('new_water depth requires historical data before downloading elevation')
    if not np.isfinite(buffer_m) or buffer_m < 0 or max_tiles_per_scene < 1:
        raise ValueError('Invalid auxiliary buffer or tile limit')
    cache_dir = Path(cache_dir)
    prepared = {}; records = []
    for name,image in images.items():
        _check_cancel(cancel)
        options = dict(base)
        need_history = download_history and not options.get('reference_path')
        need_elevation = download_elevation and not options.get('elevation_path')
        record = dict(name=name,image=str(Path(image).resolve()),products={})
        if need_history or need_elevation:
            with rasterio.open(image) as source:
                if not source.crs:
                    raise ValueError(f'Scene CRS required for auxiliary download: {image}')
                bounds = validate_bounds(transform_bounds(source.crs,'EPSG:4326',*source.bounds,densify_pts=21))
            west,south,east,north = bounds
            dy = buffer_m/111320
            dx = dy/max(math.cos(math.radians((south+north)/2)),.05)
            terrain_bounds = validate_bounds((max(-180,west-dx),max(-90,south-dy),min(180,east+dx),min(90,north+dy)))
            record['bounds_wgs84'] = list(bounds)
            for product,needed,extent in (('worldcover',need_history,bounds),('copernicus',need_elevation,terrain_bounds)):
                if not needed:
                    continue
                specs = tile_specs(product,extent,worldcover_year)
                if len(specs) > max_tiles_per_scene:
                    raise ValueError(f'{name}: {len(specs)} {product} tiles exceed max_tiles_per_scene')
                key = hashlib.sha256(json.dumps(dict(product=product,bounds=extent,
                    year=worldcover_year if product=='worldcover' else None),sort_keys=True).encode()).hexdigest()[:20]
                paths = [download_tile(s['url'],cache_dir/'tiles'/product/s['filename'],timeout=timeout,
                         retries=retries,log=log,cancel=cancel) for s in specs]
                output = mosaic_tiles(paths,extent,cache_dir/'scenes'/key/(product+'.tif'),
                         water=product=='worldcover',max_pixels=max_mosaic_pixels,log=log,cancel=cancel)
                record['products'][product] = dict(path=str(output.resolve()),tiles=specs,
                    year=worldcover_year if product=='worldcover' else None)
                if product=='worldcover':
                    options.update(reference_path=str(output.resolve()),reference_water_values=(1,),reference_dry_values=(0,))
                else:
                    options.update(elevation_path=str(output.resolve()),elevation_kind='DSM',elevation_scale=1.)
        if require_depth_reference and options.get('elevation_path') and options.get('depth_scope','new_water') == 'new_water' and not options.get('reference_path'):
            raise ValueError(f'{name}: new_water depth requires historical data; enable download_history or supply reference_path')
        prepared[name] = options
        record['analysis_options'] = {k:str(v) if isinstance(v,Path) else v for k,v in options.items()}
        records.append(record)
    cache_dir.mkdir(parents=True,exist_ok=True)
    payload = dict(status='complete',scenes=records)
    key = hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False).encode()).hexdigest()[:20]
    manifest = cache_dir/('batch_'+key+'.json')
    _write_json(manifest,payload)
    log(f'辅助数据清单：{manifest}')
    return prepared
