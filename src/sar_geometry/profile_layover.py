"""BC 条带叠掩。原始 XML 与椭球高 DEM 定义剖面，SAR 地图定义输出范围。"""
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
import sys

import numpy as np
import rasterio
from affine import Affine
from pyproj import CRS, Transformer
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import reproject, transform_bounds
from scipy.ndimage import distance_transform_edt, map_coordinates, spline_filter

from .core import scan_gc_map2_v57_shadow_layover
from .profile_regrid import build_profile_mask, load_kernel
from .profile_segment import source_segment
from .profile_orbit import body_fields, HermiteOrbit, SixChannelOrbit, solve_xyz, solve_hermite_geometry
from .scene import make_terrain_raster_grid
from utils.files import write_json

def geometry_fields(source, metadata, folder, stack, log, orbit_mode):
    """在六倍 DEM 节点求多视行列、横轨距离（m）和视角（rad）。

    方位、距离索引从多视像元中心起算。临时数组按纬度分块写入，共享同一
    DEM 样条边界。高程缺测节点保留 NaN，在反求后的扫描和输出中传播。
    """
    radar, states = metadata['grid'], metadata['orbit']
    window, _, extrema = source_segment(source, radar, states)
    with rasterio.open(source) as ds:
        source_transform = ds.window_transform(window)
        first_lon,first_lat = ds.transform*(window.col_off+.5,window.row_off+.5)
        source_dem = ds.read(1,window=window,masked=True)
    z = np.asarray(source_dem.data,dtype='f8')
    gaps = np.ma.getmaskarray(source_dem) | ~np.isfinite(z)
    if np.all(gaps):
        raise ValueError('No valid elevation in the source segment')
    if np.any(gaps):
        nearest = distance_transform_edt(gaps,return_distances=False,return_indices=True)
        z[gaps] = z[tuple(nearest[:,gaps])]
    valid = None if not gaps.any() else ~gaps
    coefficients = spline_filter(z,order=2,mode='mirror')
    factor = 6
    shape = tuple((k-1)*factor+1 for k in z.shape)
    dx,dy = source_transform.a/factor,source_transform.e/factor
    transform = Affine(dx,0,first_lon-dx/2,0,dy,first_lat-dy/2)
    az_dt = radar['line_time']
    t0 = radar['time0']
    spacing = radar['range_spacing']
    mli_shape = (min(radar['rows'],int(np.floor((states[-1,0]-t0)/az_dt))+1)//3,radar['columns']//3)
    first_az, first_range = t0+az_dt, radar['range0']+spacing
    line_time = 3*az_dt
    t_center = first_az+line_time*mli_shape[0]/2
    hermite_center = t0+(mli_shape[0]*3-1)*az_dt/2
    physical = SixChannelOrbit(states) if orbit_mode=='bc_mli' else HermiteOrbit(states)
    registration = SixChannelOrbit(states) if orbit_mode=='bc_azimuth' else None
    project = Transformer.from_crs(4979,4978,always_xy=True)
    fields = {}
    for name,dtype in [('az','f4'),('rg','f4'),('yb','f8'),('look','f4')]:
        arr = np.memmap(folder/(name+'.bin'),mode='w+',dtype=dtype,shape=shape)
        stack.callback(arr._mmap.close)
        fields[name] = arr
    col = np.arange(shape[1])[None,:]
    for start in range(0,shape[0],40):
        end = min(start+40,shape[0])
        row,cols = np.broadcast_arrays(np.arange(start,end)[:,None],col)
        coords = np.stack((row.ravel()/factor,cols.ravel()/factor))
        h = map_coordinates(coefficients,coords,order=2,mode='mirror',prefilter=False,output=np.float32)
        good = (np.ones(len(h),bool) if valid is None else
                map_coordinates(valid,coords,order=0,mode='nearest',prefilter=False))
        xyz = np.stack(project.transform(first_lon+cols.ravel()[good]*dx,
                       first_lat+row.ravel()[good]*dy,h[good]),axis=1)
        for arr in fields.values():
            arr[start:end] = np.nan
        if len(xyz):
            if orbit_mode=='bc_mli':
                t,pos,vel = solve_xyz(xyz,physical,t_center)
            else:
                t,pos,vel = solve_hermite_geometry(xyz,physical,hermite_center)
            r = np.linalg.norm(xyz-pos,axis=1)
            if orbit_mode!='bc_mli':
                # The physical Hermite field carries the MLI-time coordinate as float32.
                body_time = ((t-first_az)/line_time).astype('f4').astype('f8')*line_time+first_az
                pos,vel,_,_ = physical.state(body_time)
            y,look,_ = body_fields(xyz,np.column_stack((pos,vel)),radar['look_side'])
            if registration is not None:
                # BC 多视轴只用于登记方位行。r、y、look 仍来自原时刻的物理轨道。
                t,_,_ = solve_xyz(xyz,registration,t_center)
            for name,v in [('az',(t-first_az)/line_time),('rg',(r-first_range)/(3*spacing)),
                           ('yb',y),('look',look)]:
                fields[name][start:end].reshape(-1)[good] = v
        if start % 1000 == 0:
            log(f'原生高程格点定位 {start}/{shape[0]} 行')
    return fields,transform,mli_shape,3*spacing,window,extrema


def classify(fields, mask, grid):
    """将内部扫描的叠掩位回填到 DEM 节点，返回 0、1、255 分类图。

    四个邻点都有效时取双线性权重，支持度恰等于 0.50 也判为叠掩。
    邻域有缺测时沿用最近节点。最近节点也缺测或超出幅宽时写 255。
    """
    shape = fields['az'].shape
    result = np.full(shape,255,'u1')
    lay = (mask & 4) != 0
    for start in range(0,shape[0],100):
        s = slice(start,start+100)
        row,rg,yb = (fields[n][s] for n in ('az','rg','yb'))
        col = (yb-grid.y0)/grid.dy
        ok = ((row>-.5)&(row<grid.az_lines-.5)&(rg>-.5)&(rg<grid.radar_width-.5)&
              (col>-.5)&(col<grid.columns-.5))
        y,x = row[ok].astype('f8'),col[ok]
        r,c = np.floor(y+.5).astype('i4'),np.floor(x+.5).astype('i4')
        val = np.where(mask[r,c]==0,255,lay[r,c]).astype('u1')
        x,y = np.clip(x,0,grid.columns-1),np.clip(y,0,grid.az_lines-1)
        r0,c0 = y.astype('i4'),x.astype('i4')
        r1,c1 = np.minimum(r0+1,grid.az_lines-1),np.minimum(c0+1,grid.columns-1)
        dy,dx = y-r0,x-c0
        w = (1-dy)*((1-dx)*lay[r0,c0]+dx*lay[r0,c1])+dy*((1-dx)*lay[r1,c0]+dx*lay[r1,c1])
        good = (mask[r0,c0]!=0)&(mask[r1,c0]!=0)&(mask[r0,c1]!=0)&(mask[r1,c1]!=0)
        val[good] = w[good]>=.5
        result[s][ok] = val
    return result


def write_outputs(folder, native, transform, org_path, scene):
    """保存经纬度分类与 10 m 投影图，按 SAR 有效掩膜裁切，类别用最近邻。"""
    native_profile = dict(driver='GTiff',crs='EPSG:4326',transform=transform,count=1,
                          width=native.shape[1],height=native.shape[0],dtype='uint8',
                          nodata=255,compress='lzw',tiled=True)
    with rasterio.open(folder/'layover_native.tif','w',**native_profile) as dst:
        dst.write(native,1)
    with rasterio.open(org_path) as org:
        crs = CRS.from_user_input(scene['output_crs'])
        if not crs.is_projected or any(axis.unit_conversion_factor!=1 for axis in crs.axis_info[:2]):
            raise ValueError('The 10 m output requires a projected CRS in metres')
        bounds = transform_bounds(org.crs,crs,*org.bounds,densify_pts=21)
        p = make_terrain_raster_grid(bounds,10,crs)
        p.update(dtype='uint8',nodata=255)
        out = np.full((p['height'],p['width']),255,'u1')
        reproject(native,out,src_transform=transform,src_crs='EPSG:4326',src_nodata=255,
                  dst_transform=p['transform'],dst_crs=crs,dst_nodata=255,resampling=Resampling.nearest)
        with WarpedVRT(org,crs=crs,transform=p['transform'],width=p['width'],height=p['height'],
                       src_nodata=scene.get('org_nodata',org.nodata),nodata=np.nan,dtype='float32',
                       resampling=Resampling.nearest) as vrt, rasterio.open(folder/'layover_10m.tif','w',**p) as dst:
            for _,win in dst.block_windows(1):
                data = vrt.read(1,window=win,masked=True)
                tile = out[win.toslices()]
                tile[np.ma.getmaskarray(data)|~np.isfinite(data.data)] = 255
                dst.write(tile,1,window=win)
    return dict(height=p['height'],width=p['width'],crs=crs.to_string(),pixel_size_m=10.)


def run_profile_layover(config, metadata, log=print):
    paths, scene = config['paths'],config['scene']
    orbit_mode = scene.get('profile_orbit','xml_hermite')
    if orbit_mode not in ('xml_hermite','bc_mli','bc_azimuth'):
        raise ValueError('profile_orbit must be xml_hermite, bc_mli or bc_azimuth')
    if config['model']['height_datum']!='WGS84_ellipsoid' or config['model']['doppler_mode']!='zero':
        raise ValueError('profile_layover requires an ellipsoid-height DEM and zero-Doppler grid')
    if scene.get('resolution_m',10)!=10:
        raise ValueError('profile_layover uses a 10 metre SAR comparison grid')
    dem = paths.get('DEM_PATH')
    if dem is None:
        tiles = paths['DEM_TILES']
        if len(tiles)!=1:
            raise ValueError('Provide the source-post DEM mosaic through DEM_PATH or one DEM_TILES entry')
        dem = tiles[0]
    core = Path(paths['CORE_PATH'])
    suffix = '.dll' if sys.platform=='win32' else '.dylib' if sys.platform=='darwin' else '.so'
    library = paths.get('REGRID_PATH') or core.with_name('sar_profile_regrid'+suffix)
    kernel = load_kernel(library)
    output = Path(paths['OUTPUT_DIR'])
    output.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix='.profile_',dir=output) as tmp:
        folder = Path(tmp)
        with ExitStack() as stack:
            fields,transform,shape,spacing,win,limits = geometry_fields(dem,metadata,folder,stack,log,orbit_mode)
            scan = lambda angle,r: scan_gc_map2_v57_shadow_layover(core,angle,r,1)
            mask,grid = build_profile_mask(fields,shape,spacing,scan,kernel,on_progress=lambda y,n:
                                         log(f'叠掩剖面扫描 {y}/{n} 行') if y%1024==0 or y==n else None)
            native = classify(fields,mask,grid)
            target = write_outputs(folder,native,transform,paths['ORG_PATH'],scene)
            report = dict(mode='profile_layover',source_dem=str(dem),source_annotation=str(paths['XML_PATH']),
                          sar_footprint=str(paths['ORG_PATH']),look_side=metadata['description']['look_side'],
                          profile_orbit=orbit_mode,dem_oversampling=[6,6],
                          dem_interpolator='quadratic cardinal B-spline, whole-sample mirror',azimuth_decimation=1,
                          range_oversampling=.8,multilook=[3,3],vote_threshold=.5,source_height_bounds_m=limits,
                          source_window_col_row_w_h=list(win.flatten()),profile_grid=asdict(grid),
                          native_shape=list(native.shape),sar_grid=target,classes={'0':'clear','1':'layover','255':'unknown'})
        write_json(folder/'geometry_quality.json', report)
        for name in ('layover_native.tif','layover_10m.tif','geometry_quality.json'):
            (folder/name).replace(output/name)
    return report
