"""逐景DEM高程包络：同斜距支持域调查与椭球高极值计算。

未知地形不猜填；只有源DEM覆盖调查域且没有缺测时才生成区域包络。
"""
from pathlib import Path
import json
import numpy as np
import rasterio
from rasterio.windows import from_bounds, Window
from rasterio.warp import transform_bounds
from shapely.geometry import box
from shapely.ops import unary_union
from .core import interpolate_hermite_orbit_states
from .profiles import TO_GEODETIC, zero_doppler_observation_frame, intersect_profile_ray_with_height_surface
from utils.files import write_json


def bound_same_range_terrain(metadata, executable, height_bounds, options):
    """估计成像期间同斜距分支的地理包围域，返回UTM/经纬度界限。

    只考虑XML指定观测侧。逐时刻、地表角采样，并用高程上下界之间的整条
    径向线段求距离界限，避免只检查两个端点漏掉内部最小距离。额外外扩由
    coverage_guard_m明确给出；这是有保护带的数值覆盖检查，不是连续域证明。
    """
    grid = metadata['grid']
    first = max(0., float(metadata['orbit'][0, 0]))
    last = min((grid['rows']-1)*grid['line_time'], float(metadata['orbit'][-1, 0]))
    times = np.linspace(first, last, options['time_samples'])
    states = interpolate_hermite_orbit_states(executable, metadata['orbit'], times)
    origin, e0, e1 = zero_doppler_observation_frame(states, grid['look_side'])
    theta = np.linspace(1e-5, options['theta_max_rad'], options['theta_samples'])
    direction = np.cos(theta)[None, :, None]*e0[:, None] + np.sin(theta)[None, :, None]*e1[:, None]
    low = intersect_profile_ray_with_height_surface(origin[:, None], direction, height_bounds[0])
    high = intersect_profile_ray_with_height_surface(origin[:, None], direction, height_bounds[1])
    low_radius = np.linalg.norm(low-origin[:, None], axis=-1)
    high_radius = np.linalg.norm(high-origin[:, None], axis=-1)
    nearest_radius = np.clip(np.sum((states[:, None, :3]-origin[:, None])*direction, axis=-1), low_radius, high_radius)
    minimum = np.linalg.norm(origin[:, None]+nearest_radius[..., None]*direction-states[:, None, :3], axis=-1)
    maximum = np.maximum(np.linalg.norm(low-states[:, None, :3], axis=-1), np.linalg.norm(high-states[:, None, :3], axis=-1))
    near = grid['range0']
    far = near + (grid['columns']-1)*grid['range_spacing']
    possible = (minimum <= far) & (maximum >= near)
    if not possible.any() or possible[:, [0, -1]].any():
        raise ValueError('同斜距候选范围触及角采样边界，不能用该范围确定区域包络')
    points = np.concatenate((low[possible], high[possible]))
    lon, lat, height = TO_GEODETIC.transform(*points.T)
    from pyproj import Transformer
    x, y, _ = Transformer.from_crs(4979, options['output_crs'], always_xy=True).transform(lon, lat, height)
    guard = options['coverage_guard_m']
    utm = [float(x.min()-guard), float(y.min()-guard), float(x.max()+guard), float(y.max()+guard)]
    geographic = transform_bounds(options['output_crs'], 4326, *utm, densify_pts=41)
    return dict(utm_bounds_with_guard_m=utm, geographic_bounds_with_guard=list(geographic),
                time_samples=options['time_samples'], theta_samples=options['theta_samples'],
                coverage_guard_m=guard, global_height_bounds_m=height_bounds,
                limitation='离散轨道/角度采样加保护带；不构成连续域完备性证明')


def scan_regional_dem_envelope(dem_paths, geoid_path, domain, margin_m, rounding_m, cache_dir=None):
    """扫描完整源瓦片的正高H与局部格网N，以h=H+N构造椭球高界限。

    DEM必须覆盖候选域且没有缺测；不会用当前20公里支持域的极值代表外围。
    极值按Hmin+Nmin/Hmax+Nmax保守组合，另加显式工程裕度后向外取整。
    margin_m不是统计置信区间，也不能保证真实地形误差绝不超过该裕度。
    cache_dir可共享完整瓦片极值；文件名/大小/mtime改变时重读，不因根目录迁移拦截。
    """
    if margin_m < 0 or rounding_m <= 0:
        raise ValueError('高程裕度须非负，取整间隔须为正')
    minimum, maximum = np.inf, -np.inf
    footprints, sources = [], []
    for path in dem_paths:
        path = Path(path)
        stat = path.stat()
        identity = dict(name=path.name, size=stat.st_size, mtime_ns=stat.st_mtime_ns)
        record = Path(cache_dir)/(path.name+'.height.json') if cache_dir is not None else None
        cached = json.loads(record.read_text(encoding='utf-8')) if record is not None and record.exists() else {}
        with rasterio.open(path) as src:
            if not src.crs or src.crs.to_epsg() != 4326:
                raise ValueError('区域包络扫描目前要求明确的EPSG:4326源DEM')
            identity.update(shape=list(src.shape), transform=list(src.transform),
                            scale=src.scales[0], offset=src.offsets[0])
            if cached.get('identity') == identity:
                missing, low, high = cached['missing_pixels'], cached['minimum'], cached['maximum']
            else:
                missing, low, high = 0, np.inf, -np.inf
                for _, window in src.block_windows(1):
                    values = src.read(1, window=window, masked=True, out_dtype='float64')
                    values = values*src.scales[0]+src.offsets[0]
                    valid = ~np.ma.getmaskarray(values) & np.isfinite(values.data)
                    missing += int(valid.size-valid.sum())
                    if valid.any():
                        low = min(low, float(values.data[valid].min()))
                        high = max(high, float(values.data[valid].max()))
                if record is not None and not missing:
                    write_json(record, dict(identity=identity, missing_pixels=missing, minimum=low, maximum=high))
            if missing:
                raise ValueError(f'源DEM有{missing}个缺测像元，不能据此确认外围高程包络：{path}')
            minimum, maximum = min(minimum, low), max(maximum, high)
            footprints.append(box(*src.bounds))
            sources.append(dict(path=str(path), bounds=list(src.bounds), missing_pixels=missing))
    if not unary_union(footprints).covers(box(*domain['geographic_bounds_with_guard'])):
        raise ValueError('源DEM未覆盖潜在同斜距地形的保护域，需要补充瓦片，不能直接缩小高程包络')
    # 按源格网分辨率读取各瓦片覆盖的N窗口，并带一格插值邻域，不只采样四角。
    nmin, nmax = np.inf, -np.inf
    with rasterio.open(geoid_path) as src:
        # PROJ高程改正格网常标为4979；其水平坐标同为WGS84经纬度，波段仍为N。
        if src.crs.to_epsg() not in (4326, 4979):
            raise ValueError('大地水准面格网须明确为WGS84经纬度')
        for footprint in footprints:
            w = from_bounds(*footprint.bounds, src.transform)
            left, top = int(np.floor(w.col_off))-1, int(np.floor(w.row_off))-1
            right, bottom = int(np.ceil(w.col_off+w.width))+1, int(np.ceil(w.row_off+w.height))+1
            if left < 0 or top < 0 or right > src.width or bottom > src.height:
                raise ValueError('高程转换格网没有完整覆盖源DEM及插值邻域')
            values = src.read(1, window=Window(left, top, right-left, bottom-top), masked=True, out_dtype='float64')
            values = values*src.scales[0]+src.offsets[0]
            if np.ma.getmaskarray(values).any() or not np.isfinite(values.data).all():
                raise ValueError('大地水准面格网存在缺测，不能确认高程包络')
            nmin = min(nmin, float(values.min()))
            nmax = max(nmax, float(values.max()))
    bounds = [float(np.floor((minimum+nmin-margin_m)/rounding_m)*rounding_m),
              float(np.ceil((maximum+nmax+margin_m)/rounding_m)*rounding_m)]
    return dict(dem_sources=sources, orthometric_height_range_m=[minimum, maximum],
                geoid_undulation_range_m=[nmin, nmax], height_bounds_m=bounds,
                engineering_margin_m=margin_m, outward_rounding_m=rounding_m,
                supported_domain=domain, status='候选区域界限，受DEM和数值覆盖假设限制')


