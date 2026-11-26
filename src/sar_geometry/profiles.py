"""零Doppler观测剖面：椭球/ECEF构造、DEM双线性采样和有界支持域。

本模块没有将DEM行当成雷达方向，也不按斜距排序。每个时刻的地面点满足
V·(P-S)=0；地表角theta按近侧到远侧递增。非零Doppler斜视需要另一种曲面。
"""

from contextlib import closing
from functools import partial
import numpy as np
from pyproj import CRS, Transformer
from scipy.ndimage import map_coordinates
import rasterio
from rasterio.windows import Window
from .core import (
    interpolate_hermite_orbit_states,
    compute_radar_shadow_layover_batch,
    solve_range_doppler_points,
)

A = 6378137.0
B = 6356752.314245179
TO_GEODETIC = Transformer.from_crs(4978, 4979, always_xy=True)


def zero_doppler_observation_frame(states, look_side):
    """ECEF状态→零Doppler平面原点及径向/横向基，全部单位米。

    平面原点O是地心到V·P=V·S的垂足；e0朝卫星，e1朝指定观测侧。
    因而P=O+rho*(cos(theta)*e0+sin(theta)*e1)严格位于该平面。
    """
    position, velocity_unit = states[..., :3], states[..., 3:]
    velocity_unit = velocity_unit / np.linalg.norm(velocity_unit, axis=-1, keepdims=True)
    origin = np.sum(position * velocity_unit, axis=-1, keepdims=True) * velocity_unit
    radial_basis = position - origin
    radial_basis = radial_basis / np.linalg.norm(radial_basis, axis=-1, keepdims=True)
    cross_track_basis = look_side * np.cross(velocity_unit, radial_basis)
    return origin, radial_basis, cross_track_basis


def intersect_profile_ray_with_wgs84_ellipsoid(origin, direction):
    """射线O+rho*d与WGS84椭球的正根；用于高程迭代初值，不用球体替代椭球。"""
    axes = np.array([A, A, B])
    scaled_origin = origin / axes
    scaled_direction = direction / axes
    quadratic_a = np.sum(scaled_direction * scaled_direction, axis=-1)
    quadratic_b = 2 * np.sum(scaled_origin * scaled_direction, axis=-1)
    quadratic_c = np.sum(scaled_origin * scaled_origin, axis=-1) - 1
    discriminant = quadratic_b * quadratic_b - 4 * quadratic_a * quadratic_c
    return (-quadratic_b + np.sqrt(np.maximum(discriminant, 0))) / (2 * quadratic_a)


def intersect_profile_ray_with_height_surface(origin, direction, height):
    """固定椭球高曲面与平面径向射线求交，用于未观测地形的上下包络。

    使用PROJ的椭球高，沿射线迭代；这里的rho修正只作数值迭代，最终检查高程残差。
    """
    radius = intersect_profile_ray_with_wgs84_ellipsoid(origin, direction) + height
    for _ in range(5):
        ecef_points = origin + radius[..., None] * direction
        _, _, ellipsoid_height = TO_GEODETIC.transform(
            ecef_points[..., 0], ecef_points[..., 1], ecef_points[..., 2]
        )
        radius += height - ellipsoid_height
    return origin + radius[..., None] * direction


class ElevationSampler:
    """按请求包围窗读取DEM，缓存有限个GDAL块；不把多瓦片DEM整体载入内存。

    输入经纬度，输出椭球高米；双线性四邻域中任一无效即返回NaN，不跨洞填值。
    DEM须已完成明确的高程基准转换，read阶段不重复归一化。
    """

    def __init__(self, path):
        self.src = rasterio.open(path)
        self.transformer = Transformer.from_crs(4326, self.src.crs, always_xy=True)

    def close(self):
        self.src.close()

    def sample(self, lon, lat):
        # 细网格上，斜穿整景的细长剖面对应一个很大的轴对齐包围窗。
        # 沿横向拆分取高程，每个窗仍读足双线性邻域；整条剖面的折返判定在拼回后进行。
        if np.ndim(lon) == 2 and lon.shape[1] > 512:
            result = np.empty(lon.shape, dtype='float64')
            for start in range(0, lon.shape[1], 512):
                part = slice(start, start + 512)
                result[:, part] = self.sample(lon[:, part], lat[:, part])
            return result
        x, y = self.transformer.transform(lon, lat)
        inv = ~self.src.transform
        col = inv.a * x + inv.b * y + inv.c - 0.5
        row = inv.d * x + inv.e * y + inv.f - 0.5
        finite = np.isfinite(col) & np.isfinite(row)
        good = finite & (col >= 0) & (row >= 0) & (col < self.src.width - 1) & (row < self.src.height - 1)
        result = np.full(np.shape(lon), np.nan)
        if not good.any():
            return result
        c0 = int(np.floor(col[good].min()))
        r0 = int(np.floor(row[good].min()))
        c1 = int(np.ceil(col[good].max())) + 2
        r1 = int(np.ceil(row[good].max())) + 2
        win = Window(c0, r0, min(c1, self.src.width) - c0, min(r1, self.src.height) - r0)
        data = self.src.read(1, window=win, masked=True, out_dtype='float64').filled(np.nan)
        data = data * self.src.scales[0] + self.src.offsets[0]
        result[good] = map_coordinates(
            data, [row[good] - r0, col[good] - c0], order=1, prefilter=False, mode='constant', cval=np.nan
        )
        return result


def build_zero_doppler_terrain_profiles(
    executable, model, sampler, times, theta, height_bounds, tolerance_m=0.05, *, layover_scan='interval'
):
    """生成完整横向剖面并批量分类，返回掩膜/位置/有效标记及数值残差。

    方位行可分批，横向列保持完整。未知外围地形采用配置的全域椭球高上/下界；
    只有超出这些边界可能产生的遮挡/同距离范围，才能确定为可见或无叠掩。
    """
    if model['doppler_mode'] != 0:
        raise ValueError('观测剖面当前仅支持显式零Doppler；不能将斜视模型当零Doppler处理')
    states = interpolate_hermite_orbit_states(executable, model['orbit'], times)
    origin, radial_basis, cross_track_basis = zero_doppler_observation_frame(states, model['look_side'])
    direction = (
        np.cos(theta)[None, :, None] * radial_basis[:, None]
        + np.sin(theta)[None, :, None] * cross_track_basis[:, None]
    )
    radius = intersect_profile_ray_with_wgs84_ellipsoid(origin[:, None], direction)
    valid = np.ones(radius.shape, bool)
    residual = np.full(radius.shape, np.nan)
    # NumPy一次处理多条剖面；DEM只读覆盖这些剖面的小窗。
    for _ in range(10):
        surface_points = origin[:, None] + radius[..., None] * direction
        lon, lat, ellipsoid_height = TO_GEODETIC.transform(
            surface_points[..., 0], surface_points[..., 1], surface_points[..., 2]
        )
        terrain = sampler.sample(lon, lat)
        valid &= np.isfinite(terrain)
        residual = terrain - ellipsoid_height
        if np.any(valid & ((terrain < height_bounds[0]) | (terrain > height_bounds[1]))):
            raise ValueError('DEM超出配置高程包络，不能继续使用该包络判断支持域')
        if not np.any(valid & (np.abs(residual) > tolerance_m)):
            break
        radius += np.where(valid, residual, 0)
    valid &= np.abs(residual) <= tolerance_m
    line_of_sight = surface_points - states[:, None, :3]
    ranges = np.linalg.norm(line_of_sight, axis=-1)
    angles = np.arctan2(
        np.sum(line_of_sight * radial_basis[:, None], axis=-1),
        np.sum(line_of_sight * cross_track_basis[:, None], axis=-1),
    )
    points = np.stack(
        (np.broadcast_to(theta, ranges.shape) * A, ranges, angles, valid.astype(float)), axis=-1
    )
    # theta限制在近地可见单调区；不将地球背面或过地平线区域混入此前包络。
    if theta[0] <= 0 or theta[-1] >= 0.2:
        raise ValueError('首版剖面支持0<theta<0.2弧度；超出时需扩展几何边界模型')
    # DEM矩形边缘与斜向剖面相交位置各异；缺测首尾用包络表达，不冒充已知地形。
    first = np.argmax(valid, axis=1)
    last = valid.shape[1] - 1 - np.argmax(valid[:, ::-1], axis=1)
    near_dir = direction[np.arange(len(times)), first]
    far_dir = direction[np.arange(len(times)), last]
    near_hi = intersect_profile_ray_with_height_surface(origin, near_dir, height_bounds[1]) - states[:, :3]
    near_lo = intersect_profile_ray_with_height_surface(origin, near_dir, height_bounds[0]) - states[:, :3]
    far_hi = intersect_profile_ray_with_height_surface(origin, far_dir, height_bounds[1]) - states[:, :3]
    bounds = np.column_stack(
        (
            np.arctan2(np.sum(near_hi * radial_basis, axis=-1), np.sum(near_hi * cross_track_basis, axis=-1)),
            np.linalg.norm(near_lo, axis=-1),
            np.linalg.norm(far_hi, axis=-1),
        )
    )
    masks = compute_radar_shadow_layover_batch(executable, points, bounds, layover_scan=layover_scan)
    error = float(np.max(np.abs(residual[valid]))) if valid.any() else None
    return masks, surface_points, valid, error


def map_ecef_to_zero_doppler_profile(executable, model, points, solved):
    """把已解得时刻的DEM点映射到(t,theta)，不以斜距替代有折返的地表横向坐标。"""
    valid_geometry = np.isin(solved[:, 5], (0, 6))
    theta = np.full(len(points), np.nan)
    if valid_geometry.any():
        states = interpolate_hermite_orbit_states(executable, model['orbit'], solved[valid_geometry, 2])
        origin, radial_basis, cross_track_basis = zero_doppler_observation_frame(states, model['look_side'])
        surface_vectors = points[valid_geometry] - origin
        theta[valid_geometry] = np.arctan2(
            np.sum(surface_vectors * cross_track_basis, axis=-1),
            np.sum(surface_vectors * radial_basis, axis=-1),
        )
    return solved[:, 2], theta, valid_geometry


class DemPostingGrid:
    """由源 DEM 像元中心和指定细分倍数定义叠掩回填格点。"""

    def __init__(self, tiles, factor):
        if type(factor) is not int or factor < 1 or not tiles:
            raise ValueError('map_posting_factor 需要正整数和源 DEM')
        with rasterio.open(tiles[0]) as ref:
            crs, affine = ref.crs, ref.transform
        if crs is None or affine.b or affine.d or affine.a <= 0 or affine.e >= 0:
            raise ValueError('源 DEM 格点回填需要具有 CRS 的北向上网格')
        for path in tiles[1:]:
            with rasterio.open(path) as src:
                t = src.transform
                offset = (~affine) * (t.c, t.f)
                if (src.crs != crs or t.b or t.d or
                        not np.allclose([t.a, t.e], [affine.a, affine.e], atol=0, rtol=1e-9) or
                        not np.allclose(offset, np.rint(offset), atol=1e-5, rtol=0)):
                    raise ValueError('源 DEM 瓦片需要共用网格间距和像元中心格点')
        self.center = affine * (.5, .5)
        self.step = (affine.a / factor, affine.e / factor)
        self.identity = dict(crs=crs.to_wkt(), center=self.center, step=self.step)
        self.from_ecef = Transformer.from_crs(4978, CRS.from_user_input(crs).to_3d(), always_xy=True)
        self.to_geo = Transformer.from_crs(crs, 4326, always_xy=True)
        self.to_ecef = Transformer.from_crs(4979, 4978, always_xy=True)

    def surface_points(self, points, sampler):
        """N×3 ECEF 米坐标→最近细分 DEM 中心的地表 ECEF 点。缺高程时返回 NaN。"""
        x, y, _ = self.from_ecef.transform(*points.T)
        x = self.center[0] + np.floor((x - self.center[0]) / self.step[0] + .5) * self.step[0]
        y = self.center[1] + np.floor((y - self.center[1]) / self.step[1] + .5) * self.step[1]
        lon, lat = self.to_geo.transform(x, y)
        h = sampler.sample(lon, lat)
        return np.column_stack(self.to_ecef.transform(lon, lat, h))

    def bind(self, stack, dem):
        sampler = stack.enter_context(closing(ElevationSampler(dem)))
        return partial(self.surface_points, sampler=sampler)


def sample_zero_doppler_visibility(executable, model, points, times, theta, masks, *,
                                   subpixel_layover=False, post_points=None, layover_vote_threshold=0.5):
    """求原点的覆盖与阴影，可选在源 DEM 细分格点复查已知叠掩。"""
    if not 0 < layover_vote_threshold <= 1 or (not subpixel_layover and layover_vote_threshold != 0.5):
        raise ValueError('叠掩表决阈值应在 (0, 1]，自定义阈值需要 ordered_subpixel 模式')
    value, good, covered = _query_profile_masks(
        executable, model, points, times, theta, masks, subpixel_layover, layover_vote_threshold)
    if post_points is not None:
        indices = np.flatnonzero(value[:, 1] != 255)
        if indices.size:
            posted = post_points(points[indices])
            finite = np.isfinite(posted).all(axis=1)
            if finite.any():
                result, _, _ = _query_profile_masks(
                    executable, model, posted[finite], times, theta, masks, subpixel_layover, layover_vote_threshold)
                known = result[:, 1] != 255
                value[indices[finite][known], 1] = result[known, 1]
    return value, good, covered


def _query_profile_masks(executable, model, points, times, theta, masks, subpixel_layover, layover_vote_threshold):
    """将N×3 ECEF点映射到完整观测剖面，返回N×2类别、收敛与幅宽内标记。

    坐标为米/秒/弧度；类别为0可见、1遮挡、255未知。仅对连续坐标求解，
    默认采用最近邻。亚像元模式仅对四邻域均已知的叠掩按双线性权重表决。
    阴影及含未知邻点的叠掩使用最近邻。域外及失败保留255。
    """
    solved = solve_range_doppler_points(executable, model, points)
    t, angle, good = map_ecef_to_zero_doppler_profile(executable, model, points, solved)
    row_coord = np.where(good, (t - times[0]) / (times[1] - times[0]), -1)
    col_coord = np.where(good, (angle - theta[0]) / (theta[1] - theta[0]), -1)
    row = np.rint(row_coord).astype('int64')
    col = np.rint(col_coord).astype('int64')
    covered = solved[:, 5] == 0
    sample = covered & good & (row >= 0) & (row < len(times)) & (col >= 0) & (col < len(theta))
    value = np.full((len(points), 2), 255, dtype='uint8')
    value[sample] = masks[row[sample], col[sample]]
    if subpixel_layover:
        y = np.clip(row_coord[sample], 0, len(times) - 1)
        x = np.clip(col_coord[sample], 0, len(theta) - 1)
        r0, c0 = np.floor(y).astype('int64'), np.floor(x).astype('int64')
        r1, c1 = np.minimum(r0 + 1, len(times) - 1), np.minimum(c0 + 1, len(theta) - 1)
        top_left, top_right = masks[r0, c0, 1], masks[r0, c1, 1]
        bottom_left, bottom_right = masks[r1, c0, 1], masks[r1, c1, 1]
        known = (top_left < 2) & (top_right < 2) & (bottom_left < 2) & (bottom_right < 2)
        dy, dx = y - r0, x - c0
        support = ((1 - dy) * ((1 - dx) * top_left + dx * top_right)
                   + dy * ((1 - dx) * bottom_left + dx * bottom_right))
        value[np.flatnonzero(sample)[known], 1] = support[known] >= layover_vote_threshold
    return value, good, covered
