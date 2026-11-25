"""DFC Track 1 同期 VV 与地形的输入对照。裁片使用比赛定义的 10 m 网格。"""

import numpy as np
from affine import Affine

from datasets.relative_normalization import LOG_REPRESENTATION, apply_vv_train_p02_p98_dem_m


DFC_GRID_SPACING_M = 10.0
DFC_SLOPE_RADIUS = 3
DFC_TERRAIN_INPUTS = ("vv_only", "elevation", "slope", "elevation_slope")
DFC_TERRAIN_CHANNELS = ("VV", "DEM", "SLOPE")


def dfc_terrain_protocol(mode):
    """记录检查点可核对的物理单位、输入顺序、差分尺度和掩膜约定。"""
    if mode not in DFC_TERRAIN_INPUTS:
        raise ValueError(f"未知 DFC 地形输入 {mode!r}，可选 {DFC_TERRAIN_INPUTS}")
    return dict(version=1, mode=mode, channels=list(DFC_TERRAIN_CHANNELS),
                elevation_unit="m", slope_unit="degree", grid_spacing_m=DFC_GRID_SPACING_M,
                radius_pixels=DFC_SLOPE_RADIUS, dem_grid="DFC_Track1_resampled_Copernicus_DEM",
                slope_method="central_or_one_sided_finite_difference",
                valid_support="VV_DEM_valid_and_contiguous_axis_neighbours_in_both_axes",
                inactive_channels="zero")


def dfc_terrain_from_checkpoint(checkpoint):
    """读取新输入协议；旧权重继续使用原有通道和 VV 变换。"""
    protocol = checkpoint.get("dfc_terrain")
    if protocol is None:
        if "SLOPE" in (checkpoint.get("input_channels") or ()):
            raise ValueError("坡度通道检查点缺少 DFC 地形输入协议")
        return None
    if not isinstance(protocol, dict):
        raise ValueError("无效的 DFC 地形检查点协议")
    mode = protocol.get("mode")
    if protocol != dfc_terrain_protocol(mode):
        raise ValueError("检查点的 DFC 地形尺度、通道或掩膜协议不一致")
    if (checkpoint.get("patch_representation") != LOG_REPRESENTATION
            or tuple(checkpoint.get("bands") or ()) != (1, 4)
            or tuple(checkpoint.get("input_channels") or ()) != DFC_TERRAIN_CHANNELS
            or checkpoint.get("in_channels", 3) != 3
            or checkpoint.get("model_kwargs", {}).get("in_channels", 3) != 3):
        raise ValueError("DFC 地形检查点需要相对对数 VV、第 1/4 源波段和三个模型输入通道")
    return mode


def validate_dfc_terrain_grid(dataset):
    """无地理参考的原始 DFC 裁片按标称 10 m；有参考的栅格核对实际间距。"""
    transform = dataset.transform
    if dataset.crs is None and transform == Affine.identity():
        return
    if dataset.crs is None or not dataset.crs.is_projected:
        raise ValueError("DFC 坡度要求原始无地理参考裁片或 10 m 投影网格")
    metres = dataset.crs.linear_units_factor[1]
    axes = np.array([[transform.a, transform.d], [transform.b, transform.e]]) * metres
    if not np.allclose(axes @ axes.T, np.eye(2) * DFC_GRID_SPACING_M ** 2, rtol=1e-5, atol=1e-5):
        raise ValueError("DFC 坡度需要相互正交的 10 m 像元")


def _neighbours(valid, axis):
    """返回负、正方向从当前像元到差分端点之间的连续有效域。"""
    radius = DFC_SLOPE_RADIUS
    mask = np.moveaxis(valid, axis, -1)
    negative = mask.copy()
    positive = mask.copy()
    negative[:, :radius] = False
    positive[:, -radius:] = False
    for shift in range(1, radius + 1):
        negative[:, shift:] &= mask[:, :-shift]
        positive[:, :-shift] &= mask[:, shift:]
    return np.moveaxis(negative, -1, axis), np.moveaxis(positive, -1, axis)


def _gradient(height, axis, negative, positive):
    """双侧均有效时跨 60 m，单侧有效时从中心跨 30 m。"""
    radius = DFC_SLOPE_RADIUS
    z = np.moveaxis(height, axis, -1)
    padded = np.pad(z, ((0, 0), (radius, radius)))
    before = np.moveaxis(padded[:, :z.shape[1]], -1, axis)
    after = np.moveaxis(padded[:, 2 * radius:], -1, axis)
    rise = np.where(positive, after, height) - np.where(negative, before, height)
    run = (negative.astype(np.float32) + positive) * (radius * DFC_GRID_SPACING_M)
    return np.divide(rise, run, out=np.zeros_like(height), where=run > 0)


def _support(valid, neighbours):
    rows, cols = neighbours
    return valid & (rows[0] | rows[1]) & (cols[0] | cols[1])


def dfc_terrain_valid_mask(valid):
    valid = np.asarray(valid, dtype=bool)
    if valid.ndim != 2 or not valid.size:
        raise ValueError("DFC 地形需要非空的二维有效掩膜")
    return _support(valid, (_neighbours(valid, 0), _neighbours(valid, 1)))


def apply_dfc_terrain(image, valid, mode):
    """[VV, 米制高程] 转为同一有效域上的三个通道，VV 数值保持输入尺度。"""
    dfc_terrain_protocol(mode)
    image = np.asarray(image, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if image.ndim != 3 or image.shape[0] != 2 or image.shape[1:] != valid.shape or not valid.size:
        raise ValueError("DFC 地形需要非空 [2,H,W] 的 VV、高程及 [H,W] 有效掩膜")
    if not np.isfinite(image[:, valid]).all():
        raise ValueError("有效 VV 和高程必须为有限数")
    neighbours = (_neighbours(valid, 0), _neighbours(valid, 1))
    common = _support(valid, neighbours)
    result = np.zeros((len(DFC_TERRAIN_CHANNELS), *valid.shape), dtype=np.float32)
    result[0, common] = image[0, common]
    if mode in ("elevation", "elevation_slope"):
        result[1, common] = image[1, common]
    if mode in ("slope", "elevation_slope"):
        height = np.where(valid, image[1], 0.0)
        dy = _gradient(height, 0, *neighbours[0])
        dx = _gradient(height, 1, *neighbours[1])
        result[2, common] = np.degrees(np.arctan(np.hypot(dx[common], dy[common])))
    return result, common


def prepare_dfc_terrain(image, valid, mode, relative_normalization):
    """在训练和推理时使用同一地形有效域及训练集拟合的 VV 相对对数尺度。"""
    terrain, common = apply_dfc_terrain(image, valid, mode)
    terrain[0] = apply_vv_train_p02_p98_dem_m(image, common, relative_normalization)[0]
    return terrain, common
