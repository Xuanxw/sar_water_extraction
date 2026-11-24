"""独立C++核心的唯一Python调用接口；不含重复数学实现。"""

from pathlib import Path
import struct
import subprocess
import os
import numpy as np

PARAMETERS = (
    'wavelength',
    'range0',
    'range_spacing',
    'time0',
    'line_time',
    'columns',
    'rows',
    'look_side',
    'doppler_mode',
    'doppler_sign',
    'max_iterations',
    'tolerance_hz',
    'tolerance_pixels',
    'max_step_seconds',
)
STATUS_NAMES = {
    0: 'valid',
    1: 'invalid_input',
    2: 'orbit_bounds',
    3: 'not_converged',
    4: 'wrong_side',
    5: 'doppler_bounds',
    6: 'outside_radar',
}
# 仅供资源监测读取；不参与数值结果。子进程退出后立即移除。
ACTIVE_CHILDREN = set()


def call_geometry_core(executable, command, payload, count, columns):
    """二进制管道不传机器路径给C++，避免核心内的中文路径和编码差异。"""
    executable = Path(executable)
    if not executable.is_file():
        raise FileNotFoundError(f'请先构建独立核心：{executable}')
    env = os.environ.copy()
    env.setdefault('OMP_NUM_THREADS', '4')
    with subprocess.Popen(
        [str(executable), command],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    ) as process:
        ACTIVE_CHILDREN.add(process.pid)
        try:
            stdout, stderr = process.communicate(payload)
        finally:
            ACTIVE_CHILDREN.discard(process.pid)
        if process.returncode:
            raise RuntimeError(stderr.decode('utf-8', errors='replace').strip())
    if len(stdout) != count * columns * 8:
        raise RuntimeError('C++核心返回数据长度不完整')
    return np.frombuffer(stdout, dtype='<f8').reshape(count, columns).copy()


def solve_range_doppler_points(executable, model, points, normals=None):
    """求解地面点的距离—Doppler约束，输出零起始雷达像元坐标。

    Parameters: executable - C++几何核心路径。
                model      - 轨道、Doppler及雷达网格参数；单位米、秒、Hz。
                points     - (N, 3) ECEF地面位置，单位米。
                normals    - 可选(N, 3)地表法向；缺失时仅局部入射角为未知。
    Returns:    result     - (N, 8)数组：距离/方位像元、时间、Hz残差、迭代数、
                             状态码、局部入射角度、时间步长像元。状态定义见STATUS_NAMES。
    """
    points = np.asarray(points, dtype='<f8')
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('points必须是N×3的ECEF坐标')
    normals = np.full_like(points, np.nan) if normals is None else np.asarray(normals, dtype='<f8')
    if normals.shape != points.shape:
        raise ValueError('normal和point尺寸不同')
    orbit = np.asarray(model['orbit'], dtype='<f8')
    doppler = np.asarray(model['doppler'], dtype='<f8')
    if orbit.ndim != 2 or orbit.shape[1] != 7 or doppler.ndim != 2 or doppler.shape[1] < 2:
        raise ValueError('orbit应为N×7，doppler为M×(2+系数数)')
    payload = b''.join(
        (
            struct.pack('<QQ', 1, len(points)),
            np.asarray([model[k] for k in PARAMETERS], dtype='<f8').tobytes(),
            struct.pack('<Q', len(orbit)),
            orbit.tobytes(),
            struct.pack('<QQ', len(doppler), doppler.shape[1] - 2),
            doppler.tobytes(),
            np.column_stack((points, normals)).astype('<f8').tobytes(),
        )
    )
    return call_geometry_core(executable, 'solve', payload, len(points), 8)


def compute_radar_shadow_layover(
    executable, points, support_complete, angle_epsilon=1e-10, range_epsilon=1e-6
):
    """已构造观测剖面的实验接口；输入s/斜距米/向上仰角弧度/valid，输出阴影、叠掩。"""
    points = np.asarray(points, dtype='<f8')
    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError('profile必须为N×4，且保持地表顺序')
    payload = (
        struct.pack('<QQddd', 1, len(points), float(support_complete), angle_epsilon, range_epsilon)
        + points.tobytes()
    )
    return call_geometry_core(executable, 'visibility', payload, len(points), 2).astype('uint8')


def interpolate_hermite_orbit_states(executable, orbit, times):
    """批量插值同一Hermite轨道，返回N×6的ECEF位置米/速度米每秒。"""
    orbit = np.asarray(orbit, dtype='<f8')
    times = np.asarray(times, dtype='<f8').ravel()
    payload = struct.pack('<QQQ', 1, len(times), len(orbit)) + orbit.tobytes() + times.tobytes()
    return call_geometry_core(executable, 'states', payload, len(times), 6)


def scan_gc_map2_v57_shadow_layover(executable, angle, slant_range, direction):
    """复现v5.7内部float32扫描，返回行×列的0/1/4/16/20字节。

    angle、slant_range是已经重排到同一内部网格的二维角度场和斜距米；0为目标
    内部缺测哨兵，NaN/Inf不支持。direction为该角度场的有向比较系数±1，不能
    根据左右视猜填。本接口只还原扫描，不把现有剖面构造宣称为GAMMA的内部网格。
    """
    angle = np.asarray(angle, dtype='float32')
    slant_range = np.asarray(slant_range, dtype='float32')
    if angle.ndim != 2 or angle.shape != slant_range.shape or angle.shape[1] == 0:
        raise ValueError('angle和slant_range必须是同尺寸的二维完整横向剖面')
    if direction not in (-1, 1) or not np.isfinite(angle).all() or not np.isfinite(slant_range).all():
        raise ValueError('扫描需要有限float32值及明确的direction=±1；0沿用内部缺测约定')
    columns = angle.shape[1]
    payload = (
        struct.pack('<QQQd', 1, angle.size, columns, direction)
        + angle.astype('<f8').tobytes()
        + slant_range.astype('<f8').tobytes()
    )
    return call_geometry_core(executable, 'gamma57_scan', payload, angle.size, 1).reshape(angle.shape).astype('uint8')


def compute_radar_shadow_layover_batch(executable, points, bounds, angle_epsilon=1e-10, range_epsilon=1e-6):
    """完整方位剖面批处理。points行×列×4，bounds行×3；输出行×列×2的0/1/255。"""
    profile_points = np.asarray(points, dtype='<f8')
    support_bounds = np.asarray(bounds, dtype='<f8')
    if (
        profile_points.ndim != 3
        or profile_points.shape[2] != 4
        or support_bounds.shape != (profile_points.shape[0], 3)
    ):
        raise ValueError('剖面或支持域边界尺寸不一致')
    rows, cols = profile_points.shape[:2]
    payload = (
        struct.pack('<QQQdd', 1, rows * cols, cols, angle_epsilon, range_epsilon)
        + support_bounds.tobytes()
        + profile_points.tobytes()
    )
    return (
        call_geometry_core(executable, 'profiles', payload, rows * cols, 2)
        .reshape(rows, cols, 2)
        .astype('uint8')
    )
