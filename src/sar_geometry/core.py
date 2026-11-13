"""独立C++核心的唯一Python调用接口；不含重复数学实现。"""
from pathlib import Path
import struct
import subprocess
import os
import numpy as np

PARAMETERS = ('wavelength','range0','range_spacing','time0','line_time','columns','rows',
              'look_side','doppler_mode','doppler_sign','max_iterations','tolerance_hz',
              'tolerance_pixels','max_step_seconds')
STATUS_NAMES = {0:'valid',1:'invalid_input',2:'orbit_bounds',3:'not_converged',
                4:'wrong_side',5:'doppler_bounds',6:'outside_radar'}
# 仅供资源监测读取；不参与数值结果。子进程退出后立即移除。
ACTIVE_CHILDREN=set()


def execute(executable, command, payload, count, columns):
    """二进制管道不传机器路径给C++，避免核心内的中文路径和编码差异。"""
    executable = Path(executable)
    if not executable.is_file():
        raise FileNotFoundError(f'请先构建独立核心：{executable}')
    env=os.environ.copy();env.setdefault('OMP_NUM_THREADS','4')
    with subprocess.Popen([str(executable),command],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env) as process:
        ACTIVE_CHILDREN.add(process.pid)
        try:stdout,stderr=process.communicate(payload)
        finally:ACTIVE_CHILDREN.discard(process.pid)
        if process.returncode:
            raise RuntimeError(stderr.decode('utf-8',errors='replace').strip())
    if len(stdout) != count*columns*8:
        raise RuntimeError('C++核心返回数据长度不完整')
    return np.frombuffer(stdout,dtype='<f8').reshape(count,columns).copy()


def solve_points(executable, model, points, normals=None):
    """地面ECEF米坐标→零起始雷达像元；法向缺失只使局部入射角未知。"""
    points = np.asarray(points,dtype='<f8')
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('points必须是N×3的ECEF坐标')
    normals = np.full_like(points,np.nan) if normals is None else np.asarray(normals,dtype='<f8')
    if normals.shape != points.shape:
        raise ValueError('normal和point尺寸不同')
    orbit = np.asarray(model['orbit'],dtype='<f8')
    doppler = np.asarray(model['doppler'],dtype='<f8')
    if orbit.ndim!=2 or orbit.shape[1]!=7 or doppler.ndim!=2 or doppler.shape[1]<2:
        raise ValueError('orbit应为N×7，doppler为M×(2+系数数)')
    payload = b''.join((struct.pack('<QQ',1,len(points)),
                       np.asarray([model[k] for k in PARAMETERS],dtype='<f8').tobytes(),
                       struct.pack('<Q',len(orbit)),orbit.tobytes(),
                       struct.pack('<QQ',len(doppler),doppler.shape[1]-2),doppler.tobytes(),
                       np.column_stack((points,normals)).astype('<f8').tobytes()))
    return execute(executable,'solve',payload,len(points),8)


def classify_profile(executable, points, support_complete, angle_epsilon=1e-10,range_epsilon=1e-6):
    """已构造观测剖面的实验接口；输入s/斜距米/向上仰角弧度/valid，输出阴影、叠掩。"""
    points = np.asarray(points,dtype='<f8')
    if points.ndim!=2 or points.shape[1]!=4:
        raise ValueError('profile必须为N×4，且保持地表顺序')
    payload = struct.pack('<QQddd',1,len(points),float(support_complete),angle_epsilon,range_epsilon)+points.tobytes()
    return execute(executable,'visibility',payload,len(points),2).astype('uint8')


def orbit_states(executable, orbit, times):
    """批量插值同一Hermite轨道，返回N×6的ECEF位置米/速度米每秒。"""
    orbit=np.asarray(orbit,dtype='<f8');times=np.asarray(times,dtype='<f8').ravel()
    payload=struct.pack('<QQQ',1,len(times),len(orbit))+orbit.tobytes()+times.tobytes()
    return execute(executable,'states',payload,len(times),6)


def classify_profiles(executable, points, bounds, angle_epsilon=1e-10, range_epsilon=1e-6):
    """完整方位剖面批处理。points行×列×4，bounds行×3；输出行×列×2的0/1/255。"""
    p=np.asarray(points,dtype='<f8');b=np.asarray(bounds,dtype='<f8')
    if p.ndim!=3 or p.shape[2]!=4 or b.shape!=(p.shape[0],3):
        raise ValueError('剖面或支持域边界尺寸不一致')
    rows,cols=p.shape[:2]
    payload=struct.pack('<QQQdd',1,rows*cols,cols,angle_epsilon,range_epsilon)+b.tobytes()+p.tobytes()
    return execute(executable,'profiles',payload,rows*cols,2).reshape(rows,cols,2).astype('uint8')
