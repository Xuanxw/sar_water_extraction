"""零Doppler观测剖面：椭球/ECEF构造、DEM双线性采样和有界支持域。

本模块没有将DEM行当成雷达方向，也不按斜距排序。每个时刻的地面点满足
V·(P-S)=0；地表角theta按近侧到远侧递增。非零Doppler斜视需要另一种曲面。
"""
import numpy as np
from pyproj import Transformer
from scipy.ndimage import map_coordinates
import rasterio
from rasterio.windows import Window
from .core import orbit_states, classify_profiles

A=6378137.0
B=6356752.314245179
TO_GEODETIC=Transformer.from_crs(4978,4979,always_xy=True)


def observation_frame(states, look_side):
    """ECEF状态→零Doppler平面原点及径向/横向基，全部单位米。

    平面原点O是地心到V·P=V·S的垂足；e0朝卫星，e1朝指定观测侧。
    因而P=O+rho*(cos(theta)*e0+sin(theta)*e1)严格位于该平面。
    """
    s,v=states[...,:3],states[...,3:]
    v=v/np.linalg.norm(v,axis=-1,keepdims=True)
    o=np.sum(s*v,axis=-1,keepdims=True)*v
    e0=s-o;e0=e0/np.linalg.norm(e0,axis=-1,keepdims=True)
    e1=look_side*np.cross(v,e0)
    return o,e0,e1


def ellipsoid_radius(origin, direction):
    """射线O+rho*d与WGS84椭球的正根；用于高程迭代初值，不用球体替代椭球。"""
    axes=np.array([A,A,B]);o=origin/axes;d=direction/axes
    aa=np.sum(d*d,axis=-1);bb=2*np.sum(o*d,axis=-1);cc=np.sum(o*o,axis=-1)-1
    disc=bb*bb-4*aa*cc
    return (-bb+np.sqrt(np.maximum(disc,0)))/(2*aa)


def surface_at_height(origin, direction, height):
    """固定椭球高曲面与平面径向射线求交，用于未观测地形的上下包络。

    使用PROJ的椭球高，沿射线迭代；这里的rho修正只作数值迭代，最终检查高程残差。
    """
    rho=ellipsoid_radius(origin,direction)+height
    for _ in range(5):
        p=origin+rho[...,None]*direction
        _,_,h=TO_GEODETIC.transform(p[...,0],p[...,1],p[...,2])
        rho+=height-h
    return origin+rho[...,None]*direction


class ElevationSampler:
    """按请求包围窗读取DEM，缓存有限个GDAL块；不把多瓦片DEM整体载入内存。

    输入经纬度，输出椭球高米；双线性四邻域中任一无效即返回NaN，不跨洞填值。
    DEM须已完成明确的高程基准转换，read阶段不重复归一化。
    """
    def __init__(self, path):
        self.src=rasterio.open(path)
        self.transformer=Transformer.from_crs(4326,self.src.crs,always_xy=True)

    def close(self):
        self.src.close()

    def sample(self, lon, lat):
        x,y=self.transformer.transform(lon,lat)
        inv=~self.src.transform
        col=inv.a*x+inv.b*y+inv.c-.5;row=inv.d*x+inv.e*y+inv.f-.5
        finite=np.isfinite(col)&np.isfinite(row)
        good=finite&(col>=0)&(row>=0)&(col<self.src.width-1)&(row<self.src.height-1)
        result=np.full(np.shape(lon),np.nan)
        if not good.any():return result
        c0=int(np.floor(col[good].min()));r0=int(np.floor(row[good].min()))
        c1=int(np.ceil(col[good].max()))+2;r1=int(np.ceil(row[good].max()))+2
        win=Window(c0,r0,min(c1,self.src.width)-c0,min(r1,self.src.height)-r0)
        data=self.src.read(1,window=win,masked=True,out_dtype='float64').filled(np.nan)
        data=data*self.src.scales[0]+self.src.offsets[0]
        result[good]=map_coordinates(data,[row[good]-r0,col[good]-c0],order=1,prefilter=False,mode='constant',cval=np.nan)
        return result


def build_profiles(executable, model, sampler, times, theta, height_bounds, tolerance_m=.05):
    """生成完整横向剖面并批量分类，返回掩膜/位置/有效标记及数值残差。

    方位行可分批，横向列保持完整。未知外围地形采用配置的全域椭球高上/下界；
    只有超出这些边界可能产生的遮挡/同距离范围，才能确定为可见或无叠掩。
    """
    if model['doppler_mode']!=0:
        raise ValueError('观测剖面当前仅支持显式零Doppler；不能将斜视模型当零Doppler处理')
    state=orbit_states(executable,model['orbit'],times)
    o,e0,e1=observation_frame(state,model['look_side'])
    direction=np.cos(theta)[None,:,None]*e0[:,None]+np.sin(theta)[None,:,None]*e1[:,None]
    rho=ellipsoid_radius(o[:,None],direction)
    valid=np.ones(rho.shape,bool);residual=np.full(rho.shape,np.nan)
    # NumPy一次处理多条剖面；DEM只读覆盖这些剖面的小窗。
    for _ in range(10):
        p=o[:,None]+rho[...,None]*direction
        lon,lat,h=TO_GEODETIC.transform(p[...,0],p[...,1],p[...,2])
        terrain=sampler.sample(lon,lat)
        valid &= np.isfinite(terrain)
        residual=terrain-h
        if np.any(valid & ((terrain<height_bounds[0])|(terrain>height_bounds[1]))):
            raise ValueError('DEM超出配置高程包络，不能继续使用该包络判断支持域')
        if not np.any(valid & (np.abs(residual)>tolerance_m)):break
        rho+=np.where(valid,residual,0)
    valid &= np.abs(residual)<=tolerance_m
    l=p-state[:,None,:3]
    ranges=np.linalg.norm(l,axis=-1)
    angles=np.arctan2(np.sum(l*e0[:,None],axis=-1),np.sum(l*e1[:,None],axis=-1))
    points=np.stack((np.broadcast_to(theta,ranges.shape)*A,ranges,angles,valid.astype(float)),axis=-1)
    # theta限制在近地可见单调区；不将地球背面或过地平线区域混入此前包络。
    if theta[0]<=0 or theta[-1]>=.2:
        raise ValueError('首版剖面支持0<theta<0.2弧度；超出时需扩展几何边界模型')
    # DEM矩形边缘与斜向剖面相交位置各异；缺测首尾用包络表达，不冒充已知地形。
    first=np.argmax(valid,axis=1);last=valid.shape[1]-1-np.argmax(valid[:,::-1],axis=1)
    near_dir=direction[np.arange(len(times)),first];far_dir=direction[np.arange(len(times)),last]
    near_hi=surface_at_height(o,near_dir,height_bounds[1])-state[:,:3]
    near_lo=surface_at_height(o,near_dir,height_bounds[0])-state[:,:3]
    far_hi=surface_at_height(o,far_dir,height_bounds[1])-state[:,:3]
    bounds=np.column_stack((np.arctan2(np.sum(near_hi*e0,axis=-1),np.sum(near_hi*e1,axis=-1)),
                            np.linalg.norm(near_lo,axis=-1),np.linalg.norm(far_hi,axis=-1)))
    masks=classify_profiles(executable,points,bounds)
    error=float(np.max(np.abs(residual[valid]))) if valid.any() else None
    return masks,p,valid,error


def profile_coordinates(executable, model, points, solved):
    """把已解得时刻的DEM点映射到(t,theta)，不以斜距替代有折返的地表横向坐标。"""
    good=np.isin(solved[:,5],(0,6));theta=np.full(len(points),np.nan)
    if good.any():
        state=orbit_states(executable,model['orbit'],solved[good,2])
        o,e0,e1=observation_frame(state,model['look_side']);d=points[good]-o
        theta[good]=np.arctan2(np.sum(d*e1,axis=-1),np.sum(d*e0,axis=-1))
    return solved[:,2],theta,good
