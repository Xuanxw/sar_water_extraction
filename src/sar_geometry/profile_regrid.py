"""Independent equal-cross-track profiling, for native co-registered DEM geometry.

No reference classification or GAMMA lookup tables enter these functions.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, spline_filter


@dataclass(frozen=True)
class Grid:
    az_lines: int
    radar_width: int
    columns: int
    y0: float
    dy: float


def load_kernel(library_path):
    kernel=ctypes.CDLL(str(Path(library_path).resolve())).gather
    f64=np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    f32=np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
    u8=np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS")
    kernel.argtypes=[f64]*4+[u8]+[ctypes.c_int]*4+[f64]*2+[ctypes.c_int]*6+[ctypes.c_double]*2+[f32]*2
    kernel.restype=None
    return kernel


def _design(a, k, h, w):
    a=2.*a/max(h-1,1)-1.
    k=2.*k/max(w-1,1)-1.
    return np.column_stack([np.ones_like(k),a,k,a*a,a*k,k*k,a**3,a*a*k,a*k*k,k**3])


def derive_grid(fields, mli_shape, spacing_m, r_ovr=.8):
    """从幅宽内 DEM 节点的横轨距离（m）构造逐多视方位行的剖面网格。"""
    lines, width = mli_shape
    low,high=np.inf,-np.inf
    az, rg, yb = (fields[k] for k in ('az','rg','yb'))
    for row in range(0,len(az),128):
        s=slice(row,row+128)
        # Extrema apply swath checks to FLOAT LUT coordinates in the reference implementation.
        r=rg[s].astype('f4',copy=False)
        a=az[s].astype('f4',copy=False)
        ok=(a>=0)&(a<lines)&(r>=0)&(r<width)&np.isfinite(yb[s])
        if np.any(ok):
            low=min(low,float(np.min(yb[s],where=ok,initial=np.inf)))
            high=max(high,float(np.max(yb[s],where=ok,initial=-np.inf)))
    if not np.isfinite(low) or low>=high:
        raise ValueError('empty or degenerate swath coordinates')
    margin=(high-low)/50.
    step=spacing_m/r_ovr
    return Grid(lines,width,int(np.ceil((high-low+2*margin)/step)), low-margin,step)


def _inverse_seed(fields, grid):
    """拟合从内部横向列、方位行到 DEM 行列的迭代初值，不参与最终分类。"""
    # The fit supplies an initial guess only. The inverse coordinate fields decide
    # the accepted position and the classification.
    stride=max(1,min(fields['az'].shape)//96)
    a=fields['az'][::stride,::stride]
    y=fields['yb'][::stride,::stride]
    idx=np.nonzero(np.isfinite(a)&np.isfinite(y)&(y!=0))
    k=(y[idx]-grid.y0)/grid.dy
    design=_design(a[idx].astype('f8'),k.astype('f8'),grid.az_lines,grid.columns)
    coords=np.column_stack(idx).astype('f8')*stride
    params=np.linalg.lstsq(design,coords,rcond=None)[0]
    resid=np.max(np.abs(design@params-coords),axis=0)
    return params,np.maximum(np.ceil(resid).astype('i4')+12,16)


def _coefficients(fields, s):
    result=[]
    valid=np.ones(fields['az'][s].shape,'u1')
    for name in ('az','yb','rg','look'):
        val=fields[name][s]
        gaps=~np.isfinite(val)
        if np.all(gaps):
            return None
        if np.any(gaps):
            valid[gaps]=0
            # Filled edge support only. Reject inverse roots at invalid source nodes below.
            idx=distance_transform_edt(gaps,return_distances=False,return_indices=True)
            val=val[tuple(idx)]
        if name=='look':
            # Store signed/side-normalized angle in degrees on the internal grid,
            # so a float32 tie has the same physical width as in the scanner.
            val=np.rad2deg(val.astype('f8',copy=False))
        result.append(spline_filter(val,order=2,mode='mirror'))
    return [*result,valid]


def build_profile_mask(fields,mli_shape,spacing_m,scan,kernel,r_ovr=.8,grid=None,block_rows=128,
                       block_cols=768,on_progress=None):
    """Return raw scanner bit codes on an (azimuth, cross-track) grid, and the grid.

    fields has full DEM-node arrays az and rg (continuous MLI indices), yb
    (positive transverse LOS component in m), look (positive off-nadir radians).
    scan(angle_f32,range_index_f32) returns uint8 codes on a full-width stripe.
    反算可以横向分块，折返扫描必须贯穿整行。先拼齐各块的连续几何量，
    再一次扫描，保证长折返两端落在不同块时仍能成对识别。
    """
    if set(fields)!={'az','yb','rg','look'} or len({v.shape for v in fields.values()})!=1:
        raise ValueError('expected four co-registered native fields')
    grid=derive_grid(fields,mli_shape,spacing_m,r_ovr) if grid is None else grid
    params,margin=_inverse_seed(fields,grid)
    height,width=fields['az'].shape
    result=np.zeros((grid.az_lines,grid.columns),np.uint8)
    p,q=(np.ascontiguousarray(params[:,i]) for i in (0,1))
    for a0 in range(0,grid.az_lines,block_rows):
        n=min(grid.az_lines-a0,block_rows)
        rad=np.zeros((n,grid.columns),'f4')
        angle=np.zeros_like(rad)
        for k0 in range(0,grid.columns,block_cols):
            m=min(grid.columns-k0,block_cols)
            aa,kk=np.meshgrid(np.linspace(a0,a0+n-1,5),np.linspace(k0,k0+m-1,5),indexing='ij')
            rc=_design(aa.ravel(),kk.ravel(),grid.az_lines,grid.columns)@params
            top,left=np.maximum(0,np.floor(rc.min(axis=0)).astype(int)-margin)
            bot,right=np.minimum((height,width),np.ceil(rc.max(axis=0)).astype(int)+margin+1)
            if bot<=top+2 or right<=left+2:
                continue
            s=np.s_[top:bot,left:right]
            coeffs=_coefficients(fields,s)
            if coeffs is None:
                continue
            rb=np.zeros((n,m),'f4')
            ab=np.zeros_like(rb)
            kernel(*coeffs,bot-top,right-left,top,left,p,q,
                   a0,k0,n,m,grid.az_lines,grid.columns,grid.y0,grid.dy,rb,ab)
            rad[:,k0:k0+m]=rb
            angle[:,k0:k0+m]=ab
        result[a0:a0+n]=scan(angle,rad)
        if on_progress:
            on_progress(a0+n,grid.az_lines)
    return result,grid


def map_to_dem(fields,mask,grid,out=None,rows=128):
    """Nearest internal class at native DEM nodes, with rounded MLI swath checks."""
    if out is None:
        out=np.zeros(fields['az'].shape,'u1')
    for r0 in range(0,len(out),rows):
        s=slice(r0,r0+rows)
        a,y,r=(fields[k][s] for k in ('az','yb','rg'))
        k=(y-grid.y0)/grid.dy
        ok=(a>-.5)&(a<grid.az_lines-.5)&(r>-.5)&(r<grid.radar_width-.5)&(k>-.5)&(k<grid.columns-.5)
        tile=out[s]
        tile[:]=0
        tile[ok]=mask[np.floor(a[ok]+.5).astype(np.intp),np.floor(k[ok]+.5).astype(np.intp)]
    return out
