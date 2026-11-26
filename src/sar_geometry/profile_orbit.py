"""Zero-Doppler orbital body frame for BC data, WGS84, ECEF, metres and seconds."""
import numpy as np
from scipy.interpolate import CubicSpline, CubicHermiteSpline


def geodetic_up(pos):
    """Outward ellipsoid normal at the subsatellite geodetic latitude."""
    a,e2=6378137.,0.00669437999014
    x,y,z=np.moveaxis(np.asarray(pos,dtype='f8'),-1,0)
    rho=np.hypot(x,y)
    lat=np.arctan2(z,rho*(1.-e2))
    for _ in range(4):
        s=np.sin(lat)
        n=a/np.sqrt(1.-e2*s*s)
        lat=np.arctan2(z+e2*n*s,rho)
    lon=np.arctan2(y,x)
    return np.stack((np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)),axis=-1)


def body_fields(xyz,state,look_side):
    """Return positive cross-track distance, positive off-nadir radians, range.

    State has the last axis (satellite ECEF position, satellite ECEF velocity).
    look_side = +1 right or -1 left. The passed state is at the point's
    azimuth acquisition time. Returned angle is atan2(Yb,Zb) made positive
    on the imaged side; multiply radians by 180/pi before casting scan samples.
    """
    p=np.asarray(state,dtype='f8')[...,:3]
    v=np.asarray(state,dtype='f8')[...,3:]
    along=v/np.linalg.norm(v,axis=-1,keepdims=True)
    right=np.cross(along,geodetic_up(p))
    right/=np.linalg.norm(right,axis=-1,keepdims=True)
    down=np.cross(along,right)
    los=np.asarray(xyz,dtype='f8')-p
    y=look_side*np.einsum('...i,...i->...',los,right)
    z=np.einsum('...i,...i->...',los,down)
    return y,np.arctan2(y,z),np.linalg.norm(los,axis=-1)


class SixChannelOrbit:
    def __init__(self, state):
        raw=np.asarray(state,dtype='f8')
        t=raw[:,0].copy()
        n=len(t)
        if n<=31:
            raise ValueError('The tested BC MLI convention uses more than 31 state vectors')
        t=t[0]+np.arange(n)*round((t[-1]-t[0])/(n-1),6)
        self.t=t
        self.pos=np.round(raw[:,1:4],4)
        self.vel=np.round(raw[:,4:7],5)
        # On equally spaced nodes, the interpolating cardinal cubic with mirror
        # extension has zero slope at each endpoint.
        bc=((1,np.zeros(6)),(1,np.zeros(6)))
        self.spline=CubicSpline(t,np.column_stack((self.pos,self.vel)),bc_type=bc)

    def state(self,t):
        t=np.asarray(t)
        tc=np.clip(t,self.t[0],self.t[-1])
        sampled=self.spline(tc)
        ds=self.spline(tc,1)
        p,v=sampled[...,:3],sampled[...,3:]
        # Linear extrapolation supplies boundary support outside the observed
        # arc. The image swath remains restricted to measured acquisition times.
        p=p+(t-tc)[...,None]*v
        dp=np.where((t==tc)[...,None],ds[...,:3],v)
        return p,v,dp,ds[...,3:]


class HermiteOrbit:
    def __init__(self,state):
        self.t=state[:,0]
        self.spline=CubicHermiteSpline(self.t,state[:,1:4],state[:,4:7])

    def state(self,t):
        v=self.spline(t,1)
        return self.spline(t),v,v,self.spline(t,2)


def solve_hermite_geometry(xyz,orbit,t_center):
    """Evaluate the physical profile's five-step zero-Doppler Newton solve."""
    time=np.full(len(xyz),float(t_center))
    dot=lambda a,b: np.einsum('ij,ij->i',a,b)
    for _ in range(5):
        pos,vel,_,acc=orbit.state(time)
        los=xyz-pos
        delta=dot(vel,los)/(dot(vel,vel)-dot(acc,los))
        if np.nanmax(np.abs(delta))<1e-10:
            break
        time+=delta
    pos,vel,_,_=orbit.state(time)
    return time,pos,vel


def solve_xyz(xyz,orbit,t_center):
    """Zero-Doppler time and interpolated ECEF position and velocity per point."""
    xyz=np.asarray(xyz,dtype='f8')
    t=np.full(xyz.shape[:-1],float(t_center))
    dot=lambda a,b: np.einsum('...i,...i->...',a,b)
    for _ in range(12):
        p,v,dp,dv=orbit.state(t)
        los=xyz-p
        delta=dot(v,los)/(dot(v,dp)-dot(dv,los))
        # Mirror-extended B-splines may have a zero endpoint derivative.
        delta=np.clip(np.where(np.isfinite(delta),delta,0),-2.,2.)
        t+=delta
        if np.max(np.abs(delta))<1e-9:
            break
    p,v,_,_=orbit.state(t)
    return t,p,v
