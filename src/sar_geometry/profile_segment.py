"""Find the native segment from source DEM and BC zero-Doppler stripmap geometry."""
import numpy as np
from scipy.interpolate import CubicHermiteSpline
from pyproj import Transformer
import rasterio


def source_segment(source, grid, states, looks=(3,3), frame=8):
    """Return source window, rough search window, and terrain elevation bounds.

    grid/states are the grid and orbit arrays returned by read_bc_metadata.
    looks is (azimuth, range). The imaged azimuth is cropped to complete looks
    supported by the state vectors, matching the configured BC workflow.
    """
    az_looks, rg_looks = looks
    t0, dt = grid['time0'], grid['line_time']
    if t0 < states[0,0]:
        raise ValueError('First radar line precedes orbit support')
    nz = min(grid['rows'],int(np.floor((states[-1,0]-t0)/dt))+1)//az_looks
    nx = grid['columns']//rg_looks
    if nz < 1 or nx < 1:
        raise ValueError('No complete multilook cells inside the orbit interval')
    start = t0+(az_looks-1)*.5*dt
    times = np.linspace(start,start+(nz-1)*az_looks*dt,100)
    near = grid['range0']+(rg_looks-1)*.5*grid['range_spacing']
    far = near+(nx-1)*rg_looks*grid['range_spacing']
    spline = CubicHermiteSpline(states[:,0],states[:,1:4],states[:,4:7],axis=0)
    sat, vel = spline(times),spline(times,1)
    unit = lambda x: x/np.linalg.norm(x,axis=1)[:,None]
    v = unit(vel)
    nadir = unit(-sat + v*np.sum(v*sat,axis=1)[:,None])
    cross = grid['look_side']*np.cross(nadir,v)
    geodetic = Transformer.from_crs(4978,4979,always_xy=True)

    def geolocate(distance, height):
        lo, hi = np.zeros(len(times)), np.full(len(times),1.5)
        for _ in range(44):
            angle = (lo+hi)/2
            ray = np.cos(angle)[:,None]*nadir + np.sin(angle)[:,None]*cross
            lon, lat, h = geodetic.transform(*(sat+distance*ray).T)
            lo = np.where(h < height,angle,lo)
            hi = np.where(h >= height,angle,hi)
        return lon,lat

    with rasterio.open(source) as ds:
        if ds.crs.to_epsg()!=4326 or ds.transform.b or ds.transform.d:
            raise ValueError('Unrotated geographic WGS 84 source required')

        def box(min_z,max_z,pad):
            near_lon,near_lat=geolocate(near,min_z)
            far_lon,far_lat=geolocate(far,max_z)
            lon=np.concatenate([near_lon,far_lon]);lat=np.concatenate([near_lat,far_lat])
            c,r=(~ds.transform)*(lon,lat)
            c0=max(0,int(np.floor(c.min()-.5))-pad)
            r0=max(0,int(np.floor(r.min()-.5))-pad)
            c1=min(ds.width-1,int(np.ceil(c.max()-.5))+pad)
            r1=min(ds.height-1,int(np.ceil(r.max()-.5))+pad)
            return rasterio.windows.Window(c0,r0,c1-c0+1,r1-r0+1)

        rough = box(-500.,9000.,frame+8)
        a=ds.read(1,window=rough,masked=True)
        a=np.ma.masked_invalid(a)
        if a.count()==0:
            raise ValueError('No valid height in the imaged terrain bounds')
        extrema=(float(a.min()),float(a.max()))
        return box(*extrema,frame),rough,extrema
