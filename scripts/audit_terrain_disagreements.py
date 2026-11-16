"""密云地形差异核查：三维视线穿地检查、岸线分层及局部图件；生产成果只读。"""
from pathlib import Path
import sys,json,csv
from time import perf_counter
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from scipy.ndimage import label,distance_transform_edt,binary_erosion
from pyproj import Transformer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from sar_geometry.main import load_config
from sar_geometry.metadata import read_bc_metadata,build_bc_range_doppler_model
from sar_geometry.core import solve_range_doppler_points,interpolate_hermite_orbit_states
from sar_geometry.profiles import ElevationSampler,TO_GEODETIC
from sar_geometry.scene_validation import warped_array
from utils.files import write_json, sha256

CONFIG_PATH=Path(r'E:\Code_projects\sar_water_extraction\configs\北京密云区\sar_geometry_visibility_7p5m.yaml')
LOCAL_DIR=Path(r'E:\Code_projects\sar_water_extraction\codex_analysis\sar_geometry\gui_direction_corrected_025242')
OUTPUT_DIR=LOCAL_DIR/'regional_audit'
RAY_STEP_M=7.5  # 沿视线的距离采样；并非DEM真实分辨率。
CLEARANCE_TOLERANCE_M=1.0


def trace_sar_line_of_sight(points,states,sampler,step=RAY_STEP_M):
    """直线P→S每step米采样，检查椭球高是否低于DEM；返回独立遮挡证据。

    仅检查射线低于11000m的部分，此后高于本案例10000m地形上界。
    首个采样点从step开始，避免目标自身相交。缺测或1m以内擦边记未知。
    """
    results=[];curves=[]
    for p,state in zip(points,states):
        delta=state[:3]-p;direction=delta/np.linalg.norm(delta)
        # 本案例SAR入射角约40度，20km射线段足以升至11000m；逐例检查终点。
        d=np.arange(step,20000+step,step);q=p+d[:,None]*direction
        lon,lat,h=TO_GEODETIC.transform(q[:,0],q[:,1],q[:,2]);keep=h<11000
        terrain=sampler.sample(lon[keep],lat[keep]);h=h[keep];d=d[keep]
        gap=~np.isfinite(terrain);clearance=h-terrain
        k=int(np.nanargmin(clearance)) if np.isfinite(clearance).any() else 0
        minimum=float(clearance[k]) if np.isfinite(clearance).any() else None
        blocked=minimum is not None and minimum < -CLEARANCE_TOLERANCE_M
        known=(not gap.any()) and not keep[-1] and minimum is not None
        status='blocked' if blocked else ('clear' if known and minimum>CLEARANCE_TOLERANCE_M else 'uncertain')
        results.append(dict(ray_status=status,min_clearance_m=minimum,obstacle_distance_m=float(d[k]),
                            missing_samples=int(gap.sum())))
        curves.append((d,h,terrain))
    return results,curves


def main():
    """按完整差异集合核查阴影；岸线统计与图件均使用冻结的相同地理网格。"""
    started=perf_counter();OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    c=load_config(CONFIG_PATH);paths=c['paths'];profile=Path(paths['OUTPUT_DIR']);base=Path(paths['BASELINE_DIR'])
    watched=[base/'water.tif',base/'terrain_exclusion.tif',LOCAL_DIR/'terrain_exclusion.tif',profile/'terrain_exclusion.tif']
    hashes={str(p):sha256(p) for p in watched}
    with rasterio.open(profile/'data_cache/reference.tif') as ref:
        tr=ref.transform;crs=ref.crs;shape=ref.shape
        valid=(ref.read(1)==1)&(warped_array(profile/'radar_coverage.tif',ref,'uint8',255)==1)
        local={n:warped_array(LOCAL_DIR/(n+'.tif'),ref,'uint8',255) for n in ['shadow_mask','layover_mask','terrain_exclusion','terrain_class']}
        new={n:warped_array(profile/(n+'.tif'),ref,'uint8',255) for n in ['shadow_mask','layover_mask','terrain_exclusion']}
        history=warped_array(base/'history.tif',ref)==1
        water=warped_array(base/'water.tif',ref,'uint8',255)==1
        raw=warped_array(base/'raw.tif',ref,'uint8',255)==1
        slope=warped_array(paths['GUI_SLOPE_PATH'],ref);aspect=warped_array(paths['GUI_ASPECT_PATH'],ref)
        dem=warped_array(profile/'data_cache/dem_ellipsoid.tif',ref)
        sar=warped_array(paths['ORG_PATH'],ref)
    # 最大历史连通水域用于定位密云水库；其时间与当期不同，不作为真实岸线。
    components,count=label(history);sizes=np.bincount(components.ravel());sizes[0]=0
    reservoir=components==int(sizes.argmax())
    inside=distance_transform_edt(reservoir)*30;outside=distance_transform_edt(~reservoir)*30
    shore=(inside<=90)&(outside<=90)&valid
    core=(inside>90)&valid
    ground=(outside>90)&valid
    regions={'reservoir_reference_core':core,'reference_shore_plus_minus_90m':shore,'outside_reservoir':ground}
    report=dict(reference='Largest connected historical-water component; not current ground truth',zones={},ray_step_m=RAY_STEP_M)
    settings=json.loads((LOCAL_DIR/'terrain_geometry.json').read_text(encoding='utf-8'))
    bearing=settings['ground_to_sensor_azimuth_deg'];theta=settings['incidence_deg']
    phi=np.deg2rad((aspect-bearing+180)%360-180)
    range_slope=np.rad2deg(np.arctan(np.tan(np.deg2rad(slope))*np.cos(phi)))
    for zone,mask in regions.items():
        counts={}
        for n in new:
            good=mask&(local[n]<2)&(new[n]<2)
            counts[n]=dict(pixels=int(good.sum()),both=int((good&(local[n]==1)&(new[n]==1)).sum()),
                local_only=int((good&(local[n]==1)&(new[n]==0)).sum()),profile_only=int((good&(local[n]==0)&(new[n]==1)).sum()))
        counts['existing_water_pixels']=int((mask&water).sum())
        counts['local_exclusion_on_existing_water']=int((mask&water&(local['terrain_exclusion']==1)).sum())
        counts['profile_exclusion_on_existing_water']=int((mask&water&(new['terrain_exclusion']==1)).sum())
        counts['existing_water_outside_base_q10']=int((mask&water&~raw).sum())
        report['zones'][zone]=counts
    common=valid&(local['shadow_mask']<2)&(new['shadow_mask']<2)
    selected=common&((local['shadow_mask']==1)|(new['shadow_mask']==1))
    rr,cc=np.nonzero(selected)
    # 所有阴影并集像元均核查，另抽取最多48个岸线排除差异点，固定随机种子。
    bank=np.argwhere(shore&(local['terrain_exclusion']<2)&(new['terrain_exclusion']<2)&(local['terrain_exclusion']!=new['terrain_exclusion']))
    if len(bank)>48:bank=bank[np.random.default_rng(42).choice(len(bank),48,replace=False)]
    positions=np.vstack((np.column_stack((rr,cc)),bank));positions=np.unique(positions,axis=0)
    x=tr.c+(positions[:,1]+.5)*tr.a;y=tr.f+(positions[:,0]+.5)*tr.e
    to_ll=Transformer.from_crs(crs,4326,always_xy=True);lon,lat=to_ll.transform(x,y)
    sampler=ElevationSampler(profile/'data_cache/dem_ellipsoid.tif')
    h=sampler.sample(lon,lat);p=np.stack(Transformer.from_crs(4979,4978,always_xy=True).transform(lon,lat,h),axis=-1)
    meta=read_bc_metadata(paths['XML_PATH']);model=build_bc_range_doppler_model(meta,dict(c['model'],radar_grid_confirmed=True))
    solved=solve_range_doppler_points(paths['CORE_PATH'],model,p);states=interpolate_hermite_orbit_states(paths['CORE_PATH'],model['orbit'],solved[:,2])
    rows=[];curves=[]
    for start in range(0,len(p),32):
        values,lines=trace_sar_line_of_sight(p[start:start+32],states[start:start+32],sampler)
        curves.extend(lines)
        for j,v in enumerate(values,start):
            r,col=positions[j];a,b=int(local['shadow_mask'][r,col]),int(new['shadow_mask'][r,col])
            group='both_shadow' if a==b==1 else ('local_shadow_only' if a==1 else ('profile_shadow_only' if b==1 else 'shore_exclusion_difference'))
            rows.append(dict(id=j,row=int(r),col=int(col),longitude=float(lon[j]),latitude=float(lat[j]),group=group,
                 near_reference_shore=bool(shore[r,col]),reference_core=bool(core[r,col]),local_class=int(local['terrain_class'][r,col]),
                 local_shadow=a,profile_shadow=b,local_layover=int(local['layover_mask'][r,col]),profile_layover=int(new['layover_mask'][r,col]),
                 slope_deg=float(slope[r,col]),range_slope_deg=float(range_slope[r,col]),amplitude=float(sar[r,col]),**v))
        if start%128==0:print(f'三维视线核查 {min(start+32,len(p))}/{len(p)}',flush=True)
    sampler.close()
    report['ray_groups']={}
    for group in sorted({r['group'] for r in rows}):
        subset=[r for r in rows if r['group']==group]
        report['ray_groups'][group]={status:sum(r['ray_status']==status for r in subset) for status in ['blocked','clear','uncertain']}
    lay=valid&(local['layover_mask']==1)&(new['layover_mask']==0)
    report['local_only_layover']=dict(pixels=int(lay.sum()),range_slope_below_fixed_incidence=int((lay&(range_slope<theta)).sum()),
        range_slope_below_incidence_by_5deg=int((lay&(range_slope<theta-5)).sum()))
    shore_only=shore&(local['terrain_exclusion']==1)&(new['terrain_exclusion']==0)
    report['shore_local_only_classes']={str(int(v)):int(n) for v,n in zip(*np.unique(local['terrain_class'][shore_only],return_counts=True))}
    with (OUTPUT_DIR/'ray_points.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    # 各组挑有明确射线证据、空间分散的示例；图件只是证据展示，不冒充人工真值。
    examples=[]
    for group,status in [('profile_shadow_only','blocked'),('local_shadow_only','clear'),('local_shadow_only','blocked'),
                         ('shore_exclusion_difference','clear')]:
        candidates=[r for r in rows if r['group']==group and r['ray_status']==status]
        if group=='shore_exclusion_difference':candidates.sort(key=lambda v:-v['slope_deg'])
        else:candidates.sort(key=lambda v:v['min_clearance_m'] if status=='blocked' else -v['min_clearance_m'])
        for row in candidates:
            if all(np.hypot(row['row']-q['row'],row['col']-q['col'])>30 for q in examples):
                examples.append(row)
                if sum(q['group']==group and q['ray_status']==status for q in examples)>=2:break
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import LightSource,ListedColormap
    for number,row in enumerate(examples,1):
        r,col=row['row'],row['col'];r0,r1=max(0,r-45),min(shape[0],r+46);c0,c1=max(0,col-45),min(shape[1],col+46)
        sl=np.s_[r0:r1,c0:c1];point=(col-c0,r-r0)
        fig=Figure(figsize=(15,9));FigureCanvasAgg(fig);axes=fig.subplots(2,3)
        # 只为示例小窗读取5m SAR快视图；与其余30m面板保持同一空间范围。
        from rasterio.transform import from_origin
        native_transform=from_origin(tr.c+c0*30,tr.f-r0*30,5,5)
        with rasterio.open(paths['ORG_PATH']) as src, WarpedVRT(src,crs=crs,transform=native_transform,
                width=(c1-c0)*6,height=(r1-r0)*6,dtype='float32',nodata=np.nan,resampling=Resampling.average) as vrt:
            img=vrt.read(1)
        finite=img[np.isfinite(img)&(img>0)];lo,hi=np.percentile(finite,[2,98])
        axes[0,0].imshow(img,cmap='gray',vmin=lo,vmax=hi,extent=(-.5,c1-c0-.5,r1-r0-.5,-.5))
        axes[0,0].set_title('SAR amplitude P2-P98 (5m overview)')
        axes[0,1].imshow(LightSource(315,45).hillshade(dem[sl],dx=30,dy=30),cmap='gray')
        axes[0,1].contour(history[sl],levels=[.5],colors=['cyan'],linewidths=.5);axes[0,1].set_title('DEM hillshade / cyan historical water')
        for ax,n,title in [(axes[0,2],'terrain_exclusion','Exclusion'),(axes[1,0],'shadow_mask','Shadow'),(axes[1,1],'layover_mask','Layover')]:
            a,b=local[n][sl],new[n][sl];v=np.where((a<2)&(b<2),a+2*b,4)
            ax.imshow(v,cmap=ListedColormap(['white','orange','blue','purple','gray']),vmin=0,vmax=4)
            ax.set_title(title+': orange local / blue profile / purple both')
        for ax in axes.ravel()[:-1]:ax.plot(*point,'r+',markersize=9);ax.axis('off')
        d,ray,terrain=curves[row['id']];limit=max(500,min(5000,row['obstacle_distance_m']+500));use=d<=limit
        axes[1,2].plot(d[use],terrain[use],label='DEM ellipsoid height');axes[1,2].plot(d[use],ray[use],label='LOS to sensor')
        axes[1,2].set_xlabel('Distance from target toward sensor (m)');axes[1,2].set_ylabel('Height (m)');axes[1,2].legend()
        fig.suptitle(f'Case {number} | {row["group"]} | {row["longitude"]:.6f}, {row["latitude"]:.6f}\n'
                     f'Independent ray: {row["ray_status"]}, minimum clearance={row["min_clearance_m"]:.2f}m | slope={row["slope_deg"]:.1f}, range slope={row["range_slope_deg"]:.1f}')
        fig.tight_layout();fig.savefig(OUTPUT_DIR/f'case_{number:02}.png',dpi=120);fig.clear()
        row['figure']=f'case_{number:02}.png'
    report['examples']=examples;report['elapsed_seconds']=perf_counter()-started
    report['original_products_unchanged']=all(sha256(p)==hashes[str(p)] for p in watched)
    write_json(OUTPUT_DIR/'audit.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='examples'},ensure_ascii=False,indent=2),flush=True)
    return report


if __name__=='__main__':main()
