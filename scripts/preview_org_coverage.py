"""独立输出一景ORG的裁剪前覆盖快视图；只读原图和ROI，不运行滤波/分割。"""
import json
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.features import geometry_mask, geometry_window
from pyproj import Transformer
from shapely.geometry import Polygon, mapping
from shapely.ops import transform as transform_geometry
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.transforms import Affine2D
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import matplotlib

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from configuration import load_yaml
from thres_segmentation.org_io import read_aoi, signature
from thres_segmentation.water_analysis import pixel_areas, area_sum

# 执行设置：只导出指定景，不扫描并批量生成全部影像。
CONFIG_PATH=Path(r'E:\Code_projects\sar_water_extraction\configs\北京密云区\threshold_segmentation.yaml')
SCENE_ID='BC5-SM-ORG-VV-20250730T142126-002563-000124-000A03-01'
OUTPUT_DIR=Path(r'E:\InSAR\水体提取\北京密云区\outputs\coverage_before_roi')
MAX_SIDE=1600
AMPLITUDE_DISPLAY_RANGE=(0,4096)  # 固定显示范围，不做自动百分位拉伸或归一化。


def draw_boundary(ax,geometry,color,linewidth=1.3,linestyle='-'):
    """支持多部件行政区及内环。"""
    polygons=[geometry] if geometry.geom_type=='Polygon' else geometry.geoms
    for polygon in polygons:
        for ring in [polygon.exterior,*polygon.interiors]:
            x,y=ring.xy;ax.plot(x,y,color=color,linewidth=linewidth,linestyle=linestyle)


def main():
    config=load_yaml(CONFIG_PATH);paths=config['paths']
    records=json.loads((paths['OUTPUT_DIR']/'inventory.json').read_text(encoding='utf-8'))
    record=next(r for r in records if r['scene_id']==SCENE_ID)
    source=Path(record['path'])
    if record['member']:
        # 优先复用已有的完整解压TIFF，不重复解压，也不读取已裁剪的幅度图作原图。
        extracted=paths['CACHE_DIR']/'extracted'/signature(record['source_stamp'])/Path(record['member']).name
        source=str(extracted) if extracted.is_file() else '/vsizip/'+source.as_posix()+'/'+record['member']
    aoi,area_crs=read_aoi(paths['AOI_PATH'],config['org']['output_crs'])
    print(f'原始未裁剪影像：{source}',flush=True)
    crop=paths['CACHE_DIR']/'scenes'/SCENE_ID/'amplitude.tif'
    with rasterio.open(source) as src:
        if not src.crs:raise ValueError('原影像缺少CRS，无法与ROI对照')
        to_native=Transformer.from_crs(area_crs,src.crs,always_xy=True).transform
        to_area=Transformer.from_crs(src.crs,area_crs,always_xy=True).transform
        roi=transform_geometry(to_native,aoi)
        extent=Polygon([src.transform*p for p in ((0,0),(src.width,0),(src.width,src.height),(0,src.height))])
        rectangular_coverage=transform_geometry(to_area,extent).intersection(aoi).area/aoi.area
        ratio=min(1,MAX_SIDE/max(src.shape));height=max(1,round(src.height*ratio));width=max(1,round(src.width*ratio))
        # 均值多视用于幅度预览；最近邻原始采样单独用于覆盖诊断，避免均值掩盖零值边界。
        image=src.read(1,out_shape=(height,width),masked=True,resampling=Resampling.average)
        sampled=src.read(1,out_shape=(height,width),masked=True,resampling=Resampling.nearest)
        transform=src.transform*Affine.scale(src.width/width,src.height/height)
        valid=~np.ma.getmaskarray(sampled)&np.isfinite(sampled.data)
        positive=valid&(sampled.data>0)
        inside=geometry_mask([mapping(roi)],out_shape=(height,width),transform=transform,invert=True)
        areas=pixel_areas(src.crs,transform,(height,width))
        positive_coverage=area_sum(positive&inside,areas)/aoi.area
        crop_outline=None;crop_check='没有裁剪缓存，仅展示原始覆盖'
        if crop.is_file():
            expected=geometry_window(src,[mapping(transform_geometry(to_native,aoi.buffer(config['org']['crop_buffer_m'])))])
            with rasterio.open(crop) as cut:
                ok=(cut.crs==src.crs and cut.shape==(int(expected.height),int(expected.width))
                    and cut.transform.almost_equals(src.window_transform(expected)))
                crop_check='裁剪网格与ROI外扩窗口一致' if ok else '裁剪网格与当前ROI外扩窗口不一致，需核查'
                crop_outline=Polygon([cut.transform*p for p in ((0,0),(cut.width,0),(cut.width,cut.height),(0,cut.height))])
                print(f'原尺寸={src.width}×{src.height}；裁剪后={cut.width}×{cut.height}；{crop_check}',flush=True)
        print(f'ROI面积={aoi.area/1e6:.3f} km²；原图矩形范围覆盖ROI={rectangular_coverage:.2%}',flush=True)
        print(f'非零正值信号覆盖ROI约={positive_coverage:.2%}（降采样估算，零值不直接判为NoData）；原NoData={src.nodata}',flush=True)
        native_crs=src.crs;native_shape=src.shape
    matplotlib.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus']=False
    fig=Figure(figsize=(15,9),facecolor='white');FigureCanvasAgg(fig)
    axes=fig.subplots(1,2)
    affine=Affine2D.from_values(transform.a,transform.d,transform.b,transform.e,transform.c,transform.f)
    axes[0].imshow(image,cmap='gray',vmin=AMPLITUDE_DISPLAY_RANGE[0],vmax=AMPLITUDE_DISPLAY_RANGE[1],
                   origin='upper',extent=(0,width,height,0),transform=affine+axes[0].transData,interpolation='nearest')
    axes[0].set_title(f'原始ORG幅度（多视快视图）\n固定{AMPLITUDE_DISPLAY_RANGE[0]}–{AMPLITUDE_DISPLAY_RANGE[1]}显示，无自动拉伸')
    codes=np.where(~valid,0,np.where(positive,2,1))
    axes[1].imshow(codes,cmap=ListedColormap(['#d9d9d9','#252525','#7cbea1']),vmin=0,vmax=2,
                   origin='upper',extent=(0,width,height,0),transform=affine+axes[1].transData,interpolation='nearest')
    axes[1].set_title('原始信号覆盖诊断（降采样）\n绿色：正值；黑色：有效非正值；灰色：掩膜/非有限值')
    combined=roi.union(extent).bounds;pad=max(combined[2]-combined[0],combined[3]-combined[1])*.04
    for ax in axes:
        draw_boundary(ax,roi,'#e12b32',1.6)
        draw_boundary(ax,extent,'#3b82f6',1.0,':')
        if crop_outline is not None:draw_boundary(ax,crop_outline,'#ffbb00',1.0,'--')
        ax.set_xlim(combined[0]-pad,combined[2]+pad);ax.set_ylim(combined[1]-pad,combined[3]+pad)
        ax.set_facecolor('#eceff3');ax.tick_params(labelsize=8)
        ax.set_aspect(1/np.cos(np.deg2rad((combined[1]+combined[3])/2)) if native_crs.is_geographic else 'equal')
        ax.set_xlabel('经度' if native_crs.is_geographic else 'X');ax.set_ylabel('纬度' if native_crs.is_geographic else 'Y')
    fig.legend(handles=[Line2D([0],[0],color='#e12b32',label='密云ROI边界'),
        Line2D([0],[0],color='#3b82f6',linestyle=':',label='原图矩形范围'),
        Line2D([0],[0],color='#ffbb00',linestyle='--',label='实际裁剪框')],loc='lower center',ncol=3,bbox_to_anchor=(.5,.10))
    fig.suptitle(f'{SCENE_ID}\n裁剪前覆盖核查｜原图 {native_shape[1]}×{native_shape[0]}｜{native_crs}',fontsize=12)
    fig.text(.5,.045,f'原图矩形范围覆盖ROI {rectangular_coverage:.2%}；非零正值信号覆盖约 {positive_coverage:.2%}\n'
             f'{crop_check}；未覆盖区域不是非水体。零值未声明NoData，不据此修改原图或提取掩膜。',ha='center',fontsize=10)
    fig.subplots_adjust(left=.06,right=.98,bottom=.18,top=.86,wspace=.18)
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    target=OUTPUT_DIR/(SCENE_ID+'_before_roi.png');fig.savefig(target,dpi=140)
    print(f'输出：{target}',flush=True)
    return target


if __name__=='__main__':main()
