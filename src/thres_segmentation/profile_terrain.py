"""将独立观测剖面掩膜供阈值流程消费，路径和科学参数由案例YAML显式提供。"""
from pathlib import Path
import json
from contextlib import ExitStack
import numpy as np
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.windows import Window
from sar_geometry.scene import run_zero_doppler_terrain_visibility
from sar_geometry.metadata import read_bc_metadata
from utils.files import write_json

PROFILE_METHOD_VERSION='radar_profile_zero_doppler_v2'


def align_profile_masks(reference, geometry_dir, output, block_size=256):
    """分类最近邻对齐到SAR原网格；255保留未知，不把未知转换成排除或陆地。

    只改变网格，不增加30m DEM的真实信息。阴影和叠掩各自输出，联合掩膜由独立核心定义。
    """
    output=Path(output);geometry_dir=Path(geometry_dir)
    products={name:output/(name+'.tif') for name in ('shadow_mask','layover_mask','terrain_exclusion')}
    with ExitStack() as stack:
        src=stack.enter_context(rasterio.open(reference));profile=src.profile.copy()
        profile.update(driver='GTiff',count=1,dtype='uint8',nodata=255,compress='lzw',BIGTIFF='IF_SAFER')
        readers={};writers={}
        for name,path in products.items():
            base=stack.enter_context(rasterio.open(geometry_dir/(name+'.tif')))
            readers[name]=stack.enter_context(WarpedVRT(base,crs=src.crs,transform=src.transform,width=src.width,height=src.height,
                dtype='uint8',nodata=255,resampling=Resampling.nearest))
            writers[name]=stack.enter_context(rasterio.open(path.with_suffix('.partial.tif'),'w',**profile))
        for y in range(0,src.height,block_size):
            win=Window(0,y,src.width,min(block_size,src.height-y));valid=src.read_masks(1,window=win)>0
            for name in products:
                value=readers[name].read(1,window=win)
                writers[name].write(np.where(valid,value,255).astype('uint8'),1,window=win)
    for p in products.values():p.with_suffix('.partial.tif').replace(p)
    return products


def generate_profile_terrain(reference, config, folder):
    """按本景ORG注释中的原雷达网格计算候选掩膜；缺项报错，不退回局部坡度。

    只生成地形文件。阈值/历史水体/矢量处理仍由原主流程执行，30m未知域不作硬剔除。
    """
    folder=Path(folder);paths=config['paths'];options=config['org']['profile_geometry']
    metadata=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
    # XML已由本景输入解析流程保存在成果目录，避免依赖原ZIP是否仍在原位置。
    identifier=Path(metadata['source_image']).stem
    annotation=read_bc_metadata(folder/'source.xml',xml_member=identifier+'.xml')
    geometry_output=Path(paths['PROFILE_GEOMETRY_CACHE_DIR'])/identifier
    settings=dict(options['model'])
    geometry=dict(paths=dict(XML_PATH=folder/'source.xml',ORG_PATH=Path(reference),DEM_TILES=paths['PROFILE_DEM_TILES'],
        GEOID_PATH=options['geoid_url'],OUTPUT_DIR=geometry_output,CORE_PATH=paths['GEOMETRY_CORE_PATH']),
        model=settings,scene=dict(options['scene']),execution=dict(options['execution']))
    if not settings['radar_grid_confirmed'] and not geometry['scene'].get('allow_unconfirmed_geometry'):
        raise ValueError('必须明确接受并记录本地区零Doppler候选模型的假设')
    print('观测剖面地形缓存：'+str(geometry_output),flush=True)
    # 控制GDAL缓存并保留阶段日志，整条横向剖面不拆分。
    log_file=geometry_output/'processing.log';geometry_output.mkdir(parents=True,exist_ok=True)
    def log(message):
        print(message,flush=True)
        with log_file.open('a',encoding='utf-8') as f:f.write(message+'\n')
    with rasterio.Env(GDAL_CACHEMAX=64*1024*1024):report=run_zero_doppler_terrain_visibility(geometry,annotation,log)
    products=align_profile_masks(reference,geometry_output,folder,config['processing']['block_size'])
    write_json(folder/'terrain_geometry.json',dict(method=PROFILE_METHOD_VERSION,geometry_quality=report,
        high_slope_hard_exclusion=False,unknown_policy='255不参与硬排除',geometry_output=str(geometry_output),
        note='零Doppler观测剖面候选；不使用15度坡度硬剔除，历史岸线不作为真值。'))
    return products
