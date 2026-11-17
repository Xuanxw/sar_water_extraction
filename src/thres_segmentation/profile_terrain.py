"""将独立观测剖面掩膜供阈值流程消费，路径和科学参数由案例YAML显式提供。"""
from pathlib import Path
from time import perf_counter
import json
from contextlib import ExitStack
import numpy as np
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.windows import Window
from filelock import FileLock
from pyproj import CRS, Transformer
from scipy.ndimage import maximum_filter, minimum_filter
from shapely.geometry import box
from shapely.ops import unary_union
from sar_geometry.scene import run_zero_doppler_terrain_visibility, combine_shadow_layover_masks
from sar_geometry.metadata import read_bc_metadata, build_bc_range_doppler_model
from sar_geometry.profiles import sample_zero_doppler_visibility
from sar_geometry.terrain_support import bound_same_range_terrain, scan_regional_dem_envelope
from thres_segmentation.auxiliary_data import tile_specs, download_tile
from utils.files import write_json

PROFILE_METHOD_VERSION='radar_profile_zero_doppler_v3'


def visibility_transition_band(shadow, layover, radius=1):
    """在计算网格标记类别转折带，含0/1/255交界；稳定核心不扩张、不填洞。

    输入两个同尺寸uint8数组，输出bool。调用方读取radius行halo后再截去halo，
    避免分块边缘产生假边界。该标记只决定哪里重算，不是新的排除掩膜。
    """
    size = radius * 2 + 1
    boundary = np.zeros(shadow.shape, dtype=bool)
    for value in (shadow, layover):
        boundary |= maximum_filter(value, size=size, mode='nearest') != minimum_filter(
            value, size=size, mode='nearest'
        )
    return boundary


def refine_sar_visibility_boundaries(reference, geometry_dir, output, geometry, annotation, block_size, log):
    """将细网格稳定类别对齐至SAR，转折带按原生像元中心重新求解RD。

    DEM仅按块双线性求椭球高；阴影/叠掩不做双线性插值。复用完整观测剖面，
    不在局部窗口内重建阴影。未知仍为255，真实边界精度受源DEM及剖面采样限制。
    """
    started = perf_counter()
    geometry_dir = Path(geometry_dir)
    output = Path(output)
    band_path = geometry_dir / 'visibility_transition.tif'
    radius = 1
    # 先在细网格建立边界带。halo覆盖两侧，分块大小不改变标记结果。
    with rasterio.Env(GDAL_CACHEMAX=64 * 1024 * 1024), rasterio.open(geometry_dir / 'shadow_mask.tif') as shadow, rasterio.open(
        geometry_dir / 'layover_mask.tif'
    ) as layover:
        profile = dict(shadow.profile, dtype='uint8', nodata=None)
        with rasterio.open(band_path.with_suffix('.partial.tif'), 'w', **profile) as dst:
            for row in range(0, shadow.height, 128):
                height = min(128, shadow.height - row)
                win = Window(0, row - radius, shadow.width, height + 2 * radius)
                s = shadow.read(1, window=win, boundless=True, fill_value=255)
                l = layover.read(1, window=win, boundless=True, fill_value=255)
                boundary = visibility_transition_band(s, l, radius)[radius:radius + height]
                dst.write(boundary.astype('uint8'), 1, window=Window(0, row, shadow.width, height))
    band_path.with_suffix('.partial.tif').replace(band_path)
    axes = np.load(geometry_dir / 'data_cache/profile_axes.npz')
    times, theta = axes['times'], axes['theta']
    axes.close()
    masks = np.load(geometry_dir / 'data_cache/profiles.npy', mmap_mode='r')
    settings = dict(geometry['model'], radar_grid_confirmed=True)
    model = build_bc_range_doppler_model(annotation, settings)
    names = ('shadow_mask', 'layover_mask', 'terrain_exclusion')
    products = {name: output / (name + '.tif') for name in names}
    refined_count = valid_count = 0
    # 转折带严格使用与整景相同的DEM/轨道/轴；只减少稳定核心的重复求解。
    with rasterio.Env(GDAL_CACHEMAX=64 * 1024 * 1024), ExitStack() as stack:
        stack.callback(masks._mmap.close)
        src = stack.enter_context(rasterio.open(reference))
        native_grid = dict(crs=str(src.crs), width=src.width, height=src.height, transform=list(src.transform))
        profile = dict(src.profile, driver='GTiff', count=1, dtype='uint8', nodata=255,
                       compress='lzw', BIGTIFF='IF_SAFER')
        readers = {}
        for name in (*names[:2], 'visibility_transition'):
            base = stack.enter_context(rasterio.open(geometry_dir / (name + '.tif')))
            readers[name] = stack.enter_context(WarpedVRT(
                base, crs=src.crs, transform=src.transform, width=src.width, height=src.height,
                resampling=Resampling.nearest, nodata=255 if name in names else 0,
            ))
        dem = stack.enter_context(rasterio.open(geometry_dir / 'data_cache/dem_ellipsoid.tif'))
        heights = stack.enter_context(WarpedVRT(
            dem, crs=src.crs, transform=src.transform, width=src.width, height=src.height,
            dtype='float32', nodata=np.nan, resampling=Resampling.bilinear,
        ))
        writers = {name: stack.enter_context(rasterio.open(
            path.with_suffix('.partial.tif'), 'w', **profile
        )) for name, path in products.items()}
        to_ecef = Transformer.from_crs(CRS.from_user_input(src.crs).to_3d(), 4978, always_xy=True)
        # 限制到小块，避免原生整景的浮点坐标/RD临时数组触发OOM。
        for row in range(0, src.height, block_size):
            height = min(block_size, src.height - row)
            for col in range(0, src.width, block_size):
                win = Window(col, row, min(block_size, src.width - col), height)
                image = src.read(1, window=win, masked=True)
                valid = ~np.ma.getmaskarray(image) & np.isfinite(image.data)
                s = readers['shadow_mask'].read(1, window=win)
                l = readers['layover_mask'].read(1, window=win)
                refine = valid & (readers['visibility_transition'].read(1, window=win) == 1)
                if refine.any():
                    rr, cc = np.nonzero(refine)
                    a = src.transform
                    x = a.c + (cc + col + 0.5) * a.a + (rr + row + 0.5) * a.b
                    y = a.f + (cc + col + 0.5) * a.d + (rr + row + 0.5) * a.e
                    h = heights.read(1, window=win, masked=True).filled(np.nan)[refine]
                    points = np.column_stack(to_ecef.transform(x, y, h))
                    value, _, _ = sample_zero_doppler_visibility(
                        geometry['paths']['CORE_PATH'], model, points, times, theta, masks
                    )
                    s[refine], l[refine] = value[:, 0], value[:, 1]
                s[~valid] = l[~valid] = 255
                union = combine_shadow_layover_masks(s, l)
                for name, value in zip(names, (s, l, union)):
                    writers[name].write(value, 1, window=win)
                refined_count += int(refine.sum())
                valid_count += int(valid.sum())
            if row == 0 or row + height == src.height or row // block_size % 8 == 0:
                log(f'[04b] SAR原生边界复核 {row + height}/{src.height}行；'
                    f'重算{refined_count:,}点；{perf_counter() - started:.1f}s')
    del masks
    for path in products.values():
        path.with_suffix('.partial.tif').replace(path)
    return products, dict(
        method='exact_rd_at_native_sar_transition_pixels',
        transition_radius_cells=radius,
        native_grid=native_grid,
        transition_raster=str(band_path),
        valid_pixels=valid_count,
        refined_pixels=refined_count,
        elapsed_seconds=perf_counter() - started,
        note='稳定核心最近邻；类别转折带重算RD后采样完整剖面；未增加源DEM真实分辨率。',
    )


def prepare_profile_height_envelope(geometry, annotation, config, log):
    """按本景雷达距离/轨道选择DEM，计算外围地形的椭球高包络，单位米。

    先用宽高程界限调查潜在同斜距分支，再核查所需DEM覆盖并扫描H和N。
    20公里只定义剖面计算的DEM外扩量，不限制调查范围或水体输出范围。
    只修改本景geometry副本；不改原YAML，缺测/缺瓦片不静默套用固定包络。
    """
    settings = config['org']['profile_geometry'].get('height_envelope', {})
    mode = settings.get('mode', 'fixed')
    if mode == 'fixed':
        return None
    if mode != 'regional_dem':
        raise ValueError('height_envelope.mode须为fixed或regional_dem')
    if geometry['model']['height_datum'] != 'EGM2008':
        raise ValueError('自动区域包络当前使用EGM2008正高DEM及显式格网N，不能混用其他高程基准')
    survey = dict(settings['coverage_survey'], output_crs=geometry['scene']['output_crs'])
    domain = bound_same_range_terrain(annotation, geometry['paths']['CORE_PATH'],
                settings['global_height_bounds_m'], survey)
    log('[01] 本景同斜距地形调查域：'+str(domain['geographic_bounds_with_guard']))
    cache = Path(config['paths']['AUXILIARY_CACHE_DIR'])
    cache.mkdir(parents=True, exist_ok=True)
    auxiliary = config['org']['auxiliary']
    with FileLock(str(cache/'org_auxiliary.lock')):
        # 完整本地DEM可直接复用；未覆盖时按本景调查域补充共享Copernicus瓦片。
        sources = [Path(p) for p in geometry['paths']['DEM_TILES']]
        footprints = []
        for path in sources:
            with rasterio.open(path) as src:
                if src.crs.to_epsg() != 4326:
                    raise ValueError('自动区域包络需要WGS84经纬度源DEM')
                footprints.append(box(*src.bounds))
        target = box(*domain['geographic_bounds_with_guard'])
        if not footprints or not unary_union(footprints).covers(target):
            specs = tile_specs('copernicus', domain['geographic_bounds_with_guard'])
            if len(specs) > auxiliary['max_tiles']:
                raise ValueError('本景地形调查域超过辅助瓦片数量上限，需检查雷达范围或调整max_tiles')
            sources = []
            for spec in specs:
                destination = cache/'tiles/copernicus'/spec['filename']
                if auxiliary['download_missing']:
                    destination = download_tile(spec['url'], destination, timeout=auxiliary['timeout'],
                                    retries=auxiliary['retries'], log=log)
                elif not destination.is_file():
                    raise FileNotFoundError(f'本景地形支持缺少DEM且禁止下载：{destination}')
                sources.append(destination)
        envelope = scan_regional_dem_envelope(sources, geometry['paths']['GEOID_PATH'], domain,
                       settings['margin_m'], settings['rounding_m'], cache_dir=cache/'terrain_height_stats')
    geometry['paths']['DEM_TILES'] = sources
    geometry['scene']['height_bounds_m'] = envelope['height_bounds_m']
    envelope['support_buffer_m'] = geometry['scene']['support_buffer_m']
    write_json(Path(geometry['paths']['OUTPUT_DIR'])/'regional_height_envelope.json', envelope)
    log('[01] 本景自动椭球高包络：'+str(envelope['height_bounds_m'])+' m；源DEM '+str(len(sources))+'块')
    return envelope


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

    只生成地形文件。阈值/历史水体/矢量处理仍由原主流程执行，未知域不作硬剔除。
    """
    terrain_started = perf_counter()
    folder=Path(folder);paths=config['paths'];options=config['org']['profile_geometry']
    metadata=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
    # XML已由本景输入解析流程保存在成果目录，避免依赖原ZIP是否仍在原位置。
    identifier=Path(metadata['source_image']).stem
    annotation=read_bc_metadata(folder/'source.xml',xml_member=identifier+'.xml')
    geometry_output=Path(paths['PROFILE_GEOMETRY_CACHE_DIR'])/identifier
    settings=dict(options['model'])
    geometry=dict(paths=dict(XML_PATH=folder/'source.xml',ORG_PATH=Path(reference),DEM_TILES=paths.get('PROFILE_DEM_TILES',[]),
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
    with rasterio.Env(GDAL_CACHEMAX=64*1024*1024, GDAL_HTTP_TIMEOUT=config['org']['auxiliary']['timeout']):
        envelope = prepare_profile_height_envelope(geometry, annotation, config, log)
        report = run_zero_doppler_terrain_visibility(geometry, annotation, log)
    if envelope is not None:
        report = dict(report, height_envelope=envelope)
        write_json(geometry_output/'geometry_quality.json', report)
    if geometry['scene'].get('refine_sar_boundaries', False):
        products, refinement = refine_sar_visibility_boundaries(
            reference, geometry_output, folder, geometry, annotation,
            options['execution'].get('native_block_size', 256), log,
        )
        report = dict(report, native_boundary_refinement=refinement)
        # 剖面/映射累计耗时覆盖续跑前后；本次墙钟耗时单独记录，避免把缓存命中误称为冷启动性能。
        report['geometry_and_native_seconds'] = report['profile_seconds'] + report['mapping_seconds'] + refinement['elapsed_seconds']
        report['current_run_terrain_seconds'] = perf_counter() - terrain_started
        report['geometry_and_native_budget_met'] = (
            report['geometry_and_native_seconds'] <= geometry['execution']['time_budget_seconds']
        )
        write_json(geometry_output / 'geometry_quality.json', report)
        budget = geometry['execution']['time_budget_seconds']
        log(f'[05b] 累计几何计算加原生复核：{report["geometry_and_native_seconds"]:.2f}s；'
            f'{budget:g}秒目标={report["geometry_and_native_budget_met"]}；'
            f'本次地形阶段{report["current_run_terrain_seconds"]:.2f}s')
    else:
        products=align_profile_masks(reference,geometry_output,folder,config['processing']['block_size'])
    write_json(folder/'terrain_geometry.json',dict(method=PROFILE_METHOD_VERSION,geometry_quality=report,
        high_slope_hard_exclusion=False,unknown_policy='255不参与硬排除',geometry_output=str(geometry_output),
        note='零Doppler观测剖面候选；不使用15度坡度硬剔除，历史岸线不作为真值。'))
    return products
