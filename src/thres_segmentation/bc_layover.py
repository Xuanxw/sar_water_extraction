"""将 BC 条带剖面叠掩与同景参考水体实验的阴影合并到分割网格。"""
import json
import shutil
from pathlib import Path

import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform, reproject

from sar_geometry.metadata import read_bc_metadata
from sar_geometry.profile_layover import run_profile_layover
from sar_geometry.scene import combine_shadow_layover_masks
from thres_segmentation.product_layout import scene_product
from utils.files import sha256, write_json
from utils.raster import windows


def prepare_geographic_dem(source, destination):
    """将上一版逐景椭球高 DEM（米）双线性重投影为 1 角秒经纬网，高程基准沿用源文件。"""
    pending = destination.with_suffix('.partial.tif')
    with rasterio.open(source) as src:
        transform, width, height = calculate_default_transform(
            src.crs, 'EPSG:4326', src.width, src.height, *src.bounds, resolution=1 / 3600)
        with rasterio.open(pending, 'w', driver='GTiff', count=1, dtype='float32', nodata=-9999,
                           crs='EPSG:4326', transform=transform, width=width, height=height) as dst:
            reproject(rasterio.band(src, 1), rasterio.band(dst, 1), resampling=Resampling.bilinear)
    pending.replace(destination)


def align_layover_with_shadow(reference, layover, shadow, folder, block_size):
    """在原水体网格上合并旧阴影与最近邻对齐的 10 m 叠掩，保留 0/1/255 语义。"""
    products = {key: folder / (key + '.tif') for key in ('shadow_mask', 'layover_mask', 'terrain_exclusion')}
    pending = {key: path.with_suffix('.partial.tif') for key, path in products.items()}
    with rasterio.open(reference) as sar, rasterio.open(shadow) as old, rasterio.open(layover) as new:
        if (sar.crs, sar.transform, sar.shape) != (old.crs, old.transform, old.shape) or old.nodata != 255:
            raise ValueError(f'参考阴影与本景分割网格或 NoData 不匹配：{shadow}')
        profile = dict(driver='GTiff', width=sar.width, height=sar.height, transform=sar.transform,
                       crs=sar.crs, count=1, dtype='uint8', nodata=255, compress='lzw', tiled=True)
        with WarpedVRT(new, crs=sar.crs, transform=sar.transform, width=sar.width, height=sar.height,
                       src_nodata=255, nodata=255, resampling=Resampling.nearest) as aligned, \
             rasterio.open(pending['layover_mask'], 'w', **profile) as lay_dst, \
             rasterio.open(pending['terrain_exclusion'], 'w', **profile) as union_dst:
            for win in windows(sar, block_size):
                lay = aligned.read(1, window=win)
                valid = sar.read_masks(1, window=win) != 0
                lay[~valid] = 255
                union = combine_shadow_layover_masks(old.read(1, window=win), lay)
                union[~valid] = 255
                lay_dst.write(lay, 1, window=win)
                union_dst.write(union, 1, window=win)
    shutil.copyfile(shadow, pending['shadow_mask'])
    for key, target in products.items():
        pending[key].replace(target)
    return products


def generate_bc_layover_terrain(reference, config, folder):
    """按校正幅度图及其配套 XML 重算整景叠掩，再接入本景裁剪后的水体网格。"""
    paths = config['paths']
    record = json.loads((folder / 'input.json').read_text(encoding='utf-8'))
    previous = paths['REUSE_RESULTS_DIR'] / record['date'] / record['scene_id']
    old_record = json.loads(scene_product(previous, 'input.json').read_text(encoding='utf-8'))
    image = Path(record['path'])
    xml = image.with_suffix('.xml')
    xml_hash = sha256(xml)
    if old_record['source_stamp'] != record['source_stamp'] or xml_hash != sha256(scene_product(previous, 'source.xml')):
        raise ValueError(f'本景幅度图或 XML 与参考水体实验不一致：{image}')
    old_terrain = json.loads(scene_product(previous, 'terrain_geometry.json').read_text(encoding='utf-8'))
    old_geometry = Path(old_terrain['geometry_output'])
    source_dem = old_geometry / 'data_cache' / 'dem_ellipsoid.tif'
    shadow = scene_product(previous, 'shadow_mask.tif')
    geometry_dir = paths['PROFILE_GEOMETRY_CACHE_DIR'] / record['scene_id']
    if geometry_dir.resolve() == old_geometry.resolve():
        raise ValueError('新版叠掩需要独立的几何缓存目录')
    geometry_dir.mkdir(parents=True, exist_ok=True)
    geographic_dem = geometry_dir / 'dem_ellipsoid_1arcsec.tif'
    prepare_geographic_dem(source_dem, geographic_dem)
    geometry = dict(paths=dict(XML_PATH=xml, ORG_PATH=image, DEM_PATH=geographic_dem,
                               CORE_PATH=paths['GEOMETRY_CORE_PATH'], OUTPUT_DIR=geometry_dir),
                    model=dict(height_datum='WGS84_ellipsoid', doppler_mode='zero'),
                    scene=dict(profile_orbit='bc_azimuth', output_crs=config['org']['output_crs'],
                               resolution_m=10, org_nodata=0))
    report = run_profile_layover(geometry, read_bc_metadata(xml), log=lambda message: print(message, flush=True))
    products = align_layover_with_shadow(reference, geometry_dir / 'layover_10m.tif', shadow,
                                         folder, config['processing']['block_size'])
    write_json(folder / 'terrain_geometry.json', dict(
        method=config['org']['profile_geometry']['layover_method'], geometry_output=str(geometry_dir),
        geometry_quality=report, source_dem_ellipsoid=str(source_dem), xml_sha256=xml_hash,
        shadow_reference=str(shadow), layover_resampling='nearest', unknown_policy='255不参与硬排除'))
    return products
