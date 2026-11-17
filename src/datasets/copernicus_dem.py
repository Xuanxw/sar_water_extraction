"""将当地 Copernicus GLO-30 瓦片按目标地图网格读取；高程单位为米。"""
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from shapely.geometry import box
from shapely.ops import unary_union
from shapely.strtree import STRtree


class DemTileCatalog:
    """仅扫描一次本地瓦片头，按场景空间范围查找候选，避免逐景打开整个缓存。"""

    def __init__(self, source):
        source = Path(source)
        if not source.exists():
            raise FileNotFoundError(f"本地 DEM 路径不存在：{source}")
        self.paths = sorted(source.glob("Copernicus_DSM_COG_*_DEM.tif")) if source.is_dir() else [source]
        if not self.paths:
            raise FileNotFoundError(f"未找到本地 Copernicus GLO-30 DEM 瓦片：{source}")
        bounds = []
        for path in self.paths:
            with rasterio.open(path) as dem:
                if dem.crs is None or dem.count != 1:
                    raise ValueError(f"DEM 需具备地理参考且为单波段：{path}")
                bounds.append(box(*transform_bounds(dem.crs, "EPSG:4326", *dem.bounds)))
        self.index = STRtree(bounds)

    def covering(self, scene):
        if not scene.crs or scene.transform == rasterio.Affine.identity():
            raise ValueError(f"目标影像缺少地理网格：{scene.name}")
        extent = box(*transform_bounds(scene.crs, "EPSG:4326", *scene.bounds))
        matches = sorted(self.index.query(extent, predicate="intersects"))
        if not matches or not unary_union(self.index.geometries.take(matches)).covers(extent):
            raise ValueError(f"本地 DEM 瓦片未覆盖影像范围：{Path(scene.name).name}；经纬范围 {extent.bounds}")
        return [self.paths[i] for i in matches]


def aligned_dem_tiles(source, scene, stack):
    """按场景 CRS、仿射变换及行列数创建双线性重采样视图；缺测输出 NaN。"""
    paths = DemTileCatalog(source).covering(scene) if isinstance(source, (str, Path)) else source
    tiles = []
    for path in paths:
        dem = stack.enter_context(rasterio.open(path))
        if dem.scales[0] != 1 or dem.offsets[0] != 0:
            raise ValueError(f"DEM 必须已按米存储，带比例或偏移的栅格请先解码：{path}")
        tile = WarpedVRT(dem, crs=scene.crs, transform=scene.transform,
                         width=scene.width, height=scene.height, dtype="float32",
                         nodata=float("nan"), resampling=Resampling.bilinear)
        tiles.append(stack.enter_context(tile))
    return tiles


def read_elevation(tiles, window):
    """合并窗口内各瓦片的有效米值；保留零和负高程，内部空洞为 NaN。"""
    heights = np.full((int(window.height), int(window.width)), np.nan, dtype="float32")
    for tile in tiles:
        values = tile.read(1, window=window, masked=True).filled(np.nan)
        np.copyto(heights, values, where=~np.isfinite(heights))
    return heights
