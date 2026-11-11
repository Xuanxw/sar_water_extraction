"""从原 SAR GUI 脚本迁移，保留已有处理逻辑。"""
from osgeo import gdal, ogr, osr
import os
from pathlib import Path
import glob
import time
import json
import hashlib
import math
import shutil
import ssl
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, transform as warp_transform, transform_bounds
from rasterio.crs import CRS
from rasterio import features
import geopandas as gpd
from shapely.geometry import LineString, Polygon
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PySide6.QtCore import QThread, Signal
gdal.SetConfigOption("GDAL_MEM_ENABLE_OPEN", "YES")
from thres_segmentation.threshold import percentile_mask

class SARWorker(QThread):
    log_signal = Signal(str)
    progress_signal = Signal(int)
    finished_signal = Signal(bool, str)
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 强制使用 Agg 后端，防止 Matplotlib 在后台线程中弹窗或崩溃
        import matplotlib
        matplotlib.use('Agg') 
        # 🔑 新增：确保后台线程也使用宋体
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['SimSun', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

    OVERPASS_ENDPOINTS = (
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://overpass.nchc.org.tw/api/interpreter",
    )

    @staticmethod
    def build_sar_coverage_mask(raw, raster_mask=None):
        """区分外围黑边零值与影像覆盖区内部的异常零值。

        只把和栅格边界连通的非正值区域判作外围 NoData。被有效信号包围的
        零值/非有限值属于成像漏洞，仍保留在覆盖范围内；后续仅作数值中和，
        不执行逐像素空间插值。
        """
        signal_valid = np.isfinite(raw) & (raw > 0)
        if raster_mask is not None:
            signal_valid &= raster_mask.astype(bool)

        # 仅为识别 footprint 拓扑做一次小尺度闭运算，阻断零散内部0值串成细线后
        # 意外连到外围黑边；不改变 raw 数值，也不直接作为水体判定依据。
        topology_seed = cv2.morphologyEx(
            signal_valid.astype(np.uint8), cv2.MORPH_CLOSE,
            np.ones((5, 5), dtype=np.uint8), iterations=1)
        # 外围增加一圈 0，从左上角洪水填充所有与图像外部连通的无信号像素。
        padded = np.pad(topology_seed, 1, mode='constant', constant_values=0)
        cv2.floodFill(padded, None, seedPoint=(0, 0), newVal=2,
                      loDiff=0, upDiff=0, flags=4)
        exterior_nodata = padded[1:-1, 1:-1] == 2
        coverage = ~exterior_nodata
        internal_gap_count = int(np.count_nonzero(coverage & ~signal_valid))
        del topology_seed, padded
        return coverage, signal_valid, internal_gap_count

    @staticmethod
    def neutralize_internal_sar_zeros(raw, coverage, signal_valid, sample_step=8):
        """不做空间插值，仅用抽样中位数中和覆盖区内部零值。

        内部零值仍由 signal_valid 掩膜保留其位置语义；中位数只供 Lee
        数值计算使用，防止 log10(0) 产生极暗伪水体。该方法只有一次抽样和一次
        布尔赋值，计算量远小于多轮邻域插值。
        """
        working = (raw if raw.dtype == np.float32 and raw.flags.writeable
                   else raw.astype(np.float32, copy=True))
        sampled = working[::sample_step, ::sample_step]
        sampled_values = sampled[np.isfinite(sampled) & (sampled > 0)]
        if sampled_values.size == 0:
            raise ValueError("SAR 影像中没有可用于计算的正值像素。")
        neutral_value = float(np.median(sampled_values))
        del sampled_values, sampled

        # internal_gap_mask 只在本函数内短暂存在；返回并继续复用 signal_valid，
        # 使异常零值不参与噪声、百分位和阈值统计。
        internal_gap_mask = coverage & ~signal_valid
        gap_count = int(np.count_nonzero(internal_gap_mask))
        working[internal_gap_mask] = neutral_value
        del internal_gap_mask
        return working, signal_valid, gap_count, neutral_value

    @staticmethod
    def valid_data_bounds(valid_mask, transform):
        rows = np.flatnonzero(np.any(valid_mask, axis=1))
        cols = np.flatnonzero(np.any(valid_mask, axis=0))
        if not rows.size or not cols.size:
            raise ValueError("SAR 影像中没有有效像素。")
        r0, r1, c0, c1 = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
        corners = [transform * (c0, r0), transform * (c1, r0),
                   transform * (c0, r1), transform * (c1, r1)]
        xs, ys = zip(*corners)
        return min(xs), min(ys), max(xs), max(ys)

    @staticmethod
    def copernicus_tile_name(lat_deg, lon_deg):
        """返回 Copernicus GLO-30 公共 COG 的1°分幅名称。"""
        lat_prefix = "N" if lat_deg >= 0 else "S"
        lon_prefix = "E" if lon_deg >= 0 else "W"
        return (f"Copernicus_DSM_COG_10_{lat_prefix}{abs(int(lat_deg)):02d}_00_"
                f"{lon_prefix}{abs(int(lon_deg)):03d}_00_DEM")

    @classmethod
    def copernicus_tiles_for_bounds(cls, bounds_wgs84):
        """列出覆盖 WGS84 包围框的全部 Copernicus GLO-30 1°瓦片。"""
        min_lon, min_lat, max_lon, max_lat = bounds_wgs84
        if not (-180.0 <= min_lon < max_lon <= 180.0 and
                -90.0 <= min_lat < max_lat <= 90.0):
            raise ValueError(f"DEM下载范围无效或跨越180°经线：{bounds_wgs84}")
        tiles = []
        for lat_deg in range(int(math.floor(min_lat)), int(math.ceil(max_lat))):
            for lon_deg in range(int(math.floor(min_lon)), int(math.ceil(max_lon))):
                tiles.append((lat_deg, lon_deg, cls.copernicus_tile_name(lat_deg, lon_deg)))
        return tiles

    @staticmethod
    def buffered_wgs84_bounds(bounds_wgs84, buffer_m):
        """以近似米制距离扩展经纬度范围，给坡度边缘计算保留邻域。"""
        min_lon, min_lat, max_lon, max_lat = bounds_wgs84
        center_lat = (min_lat + max_lat) / 2.0
        lat_pad = max(float(buffer_m), 0.0) / 111320.0
        lon_scale = max(math.cos(math.radians(center_lat)), 0.05)
        lon_pad = max(float(buffer_m), 0.0) / (111320.0 * lon_scale)
        return (max(-180.0, min_lon - lon_pad), max(-89.9999, min_lat - lat_pad),
                min(180.0, max_lon + lon_pad), min(89.9999, max_lat + lat_pad))

    def download_copernicus_dem_tiles(self, bounds_wgs84, cache_dir):
        """下载并校验 Copernicus GLO-30 COG；完整瓦片永久保留供后续复用。"""
        import certifi

        tile_cache_dir = os.path.join(cache_dir, "copernicus_glo30_tiles")
        os.makedirs(tile_cache_dir, exist_ok=True)
        context = ssl.create_default_context(cafile=certifi.where())
        tile_specs = self.copernicus_tiles_for_bounds(bounds_wgs84)
        self.log_signal.emit(
            f"🌐 Copernicus GLO-30 覆盖范围需要 {len(tile_specs)} 个1°瓦片。")
        downloaded_tiles = []
        failed_tiles = []

        for lat_deg, lon_deg, base_name in tile_specs:
            if self.isInterruptionRequested():
                raise InterruptedError("任务已手动停止")
            filename = base_name + ".tif"
            local_path = os.path.join(tile_cache_dir, filename)
            partial_path = local_path + ".part"
            url = ("https://copernicus-dem-30m.s3.amazonaws.com/"
                   f"{base_name}/{filename}")

            cache_valid = False
            if os.path.exists(local_path) and os.path.getsize(local_path) > 1024:
                try:
                    with rasterio.open(local_path) as src:
                        cache_valid = src.count >= 1 and src.width > 0 and src.height > 0
                except Exception:
                    cache_valid = False
            if cache_valid:
                self.log_signal.emit(f"   ♻️ 使用DEM瓦片缓存：{filename}")
                downloaded_tiles.append(local_path)
                continue

            last_error = None
            for attempt in range(1, 4):
                try:
                    if os.path.exists(partial_path):
                        os.remove(partial_path)
                    self.log_signal.emit(
                        f"   📥 下载 {filename}（第 {attempt}/3 次）...")
                    request = urllib.request.Request(
                        url, headers={"User-Agent": "TY-SARWater/1.0"})
                    with urllib.request.urlopen(
                            request, timeout=300, context=context) as response, \
                            open(partial_path, "wb") as output:
                        shutil.copyfileobj(response, output, length=1024 * 1024)
                    with rasterio.open(partial_path) as src:
                        if src.count < 1 or src.width <= 0 or src.height <= 0:
                            raise ValueError("下载文件不是有效的DEM栅格")
                    os.replace(partial_path, local_path)
                    downloaded_tiles.append(local_path)
                    self.log_signal.emit(f"   ✅ DEM瓦片已缓存：{local_path}")
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    if os.path.exists(partial_path):
                        os.remove(partial_path)
                    self.log_signal.emit(f"   ⚠️ {filename} 下载失败：{exc}")
                    if attempt < 3:
                        time.sleep(attempt)
            if last_error is not None:
                failed_tiles.append((filename, str(last_error)))

        if not downloaded_tiles:
            details = "；".join(f"{name}: {error}" for name, error in failed_tiles)
            raise RuntimeError(f"Copernicus GLO-30 下载失败，未获得有效DEM瓦片。{details}")
        if failed_tiles:
            self.log_signal.emit(
                f"⚠️ 有 {len(failed_tiles)} 个DEM瓦片不可用；对应区域按DEM未知区处理。")
        return downloaded_tiles

    @staticmethod
    def terrain_metric_crs(sar_crs, bounds_wgs84):
        """优先复用米制SAR投影，否则根据影像中心自动选择UTM。"""
        if sar_crs is not None:
            candidate = CRS.from_user_input(sar_crs)
            if candidate.is_projected:
                units = (candidate.linear_units or "").lower()
                if units in ("metre", "meter", "metres", "meters", "m"):
                    return candidate
        center_lon = (bounds_wgs84[0] + bounds_wgs84[2]) / 2.0
        center_lat = (bounds_wgs84[1] + bounds_wgs84[3]) / 2.0
        zone = min(max(int((center_lon + 180.0) // 6.0) + 1, 1), 60)
        epsg = (32600 if center_lat >= 0 else 32700) + zone
        return CRS.from_epsg(epsg)

    @staticmethod
    def resolve_local_slope_aspect(folder_path):
        """兼容原viz命名和任意包含slope/aspect关键字的GeoTIFF。"""
        if not folder_path or not os.path.isdir(folder_path):
            raise FileNotFoundError(f"本地坡度/坡向文件夹不存在：{folder_path}")
        slope_path = os.path.join(folder_path, "viz.hh_slope.tif")
        aspect_path = os.path.join(folder_path, "viz.hh_aspect.tif")
        if not os.path.exists(slope_path):
            candidates = sorted(glob.glob(os.path.join(folder_path, "*slope*.tif")) +
                                glob.glob(os.path.join(folder_path, "*slope*.tiff")))
            slope_path = candidates[0] if candidates else None
        if not os.path.exists(aspect_path):
            candidates = sorted(glob.glob(os.path.join(folder_path, "*aspect*.tif")) +
                                glob.glob(os.path.join(folder_path, "*aspect*.tiff")))
            aspect_path = candidates[0] if candidates else None
        if not slope_path or not aspect_path:
            raise FileNotFoundError(
                f"文件夹中未同时找到 *slope.tif 和 *aspect.tif：{folder_path}")
        return slope_path, aspect_path

    @staticmethod
    def resolve_local_dem_sources(path):
        """本地DEM既可选择单个GeoTIFF，也可选择包含多个分幅的文件夹。"""
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"本地DEM不存在：{path}")
        if os.path.isfile(path):
            return [path]
        candidates = sorted(glob.glob(os.path.join(path, "*.tif")) +
                            glob.glob(os.path.join(path, "*.tiff")))
        candidates = [item for item in candidates
                      if "slope" not in os.path.basename(item).lower() and
                      "aspect" not in os.path.basename(item).lower()]
        if not candidates:
            raise FileNotFoundError(f"本地DEM文件夹中没有可用GeoTIFF：{path}")
        return candidates

    def prepare_terrain_layers(self, cfg, bounds_wgs84, sar_crs):
        """将下载或本地DEM处理为约30米米制坡度/坡向GeoTIFF，并复用缓存。"""
        mode = cfg.get("terrain_source_mode", "local_layers")
        if mode == "local_layers":
            folder_path = cfg.get("dem_folder")
            if not folder_path:
                folder_path = os.path.join(os.path.dirname(cfg["folder_path"]), "viz")
            slope_path, aspect_path = self.resolve_local_slope_aspect(folder_path)
            self.log_signal.emit(f"📂 使用本地坡度：{slope_path}")
            self.log_signal.emit(f"📂 使用本地坡向：{aspect_path}")
            return slope_path, aspect_path

        buffer_m = float(cfg.get("dem_buffer_m", 1000.0))
        resolution_m = float(cfg.get("dem_resolution_m", 30.0))
        buffered_bounds = self.buffered_wgs84_bounds(bounds_wgs84, buffer_m)
        metric_crs = self.terrain_metric_crs(sar_crs, buffered_bounds)
        metric_bounds = transform_bounds(
            "EPSG:4326", metric_crs, *buffered_bounds, densify_pts=21)

        if mode == "download":
            cache_dir = (cfg.get("dem_cache_dir") or
                         os.path.join(cfg["folder_path"], "dem_cache"))
            os.makedirs(cache_dir, exist_ok=True)
            source_paths = self.download_copernicus_dem_tiles(buffered_bounds, cache_dir)
            source_name = "cop30"
            source_description = "Copernicus DEM GLO-30 (EGM2008 orthometric height)"
        elif mode == "local_dem":
            source_paths = self.resolve_local_dem_sources(cfg.get("local_dem_path"))
            source_name = "local_dem"
            source_description = "Local DEM (elevation unit assumed to be metre)"
        else:
            raise ValueError(f"未知地形数据来源模式：{mode}")

        source_signature = []
        for path in source_paths:
            stat = os.stat(path)
            source_signature.append(
                (os.path.abspath(path), int(stat.st_size), int(stat.st_mtime)))
        cache_payload = {
            "mode": mode,
            "sources": source_signature,
            "bounds": [round(value, 3) for value in metric_bounds],
            "crs": metric_crs.to_string(),
            "resolution_m": resolution_m,
            "algorithm": "ZevenbergenThorne",
        }
        cache_key = hashlib.sha1(
            json.dumps(cache_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()[:12]
        terrain_dir = os.path.join(cfg["output_dir"], "terrain")
        os.makedirs(terrain_dir, exist_ok=True)
        prefix = f"{source_name}_{cache_key}"
        projected_dem_path = os.path.join(terrain_dir, f"{prefix}_dem_projected.tif")
        self.analysis_elevation_path = projected_dem_path
        slope_path = os.path.join(terrain_dir, f"{prefix}_slope.tif")
        aspect_path = os.path.join(terrain_dir, f"{prefix}_aspect.tif")

        cached_outputs = (projected_dem_path, slope_path, aspect_path)
        cache_valid = all(os.path.exists(path) and os.path.getsize(path) > 1024
                          for path in cached_outputs)
        if cache_valid:
            try:
                for path in cached_outputs:
                    with rasterio.open(path) as src:
                        if src.width <= 0 or src.height <= 0:
                            raise ValueError(f"无效缓存：{path}")
                self.log_signal.emit(f"♻️ 使用已计算的DEM地形成果缓存：{terrain_dir}")
                return slope_path, aspect_path
            except Exception as exc:
                self.log_signal.emit(f"⚠️ 地形成果缓存无效，将重新计算：{exc}")

        self.log_signal.emit(
            f"🗺️ 正在将DEM拼接并投影至 {metric_crs.to_string()}，分辨率 {resolution_m:g} m...")
        projected_partial = projected_dem_path + ".part.tif"
        if os.path.exists(projected_partial):
            os.remove(projected_partial)
        warp_options = gdal.WarpOptions(
            format="GTiff", dstSRS=metric_crs.to_wkt(), outputBounds=metric_bounds,
            xRes=resolution_m, yRes=resolution_m, targetAlignedPixels=True,
            resampleAlg="bilinear", dstNodata=-9999.0, outputType=gdal.GDT_Float32,
            multithread=True, warpMemoryLimit=512,
            creationOptions=["TILED=YES", "COMPRESS=DEFLATE", "PREDICTOR=3", "BIGTIFF=IF_SAFER"])
        warp_result = gdal.Warp(projected_partial, source_paths, options=warp_options)
        if warp_result is None:
            raise RuntimeError("DEM拼接、裁剪或投影失败。请检查DEM坐标系和覆盖范围。")
        warp_result = None
        os.replace(projected_partial, projected_dem_path)

        with rasterio.open(projected_dem_path) as src:
            preview = src.read(1, masked=True, out_shape=(
                min(src.height, 1024), min(src.width, 1024)))
            if np.ma.count(preview) == 0:
                raise ValueError("DEM与SAR有效范围不重叠，投影结果全部为空值。")
        self.log_signal.emit(
            f"📐 正在 {resolution_m:g} 米投影网格上计算坡度和坡向...")
        creation_options = [
            "TILED=YES", "COMPRESS=DEFLATE", "PREDICTOR=3", "BIGTIFF=IF_SAFER"]
        slope_options = gdal.DEMProcessingOptions(
            format="GTiff", computeEdges=True, alg="ZevenbergenThorne",
            slopeFormat="degree", creationOptions=creation_options)
        aspect_options = gdal.DEMProcessingOptions(
            format="GTiff", computeEdges=True, alg="ZevenbergenThorne",
            zeroForFlat=True, creationOptions=creation_options)
        slope_partial = slope_path + ".part.tif"
        aspect_partial = aspect_path + ".part.tif"
        for partial_path in (slope_partial, aspect_partial):
            if os.path.exists(partial_path):
                os.remove(partial_path)
        slope_result = gdal.DEMProcessing(
            slope_partial, projected_dem_path, "slope", options=slope_options)
        aspect_result = gdal.DEMProcessing(
            aspect_partial, projected_dem_path, "aspect", options=aspect_options)
        if slope_result is None or aspect_result is None:
            slope_result = None
            aspect_result = None
            raise RuntimeError("GDAL坡度/坡向计算失败。")
        slope_result = None
        aspect_result = None
        os.replace(slope_partial, slope_path)
        os.replace(aspect_partial, aspect_path)

        common_tags = {
            "terrain_cache_key": cache_key,
            "source": source_description,
            "processing_crs": metric_crs.to_string(),
            "processing_resolution_m": f"{resolution_m:g}",
            "derivative_algorithm": "ZevenbergenThorne",
        }
        for path, terrain_kind in ((projected_dem_path, "projected_dem"),
                                   (slope_path, "slope_degree"),
                                   (aspect_path, "aspect_azimuth_degree")):
            with rasterio.open(path, "r+") as dst:
                dst.update_tags(**common_tags, terrain_kind=terrain_kind)
        manifest_path = os.path.join(terrain_dir, f"{prefix}_manifest.json")
        manifest = dict(cache_payload)
        manifest.update({
            "source_description": source_description,
            "buffered_bounds_wgs84": list(buffered_bounds),
            "projected_dem": projected_dem_path,
            "slope": slope_path,
            "aspect": aspect_path,
        })
        with open(manifest_path, "w", encoding="utf-8") as output:
            json.dump(manifest, output, ensure_ascii=False, indent=2)
        self.log_signal.emit(f"✅ 坡度GeoTIFF：{slope_path}")
        self.log_signal.emit(f"✅ 坡向GeoTIFF：{aspect_path}")
        return slope_path, aspect_path

    @staticmethod
    def split_bbox(bbox):
        minx, miny, maxx, maxy = bbox
        midx, midy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
        return [(minx, miny, midx, midy), (midx, miny, maxx, midy),
                (minx, midy, midx, maxy), (midx, midy, maxx, maxy)]

    def request_overpass_tile(self, feature_type, bbox, endpoint):
        import certifi
        minx, miny, maxx, maxy = bbox
        osm_bbox = f"{miny:.8f},{minx:.8f},{maxy:.8f},{maxx:.8f}"
        tag = "building" if feature_type == "building" else "highway"
        query = f'[out:json][timeout:120];way["{tag}"]({osm_bbox});out tags geom;'
        request = urllib.request.Request(
            endpoint,
            data=urllib.parse.urlencode({"data": query}).encode("utf-8"),
            headers={"User-Agent": "TY-SARWater/1.0", "Accept": "application/json"})
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(request, timeout=150, context=context) as response:
            return json.load(response).get("elements", [])

    def download_osm_recursive(self, feature_type, bbox, depth=0, max_depth=4):
        """多镜像下载；遇到 429/504/超时时把 bbox 四分递归，避免整城道路查询超限。"""
        last_error = None
        for endpoint in self.OVERPASS_ENDPOINTS:
            try:
                return self.request_overpass_tile(feature_type, bbox, endpoint)
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ssl.SSLError) as exc:
                last_error = exc
                self.log_signal.emit(
                    f"⚠️ Overpass 请求失败（层级{depth}，{endpoint.split('/')[2]}）：{exc}")
        if depth >= max_depth:
            raise RuntimeError(f"OSM 分块下载达到最大拆分层级：{last_error}")
        self.log_signal.emit(f"🔀 将当前范围四分后重试（层级 {depth + 1}/{max_depth}）...")
        elements = []
        for child_bbox in self.split_bbox(bbox):
            if self.isInterruptionRequested():
                raise InterruptedError("任务已手动停止")
            elements.extend(self.download_osm_recursive(feature_type, child_bbox, depth + 1, max_depth))
        return elements

    def download_osm_layer(self, feature_type, bbox_wgs84, cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
        key = "_".join(f"{value:.5f}" for value in bbox_wgs84)
        cache_path = os.path.join(cache_dir, f"osm_{feature_type}_{key}.gpkg")
        if os.path.exists(cache_path) and os.path.getsize(cache_path) > 100:
            self.log_signal.emit(f"♻️ 使用 OSM 缓存: {os.path.basename(cache_path)}")
            return gpd.read_file(cache_path), cache_path

        self.log_signal.emit(f"🌐 分块下载 OSM {feature_type} 数据...")
        elements = self.download_osm_recursive(feature_type, bbox_wgs84)
        records, seen_ids = [], set()
        for element in elements:
            osm_id = element.get("id")
            if osm_id in seen_ids:
                continue
            seen_ids.add(osm_id)
            coords = [(point["lon"], point["lat"]) for point in element.get("geometry", [])]
            if feature_type == "building":
                if len(coords) < 4:
                    continue
                if coords[0] != coords[-1]:
                    coords.append(coords[0])
                geom = Polygon(coords)
                road_class = None
            else:
                if len(coords) < 2:
                    continue
                geom = LineString(coords)
                road_class = element.get("tags", {}).get("highway")
            if geom.is_empty or not geom.is_valid:
                continue
            records.append({"osm_id": osm_id, "class": road_class, "geometry": geom})
        if records:
            gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
            gdf.to_file(cache_path, layer=feature_type, driver="GPKG")
        else:
            gdf = gpd.GeoDataFrame(
                {"osm_id": [], "class": [], "geometry": []}, geometry="geometry", crs="EPSG:4326")
        self.log_signal.emit(f"✅ OSM {feature_type} 获取完成：{len(gdf)} 个要素。")
        return gdf, cache_path if records else "无数据"

    def build_osm_masks(self, cfg, bbox_wgs84, sar_crs, sar_transform, sar_shape, valid):
        cache_dir = os.path.join(cfg['output_dir'], "osm_mask_cache")
        building_mask = None
        road_mask = None
        if cfg.get('enable_building_mask', False):
            building_mask = np.zeros(sar_shape, dtype=np.uint8)
            buildings, _ = self.download_osm_layer("building", bbox_wgs84, cache_dir)
            if not buildings.empty:
                buildings = buildings.to_crs(sar_crs)
                building_mask = features.rasterize(
                    ((geom, 1) for geom in buildings.geometry), out_shape=sar_shape,
                    transform=sar_transform, fill=0, dtype=np.uint8)
            self.log_signal.emit(f"🏢 建筑掩膜覆盖 {np.count_nonzero(building_mask)} 个像素。")
        if cfg.get('enable_road_mask', False):
            road_mask = np.zeros(sar_shape, dtype=np.uint8)
            roads, _ = self.download_osm_layer("road", bbox_wgs84, cache_dir)
            if not roads.empty:
                metric_crs = roads.estimate_utm_crs()
                if metric_crs is None:
                    raise ValueError("无法确定道路缓冲所需的米制坐标系。")
                buffered = roads.to_crs(metric_crs).geometry.buffer(
                    max(cfg.get('road_width', 8.0) / 2.0, 0.5)).to_crs(sar_crs)
                road_mask = features.rasterize(
                    ((geom, 1) for geom in buffered), out_shape=sar_shape,
                    transform=sar_transform, fill=0, dtype=np.uint8)
            self.log_signal.emit(f"🛣️ 道路掩膜覆盖 {np.count_nonzero(road_mask)} 个像素。")
        if building_mask is not None:
            building_mask[~valid] = 0
        if road_mask is not None:
            road_mask[~valid] = 0
        return building_mask, road_mask

    @staticmethod
    def worldcover_tiles(bounds_wgs84):
        """返回覆盖经纬度范围的 ESA WorldCover 3°×3°瓦片编号。"""
        min_lon, min_lat, max_lon, max_lat = bounds_wgs84
        start_lat = int(math.floor(min_lat / 3.0) * 3)
        end_lat = int(math.floor(max_lat / 3.0) * 3) + 3
        start_lon = int(math.floor(min_lon / 3.0) * 3)
        end_lon = int(math.floor(max_lon / 3.0) * 3) + 3
        tiles = []
        for lat in range(start_lat, end_lat, 3):
            for lon in range(start_lon, end_lon, 3):
                ns = 'N' if lat >= 0 else 'S'
                ew = 'E' if lon >= 0 else 'W'
                tiles.append(f"{ns}{abs(lat):02d}{ew}{abs(lon):03d}")
        return tiles

    def build_farmland_mask(self, cfg, bounds_wgs84, sar_crs, sar_transform,
                            sar_shape, valid):
        """按 SAR 网格下载/缓存 WorldCover 40类农田，并保护80类历史水体。"""
        import certifi
        output_dir = cfg['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        cache_path = os.path.join(output_dir, "farmland_mask.tif")
        protect_water = bool(cfg.get('protect_historical_water', True))
        expected_protection = "1" if protect_water else "0"

        if os.path.exists(cache_path):
            try:
                with rasterio.open(cache_path) as src:
                    cache_ok = (
                        src.shape == sar_shape and src.crs == sar_crs and
                        src.transform.almost_equals(sar_transform) and
                        src.tags().get("worldcover_version") == "2021_v200" and
                        src.tags().get("protect_historical_water") == expected_protection)
                    if cache_ok:
                        farmland = src.read(1)
                        farmland = (farmland == 1).astype(np.uint8)
                        farmland[~valid] = 0
                        self.log_signal.emit(
                            f"♻️ 使用 ESA WorldCover 农田掩膜缓存：{cache_path}")
                        return farmland
            except Exception as exc:
                self.log_signal.emit(f"⚠️ 农田掩膜缓存无效，将重新生成：{exc}")

        farmland = np.zeros(sar_shape, dtype=np.uint8)
        historical_water = (np.zeros(sar_shape, dtype=np.uint8)
                            if protect_water else None)
        tiles = self.worldcover_tiles(bounds_wgs84)
        tile_cache_dir = os.path.join(output_dir, "worldcover_tile_cache")
        os.makedirs(tile_cache_dir, exist_ok=True)
        self.log_signal.emit(
            f"🌾 正在按 SAR 有效范围读取 ESA WorldCover 2021 农田数据（{len(tiles)}个瓦片）...")
        context = ssl.create_default_context(cafile=certifi.where())
        successful_tiles = 0
        for tile in tiles:
            if self.isInterruptionRequested():
                raise InterruptedError("任务已手动停止")
            filename = f"ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
            url = ("https://esa-worldcover.s3.eu-central-1.amazonaws.com/"
                   f"v200/2021/map/{filename}")
            local_tile = os.path.join(tile_cache_dir, filename)
            partial_tile = local_tile + ".part"
            try:
                if not os.path.exists(local_tile) or os.path.getsize(local_tile) < 1024:
                    self.log_signal.emit(f"   📥 下载 WorldCover {tile} 到本地缓存...")
                    request = urllib.request.Request(
                        url, headers={"User-Agent": "TY-SARWater/1.0"})
                    try:
                        with urllib.request.urlopen(
                                request, timeout=300, context=context) as response, \
                                open(partial_tile, 'wb') as output:
                            shutil.copyfileobj(response, output, length=1024 * 1024)
                        os.replace(partial_tile, local_tile)
                    except Exception:
                        if os.path.exists(partial_tile):
                            os.remove(partial_tile)
                        raise
                else:
                    self.log_signal.emit(f"   ♻️ 使用 WorldCover 瓦片缓存 {tile}。")
                with rasterio.open(local_tile) as src:
                        left = max(bounds_wgs84[0], src.bounds.left)
                        bottom = max(bounds_wgs84[1], src.bounds.bottom)
                        right = min(bounds_wgs84[2], src.bounds.right)
                        top = min(bounds_wgs84[3], src.bounds.top)
                        if left >= right or bottom >= top:
                            continue
                        raw_window = src.window(left, bottom, right, top)
                        col0 = max(int(math.floor(raw_window.col_off)), 0)
                        row0 = max(int(math.floor(raw_window.row_off)), 0)
                        col1 = min(int(math.ceil(raw_window.col_off + raw_window.width)), src.width)
                        row1 = min(int(math.ceil(raw_window.row_off + raw_window.height)), src.height)
                        window = rasterio.windows.Window(
                            col0, row0, col1 - col0, row1 - row0)
                        landcover = src.read(1, window=window)
                        source_transform = src.window_transform(window)
                        source_crs = src.crs
                crop_source = (landcover == 40).astype(np.uint8)
                reproject(
                    source=crop_source, destination=farmland,
                    src_transform=source_transform, src_crs=source_crs,
                    dst_transform=sar_transform, dst_crs=sar_crs,
                    resampling=Resampling.nearest, init_dest_nodata=False)
                if protect_water:
                    water_source = (landcover == 80).astype(np.uint8)
                    reproject(
                        source=water_source, destination=historical_water,
                        src_transform=source_transform, src_crs=source_crs,
                        dst_transform=sar_transform, dst_crs=sar_crs,
                        resampling=Resampling.nearest, init_dest_nodata=False)
                    del water_source
                del landcover, crop_source
                successful_tiles += 1
                self.log_signal.emit(f"   ✅ WorldCover {tile} 读取完成。")
            except Exception as exc:
                self.log_signal.emit(f"   ⚠️ WorldCover {tile} 读取失败：{exc}")
        if successful_tiles == 0:
            raise RuntimeError("ESA WorldCover 农田数据下载失败，未获得任何有效瓦片。")

        protected_pixels = 0
        if protect_water:
            # 保护历史水体及其一个 WorldCover 像元邻域，避免分类边界处的农田
            # 标签把河岸/塘岸真实水体切掉。
            historical_water = cv2.dilate(
                historical_water, np.ones((3, 3), dtype=np.uint8), iterations=1)
            protected_pixels = int(np.count_nonzero(
                (farmland == 1) & (historical_water == 1)))
            farmland[historical_water == 1] = 0
            del historical_water
        farmland[~valid] = 0
        profile = {
            "driver": "GTiff", "height": sar_shape[0], "width": sar_shape[1],
            "count": 1, "dtype": rasterio.uint8, "crs": sar_crs,
            "transform": sar_transform, "nodata": 255, "compress": "lzw",
        }
        cache_output = farmland.copy(); cache_output[~valid] = 255
        with rasterio.open(cache_path, 'w', **profile) as dst:
            dst.write(cache_output, 1)
            dst.update_tags(
                worldcover_version="2021_v200", worldcover_class="40",
                protect_historical_water=expected_protection)
        del cache_output
        self.log_signal.emit(
            f"✅ 农田掩膜覆盖 {np.count_nonzero(farmland)} 个像素；"
            f"历史水体保护排除 {protected_pixels} 个重叠像素。")
        return farmland

    @staticmethod
    def _chaikin_ring_ogr(ring, iterations=2, offset=0.25):
        """使用 NumPy 对 OGR 闭合环执行 Chaikin 拐角切割。"""
        point_count = ring.GetPointCount()
        if point_count < 4:
            return ring.Clone()
        coords = np.empty((point_count, 2), dtype=np.float64)
        for index in range(point_count):
            point = ring.GetPoint(index)
            coords[index] = point[0], point[1]
        if np.allclose(coords[0], coords[-1]):
            coords = coords[:-1]
        if len(coords) < 3:
            return ring.Clone()
        for _ in range(max(1, int(iterations))):
            next_coords = np.roll(coords, -1, axis=0)
            first = (1.0 - offset) * coords + offset * next_coords
            second = offset * coords + (1.0 - offset) * next_coords
            smoothed = np.empty((len(coords) * 2, 2), dtype=np.float64)
            smoothed[0::2] = first
            smoothed[1::2] = second
            coords = smoothed
        output = ogr.Geometry(ogr.wkbLinearRing)
        for x, y in coords:
            output.AddPoint_2D(float(x), float(y))
        output.CloseRings()
        return output

    @classmethod
    def _smooth_polygon_ogr(cls, geometry, iterations=2):
        """平滑 Polygon/MultiPolygon；其它类型原样复制。"""
        geometry_name = geometry.GetGeometryName().upper()
        if geometry_name == "POLYGON":
            output = ogr.Geometry(ogr.wkbPolygon)
            for ring_index in range(geometry.GetGeometryCount()):
                output.AddGeometry(cls._chaikin_ring_ogr(
                    geometry.GetGeometryRef(ring_index), iterations=iterations))
            return output
        if geometry_name == "MULTIPOLYGON":
            output = ogr.Geometry(ogr.wkbMultiPolygon)
            for polygon_index in range(geometry.GetGeometryCount()):
                output.AddGeometry(cls._smooth_polygon_ogr(
                    geometry.GetGeometryRef(polygon_index), iterations=iterations))
            return output
        return geometry.Clone()

    @classmethod
    def polygonize_water_gdal(cls, label_roi, roi_transform, crs, shp_path,
                              smooth_vector=True, smooth_iterations=2):
        height, width = label_roi.shape
        mem = gdal.GetDriverByName("MEM")
        src_ds = mem.Create("", width, height, 1, gdal.GDT_Byte)
        src_ds.SetGeoTransform(roi_transform.to_gdal())
        wkt = crs.to_wkt() if crs is not None else ""
        if wkt:
            src_ds.SetProjection(wkt)
        src_band = src_ds.GetRasterBand(1); src_band.WriteArray(label_roi); src_band.SetNoDataValue(255)
        mask_ds = mem.Create("", width, height, 1, gdal.GDT_Byte)
        mask_band = mask_ds.GetRasterBand(1); mask_band.WriteArray((label_roi == 1).astype(np.uint8))
        driver = ogr.GetDriverByName("ESRI Shapefile")
        if driver is None:
            raise RuntimeError("当前 GDAL 缺少 ESRI Shapefile 驱动。")
        # 某些 Windows GDAL 构建不能通过 OGR 在中文路径直接创建数据源。
        # 先写入系统英文临时目录，关闭数据源后再移动全部伴随文件。
        temp_dir = tempfile.mkdtemp(prefix="sarwater_polygonize_")
        temp_shp = os.path.join(temp_dir, "water_bodies.shp")
        vector_ds = driver.CreateDataSource(temp_shp)
        if vector_ds is None:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise RuntimeError("OGR 无法在临时目录创建 Shapefile 数据源。")
        srs = None
        if wkt:
            srs = osr.SpatialReference(); srs.ImportFromWkt(wkt)
        layer = vector_ds.CreateLayer("water_bodies", srs=srs, geom_type=ogr.wkbPolygon)
        if layer is None:
            vector_ds = None
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise RuntimeError("OGR 无法创建 water_bodies 图层。")
        layer.CreateField(ogr.FieldDefn("water", ogr.OFTInteger))
        err = gdal.Polygonize(src_band, mask_band, layer, 0, ["8CONNECTED=8"], callback=None)
        if err != gdal.CE_None:
            raise RuntimeError(f"GDAL Polygonize 返回错误码 {err}")
        count = layer.GetFeatureCount()
        smoothed_count = 0
        smooth_fallback_count = 0
        if smooth_vector and count:
            # 半个像元的拓扑保持简化先消除密集阶梯节点，再做 Chaikin 平滑。
            # 使用仿射变换行列向量长度的均值，兼容旋转影像和地理/投影坐标系。
            pixel_x = float(np.hypot(roi_transform.a, roi_transform.d))
            pixel_y = float(np.hypot(roi_transform.b, roi_transform.e))
            simplify_tolerance = 0.5 * (pixel_x + pixel_y) / 2.0
            layer.ResetReading()
            layer.StartTransaction()
            # OGR 的 IsValid 会为每个无效孔洞输出一条 GEOS 警告；批处理中静默该
            # 已被回退逻辑处理的提示，避免数万个要素时终端输出拖慢处理。
            gdal.PushErrorHandler("CPLQuietErrorHandler")
            try:
                for feature in layer:
                    original = feature.GetGeometryRef()
                    if original is None or original.IsEmpty():
                        continue
                    simplified = original.SimplifyPreserveTopology(simplify_tolerance)
                    if simplified is None or simplified.IsEmpty():
                        simplified = original.Clone()
                    smoothed = cls._smooth_polygon_ogr(
                        simplified, iterations=smooth_iterations)
                    # 极窄多边形或孔洞可能在曲线化后失效；只对这种少数情况回退到
                    # 拓扑保持简化结果，避免用昂贵的全量 Buffer(0) 修复。
                    if smoothed is None or smoothed.IsEmpty() or not smoothed.IsValid():
                        smoothed = simplified
                        smooth_fallback_count += 1
                    else:
                        smoothed_count += 1
                    feature.SetGeometry(smoothed)
                    layer.SetFeature(feature)
                layer.CommitTransaction()
            finally:
                gdal.PopErrorHandler()
        layer.SyncToDisk()
        vector_ds = mask_ds = src_ds = None
        output_dir = os.path.dirname(shp_path)
        output_base = os.path.splitext(os.path.basename(shp_path))[0]
        for name in os.listdir(temp_dir):
            source = os.path.join(temp_dir, name)
            extension = os.path.splitext(name)[1]
            destination = os.path.join(output_dir, output_base + extension)
            shutil.copy2(source, destination)
        shutil.rmtree(temp_dir, ignore_errors=True)
        return count, smoothed_count, smooth_fallback_count
        
    def plot_professional_map(self, data, bounds, title, cmap, norm, labels_dict,
                              pie_data, pie_colors, alpha, look_dir, save_path, crs=None, bg_image=None):
        """专业级地图绘制函数（支持 1.2 倍缩放 + SAR 背景图叠加）"""
        MAX_DIM = 2000
        h, w = data.shape
        # 1. 计算 1.2 倍缩放后的显示范围 (Padded Extent)
        cx = (bounds.left + bounds.right) / 2
        cy = (bounds.bottom + bounds.top) / 2
        width = (bounds.right - bounds.left) * 1.2  # 扩大 20%
        height = (bounds.top - bounds.bottom) * 1.2
        padded_extent = [
            cx - width / 2, cx + width / 2,
            cy - height / 2, cy + height / 2
        ]
        # 2. 降采样 (保持 data 和 bg_image 尺寸一致)
        if max(h, w) > MAX_DIM:
            scale = MAX_DIM / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            data = cv2.resize(data, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            if bg_image is not None:
                # 背景图使用 AREA 插值，使灰度过渡更平滑
                bg_image = cv2.resize(bg_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        fig = plt.figure(figsize=(12, 9))
        fig.patch.set_facecolor('white')
        # 布局：饼图向左微调，覆盖主图边缘
        ax_map = fig.add_axes([0.05, 0.15, 0.62, 0.80])
        ax_pie = fig.add_axes([0.64, 0.15, 0.25, 0.30])
        legend_ax = fig.add_axes([0.05, 0.05, 0.90, 0.05])
        legend_ax.axis('off')
        # 3. 智能掩膜逻辑 (决定哪些区域透明，露出背景或画布)
        # 背景图掩膜：排除 SAR 范围外的黑边 (假设 SAR 信号 > 0.002)
        mask_bg = (bg_image > 0.002) if bg_image is not None else None
        # 前景图掩膜：根据标题决定显示内容
        if "水体" in title:
            # 水体图：只画水体 (value=1)，背景 (value=0) 和 NoData (value=255) 透明，露出 SAR 底图
            mask_fg = (data == 1)
        elif "畸变" in title:
            # 畸变图：画分类 (1-4)，排除外部 (5) 和无效 (0)
            mask_fg = (data > 0) & (data < 5)
        else:
            mask_fg = None
        # 4. 绘制背景图 (SAR 灰度)
        if bg_image is not None and mask_bg is not None:
            # 构建 RGBA 数组 (高度, 宽度, 4通道)
            h_bg, w_bg = bg_image.shape
            rgba_bg = np.zeros((h_bg, w_bg, 4), dtype=np.uint8)
            # R, G, B 通道都设为灰度值
            rgba_bg[:, :, 0] = bg_image
            rgba_bg[:, :, 1] = bg_image
            rgba_bg[:, :, 2] = bg_image
            # Alpha 通道 (透明度): 
            # mask_bg 为 True 的地方设为 150 (约 60% 不透明度，让背景暗一点)
            # mask_bg 为 False 的地方设为 0 (完全透明)
            rgba_bg[:, :, 3] = np.where(mask_bg, 150, 0).astype(np.uint8)
            # 绘制背景，不再使用 mask 参数
            ax_map.imshow(rgba_bg, extent=padded_extent, origin='upper', 
                        interpolation='nearest', zorder=1)
        # 5. 绘制前景分类图
        # 将 data 转换为 RGBA 颜色
        # 1. 使用 norm 和 cmap 将数据映射为颜色 (返回 0-1 之间的浮点数 RGBA 数组)
        data_float = data.astype(np.float32)
        colors = cmap(norm(data_float)) # 形状为 (H, W, 4)
        rgba_fg = colors.copy()
        # 2. 手动设置 Alpha 通道 (透明度)
        # mask_fg 为 True 的地方，透明度设为 fg_alpha (通常 1.0 或 0.85)
        # mask_fg 为 False 的地方，透明度设为 0.0 (完全透明)
        fg_alpha = 0.85 if "畸变" in title else 1.0
        rgba_fg[:, :, 3] = np.where(mask_fg, fg_alpha, 0.0)
        # 绘制前景，不再使用 mask 参数
        ax_map.imshow(rgba_fg, extent=padded_extent, origin='upper', 
                    interpolation='nearest', zorder=2)
        ax_map.set_aspect('auto')
        ax_map.axis('off')
        ax_map.set_title(title, fontsize=18, fontweight='bold', pad=15)
        # --- 指北针 ---
        ax_map.text(0.05, 0.95, 'N', transform=ax_map.transAxes, fontsize=22, fontweight='bold', ha='center', zorder=12)
        ax_map.plot([0.05, 0.05], [0.88, 0.93], transform=ax_map.transAxes, color='black', lw=2.5, zorder=12)
        ax_map.plot([0.05, 0.04], [0.931, 0.91], transform=ax_map.transAxes, color='black', lw=2, zorder=12)
        ax_map.plot([0.05, 0.06], [0.931, 0.91], transform=ax_map.transAxes, color='black', lw=2, zorder=12)
        # --- 动态比例尺 (基于缩放后的 padded_extent 宽度计算) ---
        if crs and crs.is_geographic:
            xs, ys = warp_transform(crs, "EPSG:3857", [padded_extent[0], padded_extent[1]], [padded_extent[2], padded_extent[2]])
            map_width_m = np.sqrt((xs[1]-xs[0])**2 + (ys[1]-ys[0])**2)
        else:
            map_width_m = padded_extent[1] - padded_extent[0]
        map_width_km = map_width_m / 1000.0
        target_len_km = map_width_km / 5  # 比例尺长度约为图宽的 1/5
        scale_steps = [1, 2, 5, 10, 20, 50, 100, 200, 500]
        bar_len_km = min(scale_steps, key=lambda x: abs(x - target_len_km))
        bar_len_ratio = (bar_len_km * 1000) / map_width_m
        x, y = 0.05, 0.08
        bg = patches.Rectangle((x-0.01, y-0.02), bar_len_ratio + 0.02, 0.05, transform=ax_map.transAxes,
                               facecolor='white', edgecolor='gray', alpha=0.9, zorder=10)
        ax_map.add_patch(bg)
        ax_map.plot([x, x + bar_len_ratio], [y, y], transform=ax_map.transAxes, color='black', lw=3, zorder=11)
        ax_map.plot([x, x], [y-0.01, y+0.01], transform=ax_map.transAxes, color='black', lw=2, zorder=11)
        ax_map.plot([x+bar_len_ratio, x+bar_len_ratio], [y-0.01, y+0.01], transform=ax_map.transAxes, color='black', lw=2, zorder=11)
        ax_map.text(x, y-0.025, '0', ha='center', va='top', fontsize=11, transform=ax_map.transAxes, zorder=11)
        ax_map.text(x+bar_len_ratio, y-0.025, f'{bar_len_km} km', ha='center', va='top', fontsize=11, fontweight='bold', transform=ax_map.transAxes, zorder=11)
        # --- 航向与 LOS 箭头 ---
        heading_az = alpha % 360
        los_az = (alpha + 90) % 360 if look_dir.lower() == "right" else (alpha - 90) % 360
        origin_x, origin_y = 0.85, 0.85
        arrow_len = 0.10
        dx_h = arrow_len * np.sin(np.deg2rad(heading_az))
        dy_h = arrow_len * np.cos(np.deg2rad(heading_az))
        dx_l = arrow_len * np.sin(np.deg2rad(los_az))
        dy_l = arrow_len * np.cos(np.deg2rad(los_az))
        ax_map.plot(origin_x, origin_y, 'ko', markersize=6, transform=ax_map.transAxes, zorder=11)
        ax_map.add_patch(patches.FancyArrowPatch((origin_x, origin_y), (origin_x + dx_h, origin_y + dy_h),
                                                arrowstyle='-|>', mutation_scale=18, color='black', lw=2, transform=ax_map.transAxes, zorder=11))
        ax_map.add_patch(patches.FancyArrowPatch((origin_x, origin_y), (origin_x + dx_l, origin_y + dy_l),
                                                arrowstyle='-|>', mutation_scale=18, color='red', lw=2, transform=ax_map.transAxes, zorder=11))
        ax_map.text(origin_x + dx_h * 0.5 - 0.03, origin_y + dy_h * 0.5 + 0.02, '航向 (Heading)',
                    ha='center', va='center', fontsize=11, fontweight='bold', color='black',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=2), transform=ax_map.transAxes, zorder=12)
        ax_map.text(origin_x + dx_l * 0.5 + 0.03, origin_y + dy_l * 0.5 - 0.02, 'LOS向 (Look)',
                    ha='center', va='center', fontsize=11, fontweight='bold', color='red',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=2), transform=ax_map.transAxes, zorder=12)
        # --- 饼图 ---
        wedges, texts, autotexts = ax_pie.pie(pie_data, colors=pie_colors, startangle=90,
                                              autopct='%1.1f%%', pctdistance=0.75,
                                              wedgeprops=dict(edgecolor='w', linewidth=2), radius=1)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(11)
        ax_pie.set_title("类别占比统计", fontsize=13, fontweight='bold', pad=10)
        # --- 图例 ---
        items = [patches.Patch(color=color, label=label) for label, color in zip(labels_dict.values(), pie_colors)]
        legend_ax.legend(handles=items, ncol=len(items), loc='center', fontsize=12, frameon=False, handlelength=1.5, handleheight=1.5)
        plt.savefig(save_path, dpi=200, facecolor='white')
        plt.close(fig)
        import gc; gc.collect()
        
    def run(self):
        try:
            cfg = self.config
            self.log_signal.emit(f"🚀 开始 SAR 自动化处理管线...")
            self.progress_signal.emit(5)
            self.log_signal.emit(f"📂 目标文件夹: {cfg['folder_path']}")
            # 🔑 中断检查点 1
            if self.isInterruptionRequested():
                self.finished_signal.emit(False, "任务已手动停止")
                return
            # --- 1. 数据准备：文件识别、参数解析、SAR与本地辅助数据对齐 ---
            self.log_signal.emit("🔍 正在自动扫描文件...")
            tiff_files = glob.glob(os.path.join(cfg['folder_path'], "*.tiff")) + glob.glob(os.path.join(cfg['folder_path'], "*.tif"))
            xml_files = glob.glob(os.path.join(cfg['folder_path'], "*.xml"))
            if not tiff_files or not xml_files:
                raise FileNotFoundError("❌ 未在文件夹中找到 .tiff/.tif 或 .xml 文件！")
            sar_tif = tiff_files[0]
            xml_path = xml_files[0]
            self.log_signal.emit(f"✅ 找到影像: {os.path.basename(sar_tif)}")
            self.log_signal.emit(f"✅ 找到参数: {os.path.basename(xml_path)}")
            if cfg.get('use_ui_params', False):
                alpha, theta, look_dir = cfg['alpha'], cfg['theta'], cfg['look_dir']
                self.log_signal.emit("⚙️ 使用界面手动配置的几何参数...")
            else:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                alpha = float(root.find('.//platformHeading').text)
                theta = float(root.find('.//incidenceAngleMidSwath').text)
                look_dir = root.find('.//look_side').text.strip().lower()
                self.log_signal.emit(f" 解析XML参数 -> 航向角:{alpha}, 入射角:{theta}, 观测方向:{look_dir}")
            # 🔑 中断检查点 2
            if self.isInterruptionRequested():
                self.finished_signal.emit(False, "任务已手动停止")
                return
            # 1.1 读取SAR影像及有效范围。
            self.log_signal.emit("\n📥 正在读取 SAR 影像作为空间基准...")
            with rasterio.open(sar_tif) as src:
                raw = src.read(1).astype(np.float32)
                sar_profile = src.meta.copy()
                sar_transform = src.transform
                sar_crs = src.crs
                sar_shape = raw.shape
                sar_bounds = src.bounds
            zero_mask_start = time.perf_counter()
            valid, signal_valid, internal_zero_gap_count = self.build_sar_coverage_mask(raw)
            raw_work, analysis_valid, repaired_gap_count, neutral_value = \
                self.neutralize_internal_sar_zeros(raw, valid, signal_valid)
            exterior_count = int(np.count_nonzero(~valid))
            self.log_signal.emit(
                f"✅ SAR覆盖区占比: {np.sum(valid) / raw.size * 100:.2f}%；"
                f"外围黑边 {exterior_count} 像素；内部异常零值 {repaired_gap_count} 像素"
                f"仅作数值中和（抽样中位数 {neutral_value:.6g}），未执行空间插值；"
                f"覆盖掩膜与中和共耗时 {time.perf_counter() - zero_mask_start:.2f} 秒。")
            del signal_valid, internal_zero_gap_count
            #  新增：将 SAR 影像边界转换为 WGS84 经纬度
            self.log_signal.emit("\n🌍 正在转换 SAR 影像边界至 WGS84 坐标系...")
            if sar_crs is not None:
                try:
                    # 使用 rasterio.warp.transform_bounds 进行坐标转换
                    wgs84_bounds = transform_bounds(
                        sar_crs, 
                        "EPSG:4326",  # WGS84
                        sar_bounds.left, 
                        sar_bounds.bottom, 
                        sar_bounds.right, 
                        sar_bounds.top,
                        densify_pts=21  # 增加采样点以提高精度
                    )
                    self.log_signal.emit(f"✅ SAR 影像范围 (原始坐标系): Lon[{sar_bounds.left:.4f}, {sar_bounds.right:.4f}], Lat[{sar_bounds.bottom:.4f}, {sar_bounds.top:.4f}]")
                    self.log_signal.emit(f"✅ SAR 影像范围 (WGS84): Lon[{wgs84_bounds[0]:.6f}, {wgs84_bounds[2]:.6f}], Lat[{wgs84_bounds[1]:.6f}, {wgs84_bounds[3]:.6f}]")
                    # 计算中心点
                    center_lon = (wgs84_bounds[0] + wgs84_bounds[2]) / 2
                    center_lat = (wgs84_bounds[1] + wgs84_bounds[3]) / 2
                    self.log_signal.emit(f"📍 影像中心点 (WGS84): Lon={center_lon:.6f}, Lat={center_lat:.6f}")
                    # 计算覆盖范围
                    lon_span = wgs84_bounds[2] - wgs84_bounds[0]
                    lat_span = wgs84_bounds[3] - wgs84_bounds[1]
                    self.log_signal.emit(f"📏 覆盖范围: 经度跨度={lon_span:.6f}°, 纬度跨度={lat_span:.6f}°")
                except Exception as e:
                    self.log_signal.emit(f"⚠️ 坐标转换失败: {str(e)}")
                    wgs84_bounds = sar_bounds
            else:
                self.log_signal.emit("⚠️ SAR 影像缺少坐标系信息，无法转换")
                wgs84_bounds = sar_bounds
            building_mask = None
            road_mask = None
            farmland_mask = None
            need_osm = (cfg.get('enable_building_mask', False) or
                        cfg.get('enable_road_mask', False))
            need_farmland = cfg.get('enable_farmland_mask', False)
            need_terrain = cfg.get('enable_terrain', False)
            if need_osm or need_farmland or need_terrain:
                if sar_crs is None:
                    raise ValueError("启用道路/建筑/农田/地形掩膜时，SAR 必须包含坐标系信息。")
                tight_bounds = self.valid_data_bounds(valid, sar_transform)
                context_bounds = transform_bounds(
                    sar_crs, "EPSG:4326", *tight_bounds, densify_pts=21)
                os.makedirs(cfg['output_dir'], exist_ok=True)

            if need_osm:
                self.log_signal.emit(
                    f"📍 OSM 下载范围: Lon[{context_bounds[0]:.6f}, {context_bounds[2]:.6f}], "
                    f"Lat[{context_bounds[1]:.6f}, {context_bounds[3]:.6f}]")
                building_mask, road_mask = self.build_osm_masks(
                    cfg, context_bounds, sar_crs, sar_transform, sar_shape, valid)
                mask_meta = sar_profile.copy()
                mask_meta.update(dtype=rasterio.uint8, count=1, nodata=255, compress='lzw')
                if cfg.get('enable_building_mask', False):
                    output = building_mask.copy(); output[~valid] = 255
                    with rasterio.open(os.path.join(cfg['output_dir'], "building_mask.tif"), 'w', **mask_meta) as dst:
                        dst.write(output, 1)
                    del output
                if cfg.get('enable_road_mask', False):
                    output = road_mask.copy(); output[~valid] = 255
                    with rasterio.open(os.path.join(cfg['output_dir'], "road_mask.tif"), 'w', **mask_meta) as dst:
                        dst.write(output, 1)
                    del output

            if need_farmland:
                farmland_mask = self.build_farmland_mask(
                    cfg, context_bounds, sar_crs, sar_transform, sar_shape, valid)
            # 🔑 中断检查点 3
            if self.isInterruptionRequested():
                self.finished_signal.emit(False, "任务已手动停止")
                return
            # 1.2 准备与SAR网格对齐的地形排除掩膜，阶段3再应用。
            exclude_mask = None
            aligned_cls = None
            if cfg['enable_terrain']:
                slope_tif, aspect_tif = self.prepare_terrain_layers(
                    cfg, context_bounds, sar_crs)
                self.log_signal.emit("\n🏔️ 正在把坡度/坡向对齐到SAR网格并计算地形畸变...")
                # 1. 读取坡度/坡向地形成果
                with rasterio.open(slope_tif) as src:
                    # masked=True 同时识别数据集 nodata 和内部 mask，不能只依赖数值阈值。
                    slope_ma = src.read(1, masked=True).astype(np.float32)
                    slope = slope_ma.filled(np.nan)
                    slope_valid = (~np.ma.getmaskarray(slope_ma)) & np.isfinite(slope)
                    slope_transform = src.transform
                    slope_crs = src.crs
                with rasterio.open(aspect_tif) as src:
                    aspect_ma = src.read(1, masked=True).astype(np.float32)
                    aspect = aspect_ma.filled(np.nan)
                    aspect_valid = (~np.ma.getmaskarray(aspect_ma)) & np.isfinite(aspect)
                    aspect_transform = src.transform
                    aspect_crs = src.crs
                # 坡度/坡向的物理范围校验可补捉未在元数据中声明的 -9999 等无效值。
                slope_valid &= (slope >= 0.0) & (slope <= 90.0)
                aspect_valid &= (aspect >= 0.0) & (aspect <= 360.0)
                slope[~slope_valid] = np.nan
                aspect[~aspect_valid] = np.nan
                # 2. 🔑 核心优化：创建 SAR 高分辨率目标网格 (自动实现裁剪与对齐)
                aligned_slope = np.full(sar_shape, np.nan, dtype=np.float32)
                aligned_aspect = np.full(sar_shape, np.nan, dtype=np.float32)
                aligned_slope_valid = np.zeros(sar_shape, dtype=np.uint8)
                aligned_aspect_valid = np.zeros(sar_shape, dtype=np.uint8)
                # 3. 使用 Rasterio C 底层重采样
                # 坡度使用双线性插值 (Bilinear)，保证坡度值平滑过渡
                reproject(source=slope, destination=aligned_slope,
                        src_transform=slope_transform, src_crs=slope_crs if slope_crs else sar_crs,
                        dst_transform=sar_transform, dst_crs=sar_crs,
                        src_nodata=np.nan, dst_nodata=np.nan, init_dest_nodata=True,
                        resampling=Resampling.bilinear)
                # 坡向使用最近邻，避免跨越0°/360°时被普通线性插值成错误方向。
                reproject(source=aspect, destination=aligned_aspect,
                        src_transform=aspect_transform, src_crs=aspect_crs if aspect_crs else sar_crs,
                        dst_transform=sar_transform, dst_crs=sar_crs,
                        src_nodata=np.nan, dst_nodata=np.nan, init_dest_nodata=True,
                        resampling=Resampling.nearest)
                # 单独重投影有效性掩膜，严格保留 DEM 原始空值 footprint。
                # 这样水面上的空 slope/aspect 不会被邻近陆地的双线性插值“补成有效”。
                reproject(source=slope_valid.astype(np.uint8), destination=aligned_slope_valid,
                        src_transform=slope_transform, src_crs=slope_crs if slope_crs else sar_crs,
                        dst_transform=sar_transform, dst_crs=sar_crs,
                        src_nodata=0, dst_nodata=0, init_dest_nodata=True,
                        resampling=Resampling.nearest)
                reproject(source=aspect_valid.astype(np.uint8), destination=aligned_aspect_valid,
                        src_transform=aspect_transform, src_crs=aspect_crs if aspect_crs else sar_crs,
                        dst_transform=sar_transform, dst_crs=sar_crs,
                        src_nodata=0, dst_nodata=0, init_dest_nodata=True,
                        resampling=Resampling.nearest)
                # 释放原始低分辨率内存
                del slope, aspect, slope_valid, aspect_valid, slope_ma, aspect_ma
                # 4. 在SAR网格上计算局部入射角与畸变分类
                self.log_signal.emit("📐 正在SAR网格上计算局部入射角...")
                rad_alpha = np.deg2rad(alpha)
                rad_theta = np.deg2rad(theta)
                rad_aspect = np.deg2rad(aligned_aspect)
                rad_slope = np.deg2rad(aligned_slope)
                look_az = (alpha + 90) % 360 if look_dir.lower() == 'right' else (alpha - 90) % 360
                # 将方向差规范到[-180°, 180°)，正确处理359°与1°跨北方向的情况。
                phi = (aligned_aspect - look_az + 180.0) % 360.0 - 180.0
                # 计算 cos_xi (局部入射角余弦)
                cos_xi = (np.cos(rad_theta) * np.cos(rad_slope) +
                        np.sin(rad_theta) * np.sin(rad_slope) * np.cos(np.deg2rad(phi)))
                # 5. 生成高分辨率分类掩膜
                classification = np.zeros(sar_shape, dtype=np.uint8)
                valid_terrain = ((aligned_slope_valid == 1) &
                                 (aligned_aspect_valid == 1) &
                                 np.isfinite(aligned_slope) &
                                 np.isfinite(aligned_aspect) &
                                 np.isfinite(cos_xi))
                classification[~valid_terrain] = 0  # 无效区
                classification[valid_terrain & (cos_xi <= 0)] = 1  # 阴影
                facing = (cos_xi > 0) & (np.abs(phi) <= 90)
                classification[valid_terrain & facing & (aligned_slope > theta)] = 2  # 叠掩
                classification[valid_terrain & facing & (aligned_slope <= theta) & (aligned_slope > cfg['slope_thresh'])] = 3  # 透视收缩/高坡度
                classification[valid_terrain & (classification == 0)] = 4  # 正常可视区
                # 6. 构建最终排除掩膜 (非正常区)
                # 只允许“有有效 DEM 证据”的畸变/高坡区域参与排除。
                # classification==0 表示 DEM 无值/未知，必须保留 SAR 提取结果。
                exclude_mask = valid_terrain & (classification != 4)
                aligned_cls = classification  # 用于后续出图
                dem_valid_count = np.count_nonzero(valid_terrain & valid)
                terrain_excluded_count = np.count_nonzero(exclude_mask & valid)
                dem_unknown_count = np.count_nonzero((~valid_terrain) & valid)
                safe_ratio = ((dem_valid_count - terrain_excluded_count) / dem_valid_count * 100
                              if dem_valid_count else 0.0)
                unknown_ratio = dem_unknown_count / np.count_nonzero(valid) * 100
                self.log_signal.emit(
                    f"✅ 地形畸变计算完成！DEM有效区安全占比: {safe_ratio:.2f}%；"
                    f"DEM无值区占SAR有效区: {unknown_ratio:.2f}%（按未知处理，不删除水体）"
                )
                self.progress_signal.emit(40)
            else:
                self.log_signal.emit("\n⚠️ 地形滤除开关已关闭，跳过 DEM 读取...")
            # 🔑 中断检查点 4
            if self.isInterruptionRequested():
                self.finished_signal.emit(False, "任务已手动停止")
                return
            # --- 2. 基础预测：Lee滤波、百分位归一化、阈值分割（不等于辐射校正） ---
            self.log_signal.emit("\n💧 执行 Lee 滤波去斑...")
            fill_val = neutral_value
            temp = raw_work.copy(); temp[~valid] = fill_val
            kernel = np.ones((cfg['lee_win'], cfg['lee_win']), dtype=np.float32) / (cfg['lee_win'] ** 2)
            m1 = cv2.filter2D(temp, -1, kernel); m2 = cv2.filter2D(temp ** 2, -1, kernel)
            local_var = np.maximum(m2 - m1 ** 2, 0.0)
            noise = np.var(raw_work[analysis_valid]) * cfg['noise_ratio']
            k = local_var / (local_var + noise + 1e-10)
            lee = m1 + k * (temp - m1); lee[~valid] = np.nan
            self.log_signal.emit("📐 执行 dB 对数转换与百分位归一化...")
            db = np.full_like(lee, np.nan, dtype=np.float32)
            db[analysis_valid] = 10.0 * np.log10(lee[analysis_valid] + 1e-6)
            valid_db = db[analysis_valid]
            low, high = np.percentile(valid_db, [cfg['p_low'], cfg['p_high']])
            norm = (db - low) / (high - low + 1e-6)
            norm = np.clip(norm, 0.0, 1.0); norm[~analysis_valid] = np.nan
            self.progress_signal.emit(60)
            #  中断检查点 5
            if self.isInterruptionRequested():
                self.finished_signal.emit(False, "任务已手动停止")
                return
            # 2.1 生成原始水体标签；此时尚未施加辅助排除与形态学。
            # 🔑 新增：高反射强度水体提取模式判断
            extraction_mode = cfg.get('extraction_mode', 'bottom')  # 'bottom' 或 'top'
            water_pct = cfg['water_pct']
            
            if extraction_mode == 'top':
                self.log_signal.emit(f"\n🏷️ 生成标签 - 高反射强度提取模式 (顶部 {water_pct}%，应用掩膜)...")
            else:
                self.log_signal.emit(f"\n️ 生成标签 - 低反射强度提取模式 (底部 {water_pct}%，应用掩膜)...")
                
            label = np.full(norm.shape, 255, dtype=np.uint8)
            valid_norm = norm[analysis_valid]
            
            # 🔑 核心逻辑：直接在源头决定提取底部还是顶部
            if extraction_mode == 'top':
                # 高反射强度提取：取顶部 water_pct% (高亮度区域)
                # 例如 water_pct=25，则取最高的 25%
                water_mask, thresh = percentile_mask(valid_norm, np.ones(valid_norm.shape, dtype=bool), water_pct, "top")
                # water_mask 已由共用百分位函数计算  # 大于等于阈值的为水体
                self.log_signal.emit(f"   阈值计算: 取顶部 {water_pct}% (亮度 >= {thresh:.4f})")
            else:
                # 低反射强度提取：取底部 water_pct% (低亮度区域，默认模式)
                water_mask, thresh = percentile_mask(valid_norm, np.ones(valid_norm.shape, dtype=bool), water_pct, "bottom")
                # water_mask 已由共用百分位函数计算  # 小于阈值的为水体
                self.log_signal.emit(f"   阈值计算: 取底部 {water_pct}% (亮度 < {thresh:.4f})")
                
            label[valid] = 0
            label[analysis_valid] = water_mask.astype(np.uint8)
            # 此处暂不应用地形/道路/建筑强制排除掩膜，也不做形态学和面积筛选。
            # 先填原始提取结果中的小孔，避免把后续人为掩膜形成的空区误当作孔洞补回。
            label[~valid] = 255
            self.progress_signal.emit(75)
            # 🔑 中断检查点 6
            if self.isInterruptionRequested():
                self.finished_signal.emit(False, "任务已手动停止")
                return
            # --- 3. 后处理：填孔 → 辅助排除 → 形态学并重施掩膜 → 连通域筛选 ---
            # 辅助数据用途：DEM/DSM→坡度/坡向，结合航向/入射角/视向构造地形排除候选；
            # OSM建筑面/道路缓冲→建筑与道路排除；WorldCover 40类→农田排除，80类→历史水体保护。
            # 水深另需DEM/DSM高程本身，不能以坡度/坡向文件代替；在成果分析时执行。
            # 后处理1：仅填有效区内小孔；NoData、影像边界与排除区不能当作孔洞补回。
            fill_holes_enabled = cfg.get('fill_holes', False) and cfg.get('hole_thresh', 100) > 0
            if fill_holes_enabled:
                hole_start = time.perf_counter()
                water_rows = np.flatnonzero(np.any(label == 1, axis=1))
                water_cols = np.flatnonzero(np.any(label == 1, axis=0))
                if water_rows.size and water_cols.size:
                    r0, r1 = max(int(water_rows[0]) - 1, 0), min(int(water_rows[-1]) + 2, sar_shape[0])
                    c0, c1 = max(int(water_cols[0]) - 1, 0), min(int(water_cols[-1]) + 2, sar_shape[1])
                    roi = label[r0:r1, c0:c1]
                    roi_valid = valid[r0:r1, c0:c1]
                    # 只分析 SAR 有效区内的普通背景值 0；原始 NoData(255) 完全不参与
                    # 填孔，避免将影像数据漏洞和道路周边无值区误补成大块水体。
                    valid_background = ((roi == 0) & roi_valid).astype(np.uint8)
                    count, hole_ids, stats, _ = cv2.connectedComponentsWithStats(
                        valid_background, connectivity=8)
                    areas = stats[:, cv2.CC_STAT_AREA]
                    fill_id = areas <= cfg['hole_thresh']; fill_id[0] = False
                    border_ids = np.unique(np.concatenate(
                        (hole_ids[0], hole_ids[-1], hole_ids[:, 0], hole_ids[:, -1])))
                    fill_id[border_ids] = False
                    # 紧邻原始 NoData 的有效背景也不视为可靠孔洞，防止水体沿数据漏洞边缘扩张。
                    nodata_neighbors = cv2.dilate(
                        (~roi_valid).astype(np.uint8), np.ones((3, 3), dtype=np.uint8),
                        iterations=1).astype(bool)
                    nodata_touch_ids = np.unique(
                        hole_ids[nodata_neighbors & (valid_background == 1)])
                    fill_id[nodata_touch_ids] = False
                    fill_pixels = fill_id[hole_ids]
                    roi[fill_pixels & roi_valid] = 1
                    # 只有经孔洞面积规则确认并填充的内部零值，才提升为后续可保留像素；
                    # 未填充的内部零值在形态学后仍会恢复为背景，防止闭运算越权补洞。
                    roi_analysis_valid = analysis_valid[r0:r1, c0:c1]
                    roi_analysis_valid[fill_pixels & roi_valid] = True
                    # 无论前面做过什么处理，原始 SAR 无效区始终恢复成 NoData。
                    roi[~roi_valid] = 255
                    self.log_signal.emit(
                        f"✅ 填充 {np.count_nonzero(fill_id)} 个有效区小孔洞，耗时 "
                        f"{time.perf_counter() - hole_start:.2f} 秒；原始NoData全部保持不变。")
                    del (valid_background, hole_ids, stats, areas, fill_id, border_ids,
                         nodata_neighbors, nodata_touch_ids, fill_pixels,
                         roi_analysis_valid)

            # 后处理2：应用建筑、道路、地形与可选农田排除。
            if cfg.get('enable_building_mask', False):
                before = int(np.count_nonzero((label == 1) & (building_mask == 1)))
                label[building_mask == 1] = 0
                self.log_signal.emit(f"🏢 填孔后重新应用建筑掩膜：删除 {before} 个水体像素。")
            if cfg.get('enable_road_mask', False):
                before = int(np.count_nonzero((label == 1) & (road_mask == 1)))
                label[road_mask == 1] = 0
                self.log_signal.emit(f"🛣️ 填孔后重新应用道路掩膜：删除 {before} 个水体像素。")

            if exclude_mask is not None:
                before = int(np.count_nonzero((label == 1) & exclude_mask))
                label[exclude_mask] = 0
                self.log_signal.emit(f"⛰️ 填孔后应用有效DEM地形掩膜：删除 {before} 个水体像素。")
            farmland_mode = cfg.get('farmland_mode', 'conservative')
            if farmland_mask is not None and farmland_mode == 'all':
                before = int(np.count_nonzero((label == 1) & (farmland_mask == 1)))
                label[farmland_mask == 1] = 0
                self.log_signal.emit(f"🌾 全量农田掩膜：删除 {before} 个水体像素。")

            # 后处理3：形态学必须放在所有强制排除掩膜之后，才能清除掩膜边缘的毛刺和窄小残片。
            # 但闭运算可能跨越道路等窄带重新连接水体，所以运算后必须再次施加全部掩膜。
            if cfg['enable_morphology']:
                morph_start = time.perf_counter()
                morphology_input = (label == 1).astype(np.uint8)
                k_morph = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (cfg['morph_kernel_size'], cfg['morph_kernel_size']))
                morphology_input = cv2.morphologyEx(
                    morphology_input, cv2.MORPH_OPEN, k_morph)
                morphology_input = cv2.morphologyEx(
                    morphology_input, cv2.MORPH_CLOSE, k_morph)
                label[valid] = morphology_input[valid]
                del morphology_input
                self.log_signal.emit(
                    f"🧹 掩膜后形态学清理完成，耗时 "
                    f"{time.perf_counter() - morph_start:.2f} 秒。")

            # 形态学后再次强制恢复所有排除区，确保道路、建筑、地形掩膜不会被闭运算补回。
            label[~analysis_valid] = 0
            if exclude_mask is not None:
                label[exclude_mask] = 0
            if cfg.get('enable_building_mask', False):
                label[building_mask == 1] = 0
            if cfg.get('enable_road_mask', False):
                label[road_mask == 1] = 0
            if farmland_mask is not None and farmland_mode == 'all':
                label[farmland_mask == 1] = 0

            # 后处理4：最小连通面积筛选必须是矢量化之前的最后一道水体清理：道路/建筑/地形
            # 掩膜切割后新产生的小碎片也会在这里被删除。
            if cfg['min_area'] > 0 or (farmland_mask is not None and
                                      farmland_mode == 'conservative'):
                area_start = time.perf_counter()
                water_binary = (label == 1).astype(np.uint8)
                num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(
                    water_binary, connectivity=8)
                component_areas = stats[:, cv2.CC_STAT_AREA]
                keep_component = component_areas >= max(cfg['min_area'], 0)
                keep_component[0] = False
                farmland_removed_count = 0
                if farmland_mask is not None and farmland_mode == 'conservative':
                    # 分行块统计各连通域与农田的交叠量，避免3亿像素全图
                    # float64 weights 临时数组造成数GB峰值内存。
                    farmland_pixels = np.zeros(num_labels, dtype=np.int64)
                    overlap_block_rows = 512
                    for block_row in range(0, sar_shape[0], overlap_block_rows):
                        block_end = min(block_row + overlap_block_rows, sar_shape[0])
                        block_farmland = farmland_mask[block_row:block_end] == 1
                        if np.any(block_farmland):
                            block_labels = labels_im[block_row:block_end][block_farmland]
                            farmland_pixels += np.bincount(
                                block_labels, minlength=num_labels)
                            del block_labels
                        del block_farmland
                    overlap_ratio = np.divide(
                        farmland_pixels, component_areas,
                        out=np.zeros(num_labels, dtype=np.float64),
                        where=component_areas > 0)
                    farmland_remove = (
                        (component_areas <= cfg.get('farmland_small_area', 1000)) &
                        (overlap_ratio >= cfg.get('farmland_overlap_ratio', 0.5)))
                    farmland_remove[0] = False
                    farmland_removed_count = int(np.count_nonzero(
                        keep_component & farmland_remove))
                    keep_component[farmland_remove] = False
                    del farmland_pixels, overlap_ratio, farmland_remove
                kept_count = int(np.count_nonzero(keep_component))
                removed_count = max(num_labels - 1 - kept_count, 0)
                cleaned_water = keep_component[labels_im]
                label[valid] = cleaned_water[valid]
                del (water_binary, labels_im, stats, component_areas,
                     keep_component, cleaned_water)
                self.log_signal.emit(
                    f"🧽 最终最小连通面积筛选完成：保留 {kept_count} 个水体，"
                    f"删除 {removed_count} 个小碎片，耗时 "
                    f"{time.perf_counter() - area_start:.2f} 秒。")
                if farmland_mask is not None and farmland_mode == 'conservative':
                    self.log_signal.emit(
                        f"🌾 保守农田筛选：其中删除 {farmland_removed_count} 个"
                        f"农田重叠比例不低于 {cfg.get('farmland_overlap_ratio', 0.5)*100:.0f}%、"
                        f"面积不超过 {cfg.get('farmland_small_area', 1000)} 像素的水体斑块。")

            # 所有清理完成后恢复外围 NoData，并统计真正的最终结果。
            label[~valid] = 255
            water_ratio = np.sum(label == 1) / np.sum(label != 255) * 100
            self.log_signal.emit(f"✅ 最终有效水体占比: {water_ratio:.2f}%")
            self.progress_signal.emit(80)
            # 防御性校验：最终水体不得与已启用的强制排除掩膜重叠。
            if cfg.get('enable_building_mask', False):
                assert not np.any((label == 1) & (building_mask == 1)), "建筑掩膜应用失败"
            if cfg.get('enable_road_mask', False):
                assert not np.any((label == 1) & (road_mask == 1)), "道路掩膜应用失败"
            if farmland_mask is not None and farmland_mode == 'all':
                assert not np.any((label == 1) & (farmland_mask == 1)), "农田掩膜应用失败"
            # 🔑 中断检查点 7
            if self.isInterruptionRequested():
                self.finished_signal.emit(False, "任务已手动停止")
                return
            # --- 4. 成果输出：最终掩膜/矢量、面积/历史变化/几何水深、地图 ---
            os.makedirs(cfg['output_dir'], exist_ok=True)
            out_label = os.path.join(cfg['output_dir'], "SARWater.tif")
            label_meta = sar_profile.copy()
            label_meta.update(dtype=rasterio.uint8, nodata=255, compress='lzw')
            self.log_signal.emit("🔄 正在栅格转矢量...")
            vector_start = time.perf_counter()
            with rasterio.open(out_label, 'w', **label_meta) as dst:
                dst.write(label, 1)
            shp_path = os.path.join(cfg['output_dir'], "water_bodies.shp")
            shp_driver = ogr.GetDriverByName("ESRI Shapefile")
            if os.path.exists(shp_path): shp_driver.DeleteDataSource(shp_path)
            water_rows = np.flatnonzero(np.any(label == 1, axis=1))
            water_cols = np.flatnonzero(np.any(label == 1, axis=0))
            feature_count = smoothed_count = smooth_fallback_count = 0
            if water_rows.size and water_cols.size:
                r0, r1 = int(water_rows[0]), int(water_rows[-1]) + 1
                c0, c1 = int(water_cols[0]), int(water_cols[-1]) + 1
                label_roi = np.ascontiguousarray(label[r0:r1, c0:c1])
                roi_transform = sar_transform * rasterio.Affine.translation(c0, r0)
                feature_count, smoothed_count, smooth_fallback_count = self.polygonize_water_gdal(
                    label_roi, roi_transform, sar_crs, shp_path,
                    smooth_vector=cfg.get('smooth_vector', True),
                    smooth_iterations=cfg.get('smooth_iterations', 2))
                del label_roi
            self.log_signal.emit(
                f"✅ GDAL栅格转矢量完成！共 {feature_count} 个多边形，"
                f"耗时 {time.perf_counter() - vector_start:.2f} 秒。")
            if cfg.get('smooth_vector', True) and feature_count:
                self.log_signal.emit(
                    f"〰️ 矢量边界平滑完成：{smoothed_count} 个曲线化，"
                    f"{smooth_fallback_count} 个因几何有效性回退为拓扑简化边界。")
            # 矢量输出后生成无真值核查包；沿用最终分类，不把外部参考当真值。
            from thres_segmentation.quality_review import export_quality_review, load_review_options
            review=dict(cfg.get('quality_review_options') or load_review_options())
            if review.pop('enabled',False):
                from datetime import datetime
                review_dir=Path(cfg['output_dir'])/'quality_review'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                export_quality_review(out_label,review_dir,scene_id=Path(sar_tif).stem,
                    metadata={'source_image':str(sar_tif),'water_vector':str(shp_path),
                              'stage':'after_postprocessing'},log=self.log_signal.emit,**review)
            from thres_segmentation.water_analysis import analyze_water
            analysis_options = dict(cfg.get('analysis_options') or {})
            if cfg.get('enable_depth') and not analysis_options.get('elevation_path'):
                analysis_options['elevation_path'] = getattr(self, 'analysis_elevation_path', None)
                if not analysis_options['elevation_path']:
                    raise ValueError('水深分析需要DEM/DSM高程，不能使用坡度/坡向代替')
                if cfg.get('terrain_source_mode') == 'download':
                    analysis_options['elevation_kind'] = 'DSM'
            from datetime import datetime
            analysis_dir = os.path.join(cfg['output_dir'], 'analysis_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
            self.log_signal.emit('正在统计面积、历史变化和可选水深...')
            result = analyze_water(out_label, analysis_dir, cancel=self.isInterruptionRequested, **analysis_options)
            if result['current_water']:
                self.log_signal.emit(f"当前水体面积：{result['current_water']['km2']:.6f} km²")
            if 'change' in result:
                self.log_signal.emit(f"新增：{result['change']['new_water']['km2']:.6f} km²；"
                                     f"消退：{result['change']['lost_water']['km2']:.6f} km²（共同有效覆盖内）")
            self.log_signal.emit(f'分析输出：{analysis_dir}')
            self.progress_signal.emit(95)
            # 🔑 中断检查点 8
            if self.isInterruptionRequested():
                self.finished_signal.emit(False, "任务已手动停止")
                return
            # 4.1 基于最终掩膜制图；面积/变化以分析栅格为准，不以平滑矢量为准。
            self.log_signal.emit("\n🎨 正在生成专业级汇报地图 (1.2倍视角 + SAR背景)...")
            # 利用阶段2生成的norm数据作为背景图。
            # norm 范围 0-1，NaN 为无效区。转换为 0-255 的 uint8 灰度图
            # np.nan_to_num 将 NaN 转为 0 (黑色)，后续绘图时通过 mask 排除
            bg_image = (np.nan_to_num(norm, nan=0.0) * 255).astype(np.uint8)
            # 8.1 绘制地形畸变分类图
            if cfg['enable_terrain'] and aligned_cls is not None:
                # 处理 SAR 掩膜范围，将外部区域设为 5 (白色/无效)
                aligned_cls_plot = aligned_cls.copy()
                aligned_cls_plot[~valid] = 5 
                cls_cmap = plt.cm.colors.ListedColormap(['#2F2F2F', '#FF3333', '#FF9900', '#FFCC00', '#32CD32', '#FFFFFF'])
                cls_norm = plt.cm.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5], cls_cmap.N)
                cls_labels = {0:'无效/高坡度', 1:'阴影区(Shadow)', 2:'叠掩区(Layover)', 3:'透视收缩(Foreshortening)', 4:'正常可视区(Normal)'}
                cls_colors = ['#2F2F2F', '#FF3333', '#FF9900', '#FFCC00', '#32CD32']
                # 统计饼图 (只统计 SAR 有效范围内，即值 < 5 的像素)
                valid_terrain_pixels = np.sum(aligned_cls_plot < 5)
                pie_cls_data = []
                for i in range(5):
                    count = np.sum(aligned_cls_plot == i)
                    pie_cls_data.append((count / valid_terrain_pixels * 100) if valid_terrain_pixels > 0 else 0)
                terrain_map_path = os.path.join(cfg['output_dir'], "Map_Terrain_Distortion.png")
                self.plot_professional_map(
                    data=aligned_cls_plot, bounds=sar_bounds, title="SAR 地形几何畸变分类图",
                    cmap=cls_cmap, norm=cls_norm, labels_dict=cls_labels,
                    pie_data=pie_cls_data, pie_colors=cls_colors,
                    alpha=alpha, look_dir=look_dir,
                    save_path=terrain_map_path, crs=sar_crs,
                    bg_image=bg_image  # 🔑 传入背景图
                )
                self.log_signal.emit(f"️ 地形畸变图已保存: {terrain_map_path}")
            # 8.2 绘制水体提取结果图
            water_cmap = plt.cm.colors.ListedColormap(['#E0E0E0', '#00BFFF', '#F5F5F5'])
            water_norm = plt.cm.colors.BoundaryNorm([-0.5, 0.5, 1.5, 255.5], water_cmap.N)
            water_labels = {0:'背景/掩膜区', 1:'提取水体', 255:'NoData'}
            water_colors = ['#E0E0E0', '#00BFFF', '#F5F5F5']
            valid_pixels = np.sum(label != 255)
            water_pixels = np.sum(label == 1)
            bg_pixels = valid_pixels - water_pixels
            nodata_pixels = np.sum(label == 255)
            # 饼图统计 (基于有效像素)
            pie_water_data = [
                bg_pixels / valid_pixels * 100 if valid_pixels > 0 else 0,
                water_pixels / valid_pixels * 100 if valid_pixels > 0 else 0,
                nodata_pixels / label.size * 100
            ]
            water_map_path = os.path.join(cfg['output_dir'], "Map_Water_Extraction.png")
            self.plot_professional_map(
                data=label, bounds=sar_bounds, title="SAR 水体自动化提取结果图",
                cmap=water_cmap, norm=water_norm, labels_dict=water_labels,
                pie_data=pie_water_data, pie_colors=water_colors,
                alpha=alpha, look_dir=look_dir,
                save_path=water_map_path, crs=sar_crs,
                bg_image=bg_image  # 🔑 传入背景图
            )
            self.log_signal.emit(f"🖼️ 水体提取图已保存: {water_map_path}")
            self.progress_signal.emit(100)
            self.finished_signal.emit(True, "处理完成！请查看输出目录。")
            del raw, raw_work, analysis_valid, lee, db, norm, label
            import gc; gc.collect()
        except Exception as e:
            import traceback
            self.finished_signal.emit(False, f"处理失败: {str(e)}\n{traceback.format_exc()}")
