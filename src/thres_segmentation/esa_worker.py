"""从原 SAR GUI 脚本迁移，保留已有处理逻辑。"""
import os
import glob
import math
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio import features
import geopandas as gpd
from shapely.geometry import shape
from PySide6.QtCore import QThread, Signal
from thres_segmentation.gis_worker import SARWorker

class ESAWaterWorker(QThread):
    log_signal = Signal(str)
    finished_signal = Signal(bool, str)
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
    def get_esa_tiles(self, bounds):
        """计算需要的 ESA WorldCover 瓦片编号"""
        import math
        min_lon, min_lat, max_lon, max_lat = bounds
        start_lat = int(math.floor(min_lat / 3.0) * 3)
        end_lat = int(math.floor(max_lat / 3.0) * 3) + 3
        start_lon = int(math.floor(min_lon / 3.0) * 3)
        end_lon = int(math.floor(max_lon / 3.0) * 3) + 3
        tiles = []
        for lat in range(start_lat, end_lat, 3):
            for lon in range(start_lon, end_lon, 3):
                ns = 'N' if lat >= 0 else 'S'
                ew = 'E' if lon >= 0 else 'W'
                tile_name = f"{ns}{abs(lat):02d}{ew}{abs(lon):03d}"
                tiles.append(tile_name)
        return tiles
        
    def download_esa_water(self, tiles, target_bounds, target_crs="EPSG:4326"):
        """下载 ESA 水体数据并生成二值掩膜"""
        import math
        sources = []
        self.log_signal.emit(f"📥 计划下载 {len(tiles)} 个瓦片...")
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
            for tile in tiles:
                # 🔑 中断检查点
                if self.isInterruptionRequested():
                    return None, None, None
                url = f"https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
                self.log_signal.emit(f"    获取: {tile} ...")
                try:
                    with rasterio.open(url) as src:
                        data = src.read(1)
                        transform = src.transform
                        crs = src.crs
                        sources.append((data, transform, crs))
                        self.log_signal.emit("OK")
                except Exception as e:
                    self.log_signal.emit(f"失败 ({e})")
                    continue
        if not sources:
            raise RuntimeError("未能下载任何有效的瓦片数据。")
        self.log_signal.emit("🔄 正在重投影并拼接数据...")
        minx, miny, maxx, maxy = target_bounds
        res = 0.00009  # 约 10米 @ 45deg
        width = int(math.ceil((maxx - minx) / res))
        height = int(math.ceil((maxy - miny) / res))
        dst_transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, width, height)
        final_data = np.full((height, width), fill_value=255, dtype=np.uint8)
        for i, (src_data, src_transform, src_crs) in enumerate(sources):
            src_bounds = rasterio.transform.array_bounds(src_data.shape[0], src_data.shape[1], src_transform)
            if src_bounds[2] < minx or src_bounds[0] > maxx or src_bounds[3] < miny or src_bounds[1] > maxy:
                continue
            inter_minx = max(minx, src_bounds[0])
            inter_miny = max(miny, src_bounds[1])
            inter_maxx = min(maxx, src_bounds[2])
            inter_maxy = min(maxy, src_bounds[3])
            row_start = int(math.floor((maxy - inter_maxy) / res))
            row_end = int(math.ceil((maxy - inter_miny) / res))
            col_start = int(math.floor((inter_minx - minx) / res))
            col_end = int(math.ceil((inter_maxx - minx) / res))
            row_start, row_end = max(0, row_start), min(height, row_end)
            col_start, col_end = max(0, col_start), min(width, col_end)
            if row_start >= row_end or col_start >= col_end:
                continue
            h_sub = row_end - row_start
            w_sub = col_end - col_start
            sub_transform = rasterio.transform.from_bounds(inter_minx, inter_miny, inter_maxx, inter_maxy, w_sub, h_sub)
            sub_data = np.full((h_sub, w_sub), 255, dtype=np.uint8)
            reproject(
                source=src_data, destination=sub_data,
                src_transform=src_transform, src_crs=src_crs,
                dst_transform=sub_transform, dst_crs=target_crs,
                resampling=Resampling.nearest,
                src_nodata=255, dst_nodata=255
            )
            mask_valid = sub_data != 255
            final_data[row_start:row_end, col_start:col_end][mask_valid] = sub_data[mask_valid]
        # 生成二值掩膜: 水=0, 陆=1, 背景=255
        esa_class_water = 80  # ESA WorldCover v200 中水体对应的值
        binary_mask = np.ones_like(final_data, dtype=np.uint8)
        binary_mask[final_data == esa_class_water] = 0
        binary_mask[final_data == 255] = 255
        return binary_mask, dst_transform, target_crs
        
    def raster_to_vector(self, binary_mask, transform, crs, min_pixels=5):
        """将二值栅格转换为矢量多边形 (仅提取水体)"""
        self.log_signal.emit("🔄 正在执行栅格转矢量 (Polygonize)...")
        geometries = []
        values = []
        results = features.shapes(
            binary_mask,
            transform=transform,
            connectivity=8,
            mask=binary_mask == 0  # 只处理值为0的像素
        )
        count = 0
        invalid_count = 0
        for geom_dict, value in results:
            try:
                geom = shape(geom_dict)
                if not geom.is_valid:
                    geom = geom.buffer(0)
                    if not geom.is_valid or geom.is_empty:
                        invalid_count += 1
                        continue
                if geom.is_empty:
                    continue
                geometries.append(geom)
                values.append(value)
                count += 1
            except Exception as e:
                invalid_count += 1
                continue
        self.log_signal.emit(f"    检测到 {count} 个有效水体多边形 (跳过 {invalid_count} 个无效/空几何体)。")
        if not geometries:
            return None
        gdf = gpd.GeoDataFrame(
            {"water_class": values},
            geometry=geometries,
            crs=crs
        )
        return gdf
        
    def run(self):
        try:
            cfg = self.config
            output_dir = cfg['output_dir']
            # 🔑 中断检查点
            if self.isInterruptionRequested():
                self.finished_signal.emit(False, "任务已手动停止")
                return
            # 确定范围
            if cfg.get('use_sar_bounds', False):
                # 读取 SAR 影像获取边界及有效覆盖范围
                self.log_signal.emit(" 正在读取 SAR 影像边界及有效覆盖区域...")
                sar_folder = cfg['sar_folder']
                tiff_files = glob.glob(os.path.join(sar_folder, "*.tiff")) + glob.glob(os.path.join(sar_folder, "*.tif"))
                if not tiff_files:
                    raise FileNotFoundError("未在 SAR 文件夹中找到 TIFF 文件！")
                with rasterio.open(tiff_files[0]) as src:
                    # 获取原始边界和坐标系
                    bounds = src.bounds
                    sar_crs = src.crs
                    sar_transform = src.transform
                    # 🔑 关键：读取数据并生成有效掩膜（识别非黑边区域）
                    raw = src.read(1)
                    # 只剔除与栅格边界连通的零值黑边；内部零值属于覆盖区。
                    coverage_mask, _, internal_gap_count = SARWorker.build_sar_coverage_mask(
                        raw)
                    valid_mask = coverage_mask.astype(np.uint8)
                    self.log_signal.emit(
                        f"✅ SAR覆盖识别：外围黑边 {np.count_nonzero(~coverage_mask)} 像素；"
                        f"内部零值漏洞 {internal_gap_count} 像素（ESA裁剪时保留）。")
                    # 找到有效数据的边界（去除黑边）
                    rows = np.any(valid_mask, axis=1)
                    cols = np.any(valid_mask, axis=0)
                    if not np.any(rows) or not np.any(cols):
                        raise ValueError("SAR 影像中没有有效数据！")
                    # 计算有效数据的像素范围
                    row_min, row_max = np.where(rows)[0][[0, -1]]
                    col_min, col_max = np.where(cols)[0][[0, -1]]
                    # 将像素坐标转换为地理坐标
                    # 左上角（行最小，列最小）
                    top_left = src.transform * (col_min, row_min)
                    # 右下角（行最大，列最大）
                    bottom_right = src.transform * (col_max + 1, row_max + 1)
                    # 创建有效区域边界
                    valid_bounds = (top_left[0], bottom_right[1], bottom_right[0], top_left[1])
                    self.log_signal.emit(f"✅ SAR 原始影像范围: Lon[{bounds.left:.4f}, {bounds.right:.4f}], Lat[{bounds.bottom:.4f}, {bounds.top:.4f}]")
                    self.log_signal.emit(f"✅ SAR 有效覆盖区域: Lon[{valid_bounds[0]:.4f}, {valid_bounds[2]:.4f}], Lat[{valid_bounds[1]:.4f}, {valid_bounds[3]:.4f}]")
                # 🔑 关键：将边界转换为 WGS84 (EPSG:4326)
                if sar_crs is not None and sar_crs.to_string() != "EPSG:4326":
                    self.log_signal.emit(f"🔄 将坐标系从 {sar_crs.to_string()} 转换为 EPSG:4326...")
                    bounds = transform_bounds(sar_crs, "EPSG:4326", 
                                            valid_bounds[0], valid_bounds[1], 
                                            valid_bounds[2], valid_bounds[3],
                                            densify_pts=21)
                    self.log_signal.emit(f"✅ WGS84 有效覆盖区域: Lon[{bounds[0]:.6f}, {bounds[2]:.6f}], Lat[{bounds[1]:.6f}, {bounds[3]:.6f}]")
                else:
                    bounds = valid_bounds
            else:
                bounds = cfg['bounds']
                self.log_signal.emit(f"📍 使用手动范围: Lon[{bounds[0]:.4f}, {bounds[2]:.4f}], Lat[{bounds[1]:.4f}, {bounds[3]:.4f}]")
            # 🔑 中断检查点
            if self.isInterruptionRequested():
                self.finished_signal.emit(False, "任务已手动停止")
                return
            # 计算瓦片
            tiles = self.get_esa_tiles(bounds)
            if not tiles:
                raise ValueError("计算出的瓦片列表为空。")
            # 下载并处理数据
            binary_mask, out_transform, out_crs = self.download_esa_water(tiles, bounds)
            # 🔑 如果下载过程中被中断，直接返回
            if binary_mask is None:
                self.finished_signal.emit(False, "任务已手动停止")
                return
            # 🔑 中断检查点
            if self.isInterruptionRequested():
                self.finished_signal.emit(False, "任务已手动停止")
                return
            # 🔑 关键优化：如果使用了 SAR 边界，则利用掩膜裁剪掉黑边区域
            if cfg.get('use_sar_bounds', False):
                self.log_signal.emit("✂️ 正在将 ESA 水体数据精确裁剪至 SAR 影像有效覆盖区域...")
                # 重新读取 SAR 影像获取变换参数
                with rasterio.open(tiff_files[0]) as src:
                    sar_transform = src.transform
                    sar_crs = src.crs
                    raw = src.read(1)
                    coverage_mask, _, _ = SARWorker.build_sar_coverage_mask(raw)
                    # ESA 只按真实覆盖 footprint 裁剪，不按内部像素值是否为0裁剪。
                    sar_valid_mask = coverage_mask.astype(np.uint8)
                # 创建与 binary_mask 同形状的目标网格
                target_sar_mask = np.zeros_like(binary_mask, dtype=np.uint8)
                # 将 SAR 的有效掩膜重采样到 ESA 水体数据的网格上
                reproject(
                    source=sar_valid_mask,
                    destination=target_sar_mask,
                    src_transform=sar_transform,
                    src_crs=sar_crs,
                    dst_transform=out_transform,
                    dst_crs=out_crs,
                    resampling=Resampling.nearest  # 使用最近邻，保持 0/1 值
                )
                # 应用裁剪：SAR 未覆盖的区域 (target_sar_mask == 0) 设为 NoData (255)
                masked_count = np.sum(target_sar_mask == 0)
                binary_mask[target_sar_mask == 0] = 255
                self.log_signal.emit(f"✅ 裁剪完成，已移除 {masked_count} 个黑边像素。")
            # 保存 TIFF
            tiff_path = os.path.join(output_dir, "esa_water_mask.tif")
            self.log_signal.emit(f"💾 正在保存栅格结果: {tiff_path}")
            with rasterio.open(
                tiff_path, 'w', driver='GTiff',
                height=binary_mask.shape[0], width=binary_mask.shape[1],
                count=1, dtype=binary_mask.dtype,
                crs=out_crs, transform=out_transform,
                compress='lzw', nodata=255
            ) as dst:
                dst.write(binary_mask, 1)
            self.log_signal.emit(f"    ✅ 栅格文件已保存: {tiff_path}")
            #  中断检查点
            if self.isInterruptionRequested():
                self.finished_signal.emit(False, "任务已手动停止")
                return
            # 栅格转矢量
            shp_path = os.path.join(output_dir, "esa_water_bodies.shp")
            water_gdf = self.raster_to_vector(binary_mask, out_transform, out_crs, cfg.get('min_water_area', 5))
            if water_gdf is not None:
                self.log_signal.emit(f"💾 正在保存矢量结果: {shp_path}")
                water_gdf['area_sq_deg'] = water_gdf.geometry.area
                water_gdf['perimeter'] = water_gdf.geometry.length
                water_gdf.to_file(shp_path, driver='ESRI Shapefile')
                self.log_signal.emit(f"    ✅ 矢量文件已保存: {shp_path}")
                self.log_signal.emit(f"    包含 {len(water_gdf)} 个水体要素。")
            else:
                self.log_signal.emit("    ️ 未检测到有效水体，跳过矢量文件生成。")
            self.finished_signal.emit(True, f"ESA 水体数据下载完成！\n输出文件:\n  - {tiff_path}\n  - {shp_path}")
        except Exception as e:
            import traceback
            self.finished_signal.emit(False, f"下载失败: {str(e)}\n{traceback.format_exc()}")
