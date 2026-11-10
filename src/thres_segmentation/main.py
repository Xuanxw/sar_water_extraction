"""独立阈值分割入口：python src/thres_segmentation/main.py；修改顶部变量后运行。"""
import sys
from datetime import datetime
from pathlib import Path

# 仅用于定位Python模块；下面的数据和输出路径均使用绝对路径。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ---------- 1. 运行模式与输入数据 ----------
MODE = "dataset"  # dataset：准备辅助数据后处理；download：只准备辅助数据；gui：基础界面。
DATA_LAYOUT = "full_scene"  # BC整景使用full_scene；旧DFC切片用patch。
# 与image_process共用成果根目录，不在这里维护另一套滤波/归一化参数。
from configuration import load_yaml
PREPROCESS_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "image_process.yaml"
PREPROCESSING = load_yaml(PREPROCESS_CONFIG)
NORMALIZED_ROOT = PREPROCESSING["paths"]["NORMALIZED_ROOT"]
IMAGE_DIR = NORMALIZED_ROOT / "images/val"
LABEL_DIR = NORMALIZED_ROOT / "labels/val"  # 无标签时设None。
SPLIT_MANIFEST = NORMALIZED_ROOT / "split_manifest.csv"
BANDS = {"VV": 1}  # 离线成果中的波段编号；单VV是第1波段。

# 整景模式从清单source_image读取原始GeoTIFF/ZIP和同名XML，不在prepared目录猜测meta。
SCENE_PREFIX = None  # None处理全部BC验证景；只处理BC3时设"BC3-"。
BLOCK_SIZE = 1024  # 只控制分块读取；每景仍使用统一阈值，不落盘切片。
CACHE_DIR = Path(r"E:\Code_projects\sar_water_extraction\data_cache\threshold_bc")
PREVIEW_SIZE = 1200  # PNG最长边，TIFF和指标始终使用原始分辨率。
BOUNDARY_RADII = (1, 3, 5)  # 原始像素单位；整景输出缓冲带全像素计数。
# 精确百分位使用临时磁盘缓冲，峰值约为单景有效像素数×8字节；完成/异常均自动清理。

# 1.1 外部辅助数据准备（不依赖GUI，和监督后处理共用下载器/缓存）
AUXILIARY_OPTIONS = dict(
    download_history=False,  # True：下载WorldCover并生成各景历史水体掩膜。
    download_elevation=False,  # True：下载Copernicus GLO-30 DSM，直接接入水深分析。
    worldcover_year=2021,  # 支持2020(v100)、2021(v200)，不是任意日期的灾前影像。
    buffer_m=1000.,  # 高程范围外扩；历史水体按影像包围框准备。
    max_tiles_per_scene=64, max_mosaic_pixels=100000000,
    timeout=60, retries=3,
)
AUXILIARY_CACHE_DIR = Path(r'E:\Code_projects\sar_water_extraction\data_cache\auxiliary')
# 开启后：选中清单→下载/复用缓存→逐景裁剪→将辅助路径传入面积/变化/水深。
# 指定ANALYSIS_OPTIONS中的本地reference_path/elevation_path时优先使用本地，不重复下载。

# ---------- 2. 传统阈值与预处理 ----------
WATER_PERCENT = 10  # 逐图取暗值侧10%分位阈值，不是模型概率，不根据标签自动调参。
THRESHOLD_MODE = "bottom"  # bottom提取暗值侧；top提取亮值侧。
# 预处理已由image_process完成：ORG幅度平方→SNAP Lee→恢复幅度→train统计z-score。
# 此处不再次滤波/转dB/P1-P99拉伸；负z-score和合法0均保留。
# WATER_PERCENT仍是整景暗/亮像元百分位，不是概率阈值；结果不保证等于真实水体占比。

# ---------- 3. 辅助修正、面积、历史变化与几何水深 ----------
USE_LANDCOVER = False  # BC只有VV，没有DFC第5波段；整景入口当前不做土地覆盖修正。
LANDCOVER_BAND = 5  # 当前DFC的土地覆盖波段；与上述SAR输入波段分开设置。

# 3.1 面积/变化/水深：外部历史数据只用于对比，默认不强制补入当前阈值结果。
# 面积始终统计；指定的本地文件优先，自动下载时按各景范围准备对应参考数据。
ANALYSIS_OPTIONS = dict(
    reference_path=None,  # 历史栅格或水体 SHP/GPKG/GeoJSON。
    reference_water_values=(1,), reference_dry_values=(0,),
    # 旧GUI下载的 esa_water_mask.tif 使用 water=(0,), dry=(1,)！新下载统一1水体/0非水体。
    # 原始 WorldCover：water=(80,), dry=(10,20,30,40,50,60,70,90,95,100)。
    reference_coverage_path=None,  # 矢量调查覆盖面；面外属于未知。
    vector_assume_complete=False,  # 仅确认矢量完整覆盖当前影像时设True。
    elevation_path=None,  # DEM/DSM GeoTIFF；None不计算水深。
    elevation_kind='DEM', elevation_scale=1.,  # 高程乘该系数转换为米。
    depth_scope='new_water',  # 默认仅共同覆盖内新增水体；需历史参考。
    # all_water必须显式选择；普通DEM无法提供永久水体水底地形。
    depth_neighbors=100, depth_min_boundary=3, depth_max_distance_m=5000.,
    depth_max_pixels=25000000,  # 水深整景连通性内存保护；面积/变化保持分块。
)

# ---------- 4. 评价、输出与可视化 ----------
OUTPUT_ROOT = Path("E:/Code_projects/sar_water_extraction/outputs/thres_seg/quality")
OUTPUT_DIR = None  # None在OUTPUT_ROOT下新建时间戳目录；也可指定新的绝对目录。
SAVE_MASKS = True  # 保存最终TIFF：0非水体、1水体、255无效。
MAX_VISUALIZATIONS = None  # None全部出图；0不出图；整数只限制出图数，不限制精度统计。


def main():
    """将顶部配置传入独立处理流程；标签仅用于评价，不参与阈值选择。"""
    if MODE == "gui":
        print("GUI仍为历史独立GIS流程；与image_process统一的BC处理请使用MODE=dataset。", flush=True)
        from thres_segmentation.gui import run_gui
        return run_gui()
    if MODE not in ('dataset','download'):
        raise ValueError("MODE 使用 dataset、download 或 gui")
    if DATA_LAYOUT not in ('full_scene','patch'):
        raise ValueError('DATA_LAYOUT使用full_scene或patch')
    if DATA_LAYOUT == 'full_scene' and (SPLIT_MANIFEST is None or USE_LANDCOVER):
        raise ValueError('整景需要SPLIT_MANIFEST，且USE_LANDCOVER=False')
    scene_analysis_options = None
    if MODE == 'download' or AUXILIARY_OPTIONS['download_history'] or AUXILIARY_OPTIONS['download_elevation']:
        from thres_segmentation.auxiliary_data import selected_images, prepare_auxiliary_inputs
        if MODE == 'download' and not any(AUXILIARY_OPTIONS[k] for k in ('download_history','download_elevation')):
            raise ValueError('download模式需开启download_history或download_elevation')
        images = selected_images(IMAGE_DIR,SPLIT_MANIFEST,SCENE_PREFIX if DATA_LAYOUT == 'full_scene' else None)
        scene_analysis_options = prepare_auxiliary_inputs(images,AUXILIARY_CACHE_DIR,
            analysis_options=ANALYSIS_OPTIONS,require_depth_reference=MODE!='download',**AUXILIARY_OPTIONS)
        if MODE == 'download':
            return scene_analysis_options
    from thres_segmentation.pipeline import run_segmentation

    output_dir = OUTPUT_DIR if OUTPUT_DIR is not None else OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    threshold_options = dict(water_percent=WATER_PERCENT, mode=THRESHOLD_MODE,
                             preprocessing="offline_normalized")
    print(f"预处理配置：{PREPROCESS_CONFIG}；直接复用离线归一化成果，不启动SNAP", flush=True)
    print(f"输入：{IMAGE_DIR}\n标签：{LABEL_DIR}\n输出：{output_dir}", flush=True)
    print(f"波段：{BANDS}；出图数量：{'全部' if MAX_VISUALIZATIONS is None else MAX_VISUALIZATIONS}", flush=True)
    if DATA_LAYOUT == "full_scene":
        if USE_LANDCOVER:
            raise ValueError("BC整景无内置土地覆盖波段，请设USE_LANDCOVER=False")
        from thres_segmentation.scene_pipeline import run_scene_segmentation
        return run_scene_segmentation(
            image_dir=IMAGE_DIR, label_dir=LABEL_DIR, split_manifest=SPLIT_MANIFEST,
            output_dir=output_dir, bands=BANDS, threshold_options=threshold_options,
            block_size=BLOCK_SIZE, cache_dir=CACHE_DIR, preview_size=PREVIEW_SIZE,
            save_masks=SAVE_MASKS, max_visualizations=MAX_VISUALIZATIONS,
            scene_prefix=SCENE_PREFIX, boundary_radii=BOUNDARY_RADII,
            analysis_options=ANALYSIS_OPTIONS, scene_analysis_options=scene_analysis_options)
    if DATA_LAYOUT != "patch":
        raise ValueError("DATA_LAYOUT使用full_scene或patch")
    return run_segmentation(
        image_dir=IMAGE_DIR, label_dir=LABEL_DIR, bands=BANDS, output_dir=output_dir,
        threshold_options=threshold_options, use_landcover=USE_LANDCOVER,
        landcover_band=LANDCOVER_BAND, save_masks=SAVE_MASKS,
        max_visualizations=MAX_VISUALIZATIONS, split_manifest=SPLIT_MANIFEST,
        analysis_options=ANALYSIS_OPTIONS, scene_analysis_options=scene_analysis_options)


if __name__ == "__main__":
    main()
