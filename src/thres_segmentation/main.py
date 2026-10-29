"""独立阈值分割入口：python src/thres_segmentation/main.py；直接修改下面的变量。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path("E:/Code_projects/sar_water_extraction")
MODE = "dataset"  # dataset：按目录处理影像；gui：原地理编码 SAR 图形界面。
IMAGE_DIR = Path("E:/Code_projects/sar_water_extraction/data/Track1/train/images/val")
LABEL_DIR = Path("E:/Code_projects/sar_water_extraction/data/Track1/train/labels/val")  # 无标签影像设为 None，例如处理 data/images。
SPLIT_MANIFEST = Path("E:/Code_projects/sar_water_extraction/data/Track1/train/split_manifest.csv")  # 只处理 active=1 的 val；外部数据设 None。
BANDS = {"VV": 1, "VH": 2}  # 每个波段独立分割；单波段数据可设为 {"VV": 1}。
OUTPUT_DIR = None  # None 输出到 outputs/thres_seg/quality/时间戳，不依附训练实验。
THRESHOLD_OPTIONS = dict(water_percent=10, mode="bottom", lee_window=5,
                         noise_ratio=.12, stretch=(1, 99), power_to_db=False)
# water_percent 是固定的逐图百分位比例，不是网络概率阈值，不使用标签自动选参。
# lee_window=1 关闭滤波；mode="top" 提取亮值侧；存储单位未确认时不默认转 dB。
USE_LANDCOVER = True  # 当前 DFC 的 band 5 可选补水；纯 SAR/单波段影像设为 False。
LANDCOVER_BAND = 5
SAVE_MASKS = True  # 保存最终 TIFF：0非水体、1水体、255无效。
MAX_VISUALIZATIONS = 10  # None 绘制全部；所有输入影像仍参与处理和精度统计。


def main():
    if MODE == "gui":
        from thres_segmentation.gui import run_gui
        return run_gui()
    if MODE != "dataset":
        raise ValueError("MODE 使用 dataset 或 gui")
    from thres_segmentation.pipeline import run_segmentation
    return run_segmentation(
        image_dir=IMAGE_DIR, label_dir=LABEL_DIR, bands=BANDS, output_dir=OUTPUT_DIR,
        threshold_options=THRESHOLD_OPTIONS, use_landcover=USE_LANDCOVER,
        landcover_band=LANDCOVER_BAND, save_masks=SAVE_MASKS,
        max_visualizations=MAX_VISUALIZATIONS, split_manifest=SPLIT_MANIFEST)


if __name__ == "__main__":
    main()
