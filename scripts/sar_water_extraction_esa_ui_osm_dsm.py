"""旧 GUI 入口兼容保留；数据集独立分割请运行 src/thres_seg/main.py。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from thres_seg.gui import run_gui, SARWaterExtractApp
from thres_seg.gis_worker import SARWorker
from thres_seg.esa_worker import ESAWaterWorker

if __name__ == "__main__":
    sys.exit(run_gui())
