"""已有ORG成果的补处理入口；全部算法、续跑和归档均使用正式main.py。"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from thres_segmentation.main import main
from configuration import config_argument

if __name__=='__main__':
    main(config_argument())
