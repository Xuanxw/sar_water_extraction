"""已有ORG成果的补处理入口；全部算法、续跑和归档均使用正式main.py。"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from thres_segmentation.main import main,CONFIG_PATH

if __name__=='__main__':
    main(sys.argv[1] if len(sys.argv)>1 else CONFIG_PATH)
