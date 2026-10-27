"""兼容原命令；评价参数和实现已移到 src/evaluation/model_eval.py。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluation.model_eval import main

if __name__ == "__main__":
    main()
