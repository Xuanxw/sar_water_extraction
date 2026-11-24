"""独立GAMMA参考产品对照入口；只读取已有成果，不运行水体提取。"""

from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sar_geometry.gamma_reference import compare_gc_map2_masks


def main(config_path):
    """读取YAML，以同网格的阴影/叠掩图和GAMMA ls_map生成独立差异成果。"""
    with Path(config_path).open(encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    return compare_gc_map2_masks(config)


if __name__ == '__main__':
    config_path = sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[2] / 'configs/sar_geometry_gamma_compare.yaml'
    main(config_path)
