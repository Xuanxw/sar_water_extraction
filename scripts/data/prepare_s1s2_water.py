"""用训练 YAML 准备 S1S2-Water VV＋DEM 整景数据及窗口索引。"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from configuration import config_argument, load_yaml
from datasets.s1s2_water import prepare_dataset


def main(config_path):
    config = load_yaml(config_path)
    paths, data = config["paths"], config["data"]
    target = paths["IMAGE_DIR"].parent
    if (data.get("SCENE_SOURCE") != "s1s2_water" or data["DATA_LAYOUT"] != "full_scene"
            or data["POLARIZATIONS"] != ["VV"] or data["POLARIZATION_BANDS"]["VV"] != 1 or data["DEM_BAND"] != 2
            or paths["IMAGE_DIR"] != target / "images" or paths["LABEL_DIR"] != target / "labels"
            or paths["MANIFEST_PATHS"] != [target / "split_manifest.csv"]):
        raise ValueError("请使用 S1S2-Water VV＋DEM 整景训练配置，准备目录需包含 images、labels 和 split_manifest.csv")
    scene = data["SCENE_OPTIONS"]
    prepare_dataset(paths["DATA_ROOT"], target, paths["WINDOW_CACHE_DIR"], scene["window_size"], scene["halo"])


if __name__ == "__main__":
    main(config_argument())
