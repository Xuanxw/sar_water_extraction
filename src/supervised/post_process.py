"""独立后处理入口：修改configs/post_process.yaml，再手动运行本文件。"""
import sys
import shutil
from datetime import datetime
from uuid import uuid4
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configuration import load_yaml

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "post_process.yaml"
CONFIG = load_yaml(CONFIG_PATH)
PATHS = CONFIG["paths"]


def main():
    """只读取已保存模型；后处理图件不覆盖训练每轮的验证图件。"""
    checkpoints = PATHS["POSTPROCESS_CHECKPOINTS"]
    if not checkpoints or any(not path.is_file() for path in checkpoints):
        raise FileNotFoundError("请在post_process.yaml中指定存在的POSTPROCESS_CHECKPOINTS")
    layout = CONFIG["run"]["DATA_LAYOUT"]
    if layout not in ("full_scene", "patch") or (layout == "full_scene" and len(checkpoints) != 1):
        raise ValueError("DATA_LAYOUT使用full_scene/patch；整景一次处理一个模型")
    output = PATHS["OUTPUT_DIR"] or checkpoints[0].parent / "postprocess" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(CONFIG_PATH, output / f"post_process_config_{datetime.now():%Y%m%d_%H%M%S_%f}_{uuid4().hex[:8]}.yaml")
    quality = output / "quality"
    print(f"后处理配置：{CONFIG_PATH}\n模型：{checkpoints}\n后处理结果及图件：{quality}", flush=True)
    if layout == "full_scene":
        from supervised.evaluation.scene_quality import run_scene_quality
        return run_scene_quality(checkpoints[0], image_dir=PATHS["IMAGE_DIR"], label_dir=PATHS["LABEL_DIR"],
            output_dir=quality, **CONFIG["quality"]["SCENE_QUALITY_OPTIONS"])
    from supervised.evaluation.postprocess import run_postprocess
    options = dict(CONFIG["quality"]["POSTPROCESS_OPTIONS"])
    analysis = dict(options["analysis_options"])
    analysis.update(reference_path=PATHS["REFERENCE_PATH"], reference_coverage_path=PATHS["REFERENCE_COVERAGE_PATH"],
                    elevation_path=PATHS["ELEVATION_PATH"])
    options.update(image_dir=PATHS["IMAGE_DIR"], label_dir=PATHS["LABEL_DIR"], output_dir=quality,
                   auxiliary_cache_dir=PATHS["AUXILIARY_CACHE_DIR"], analysis_options=analysis)
    return run_postprocess(checkpoints, **options)


if __name__ == "__main__":
    main()
