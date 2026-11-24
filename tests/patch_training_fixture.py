"""用四个合成 VV/VH 切片走通 YAML 训练、保存和 BN 校准入口。"""
from pathlib import Path
from unittest.mock import patch

import numpy as np
import rasterio
import torch
import yaml
from rasterio.transform import from_origin

from scripts.models import calibrate_bn
from supervised import train


def train_patch_model(root, architecture, backbone):
    template = Path(__file__).resolve().parents[1] / "configs/s1/train_mobilenetv2_unet_trainnorm80.yaml"
    config = yaml.safe_load(template.read_text(encoding="utf-8"))
    config["paths"].update(DATA_ROOT=str(root), IMAGE_DIR="images", LABEL_DIR="labels",
                           OUTPUT_ROOT="runs", OUTPUT_DIR="runs", WINDOW_CACHE_DIR="cache",
                           MANIFEST_PATHS=[], RESUME_CHECKPOINT=None)
    config["data"].update(POLARIZATIONS=["VV", "VH"], POLARIZATION_BANDS={"VV": 1, "VH": 2},
                          DEM_BAND=None, PATCH_REPRESENTATION="stored", VAL_RATIO=0.5)
    config["model"].update(ARCHITECTURE=architecture, BACKBONE=backbone, PRETRAINED=False, STAGE="full")
    config["training"].update(EPOCHS=1, BATCH_SIZE=2, CPU_THREADS=2, LOSS_NAME="ce",
                              TRAIN_AUGMENTATION=False, SAMPLING="random", CLASS_WEIGHT_MODE="none")
    config["quality"].update(SAVE_VALIDATION_FIGURES=False, SAVE_FEATURE_MAPS=False)
    config_path = root / "train.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")

    (root / "images").mkdir()
    (root / "labels").mkdir()
    rng = np.random.default_rng(42)
    grid = dict(driver="GTiff", width=64, height=64, crs="EPSG:32650",
                transform=from_origin(500000, 4000000, 10, 10))
    for i in range(4):
        image = rng.standard_normal((2, 64, 64)).astype("float32")
        label = (image[0] > 0).astype("uint8")
        label[:2] = 255
        with rasterio.open(root / "images" / f"{i}.tif", "w", **grid, count=2, dtype="float32") as dst:
            dst.write(image)
        with rasterio.open(root / "labels" / f"{i}.tif", "w", **grid, count=1, dtype="uint8", nodata=255) as dst:
            dst.write(label, 1)

    # configure 会绑定模块变量，测试结束后恢复调用方原有配置。
    with patch.dict(vars(train)), patch.object(torch.cuda, "is_available", return_value=False):
        train.configure(config_path)
        trainer = train.create_trainer()
        trainer.train()
        run_dir, metrics = trainer.run_dir, trainer.phase_metrics["val"]
        del trainer
        with patch.multiple(calibrate_bn, RUN_DIR=run_dir, CHECKPOINT_NAME="last.pth",
                            EXPECTED_EPOCH=1, RESUME_DIR=None):
            return calibrate_bn.BNCalibrator(), metrics
