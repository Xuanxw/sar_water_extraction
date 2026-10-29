"""ai4g-flood 网络结构的 DFC 水体对照：运行 python src/supervised/experimental/experimental_model_unet_mobilenet.py。

参考 Microsoft ai4g-flood/src/utils/model.py（MIT）：U-Net + MobileNetV2。
原仓库将灾前/灾后 SAR 转为双通道二值变化图，并公开推理流程。
这里按本项目需求输入单时相 VV/VH，采用训练集标准化、单头 BCE 和固定分组划分；
这是结构适配实验，不是原洪水流程或原训练配方的完整复现。
"""
from pathlib import Path
import sys
import gc

import torch
from torch import nn
import segmentation_models_pytorch as smp
from termcolor import cprint

# 支持按文件路径直接运行，也支持包导入。
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from supervised.train import WaterSegmentationTrainer


ROOT = Path("E:/Code_projects/sar_water_extraction")
IMAGE_DIR = Path("E:/Code_projects/sar_water_extraction/data/Track1/train/images")
LABEL_DIR = Path("E:/Code_projects/sar_water_extraction/data/Track1/train/labels")
# 训练器读取同一 split_manifest.csv，自动追加 _grouped 和运行时间戳。
OUTPUT_DIR = Path("E:/Code_projects/sar_water_extraction/outputs/ai4g_mobilenetv2_unet_dfc_hard_sampling")
BANDS = (1, 2)  # 当前 DFC 文件中的 VV、VH；保持512×512整幅输入，不缩放成128×128。
PRETRAINED = True  # 仅加载 MobileNetV2 的 ImageNet 权重，不加载 ai4g_sar_model.ckpt。
BATCH_SIZE = 2
EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
SEED = 42
TRAIN_AUGMENTATION = True  # 与 grouped PVT 相同：90度旋转和水平翻转。
SAMPLING = "hard"
HARD_WARMUP_EPOCHS = 2
HARD_SAMPLE_RATIO = 0.5
HARD_POOL_RATIO = 0.25
RUN_POSTPROCESS = True
POSTPROCESS_OPTIONS = dict(
    threshold=0.5, use_landcover=True, landcover_band=5, water_class=80,
    save_masks=True, save_visuals=True, max_visualizations=None,
)  # 同时输出 raw 和土地覆盖修正结果；模型比较优先看 raw。
SOURCE_COMMIT = "b3c274e91911e3dcfd6c9f063b73d13b56d00bde"


class AI4GWaterUNet(nn.Module):
    """复用 SMP 的完整 U-Net + MobileNetV2，输出一个未激活的水体 logit。"""

    def __init__(self, in_channels=2, num_classes=1, pretrained=True):
        super().__init__()
        # 编码深度、解码通道、BatchNorm 和最近邻上采样采用 SMP U-Net 默认结构。
        # 显式写出配置便于复核；输入层的双通道预训练适配由 SMP 完成。
        self.model = smp.Unet(
            encoder_name="mobilenet_v2", encoder_depth=5,
            encoder_weights="imagenet" if pretrained else None,
            decoder_channels=(256, 128, 64, 32, 16),
            decoder_use_norm="batchnorm", decoder_attention_type=None,
            decoder_interpolation="nearest",
            in_channels=in_channels, classes=num_classes, activation=None,
        )

    def forward(self, image):
        """[B,2,H,W] 标准化 SAR → [B,1,H,W]；BCE 内部处理 sigmoid。"""
        return self.model(image)


def create_trainer():
    """复用项目训练器；参数在本文件定义，不随 train.py 的实验开关变化。"""
    return WaterSegmentationTrainer(
        model_class=AI4GWaterUNet,
        model_kwargs=dict(in_channels=len(BANDS), num_classes=1, pretrained=PRETRAINED),
        model_info=dict(
            model_name="ai4g_mobilenetv2_unet_dfc", backbone="mobilenet_v2",
            decoder="SMP U-Net", normalization="BatchNorm", smp_version=smp.__version__,
            pretrained=PRETRAINED, supervision="single_head_bce",
            input_representation="single_date_DFC_stored_values; train_mean_std",
            implementation="ai4g architecture adapted to DFC; project training protocol",
            source_repository="https://github.com/microsoft/ai4g-flood",
            source_commit=SOURCE_COMMIT,
        ),
        image_dir=IMAGE_DIR, label_dir=LABEL_DIR, output_dir=OUTPUT_DIR,
        bands=BANDS, batch_size=BATCH_SIZE, epochs=EPOCHS, learning_rate=LEARNING_RATE,
        seed=SEED, val_ratio=0.2, max_samples=None,
        loss_name="bce", head_loss_weights=(1.0,),
        optimizer_name="adamw", weight_decay=WEIGHT_DECAY, cosine_schedule=True,
        train_augmentation=TRAIN_AUGMENTATION, sampling=SAMPLING,
        hard_warmup_epochs=HARD_WARMUP_EPOCHS,
        hard_sample_ratio=HARD_SAMPLE_RATIO, hard_pool_ratio=HARD_POOL_RATIO,
    )


def main():
    """训练后重载最佳 F1 权重，沿用现有 quality 的整体、边界与可视化评价。"""
    cprint("实验：U-Net + MobileNetV2；DFC 单时相水体适配；单头 BCE", "cyan", flush=True)
    trainer = create_trainer()
    trainer.train()
    run_dir = trainer.run_dir
    if RUN_POSTPROCESS:
        options = dict(POSTPROCESS_OPTIONS, image_dir=trainer.image_dir,
                       label_dir=trainer.label_dir, use_saved_val=True, with_labels=True,
                       output_dir=None, batch_size=trainer.batch_size, use_amp=trainer.use_amp)
        # 释放优化器和训练模型后再加载最佳模型，避免占用双份显存。
        del trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        from supervised.evaluation.postprocess import run_postprocess
        run_postprocess([run_dir / "best_f1.pth"], **options)
    return run_dir


if __name__ == "__main__":
    main()
