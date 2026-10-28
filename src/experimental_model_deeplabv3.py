"""PVT + DeepLabv3+ 风格水体实验：python src/experimental_model_deeplabv3.py。

保留 PVT-v2-B2 的四级编码，在最深层做 ASPP，再融合四分之一尺度浅层特征。
这是输出步长32的适配版，不是原论文的步长16/8或空洞可分离卷积完整复现。
训练协议沿用 grouped PVT 基线，解码器改为单头 BCE；参数直接在下方修改。
"""
from pathlib import Path
import gc

import torch
from torch import nn
from torch.nn import functional as F
import timm
from termcolor import cprint

from models.common import BasicConv2d
from models.pvt import build_pvt_backbone
from train import WaterSegmentationTrainer


ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "data/Track1/train/images"
LABEL_DIR = ROOT / "data/Track1/train/labels"
# 公共训练器自动追加 _grouped 和时间戳，不覆盖既有实验。
OUTPUT_DIR = ROOT / "outputs/pvt_deeplabv3plus_hard_sampling"
BANDS = (1, 2)  # 当前 DFC 的 VV、VH；单极化可改成 (1,) 或 (2,)。
BACKBONE = "pvt_v2_b2.in1k"
PRETRAINED = True  # 仅用 ImageNet 骨干初始化，不接着已有水体模型续训。
DECODER_CHANNELS = 256
LOW_LEVEL_CHANNELS = 48
# PVT 最深层步长32，512输入对应16×16；相对步长16的6/12/18减半。
# 这是固定配置，不根据验证结果动态改变；不直接照搬大空洞率。
ASPP_RATES = (3, 6, 9)
ASPP_DROPOUT = 0.1
BATCH_SIZE = 2
EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
SEED = 42
TRAIN_AUGMENTATION = True  # 与控制组相同：90度旋转和水平翻转。
SAMPLING = "hard"
HARD_WARMUP_EPOCHS = 2
HARD_SAMPLE_RATIO = 0.5
HARD_POOL_RATIO = 0.25
RUN_POSTPROCESS = True
POSTPROCESS_OPTIONS = dict(
    threshold=0.5, use_landcover=True, landcover_band=5, water_class=80,
    save_masks=True, save_visuals=True, max_visualizations=None,
)  # 保存 raw、土地覆盖修正、整体/边界指标及PNG；比较网络时优先看 raw。


class ASPP(nn.Module):
    """空洞空间金字塔：1×1、三个空洞3×3、全局池化，共五条上下文分支。"""

    def __init__(self, in_channels, channels=256, rates=(3, 6, 9), dropout=0.1):
        super().__init__()
        self.branches = nn.ModuleList([
            BasicConv2d(in_channels, channels, 1),
            *[BasicConv2d(in_channels, channels, 3, dilation=rate) for rate in rates],
        ])
        # 池化后仅剩1×1；此支路不使用BN，允许最后一批只有一张图。
        # 其余空间卷积仍用公共 BasicConv2d 的 BatchNorm，不整体切换归一化方式。
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, channels, 1),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            BasicConv2d((len(rates) + 2) * channels, channels, 1),
            nn.Dropout(dropout),
        )

    def forward(self, feature):
        """[B,512,H/32,W/32] → [B,256,H/32,W/32]，各分支空间尺寸不变。"""
        branches = [branch(feature) for branch in self.branches]
        context = F.interpolate(self.global_pool(feature), size=feature.shape[-2:],
                                mode="bilinear", align_corners=False)
        return self.project(torch.cat([*branches, context], dim=1))


class PVTDeepLabV3Plus(nn.Module):
    """PVT编码器 + ASPP + 浅层跳连；直接预测水体，不使用粗预测残差或UAFM。"""

    def __init__(self, in_channels=2, num_classes=1, pretrained=True,
                 backbone_name="pvt_v2_b2.in1k", decoder_channels=256,
                 low_level_channels=48, atrous_rates=(3, 6, 9), dropout=0.1):
        super().__init__()
        self.backbone = build_pvt_backbone(in_channels, pretrained, backbone_name)
        feature_channels = self.backbone.feature_info.channels()
        self.aspp = ASPP(feature_channels[-1], decoder_channels, atrous_rates, dropout)
        self.low_projection = BasicConv2d(feature_channels[0], low_level_channels, 1)
        # 两次普通3×3卷积融合上下文与定位信息；不是空洞可分离卷积版本。
        self.fusion = nn.Sequential(
            BasicConv2d(decoder_channels + low_level_channels, decoder_channels),
            BasicConv2d(decoder_channels, decoder_channels),
        )
        self.segmentation_head = nn.Conv2d(decoder_channels, num_classes, 1)

    def forward(self, image):
        """512输入：深层16×16 → ASPP → 与128×128浅层融合 → 单通道512×512 logits。"""
        features = self.backbone(image)
        high = self.aspp(features[-1])
        low = self.low_projection(features[0])
        # 按实际大小对齐，兼容不能被32整除的输入；不改动PVT的输出步长。
        high = F.interpolate(high, size=low.shape[-2:], mode="bilinear", align_corners=False)
        fused = self.fusion(torch.cat((high, low), dim=1))
        logits = self.segmentation_head(fused)
        # 输出未激活logits，sigmoid由BCE/评价入口处理，避免重复激活。
        return F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)


def create_trainer():
    """显式指定训练协议，不受 train.py 的 METHOD 与解码器开关影响。"""
    output_dir = OUTPUT_DIR
    if BANDS != (1, 2):
        output_dir = output_dir.with_name(output_dir.name + "_bands_" + "_".join(map(str, BANDS)))
    return WaterSegmentationTrainer(
        model_class=PVTDeepLabV3Plus,
        model_kwargs=dict(in_channels=len(BANDS), num_classes=1, pretrained=PRETRAINED,
                          backbone_name=BACKBONE, decoder_channels=DECODER_CHANNELS,
                          low_level_channels=LOW_LEVEL_CHANNELS, atrous_rates=ASPP_RATES,
                          dropout=ASPP_DROPOUT),
        model_info=dict(
            model_name="pvt_deeplabv3plus", backbone=BACKBONE,
            decoder="DeepLabv3+ style ASPP + low-level fusion", output_stride=32,
            decoder_channels=DECODER_CHANNELS, low_level_channels=LOW_LEVEL_CHANNELS,
            aspp_rates=ASPP_RATES, aspp_dropout=ASPP_DROPOUT,
            normalization="PVT LayerNorm + decoder BatchNorm; global pool branch without BN",
            decoder_convolution="standard Conv2d, not separable", align_corners=False,
            timm_version=timm.__version__, pretrained=PRETRAINED, supervision="single_head_bce",
            input_representation="single_date_DFC_stored_values; train_mean_std",
            implementation="PVT OS32 + DeepLabv3+ style decoder; project architecture comparison",
            decoder_reference="https://arxiv.org/abs/1802.02611",
        ),
        image_dir=IMAGE_DIR, label_dir=LABEL_DIR, output_dir=output_dir,
        bands=BANDS, batch_size=BATCH_SIZE, epochs=EPOCHS, learning_rate=LEARNING_RATE,
        seed=SEED, val_ratio=0.2, max_samples=None,
        loss_name="bce", head_loss_weights=(1.0,),
        optimizer_name="adamw", weight_decay=WEIGHT_DECAY, cosine_schedule=True,
        train_augmentation=TRAIN_AUGMENTATION, sampling=SAMPLING,
        hard_warmup_epochs=HARD_WARMUP_EPOCHS,
        hard_sample_ratio=HARD_SAMPLE_RATIO, hard_pool_ratio=HARD_POOL_RATIO,
    )


def main():
    """训练后释放显存并重载best_f1.pth，在保存的val上输出quality结果。"""
    cprint("实验：PVT + DeepLabv3+风格解码器；步长32；单头BCE", "cyan", flush=True)
    trainer = create_trainer()
    trainer.train()
    run_dir = trainer.run_dir
    if RUN_POSTPROCESS:
        options = dict(POSTPROCESS_OPTIONS, image_dir=trainer.image_dir,
                       label_dir=trainer.label_dir, use_saved_val=True, with_labels=True,
                       output_dir=None, batch_size=trainer.batch_size, use_amp=trainer.use_amp)
        del trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        from evaluation.postprocess import run_postprocess
        run_postprocess([run_dir / "best_f1.pth"], **options)
    return run_dir


if __name__ == "__main__":
    main()
