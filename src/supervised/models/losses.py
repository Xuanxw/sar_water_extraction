"""水体分割损失：单输出头的计算放在这里，多输出监督仍由 train.py 求和。"""
import torch
from torch import nn
from torch.nn import functional as F


class WaterSegmentationLoss(nn.Module):
    """ce 用于历史双类别模型；bce、bce_dice 用于单通道水体模型。"""

    def __init__(self, name="bce", ignore_index=-1, bce_weight=1.0, dice_weight=1.0, class_weights=(1.0, 1.0)):
        super().__init__()
        if name not in ("ce", "bce", "bce_dice"):
            raise ValueError("损失策略应为 ce、bce 或 bce_dice")
        self.name, self.ignore_index = name, ignore_index
        self.bce_weight, self.dice_weight = bce_weight, dice_weight
        # 顺序为非水体、水体；这是像素类别权重，与多头监督权重不同。
        self.class_weights = tuple(float(value) for value in class_weights)
        if len(self.class_weights) != 2 or any(value <= 0 or not torch.isfinite(torch.tensor(value)) for value in self.class_weights):
            raise ValueError("class_weights 需要两个有限正数，顺序为非水体、水体")
        if name == "ce" and self.class_weights != (1.0, 1.0):
            raise ValueError("类别加权当前仅用于单通道 BCE")
        self.binary = name != "ce"

    def forward(self, logits, label):
        """logits 为 [B,C,H,W]，label 为 [B,H,W]；忽略像素不参与任何损失。"""
        valid = label != self.ignore_index
        if not valid.any():
            return logits.sum() * 0  # 全忽略时仍保留计算图，返回零损失。
        if not self.binary:
            return F.cross_entropy(logits, label, ignore_index=self.ignore_index)
        if logits.shape[1] != 1:
            raise ValueError("bce / bce_dice 要求模型输出一个水体 logit 通道")

        # 用 float32 累计，避免 AMP 下大幅面像素求和溢出；不先 sigmoid 再算 BCE。
        score = logits[:, 0].float()
        target = label.clamp_min(0).float()
        pixel_loss = F.binary_cross_entropy_with_logits(score, target, reduction="none")
        background_weight, water_weight = self.class_weights
        pixel_weight = torch.where(target == 1, water_weight, background_weight)
        # 除以有效像素数；权重由全体train标签预先确定，不在每批/val上重新估计。
        bce = (pixel_loss * pixel_weight)[valid].mean()
        if self.name == "bce":
            return bce  # 类别权重均为1时，与原始有效像素BCE均值一致。

        # Soft Dice 使用连续概率，不能用阈值化预测，否则无法正常反向传播。
        probability = score.sigmoid().masked_fill(~valid, 0)
        target = target.masked_fill(~valid, 0)
        intersection = (probability * target).sum(dim=(1, 2))
        denominator = probability.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
        # 每幅图分别计算后求均值；smooth=1，兼容无水图，不混合整个 batch 的水体面积。
        dice = 1 - (2 * intersection + 1) / (denominator + 1)
        dice = dice[valid.any(dim=(1, 2))].mean()
        return self.bce_weight * bce + self.dice_weight * dice
