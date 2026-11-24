"""S1 Water 编码器配合 UNet++ 嵌套跳连，作为独立结构对照。

连接规则依据 Zhou et al., https://arxiv.org/abs/1807.10165 的式 (1)。
保留现有 S1 编码器的六个尺度及 BN/Dropout；每个嵌套节点使用两层卷积。
只监督最深的全分辨率输出，不复现论文的深监督训练配方。
"""
import torch
from torch import nn

from .sentinel1_water_unet import CENTER_FILTERS, ENCODER_FILTERS, ConvBNReLU


class UNetPlusPlus(nn.Module):
    """输入已预处理的 NCHW 浮点特征，输出同网格 N×K×H×W 无单位 logits。

    输入通道可为 VV、VV+DEM 或 VV+VH；数值变换和有效掩膜由数据层处理。
    编码器宽度固定为 32/64/128/256/512/1024，与 S1 Water U-Net 对齐。
    小于 32 或非 32 整倍数的尺寸右下补零，预测后裁回，不重采样网格。
    """

    def __init__(self, in_channels=1, num_classes=1, dropout=0.5):
        super().__init__()
        self.down = nn.ModuleList()
        channels = in_channels
        for width in ENCODER_FILTERS:
            self.down.append(nn.Sequential(ConvBNReLU(channels, width), ConvBNReLU(width, width)))
            channels = width
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(dropout)
        self.center = nn.Sequential(
            ConvBNReLU(channels, CENTER_FILTERS), ConvBNReLU(CENTER_FILTERS, CENTER_FILTERS)
        )
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        widths = (*ENCODER_FILTERS, CENTER_FILTERS)
        self.nested = nn.ModuleDict()
        # 按列构建：同列各节点只依赖编码器或之前的列，注册顺序也是前向顺序。
        for column in range(1, len(widths)):
            for level in range(len(widths) - column):
                inputs = column * widths[level] + widths[level + 1]
                self.nested[f"x{level}_{column}"] = nn.Sequential(
                    ConvBNReLU(inputs, widths[level]), ConvBNReLU(widths[level], widths[level])
                )
        self.classify = nn.Conv2d(widths[0], num_classes, kernel_size=1)

    def forward(self, x):
        """密集融合所有同尺度前序节点与下一级节点；返回 logits，外部统一 sigmoid。"""
        height, width = x.shape[-2:]
        if height % 32 or width % 32:
            x = nn.functional.pad(x, (0, -width % 32, 0, -height % 32))
        nodes = {}
        for level, block in enumerate(self.down):
            x = block(x)
            nodes[level, 0] = x
            x = self.drop(self.pool(x))
        depth = len(self.down)
        nodes[depth, 0] = self.center(x)
        for column in range(1, depth + 1):
            for level in range(depth + 1 - column):
                # X(i,j) = H([X(i,0), ..., X(i,j-1), up(X(i+1,j-1))])。
                inputs = [nodes[level, previous] for previous in range(column)]
                inputs.append(self.upsample(nodes[level + 1, column - 1]))
                nodes[level, column] = self.nested[f"x{level}_{column}"](torch.cat(inputs, dim=1))
        return self.classify(nodes[0, depth]).float()[..., :height, :width]
