"""Adapted from fabien-w/sentinel1-water-unet, src/s1water/model.py.
https://doi.org/10.3390/rs16061056

MIT License

Copyright (c) 2024-2026 Fabien H. Wagner and CTrees

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

The model weights distributed on Hugging Face are licensed separately under
the Creative Commons Attribution 4.0 International license (CC BY 4.0).
"""

from __future__ import annotations

import torch
from torch import nn

# Keras BatchNormalization default epsilon (PyTorch default is 1e-5).
KERAS_BN_EPS = 1e-3

ENCODER_FILTERS = (32, 64, 128, 256, 512)
CENTER_FILTERS = 1024


class ConvBNReLU(nn.Sequential):
    """3x3 convolution ('same' padding) + batch normalisation + ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch, eps=KERAS_BN_EPS, momentum=0.01),
            nn.ReLU(inplace=True),
        )


class UNetS1Water(nn.Module):
    """Wagner 水体 U-Net，按实际极化数构建第一层，返回单通道 logits。"""

    def __init__(self, in_channels: int = 1, num_classes: int = 1, dropout: float = 0.5):
        super().__init__()
        self.down = nn.ModuleList()
        ch = in_channels
        for f in ENCODER_FILTERS:
            self.down.append(nn.Sequential(ConvBNReLU(ch, f), ConvBNReLU(f, f)))
            ch = f
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(dropout)
        self.center = nn.Sequential(ConvBNReLU(ch, CENTER_FILTERS), ConvBNReLU(CENTER_FILTERS, CENTER_FILTERS))
        ch = CENTER_FILTERS
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.up = nn.ModuleList()
        for f in reversed(ENCODER_FILTERS):
            # concatenation of the skip connection (f channels) and the upsampled map (ch channels)
            self.up.append(nn.Sequential(ConvBNReLU(f + ch, f), ConvBNReLU(f, f), ConvBNReLU(f, f)))
            ch = f
        self.classify = nn.Conv2d(ch, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入 NCHW，在右侧和底部补齐五次池化的步长，再裁回原图网格。"""
        height, width = x.shape[-2:]
        if height % 32 or width % 32:
            x = nn.functional.pad(x, (0, -width % 32, 0, -height % 32))
        skips = []
        for block in self.down:
            x = block(x)
            skips.append(x)
            x = self.drop(self.pool(x))
        x = self.center(x)
        for block in self.up:
            x = torch.cat([skips.pop(), self.upsample(x)], dim=1)
            x = block(x)
        # 项目统一在 BCE 中处理 logits，验证时再应用 sigmoid。
        return self.classify(x).float()[..., :height, :width]
