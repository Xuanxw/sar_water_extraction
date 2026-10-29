"""用于水体二分类的四级 U-Net：编码、跳跃连接、解码。"""
import torch
from torch import nn
from torch.nn import functional as F


from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from supervised.models.common import DoubleConv


class UNet(nn.Module):
    """输入 [B,C,H,W]，输出 [B,2,H,W] 的 logits，不在模型内做 softmax。"""

    def __init__(self, in_channels=2, num_classes=2, base_channels=32):
        """构建 U-Net 模型，in_channels 为输入波段数，num_classes 为输出类别数。"""
        super().__init__() # 初始化模型管理机制
        c = base_channels
        self.pool = nn.MaxPool2d(2) # 缩小空间尺寸，2x2最大池化
        self.encoder1 = DoubleConv(in_channels, c)
        self.encoder2 = DoubleConv(c, c * 2)
        self.encoder3 = DoubleConv(c * 2, c * 4)
        self.encoder4 = DoubleConv(c * 4, c * 8)
        self.bottleneck = DoubleConv(c * 8, c * 16)

        # 转置卷积扩大空间尺寸，同时减少通道数。
        # eg: 从[B, 512, 32, 32] 转到 [B, 256, 64, 64]
        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, 2, stride=2)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.decoder4 = DoubleConv(c * 16, c * 8)
        self.decoder3 = DoubleConv(c * 8, c * 4)
        self.decoder2 = DoubleConv(c * 4, c * 2)
        self.decoder1 = DoubleConv(c * 2, c)
        self.classifier = nn.Conv2d(c, num_classes, 1)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """兼容旧 Sequential 命名，已有 BN 检查点无需重训或手动改名。"""
        # 旧双卷积中 0/1/3/4 分别为第一卷积、第一 BN、第二卷积、第二 BN。
        old_names = {"0": "conv1.conv", "1": "conv1.bn", "3": "conv2.conv", "4": "conv2.bn"}
        converted = {}
        for name, value in state_dict.items():
            parts = name.split(".")
            if len(parts) >= 3 and parts[1] in old_names:
                name = ".".join([parts[0], old_names[parts[1]], *parts[2:]])
            converted[name] = value
        return super().load_state_dict(converted, strict=strict, assign=assign)

    def forward(self, image):
        """先下采样提取上下文，再拼接同尺度编码特征，恢复像素级预测。"""
        # 512×512 输入对应 512、256、128、64、32 五种空间尺度。
        e1 = self.encoder1(image)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        center = self.bottleneck(self.pool(e4))

        d4 = self.up4(center)
        # 插值对齐跳跃连接的尺寸，也支持高宽不能被 16 整除的输入。
        d4 = F.interpolate(d4, size=e4.shape[-2:], mode="bilinear", align_corners=False)
        # 跳跃连接：torch.cat按通道拼接编码器特征和解码器特征
        # 编码特征 e4：[B, 256, 64, 64]
        # 解码特征 d4：[B, 256, 64, 64]
        #            ↓ 拼接
        #     [B, 512, 64, 64]
        #            ↓ DoubleConv
        #     [B, 256, 64, 64]
        # 拼接使解码阶段同时获得：
        # - 深层特征带来的大范围上下文；
        # - 编码阶段保留下来的空间细节。
        # 对水体分割而言，这有助于在识别大范围水域的同时恢复河岸、窄河道等细节。
        # 不过，普通 U-Net 的拼接本身不会显式判断哪个特征更可信。
        d4 = self.decoder4(torch.cat([e4, d4], dim=1)) 
        d3 = self.up3(d4)
        d3 = F.interpolate(d3, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.decoder3(torch.cat([e3, d3], dim=1))
        d2 = self.up2(d3)
        d2 = F.interpolate(d2, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.decoder2(torch.cat([e2, d2], dim=1))
        d1 = self.up1(d2)
        d1 = F.interpolate(d1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.decoder1(torch.cat([e1, d1], dim=1))
        return self.classifier(d1) # 生成每个像素的类别分数


if __name__ == "__main__":
    model = UNet()
    # 生成标准正态分布的随机张量，模拟一幅双通道输入。
    # 只用于检查网络能否运行，不是真实 SAR 数据。
    image = torch.randn(1, 2, 512, 512) 
    with torch.no_grad(): # 关闭梯度记录，减少这次尺寸检查的内存开销
        output = model(image)
    print(f"输入：{image.shape}")
    print(f"输出：{output.shape}")
    # 依次取出模型参数，使用 numel() 统计每个参数张量中的元素数，再求和得到参数总量。
    # 格式中的 :, 只是让数字显示为带千位分隔符的形式。
    print(f"参数量：{sum(parameter.numel() for parameter in model.parameters()):,}")
