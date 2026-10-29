"""共用卷积、双卷积、FPN 与语义解码头，每种方法集中在一个类中。"""
from torch import nn
from torch.nn import functional as F


class BasicConv2d(nn.Module):
    """卷积 → BatchNorm → ReLU；默认 3×3，适用于当前使用的奇数卷积核。"""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=None, dilation=1):
        super().__init__()
        # 未显式指定时按卷积核和空洞率补零，步长 1 下保持空间尺寸。
        if padding is None:
            padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feature):
        """提取局部特征，归一化后引入非线性。"""
        return self.relu(self.bn(self.conv(feature)))


class DoubleConv(nn.Module):
    """双卷积块：连续两个基础卷积块，供编码器和解码器复用。"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = BasicConv2d(in_channels, out_channels)
        self.conv2 = BasicConv2d(out_channels, out_channels)

    def forward(self, image):
        """[B,输入通道,H,W] → [B,输出通道,H,W]。"""
        return self.conv2(self.conv1(image))


class FPN(nn.Module):
    """作者实际使用的四级 FPN neck：1×1 侧向投影、高层上采样相加、3×3 输出。"""

    def __init__(self, channels=32):
        super().__init__()
        self.lateral_convs = nn.ModuleList([nn.Conv2d(channels, channels, 1) for _ in range(4)])
        self.fpn_convs = nn.ModuleList([nn.Conv2d(channels, channels, 3, padding=1) for _ in range(4)])

    def forward(self, features):
        """四层输入和输出均保持各自尺度；高层语义逐级传到浅层。"""
        laterals = [conv(feature) for conv, feature in zip(self.lateral_convs, features)]
        for index in range(3, 0, -1):
            laterals[index - 1] = laterals[index - 1] + F.interpolate(
                laterals[index], size=laterals[index - 1].shape[-2:], mode="nearest",
            )
        return tuple(conv(feature) for conv, feature in zip(self.fpn_convs, laterals))


class SemanticFPNDecoder(nn.Module):
    """将步长为 4、8、16、32 的四层特征对齐到 1/4 尺度，输出分类分数。"""

    def __init__(self, channels=32, num_classes=2):
        super().__init__()
        self.scale_heads = nn.ModuleList()
        # 对应作者的 head_length=[1,1,2,3]，每个卷积后接 BN 和 ReLU。
        for level in range(4):
            layers = []
            for _ in range(max(1, level)):
                layers.extend([
                    nn.Conv2d(channels, channels, 3, padding=1),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                ])
                if level > 0:
                    layers.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
            self.scale_heads.append(nn.Sequential(*layers))
        self.cls_seg = nn.Conv2d(channels, num_classes, 1)

    def forward(self, features):
        """四个 [B,32,H_i,W_i] → [B,类别数,H/4,W/4]；不做 softmax。"""
        result = self.scale_heads[0](features[0])
        for head, feature in zip(self.scale_heads[1:], features[1:]):
            # 显式按目标大小对齐，也能处理不能被 32 整除的影像尺寸。
            result = result + F.interpolate(
                head(feature), size=result.shape[-2:], mode="bilinear", align_corners=False,
            )
        return self.cls_seg(result)
