"""ResNet 骨干的水体语义分割网络。"""
import timm
from torch import nn
from torch.nn import functional as F

from .common import SemanticFPNDecoder


class ResNet50WaterNet(nn.Module):
    """ResNet-50 基线；构造和前向计算集中在同一个类中。"""

    def __init__(self, in_channels=2, num_classes=2, decoder_channels=32, pretrained=True):
        super().__init__()
        # 跳过步长 2 的 stem，使用步长 4/8/16/32 的四层特征。
        self.backbone = timm.create_model(
            "resnet50.tv_in1k", pretrained=pretrained, in_chans=in_channels,
            features_only=True, out_indices=(1, 2, 3, 4),
        )
        self.projections = nn.ModuleList([
            nn.Conv2d(channels, decoder_channels, 1)
            for channels in self.backbone.feature_info.channels()
        ])
        self.decoder = SemanticFPNDecoder(decoder_channels, num_classes)

    def forward(self, image):
        """骨干提取四尺度特征，再投影、解码并恢复输入尺寸。"""
        features = self.backbone(image)
        features = [projection(feature) for projection, feature in zip(self.projections, features)]
        logits = self.decoder(features)
        return F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=True)
