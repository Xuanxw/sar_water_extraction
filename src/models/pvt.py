"""PVT 系列水体分割：骨干构造、基础网络和 UANet 各模块。

按类阅读即可：PVTWaterNet 为简单基线，PVTUANet 为分阶段完整网络；
MBDC、PSM、CGM、UncertaintyRank、UAFM 的初始化和前向都在各自类中。
作者实现来源与 SAR 适配差异见 docs/stages/uanet_stages.md。
"""
import torch
import timm
from torch import nn
from torch.nn import functional as F

from .common import BasicConv2d, FPN, SemanticFPNDecoder


def build_pvt_backbone(in_channels=2, pretrained=True, backbone_name="pvt_v2_b2.in1k"):
    """返回步长 4/8/16/32、通道 64/128/320/512 的特征骨干。"""
    if backbone_name not in tuple(f"pvt_v2_b{i}.in1k" for i in (2, 3, 4, 5)):
        raise ValueError("当前解码器支持 PVT-v2 B2/B3/B4/B5，它们具有相同的四级通道数")
    # timm 负责 ImageNet 权重与输入通道适配；标准 B2 不是 B2-Li。
    return timm.create_model(
        backbone_name, pretrained=pretrained, in_chans=in_channels,
        features_only=True, out_indices=(0, 1, 2, 3), drop_path_rate=0.1,
    )


class PVTWaterNet(nn.Module):
    """PVT 基线：双通道 SAR → PVT → 通道投影 → 语义解码。"""

    def __init__(self, in_channels=2, num_classes=2, decoder_channels=32, pretrained=True):
        super().__init__()
        self.backbone = build_pvt_backbone(in_channels, pretrained)
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


class MBDC(nn.Module):
    """三分支空洞卷积，扩大感受野，并将不同尺度统一到相同通道数。"""

    def __init__(self, in_channels, channels=32, dilations=((4, 8, 16), (2, 4, 8))):
        super().__init__()
        # 两组分别是 branch1 / branch2 内三层卷积的空洞率；默认值保持历史模型。
        rate1, rate2 = dilations
        self.branch0 = nn.Sequential(BasicConv2d(in_channels, channels, 1))
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channels, channels, 1),
            BasicConv2d(channels, channels, 3, dilation=rate1[0]),
            BasicConv2d(channels, channels, 3, dilation=rate1[1]),
            BasicConv2d(channels, channels, 3, dilation=rate1[2]),
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, channels, 1),
            BasicConv2d(channels, channels, 3, dilation=rate2[0]),
            BasicConv2d(channels, channels, 3, dilation=rate2[1]),
            BasicConv2d(channels, channels, 3, dilation=rate2[2]),
        )
        self.conv_cat = BasicConv2d(channels * 3, channels, 3, padding=1)
        self.conv_res = BasicConv2d(in_channels, channels, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feature):
        """三个分支按通道拼接，经卷积压缩后与投影残差相加。"""
        branches = torch.cat((self.branch0(feature), self.branch1(feature), self.branch2(feature)), dim=1)
        return self.relu(self.conv_cat(branches) + self.conv_res(feature))


class PSM(nn.Module):
    """预测引导的空间增强；保留作者逐通道的空间注意力计算。"""

    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, feature, logits):
        """深层 [B,C,H,W] 与同尺寸前景分数构造 [B,HW,HW] 注意力。"""
        foreground = logits[:, 1:2]
        batch, channels, height, width = feature.shape
        key = foreground.reshape(batch, 1, height * width)
        enhanced = []
        for channel in range(channels):
            value = feature[:, channel:channel + 1].reshape(batch, 1, height * width)
            attention = self.softmax(torch.bmm(value.transpose(1, 2), key))
            output = torch.bmm(value, attention.transpose(1, 2))
            enhanced.append(output.reshape(batch, 1, height, width))
        return feature + self.gamma * torch.cat(enhanced, dim=1)


class CGM(nn.Module):
    """预测引导的通道增强；gamma 初始为 0，从恒等映射开始学习。"""

    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, feature, logits):
        """每个通道与前景分数做空间内积，得到 [B,C,1,1] 通道权重。"""
        batch, channels, _, _ = feature.shape
        foreground = logits[:, 1:2].reshape(batch, 1, -1)
        weights = torch.bmm(feature.reshape(batch, channels, -1), foreground.transpose(1, 2))
        weights = weights.sigmoid().unsqueeze(-1)
        return feature + self.gamma * weights * feature


class UncertaintyRank(nn.Module):
    """前景/背景五级权重；默认保留旧代码，水体分支按论文文字包含分箱左端点。"""

    def __init__(self, include_midpoint=False):
        super().__init__()
        self.include_midpoint = include_midpoint

    def forward(self, foreground_logits):
        """输入单通道 logits；输出两个同尺寸权重图，不向分段等级回传梯度。"""
        probability = foreground_logits.sigmoid()
        fore_distance = probability - 0.5
        back_distance = 0.5 - probability
        fore_rank = torch.zeros_like(probability)
        back_rank = torch.zeros_like(probability)
        # 越靠近 0.5，等级越高；p=0.5 时旧代码两侧为 0，水体分支两侧为 5。
        for threshold, rank in ((0.0, 5), (0.1, 4), (0.2, 3), (0.3, 2), (0.4, 1)):
            # 水体论文的文字分箱包含左端点；旧 UANet 保留公开代码的严格 >。
            fore_mask = fore_distance >= threshold if self.include_midpoint else fore_distance > threshold
            back_mask = back_distance >= threshold if self.include_midpoint else back_distance > threshold
            fore_rank[fore_mask] = rank
            back_rank[back_mask] = rank
        return fore_rank.detach(), back_rank.detach()


class UAFM(nn.Module):
    """不确定性融合：分别增强前景和背景特征，再融合高低层并产生细化预测。"""

    def __init__(self, channels=32, num_classes=2):
        super().__init__()
        self.water_paper = num_classes == 1
        self.rank = UncertaintyRank(include_midpoint=self.water_paper)
        # 水体论文式(5)(6)：1×1 压缩；旧建筑代码保留 3×3。
        kernel = 1 if self.water_paper else 3
        self.conv_high = BasicConv2d(channels * 2, channels, kernel)
        self.conv_low = BasicConv2d(channels * 2, channels, kernel)
        self.conv_fusion = nn.Conv2d(channels * 2, channels, 3, padding=1)
        # 水体论文式(7)用 3×3 产生单通道预测。
        self.seg_out = nn.Conv2d(channels, num_classes, 3, padding=1) if self.water_paper else nn.Conv2d(channels, num_classes, 1)

    def forward(self, feature_low, feature_high, logits):
        """高层预测引导两个尺度，返回融合特征及单/双通道 logits。"""
        foreground = logits[:, -1:]  # 双类别取水体通道；单通道直接取该 logit。
        fore_high, back_high = self.rank(foreground)
        high = torch.cat((fore_high * feature_high, back_high * feature_high), dim=1)
        high = F.interpolate(self.conv_high(high), size=feature_low.shape[-2:],
                             mode="bilinear", align_corners=True)

        # 水体论文与建筑公开代码在等级图的上采样顺序上不同。
        if self.water_paper:
            # 水体论文式(6)：最近邻上采样等级图，而不是插值 logit 后重新分级。
            fore_low = F.interpolate(fore_high, size=feature_low.shape[-2:], mode="nearest")
            back_low = F.interpolate(back_high, size=feature_low.shape[-2:], mode="nearest")
        else:
            low_logits = F.interpolate(foreground, size=feature_low.shape[-2:],
                                       mode="bilinear", align_corners=True)
            fore_low, back_low = self.rank(low_logits)
        low = self.conv_low(torch.cat((fore_low * feature_low, back_low * feature_low), dim=1))
        fusion = self.conv_fusion(torch.cat((high, low), dim=1))
        return fusion, self.seg_out(fusion)


class HighResolutionRefinement(nn.Module):
    """在约 1/2 尺度结合 SAR 局部纹理与最终语义预测，输出全分辨率 logit 修正量。"""

    def __init__(self, in_channels=2, channels=16):
        super().__init__()
        # 输入已经由训练/验证入口按训练集均值和标准差标准化，这里不重复归一化。
        # 使用普通卷积和 ReLU，避免为这一小分支引入额外的 BatchNorm 统计量。
        self.shallow = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(channels + 1, channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.delta = nn.Conv2d(channels, 1, 1)
        # 初始修正量严格为 0，使新网络从原来的四头输出开始学习。
        # 首次反向传播先更新此层，随后梯度才能传入浅层与融合卷积。
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, image, final_logits):
        """输入标准化 SAR 和 1/4 尺度最终 logits；返回与原图等大的修正 logits。"""
        detail = self.shallow(image)
        guidance = F.interpolate(final_logits, size=detail.shape[-2:],
                                 mode="bilinear", align_corners=True)
        # 拼接的是 logits，而不是概率；保留梯度，使最终头损失共同训练两条路径。
        correction = self.delta(self.fusion(torch.cat((detail, guidance), dim=1)))
        return F.interpolate(correction, size=image.shape[-2:],
                             mode="bilinear", align_corners=True)


class PVTUANet(nn.Module):
    """共享 PVT 结构：双通道为 UANet，单通道为 UAFNet；最后输出用于评价。"""

    def __init__(self, in_channels=2, num_classes=2, decoder_channels=32,
                 pretrained=True, stage="full", backbone_name="pvt_v2_b2.in1k", final_residual=False,
                 mbdc_dilations=None, high_resolution_refinement=False):
        super().__init__()
        if stage not in ("mbdc", "fpn", "uafm", "full"):
            raise ValueError("stage 应为 mbdc、fpn、uafm 或 full")
        if num_classes not in (1, 2):
            raise ValueError("num_classes=1 为水体 BCE，2 为原双类别 CE")
        if num_classes == 1 and stage != "uafm":
            raise ValueError("水体 UAFNet 使用 stage=uafm，不添加建筑网络 PSM/CGM")
        if final_residual and (num_classes != 1 or stage != "uafm"):
            raise ValueError("最终头残差对照只用于单通道 UAFNet")
        if high_resolution_refinement and (num_classes != 1 or stage != "uafm"):
            raise ValueError("高分辨率细化分支只用于单通道 UAFNet")
        self.high_resolution_refinement = high_resolution_refinement
        self.final_residual = final_residual  # 默认关闭，旧检查点仍按原结构推理。
        self.stage = stage
        channels = decoder_channels
        self.backbone = build_pvt_backbone(in_channels, pretrained, backbone_name)
        # None 为历史四级同率；显式表按浅到深排列，重载时由检查点恢复。
        self.mbdc_dilations = (mbdc_dilations if mbdc_dilations is not None
                               else (((4, 8, 16), (2, 4, 8)),) * 4)
        if len(self.mbdc_dilations) != 4:
            raise ValueError("MBDC 空洞率表需要按浅到深提供四级配置")
        # 对应作者 conv_2 到 conv_5；保持四个尺度，全部变为 channels 通道。
        self.conv_2 = nn.Sequential(MBDC(64, channels, self.mbdc_dilations[0]))
        self.conv_3 = nn.Sequential(MBDC(128, channels, self.mbdc_dilations[1]))
        self.conv_4 = nn.Sequential(MBDC(320, channels, self.mbdc_dilations[2]))
        self.conv_5 = nn.Sequential(MBDC(512, channels, self.mbdc_dilations[3]))
        if stage != "mbdc":
            self.neck = FPN(channels)
        self.decoder = SemanticFPNDecoder(channels, num_classes)
        if stage == "full":
            self.psm = PSM()
            self.cgm = CGM()
        if stage in ("uafm", "full"):
            self.ufm_layer4 = UAFM(channels, num_classes)
            self.ufm_layer3 = UAFM(channels, num_classes)
            self.ufm_layer2 = UAFM(channels, num_classes)
        if final_residual:
            # 最后一个预测卷积现在输出修正量：初始为 0，因此最终预测从粗预测开始。
            # 只清零该卷积，保留前三个预测头和其他特征层原有初始化。
            nn.init.zeros_(self.ufm_layer2.seg_out.weight)
            nn.init.zeros_(self.ufm_layer2.seg_out.bias)
        # 原 PVT 类还声明了不参与 forward 的 ufm_layer1；这里不注册这个闲置模块。
        # 放在所有原有层初始化之后：相同随机种子下，原有路径的初始权重保持一致。
        if high_resolution_refinement:
            self.highres_refinement = HighResolutionRefinement(in_channels)

    def forward(self, image):
        """512 输入时：128/64/32/16 四尺度；所有预测最终恢复到输入大小。"""
        size = image.shape[-2:]
        e2, e3, e4, e5 = self.backbone(image)
        f2 = self.conv_2(e2)
        f3 = self.conv_3(e3)
        f4 = self.conv_4(e4)
        f5 = self.conv_5(e5)
        features = (f2, f3, f4, f5)
        coarse = self.decoder(features if self.stage == "mbdc" else self.neck(features))
        coarse_full = F.interpolate(coarse, size=size, mode="bilinear", align_corners=True)
        if self.stage in ("mbdc", "fpn"):
            return coarse_full

        coarse_deep = F.interpolate(coarse, size=f5.shape[-2:], mode="bilinear", align_corners=True)
        if self.stage == "full":
            f5 = self.cgm(self.psm(f5, coarse_deep), coarse_deep)
        fusion4, prediction4 = self.ufm_layer4(f4, f5, coarse_deep)
        fusion3, prediction3 = self.ufm_layer3(f3, fusion4, prediction4)
        _, prediction2 = self.ufm_layer2(f2, fusion3, prediction3)
        if self.final_residual:
            # 两者都在 1/4 分辨率相加 logits，不相加概率，也不使用 16×16 引导图作基底。
            # 保留 coarse 的梯度，让最终损失可以直接训练 FPN 和粗解码器。
            prediction2 = coarse + prediction2
        final_full = F.interpolate(prediction2, size=size, mode="bilinear", align_corners=True)
        if self.high_resolution_refinement:
            # prediction2 已包含上面的 coarse 残差；此处只加细节修正，不再重复加粗预测。
            final_full = final_full + self.highres_refinement(image, prediction2)
        return (coarse_full,
                F.interpolate(prediction4, size=size, mode="bilinear", align_corners=True),
                F.interpolate(prediction3, size=size, mode="bilinear", align_corners=True),
                final_full)
