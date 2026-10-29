"""PVT 系列水体分割：骨干构造、基础网络和 UANet 各模块。

按类阅读即可：PVTWaterNet 为简单基线，PVTUANet 为分阶段完整网络，PVTDeepLabV3Plus 为 ASPP 解码基线；
MBDC、PSM、CGM、UncertaintyRank、UAFM 的初始化和前向都在各自类中。
作者实现来源与 SAR 适配差异见 docs/stages/uanet_stages.md。
"""
import torch
import timm
from torch import nn
from torch.nn import functional as F

from .common import BasicConv2d, DoubleConv, FPN, SemanticFPNDecoder
from .resnet import build_resnet50_backbone


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


class NestedUNetDecoder(nn.Module):
    """U-Net++ 风格嵌套细化；保留原实验的三级输出尺度，不复制论文的监督配置。

    xij 中 i 是尺度层级（0 最浅），j 是嵌套列数；同层历史节点与下一层节点拼接。
    输入为 MBDC 的四级特征，输出依次为 1/16、1/8、1/4 尺度 logits。
    """

    def __init__(self, channels=32):
        super().__init__()
        self.x01 = DoubleConv(channels * 2, channels)
        self.x11 = DoubleConv(channels * 2, channels)
        self.x21 = DoubleConv(channels * 2, channels)
        self.x02 = DoubleConv(channels * 3, channels)
        self.x12 = DoubleConv(channels * 3, channels)
        self.x03 = DoubleConv(channels * 4, channels)
        # 与 UAFM 一样用 3×3 单通道头；没有第五个监督头。
        self.head4 = nn.Conv2d(channels, 1, 3, padding=1)
        self.head3 = nn.Conv2d(channels, 1, 3, padding=1)
        self.head2 = nn.Conv2d(channels, 1, 3, padding=1)

    def forward(self, features):
        """显式写出六个节点，便于逐层对照；按实际尺寸插值，兼容非整倍数输入。"""
        x00, x10, x20, x30 = features
        x01 = self.x01(torch.cat((x00, F.interpolate(
            x10, size=x00.shape[-2:], mode="bilinear", align_corners=True)), dim=1))
        x11 = self.x11(torch.cat((x10, F.interpolate(
            x20, size=x10.shape[-2:], mode="bilinear", align_corners=True)), dim=1))
        x21 = self.x21(torch.cat((x20, F.interpolate(
            x30, size=x20.shape[-2:], mode="bilinear", align_corners=True)), dim=1))
        x02 = self.x02(torch.cat((x00, x01, F.interpolate(
            x11, size=x00.shape[-2:], mode="bilinear", align_corners=True)), dim=1))
        x12 = self.x12(torch.cat((x10, x11, F.interpolate(
            x21, size=x10.shape[-2:], mode="bilinear", align_corners=True)), dim=1))
        x03 = self.x03(torch.cat((x00, x01, x02, F.interpolate(
            x12, size=x00.shape[-2:], mode="bilinear", align_corners=True)), dim=1))
        return self.head4(x21), self.head3(x12), self.head2(x03)


class PVTUANet(nn.Module):
    """共享 UANet/UAFNet 及嵌套细化对照；水体原路径可用 PVT 或 ResNet50 编码器。

    保留 PVTUANet 历史类名与模块名，旧检查点可严格重载，不额外复制一套解码器。
    """

    def __init__(self, in_channels=2, num_classes=2, decoder_channels=32,
                 pretrained=True, stage="full", backbone_name="pvt_v2_b2.in1k", final_residual=False,
                 mbdc_dilations=None, high_resolution_refinement=False, refinement_decoder="uafm"):
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
        if refinement_decoder not in ("uafm", "unetpp"):
            raise ValueError("refinement_decoder 应为 uafm 或 unetpp")
        if refinement_decoder == "unetpp" and (
            num_classes != 1 or stage != "uafm" or high_resolution_refinement
            or backbone_name != "pvt_v2_b2.in1k"
        ):
            raise ValueError("本轮嵌套细化对照使用 PVT-B2 单通道水体头，并关闭高分辨率分支")
        self.refinement_decoder = refinement_decoder  # 旧检查点缺省仍走 UAFM。
        self.high_resolution_refinement = high_resolution_refinement
        self.final_residual = final_residual  # 默认关闭，旧检查点仍按原结构推理。
        self.stage = stage
        channels = decoder_channels
        if backbone_name == "resnet50.tv_in1k":
            if num_classes != 1 or stage != "uafm":
                raise ValueError("ResNet50 骨干对照仅用于单通道 UAFNet")
            self.backbone = build_resnet50_backbone(in_channels, pretrained)
        else:
            self.backbone = build_pvt_backbone(in_channels, pretrained, backbone_name)
        # 只适配 MBDC 的输入通道，四级输出仍为 decoder_channels；后续解码完全复用。
        c2, c3, c4, c5 = self.backbone.feature_info.channels()
        # None 为历史四级同率；显式表按浅到深排列，重载时由检查点恢复。
        self.mbdc_dilations = (mbdc_dilations if mbdc_dilations is not None
                               else (((4, 8, 16), (2, 4, 8)),) * 4)
        if len(self.mbdc_dilations) != 4:
            raise ValueError("MBDC 空洞率表需要按浅到深提供四级配置")
        # 对应作者 conv_2 到 conv_5；保持四个尺度，全部变为 channels 通道。
        self.conv_2 = nn.Sequential(MBDC(c2, channels, self.mbdc_dilations[0]))
        self.conv_3 = nn.Sequential(MBDC(c3, channels, self.mbdc_dilations[1]))
        self.conv_4 = nn.Sequential(MBDC(c4, channels, self.mbdc_dilations[2]))
        self.conv_5 = nn.Sequential(MBDC(c5, channels, self.mbdc_dilations[3]))
        if stage != "mbdc":
            self.neck = FPN(channels)
        self.decoder = SemanticFPNDecoder(channels, num_classes)
        if stage == "full":
            self.psm = PSM()
            self.cgm = CGM()
        if refinement_decoder == "unetpp":
            # 放在共有层之后创建，同种子下 PVT、MBDC、FPN、粗解码头的初值不变。
            self.nested_decoder = NestedUNetDecoder(channels)
        elif stage in ("uafm", "full"):
            self.ufm_layer4 = UAFM(channels, num_classes)
            self.ufm_layer3 = UAFM(channels, num_classes)
            self.ufm_layer2 = UAFM(channels, num_classes)
        if final_residual:
            # 两种细化方式都输出 logit 修正量，初始最终预测等于同一个粗预测。
            final_head = (self.nested_decoder.head2 if refinement_decoder == "unetpp"
                          else self.ufm_layer2.seg_out)
            nn.init.zeros_(final_head.weight)
            nn.init.zeros_(final_head.bias)
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

        if self.refinement_decoder == "unetpp":
            # 用嵌套特征融合替换三级不确定性引导；粗预测继续经下面的残差参与最终输出。
            prediction4, prediction3, prediction2 = self.nested_decoder(features)
        else:
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
