"""根据检查点中保存的模型名称重建网络；兼容历史 U-Net 文件。"""
from .unet import UNet


def load_model(checkpoint):
    """返回严格加载权重的 CPU 模型；重建时不再次下载 ImageNet 权重。"""
    name = checkpoint.get("model_name", "unet")
    if name == "unet":
        model = UNet(checkpoint["in_channels"], base_channels=checkpoint["base_channels"])
    elif name in ("pvt_uanet", "pvt_uafnet", "resnet50_uafnet", "pvt_unetpp_refinement"):
        from .pvt import PVTUANet

        options = dict(checkpoint["model_kwargs"])
        options["pretrained"] = False
        model = PVTUANet(**options)
    elif name == "pvt_deeplabv3plus":
        from experimental_model_deeplabv3 import PVTDeepLabV3Plus

        options = dict(checkpoint["model_kwargs"])
        options["pretrained"] = False
        model = PVTDeepLabV3Plus(**options)
    elif name == "ai4g_mobilenetv2_unet_dfc":
        # 独立实验入口中的模型类；仅重建结构，避免再次下载 ImageNet 权重。
        from experimental_model_unet_mobilenet import AI4GWaterUNet

        options = dict(checkpoint["model_kwargs"])
        options["pretrained"] = False
        model = AI4GWaterUNet(**options)
    elif name in ("pvt", "resnet50"):
        # 延迟导入，让只使用 U-Net 的用户不必加载 timm。
        from .pvt import PVTWaterNet
        from .resnet import ResNet50WaterNet

        options = dict(checkpoint["model_kwargs"])
        options["pretrained"] = False  # 完整训练权重已在检查点中。
        model_class = PVTWaterNet if name == "pvt" else ResNet50WaterNet
        model = model_class(**options)
    else:
        raise ValueError(f"不支持的模型名称：{name}")
    model.load_state_dict(checkpoint["model_state_dict"])
    return model
