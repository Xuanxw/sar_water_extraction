# 模型代码结构与函数索引

模型按骨干集中在少量脚本中。每种方法的初始化和前向放在自己的类里，公共组件只保留一份定义。

## 1. 当前目录

```text
src/models/
├─ unet.py        # UNet
├─ pvt.py         # PVT 基线、PVT-UANet 及其特有模块
├─ resnet.py      # ResNet50WaterNet
├─ common.py      # 公共卷积、双卷积、FPN、语义解码头
├─ checkpoint.py  # 按元信息重建并严格加载模型
└─ __init__.py
```

数据读取位于 `datasets/water_dataset.py`；训练入口、方法选择、参数与完整训练循环都位于同一个 `train.py`。

## 2. 按类查找函数

| 文件 | 类 / 函数 | 作用与主要方法 |
|---|---|---|
| [unet.py](../src/models/unet.py) | `UNet` | `__init__` 创建网络；`forward` 编解码；`load_state_dict` 兼容历史层名 |
| [pvt.py](../src/models/pvt.py) | `build_pvt_backbone` | 创建基线与 UANet 共用的 timm PVT 骨干 |
| 同上 | `PVTWaterNet` | 本类的 `__init__`、`forward` 包含骨干、投影、解码与上采样 |
| 同上 | `MBDC` | `__init__` 定义三分支与残差；`forward` 拼接、压缩并相加 |
| 同上 | `PSM` | `__init__` 定义 γ 和 softmax；`forward` 计算空间注意力 |
| 同上 | `CGM` | `__init__` 定义 γ；`forward` 计算通道门控 |
| 同上 | `UncertaintyRank` | `forward` 生成前景/背景等级，无需自定义初始化 |
| 同上 | `UAFM` | `__init__` 定义加权、融合和分类层；`forward` 返回特征与预测 |
| 同上 | `PVTUANet` | `__init__` 按 stage 创建模块；`forward` 组织粗预测和三级细化 |
| [resnet.py](../src/models/resnet.py) | `ResNet50WaterNet` | 本类的 `__init__`、`forward` 包含 ResNet 骨干、投影、解码与上采样 |
| [common.py](../src/models/common.py) | `BasicConv2d` | `__init__` 定义卷积/BN/ReLU；`forward` 顺序执行 |
| 同上 | `DoubleConv` | `__init__` 创建两个基础块；`forward` 连续执行 |
| 同上 | `FPN` | `__init__` 定义侧向/输出卷积；`forward` 自顶向下融合 |
| 同上 | `SemanticFPNDecoder` | `__init__` 定义四条支路；`forward` 对齐、相加并分类 |
| [checkpoint.py](../src/models/checkpoint.py) | `load_model` | 根据模型名、参数重建网络，关闭预训练下载并严格加载权重 |

## 3. 阅读顺序与继承关系

```text
train.py：main() → create_trainer()，选择 METHOD="uanet"
  → train.py：创建 WaterSegmentationTrainer，无训练器子类
  → models/pvt.py：PVTUANet.__init__ / forward
       ├─ 同文件：build_pvt_backbone
       ├─ 同文件：MBDC、PSM、CGM、UncertaintyRank、UAFM
       └─ common.py：BasicConv2d、FPN、SemanticFPNDecoder
  → 返回训练器：损失、更新、验证、保存
```

先看 `PVTUANet.forward()` 确认连接，再在同一文件内定位具体模块。FPN 的构造和前向集中在 `FPN` 类中，ResNet 分割流程集中在 `ResNet50WaterNet` 类中；timm 提供骨干内部的实现，不在本项目重写。

所有完整分割模型直接继承 `nn.Module`，不再经过 `BackboneWaterNet` 中间父类。PVT 和 ResNet 的短前向流程在本类直接展开，方便顺序阅读。公共卷积、FPN、解码头仍各只有一份类定义；PVT 骨干构造也只有一份。

解码头中带 bias 的卷积保留原始 Sequential 结构，不强行替换为 `bias=False` 的 `BasicConv2d`，以保持参数语义与历史权重名称一致。

训练入口也已合并：不再定义 UNetTrainer/PVTTrainer 等子类，四个同级方法由 `METHOD` 选择，然后直接创建同一个 `WaterSegmentationTrainer`。公共训练器位于 `src/train.py`；没有迁移 Lightning 或添加配置、argparse 框架。

统一入口的调用为 `main() → create_trainer() → WaterSegmentationTrainer(...) → train()`。`OUTPUT_DIR=None` 时根据 METHOD 选择原实验目录；uanet 还根据 STAGE 分目录。METHOD 的四个值为 unet、pvt、resnet50、uanet，只有 uanet 使用 STAGE。

入口合并已用固定合成数据对照两轮训练：UNet、PVT、ResNet50、UANet-mbdc 和 UANet-full 的逐轮指标、标准化、划分、保存轮次与权重均与合并前一致。另有 7 项模型测试和 3 项统一入口选择测试通过；没有启动正式数据训练。

## 4. 导入、运行与旧权重

```python
from models.unet import UNet
from models.pvt import PVTWaterNet, PVTUANet, MBDC, PSM, CGM, UncertaintyRank, UAFM
from models.resnet import ResNet50WaterNet
from models.common import BasicConv2d, DoubleConv, FPN, SemanticFPNDecoder
from models.checkpoint import load_model
```

四种方法统一运行 `python src/train.py`；顶部 `METHOD` 选择 unet/pvt/resnet50/uanet，`STAGE` 仅对 uanet 生效。在 `src` 目录可执行 `python -m models.unet` 查看 U-Net 尺寸示例。

原来的多层拆分导入路径已收拢，项目内入口、测试和文档同步更新；外部临时脚本应按上述路径修改导入。

模型属性名、检查点元信息、数据划分、标准化、输出顺序、损失和训练参数保持不变，已有权重字典无需转换。stage 仍控制模块启用，不要求复制多套实现，也不会自动接着上一阶段训练。

本轮已用固定初始化和输入，对比 U-Net、PVT、ResNet 与 UANet 四种阶段共 7 种配置：整理前后的状态字典名称、数值和前向输出完全一致；full 对照将 PSM/CGM 的 γ 设为非零以覆盖增强路径。现有 7 项模型测试全部通过，包含梯度、严格重载和合成数据训练/校准接口检查。本轮没有启动正式数据训练。

网络计算见 [U-Net 函数说明](unet_functions_and_derivations.md) 与 [PVT-UANet 函数说明](pvt_uanet_functions_and_derivations.md)。

"""
| 模块 | 属于 Transformer 吗？ | 主要计算 |
|---|---|---|
| **PVT 骨干** | **是** | 多头空间缩减注意力、MLP、LayerNorm、残差 |
| **MBDC** | 否 | 多分支空洞卷积、拼接、残差 |
| **FPN** | 否 | 卷积、上采样、逐层相加 |
| **语义解码头** | 否 | 卷积、上采样、多尺度相加 |
| **PSM / CGM** | 不属于完整 Transformer Block | 空间注意力／通道调制 |
| **UAFM** | 否 | 不确定性加权、卷积与特征融合 |
| **普通 U-Net / ResNet** | 否 | 主要使用卷积结构 |
"""

## 5. 完整 UAFNet 的 ResNet50 编码器对照

`train.py:create_trainer()` 保持 `METHOD="uafnet"`，通过 `UAFNET_BACKBONE` 选择 PVT-v2-B2 或 `resnet50.tv_in1k`。为了严格兼容旧检查点，完整模型暂沿用 `models/pvt.py:PVTUANet` 类名；ResNet50 的构建集中在 `models/resnet.py:build_resnet50_backbone()`，与旧的简单 ResNet50 基线共用，不复制解码器，也不增加训练器子类。

512×512 SAR 输入时：

| 编码器 | 四级特征通道 | 四级空间尺寸 |
|---|---|---|
| PVT-v2-B2 | 64、128、320、512 | 128²、64²、32²、16² |
| ResNet50 | 256、512、1024、2048 | 128²、64²、32²、16² |
| 各自 MBDC 输出 | 均为32 | 128²、64²、32²、16² |

两组都经过 MBDC → FPN → 语义解码头产生粗预测。各级 MBDC 特征同时进入三级 UAFM，粗预测引导逐级细化；最后在1/4分辨率相加粗 logits 与修正 logits，再恢复512×512。训练返回四个全分辨率输出，最终头用于验证。本次高分辨率细化关闭，固定空洞率、等权 BCE 和全局困难采样保持一致。

新检查点记录 `model_name="resnet50_uafnet"` 及实际 `model_kwargs.backbone_name`，`models/checkpoint.py:load_model()` 离线严格重建。旧 PVT 与简单 ResNet50 模型名称仍可加载。ResNet50 替换是本项目的骨干消融，不等同于原论文配置；输入通道适配使 MBDC 参数量也发生必要变化。


## 2026-10-28：PVT 嵌套细化对照

当前 `train.py` 默认设置 `UAFNET_REFINEMENT_DECODER="unetpp"`，关闭高分辨率分支。保留 PVT/MBDC/FPN/粗预测，仅替换三级 UAFM 为 `models/pvt.py:NestedUNetDecoder` 的六节点密集跳跃融合；`DoubleConv` 复用 common.py。四头等权 BCE 和最终粗预测残差保持不变。

新模型在检查点中命名为 `pvt_unetpp_refinement`，由 `models/checkpoint.py` 重建；输出进入 `pvt_unetpp_hard_sampling_residual_grouped`。将开关恢复为 `uafm` 即恢复原 grouped PVT 控制路径。节点尺寸、公式和非原版复现的边界见 [结构推导第29节](pvt_uanet_functions_and_derivations.md#29-pvt--u-net-风格嵌套细化对照)。
