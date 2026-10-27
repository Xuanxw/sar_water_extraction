# SAR water extraction

## 当前实验：固定分组划分，先重跑控制基线再测试高分辨率细化

数据已按清单放入 `data/Track1/train/images/{train,val}` 和 `labels/{train,val}`；`data/Track1/train/split_manifest.csv` 记录场景、组号、固定归属和 `active`。重复副本及标签冲突文件保留在物理目录，只有 `active=1` 的记录进入训练和验证，不再随机拆分。

| 集合 | 物理文件对数 | active 样本数 |
|---|---:|---:|
| train | 1314 | 1144 |
| val | 317 | 287 |

`IMAGE_DIR`、`LABEL_DIR` 仍填写 images/labels 父目录，训练器读取固定清单；归一化只统计 1144 个有效训练样本。检查点保存逻辑文件名，实验目录保存清单快照及 SHA256，旧检查点的裸文件名仍可解析。

当前保留固定 MBDC、全局困难采样、最终头残差、四头等权 BCE，按以下顺序运行 `python src/train.py`：

| 轮次 | `UAFNET_HIGH_RESOLUTION_REFINEMENT` | 输出父目录 |
|---|---|---|
| 新划分控制组（当前默认） | `False` | `outputs/pvt_uafnet_hard_sampling_residual_grouped/` |
| 轻量高分辨率细化对照 | `True` | `outputs/pvt_uafnet_hard_sampling_residual_highres_grouped/` |

两轮均设 `UAFNET_MBDC_DILATIONS=None`，从相同 ImageNet 骨干初始化开始，其余训练参数不变。先完成控制组，再只切换上述开关；不能用旧划分模型的分数直接替代新控制组。新分支在半分辨率融合 SAR 细节与现有最终 logits，输出零初始化修正量；四个监督头不增加。实现与尺寸流见 [结构说明第28节](docs/pvt_uanet_functions_and_derivations.md#28-轻量高分辨率细化分支)。目前只有接口与合成测试通过，尚无新划分下的精度改善结论。

训练后将 `src/validation.py` 的 `CHECKPOINTS` 指向对应新实验 `best_f1.pth`，保留 `USE_SAVED_VAL=True`，比较共同 287 张验证图上的原始整体及边界指标。传统法对比通过 `src/evaluation/compare_threshold.py`：先填写新 grouped 实验的 `CHECKPOINT`；入口核对当前清单 SHA256、训练名单和验证名单，再从物理 `images/val`、`labels/val` 读取同一批287张有效图。旧划分模型会被拒绝。独立 `src/thres_seg/main.py` 同样默认只处理清单内 active val，不加载模型。旧检查点仍可按原名单复核历史结果，但其训练样本可能进入新 val，不能当作新划分的独立验证成绩。当前场景分类及去重分组仍不等于已证实地理/事件独立，不能据此直接开启区域 Bootstrap。详见 [验证说明](docs/validation_baseline.md)。

## 已完成实验：按特征尺度配置 MBDC 空洞率

本轮从“全局困难采样 + 最终残差 + 等权 BCE”出发，只修改 MBDC 的空洞率；正式训练由用户执行。保持 PVT-v2-B2、VV/VH、固定划分、种子42、batch size 2、20轮、学习率1e-4、AdamW、增强和采样规则不变，使用原来的 ImageNet 骨干初始化方式重新训练，不接着旧 `best_f1.pth` 微调。

```python
METHOD = "uafnet"
UAFNET_FINAL_RESIDUAL = True
LOSS_NAME = "bce"
SAMPLING = "hard"
HEAD_LOSS_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
UAFNET_MBDC_DILATIONS = (
    ((4, 8, 16), (2, 4, 8)),  # 512输入：128×128特征
    ((2, 4, 8), (1, 2, 4)),   # 64×64
    ((1, 2, 4), (1, 1, 2)),   # 32×32
    ((1, 1, 2), (1, 1, 1)),   # 16×16
)
```

原配置在16×16特征图上使用 dilation=16 的3×3卷积时，周围八个采样位置全部落入补零区；本轮缩小深层空洞率以恢复有效邻域采样。参数量、卷积层数、输出形状、FPN和UAFM均不变。这是待验证的结构假设，不保证精度提升，也不是原论文配置。

运行 `python src/train.py`，结果进入 `outputs/pvt_uafnet_hard_sampling_residual_scale_mbdc/时间戳/`。完整配置写入检查点的 `model_kwargs.mbdc_dilations`，评价和后处理据此重建结构。旧检查点缺少该字段时保持原固定空洞率，不受训练脚本顶部新配置影响。

**复跑下文历史 UAFNet 实验时，先设 `UAFNET_MBDC_DILATIONS=None`，再按原方案设置头权重、采样、残差和损失。** 本轮的基线选择、结果比较和尺度推导见 [结构说明第25节](docs/pvt_uanet_functions_and_derivations.md#25-按特征尺度配置-mbdc-2026-10-27)。

## 已完成实验：四个输出头的监督权重

该轮以“全局困难采样 + 最终残差 + BCE”为控制基线，只调整四个输出头的损失系数。F1由83.0543%升至83.1578%，Recall由80.4098%降至79.2793%；尚不能据此确认稳定改善。本轮结构实验恢复等权监督，便于独立归因。损失概念、公式与后续顺序见 [水体分割损失函数说明](docs/loss_functions_explained.md)。

```python
METHOD = "uafnet"
UAFNET_FINAL_RESIDUAL = True
LOSS_NAME = "bce"
SAMPLING = "hard"
HEAD_LOSS_WEIGHTS = (1.0, 0.5, 0.5, 2.0)
UAFNET_MBDC_DILATIONS = None  # 复跑历史实验使用原固定空洞率。
```

权重依次对应粗预测、第一级 UAFM、第二级 UAFM、最终预测，训练目标为 `BCE_coarse + 0.5*BCE_4 + 0.5*BCE_3 + 2*BCE_final`，不额外除以权重和。系数总和仍为4，只保持名义尺度，不保证梯度范数相同；最终残差损失也会回传粗分支。模型、划分、种子、batch size 2、20轮、学习率1e-4、AdamW、增强和困难采样规则保持控制基线设置。

按本节配置运行 `python src/train.py`，双极化结果进入 `outputs/pvt_uafnet_hard_sampling_residual_head_weights/时间戳/`。检查点新增 `head_loss_weights`；验证仍只计算最终头的普通 BCE，困难评分仍用逐图最终头 BCE，`best.pth` 取最低验证 loss、`best_f1.pth` 取最高验证 F1 的规则不变。评分规则相同不代表训练后困难池名单固定不变。

`HEAD_LOSS_WEIGHTS` 在统一入口中只对 `uafnet` 生效，其他方法保持旧监督方式。**复跑下文任一历史 UAFNet 实验时，须先设 `HEAD_LOSS_WEIGHTS=(1.0,1.0,1.0,1.0)`，再按对应章节设置采样、残差和损失。** 旧检查点仍可直接评价，不必修改或删除；不同权重的新训练使用独立目录。后续区域损失、裁剪增强仍待逐项实施，本轮未加入。

## 训练过程中的中间特征图

默认 `SAVE_FEATURE_MAPS=False`，不再输出训练中间特征图。需要专项诊断时，可在 `src/train.py` 顶部设置 `SAVE_FEATURE_MAPS=True`、`FEATURE_MAP_INTERVAL=5`、`FEATURE_MAP_SAMPLES=2`。适用于 `PVTUANet` 构建的 uanet/uafnet；第1轮、每5轮和最后一轮，输出固定验证首批的前2幅（批次不足2幅时按实际数量）。当前20轮实验对应1/5/10/15/20轮。首批标签全部无效时该批跳过，不另换样本。

每幅样本保存到 `本次实验/features/epoch_XXX/原文件名.png`，包含输入SAR最多前两个波段、真值、粗/最终水体概率，以及PVT四级、MBDC四级、FPN四级、三级UAFM融合特征。未启用的模块面板留空。所有图来自当前轮的 eval 模式验证前向，不是训练增强后的随机样本，也不更新BN统计。采集使用临时钩子，不改变模型输出、损失、采样或检查点格式；没有额外模型前向，绘图仍会占用少量时间，验证耗时包含该开销。

多通道特征用 `mean(abs(feature), dim=channel)` 汇总为二维响应，每张图单独按2%～98%分位拉伸，并标明原始尺寸与显示范围。它不是水体概率、注意力权重或完整逐通道特征，不能根据不同面板的亮度直接比较绝对激活强弱。概率面板固定0～1，SAR显示文件中的存储值，不新增dB转换。未保存大体积原始特征张量。

实现集中在 `src/evaluation/visual.py` 的 `FeatureVisualizer`，训练入口只负责选择轮次和批次。关闭时设 `SAVE_FEATURE_MAPS=False`。已启动的Python进程不会自动载入本次代码修改；新启动训练才会启用输出，已完成轮次的图不会自动补生成。

## 代码组织与统一训练入口

所有模型从 `src/train.py` 启动，在顶部直接选择方法和参数，不使用模型专用训练器子类、argparse 或配置框架。

```text
src/
├─ datasets/water_dataset.py
├─ models/
│  ├─ unet.py
│  ├─ pvt.py
│  ├─ resnet.py
│  ├─ common.py
│  └─ checkpoint.py
├─ evaluation/
│  ├─ postprocess.py # 最终质量评价与后处理，由 train.py 调用
│  ├─ model_eval.py  # 独立模型评价与概率阈值扫描
│  ├─ metrics.py     # 共用像素指标
│  ├─ boundary_eval.py # 内外缓冲带与固定点质量评价
│  └─ visual.py      # 可选中间特征图，默认关闭
├─ thres_seg/
│  ├─ main.py        # 独立阈值分割，或原 GUI 入口
│  ├─ threshold.py   # 百分位阈值与 Lee 近似去斑
│  ├─ pipeline.py    # 按目录处理影像及可选标签评价
│  ├─ gui.py
│  ├─ gis_worker.py
│  └─ esa_worker.py
└─ train.py          # 训练与后处理统一入口
scripts/
├─ evaluate.py       # 原命令的兼容入口，参数位于 evaluation/model_eval.py
└─ calibrate_bn.py   # 可选 BN 校准
```

修改 `train.py`：

```python
METHOD = "uafnet"  # 同级可选 unet、pvt、resnet50、uanet、uafnet
STAGE = "uafm"     # 只对 uanet 生效，可选 mbdc、fpn、uafm、full
```

在项目根目录运行：

```powershell
conda activate sar-water
python src/train.py
```

| METHOD | 完整模型 | 默认保存目录 |
|---|---|---|
| `unet` | `UNet` | `outputs/unet_baseline/时间戳/` |
| `pvt` | `PVTWaterNet` | `outputs/pvt_baseline/时间戳/` |
| `resnet50` | `ResNet50WaterNet` | `outputs/resnet50_baseline/时间戳/` |
| `uanet` | `PVTUANet` | `outputs/pvt_uanet_阶段/时间戳/` |
| `uafnet` | 复用 `PVTUANet` 的单通道水体论文分支 | `outputs/pvt_uafnet/时间戳/` |

表中是基础目录；`OUTPUT_DIR=None` 时还会按损失、采样、残差、高分辨率细化、尺度 MBDC、非等权监督及极化设置追加后缀。读取固定分组清单时，训练器还会追加 `_grouped`；显式指定的 `OUTPUT_DIR` 也追加该分组后缀，当前完整目录见本页开头。每次仍创建独立时间戳目录。`uanet` 是训练入口中的方法名，检查点继续记录 `model_name="pvt_uanet"`，兼容历史权重。

调用关系为 `__main__ → main() → create_trainer() → WaterSegmentationTrainer(...) → trainer.train() → evaluation.postprocess.run_postprocess(...)`；最后一步由 `RUN_MODE` 控制。`WaterSegmentationTrainer` 与入口同在 `train.py`，直接完成数据准备、标准化、训练/验证、指标与保存；已删除独立训练器文件和模型专用入口。

模型按骨干集中组织，类与方法的定位见 [模型结构与函数索引](docs/model_structure.md)。已有四种方法保留原模型和训练规则；新增 `uafnet` 的训练设置见下节。旧 U-Net 数字层名仍由 `UNet.load_state_dict` 兼容。

## UAFNet 水体论文训练

在 `src/train.py` 设置 `METHOD="uafnet"`，仍运行 `python src/train.py`。所有训练逻辑留在同一个 `WaterSegmentationTrainer` 内，没有新增训练入口或训练器子类。

依据 [UAFNet 水体论文](docs/papers/2024_UAFNet_SARFloodMapping_IGARSS_Published.pdf) 第 2.2、3.1 节实现：

- 单模型 PVT-v2-B2 → MBDC → FPN/语义解码头产生粗预测 → 三级 UAFM 细化；不添加建筑 UANet 的 PSM/CGM。
- 四个输出均为 `[B,1,H,W]` 的水体 logits，论文基线的训练损失是四个有效像素平均 BCE 等权之和；当前控制组保留等权监督及固定 MBDC；尺度空洞率为历史对照，见开头。使用数值稳定的 `BCEWithLogitsLoss`，标签 `-1` 不参与计算。
- AdamW、初始学习率 `1e-4`、每轮余弦退火；验证只用最终输出，`sigmoid(logit)>=0.5` 判为水体。
- 水体 UAFM 使用论文式(5)(6)的 1×1 特征压缩、最近邻上采样等级图、式(7)的 3×3 预测卷积；原双通道 UANet 的计算保持不变。
- 五级权重按论文文字区间 `[0,.1)` 等分箱，`p=.5` 时两侧均为等级 5。论文式(3)的 floor 写法与文字区间有边界差异，本实现明确采用文字区间，等级不缩放；等级图继续 detach。

当前仍是**参考论文的本数据集适配版**：

- 保留 `(1,2)` 两通道与训练集统计标准化，未推断极化顺序或数值单位。论文输入为 `VH,VV,VH`，只有确认文件的极化对应关系后才可设置重复波段，例如确认 1=VV、2=VH 后使用 `(2,1,2)`。
- batch size 2、20 轮、权重衰减 0.01、余弦最低学习率 0 为本项目设置，论文没有提供完整数值；不能称为原作者全部超参数。
- `TRAIN_AUGMENTATION=True` 时，仅训练批次随机做 90° 倍数旋转与水平翻转，影像和标签同步。它覆盖方形切片的八种正交变换，不是论文未详述角度和翻转方式的 12 倍扩增。验证不增强，先划分再增强。
- MBDC、FPN 解码头及高层融合特征的双线性插值复用现有实现，论文未完整明确的细节不声称逐行复现。默认训练 B2；后续成员骨干和土地覆盖处理见下节，测试增强仍未加入。

原始等权、普通采样且无最终残差的 B2 论文适配基线创建 `outputs/pvt_uafnet/时间戳/`；其他对照按设置追加目录后缀。保存 `metrics.csv`、最低验证 loss 的 `best.pth`、最高验证 F1 的 `best_f1.pth`、`last.pth`，异常时另存 `abnormal.pth`。新检查点名为 `pvt_uafnet`，模型加载器已支持；另保存优化器、调度器和 AMP 状态，但本次没有加入断点续训入口。不要直接把旧双类别 UANet 权重严格加载到单通道 UAFNet。

旧 `calibrate_bn.py` 的评价逻辑仍面向双类别 CE，不应直接用于单通道 UAFNet；本阶段先正常训练和验证，不额外校准 BN。

接口检查：`python -m unittest discover -s tests -p test_uafnet_training.py -v`。该检查使用合成小样本，不代表正式数据精度。新训练同时改变损失、优化器、调度、增强和融合细节，不能把它与旧 FPN 的差异全部归因于 UAFM。

## 后处理与多模型集成

执行入口统一为 `python src/train.py`，在其顶部配置：

```python
RUN_MODE = "train_postprocess"  # 默认：训练完成后自动后处理。
POSTPROCESS_CHECKPOINTS = []    # 仅后处理模式填写已训练模型路径。
POSTPROCESS_OPTIONS = dict(
    use_saved_val=True, with_labels=True,
    threshold=0.5, use_landcover=True, landcover_band=5, water_class=80,
    save_masks=True, save_visuals=True, max_visualizations=None,
)
```

| RUN_MODE | 执行内容 |
|---|---|
| `train_postprocess` | 训练完成后加载本次最佳模型，对原验证集后处理 |
| `train` | 只训练、验证和保存检查点 |
| `postprocess` | 不训练，使用 `POSTPROCESS_CHECKPOINTS` 中的已有模型 |

训练衔接时，单通道水体模型选择本次目录中的 `best_f1.pth`，其他双类别模型选择最低验证loss的 `best.pth`，不扫描其他实验目录，也不误用后处理原来的硬编码旧模型。先释放训练器及优化器占用，再重载最佳模型。该模式固定使用本次影像/标签目录和检查点的验证名单，沿用训练batch size和AMP设置；后处理保存在本次实验的 `quality/时间戳/`。训练失败不会继续后处理，后处理失败也不会删除已保存权重。

已有结果无需重新训练：将 `RUN_MODE="postprocess"`，填写 `POSTPROCESS_CHECKPOINTS=[r"实际实验目录/best_f1.pth"]`，仍运行同一个命令。多模型集成也通过该列表传入。默认 `max_visualizations=None` 会绘制全部验证影像，需要时间；可改成10先查看少量图，精度仍统计整个验证集。

算法仍集中在 `src/evaluation/postprocess.py`，不再直接运行该文件；其 `run_postprocess` 必须显式接收检查点列表。`src/evaluation/model_eval.py` 继续作为独立指标评价/阈值扫描入口，不会在训练结束后自动重复扫描。

当前数据的第 5 波段在完整 326 幅内部验证切片中观测到类别 `0,10,20,30,40,50,60,70,80,90,100`，与 ESA WorldCover 编码一致；由于文件没有波段描述，此位置基于本地观测配置，不代表任意六波段文件都具有相同顺序。代码 80 为永久水体（[ESA WorldCover 官方产品手册](https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/docs/WorldCover_PUM_V2.0.pdf)）；0/255 作为无效土地覆盖类别，不参与补充。更换数据前应核实波段及类别对应关系。

本实现明确采用“模型水体 **或** 有效土地覆盖水体”的合并规则，参考论文对土地覆盖水体区域的补充说明；论文未公布全部实现细节，不声称逐行一致。不把土地覆盖的非水体区域设为硬排除条件，避免删除新洪水；也未加入 DEM 高程硬阈值或小斑块删除。

默认沿用检查点的内部验证名单及原始输入归一化。终端分别报告各成员、概率集成结果、后处理结果的 F1/IoU/P/R。推理与修正不读取标签值作决策，标签仅用于最后统计。无标签数据设置 `POSTPROCESS_OPTIONS` 中的 `with_labels=False`、`use_saved_val=False`，并指定 `image_dir`（使用仅后处理模式），不会计算 F1。

`save_masks=True` 时，保存到第一个检查点目录下 `quality/时间戳/raw/` 和 `postprocessed/`，值为 0 非水体、1 水体、255 无效；原始检查点不被覆盖。源数据没有地理参考时，输出 TIFF 同样没有可靠地理定位，不能据此生成地理矢量。`save_masks=False` 时不保存掩膜；精度表与 `save_visuals` 控制的对照图仍按下文输出。

当前 B2 的完整内部验证结果（326 幅、阈值 0.5、AMP）：

| 结果 | F1 | IoU | Precision | Recall |
|---|---:|---:|---:|---:|
| 原始 B2 | 0.790527 | 0.653613 | 0.846358 | 0.741606 |
| 土地覆盖合并后 | 0.813775 | 0.686020 | 0.852546 | 0.778376 |

本次补入 217,914 个像素，均对应此验证集的水体标签，FP 未增加。这是辅助先验在当前数据上的收益，不代表纯 SAR 网络改善，也不能直接对照论文官方测试成绩；新地区、不同年份的土地覆盖可能引入误检。

多模型集成已接入同一个脚本的 `WaterEnsemble` 类：每个成员先按各自波段顺序、均值和标准差处理输入，再取最终头的水体概率等权平均，最后阈值化；不直接平均 logits，不对二值图投票。当前默认一个成员，相当于普通推理，不声称已经取得集成收益。不要同时加入同一实验、相同权重的 `best.pth` 和 `best_f1.pth`，也不要默认混入较弱的旧消融模型。使用原内部验证集时，成员必须保存相同的验证名单，避免把某成员训练过的影像用于集成评价。

后续在 `train.py` 修改 `UAFNET_BACKBONE` 为 `pvt_v2_b3.in1k`、`pvt_v2_b4.in1k` 或 `pvt_v2_b5.in1k`，分别独立训练。默认 B2 仍保存到 `outputs/pvt_uafnet/`，其他成员分别保存到 `outputs/pvt_uafnet_b3/` 等目录，检查点记录实际骨干。骨干变大后显存需求增加；先尝试 B3，再根据显存决定是否继续。当前尚未启动这些新训练，也没有 B3/B4/B5 实测精度。

集成是否采用，以同一固定验证集上的“各成员 → 集成 → 集成后处理”结果为依据；平均多个模型并不保证改善。接口检查可运行 `python -m unittest discover -s tests -p test_postprocess.py -v`。

## 损失函数对照实验

损失统一放在 `src/models/losses.py` 的 `WaterSegmentationLoss` 中；`src/train.py` 顶部用 `LOSS_NAME` 选择。当前使用 BCE、全局困难采样及最终头残差，保持等权监督及固定 MBDC；先重跑分组控制基线，再测试高分辨率细化，详见本页开头。下表与后文保留历史损失、采样实验的配置；其基础路径未计入后续残差、采样及头权重后缀。

| 策略 | 单个输出头损失 | 默认输出父目录 |
|---|---|---|
| `bce` | 原有效像素 BCEWithLogits 均值，保持历史公式 | `outputs/pvt_uafnet/` |
| `bce_dice` | `BCE_WEIGHT * BCE + DICE_WEIGHT * SoftDiceLoss` | `outputs/pvt_uafnet_bce_dice/` |
| `ce` | 原双类别交叉熵，用于 U-Net、PVT、ResNet50、UANet | 沿用各自历史目录 |

BCE+Dice 第一轮两个系数都为 1，即保留 BCE 系数并增加 Dice 项，不加类别权重。Dice 在每幅图的有效像素上计算 `1-(2*sum(p*y)+1)/(sum(p)+sum(y)+1)`，再对有效图像求均值；无水图保留，全忽略图排除，忽略像素不贡献损失或梯度。`p` 是连续 sigmoid 概率，不是阈值化预测。该历史实验对四个输出头等权求和，复跑需把 `HEAD_LOSS_WEIGHTS` 设回全1；验证仅最终头。BCE/Dice 用 float32 累计，支持原 AMP 流程。

运行 `python src/train.py` 执行当前选定方案。每次仍新建时间戳目录；B2 的纯 BCE 路径保留，新策略加后缀，B3/B4/B5 同理。显式填写 `OUTPUT_DIR` 时以该路径为准。checkpoint 保存 `loss_name`、`loss_kwargs` 和计算方式，并继续保存 `best.pth`、`best_f1.pth`、`last.pth`。新实验从相同预训练骨干初始化，不从旧 BCE 模型续训；既有检查点和评价/后处理加载方式不变。

BCE+Dice 与旧 BCE 应在同一验证名单下比较 `best_f1.pth` 的原始 F1、IoU、P/R、FP/FN，初始评价阈值均固定 0.5；之后用相同后处理规则再次比较。不同损失策略的 loss 数值不在同一尺度，不能用 val loss 大小判断谁更好；多头训练 loss 与最终头验证 loss 也不能直接比较。阈值调优独立进行，避免混入第二个变量。

## 第一轮训练改善：BCE + 困难样本采样

困难采样基线配置为 `LOSS_NAME="bce"`、`SAMPLING="hard"`、`UAFNET_FINAL_RESIDUAL=False`、`HEAD_LOSS_WEIGHTS=(1,1,1,1)`，输出到 `outputs/pvt_uafnet_hard_sampling/时间戳/`，不从已有 BCE/Dice 模型续训。设置 `SAMPLING="random"` 即回到普通采样；复跑 Dice 时需要同时设 `LOSS_NAME="bce_dice"`、`SAMPLING="random"`、`UAFNET_FINAL_RESIDUAL=False` 并保留全1头权重。当前默认实验见本页开头。

- `HARD_WARMUP_EPOCHS=2`：前两轮沿用原随机 DataLoader，完整遍历训练子集，建立每幅图的难度记录。
- `HARD_POOL_RATIO=0.25`：按最后观测到的逐图有效像素 BCE，取有分数训练样本中最高的 25% 组成困难池。评分取最终头、参数更新前的训练输出，包含原随机增强与训练模式，并非额外的固定模型全量评价。
- `HARD_SAMPLE_RATIO=0.5`：第3轮起，每轮约半数位置从整个训练子集随机无放回抽取，剩余位置从困难池随机有放回抽取，再混合打乱。普通部分也可能抽到困难样本，所以困难池占最终抽样的比例通常超过一半。
- 每轮仍抽取1305次、batch size为2时仍为653批；不会额外增加训练轮数或每轮更新机会。困难样本重复出现，一轮不再保证覆盖全部不同影像。

评分只读取训练标签；漏检和误检都会贡献 BCE，不按验证 bad case 名单、土地覆盖或 DEM 选择样本。同一图本轮重复观测的评分取均值，未抽到的样本保留上次分数及时间；无有效标签的样本不生成难度分数。这是在线估计，可能偏向水体比例高或标注有疑问的样本，不保证提升精度。

每次训练额外写 `sampling.csv`：每轮每幅训练图的抽取次数、是否属于本轮困难池、有效评分次数、最后平均 BCE、最后观测轮次；无 JSON 或额外分析目录。checkpoint 保存 `sampling_config`、难度分数、最后观测轮次、当前索引计划和困难池。索引均相对于 `train_set`，验证集始终完整顺序评价，归一化仍只统计原训练子集且不按重复抽样加权。训练集指标反映实际抽样分布，不可当作全训练集指标与旧实验直接比较。

实验完成后，用相同验证名单和阈值0.5比较原始 `best_f1.pth`，并做相同 Precision 要求下的 Recall 对照；先不改变后处理、损失权重或网络结构。默认B2与原BCE实验对照，B3/B4/B5不能和采样改动一起作为单变量实验。

## 最终头残差结构对照

复跑本节原残差基线时，显式设 `SAMPLING="hard"`、`HEAD_LOSS_WEIGHTS=(1,1,1,1)`。当前非等权监督实验保留这套残差结构与全局困难采样，输出到独立的 `_head_weights` 目录。

训练极化在 `src/train.py` 顶部统一选择，所有方法共用：

```python
POLARIZATIONS = ("VV", "VH")  # 当前双极化；VV 单极化改为 ("VV",)，注意逗号。
POLARIZATION_BANDS = {"VV": 1, "VH": 2}  # 按实际输入文件的波段顺序填写。
```

DFC 的 VV 单极化对照只修改 `POLARIZATIONS=("VV",)`，仍读取原六波段文件的第 1 波段；VH 对照可用 `("VH",)`。本地只有一个 VV 波段的已处理影像，可设置 `POLARIZATIONS=("VV",)`、`POLARIZATION_BANDS={"VV": 1}`，并修改影像和标签目录。其他极化组合按真实文件补充映射；不能复制 VV 当作 VH。本配置仅负责波段选择，不负责 SLC 定标、地理编码或 dB 转换，极化身份仍需依据数据说明确认。

网络输入通道数随选择自动变化，训练集均值和标准差按所选通道重新计算，检查点记录极化名称、文件波段和模型通道数。默认 VV/VH 保留原输出目录，其他组合增加极化后缀，例如当前加权监督的 VV 单极化目录为 `outputs/pvt_uafnet_hard_sampling_residual_head_weights_vv/时间戳/`。评价和后处理沿用检查点保存的波段与归一化参数；其输入文件的通道布局及数值表达必须与该检查点兼容。本设置不转换旧双通道检查点，也不自动微调本地新数据。单极化对照先保持损失、采样、残差开关及其他参数不变，重新运行训练。

原残差结构对照在 `train.py` 设置 `UAFNET_FINAL_RESIDUAL=True`、`LOSS_NAME="bce"`、`SAMPLING="hard"`、`HEAD_LOSS_WEIGHTS=(1,1,1,1)` 及既有采样比例、预热轮数和训练设置，运行 `python src/train.py`。结果单独保存到 `outputs/pvt_uafnet_hard_sampling_residual/时间戳/`；设置该开关为 `False` 可恢复上一轮困难采样基线，普通采样和其他损失对照也应显式关闭该开关。

结构只改变最终预测：在 1/4 分辨率计算 `final_logits = coarse_logits + correction_logits`，然后按原方式上采样。`coarse_logits` 为 FPN 解码器的高分辨率粗预测，不是16×16引导图；中间两级 UAFM 不变。修正量复用最后 UAFM 的预测卷积，将该卷积的权重和偏置初始化为零，使新训练开始时最终头等于粗预测。最终损失可以沿残差路径直接更新 FPN/粗解码器；零初始化使修正卷积前的特征层在首步可出现零梯度，修正卷积更新后恢复学习。仍使用四输出等权监督，最后一项监督的是相加后的最终预测，不额外监督修正量，也不强制水体概率只增不减。

这是本项目的残差结构实验，不是论文原结构。新实验从原预训练骨干重新训练，不续训旧最佳模型。checkpoint 的 `model_kwargs` 保存 `final_residual`；未包含该字段的历史检查点默认关闭，严格加载后的行为不变。评价与后处理仍取最后一个输出。完成后，与上一轮困难采样基线及本轮粗预测头一起比较固定阈值和相同 Precision 要求下的 Recall；零初始化和合成测试通过不代表已取得真实精度改善。

## 按水体占比分层的困难采样

上一轮全分层实验配置是 `METHOD="uafnet"`、`LOSS_NAME="bce"`、`SAMPLING="stratified_hard"`、`UAFNET_FINAL_RESIDUAL=True`，仍使用VV/VH双通道和B2骨干。运行 `python src/train.py`，输出到 `outputs/pvt_uafnet_hard_sampling_residual_stratified/时间戳/`。从同一预训练骨干重新训练，不续训旧模型；旧结果不覆盖。

在原归一化读取循环中，仅根据训练标签的有效像素计算每图水体比例，分为无水体、(0,1%]、(1%,10%]、(10%,50%]、(50%,100%]五组。前2轮仍完整随机遍历。第3轮起，困难池总规模仍为有有效BCE分数样本数的25%，按各组有分数样本人数比例分配名额；先向下取整，剩余名额用最大余数法分配，并列按组号。每组内部仍选最后实际观测BCE最高的样本。全忽略标签不分配虚构水体组，也不进入困难池。

当前1305个训练样本的五组人数是62、598、435、165、45；有有效分数且数据未改动时，326个困难池名额分别为16、149、109、41、11。困难分支仍从池内均匀有放回抽取652次，普通分支仍随机无放回抽取653次，最后打乱；每轮总数1305、batch size 2、20轮、学习率1e-4、BCE、多头监督、增强、余弦调度及原验证集均不变。这里只约束池构成，不强制每组最终抽样次数完全等于配额比例。

`sampling.csv` 新增 `water_ratio` 和 `water_group`，`sampling_groups.csv` 每轮记录各组人数、困难池人数和实际抽样次数；预热轮的困难池人数为0。检查点保存分组规则、水体比例和训练组号，可按 `train_files` 对应复核。验证集不参与分组或难度更新。`SAMPLING="hard"` 可复跑原全局困难池及原目录，`"random"` 恢复普通采样，原检查点加载方式不变。

审查依据和完整实验约定见 [docs/s1_samples/stratified_hard_sampling_experiment.md](docs/s1_samples/stratified_hard_sampling_experiment.md)。该轮已完成：整体F1从83.0543%降至82.1782%，少水体切片Recall从34.4005%升至40.0411%。因此全分层结果不替代原残差模型，下一轮采用下述温和混合方案。

## 全局困难池与少水体补充（历史对照）

复跑设置 `SAMPLING="hybrid_hard"`、`LOW_WATER_POOL_RATIO=0.20`、`HEAD_LOSS_WEIGHTS=(1,1,1,1)`，保留残差和BCE。保留全局BCE最高的261个训练样本，再从未入池且水体占比为(0,1%]的样本中补充65个；不足时按全局排名补齐，池内不重复。困难池仍326个，每轮仍为653次普通抽样加652次困难抽样。其余模型、损失、划分和训练参数保持原配置。

按上述配置运行 `python src/train.py`，结果进入 `outputs/pvt_uafnet_hard_sampling_residual_hybrid/时间戳/`；旧实验不覆盖。仍从原预训练骨干重新训练，不加载上一轮最佳模型续训。日志继续保存采样组别及次数，检查点记录混合比例和规则。该方案保留用于历史对照，当前默认已回到全局困难采样并测试头权重。

具体方案、回退方式和验收标准见 [混合采样实验](docs/s1_samples/hybrid_hard_sampling_experiment.md)。[缓冲区与固定点辅助评价](docs/s1_samples/buffer_point_quality_metrics.md)目前是评价方案说明，尚未接入计算代码，也不参与损失或选模。

## 数据目录与波段组成

本地目录文件数量已清点；抽查训练、官方验证和官方测试文件，影像均为六波段 `int16`、512×512，标签为单波段 `uint8` PNG、512×512。

| 本地目录或子集 | 数量 | 内容 | 当前用途 |
|---|---:|---|---|
| `data/Track1/train/images` | 1631 | SAR 与四种辅助数据组成的六波段 TIFF | 有标签数据的影像 |
| `data/Track1/train/labels` | 1631 | 对应单波段 PNG，0 非水体、1 水体 | 监督训练及内部验证 |
| 检查点记录的内部训练／验证划分 | 1305／326 | 从上述 1631 幅中划分，没有复制新文件夹 | 当前训练、F1 和 bad case 分析 |
| `data/Track1/val/images` | 349 | 官方 validation 六波段 TIFF，本地没有对应标签 | 预测及定性检查，不能直接算本地 F1 |
| `data/images` | 351 | 用户确认来自 Track1 test 的六波段 TIFF，本地没有对应标签 | 官方 test 预测，不能直接算本地 F1 |

根据 [2026 比赛总结图 2](docs/papers/2026_DFC2024_RapidFloodMapping_JSTARS_Published.pdf) 与[官方比赛介绍](https://xplorestaging.ieee.org/document/10582794)的 Track 1 数据层布局，对应关系如下。当地文件没有波段名称，按原始比赛布局映射，不应推广到重排后的任意 TIFF：

| 波段 | 官方布局含义 | 作用 | 当前使用 |
|---|---|---|---|
| 1 | Sentinel-1 VV，同极化后向散射 | 水面及其他地物的 SAR 散射信息 | 与波段 2 一起输入网络 |
| 2 | Sentinel-1 VH，交叉极化后向散射 | 提供另一极化的散射响应 | 与波段 1 一起输入网络 |
| 3 | MERIT DEM | 地形高程辅助信息 | 未输入网络，未作高程硬阈值修正 |
| 4 | Copernicus DEM | 另一来源的高程／表面模型 | 未输入网络，未作高程硬阈值修正 |
| 5 | ESA WorldCover 土地覆盖 | 类别图，80 为永久水体 | 可选后处理补充水体 |
| 6 | Global Surface Water 水体出现频率 | 历史水体出现情况，不是本次洪水标签 | 仅作可视化对照 |

这些“六波段”并非六个 SAR 极化通道。真实标签另存于 `labels` 文件，不是第 5 或第 6 波段。当前模型用 `(1,2)`，不是将六层全部输入。

本地文件没有 CRS、有效地理变换或物理单位说明，scale/offset 为 1/0；不能据此恢复 SAR 量化缩放、确定线性功率/dB，也不能把 `int16` 数字直接称为标准 σ⁰。第 6 波段曾观测到 101 等特殊编码，不自动视为 101% 或直接除以 100。辅助层的原始分辨率与重采样后的共同像素网格不是同一个概念。

## 查看后处理结果质量

在 `train.py` 选择相应 `RUN_MODE` 后统一运行 `python src/train.py`，在预测和后处理过程中同时输出精度表、掩膜与对照图，不再使用独立的 bad case 分析脚本，也不自动排名、选择问题样本或判断错误场景。历史 `codex_analysis/bad_cases/` 成果保留。

本次结果均保存在模型目录下的 `quality/时间戳/`：

| 输出 | 内容 |
|---|---|
| `raw/` | 阈值化后的原始预测 TIFF；多成员时对应概率集成结果 |
| `postprocessed/` | 土地覆盖修正后的 TIFF |
| `figures/` | 按原文件名保存的质量对照 PNG |
| `metrics.csv` | 有标签时，各成员、集成、后处理的全数据累计 F1、IoU、P/R 和混淆矩阵 |
| `per_image.csv` | 按影像文件顺序列出修正前后精度和水体占比，方便手工对应图片 |

`save_masks=True` 保存掩膜；`SAVE_VISUALS=True` 保存图片；`MAX_VISUALIZATIONS=None` 默认绘制全部影像，试跑时可改为 10，仅绘制前 10 幅。即使限制图片数，指标仍统计完整数据集。CSV 由后处理自动输出，不写 JSON、summary、排名或人工复查模板。

有标签时，对照图包括 SAR 波段 1/2、真值、概率、原始预测、错误分布、修正后预测及新增水体。TP 蓝、FP 红、FN 黄、忽略像素黑；新增 TP 绿、新增 FP 品红。标题显示文件名和原始／修正后 F1。SAR 显示使用 P2/P98 拉伸，不改变输入，不自动转 dB；概率范围固定为 0～1。

PNG 扩展为四行：SAR 与真值/概率；MERIT DEM、Copernicus DEM 及各自的地形梯度；WorldCover、历史水体出现频率、原始预测/错误；修正预测、新增水体及辅助解释。辅助波段按文件自身掩膜显示，不因为 SAR/标签无效而丢弃其独立信息。DEM 的 `-9999` 仅在辅助显示中作为缺测，保留零高程；整幅缺测会明确显示不可用。土地覆盖用离散类别色标；出现频率保留原始编码，101 等特殊值不解释成百分比。

当前文件缺少可靠的 XY/Z 单位，`DEM_PIXEL_SIZE_METERS=None`、`DEM_HEIGHT_SCALE=None` 时，坡度面板显示 `Elevation gradient (raw units/pixel)`，即原始高程值对像素坐标的梯度幅值，**不是角度坡度或阴影/叠掩掩膜**。确认共同切片网格的行列间距（米）和高程值到米的比例后填写这两个变量，才计算 `atan(sqrt((dz/dx)^2+(dz/dy)^2))` 并转为度；不能用辅助 DEM 原始分辨率直接代替重采样后的切片间距。缺测邻域不参与求导，切片边缘为单边差分。辅助面板不改变模型输入、预测或 F1。

先用错误图定位 FP/FN，再比较其地形、土地覆盖和历史水体特征。图像能提示改进方向，不能估算加入辅助数据后的保证收益；实际提升需固定训练/验证划分，以坡度规则或辅助输入消融实验比较 F1、FP、FN，避免根据验证标签直接修补预测。

无标签数据设为 `WITH_LABELS=False`、`USE_SAVED_VAL=False` 并指定影像目录。仍生成掩膜、概率和预测对照图，真值和错误面板明确标注不可用；`per_image.csv` 只记录预测占比，不输出 F1、FP、FN，也不生成精度汇总表。场景类型及错误成因由人工查看图像判断。

## 独立评价 F1

`src/evaluation/model_eval.py` 现支持两种模式：顶部 `MODE="scan"`（当前默认）扫描残差基线、头监督权重、尺度 MBDC 三组主线；`MODE="single"` 保留原单模型评价。运行入口均为 `python src/evaluation/model_eval.py`。

扫描使用 `SCAN_CHECKPOINTS` 指定的两个 `best_f1.pth`，固定原内部验证集、最终输出头及各自归一化，检查验证名单、数据目录、波段和忽略值一致；不加入土地覆盖后处理、不修改模型。两模型各推理一遍，按 logit 分箱累计所有阈值的精确混淆矩阵，不产生阈值数乘像素数的巨大张量。阈值为 0.01～0.99、步长 0.001；阈值相等时计为水体。

结果保存到根目录 `codex_analysis/threshold_scan/时间戳/`，不混入训练 `outputs`：`threshold_metrics.csv` 包含检查点路径、训练轮次及所有阈值的 P/R/F1/IoU/混淆矩阵；`precision_comparison.csv` 在 P≥80%/85%/90% 的候选中分别选 Recall 最高的实测点，记录实际 Precision 与对应阈值，不可达时明确标记；`pr_curve.png` 展示原始 PR 点的曲线，不做插值或精确率包络。候选阈值有限，比较的是共同精确率下限，不能声称精确率严格相等。无预测像素不作为满足 Precision 要求的点。

扫描完成后再选择部署阈值；不要同时改变损失权重或后处理规则。结果属于验证集调参，不是独立测试成绩。

单模型模式设置 `MODE="single"`，并用 `CHECKPOINT_PATH` 选择检查点，默认沿用其原内部验证集。脚本直接在顶部定义变量，不使用 argparse；单模型模式仅在终端打印评价结果。

- `CHECKPOINT_PATH`：要评价的模型，可改为 `best.pth`、`last.pth` 或旧双类别模型。
- `IMAGE_DIR`、`LABEL_DIR`：默认 None，沿用检查点目录；数据迁移或独立评价时可指定路径。
- `USE_SAVED_VAL=True`：严格使用检查点的 `val_files`，不重新随机划分。评价另外一套有标签数据时设为 False，并设置对应影像、标签目录。
- `BATCH_SIZE`、`USE_AMP`：默认 None，沿用训练记录；CPU 自动关闭 AMP。改用 float32 或其他批大小时，结果可能有少量浮点差异。
- `THRESHOLD=0.5`：单通道水体概率阈值；双类别模型沿用 argmax。正式比较时保持阈值固定。

脚本通过 `models/checkpoint.py` 恢复模型，沿用训练均值、标准差、有效像素处理，在 eval/inference_mode 下只使用最后一个输出头。整套数据累计 TN/FP/FN/TP 后计算水体 F1、IoU、Precision、Recall，另打印 loss、有效像素数和水体占比，不平均每张影像的 F1。

20261021_102433_387373 的第 19 轮 `best_f1.pth` 已重新预测完整 326 幅内部验证影像：F1=0.790527、IoU=0.653613、Precision=0.846358、Recall=0.741606，混淆矩阵与训练日志完全一致。这是内部验证结果，不是官方测试集成绩；官方无标签 test 影像不能用此脚本计算 F1。已知答案检查可运行 `python -m unittest discover -s tests -p test_evaluate.py -v`。

## PVT 第一阶段与独立 ResNet-50 对照

运行 PVT 基线时先在 `train.py` 中设置 `METHOD="pvt"`：

```powershell
conda activate sar-water
python src/train.py
```

ResNet-50 在同一个入口设置 `METHOD="resnet50"` 后运行。两者共用同一个解码头，分别保存到 `outputs/pvt_baseline/时间戳/` 与 `outputs/resnet50_baseline/时间戳/`。默认加载 ImageNet 预训练骨干，自动适配两个 SAR 输入通道；解码头随机初始化。依赖 `timm==1.0.30` 已加入 `requirements.txt`，本机 sar-water 环境已安装，两种预训练权重已验证可加载。

这一步是“骨干 + 1×1 通道投影 + 语义解码头”的阶段实验，尚未包含 MBDC、FPN neck、PSM、CGM 和 UAFM。它不是完整 UANet，也不是冠军比赛方案的完整复现。两种骨干的尺寸、原代码对应关系、预训练差异、后续模块位置和校准方法见 [PVT 阶段说明](docs/stages/pvt_stage1.md)。

两个新模型默认 batch size 2、20 轮、Adam、学习率 0.001、AMP，保持现有数据划分和评价方式。该设置是当前对照起点，不是已验证的最优训练配置。正式训练与精度验证仍需执行。

离线接口检查可运行 `python -m unittest discover -s tests -p test_backbone_models.py -v`，检查尺寸、梯度、保存加载及校准参数边界；它不代表真实数据训练精度。

## UANet 后续阶段与新 S1 应用

后续分阶段网络已接入，统一入口为 `python src/train.py`，设置 `METHOD="uanet"`，当前默认 `STAGE="mbdc"`；随后可依次改为 `fpn`、`uafm`、`full`，各阶段分别保存实验。`full` 包含公开 PVT-UANet 活动路径的 MBDC、FPN、PSM、CGM 和三级 UAFM，训练对四个输出等权求 CE 之和，验证使用最终预测。单输出基线的损失逻辑保持一致。

结构、论文/代码差异与阶段设置见 [UANet 分阶段说明](docs/stages/uanet_stages.md)。逐函数讲解、整体流程图、512×512 输入的尺寸推导，以及 PSM/CGM、不确定性等级与 UAFM 融合计算见 [PVT-UANet 函数含义与计算推导](docs/pvt_uanet_functions_and_derivations.md)，写法与已有 U-Net 说明一致。

新 Sentinel-1 SLC 经 WSL/GAMMA 处理后的 binary/GeoTIFF 输入、数值单位对齐、整景推理与水体矢量输出方案见 [GAMMA 新数据应用方案](docs/stages/sentinel1_gamma_inference_plan.md)。本轮已实现网络与接口，完整场景推理/矢量导出脚本仍待按确认的输入契约实施。

论文集中保存在 [docs/papers](docs/papers/README.md)，现有 9 份 PDF，按年份、模型、任务、来源和版本统一命名。DAD 早期预印本与 TCSVT 正式版均已保存；正式版已核验为 2026 年 36(2)、2372–2386，共 15 页，题名、作者和 DOI 一致，页面连续且可正常渲染。

## 数据检查

当前先检查数据，输入目录直接定义在 `check_data.py` 顶部的 `DATASETS` 中。每一项为集合名称、影像目录、标签目录；没有提供标签时填写 `None`。

在项目目录运行：

```powershell
conda activate sar-water
python scripts/check_data.py
```

脚本只有两个函数：`check_data` 顺序完成文件清点、逐文件读取、实际信息统计和标签配对；`main` 依次调用。读取失败通过 `try/except` 打印文件名和异常，继续检查后续文件。

所有结果直接显示在终端，不写 JSON、CSV 或其他报告文件。显示内容包括：

- 实际文件数量、读取成功和失败数量。
- 实际尺寸、波段数、数据类型、波段描述、单位、NoData、scale/offset、CRS 和仿射变换。
- 每个影像波段的原始数值范围、掩膜无效数和 NaN/Inf 数量。
- 每个标签波段的实际值计数。
- 同名重复文件、缺失标签、孤立标签和配对尺寸不一致。

递归读取 `.tif/.tiff/.png/.img/.vrt`，每次把一个文件及其掩膜读入内存，适用于当前切片数据。所有波段编号从 1 开始；范围统计排除非有限数值和文件掩膜无效像素，不应用 scale/offset，不自行判断零值或其他数值是否为填充值。掩膜无效数与 NaN/Inf 数量可能重叠，不能直接相加。标签计数保留所有有限原始值，包括有限的 NoData 编码。

不预设应有多少文件、多少波段或哪些标签值，也不推断 VV/VH、线性/dB 或水体类别。检查结果说明本地文件是否可读及是否配对；没有官方文件清单时，不能据此证明下载包完整；同尺寸不等于地理配准已验证。

第三方依赖为 NumPy 和 Rasterio，列在 `requirements.txt` 中。旧版 `outputs` 下的报告是历史文件，本脚本不会生成或更新它们。

## 数据读取

`src/datasets/water_dataset.py` 使用 Rasterio 和 PyTorch 的 `Dataset`、`DataLoader`，按需读取栅格，不把整个数据集载入内存。直接修改文件顶部的目录、`BANDS` 和 `BATCH_SIZE` 后运行：

```powershell
conda activate sar-water
python src/datasets/water_dataset.py
```

运行示例仅读取一个批次，在终端显示样本数量、形状和值域，不启动训练。

`WaterDataset(image_dir, label_dir=None, bands=None)` 返回字典：`image` 为 float32 `[C,H,W]`，`valid_mask` 为 bool `[H,W]`，`name` 为相对影像文件名。有标签时另返回 int64 `[H,W]` 的 `label`；标签根据相对路径和同名文件配对，允许影像与标签的扩展名不同。无标签集合将 `label_dir` 设为 `None`。

`bands=None` 保留全部波段，`bands=(1, 2)` 选择文件第 1、2 波段；选择通道不代表已确认其物理含义。当前不进行数值缩放、dB 转换、归一化或数据增强，保留原始标签编码；这些处理待波段及标签定义确认后加入。

影像各选中波段均有限且文件掩膜有效时，`valid_mask` 才为 True。无效影像位置填 0，对应标签置为内部忽略值 `-1`；标签自身掩膜无效或非有限的位置也置为 `-1`。后续交叉熵需设置 `ignore_index=-1`。未在文件中声明为 NoData 的数值（例如 `-9999`）不会被自行排除。读取失败或缺失标签直接显示原始异常，避免训练时静默跳过样本。

依赖列表新增 PyTorch；当前环境已有安装，无需重新安装。

## U-Net 基线

`src/models/unet.py` 已恢复四级 U-Net 的 BatchNorm 基线：双卷积、BatchNorm、ReLU、最大池化、转置卷积和跳跃连接，起始宽度为 32，共 18 个 BN。输出为两个类别的 logits，不在模型内部做 softmax。单独查看尺寸时，在 `src` 目录运行 `python -m models.unet`；训练命令不变。

训练 U-Net 时在 `src/train.py` 顶部设置 `METHOD="unet"`，运行方式为：

```powershell
conda activate sar-water
python src/train.py
```

初始设置：文件第 1、2 通道，512×512 原尺寸，batch size 2，Adam，学习率 0.001，20 轮；CUDA 可用时开启混合精度。只根据所选影像通道的有效像素计算训练部分的逐通道均值和标准差，训练、验证均用同一组统计量。无效影像位置在归一化后重新填 0，标签 -1 不参与损失和指标。

输入通道按文件编号选择，仍需结合原始数据说明确认极化与单位；不做 dB 转换，不额外把未声明的数值当作 NoData。官方说明中的输出编码为非水体 0、水体 1，因此采用交叉熵二分类。

从有标签的训练目录按固定种子 42 划出约 20% 作为内部验证；官方 val/test 无标签，不用于本地指标。当前随机划分只用于基线调试，不代表按事件独立测试，尚不能排除相邻/重叠切片造成的乐观偏差。后续应根据事件或场景信息分组，再进行正式模型比较。

终端显示每轮训练和验证的有效像素加权损失、水体 IoU、F1、Precision、Recall。指标基于该阶段所有有效像素的累计混淆矩阵，不是官方按事件或样本的比赛评分复现。

BN 训练每次在 `outputs/unet_baseline/` 下新建时间戳子目录，历史 GN 实验仍保留在 `outputs/unet_groupnorm/`。终端会显示本次实验路径，避免覆盖上一轮结果。该目录保存：

- `metrics.csv`：每轮 train/val 各一行，包含 loss、IoU、F1、Precision、Recall、混淆矩阵、有效/忽略像素数、真实/预测水体比例、全忽略批次数、学习率、AMP 缩放因子与跳过更新次数、非有限损失批次数和阶段耗时。CSV 使用未格式化的数值，终端显示四位小数。
- `best.pth`：验证损失最低的模型，选择规则保持不变。
- `last.pth`：最近一次完成验证的模型，正常训练结束时对应最后一轮。
- `abnormal.pth`：最近一次异常验证的模型。有真实水体且 F1 低于 `ABNORMAL_F1_THRESHOLD=0.001`，或验证损失非有限时保存；未触发时不会生成。它是复查线索，不是故障原因判定。

模型文件还保存本轮 train/val 指标、输入通道、归一化统计量、训练/验证文件划分、最近 20 个有效训练样本名及运行环境信息，并记录 `normalization="BatchNorm"`。不包含优化器状态，没有实现断点续训。`true_water_ratio` 和 `pred_water_ratio` 都以有效标签像素数为分母；`amp_skipped_steps` 通过缩放因子下降间接统计，缩放因子为零后不能继续可靠计数。

当前回到 BN 基线，学习率、批大小、轮数、划分、输入归一化、损失、优化器、AMP 和最佳模型选择规则均保持原设置，从相同种子重新初始化。仅新增非有限 loss 立即停止并显示文件名的检查，避免异常后继续训练。GN 检查点不能直接按当前 BN 结构加载。Codex 分析产生的图表和解释仍放在 `codex_analysis/`，不混入训练输出。

## 训练后逐层校准 BN

训练结束后运行：

```powershell
python src/calibrate_bn.py
```

脚本顶部直接设置 `EXPERIMENT_DIR`、`RUN_DIR`、`CHECKPOINT_NAME` 和 `CALIBRATION_SAMPLES`。默认仍选择最新 U-Net BN 实验中的 `last.pth`；处理 PVT 或 ResNet-50 时改为相应实验目录，或者直接指定某次 `RUN_DIR`。固定使用该检查点训练划分列表前 128 幅影像；不读取校准影像的标签，不用验证影像估计统计量。PVT 只校准解码头的 BN，骨干 LayerNorm 不参与；ResNet-50 校准骨干与解码头的 BN。

校准始终使用 float32 和 eval，按照当前 U-Net 的前向顺序逐层重估 BN 输入的逐通道均值、方差，写入 `running_mean`、`running_var`。上游已经校准后才估计下游；不把全部 BN 同时设为 train，不计算梯度，不更新卷积权重或 BN 的可学习缩放、偏置。为了过程直观，每层完整遍历一次相同训练子集，因此比同时校准所有层更慢。

校准前后均评价检查点保存的完整内部验证划分，另存到 `outputs/unet_baseline/实验时间戳/bn_calibration/校准时间戳/`：

- `metrics.csv`：before/after 两行，含 loss、IoU、F1、Precision、Recall、混淆矩阵和水体比例。
- `calibrated.pth`：新的运行统计量及原有可学习参数，记录源模型、校准样本和前后指标；不覆盖原始 best/last/abnormal。

校准不是重新训练，也不能修复已含 NaN/Inf 的权重。固定校准方案后比较原始与校准模型，不根据验证成绩反复挑选校准样本；内部验证成绩不等于独立事件测试成绩。

先将 `MAX_SAMPLES = 16`、`EPOCHS = 2`、`OUTPUT_DIR` 改为单独的试跑目录，可短程验证训练流程；正式训练改回 `MAX_SAMPLES = None`。当前基线不加数据增强、类别加权、学习率调度或早停，先确认基础流程，再逐项比较。

数据来源与编码依据：[DFC2024 官方说明](https://www.grss-ieee.org/community/technical-committees/2024-ieee-grss-data-fusion-contest/)。


## 像元精度指标与标签冲突审查（2026-10-27）

`src/evaluation/metrics.py` 是训练、独立评价、阈值扫描及后处理的共同指标入口。输入顺序为 TN、FP、FN、TP；行是真值、列是预测。总体指标先累计全部有效像元的混淆计数再计算，不直接平均逐图指标。原有四指标和模型选择规则不变。

| 字段 | 含义 / 公式 |
|---|---|
| iou / f1 / precision / recall | 水体类别IoU、F1、精确率与召回率，沿用原定义 |
| accuracy | OA=(TP+TN)/N；背景很多时可能掩盖水体错误 |
| background_iou | TN/(TN+FP+FN) |
| mean_iou | 水体IoU与背景IoU的两类平均，不是原来的单类IoU |
| specificity | TN/(TN+FP)，非水体召回率 |
| false_positive_rate | FPR=FP/(FP+TN)，真实背景中的误报比例 |
| false_negative_rate | FNR=FN/(FN+TP)，真实水体中的漏检比例 |
| false_discovery_rate | FDR=FP/(FP+TP)，预测水体中的错误比例，勿与FPR混淆 |
| balanced_accuracy | (Recall+Specificity)/2 |
| kappa / mcc | Cohen Kappa与Matthews相关系数，综合四项混淆计数 |

分母无定义时保留NaN，Figure显示N/A；无预测水体时Precision保持旧约定0。严格两类mIoU和Balanced accuracy在所需类别无定义时也为NaN。Kappa/MCC不以百分数显示。新指标追加到训练和后处理CSV及阈值扫描结果；后处理Figures顶部保留四项水体指标，右下角补充其他指标。无标签不能生成精度。上述指标不等价于岸线距离、边界F1或对象检出率。

19对同影像异标签的只读对比图和说明位于 `codex_analysis/label_conflicts_20261027/`。`overview.png`为总览，`figures/`有19张详细图，`pair_statistics.csv`给出差异面积、标签一致性IoU/Dice及几何归类。两份标签均不自动视为真值，原始数据与训练划分未修改。

## 阈值对照与边界质量评价

运行 `python src/thres_seg/main.py`，直接设置 `IMAGE_DIR`、可选 `LABEL_DIR` 和 `BANDS`。这是独立的传统阈值流程，不加载模型或检查点、不读取训练/验证名单、不自动选参。默认遍历输入目录全部影像，分别输出 VV/VH 阈值及可选 ESA 补水结果；无标签时不计算精度。结果保存到 `outputs/thres_seg/quality/时间戳/`。`evaluation/boundary_eval.py` 同时接入训练后的质量流程，输出 1/3/5 像素内外带、核心区的全部像素计数及固定采样点。新结果统一进入 `quality/`，不自动删除历史成果。详见 [阈值与边界评价说明](docs/threshold_and_boundary_evaluation.md)。

训练只关闭中间层特征图（`SAVE_FEATURE_MAPS=False`）；采样明细、异常快照、最佳模型、`last.pth` 和指标仍按原规则保存。

独立评价新入口：`python src/evaluation/model_eval.py`。旧 `python scripts/evaluate.py` 命令仍可用，但参数只在新文件顶部修改。常规训练和 quality 不依赖该入口；需要概率阈值扫描时才单独执行。

## 固定基线验证

运行 `python src/validation.py`：默认固定历史等权残差 PVT-UAFNet，输出整体、逐图、场景和边界质量，同时分开报告与训练整幅重复的样本。它与 `src/train.py` 同级，不训练、不调阈值、不使用土地覆盖补水；`thres_seg` 保持独立。配置、输出及真正按区域测试的条件见 [固定基线与验证说明](docs/validation_baseline.md)。当前内部验证的 326 张中有 64 张与训练影像完全重复，不能将其成绩称为独立泛化精度。

模型与传统阈值的统一对比入口：`python src/evaluation/compare_threshold.py`。默认使用检查点中有标签的内部 `val_files`，比较原始模型与 VV/VH 的 Lee＋固定百分位分割，统一整体混淆计数、边界带和固定采样点；`thres_seg` 继续独立。后续训练默认已恢复固定 MBDC、困难采样、最终头残差及等权 BCE。置信区间和指标选用原则见上述验证说明。
