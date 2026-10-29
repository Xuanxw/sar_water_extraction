# SAR 水体提取

本仓库用于有监督 SAR 水体语义分割，提供数据读取、模型训练、固定验证集评价、质量可视化，以及独立的传统阈值分割。当前默认模型为 **PVT + DeepLabv3+ 风格解码器**，输入为 DFC Track 1 的 VV/VH 两个波段。

README 只介绍通用用法。算法说明放在 `docs/`，实验分析放在 `codex_analysis/`，训练及预测产物放在 `outputs/`。

## 环境与运行

在 PowerShell 中运行：

```powershell
conda activate sar-water
cd E:\Code_projects\sar_water_extraction
pip install -r E:\Code_projects\sar_water_extraction\requirements.txt
python E:\Code_projects\sar_water_extraction\src\supervised\train.py
```

已有环境无需每次重新安装依赖。训练自动选择 CUDA 或 CPU；`PRETRAINED=True` 使用编码器的 ImageNet 权重，首次使用可能需要下载。读取已有水体检查点时直接恢复保存的权重，不再下载编码器权重。

主要入口都通过顶部变量配置，不使用命令行参数解析。数据目录使用明确的绝对路径；更换仓库位置时修改入口配置即可。代码中的模块导入路径、样本文件拼接和时间戳输出子目录不属于数据路径配置。

## 目录与职责

```text
sar_water_extraction/
├── src/
│   ├── supervised/
│   │   ├── train.py                   # 统一训练入口
│   │   ├── validation.py              # 固定模型验证与质量报告
│   │   ├── datasets/                  # 读取、检查、BC数据准备
│   │   ├── models/                    # 模型、损失、检查点重建
│   │   ├── evaluation/                # 指标、边界、后处理、阈值扫描
│   │   └── experimental/              # MobileNetV2 + U-Net对照
│   └── thres_segmentation/            # 独立传统阈值分割与GUI
├── scripts/
│   ├── compare_threshold.py           # 模型与传统法在同一val上对比
│   ├── evaluate.py                    # model_eval的兼容入口
│   └── ...                            # 场景分类、划分、BN校准等工具
├── tests/                             # 合成数据回归测试
├── data/                              # 原始数据与固定划分清单
├── docs/                              # 算法说明与文献
├── outputs/                           # 训练检查点和质量输出
└── codex_analysis/                    # 实验分析与对照报告
```

## 数据准备

训练入口使用以下绝对路径：

```python
IMAGE_DIR = Path(r"E:\Code_projects\sar_water_extraction\data\Track1\train\images")
LABEL_DIR = Path(r"E:\Code_projects\sar_water_extraction\data\Track1\train\labels")
```

目录结构为：

```text
data/Track1/train/
├── images/train/      # 训练影像
├── images/val/        # 验证影像
├── labels/train/      # 同名训练标签
├── labels/val/        # 同名验证标签
└── split_manifest.csv # name、split、active、group_id等字段
```

有固定划分清单时只读取 `active=1` 的样本，训练和验证使用各自子目录，不再按 `VAL_RATIO` 随机重分。更换清单需要重新检查区域重叠、重复样本及模型是否曾训练过新 val。

当前训练影像按 `POLARIZATION_BANDS={"VV":1,"VH":2}` 读取。模型输入只取所选 SAR 波段；DEM、坡度、土地覆盖等辅助波段可用于质量图或另行声明的修正，不自动进入模型。标签使用水体 `1`、非水体 `0`；文件中的 nodata/无效区域在数据读取后使用 `IGNORE_INDEX=-1` 排除。新增数据中未经标注的区域必须标为无效，不能直接当作非水体。

训练使用存储数值，并仅从训练集计算每个通道的均值、标准差。验证和推理复用检查点中保存的统计量。不会自动进行辐射校正、地理编码、单位换算或转 dB；更换传感器前需另行确认数据表示与训练输入是否兼容。

## 新传感器数据准备

运行独立准备入口，配置均在脚本顶部：

```powershell
python E:\Code_projects\sar_water_extraction\src\datasets\prepare_bc_data.py
```

默认 `MODE="split"`，只统计已采用场景的 train/val 和地理组，不读取影像、不检查地物或几何。`MODE="inspect"` 才打印前三景格式。当前读取 `I:\水体数据\水体数据_train_val划分_已确认.csv`，其中“已确认”表示用户决定纳入，不表示标签已完成验收。当前新数据路径指向 `I:\水体数据`，与原 DFC 数据分开存放。

| 输入 | 支持方式 |
|---|---|
| `.tif` / `.tiff` | 具有坐标系和地理变换的实数 GeoTIFF，两种后缀等价 |
| `.zip` | 包含唯一一个 GeoTIFF；正式准备时逐景展开到临时目录，处理后清理 |
| `.shp` 标签 | 面/多部件面及其配套属性、坐标系文件；排除精度评价点文件 |
| 裸 binary / 复数 SLC | 不直接输入；需明确尺寸、类型、地理参考及物理含义，先生成实数 GeoTIFF |

正式切块前按顺序配置：

1. 核实标签字段含义，填写 `CLASS_FIELD` / `CLASS_MAP`；属性含义不同的景放在 `SCENE_CLASS_RULES`。目标编码为 `0=确认非水体`、`1=确认水体`，不能根据字段数字猜类别。
2. 核实影像有效域。优先使用源 nodata/内部掩膜；否则填写已确认的 `INPUT_NODATA_VALUE` 或提供同网格 `VALID_MASK_DIR`。只有确认整景有效才设 `ALLOW_ALL_VALID_SOURCE=True`，不默认剔除零值。
3. 将划分 CSV 中决定纳入的景设为“是否采用=是”或“已确认”，保持已有 train/val 和地理组不变。`待确认`、`否` 不进入正式准备。
4. 设置 `MODE="prepare"`，使用新的 `OUTPUT_DIR` 后运行。保留原始 ZIP、SHP 和划分 CSV；异常时不覆盖旧结果、不自动修复无效面。

默认输出到 `I:\水体数据\prepared`：

```text
prepared/
├── images/train/、images/val/  # 单VV、float32、512×512 GeoTIFF
├── labels/train/、labels/val/  # 同名uint8 GeoTIFF，0/1/255，nodata=255
├── split_manifest.csv         # 切片、地理组、源路径、窗口、有效/类别像素数
├── scene_split.csv            # 输入景级清单副本
└── preparation_settings.txt   # 实际波段、标签映射及有效域配置
```

标签在影像原生网格上按像素中心栅格化，必要时先转换坐标系。矢量外、未映射类别、正负面冲突、影像无效域及边缘填充均为 `255`，当前 `WaterDataset` 会将其读取为 `IGNORE_INDEX=-1`。全未知切片跳过；若只有水体正样本、没有任何确认的非水体，不能作为完整二分类训练集发布。成功结束才产生正式 `split_manifest.csv`；仅有 `.partial` 时不要启动训练。

准备过程保留存储值及原生像素间距，不自动转 dB、应用 scale/offset 或重采样。1 m 与3 m切片的实际覆盖范围不同；统一格式并不等于完成跨传感器适配。未知区域仍可提供影像上下文，但不参与监督或精度计量；局部标注的成绩也不能代表整景精度。

数据准备成功后，在 `src/supervised/train.py` 设置：

```python
IMAGE_DIR = Path(r"I:\水体数据\prepared\images")
LABEL_DIR = Path(r"I:\水体数据\prepared\labels")
POLARIZATIONS = ("VV",)
POLARIZATION_BANDS = {"VV": 1}
```

同时将 `POSTPROCESS_OPTIONS` 中的 `use_landcover` 设为 `False`，新单波段数据没有 DFC 第5波段土地覆盖。现有双极化水体检查点不能直接作为单VV模型加载；本准备脚本不修改训练配置，也不启动训练。

## 训练配置

在 `src/supervised/train.py` 顶部按功能分组修改：

| 参数组 | 主要变量 | 用途 |
|---|---|---|
| 执行方式 | `METHOD`、`RUN_MODE` | 选择模型及训练/质量输出流程 |
| 数据 | `IMAGE_DIR`、`LABEL_DIR`、`POLARIZATIONS`、`POLARIZATION_BANDS` | 数据路径、极化与文件波段映射 |
| 输出 | `OUTPUT_ROOT`、`OUTPUT_DIR` | 根目录与可选绝对路径覆盖；每次运行创建新时间戳目录 |
| 训练 | `PRETRAINED`、`BATCH_SIZE`、`EPOCHS`、`LEARNING_RATE`、`SEED` | 初始化、批次、轮数、学习率和随机种子 |
| 优化 | `OPTIMIZER_NAME`、`WEIGHT_DECAY`、`COSINE_SCHEDULE`、`TRAIN_AUGMENTATION` | 单通道水体方法的优化与增强设置 |
| 损失 | `LOSS_NAME`、`BCE_WEIGHT`、`DICE_WEIGHT` | 选择损失；后两项仅 BCE+Dice 使用 |
| 采样 | `SAMPLING`、`HARD_WARMUP_EPOCHS`、`HARD_SAMPLE_RATIO`、`HARD_POOL_RATIO` | 随机/困难采样及比例 |
| DeepLab | `DEEPLAB_*` | 编码器、解码宽度、ASPP空洞率、Dropout |
| 其他网络 | `UAFNET_*`、`STAGE`、`HEAD_LOSS_WEIGHTS` 等 | 仅选中对应模型时生效 |
| 质量输出 | `POSTPROCESS_OPTIONS`、`POSTPROCESS_CHECKPOINTS` | 推理工作点、土地覆盖修正、掩膜和可视化 |

默认 `METHOD="deeplabv3"` 使用 PVT-v2-B2 编码器、ASPP 和浅层跳连，单头 BCE，解码通道256、浅层投影48、空洞率 `(3,6,9)`。输入512×512时，深层16×16特征经过ASPP，与128×128浅层特征融合，最后恢复到512×512。它是输出步长32、普通卷积的 DeepLabv3+ 风格适配，不是原论文完整配置。

其他方法可这样选择：

| `METHOD` | 模型 | `LOSS_NAME` | `SAMPLING` |
|---|---|---|---|
| `deeplabv3` | PVT + DeepLabv3+风格解码器 | `bce` / `bce_dice` | BCE可选困难采样；BCE+Dice用`random` |
| `uafnet` | UAFNet风格水体网络，可配置嵌套细化等 | `bce` / `bce_dice` | 同上 |
| `uanet` | 按`STAGE`选择模块的双类别网络 | `ce` | `random` |
| `unet` | 普通U-Net | `ce` | `random` |
| `pvt` | PVT + 简单解码头 | `ce` | `random` |
| `resnet50` | ResNet50 + 相同简单解码头 | `ce` | `random` |

切换双类别模型时，同时调整损失和采样变量。`POLARIZATIONS=("VV",)` 可训练单VV模型，但不会从缺失数据中构造VH；单极化模型应重新训练或做明确的迁移实验。

`RUN_MODE` 的三个取值：

- `train_postprocess`：训练完成后重载本次最佳检查点，再输出质量结果。
- `train`：只训练。
- `postprocess`：不训练，使用 `POSTPROCESS_CHECKPOINTS` 指定的绝对路径。

默认是重新训练，`PRETRAINED=True` 不表示续训旧水体模型。DeepLab已整合到 `src/supervised/train.py`，不再提供旧的独立DeepLab脚本。

## 训练与质量产物

每次运行保存到独立时间戳目录，已有结果不会因重跑而被覆盖。

| 文件或目录 | 含义 |
|---|---|
| `best.pth` | 验证损失最低的检查点 |
| `best_f1.pth` | 验证F1最高的检查点，单通道水体质量输出默认使用它 |
| `last.pth` | 最后一轮检查点 |
| `metrics.csv` | 每轮train/val损失、混淆计数及精度指标 |
| `split_manifest.csv` | 此次运行所用的固定划分副本 |
| 采样相关记录 | 困难池、抽样情况及样本组成，便于复核 |
| `quality/` | 掩膜、逐图精度、质量图和边界评价 |

`.pth` 保存模型权重、模型配置、波段、归一化参数及训练/验证名单等。恢复推理应通过 `models/checkpoint.py`，不要仅凭文件名猜测网络结构。

`SAVE_FEATURE_MAPS=False` 关闭中间层特征图。质量图是最终成果，不属于训练中间特征。`use_landcover=True` 会额外生成土地覆盖修正结果，评价网络本身时使用 **raw** 掩膜及指标，不把修正结果混入原始模型成绩。

## 固定模型验证与概率阈值扫描

固定阈值、输出质量结果和汇总：

```powershell
python E:\Code_projects\sar_water_extraction\src\supervised\validation.py
```

先填写顶部 `CHECKPOINTS`、`IMAGE_DIR`、`LABEL_DIR`。`USE_SAVED_VAL=True` 使用检查点中的val名单；外部有标签数据设为False并明确填写数据目录。数据路径仍可从函数调用传入；省略时公共函数可恢复检查点中的路径，直接运行脚本则使用顶部绝对路径。

快速核验或模型概率阈值扫描：

```powershell
python E:\Code_projects\sar_water_extraction\src\supervised\evaluation\model_eval.py
```

- `MODE="single"`：在固定 `THRESHOLD` 下评价一个检查点。
- `MODE="scan"`：每个模型推理一次，在同一组 `SCAN_THRESHOLDS` 下累计指标，并按 `PRECISION_TARGETS` 比较达到指定Precision要求时的最大Recall。它不运行传统阈值分割。

整体指标由全部有效像素的TN/FP/FN/TP汇总计算；逐图与分组结果用于查看困难场景。`val`用于开发和选参，不能当作独立测试集。按真实独立区域计算Bootstrap置信区间需要可靠的区域映射；不能把相关切片当成独立区域。IoU置信区间表达统计估计的不确定性，不是“是否达到95%精度”的门槛。

## 独立传统阈值分割

```powershell
python E:\Code_projects\sar_water_extraction\src\thres_segmentation\main.py
```

这是独立入口，不加载模型检查点。`MODE="dataset"` 默认处理固定val影像；`MODE="gui"` 启动原地理编码SAR图形界面。数据模式采用 Lee近似去斑 + 逐图百分位阈值，分别输出VV、VH结果，**不是Otsu**。

| `THRESHOLD_OPTIONS` 项 | 含义 |
|---|---|
| `water_percent=10` | 逐图选第10百分位作为暗值阈值；不是模型概率0.1，也不是读取标签得到水体比例 |
| `mode="bottom"` | 取低于阈值的暗像素；`top`取高于或等于第100-q百分位的亮像素 |
| `lee_window=5` | Lee近似滤波的5×5窗口；1关闭滤波 |
| `noise_ratio=0.12` | 以全图方差乘此比例估计噪声方差，参与局部平滑权重 |
| `stretch=(1,99)` | 去斑后按1/99百分位缩放至0～1并截断极端值；这是强度分数，不是水体概率 |
| `power_to_db=False` | 不转换存储域；仅确认线性功率输入时才考虑10log10转换 |

`water_percent` 是固定方案参数；同值像素和无效域会使实际水体比例不严格等于该数值。`USE_LANDCOVER` 控制是否额外使用辅助土地覆盖补水。原始阈值掩膜位于 `masks/threshold_VV`、`masks/threshold_VH`，比较原始算法时使用这些目录。

## 模型与传统阈值结果对比

```powershell
python E:\Code_projects\sar_water_extraction\scripts\compare_threshold.py
```

该脚本只在同一组val、共同有效像素上比较整体与边界精度，不使用train评价。支持“现场计算”或“读取已有掩膜”两种输入方式，也可两者混用。

| 前置参数 | 含义 |
|---|---|
| `CHECKPOINT` | 模型配置、val名单和训练划分来源；未提供模型掩膜时还用于推理 |
| `SPLIT_MANIFEST` | 当前有效划分，与检查点的清单哈希及train/val名单核对 |
| `IMAGE_DIR`、`LABEL_DIR` | val影像和人工标签的绝对目录 |
| `USE_SAVED_VAL` | True按检查点val名单；外部有标签集合设False并将`SPLIT_MANIFEST=None` |
| `MODEL_THRESHOLD` | 网络水体概率阈值，默认0.5；不是传统强度分位数 |
| `MODEL_MASK_DIR` | `None`时进行模型推理；填写路径后读取已有raw二值掩膜 |
| `THRESHOLD_MASK_DIRS` | 空字典时重新做传统分割；为VV/VH填写已有掩膜目录后，对应波段不再分割 |
| `THRESHOLD_BANDS` | 传统方法分别读取的SAR波段；同时参与共同有效域计算 |
| `THRESHOLD_OPTIONS` | 仅未提供缓存掩膜的波段使用，含义见上表 |
| `BOUNDARY_RADII` | 真值边界内外缓冲带宽度，单位像素，默认1/3/5 |
| `BOUNDARY_POINTS`、`SEED` | 各局部区域的采样点数和固定随机种子，各方法共用点位 |
| `SCENE_CSV` | 可选的场景分类表，用于分组报告 |
| `REGION_CSV`、`REGIONS_ARE_INDEPENDENT` | 外部有标签测试的真实区域分组与独立性声明 |
| `BATCH_SIZE`、`USE_AMP` | 读取/推理批量及CUDA混合精度；缓存模式不进行模型AMP推理 |
| `SAVE_MASKS`、`MAX_VISUALIZATIONS` | 是否保存掩膜及可视化数量；不改变参与精度统计的样本数 |
| `OUTPUT_DIR` | 新对比报告目录；None自动创建时间戳目录，不是输入结果目录 |

复用已有结果时，填写实际生成目录，例如：

```python
MODEL_MASK_DIR = Path(r"E:\Code_projects\sar_water_extraction\outputs\某模型\某次运行\quality\某次质量输出\raw")
THRESHOLD_MASK_DIRS = {
    "VV": Path(r"E:\Code_projects\sar_water_extraction\outputs\thres_seg\quality\某次运行\masks\threshold_VV"),
    "VH": Path(r"E:\Code_projects\sar_water_extraction\outputs\thres_seg\quality\某次运行\masks\threshold_VH"),
}
```

示例中的“某次运行”需换成真实目录。两侧都配置后不再执行网络推理或传统分割，只读取掩膜重新统计、绘图。仍读取检查点以核对划分；已有掩膜须确实来自所声明的检查点、概率阈值和处理流程，TIFF本身不能证明这些来源。

输入须为与原影像同名、同网格的单波段TIFF：0非水体、1水体、255无效，且覆盖共同影像有效域。PNG预览图或只有F1/IoU的汇总CSV不能用于重新计算局部边界精度。缓存模式不会从二值结果恢复概率或原始强度阈值，报告相应字段留空。

结果包含整体 `metrics.csv`、逐图 `per_image.csv`、实际使用的传统阈值及来源 `thresholds.csv`、处理配置 `protocol.csv`、边界 `boundary/metrics.csv` 及可视化。边界评价分别报告内侧漏检、外侧误检、内外带联合计数，并区分全像素与固定采样点统计；这些是局部分类指标，不是带距离容差的Boundary F1。

## 回归检查

在仓库根目录运行 `python -m unittest discover -s tests`。测试使用临时合成数据；模型权重重载不会重新下载预训练权重。
