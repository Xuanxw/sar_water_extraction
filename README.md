# SAR 水体提取

本仓库用于有监督 SAR 水体语义分割，提供数据读取、模型训练、固定验证集评价、质量可视化，以及独立的传统阈值分割。当前默认模型为 **PVT + DeepLabv3+ 风格解码器**，默认输入为已准备的 BC 整景单 VV 数据；DFC Track 1 的 VV/VH 作为历史可选配置保留。

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
│   │   ├── models/                    # 模型、损失、检查点重建
│   │   ├── evaluation/                # 指标、边界、后处理、阈值扫描
│   │   └── experimental/              # MobileNetV2 + U-Net对照
│   ├── datasets/                     # 读取、检查、BC数据准备与标签修订
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

历史 DFC 数据配置（需同时设 DATA_LAYOUT="patch"、CLASS_WEIGHT_MODE="none"）：

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

DFC 影像按 `POLARIZATION_BANDS={"VV":1,"VH":2}` 读取；BC 默认使用 `{"VV":1}`。模型输入只取所选 SAR 波段；DEM、坡度、土地覆盖等辅助波段可用于质量图或另行声明的修正，不自动进入模型。标签使用水体 `1`、非水体 `0`；文件中的 nodata/无效区域在数据读取后使用 `IGNORE_INDEX=-1` 排除。新增数据中未经标注的区域必须标为无效，不能直接当作非水体。

训练使用存储数值，并仅从训练集计算每个通道的均值、标准差。验证和推理复用检查点中保存的统计量。不会自动进行辐射校正、地理编码、单位换算或转 dB；更换传感器前需另行确认数据表示与训练输入是否兼容。

## 新传感器数据准备

运行独立准备入口，配置均在脚本顶部：

```powershell
python E:\Code_projects\sar_water_extraction\src\datasets\image_process.py
```

默认 `MODE="labels"` 仅重建已有标签；首次导入SAR时显式设置 `MODE="prepare"`，按整景输出，不裁剪、不缩放、不生成512×512小切片。`MODE="split"` 仅显示划分；`MODE="inspect"` 显示少量源文件的格式。输入清单为 `I:\水体数据\水体数据_train_val划分_已确认.csv`，沿用地理组及train/val归属。

支持具有地理参考的实数GeoTIFF（`.tif/.tiff`），以及内含唯一GeoTIFF的ZIP；标签为SHP面及其配套文件。ZIP逐景临时展开，完成后清理缓存，原始文件保留。

顶部主要配置：

| 变量 | 含义 |
|---|---|
| `OUTPUT_DIR` | 统一为 `I:\水体数据\prepared_scenes`；labels模式原位替换标签，prepare模式不会覆盖已完成目录 |
| `BANDS=(1,)` | 当前输出单VV，保留原始存储数值 |
| `READ_WINDOW_SIZE=1024` | 内部读写窗口，用于控制内存；不是训练窗口，不影响整景输出尺寸 |
| `CLASS_FIELD` / `CLASS_MAP` | 当前BC成果使用 `None` / `{None:1}`：所有SHP面为水体，不依据class字段区分水陆 |
| `OUTSIDE_LABEL` | 当前BC面外已确认为非水体，设为0；其他未完整标注的数据必须设为255 |
| `SCENE_CLASS_RULES` | 明确含义后可为特殊场景配置单独的字段和映射 |
| `ALLOW_ALL_VALID_SOURCE=True` | 未声明无效域时按用户要求保留全部有限数值，包括0；仍尊重源文件有效掩膜 |
| `INPUT_NODATA_VALUE` / `VALID_MASK_DIR` | 可选显式无效值或同网格有效范围掩膜 |

```text
prepared_scenes/
├── images/train/、images/val/  # 每景一个float32 GeoTIFF，原始宽高/CRS/网格
├── labels/train/、labels/val/  # 同名同尺寸uint8标签，0/1/255
├── split_manifest.csv         # 每景一行：划分、地理组、源路径、尺寸及像素数
├── scene_split.csv            # 输入划分副本
├── preparation_settings.txt   # 本次实际配置
└── skipped_scenes.csv         # 有跳过场景时记录原因
```

标签按原生网格的像素中心栅格化，必要时转换矢量坐标系。当前BC语义为面内水体1、面外非水体0；源无效像素与正负标签冲突为255。孔洞属于面外，重叠水体面取并集。图例设置为水体黑、非水体白、255为nodata；显示颜色不参与训练。其他数据若标注覆盖不完整，必须把面外设为255。没有任何有效0/1监督像素的整景标为active=0，不用于训练或精度统计。缺少类别字段或没有可映射类别的场景暂跳过，不猜测含义；无效几何不自动修复。

内部窗口逐块写入同一幅整景文件，使用压缩及大TIFF支持；不会一次把整幅影像加载到内存。整景影像和标签写完后才登记清单，支持未完成目录续跑；旧切片清单不能续跑到整景目录。仍有跳过场景或数据不完整时保留`.partial`，成功完成后才发布正式清单。已有源数据和旧切片不删除。

准备过程不转dB、不应用scale/offset、不重采样。统一存储形式不等于完成跨传感器适配；局部标注的指标不能代表整景所有区域的精度。

### 原位重建已经准备好的整景标签

`image_process.py` 的 `MODE="labels"`，执行原位标签重建，输出根目录统一为 `I:\水体数据\prepared_scenes`；不会再写入 `prepared_scenes_new`。只有首次导入原始SAR时才使用 `MODE="prepare"`。

停止读取此数据的训练/评价后，运行：

```powershell
python E:\Code_projects\sar_water_extraction\src\datasets\image_process.py
```

该入口读取 `I:\水体数据\prepared_scenes\split_manifest.csv` 中的原始SHP路径，按照对应SAR网格重新栅格化，直接替换 `prepared_scenes\labels\train` 和 `labels\val` 中的GeoTIFF。水体面内为1、面外为0，无效SAR位置为255（忽略）。色带0白、1黑，显示颜色不能用来判断像素值；不要对旧标签直接做 `1-label`。保留原SAR、原SHP、景号与train/val地理分组，不创建VRT或新版本数据目录。

顶部 `OUTPUT_DIR` 指定原位替换目录，`READ_WINDOW_SIZE` 控制内部读写内存。每景先写临时标签，完整关闭后替换原文件；全部完成后同步 `split_manifest.csv` 像素统计、`label_statistics.json` 和 `revision_settings.json`。中断后重新运行会从第一景重建；`.labels_rebuilding` 存在时训练读取器拒绝使用尚未完成的数据。

标签替换后，先运行image_process.py的preprocess模式，重建对应处理成果、窗口索引和训练统计，无须手动删除缓存；训练读取已完成缓存并计算类别权重。重新开始训练，不续训旧标签下的水体模型。旧BC/V2入口已合并移除。

**整景训练使用 `src/supervised/load/sar_scene_dataset.py`**，整景模式不使用S1读取器。`src/supervised/train.py` 默认读取 `I:\水体数据\processed_scenes\normalized\images` 和 `labels`，使用单VV PVT＋DeepLabv3+风格模型、BCE、batch size 2。旧DFC实验只有显式选择`DATA_LAYOUT="patch"`并恢复数据路径/波段配置才使用旧读取器。

整景读取参数位于`SCENE_OPTIONS`：

| 参数 | 默认值与作用 |
|---|---|
| `window_size` / `halo` | 1024 / 128；中心768×768负责监督和计分，周围上下文只作为输入 |
| `windows_per_epoch` | 4096；每轮训练抽样上限，`None`遍历全部训练窗口；当前使用随机窗口抽样，关闭困难重采样 |
| `normalization_windows` | 每个训练景固定抽2个有标注核心的窗口，使用其中全部有效SAR像素估计均值/标准差；不使用val，不按水体类别筛选 |
| `num_workers` / `max_open_scenes` | 2 / 4；每worker独立有限文件缓存、2批预取、64MiB GDAL缓存，GPU使用锁页内存 |
| `cache_dir` | `E:\Code_projects\sar_water_extraction\data_cache\bc_scenes`，保存小型窗口索引及归一化缓存，不保存小影像切片 |

首次索引用连续标签条带扫描，不加载整景数组；源文件尺寸/修改时间、VRT引用的SAR、采用清单或窗口配置变化会改变缓存键。后续复用索引。离线归一化使用固定训练窗口的**抽样估计**；原参数和抽样索引保存在preprocessing.json和检查点中，预测新数据须沿用相同协议。

训练每轮是窗口抽样轮次，不等于遍历全部原始场景。验证遍历全部有0/1标签的核心窗口，不受训练抽样上限限制；核心互不重叠，每个已标注像素仅统计一次。输出常规`metrics.csv`、最佳/最后检查点及每轮`val_scenes_epoch_*.csv`。后者用于检查各景结果，不能将未知区域算成非水体。

新实验在 `RESUME_CHECKPOINT=None` 时使用 `PRETRAINED` 决定是否加载ImageNet编码器权重；显式指定续跑检查点时恢复已有水体模型，不再下载或加载ImageNet初始化权重。`CLASS_WEIGHT_MODE="train_pixels"` 仅使用train不重叠核心的已标注像素，设置BCE类别权重 `w_c=N/(2*N_c)`，顺序为非水体、水体；未知255不计数。它让两类在全训练像素总体上的权重总量相同，并不保证每批类别平衡或效果提升。为避免叠加重加权，本轮使用 `SAMPLING="random"`；改回旧实验时设类别权重模式为`none`。

训练loss使用类别加权，val loss保持未加权，两者不能直接按数值大小判断过拟合。精度指标不加权，概率阈值仍为0.5；加权可能改变概率校准，后续阈值选择仅用val。DeepLab只有一个监督头，头权重为1；UAFNet的四头权重不影响它。

除原有`best.pth`（val loss最低）、`best_f1.pth`和`last.pth`，另保存`best_balanced_accuracy.pth`，按水体与非水体召回率的平均值选取。全水体/全非水体预测在两类均存在时该值为0.5；仍须结合两类mIoU、误差图、各景表现判断质量，文件名中的best不表示模型已经有效。

当前训练内验证采用带上下文的中心裁取拼接计量，不是重叠概率平均。训练验证本身不保存全幅预测。整景概率/掩膜/质量图由train.py的postprocess模式分派到scene_quality生成，详见下方“BC整景质量输出”。当前不自动导出矢量。

## 训练配置

在 `src/supervised/train.py` 顶部按功能分组修改：

| 参数组 | 主要变量 | 用途 |
|---|---|---|
| 执行方式 | `METHOD`、`RUN_MODE` | 选择模型及训练/质量输出流程 |
| 数据 | `IMAGE_DIR`、`LABEL_DIR`、`POLARIZATIONS`、`POLARIZATION_BANDS` | 数据路径、极化与文件波段映射 |
| 斑点滤波 | `image_process.py`中的`FILTER_INPUT_STAGE`、`FILTER_WINDOW`、`ENL` | 正式训练SPECKLE_FILTER固定none，实际滤波在离线模式完成 |
| 输出 | `OUTPUT_ROOT`、`OUTPUT_DIR`、`RESULT_FILES` | 最终实验目录及模型/指标/日志名称；运行只加时间戳，续跑沿用原目录 |
| 缓存与质量路径 | `WINDOW_CACHE_DIR`、`SNAP_GPT_PATH`、`SNAP_CACHE_DIR`、`QUALITY_CHECKPOINT`、`QUALITY_OUTPUT_DIR`、`AUXILIARY_CACHE_DIR` | 所有可配置位置集中在顶部 |
| 训练 | `RESUME_CHECKPOINT`、`PRETRAINED`、`BATCH_SIZE`、`EPOCHS`、`LEARNING_RATE`、`SEED` | 续跑/初始化、批次、总目标轮数、学习率和随机种子 |
| 优化 | `OPTIMIZER_NAME`、`WEIGHT_DECAY`、`COSINE_SCHEDULE`、`TRAIN_AUGMENTATION` | 单通道水体方法的优化与增强设置 |
| 损失 | `LOSS_NAME`、`CLASS_WEIGHT_MODE`、`BCE_WEIGHT`、`DICE_WEIGHT` | 选择损失及BCE类别权重；后两项仅 BCE+Dice 使用 |
| 采样 | `SAMPLING`、`HARD_WARMUP_EPOCHS`、`HARD_SAMPLE_RATIO`、`HARD_POOL_RATIO` | 随机/困难采样及比例 |
| DeepLab | `DEEPLAB_*` | 编码器、解码宽度、ASPP空洞率、Dropout |
| 其他网络 | `UAFNET_*`、`STAGE`、`HEAD_LOSS_WEIGHTS` 等 | 仅选中对应模型时生效 |
| 质量输出 | `POSTPROCESS_CHECKPOINTS`、`SCENE_QUALITY_OPTIONS` / `POSTPROCESS_OPTIONS` | 整景使用前者配置，旧切片使用后者；推理、计量和可视化 |

默认 `METHOD="deeplabv3"` 使用 PVT-v2-B2 编码器、ASPP 和浅层跳连，单头 BCE，解码通道256、浅层投影48、空洞率 `(3,6,9)`。输入512×512时，深层16×16特征经过ASPP，与128×128浅层特征融合，最后恢复到512×512。它是输出步长32、普通卷积的 DeepLabv3+ 风格适配，不是原论文完整配置。

图像预处理统一通过 `src/datasets/image_process.py` 手动运行。正式训练读取
`I:\水体数据\processed_scenes\normalized\images` 与 `labels`；不执行校正、滤波或标准化。
原始SAR仍保存在 `I:\水体数据\prepared_scenes`，所有处理阶段保持原始尺寸、坐标系、仿射网格与train/val划分。

```python
# image_process.py
MODE = "preprocess"
ENABLE_CALIBRATION = False  # 当前BC缺少校正系数，保持关闭。
ENABLE_FILTER = True
FILTER_INPUT_STAGE = "org"
NORMALIZE_INPUT_STAGE = "filtered"
```

默认链路为ORG幅度→SNAP功率域Lee→恢复幅度→使用train统计做z-score→保存浮点GeoTIFF。
SNAP GPT、缓存目录、窗口索引缓存、读写窗口和训练统计抽样参数均在image_process.py顶部配置。
每景输出与完成记录匹配时复用；失败景重做，已完成景不重复执行。处理未完成时目录保留
`.preprocessing_incomplete`，正式训练拒绝读取；全部成功后发布 `preprocessing.json` 和窗口索引。
归一化只从train固定抽样估计均值/标准差，对train和val使用相同参数；有限0值参与统计，
无效区域保存为NaN，标准化后的负值仍是有效数据。训练直接读取浮点值，窗口外无效填充仍置0。
重新生成正在使用的阶段前先停止读取该成果的训练。每个阶段的整景Float32影像需额外磁盘空间，未压缩约4字节×波段数×像素数，实际使用压缩。

```powershell
python src/datasets/image_process.py
python src/supervised/train.py
```

检查点保存 `inputs_normalized` 和完整 `input_preprocessing`，包括原train均值、标准差、阶段参数和上游记录。
训练、验证、BN校准和质量输出会检查输入状态并跳过重复标准化。改变处理协议后应从头训练，
不能把旧原幅度模型直接续跑到新的离线标准化数据上。保留旧检查点的原始输入路径供历史复评。

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

`PRETRAINED=True` 只控制新实验的编码器初始化，是否续跑由 `RESUME_CHECKPOINT` 决定。DeepLab已整合到 `src/supervised/train.py`，不再提供旧的独立DeepLab脚本。

### 中断后续跑

在 `src/supervised/train.py` 顶部设置 `RESUME_CHECKPOINT` 为已有实验 `last.pth` 的绝对路径；设置为 `None` 表示新实验。随后仍运行同一个训练入口：

```powershell
python E:\Code_projects\sar_water_extraction\src\supervised\train.py
```

`EPOCHS` 表示总目标轮数：检查点完成第11轮、`EPOCHS=20` 时，从第12轮继续至第20轮。恢复模型、优化器、学习率调度器、AMP和最佳指标，保留原实验目录及此前日志。当前数据划分、波段、模型和训练配置需与检查点兼容；更换标签或算法方案应建立新实验。

新版还会在一轮训练结束、验证开始前保存 `train_end.pth`。如果验证中断，可将续跑路径指向它，重新执行这一轮完整验证，然后继续训练；`last.pth` 表示训练和验证均完成的轮次。训练批次中途的进度不会逐批保存，未保存部分需要重做。

新版检查点保存随机数状态；旧检查点若缺失会明确提示，仍可恢复其已有训练状态，但不能保证与不中断时逐批完全相同。检查点通过临时文件完整写入后替换；训练报错保留traceback供排查，不静默跳过坏样本。

## 训练与质量产物

新实验保存到独立时间戳目录；续跑沿用原实验目录，继续追加指标并更新检查点。

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

### 后处理四阶段与在线辅助数据

监督入口 `src/supervised/train.py` 的 `POSTPROCESS_OPTIONS`，以及阈值入口
`src/thres_segmentation/main.py` 和对应 pipeline，按以下阶段注释：

1. **数据准备**：锁定所选验证/外部影像，恢复输入契约；按每景范围下载并缓存辅助数据。本地参考文件优先。
2. **基础预测**：监督模型恢复各检查点滤波/标准化，平均概率并阈值化；阈值模型执行 Lee、百分位拉伸及阈值分割。这些处理不等于 Sigma0/Gamma0 辐射校正。
3. **辅助修正与水体分析**：保留原始预测，按显式开关补水；统计面积，结合历史参考计算新增/消退，结合 DEM/DSM 估计几何水深。
4. **评价与输出**：标签仅用于独立计分；修正前后使用相同有效标签域、汇总混淆计数，输出掩膜、CSV、质量图及分析栅格。

监督后处理复用阈值分割的 `auxiliary_data.py` 和 `water_analysis.py`。在
`POSTPROCESS_OPTIONS` 中设置：

```python
auxiliary_options=dict(download_history=True, download_elevation=True,
                       worldcover_year=2021, buffer_m=1000.),
auxiliary_cache_dir=Path(r"E:\Code_projects\sar_water_extraction\data_cache\auxiliary"),
use_landcover=False,       # 不要求输入影像自带第5波段。
use_external_water=False,  # 只对比历史数据；显式True才将历史水体补入postprocessed。
```

保持现有 `analysis_options` 的其余参数即可。仅下载历史水体时关闭
`download_elevation`；默认 `depth_scope='new_water'` 需要历史参考。
`reference_path` 可指定本地历史栅格或水体矢量；矢量需有效调查覆盖面，或明确声明完整覆盖。
不同景已有不同辅助文件时，可直接传入 `scene_analysis_options={样本名: 分析参数}`，
此时将 `auxiliary_options=None`，避免与自动准备产生歧义。

默认下载和外部补水开关均关闭。下载 WorldCover 后只用于变化对比，
不会自动改写模型结果；历史补水可能消除真实退水，因此始终分别分析 `raw` 与
`postprocessed`，评价模型及退水时应检查 `raw`。高程当前用于水深估计，
未将坡度直接作为监督模型的硬排除条件。

监督产物新增 `postprocess.json`（样本辅助路径及开关）、
`analysis/raw/<样本>/`、`analysis/postprocessed/<样本>/`，以及 `per_image.csv` 中的
`raw_water_area_m2`、`post_water_area_m2`、面积可用状态和辅助新增像素数。
`save_masks=False` 时仍通过临时掩膜完成分析；无 CRS 且无外部参考时面积标记为不可用，
有外部参考则要求 CRS，不能用像素数冒充平方米。
当前质量图中的 DEM/WorldCover 面板仍展示输入影像内置波段；外部辅助数据的结果见 `analysis/`。

在线产品支持 WorldCover 2020/2021 静态水体参考及 Copernicus GLO-30 DSM，
不代表任意日期的灾前状态或水下地形。水深是岸线高程插值的几何估算；
输入范围、编码、缓存与质量码详见 [阈值分割说明](src/thres_segmentation/README.md)。
上述外部辅助水体和水深功能属于旧切片后处理路径。`DATA_LAYOUT='full_scene'` 时，train.py会分派到独立的窗口化scene_quality路径；两条路径的配置不混用。

## 固定模型验证与概率阈值扫描

旧DFC切片的固定阈值验证与质量汇总如下；BC整景使用train.py的postprocess模式或下面model_eval.py的单次评价，勿将整景直接交给旧validation.py整图报告：

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
python E:\Code_projects\sar_water_extraction\src\compare_threshold.py
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


## BC整景质量输出与独立优化对照

整景质量输出仍从 `src/supervised/train.py` 进入。选择 `RUN_MODE="postprocess"`，在
`POSTPROCESS_CHECKPOINTS` 填写一个整景检查点，`SCENE_QUALITY_OPTIONS` 配置输出目录、
阈值网格、保存项和可选开闭运算。不会启动训练，也不会修改原始SAR、标签或模型。
运行时按训练窗口与halo读取，核心区不重叠写回原地理网格；标签只参与评价。
当前该入口使用检查点保存的完整val景；外部无标签影像和多模型集成仍需单独适配。

- `thresholds`：一次模型推理中扫描多个概率阈值，默认0.01～0.99，步长0.01。
- `threshold`：输出raw/post掩膜的固定工作点；扫描最优点不会自动替换它。
- `opening_radius`、`closing_radius`：局部形态学半径（像素）；均为0时post与raw一致。
  这里不执行DEM/土地覆盖/OSM修正，也不进行整景连通域面积筛除。
- `boundary_radii`：真值水体内外距离带，单位像素；不同地面分辨率下不能直接当作相同米数。
- `boundary_points`：每景每侧每半径最多固定抽样点数，默认256；种子固定，独立于预测。
  两侧定额抽样改变类别占比，采样点结果与全距离带结果分别报告。
- `save_probabilities`：输出压缩float32概率GeoTIFF；数值扫描直接基于logit，避免概率量化影响判别。
- `save_visuals`：每景一张降采样预览；图上指标来自原分辨率全景计数。
- `output_dir`：相同配置可继续未完成景；修改阈值/形态学/检查点时另用目录。
- `scene_limit=None`：全val；只用于试跑的景数限制必须明确保留在配置中，不能当作完整val结果。

输出目录位于检查点旁的 `quality/`：

| 文件/目录 | 含义 |
|---|---|
| `raw/`、`postprocessed/` | 原始及修正二值GeoTIFF：水体1、非水体0、无效255 |
| `probability/` | float32水体概率，nodata=-1；概率并非已经过置信度校准 |
| `figures/` | SAR、真值、raw、post、TP/FP/FN对照预览 |
| `threshold_overall.csv` | 按全部有效像素汇总的阈值曲线，附逐景宏平均F1 |
| `threshold_scenes.csv`、`threshold_groups.csv` | 各阈值的逐景/地理组指标 |
| `precision_targets.csv`、`recall_targets.csv` | 达到指定P或R要求的实测最优点，不做插值 |
| `threshold_boundary.csv`、`post_boundary.csv` | 全距离带的严格局部分类计数，不是容差Boundary F1 |
| `threshold_boundary_points.csv`、`post_boundary_points.csv` | 固定采样点对应指标 |
| `boundary_points/` | 可复核点位的行列号、标签、logit和raw/post预测 |
| `training_parity.csv` | 完成后核对0.5工作点与原训练验证的混淆计数差 |
| `status.json` | 已完成/预期景数；complete为false时只是阶段结果 |
| `counts/` | 每景计数与完成标记，用于中断续跑，不是训练中间特征 |

类别权重在 `CLASS_WEIGHT_MODE` 选择：`none` 为等权；`train_pixels` 为原逆频率；
`train_pixels_sqrt` 为逆频率平方根。后两者均保持训练像素分布下平均权重为1，
验证损失不加权。切换策略应 `RESUME_CHECKPOINT=None` 并使用新的输出目录；
输入、划分、采样、种子、训练步数和网络保持一致。正式训练需要将RUN_MODE改为train。

BN诊断使用 `scripts/calibrate_bn.py`：在顶部设置 `RUN_DIR`、`CHECKPOINT_NAME`、
`CALIBRATION_SAMPLES`。整景只从保存的train窗口固定随机取样，逐层恢复统计量；
同一完整val比较前后结果。校准结果另存 `bn_calibration/`，属于推理对照，不可当作原训练续跑检查点。
Lee滤波保持独立实验；需要确认幅度/功率表达，不能代替辐射校正。

逐景数值审查：将统一入口顶部设为 `MODE="radiometry"` 后运行：

```powershell
python E:\Code_projects\sar_water_extraction\src\datasets\image_process.py
```

默认每景3×3内部网格中心、256×256原分辨率抽样，输出到 `codex_analysis/radiometry_audit/`。
`scene_statistics.csv`包含原始数值分位数、零值/负值比例、标签两类数值分布和元数据；
`review_scenes.csv`列出待审查项。参考范围使用训练景的正值条件中位数，优先同传感器/像元间距。正值条件统计不改变有效像素定义，包含零值的统计同时保留。
异常标记不等于错误数据，不自动删除或加权；此统计也无法单独确定DN是幅度还是功率。


### 统一的图像处理入口

修改 `src/datasets/image_process.py` 顶部 `MODE` 后手动运行同一脚本：

| MODE | 功能 |
|---|---|
| `prepare` | 首次组织完整SAR与SHP标签，支持未完成目录续跑 |
| `labels` | 原位重建标签；运行前停止读取这套标签的训练 |
| `split` | 检查清单划分、重复景号和地理组隔离 |
| `inspect` | 抽查原始影像与矢量字段 |
| `radiometry` | 抽样统计数值分布，按train参考范围列出待检查场景 |
| `calibrate` | 执行已配置的厂商/GAMMA外部命令，或登记、检查已有校正GeoTIFF |
| `filter` | 对FILTER_INPUT_STAGE指定的线性域输入执行SNAP Lee |
| `normalize` | 对NORMALIZE_INPUT_STAGE输入估计train统计并生成浮点标准化整景 |
| `preprocess` | 按ENABLE_CALIBRATION/ENABLE_FILTER开关依次完成处理，最后归一化 |
| `quicklook` | 显示ORG、校正后、滤波后、标准化后的四阶段概览，不启动这些处理 |

辐射校正必须先明确 `CALIBRATION_QUANTITY`（sigma0/beta0/gamma0）、输出域和
`CALIBRATION_REFERENCE`（系数、LUT、算法与匹配依据），然后配置：

- `CALIBRATION_PRODUCTS_DIR`：已校正GeoTIFF，按train/val和原文件名组织。
- 或 `CALIBRATION_COMMAND`：可执行参数列表，支持 `{input}`、`{output}`、`{scene_id}`
  和WSL的 `{input_wsl}`、`{output_wsl}` 占位符。不经shell拼接；外部流程必须输出同网格GeoTIFF。

当前BC ORG只有幅度，缺少可核验校正参数。未配置时calibrate明确停止，不产生伪校正影像。
幅度平方只转换到功率，不是辐射校正；当前默认关闭校正。
开启完整校正链时设置 `ENABLE_CALIBRATION=True`、`FILTER_INPUT_STAGE="calibrated"`。

阶段目录集中在顶部：`CALIBRATED_ROOT`、`FILTERED_ROOT`、`NORMALIZED_ROOT`。
分别包含images、labels、划分清单及处理记录。滤波输出使用独立GeoTIFF，后续不依赖SNAP缓存长期保留。
`WINDOW_SIZE/HALO`应与train.py一致，正式训练只读取已建索引。

快视图设置 `MODE="quicklook"`、`QUICKLOOK_LOOKS=8`、`QUICKLOOK_MAX_SIDE=1600`。
输出 `quicklooks/景号.quick_overview.png`，2×2依次为原图、校正、滤波、归一化；
缺少成果或阶段未完成时明确标记Not generated，不用原图代替。
概览按有效像素均值降采样，0参与平均、无效值不参与，保留边缘余块。
ORG幅度固定0..255直接量化，不自动拉伸；有明确后向散射物理量的阶段用固定dB范围显示，
归一化阶段直接用已保存z值、固定范围显示，不再计算一次标准化。显示变换不修改训练影像。
快视图无需检查点；参数或源成果改变时重新生成，匹配时复用。
