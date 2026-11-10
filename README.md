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

训练、后处理和影像处理入口分别读取 `configs/train.yaml`、`configs/post_process.yaml`、`configs/image_process.yaml`，无需命令行参数。所有用户可修改的路径集中在 YAML 的 `paths` 分组，使用绝对路径；模型、训练、预处理开关放在对应参数组。其他独立工具仍使用顶部变量。代码中的模块导入路径、样本文件拼接和时间戳输出子目录不属于数据路径配置。


### YAML 使用约定

- `train.yaml`：管理训练数据、续跑、模型和训练验证图件。
- `post_process.yaml`：独立管理待评估检查点、后处理输出目录、阈值扫描和修正参数。
- `image_process.yaml`：`paths` 管理源数据和各阶段目录；`run.MODE` 每次选一个任务；`processing` 管理滤波和归一化；`snap` 管理 GPT 线程、内存及成功后的缓存清理。
- YAML 中使用 `null`、`true`、`false`，列表写成 `[1]` 或多行 `-`；不填写 Python 的 `Path(...)`、`None` 或元组语法。Windows 路径可不加引号或用单引号，避免双引号解析反斜杠。
- 脚本按自身位置查找仓库内的配置，不依赖终端当前目录；运行命令不变。修改 YAML 后重新启动才生效，已运行的进程不会重新加载参数。
- SNAP 后端 `src/datasets/snap_preprocess.py` 只接收影像处理入口传来的路径，没有机器路径默认值；训练不读取 SNAP 配置。
- 每次从训练入口启动时，将 YAML 快照保存到本次结果目录，续跑也保留先前快照。`train_end.pth` 从该轮验证开始；`last.pth` 从下一轮训练开始。

## 目录与职责

```text
sar_water_extraction/
├── configs/                           # train.yaml、post_process.yaml、image_process.yaml
├── src/
│   ├── supervised/
│   │   ├── train.py                   # 训练及本轮val质量图
│   │   ├── post_process.py            # 独立模型推理、后处理和质量输出
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

运行独立准备入口，配置在 `configs/image_process.yaml`：

```powershell
python E:\Code_projects\sar_water_extraction\src\datasets\image_process.py
```

`MODE: labels` 仅重建已有标签；首次导入SAR时显式设置 `MODE="prepare"`，按整景输出，不裁剪、不缩放、不生成512×512小切片。`MODE="split"` 仅显示划分；`MODE="inspect"` 显示少量源文件的格式。输入清单由 YAML 的 `paths.SPLIT_CSV` 指定，沿用地理组及train/val归属。

支持具有地理参考的实数GeoTIFF（`.tif/.tiff`），以及内含唯一GeoTIFF的ZIP；标签为SHP面及其配套文件。ZIP逐景临时展开，完成后清理缓存，原始文件保留。

YAML 主要配置（保持原参数名，按分组填写）：

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

YAML 的 `paths.OUTPUT_DIR` 指定原位替换目录，`READ_WINDOW_SIZE` 控制内部读写内存。每景先写临时标签，完整关闭后替换原文件；全部完成后同步 `split_manifest.csv` 像素统计、`label_statistics.json` 和 `revision_settings.json`。中断后重新运行会从第一景重建；`.labels_rebuilding` 存在时训练读取器拒绝使用尚未完成的数据。

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

新实验在 `RESUME_CHECKPOINT: null` 时使用 `PRETRAINED` 决定是否加载ImageNet编码器权重；显式指定续跑检查点时恢复已有水体模型，不再下载或加载ImageNet初始化权重。`CLASS_WEIGHT_MODE="train_pixels"` 仅使用train不重叠核心的已标注像素，设置BCE类别权重 `w_c=N/(2*N_c)`，顺序为非水体、水体；未知255不计数。它让两类在全训练像素总体上的权重总量相同，并不保证每批类别平衡或效果提升。为避免叠加重加权，本轮使用 `SAMPLING="random"`；改回旧实验时设类别权重模式为`none`。

训练loss使用类别加权，val loss保持未加权，两者不能直接按数值大小判断过拟合。精度指标不加权，概率阈值仍为0.5；加权可能改变概率校准，后续阈值选择仅用val。DeepLab只有一个监督头，头权重为1；UAFNet的四头权重不影响它。

除原有`best.pth`（val loss最低）、`best_f1.pth`和`last.pth`，另保存`best_balanced_accuracy.pth`，按水体与非水体召回率的平均值选取。全水体/全非水体预测在两类均存在时该值为0.5；仍须结合两类mIoU、误差图、各景表现判断质量，文件名中的best不表示模型已经有效。

当前训练内验证采用带上下文的中心裁取拼接计量，不是重叠概率平均。训练验证本身不保存全幅预测。每轮val直接输出质量概览；完整整景概率/掩膜及修正图由独立post_process.py分派到scene_quality生成，详见下方“BC整景质量输出”。当前不自动导出矢量。

## 训练配置

在 `configs/train.yaml` 中按功能分组修改，脚本不再保存另一套参数默认值：

| 参数组 | 主要变量 | 用途 |
|---|---|---|
| 模型选择 | `METHOD` | train.py只训练；后处理另运行post_process.py |
| 数据 | `IMAGE_DIR`、`LABEL_DIR`、`POLARIZATIONS`、`POLARIZATION_BANDS` | 数据路径、极化与文件波段映射 |
| 斑点滤波 | `image_process.yaml` 的 `processing` 中的 `FILTER_INPUT_STAGE`、`FILTER_WINDOW`、`ENL` | 正式训练不调用SNAP，实际滤波在离线模式完成 |
| 输出 | `OUTPUT_ROOT`、`OUTPUT_DIR`、`RESULT_FILES` | 最终实验目录及模型/指标/日志名称；运行只加时间戳，续跑沿用原目录 |
| 缓存路径 | `WINDOW_CACHE_DIR` | 训练窗口索引缓存；后处理路径已移至post_process.yaml |
| 训练 | `RESUME_CHECKPOINT`、`PRETRAINED`、`BATCH_SIZE`、`EPOCHS`、`LEARNING_RATE`、`SEED` | 续跑/初始化、批次、总目标轮数、学习率和随机种子 |
| 优化 | `OPTIMIZER_NAME`、`WEIGHT_DECAY`、`COSINE_SCHEDULE`、`TRAIN_AUGMENTATION` | 单通道水体方法的优化与增强设置 |
| 损失 | `LOSS_NAME`、`CLASS_WEIGHT_MODE`、`BCE_WEIGHT`、`DICE_WEIGHT` | 选择损失及BCE类别权重；后两项仅 BCE+Dice 使用 |
| 采样 | `SAMPLING`、`HARD_WARMUP_EPOCHS`、`HARD_SAMPLE_RATIO`、`HARD_POOL_RATIO` | 随机/困难采样及比例 |
| DeepLab | `DEEPLAB_*` | 编码器、解码宽度、ASPP空洞率、Dropout |
| 其他网络 | `UAFNET_*`、`STAGE`、`HEAD_LOSS_WEIGHTS` 等 | 仅选中对应模型时生效 |
| 训练质量输出 | `SAVE_VALIDATION_FIGURES`、`INTERVAL`、`MAX_SCENES`、`PREVIEW_MAX_SIDE` | 本轮val预测拼概览，图件标题使用原生像素计数 |

默认 `METHOD="deeplabv3"` 使用 PVT-v2-B2 编码器、ASPP 和浅层跳连，单头 BCE，解码通道256、浅层投影48、空洞率 `(3,6,9)`。输入512×512时，深层16×16特征经过ASPP，与128×128浅层特征融合，最后恢复到512×512。它是输出步长32、普通卷积的 DeepLabv3+ 风格适配，不是原论文完整配置。

图像预处理统一通过 `src/datasets/image_process.py` 手动运行。正式训练读取
`I:\水体数据\processed_scenes\normalized\images` 与 `labels`；不执行校正、滤波或标准化。
原始SAR仍保存在 `I:\水体数据\prepared_scenes`，所有处理阶段保持原始尺寸、坐标系、仿射网格与train/val划分。

```yaml
# 修改 configs/image_process.yaml 中相应分组的字段，其余字段保留。
run:
  MODE: preprocess
processing:
  ENABLE_CALIBRATION: false
  ENABLE_FILTER: true
  FILTER_INPUT_STAGE: org
  NORMALIZE_INPUT_STAGE: filtered
```

默认链路为ORG幅度→SNAP功率域Lee→恢复幅度→使用train统计做z-score→保存浮点GeoTIFF。
SNAP GPT、缓存目录、窗口索引缓存、读写窗口和训练统计抽样参数均在configs/image_process.yaml中配置。
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

训练和后处理已经拆为两个入口，不再设置 `RUN_MODE`：

```powershell
python E:\Code_projects\sar_water_extraction\src\supervised\train.py
# 训练完成或已有模型后，按需单独运行：
python E:\Code_projects\sar_water_extraction\src\supervised\post_process.py
```

训练默认每轮验证生成全部验证景的质量概览，以及训练/验证loss、IoU、F1、Precision和Recall曲线。
`quality.INTERVAL`控制出图轮次，`MAX_SCENES: null`表示全部验证景；限制图件数量不会限制验证指标统计。
图件直接使用当轮val预测，不重复模型推理，不输出中间层特征。未验证或无标签位置明确忽略；缩略图不能作为完整整景掩膜。
历史已完成轮次不会自动补图，从下一次执行验证开始输出；完整模型复评使用独立后处理入口。

```text
某次实验/
├── best_f1.pth、last.pth、train_end.pth、metrics.csv
├── quality/                          # 训练自己的质量输出
│   ├── training_curves.png
│   └── epoch_006/
│       ├── figures/场景编号.png       # SAR、真值、概率、原始预测、误差
│       ├── per_scene.csv             # 图中指标对应的原分辨率计数
│       └── status.json               # 完整验证结束才为complete=true
└── postprocess/某次时间戳/
    ├── post_process_config_*.yaml
    └── quality/                      # 独立推理/修正的成果、指标和图件
        ├── figures/场景编号.png       # SAR、真值、raw、post、误差
        ├── raw/、postprocessed/、probability/
        └── threshold_*.csv、boundary_points/等
```

`PRETRAINED=True` 只控制新实验的编码器初始化，是否续跑由 `RESUME_CHECKPOINT` 决定。DeepLab已整合到 `src/supervised/train.py`，不再提供旧的独立DeepLab脚本。

### 中断后续跑

在 `configs/train.yaml` 的 `paths` 中设置 `RESUME_CHECKPOINT` 为已有实验 `last.pth` 的绝对路径；设置为 `null` 表示新实验。随后仍运行同一个训练入口：

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
| `quality/` | 每轮val概览图、逐景计数与训练曲线 |
| `postprocess/时间戳/quality/` | 独立后处理的整景掩膜、概率、逐景精度、质量图和边界评价 |

`.pth` 保存模型权重、模型配置、波段、归一化参数及训练/验证名单等。恢复推理应通过 `models/checkpoint.py`，不要仅凭文件名猜测网络结构。

`SAVE_FEATURE_MAPS=False` 关闭中间层特征图。质量图是最终成果，不属于训练中间特征。`use_landcover=True` 会额外生成土地覆盖修正结果，评价网络本身时使用 **raw** 掩膜及指标，不把修正结果混入原始模型成绩。

### 后处理四阶段与在线辅助数据

独立入口 `src/supervised/post_process.py` 读取 `configs/post_process.yaml` 的 `POSTPROCESS_OPTIONS`（旧patch）/`SCENE_QUALITY_OPTIONS`（BC整景），以及阈值入口
`src/thres_segmentation/main.py` 和对应 pipeline，按以下阶段注释：

1. **数据准备**：锁定所选验证/外部影像，恢复输入契约；按每景范围下载并缓存辅助数据。本地参考文件优先。
2. **基础预测**：监督模型恢复各检查点滤波/标准化，平均概率并阈值化；BC阈值入口复用离线SNAP和z-score，再执行整景百分位分割。这些处理不等于 Sigma0/Gamma0 辐射校正。
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

旧DFC切片的固定阈值验证与质量汇总如下；BC整景使用post_process.py或下面model_eval.py的单次评价，勿将整景直接交给旧validation.py整图报告：

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

这是独立入口，不加载模型检查点。`MODE="dataset"` 从 `configs/image_process.yaml` 的 `paths.NORMALIZED_ROOT` 读取固定val影像，直接复用SNAP滤波和train统计z-score成果，不再执行旧Lee近似、逐景拉伸或转dB；按整景百分位提取水体，**不是Otsu**。`MODE="gui"` 保留历史GIS流程，未使用此BC离线链路。

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

整景后处理从 `src/supervised/post_process.py` 进入。在 `configs/post_process.yaml` 的
`paths.POSTPROCESS_CHECKPOINTS` 填写一个整景检查点，`paths.OUTPUT_DIR` 指定本次后处理根目录；
`quality.SCENE_QUALITY_OPTIONS` 配置
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

OUTPUT_DIR为null时，后处理目录位于检查点旁的 `postprocess/时间戳/quality/`；整景续跑时将OUTPUT_DIR设为原`postprocess/时间戳`目录。图件默认开启：

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
验证损失不加权。切换策略应 `RESUME_CHECKPOINT: null` 并使用新的输出目录；
输入、划分、采样、种子、训练步数和网络保持一致。正式训练直接运行train.py。

BN诊断使用 `scripts/calibrate_bn.py`：在顶部设置 `RUN_DIR`、`CHECKPOINT_NAME`、
`CALIBRATION_SAMPLES`。整景只从保存的train窗口固定随机取样，逐层恢复统计量；
同一完整val比较前后结果。校准结果另存 `bn_calibration/`，属于推理对照，不可当作原训练续跑检查点。
Lee滤波保持独立实验；需要确认幅度/功率表达，不能代替辐射校正。

逐景数值审查：将 `configs/image_process.yaml` 中的 `run.MODE` 设为 `radiometry` 后运行：

```powershell
python E:\Code_projects\sar_water_extraction\src\datasets\image_process.py
```

默认每景3×3内部网格中心、256×256原分辨率抽样，输出到 `codex_analysis/radiometry_audit/`。
`scene_statistics.csv`包含原始数值分位数、零值/负值比例、标签两类数值分布和元数据；
`review_scenes.csv`列出待审查项。参考范围使用训练景的正值条件中位数，优先同传感器/像元间距。正值条件统计不改变有效像素定义，包含零值的统计同时保留。
异常标记不等于错误数据，不自动删除或加权；此统计也无法单独确定DN是幅度还是功率。


### 统一的图像处理入口

修改 `configs/image_process.yaml` 的 `run.MODE` 后手动运行同一脚本：

| MODE | 功能 |
|---|---|
| `all` | 顺序执行prepare、按开关预处理、quicklook；未完成prepare时停止，重跑复用已完成阶段 |
| `prepare` | 首次组织完整SAR与SHP标签，支持未完成目录续跑 |
| `labels` | 原位重建标签；运行前停止读取这套标签的训练 |
| `split` | 检查清单划分、重复景号和地理组隔离 |
| `inspect` | 抽查原始影像与矢量字段 |
| `radiometry` | 抽样统计数值分布，按train参考范围列出待检查场景 |
| `calibrate` | 执行已配置的厂商/GAMMA外部命令，或登记、检查已有校正GeoTIFF |
| `filter` | 对FILTER_INPUT_STAGE指定的线性域输入执行SNAP Lee |
| `normalize` | 使用本批train统计或指定参考统计，生成浮点标准化整景 |
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

阶段目录集中在 `configs/image_process.yaml` 的 `paths`：`CALIBRATED_ROOT`、`FILTERED_ROOT`、`NORMALIZED_ROOT`。
分别包含images、labels、划分清单及处理记录。滤波输出使用独立GeoTIFF，后续不依赖SNAP缓存长期保留。
`WINDOW_SIZE/HALO`应与train.py一致，正式训练只读取已建索引。

批次处理可将 `MODE` 设为 `all`，并同步设置 `PREPARE_BATCH` 和独立的准备、处理、缓存路径。
它不执行 `split/inspect/radiometry/labels`，也不启动训练或模型验证。
中断后使用同一配置重新运行，已完成prepare的manifest及匹配的阶段成果直接复用。
`NORMALIZATION_REFERENCE` 指向已完成的 `preprocessing.json` 时，复用该训练均值/标准差；
滤波配置、数据域和波段必须一致。设为 `null` 时由本批train估计，val不参与统计。
统一分类CSV只保留表头和逐景数据，分组规则及执行流程放在同目录的 `水体数据分组说明.md`。

快视图设置 `MODE="quicklook"`、`QUICKLOOK_LOOKS=8`、`QUICKLOOK_MAX_SIDE=1600`。
输出 `quicklooks/景号.quick_overview.png`，2×2依次为原图、校正、滤波、归一化；
缺少成果或阶段未完成时明确标记Not generated，不用原图代替。
概览按有效像素均值降采样，0参与平均、无效值不参与，保留边缘余块。
ORG幅度固定0..255直接量化，不自动拉伸；有明确后向散射物理量的阶段用固定dB范围显示，
归一化阶段直接用已保存z值、固定范围显示，不再计算一次标准化。显示变换不修改训练影像。
快视图无需检查点；参数或源成果改变时重新生成，匹配时复用。


SNAP提速：`configs/image_process.yaml`的`paths.SNAP_WORK_DIR`默认位于E盘NVMe SSD，
`SNAP_CACHE_DIR`仍位于I盘，原有成功缓存及filtered成果继续复用。
单景中间文件在SSD处理，完成后复制到I盘临时文件并原子发布；失败时不登记成功。
缓存到filtered目录使用直接文件复制，不重复解码/压缩，保留缓存文件的全部字节。
每个新计算缓存目录的`timings.json`记录输入准备、GPT、恢复掩膜/压缩、缓存发布耗时；
filtered影像旁的`*.copy_timing.json`记录缓存复制时间。历史成果不为补计时而重新计算。
修改后应先停止旧处理实例，再用相同MODE和缓存路径重新运行；不同时启动两套处理。
算法窗口、ENL、GPT线程和tile cache保持原设置；本改动不更改数值配置和缓存键。


归一化提速参数位于configs/image_process.yaml中：`NORMALIZE_WORK_DIR`为E盘SSD临时目录，
`NORMALIZE_READ_SIZE=1024`合并原有256像素小块读取；输出仍为256像素存储块。
`NORMALIZE_THREADS=2`启用单景并行压缩，`NORMALIZE_COMPRESSION_LEVEL=1`采用快速DEFLATE无损压缩，
文件可能稍大，但浮点值不变。文件在SSD关闭并完成压缩后复制到I盘目标临时文件，再校验发布。
统计抽样、均值/标准差及运算精度不变，性能参数不进入数值签名，已完成成果继续复用。
`*.normalize_timing.json`分别记录读取、计算、写入/压缩、复制耗时；统计参数命中缓存时不重新计算。


### 多目录整景训练

在 `configs/train.yaml` 的 `paths.MANIFEST_PATHS` 填写一个或多个绝对路径。
每个目录应有 `split_manifest.csv`、`images/train|val`、`labels/train|val` 和已完成的 `preprocessing.json`。
当前列表仅启用 batch 1。整景模式以此列表为准，`IMAGE_DIR`/`LABEL_DIR` 仍供其他单目录入口使用。

多目录模式沿用各自的 train/val，不重新划分，也不启用 A/B/C/D 交叉验证。
它复用各目录的窗口缓存，不复制SAR；实验目录中的 `split_manifest.csv` 保存实际合并清单，
`manifest_source`、`dataset_root` 记录来源，`name`/`label_name` 为绝对路径，`original_name` 保留原文件名。
实验目录同时保存预处理信息，检查点指向该清单备份，后续验证仍需要原始数据文件存在。

重复景号、跨目录地理组混入 train/val、未完成预处理、不同归一化均值/标准差或处理方案会报错。
两批独立归一化得到的参数不同，不能直接加入同一次训练；应先统一统计和预处理。
新增或移除目录属于新的数据实验，应设置 `RESUME_CHECKPOINT: null`，不要续跑旧实验。


### 重叠融合验证与续跑

`configs/train.yaml` 的 `validation_inference` 控制整景验证的滑窗融合。
例如输入窗口1024、halo128时，中心有效区为768；`stride: 384` 使中心区重叠50%。
`blend: hann` 对多个窗口的 logits 加权平均，随后统一计算概率、阈值分类和精度。
各有效像素只统计一次，quality图复用同一份最终预测；不改变训练窗口、损失或采样。
融合输入窗口只读取SAR和有效掩膜，标签留到最终输出时读取；模型推理不依赖标签。
整体、逐景与图件复用每块的同一份原分辨率计数，图件只复制概览采样像素。
这些执行优化不改变融合协议，无需为此重新训练或另开实验目录；下次启动生效。
推理按条带滚动缓存，不在内存中保存整景多窗口预测；窗口数约为旧步长的4倍，验证会更慢。

设置 `paths.RESUME_CHECKPOINT` 后运行 `python src/supervised/train.py`。
`train_end.pth` 会先重跑该轮验证，再继续下一轮训练；`last.pth` 从下一轮训练开始。
当检查点和当前配置的验证融合设置不一致时，程序在 `OUTPUT_DIR` 下新建实验目录，
保留模型、优化器、学习率调度、AMP及随机数状态，重新建立新口径的best和指标记录。
源实验不覆盖，新目录的 `resume_source.json` 和 `validation_protocol.json` 记录来源及口径。
新目录再次中断后，将 `RESUME_CHECKPOINT` 改为该新目录的最新检查点，避免重复迁移旧实验。

`configs/post_process.yaml` 中 `SCENE_QUALITY_OPTIONS.inference_options` 可指定相同的
`stride/blend`；省略时恢复检查点中的融合设置，显式 `{}` 使用旧中心区直接拼接。
融合只减少窗口间的不连续，不保证纠正阴影误检、整片水体漏检或全部岸线误差。
完整验证结果仍需检查整体、逐景和边界指标，旧硬拼接与新融合结果分开记录。

### 场景地物分类（独立于训练和后处理）

运行 `python src/datasets/classify_scenes.py`，参数集中在脚本开头的 `PATHS` 和 `OPTIONS`，不需要独立YAML。
默认读取当前 `prepared_scenes/split_manifest.csv` 中全部train/val场景；多个数据集在
`paths.MANIFESTS` 中追加，按完整景号去重，同景划分冲突会报错。原始manifest、标签及train/val不修改。
未采用行保留原因，缺影像或辅助产品下载失败逐景记录，不把缺失值当成非农田或平地。

- 土地覆盖：使用 ESA WorldCover 2021 v200 的原始11类瓦片，复用后处理的
  `data_cache/auxiliary/tiles/worldcover` 缓存。后处理生成的二值历史水体图不能替代原始分类图。
- 地形：使用 Copernicus GLO-30 DSM，复用 `tiles/copernicus` 缓存；DSM包含建筑和植被高度，
  因此坡度标签仅用于场景筛查，不代表裸地坡度、正式地貌分类或SAR叠掩/阴影判定。
- 统计：按SAR有效范围在30米局部UTM规则网格抽样估计面积占比，分块读取，不加载整景到内存。
  类别使用最近邻，DSM使用双线性重采样；坡度在带1像素邻域的块上计算。
  SAR的0值不自动无效，显式nodata、掩膜及非有限值排除。
- 地类占比分母是有WorldCover观测的有效SAR网格；覆盖率分母是所有有效SAR网格。
  坡度比例另用有效坡度网格为分母。表中占比为0到1的小数，空白表示未知，不是0。
  覆盖率不足95%不发布确定地类/地形标签。
- 同一地类达到20%标记“较多”，最大地类达到50%标记“主导”；否则为混合。
  坡度>=15度占有效坡度网格>=20%标记“山地/陡坡候选”，否则仅标记“陡坡占比较低”。
  这些阈值是可调整的项目规则，不是已验证的统一行业标准。

输出 `outputs/scene_classification/场景地物分类.csv`，包含景号、原划分、地理组、各地类占比、
数据覆盖率、平均坡度、陡坡占比、地形标签、来源年份、瓦片路径和失败原因。
`records/` 保留逐景统计回执；每完成一景更新CSV。相同影像和参数的完成景在重启后直接复用；
失败或覆盖不足景会重试。查看CSV时避免用Excel锁定文件，否则无法继续写表。
首次下载可能较大，建议在训练之外手动执行；脚本不会加载模型或启动训练。
`MAX_SCENES=3` 可先试跑，确认后改回 `None`；`DOWNLOAD_MISSING=False` 可仅使用已有缓存。
后处理本身不必先运行。

WorldCover 2021与较新的SAR存在年份差异，本表描述参考土地覆盖背景，不替代同期真值。
30米抽样可能漏掉窄小地物，需要更精细占比时将 `RESOLUTION_M` 改为10米，但计算量增加，
且不会提高GLO-30原生地形精度。全景农田占比不等于水体邻域农田占比，
不能仅凭场景标签认定漏检发生在农田中；该结论需要与预测误差空间叠加。

数据来源：[ESA WorldCover](https://esa-worldcover.org/en/data-access)，
[Copernicus GLO-30](https://registry.opendata.aws/copernicus-dem/)。
