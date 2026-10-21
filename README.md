# SAR water extraction

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
├─ train.py         # METHOD 同级选择 unet、pvt、resnet50、uanet、uafnet
└─ calibrate_bn.py  # 独立的可选 BN 校准
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

`OUTPUT_DIR=None` 时按方法自动选择上述目录；指定路径则使用该路径。每次仍创建独立时间戳目录。`uanet` 是训练入口中的方法名，检查点继续记录 `model_name="pvt_uanet"`，兼容历史权重。

调用关系为 `__main__ → create_trainer() → WaterSegmentationTrainer(...) → trainer.train()`。`WaterSegmentationTrainer` 与入口同在 `train.py`，直接完成数据准备、标准化、训练/验证、指标与保存；已删除独立训练器文件和模型专用入口。

模型按骨干集中组织，类与方法的定位见 [模型结构与函数索引](docs/model_structure.md)。已有四种方法保留原模型和训练规则；新增 `uafnet` 的训练设置见下节。旧 U-Net 数字层名仍由 `UNet.load_state_dict` 兼容。

## UAFNet 水体论文训练

在 `src/train.py` 设置 `METHOD="uafnet"`，仍运行 `python src/train.py`。所有训练逻辑留在同一个 `WaterSegmentationTrainer` 内，没有新增训练入口或训练器子类。

依据 [UAFNet 水体论文](docs/papers/2024_UAFNet_SARFloodMapping_IGARSS_Published.pdf) 第 2.2、3.1 节实现：

- 单模型 PVT-v2-B2 → MBDC → FPN/语义解码头产生粗预测 → 三级 UAFM 细化；不添加建筑 UANet 的 PSM/CGM。
- 四个输出均为 `[B,1,H,W]` 的水体 logits，训练损失是四个有效像素平均 BCE 之和。使用数值稳定的 `BCEWithLogitsLoss`，标签 `-1` 不参与计算。
- AdamW、初始学习率 `1e-4`、每轮余弦退火；验证只用最终输出，`sigmoid(logit)>=0.5` 判为水体。
- 水体 UAFM 使用论文式(5)(6)的 1×1 特征压缩、最近邻上采样等级图、式(7)的 3×3 预测卷积；原双通道 UANet 的计算保持不变。
- 五级权重按论文文字区间 `[0,.1)` 等分箱，`p=.5` 时两侧均为等级 5。论文式(3)的 floor 写法与文字区间有边界差异，本实现明确采用文字区间，等级不缩放；等级图继续 detach。

当前仍是**参考论文的本数据集适配版**：

- 保留 `(1,2)` 两通道与训练集统计标准化，未推断极化顺序或数值单位。论文输入为 `VH,VV,VH`，只有确认文件的极化对应关系后才可设置重复波段，例如确认 1=VV、2=VH 后使用 `(2,1,2)`。
- batch size 2、20 轮、权重衰减 0.01、余弦最低学习率 0 为本项目设置，论文没有提供完整数值；不能称为原作者全部超参数。
- `TRAIN_AUGMENTATION=True` 时，仅训练批次随机做 90° 倍数旋转与水平翻转，影像和标签同步。它覆盖方形切片的八种正交变换，不是论文未详述角度和翻转方式的 12 倍扩增。验证不增强，先划分再增强。
- MBDC、FPN 解码头及高层融合特征的双线性插值复用现有实现，论文未完整明确的细节不声称逐行复现。当前只训练 B2，不包含 B3/B4/B5 集成、测试增强或土地覆盖后处理。

每次创建 `outputs/pvt_uafnet/时间戳/`，保存 `metrics.csv`、最低验证 loss 的 `best.pth`、最高验证 F1 的 `best_f1.pth`、`last.pth`，异常时另存 `abnormal.pth`。新检查点名为 `pvt_uafnet`，模型加载器已支持；另保存优化器、调度器和 AMP 状态，但本次没有加入断点续训入口。不要直接把旧双类别 UANet 权重严格加载到单通道 UAFNet。

旧 `calibrate_bn.py` 的评价逻辑仍面向双类别 CE，不应直接用于单通道 UAFNet；本阶段先正常训练和验证，不额外校准 BN。

接口检查：`python -m unittest discover -s tests -p test_uafnet_training.py -v`。该检查使用合成小样本，不代表正式数据精度。新训练同时改变损失、优化器、调度、增强和融合细节，不能把它与旧 FPN 的差异全部归因于 UAFM。

## 独立评价 F1

运行 `python src/evaluate.py`，默认评价 `outputs/pvt_uafnet/20261021_102433_387373/best_f1.pth` 保存的原内部验证集。脚本直接在顶部定义变量，不使用 argparse，不写 JSON 或评价报告文件。

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
