# 独立阈值分割与水体边界质量评价

> **原创作者: 季亚楠, 2026**

`src/thres_segmentation` 是独立的传统阈值处理流程：直接读取影像，输出分割成果；可选标签仅用于评价。它不加载模型、不读取 checkpoint 中的数据划分、不执行网络对比或自动选参。

## 当前BC整景入口

直接运行 `python E:\Code_projects\sar_water_extraction\src\thres_segmentation\main.py`。
顶部默认`DATA_LAYOUT="full_scene"`，从`configs/image_process.yaml`的`paths.NORMALIZED_ROOT`读取`images/val`及对应labels、split_manifest。只读取清单active=1的val，单VV band 1，关闭土地覆盖。`SCENE_PREFIX=None`包含BC2/BC3；只处理BC3可设`"BC3-"`。这会改变评价范围，模型对照时必须使用相同名单。

`source_image`指向原始GeoTIFF或ZIP，`metadata.py`读取原始栅格头和严格同名XML。ZIP不整包解压；不猜用目录第一份XML。`metadata.csv`记录原始路径、原始CRS/变换/网格间距、标签字段的原文值及XML来源/哈希；`metadata/`保留XML副本。缺失角度保持空值，不补默认角度。XML中的pixelValue、雷达斜距/方位间距可能描述上游产品，不能覆盖当前GeoTIFF的实数类型和地理编码网格；不据此自动转dB或进行辐射校正。

当前纯SAR阈值不使用航向角/入射角进行修正。读取这些字段是为保存准确的来源，**不代表已启用DEM雷达阴影/叠掩处理**。准备影像必须与原始GeoTIFF网格一致，标签也必须同网格。

BC整景直接复用image_process生成的成果：ORG幅度平方→SNAP Lee功率域滤波→恢复幅度→使用train统计做z-score。这里不再次滤波、转dB或做逐景P1/P99拉伸。阈值阶段两遍分块读取：汇集全景有效z-score求精确百分位→按统一阈值逐块写mask。`BLOCK_SIZE=1024`控制内存，不生成小切片；不会每块独立估计噪声、拉伸或阈值。负z-score及有效0保留，未知标签255不参与精度，但影像有效的未知区域仍输出预测。

精确百分位通过临时磁盘映射缓冲原地选择，峰值磁盘约`8×单景有效像素数`字节（十亿像素约7.45GiB）；使用`CACHE_DIR`，正常完成或Python异常退出时清理。强制杀进程可能留下临时目录。此方案增加I/O以避免把多份整景浮点数组加载到内存；输出仍是原网格GeoTIFF，不缩放输入。当前按场景依次执行，默认全部val，不自动启动批处理。

`figures/`是最长边`PREVIEW_SIZE=1200`的最近邻缩略图，小水域可能在预览中消失；图中文字、整体`metrics.csv`和逐景`per_image.csv`均来自全分辨率计数。无标签区域不作为非水体评价。`boundary/metrics.csv`与`boundary/per_image.csv`为原生像素1/3/5半径内的精确计数，窗口halo避免块边界伪岸线。整景模式暂不输出旧切片流程的定额采样点和边界PNG，避免把每窗口定额当成每整景定额；边界评价使用全区域计数。

`main.py`的dataset入口统一要求已完成的离线归一化数据，patch分派也使用同一口径。下文旧DFC/GUI说明属于历史流程；历史算法通过`ThresholdSegmenter(preprocessing="legacy")`保留，不能用于已归一化输入。GUI仍为历史GIS流程，未接入新的BC离线链路。


### 与训练共享预处理的执行顺序

1. 在 `configs/image_process.yaml` 中设置 `run.MODE: preprocess`，当前未校正ORG使用 `ENABLE_CALIBRATION: false`、`ENABLE_FILTER: true`、`INPUT_DOMAIN: amplitude`、`FILTER_INPUT_STAGE: org`、`NORMALIZE_INPUT_STAGE: filtered`，手动运行 `image_process.py`。
2. 全部完成后运行 `main.py`。如果离线成果已经完成，直接执行此步即可；阈值入口不启动SNAP，不重复生成成果。
3. `WATER_PERCENT=10` 仍表示整景暗值侧第10百分位，不是0.1水体概率。阈值现在以z-score表示，可以为负数，不应限制到0～1。

均值/标准差来自训练集，验证集只复用；分位阈值本身按当前影像的有效像素计算，这是传统算法的自适应决策，不是重新归一化。有限0和负值参与，nodata/NaN排除。只有`active=1`的val参与评价，标签不参与阈值计算。

输出 `input_preprocessing.json` 备份实际成果的处理记录，`methods.csv`/`per_image.csv`明确阈值数值域；`source_image`仍用于寻找原始XML。PNG的P2/P98拉伸仅用于显示，不进入阈值计算。未完成归一化或错误地对已归一化数据使用legacy模式会报错。

z-score在标准差为正时是单调线性变换，通常不会改变百分位排序；预期差异主要来自统一SNAP滤波及取消裁剪拉伸，不能仅凭换用归一化就认定精度提高。

## 1. 文件与调用关系

```text
src/thres_segmentation/main.py
  ├─ MODE="download" → auxiliary_data.py：辅助下载、缓存、逐景裁剪
  ├─ MODE="dataset" → 可选辅助准备 → scene_pipeline.py / pipeline.py：分割、分析、保存
  │    ├─ threshold.py：离线z-score百分位分割；legacy保留旧Lee近似
  │    ├─ evaluation/landcover.py：可选土地覆盖补水
  │    ├─ evaluation/metrics.py：有标签时统计整体像素精度
  │    └─ evaluation/boundary_eval.py：有标签时统计缓冲带和固定点
  └─ MODE="gui" → gui.py → gis_worker.py：本地数据处理和日志
```

独立批处理不导入 PyTorch、模型模块或 GUI 模块。GUI 保留本地数据处理界面，仍依赖原有 GDAL、OpenCV、GeoPandas、PySide6 环境；网络下载由 main.py 的独立辅助模块负责。`scripts/sar_water_extraction_esa_ui_osm_dsm.py` 是原 GUI 的兼容入口。

## 2. 运行方式

在 `src/thres_segmentation/main.py` 直接设置：

```python
IMAGE_DIR = Path("E:/Code_projects/sar_water_extraction/data/Track1/train/images/val")
LABEL_DIR = Path("E:/Code_projects/sar_water_extraction/data/Track1/train/labels/val")
SPLIT_MANIFEST = Path("E:/Code_projects/sar_water_extraction/data/Track1/train/split_manifest.csv")
BANDS = {"VV": 1, "VH": 2}
THRESHOLD_OPTIONS = dict(water_percent=10, mode="bottom", lee_window=5,
                         noise_ratio=.12, stretch=(1, 99), power_to_db=False)
```

执行：

```powershell
conda activate sar-water
python src/thres_segmentation/main.py
```

处理无标签的 Track1 test 影像时，设置 `IMAGE_DIR=Path("E:/Code_projects/sar_water_extraction/data/images")`、`LABEL_DIR=None`、`SPLIT_MANIFEST=None`。处理纯 SAR 单波段文件时，设置实际影像目录、`BANDS={"VV":1}`、`USE_LANDCOVER=False`；不要将影像不存在的第5波段作为土地覆盖读取。

当前默认参数 q=10 是显式固定的百分位比例。不会根据本次标签自动搜索最优参数或估计逐图真实水体面积。可以手工修改 q、暗/亮分支或 Lee 窗口。

默认只处理当前固定清单中 active=1 的 val，标签只用于评价。外部影像需设置 `SPLIT_MANIFEST=None`，此时处理指定目录全部影像；外部有标签评价需另行确认完整标注范围。

输出默认保存于 `outputs/thres_seg/quality/时间戳/`；也可直接设置 `OUTPUT_DIR` 为一个尚不存在的目录。默认保存最终掩膜，只绘制前10幅图；`MAX_VISUALIZATIONS=None` 绘制全部，不改变处理范围。

## 3. 阈值法实际做了什么

原 GUI 的核心是 **Lee 去斑 → 对数/百分位拉伸 → 逐图百分位阈值**，不是 Otsu。

当前适配保留 `5×5` 局部窗口、`noise_ratio=0.12` 和 P1/P99 拉伸。局部权重为：

\[
k=\frac{v_{local}}{v_{local}+0.12v_{image}+10^{-10}},\qquad
x_{Lee}=\mu_{local}+k(x-\mu_{local}).
\]

暗水体分支使用 `score < percentile(score, q)`，亮水体分支使用 `score >= percentile(score, 100-q)`。GUI 和数据集入口共用这一阈值函数。并列值可能使实际提取比例偏离 q，恒定强度图在暗水体模式下不会被强制切出 q% 水体。

DFC 适配与原地理编码场景的差别：

| 项目 | 当前 DFC 对照 | 原 GUI |
|---|---|---|
| 输入 | 同一 TIFF 中 band 1 VV、band 2 VH，分别测试 | 原单幅地理编码 SAR 流程 |
| 有效范围 | 与模型共用文件掩膜、有限值和有效标签；合法零值保留 | 原覆盖区/黑边/内部异常零值处理 |
| 强度域 | 默认保留文件存储值，`power_to_db=False` | 原线性输入执行 `10log10` |
| 去斑 | SciPy 实现原局部均值/方差近似；可设窗口1关闭 | 原 OpenCV 实现 |
| 外部资料 | 同文件 band 5 的 WorldCover，可关闭 | 原 OSM/ESA/DEM 功能保留 |
| 地形与地理矢量 | 不从无可靠空间参考的切片生成地理定位或雷达阴影 | 依赖原地理参考及成像参数 |

因此，这是一项明确记录参数的 **DFC 存储域传统阈值基线**，不声称逐位复现原 GUI 在另一种辐射产品上的结果。Lee 在存储域的使用仍是近似；未确认线性功率前，不据此推导 ENL 或物理后向散射。传统百分位法每图使用固定比例，无法可靠适应无水体图或水体占比差异极大的图，这是对照中应观察的局限。

归一化强度不是水体概率，阈值方法的点位 `probability` 列留空；网络概率只属于未修正预测。ESA 规则补水也没有新的校准概率。

## 4. 缓冲区、采样和混淆计数

默认半径 **1、3、5 像素**，按像素中心的欧氏距离计算，不将其称为米。设 G 为真值水体并集：

| 区域 | 定义 | 重点指标 |
|---|---|---|
| `inner` | 水体内距有效非水体不超过 d 的部分 | Recall：TP/(TP+FN)，评价岸线内侧漏检 |
| `outer` | 非水体内距 G 不超过 d 的部分 | FPR：FP/(FP+TN)，评价岸线外溢 |
| `core` | 水体内距有效非水体大于 d 的部分 | Recall，评价水体内部漏检 |
| `boundary` | inner 与 outer 的并集 | 严格像素 TN/FP/FN/TP、F1 |

候选位置距影像外部或无效数据必须 **大于 d**，确保其半径 d 邻域完整。NoData、裁切边缘不当成真实岸线。没有水体的影像没有上述区域；窄水体可能没有核心。空区域计数为0、无定义精度为 NaN/N/A，不填虚构的完美分数。

仅外侧区域的 GT 全是非水体，只会有 TN/FP；必须结合内侧才有 FN/TP。这里的 `boundary f1` 是 **内外带内的严格像素 F1**，不是允许 d 像素错位的容差 Boundary F1。

每张图、每个半径、每个基础区域均匀无放回抽最多256个像素中心。均匀指区域内每个候选像素等概率，并非沿矢量周长等间距取点。随机种子42，结合稳定文件名、半径及区域标识生成点位。相同标签与有效掩膜下，不同模型、处理顺序与重复执行都会取得相同坐标。

`points.csv` 保存行列、GT、各方法预测、可用概率、实际阈值、标签与有效掩膜 SHA256、形状和种子。跨次比较须确认文件标识、哈希、形状和参数一致；标签或有效范围变化后不能当成同一批点位。当前实现确定性重新生成并保存点位，不从历史 CSV 静默套用失效坐标。

## 5. 如何查看结果

独立阈值输出为 `outputs/thres_seg/quality/时间戳/`。网络训练后的质量评价仍写入各自实验的 `quality/时间戳/`，两条流程分开运行。

| 文件 | 用途 |
|---|---|
| `metrics.csv` | 有标签时，输入目录全部影像的累计像素指标 |
| `per_image.csv` | 逐图指标，定位收益与退化的场景 |
| `methods.csv`、`files.csv` | 固定阈值参数、波段及本次输入影像清单 |
| `figures/` | SAR、GT、各方法掩膜与 TP/FP/FN 图 |
| `boundary/metrics.csv` | 每方法/半径/区域的计数、比例与逐图平均 |
| `boundary/per_image.csv` | 逐图、逐区域统计 |
| `boundary/points.csv` | 相同位置下各方法的预测核查表 |
| `boundary/figures/` | 内侧黄、外侧红、核心蓝；白色叉号为采样点 |
| `masks/` | 默认保存最终 TIFF；0=非水体，1=水体，255=无效；无地理参考时不可作地理矢量定位 |

`boundary` 表内 `scope=all_pixels` 是全区域精确统计，**正式结论优先看这一行**；`sample_points` 是固定点诊断。每图定额、内外侧分别定额采样会改变面积/类别权重，其合并 F1 不等于真实面积口径。`macro_*` 是有定义的逐图指标均值，同时保存参与均值的图像数量。

推荐顺序：先确认全局 Recall 增加是否伴随 Precision 下降；再看三个半径下 inner Recall、outer FPR；最后看 core Recall 和逐图错误分布。内侧召回上升且外侧误检不增，才支持边界恢复改善。远离水体的误检仍需全图指标发现。

无标签时仅保存分割图、TIFF、方法/文件清单及逐图预测占比，不输出 metrics.csv 或 boundary/。有标签时，精度只度量与现有标签的一致性，不能证明标签正确，也不能替代独立测试。标签不会参与阈值计算，标签缺失像素也不会影响分割掩膜。

## 6. 训练输出约定

仅关闭训练中间层特征图，默认 `SAVE_FEATURE_MAPS=False`。采样明细、异常快照、最佳模型、`last.pth` 和 `metrics.csv` 仍按原规则保存。最终 quality 掩膜、错误图与指标属于最终质量成果，仍保留；训练参数与最佳模型的保存口径不因边界评价改变。旧文件不会被自动清理。已经启动的进程不会加载修改后的开关，下一次启动生效。

## 7. 独立模型评价入口

`src/supervised/evaluation/model_eval.py` 保留 single/scan 两种模式，负责整体指标或概率阈值扫描；`boundary_eval.py` 负责空间边界区域，`postprocess.py` 负责最终质量图及辅助数据修正。常规训练无需额外执行 model_eval。旧 `scripts/evaluate.py` 仅转调新入口，修改参数请在新文件顶部进行。


## 2026-10-27：统一使用固定分组后的 val

- `python src/thres_segmentation/main.py` 仍独立运行，不依赖 checkpoint。默认输入为 `data/Track1/train/images/val` 与 `labels/val`，按 `split_manifest.csv` 仅选 `split=val, active=1` 的287张；目录里的30张停用副本不会参与统计。这里路径中的第一层 `train` 是官方有标签数据池名，最终 `/val` 才是本项目内部验证分区。
- `python scripts/compare_threshold.py` 比较模型与传统法。先把顶部 `CHECKPOINT` 填为本次 grouped 训练的 `best_f1.pth`。默认不再使用历史模型；执行前核对清单 SHA256、训练名单和完整验证名单，不允许旧划分模型在新 val 上冒充未见样本评价。
- 两种方法共享287张图、标签有效域、边界缓冲带和固定采样点，输出整体 F1/IoU/Precision/Recall 以及局部边界混淆计数。对比入口不做土地覆盖补水；独立阈值入口若启用土地覆盖，额外 `_esa` 结果须与纯阈值结果区分。
- 训练数据只用于网络拟合和归一化统计；重复核查可能读取训练影像的哈希，但它们不进入对比指标。阈值分割不利用标签决定强度阈值，也不在这次验证中自动搜索最佳参数。
- val 可用于开发和选模，反复据此调参后不能称为完全独立的最终测试；最终泛化评价仍需另留未参与选模的区域/事件。

对于外部数据，独立阈值入口可设 `SPLIT_MANIFEST=None` 并指定新目录。`run_comparison(..., split_manifest=None)` 保留显式历史复核/外部评价调用；直接执行脚本始终传入当前固定清单，不回退到旧模型。

## 8. 水体面积、历史变化和 DEM/DSM 水深（2026-11-04）

`water_analysis.py` 为三个入口共用的分析模块：整景 `scene_pipeline.py`、旧切片 `pipeline.py` 和 GUI 的 `SARWorker`。分析使用最终水体掩膜，不使用评价标签作为历史水体。历史数据不参与当前提取阈值选择。

### 配置与运行

安装 `requirements.txt` 中的 `pyproj>=3.6,<4`，或在当前运行环境执行 `python -m pip install "pyproj>=3.6,<4"`。历史矢量读写使用已有依赖 Fiona。GUI 仍需要原有 PySide6、GDAL 等依赖。

- 批处理：修改 `main.py` 的 `ANALYSIS_OPTIONS` 后照常运行。面积自动统计；`reference_path=None` 不比较变化；`elevation_path=None` 不计算水深。默认两个可选路径都为空，需填入自己的文件。
- GUI：新增“面积、变化与水深”页。选历史文件、编码、有效覆盖面，按需启用水深。高程路径留空时可复用本次 DEM 地形处理的高程；只有坡度/坡向文件时必须单独提供 DEM/DSM。main 下载的 Copernicus GLO-30 按 DSM 标记，GUI 选择该文件时也应选择 DSM。
- 手动指定本地参考时，多景使用同一套分析参数；开启 main 的辅助下载后，则逐景自动准备对应范围并传入各景路径。不同参考年份仍需分别运行。`depth_scope='new_water'` 需要历史数据；若明确要估算全部水体，可设 `'all_water'`。

已有水体掩膜也可单独分析，不必重复分割（从项目根目录设置 `PYTHONPATH=src`）：

```python
from thres_segmentation.water_analysis import analyze_water

result = analyze_water(
    mask_path=r'E:\results\SARWater.tif',  # 0非水体、1水体、255无效
    output_dir=r'E:\results\analysis_new',  # 必须是新目录
    reference_path=r'E:\reference\historical_water.tif',
    reference_water_values=(1,), reference_dry_values=(0,),
    elevation_path=r'E:\terrain\dem.tif',
    elevation_kind='DEM', elevation_scale=1., depth_scope='new_water',
)
```

**历史编码必须确认：**

| 数据来源 | 水体值 | 非水体值 |
|---|---|---|
| 当前阈值输出 / 普通二值掩膜 | 1 | 0 |
| 旧版 GUI 下载的 `esa_water_mask.tif` | 0 | 1 |
| 原始 ESA WorldCover | 80 | 10,20,30,40,50,60,70,90,95,100 |

栅格 NoData、内部有效掩膜以外、未配置的类别及覆盖外均为未知，不能作为非水体。历史类别使用最近邻对齐，DEM/DSM 使用双线性对齐到当前掩膜网格。输入必须包含正确 CRS；无 CRS 的旧切片可继续分割，但面积报告 `unavailable_missing_crs`，不伪造平方米。

历史水体矢量支持 SHP、GPKG、GeoJSON 中的 Polygon/MultiPolygon（多图层 GPKG 默认读取首层；需先导出指定水体层）。所有要素都按水体面处理，不能混入行政区或非水体要素。必须设置 `reference_coverage_path` 为调查有效范围面，或明确设置 `vector_assume_complete=True`，表示水体矢量完整覆盖当前影像、面外可视为非水体。不会根据水体面包络自动猜测有效调查范围。空水体图层在已知调查范围内表示非水体。多边形按像元中心栅格化，边界存在分辨率量化误差。

### 面积和变化口径

投影网格使用仿射行列式和 CRS 两轴单位换算平方米，支持英尺及旋转网格；这属于地图投影平面面积，保留投影变形，定量比较应使用适宜的当地/等积投影。经纬度网格使用对应椭球的逐像元多边形面积，不能把平方度当平方米。

当前总水体面积按当前有效覆盖统计；以下变化面积全部限定在 **当前与历史共同有效覆盖** 内：

- 持续水体：当前水体 ∩ 历史水体。
- 新增水体：当前水体 ∩ 历史非水体。
- 消退水体：当前非水体 ∩ 历史水体。
- 净变化：新增面积 − 消退面积 = 共同覆盖内当前水体面积 − 历史水体面积。

报告另列共同覆盖面积、历史未知面积和当前总水体面积；不能把当前全覆盖面积与历史局部面积直接相减。WorldCover 属于特定时期/定义的产品，与当前水体差异不自动等于一次洪水。相邻景/重叠切片面积不可直接求和为区域总面积，需先按时间和覆盖去重或拼接。

### 水深方法与适用边界

方法参考本地 `docs/papers/isprs-annals-XI-3-2026-493-2026.pdf` 第 3.3.1 节和 `docs/papers/1-s2.0-S1569843226003286-main.pdf` 第 3.2–3.3 节。前者给出岸线高程反距离加权水位及水位减地形高程，后者强调连通水体的水位约束。

实现步骤：对当前水体做四连通分组 → 每组取一像元外侧的有效非水体岸线 → 从对齐的 DEM/DSM 取岸线高程 → 同组内用最近至多 `depth_neighbors=100` 个岸线样本做距离平方反比加权 → `depth = water_level − elevation`。距离使用以场景中心为原点的局部方位等距投影，以米为单位；大范围区域仍有投影变形，建议按水文完整区域处理。只使用历史已知非水体岸线（存在参考时），不将历史水体、无效区或影像裁切边界当作干岸线。

至少需要 `depth_min_boundary=3` 个可用岸线样本，每个像元只使用 `depth_max_distance_m=5000` 米范围内的样本。单连通域岸线超过20万点时按顺序等间隔抽样，并在组件表记录抽样前后数量；这些是可调整的工程参数，不是经本项目实测标定的参数。不同连通域独立拟合，近岸高程不跨独立水体混用。该方法仍可能在一个复杂连通域内跨越堤坝或弯曲岸线进行欧氏距离插值，不具备水动力模型的水力连通约束。

这是借鉴论文的**单时相几何估算基线**，未实现 FLEXTH Method B 的完整坡度过滤、缺口闭合、百分位策略，也未实现 KDFIMv2 的多时相补全、地形路由和淹没历时计算。区别于论文将负值归零，本实现把实质负深度设为 NoData 并标记冲突；仅绝对值不超过 `1e-9 m` 的浮点舍入误差归零。合法零水深保留。

DEM/DSM 高程须用 `elevation_scale` 转成米；重投影只处理水平坐标，不转换垂直基准。上采样到 SAR 分辨率不会提升 DEM/DSM 的地形细节或精度。DSM 中建筑、植被会影响估计，其结果是相对地表模型表面的高度差，不能直接当裸地积水深度。普通 DEM/DSM 通常缺少永久水体水底地形，因此默认只对历史已知非水体中的新增水体计算深度；没有历史数据时若选择 `all_water`，必须自行确认水下地形适用性。岸线定位、配准、地形垂直误差、桥梁/堤防和连通性误差均会传递到水深，需用同期水位或实测深度验证。

面积和变化采用窗口读写，不改变网格或把窗口当独立场景。水深为保证整景连通性，目前使用整景数组，默认 `depth_max_pixels=25000000` 防止无界内存占用，超过则明确报错；该上限仍可能占用数 GB 内存，应使用水文完整 ROI 或在确认内存容量后调整，不能无提示地切块生成局部水位。

### 输出

批处理：`analysis/<method>/<scene>/`，即使 `SAVE_MASKS=False` 也通过临时掩膜完成分析。GUI：提取输出目录下新建带时间戳的 `analysis_*`，避免覆盖历史分析。

| 文件 | 内容 |
|---|---|
| `summary.json` / `areas.csv` | 面积 m²/ha/km²、共同覆盖口径、输入参数、质量计数和方法限制；JSON仅在成功完成后写入 |
| `water_change.tif` | 0持续非水体、1持续水体、2新增、3消退、255未知 |
| `water_changes.gpkg` | 新增/消退多边形及类别、m²面积；与栅格一致，不做边界平滑；可在处理窗口处切分为多个要素 |
| `water_depth_m.tif` | 有效水深，NoData=-9999，0为合法值 |
| `water_level_m.tif` | 插值水面高程，包含负深度冲突点的水位供诊断 |
| `boundary_distance_m.tif` | 获得足够支持时距最近使用岸线点的距离 |
| `depth_quality.tif` | 0非目标、1成功估算、2高程缺失、3岸线不足、4距离内支持不足、5负深度冲突、6排除历史水体、7历史未知、255当前无效 |
| `depth_components.csv` | 各连通水体岸线样本数、估计像元数和负深度数量 |

变化产品仅在指定历史参考时生成，水深产品仅在指定高程时生成。体积、面积加权平均水深仅对质量码1的有效估算区域积分，不代表未估算部分或永久水体的总体积。

验证：`python -m unittest discover -s tests -p test_water_analysis.py -v`，另运行现有 `test_threshold_pipeline.py` 和 `test_bc_threshold.py`。合成测试检验面积/编码/NoData/边界和数值语义，不代表真实场景的水深精度验证。

## 9. main 辅助数据下载与基础 GUI

代码中的批处理阶段统一标注为 **1 数据准备 → 2 基础预测 → 3 辅助修正/水体分析 → 4 评价与输出**。
`pipeline.py` 的切片流程保留原始阈值结果及可选内置 WorldCover 补水结果；
`scene_pipeline.py` 使用全景统一阈值和分块 I/O，外部历史数据用于对比，不强制补水。
GUI 的 `gis_worker.py` 也按四阶段注释，其阶段3保持原有顺序：
填孔 → 建筑/道路/地形/农田排除 → 形态学并恢复排除区 → 连通域筛选；阶段4输出最终水体与分析成果。
这些注释不改变现有 GIS 参数或算法顺序。

监督模型现已复用这里的下载/缓存和水体分析模块，配置入口是
`src/supervised/train.py` 的 `POSTPROCESS_OPTIONS`，用法见[项目说明](../../README.md#后处理四阶段与在线辅助数据)。
两类模型的滤波/归一化均不代表已经执行 Sigma0/Gamma0 辐射校正。

辅助数据准备由 `auxiliary_data.py` 实现，通过 `main.py` 调用，不依赖 PySide6、GeoPandas、OpenCV 或 `osgeo`，也不调用 `ESAWaterWorker`/`SARWorker`。GUI 只保留本地文件选择、处理参数、运行控制和日志；已移除专用历史下载页、自动 DEM 下载选择和需要联网的 OSM/WorldCover 掩膜开关。旧 `esa_worker.py` 及 GIS 工作线程中的历史下载代码不再由基础 GUI 触发。

在 `main.py` 顶部设置：

```python
MODE = 'dataset'  # 先下载/准备，再提取与分析；改为'download'则只准备辅助数据。
AUXILIARY_OPTIONS = dict(
    download_history=True,
    download_elevation=True,
    worldcover_year=2021,
    buffer_m=1000.,
    max_tiles_per_scene=64,
    max_mosaic_pixels=100000000,
    timeout=60,
    retries=3,
)
```

仍使用同一个命令：`python src/thres_segmentation/main.py`。下载开关默认关闭，需按需开启；`MODE='download'` 至少开启一个开关。仅下载高程时不要求历史数据；`MODE='dataset'` 中的默认 `new_water` 水深则需要历史参考，缺少时在下载前报错。

- **历史水体**：支持 [ESA WorldCover 2020 v100 和 2021 v200](https://esa-worldcover.org/en/data-access)，下载对应3°瓦片并提取80类水体；不是按任意日期下载灾前 SAR 或动态水体。两个年份的算法版本不同，年份差异也可能含分类方法差异。
- **高程**：下载 [Copernicus GLO-30 公共 COG](https://registry.opendata.aws/copernicus-dem/) 的1°瓦片，作为 DSM、米制高程传给水深模块；不作为裸地 DTM 或水下测深数据。
- **本地优先**：`ANALYSIS_OPTIONS['reference_path']` / `['elevation_path']` 已填有效本地文件时，对应下载被跳过。本地文件不存在会明确报错，不静默替换为远程产品。
- **逐景范围**：按实际 `active=1, split=val` 清单和 `SCENE_PREFIX` 选择各景；外部目录模式与分割使用相同扩展名。每景分别读取 CRS 和包围框，避免仅取第一景或把相隔很远的景拼成巨幅辅助影像。影像包围框内部的 NoData 由后续分析有效掩膜排除。
- **自动接入**：下载后为各景提供 `scene_analysis_options`，直接连接历史比较及水深模块，无需手动复制文件名。不同极化方法共享该景的辅助数据，不重复下载。

缓存目录由 `AUXILIARY_CACHE_DIR` 控制：

```text
data_cache/auxiliary/
  tiles/worldcover/       原始WorldCover瓦片和.source.json来源记录
  tiles/copernicus/       原始Copernicus瓦片和.source.json来源记录
  scenes/<范围与产品标识>/worldcover.tif 或 copernicus.tif
                          对应.manifest.json记录源文件、范围及输出签名
  batch_<标识>.json       成功准备后的逐景输入、产品、URL及analysis_options清单
```

新 `worldcover.tif` 统一为 **1水体、0非水体、255未知**，与当前提取结果一致；原始类别0、NoData及未识别类别均保留为未知。旧 GUI 的 `esa_water_mask.tif` 仍是0水体/1非水体，手动引用时必须选择旧编码。

原始瓦片以流式方式下载到唯一临时文件，校验 HTTP 长度、TIFF签名、CRS及角部栅格块后原子替换；HTML/XML错误页不能作为成功数据。缓存命中需来源URL、文件大小、修改时间匹配并通过栅格检查，下载时计算 SHA256 记录在来源文件中（不等同于服务端发布校验和）。派生裁剪按源文件签名复用，使用最细源网格角分辨率、窗口拼接，不把整瓦片读入 NumPy 内存。Copernicus 在高纬度可能含变宽经度像元，重采样不提升其真实分辨率。

某个瓦片404、访问失败或文件不完整时，准备任务报错，不将缺失瓦片默认为陆地或零高程，也不发布本次成功清单；此前成功的瓦片缓存仍可复用。没有断点续传，失败重试从该瓦片起点重新下载。跨180°经线场景需要先拆分；场景瓦片数和裁剪像元数有上限，避免错误范围触发大规模下载。缓存需预留原始瓦片与逐景辅助影像磁盘空间。

验证入口：`python -m unittest discover -s tests -p test_auxiliary_data.py -v`。下载回归使用本地真实GeoTIFF内容和受控HTTP响应，覆盖缓存复用、错误页、截断、重试、取消、两期编码、拼接、逐景选择及 main 模式传递；实际下载仍依赖网络和服务端数据可用性。
