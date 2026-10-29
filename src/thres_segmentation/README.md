# 独立阈值分割与水体边界质量评价

> **Original Author: Yanan Ji, 2026**

`src/thres_segmentation` 是独立的传统阈值处理流程：直接读取影像，输出分割成果；可选标签仅用于评价。它不加载模型、不读取 checkpoint 中的数据划分、不执行网络对比或自动选参。

## 1. 文件与调用关系

```text
src/thres_segmentation/main.py
  ├─ MODE="dataset" → pipeline.py：栅格读取、分割、结果保存
  │    ├─ threshold.py：Lee 近似滤波与百分位分割
  │    ├─ evaluation/landcover.py：可选土地覆盖补水
  │    ├─ evaluation/metrics.py：有标签时统计整体像素精度
  │    └─ evaluation/boundary_eval.py：有标签时统计缓冲带和固定点
  └─ MODE="gui" → gui.py → gis_worker.py / esa_worker.py
```

独立批处理不导入 PyTorch、模型模块或 GUI 模块。原 GUI 功能保留，仍依赖原有 GDAL、OpenCV、GeoPandas、PySide6 环境。`scripts/sar_water_extraction_esa_ui_osm_dsm.py` 是原 GUI 的兼容入口。

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
