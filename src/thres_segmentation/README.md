# 独立阈值分割

处理参数由入口`CONFIG_PATH`指定的YAML控制。通用离线val配置为`configs/threshold_segmentation.yaml`，本次密云ORG配置独立放在`configs/北京密云区/threshold_segmentation.yaml`。`main.py`只负责读取配置和调用流程，不再读取 `image_process.yaml`，也不依赖训练模型或checkpoint。旧的 `threshold_quality.yaml` 已合并删除。

## 启动

在仓库根目录执行：

```powershell
conda activate sar-water
python src/thres_segmentation/main.py
```

配置按以下分组组织：

| YAML分组 | 用途 |
|---|---|
| paths | 输入影像、可选标签、清单、缓存、输出目录、历史参考和高程路径 |
| run | dataset分割、download辅助下载、gui旧界面；full_scene整景或patch切片 |
| input | 波段编号、景号前缀筛选 |
| threshold | water_percent、暗/亮侧、输入预处理状态 |
| processing | 读取分块大小 |
| auxiliary | 历史水体/高程下载开关、年份、范围外扩、大小限制、超时和重试 |
| landcover | 旧切片的内置土地覆盖波段修正 |
| analysis | 历史类别编码、参考覆盖语义、水深参数 |
| output | 保存mask、快视图尺寸/数量、边界评价半径 |
| quality_review | 质量报告、分层随机核验点和风险区域 |

所有文件路径集中在`paths`，使用绝对路径。`LABEL_DIR: null`表示没有参考标签；`OUTPUT_DIR: null`表示在`OUTPUT_ROOT`下新建时间戳目录。YAML中的`null`相当于Python的None，开关使用true/false。

## 通用离线val数据约定

通用配置的输入是已准备好的BC整景val；密云专项入口不使用这些参数。`full_scene`要求`IMAGE_DIR`为清单对应的`images/val`，只处理`active=1, split=val`记录。`LABEL_DIR`可为空。波段编号从1开始，`source_image`用于匹配原始GeoTIFF/ZIP及严格同名XML，不猜用第一份XML。

`offline_normalized`要求输入已有完整离线预处理记录：ORG幅度平方→SNAP Lee功率域滤波→恢复幅度→统一训练统计z-score。本入口不再滤波、标准化或调用SNAP；负z-score及合法0保留。预处理需先单独运行image_process，已经完成则无需重复。

`water_percent: 10`是整景暗值侧第10百分位，不是Otsu、概率0.1或精度目标。`mode: bottom`取暗侧，`top`取亮侧。分块只限制内存，不逐块计算不同阈值。精确百分位使用临时磁盘缓冲，所需空间与有效像元数有关。

BC整景没有内置土地覆盖波段，`landcover.enabled`必须关闭。`auxiliary`下载的历史水体/DSM当前用于面积变化和几何水深分析，并不自动修正分割。原始WorldCover水体类为80；下载器生成的二值历史mask使用1水体/0非水体，参考编码需要与输入一致。

主输出包括`masks`、`figures`、`metadata.csv`、`methods.csv`、面积分析和质量核查目录。有完整参考标签时才输出精度；快视图缩放不影响原像元统计。没有真值时不会生成可解释为实测精度的F1/IoU。

## ORG逐景生产与旧GUI

`run.mode: gui`仍打开旧界面，旧GUI的影像提取参数由界面设置；质量参数共用统一YAML。旧GUI不是本次YAML命令行批处理，不应直接用于多景目录自动配对。

`run.mode: org_scenes`独立处理原始ORG，递归检查最终ZIP/TIFF、匹配原始XML、裁剪AOI外扩范围，再执行SNAP Lee、整景分位阈值、土地覆盖/坡度/历史水体候选修正。没有真值不计算F1/IoU；每景输出水体SHP和质量JSON，不自动按日期合并。

密云入口已指向专项配置，参数和操作见[密云案例说明](../../configs/北京密云区/README.md)。ORG流程以data_cache中的逐景progress.json记录阶段；产物使用相对路径，支持迁移缓存和成果，不再用全局/逐景路径签名拦截。resume=true复用已完成且文件存在的阶段；修改输入或参数后由用户设置resume=false重算。辅助瓦片按AOI共享，计算采用分块读写。独立合并工具为`scripts/merge_water_shp.py`。

ORG配置只保留`paths/run/input/threshold/processing/output/quality_review/org`，不混入训练清单、标签或水深配置。`org`包含滤波、归一化表达、辅助下载和修正参数。`scene_zscore`仅使用等价阈值，不落盘另一份归一化图；它不会改变线性幅度分位数分类。

## 无真值的每景质量核查包

`configs/threshold_segmentation.yaml`中的`quality_review` 控制质量输出，默认启用。BC整景命令行在mask保存后输出；旧GUI在后处理矢量输出后输出。本模块不修改提取结果，不计算没有参考标签支持的F1/IoU。

ORG路径：`输出目录/日期/影像编号/quality.json`和`review/review.gpkg`。离线val路径：`输出目录/quality_review/threshold_VV/影像编号/`。旧GUI路径：`输出目录/quality_review/时间戳/`。

- `quality.json`：有效范围、水体面积、占比、抽样层规模、风险统计及尚未检查的项目。
- `review.gpkg` 的 `verification_points`：水体内部、岸线内外缓冲带、其余非水体三层随机核验点，默认每层最多50点。
- `review.gpkg` 的 `suspected_regions`：水体邻近未知区，以及有对应辅助输入时的历史差异、陡坡水体、后处理变化。风险面是定位提示，不是已确认错误；分块面不代表独立水体对象。

QGIS中打开GeoPackage选择两个图层即可。人工填写点属性 `ref_class`：0非水体、1水体、-1无法判断；尚未检查保持NULL。`confidence`与`note`可记录判读把握和依据。保留`scene_id/stratum/row/col/population/sample_n/incl_prob/weight`，不要删除不确定点或自行改变点位。入样概率为每层实际点数/候选像元数，权重为其倒数；经纬度网格的面积推断还需像元面积加权。

默认岸线缓冲为5个原始像元，三层互斥且覆盖全部已知0/1像元；NoData不抽样。相同栅格、分块大小及seed可复现点位。分块读取控制内存，结果统计保持原像元精度。抽样配置不是样本量充分性的保证。

`quality_review.export_quality_review`接受已经对齐的`history_path`（0/1）、`slope_path`（度）、`raw_mask_path`（0/1）。缺失时对应检查写`not_available`和NULL，不写成零风险；不根据历史数据推断当期真值。ORG入口已传入对齐后的历史参考、坡度和原始mask；旧流程仅在明确传入时使用这些辅助检查。

质量模块本身统计mask范围；ORG入口额外统计行政区有效覆盖面积比例，并以分位数q±配置范围检查修正后水体面积敏感性。跨景重叠一致性、时序变化和雷达阴影/叠掩几何仍未评估。无覆盖不是非水体，疑似错误不等于真实错误。
