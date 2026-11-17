# 独立阈值分割

处理参数由命令行第一个参数指定的YAML控制，不再修改源码选择配置。通用离线val配置为`configs/threshold_segmentation.yaml`，本次密云ORG配置独立放在`configs/北京密云区/corrected_amplitude_otsu_autoheight.yaml`。`main.py`只负责读取配置和调用流程，不再读取 `image_process.yaml`，也不依赖训练模型或checkpoint。旧的 `threshold_quality.yaml` 已合并删除。

## 启动

在仓库根目录执行：

```powershell
conda activate sar-water
python src/thres_segmentation/main.py configs/threshold_segmentation.yaml
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

`threshold.method`选择`otsu`或`percentile`；旧配置未指定时继续使用百分位，保持已有实验可复现。`mode: bottom`取暗侧，`top`取亮侧。分块只限制内存，整景共用一个阈值。精确百分位使用临时磁盘缓冲，所需空间与有效像元数有关。

### Otsu优先、百分位回退

当前密云校正幅度ORG的独立配置为`configs/北京密云区/corrected_amplitude_otsu_autoheight.yaml`（逐景自动高程包络）：

```powershell
python src/thres_segmentation/main.py configs/北京密云区/corrected_amplitude_otsu_autoheight.yaml
```

```yaml
threshold:
  method: otsu
  otsu_domain: log_amplitude
  water_percent: 30
  mode: bottom
  otsu_bins: 1024
  otsu_min_separability: 0.5
  otsu_min_class_fraction: 0.001
  otsu_histogram_percentiles: null  # null保持原min/max；例如[0.1,99.9]限定统计范围。
```

默认linear模式在滤波后、行政区内的有效原输入域累计等宽直方图；当前密云校正幅度配置使用log_amplitude。两者均通过最大化类间方差选择阈值，不执行辐射校正。这里用NumPy累计直方图和计算方差，无新增依赖；算法定义可参见[scikit-image官方说明](https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_otsu)。离线归一化整景入口则在已有z-score域计算，旧legacy入口在原有0～1决策域计算。

以下情况回退为原百分位方法：常量影像、直方图无法形成两个非空类别、非有限候选阈值、类间方差/总方差小于0.5，或任一类别占比小于0.1%。后两项为可调统计判据，不能证明两类对应水体/非水体，也不能识别所有错误分割。没有有效SAR像元、文件读写失败、内存不足或配置错误时直接报错，不产生伪成果。

`water_percent: 30`仅指定回退Q30，不限制Otsu必须提取30%的像元。密云配置关闭历史放宽，保持两组一致的幅度Lee、自动包络及剖面地形设置、土地覆盖及面清理参数；Otsu成功和回退均使用相同后处理。结果写入独立`thres_seg_corrected_amplitude_otsu_autoheight_20261116`目录，复用本组完整阶段并共享公共辅助瓦片，支持中断后续跑。修改算法参数时另设成果及阶段缓存目录，避免复用旧分类结果。

日志输出每景实际方法、阈值和回退原因；ORG的`quality.json → extraction → statistics → threshold_selection`保存候选Otsu阈值、Q30阈值、可分性及两类占比。离线评价的`per_image.csv`也记录相应字段。Otsu成功时，敏感性面积使用候选阈值±一个直方图bin宽度；回退时使用Q30±原配置百分位差，`sensitivity_basis`区分两者。敏感性、面积变化和可分性均不是真值精度。

BC整景没有内置土地覆盖波段，`landcover.enabled`必须关闭。`auxiliary`下载的历史水体/DSM当前用于面积变化和几何水深分析，并不自动修正分割。原始WorldCover水体类为80；下载器生成的二值历史mask使用1水体/0非水体，参考编码需要与输入一致。

主输出包括`masks`、`figures`、`metadata.csv`、`methods.csv`、面积分析和质量核查目录。有完整参考标签时才输出精度；快视图缩放不影响原像元统计。没有真值时不会生成可解释为实测精度的F1/IoU。

## ORG逐景生产与旧GUI

`run.mode: gui`仍打开旧界面，旧GUI的影像提取参数由界面设置；质量参数共用统一YAML。旧GUI不是本次YAML命令行批处理，不应直接用于多景目录自动配对。

`run.mode: org_scenes`独立处理原始ORG，递归检查最终ZIP/TIFF、匹配原始XML、裁剪AOI外扩范围，再执行SNAP Lee、整景分位阈值、土地覆盖/坡度/历史水体候选修正。没有真值不计算F1/IoU；每景输出水体SHP和质量JSON，不自动按日期合并。

密云案例通过命令行选择专项配置，参数和操作见[密云案例说明](../../configs/北京密云区/README.md)。ORG流程以data_cache中的逐景progress.json记录阶段；产物使用相对路径，支持迁移缓存和成果，不再用全局/逐景路径签名拦截。resume=true复用已完成且文件存在的阶段；修改输入或参数后由用户设置resume=false重算。辅助瓦片按AOI共享，计算采用分块读写。独立合并工具为`scripts/merge_water_shp.py`。

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

质量模块本身统计mask范围；ORG入口额外统计行政区有效覆盖面积比例，并以分位数q±配置范围检查修正后水体面积敏感性。跨景重叠一致性、时序变化和沿视线的远处遮挡仍未评估。无覆盖不是非水体，疑似错误不等于真实错误。

### 水体面成图

ORG批处理保留原分割栅格`water.tif`，水体矢量仅输出清理后的`water_polygon.shp`，并配套`water_polygon.tif/json`；不再生成旧碎斑`water.shp`。
`polygon_output`配置依次控制：内部小孔填充、开闭运算、重新施加地形与土地覆盖排除、整景8连通小斑块筛除、面边界简化和平滑。
默认孔洞上限200像素、3×3核、最小水体200像素、Chaikin平滑2次；小水体可能被删除，平滑后的面边界与栅格略有差别。
`water_polygon.json`记录本次清理参数和数量；原`quality.json`仍对应原`water.tif`，不能当作新面成果的精度报告。
已有成果可执行`python scripts/export_water_polygons.py 配置.yaml`补做，复用本地辅助缓存，不重新滤波或阈值分割。处理完成的景可续跑跳过。
`merge_water_shp.py`拼接逐景`water_polygon.shp`；输入内容更新后自动重做日期合并，原始逐景结果不变。

### 原GUI地形排除

`org.postprocess.terrain_method: gui_geometry`使用本景XML的`platformHeading`、`incidenceAngleMidSwath`和`look_side`。`paths.GDALDEM_PATH`指定GDAL工具。DEM在米制网格使用ZevenbergenThorne计算坡度、坡向，再分别以双线性、最近邻对齐到SAR网格。

`terrain_class.tif`：0=DEM未知，1=局部阴影，2=叠掩，3=迎雷达高坡/透视收缩，4=正常，255=SAR无效。`terrain_exclusion.tif`中1表示排除类1/2/3；DEM未知处不删除水体。判据逐式沿用`gis_worker.py`，不等价于GAMMA完整几何模拟。

`terrain_geometry.json`保存每景角度和类别数量。水体分割与成面填孔/形态学之后都重新施加该掩膜，避免填回已排除区域。旧批处理只有全坡向坡度阈值，新的源码方案只在迎雷达方向判定高坡收缩，因此面积可能增加或减少，不能仅据面积断定精度提高。

`org.archive_intermediates: true`在完成后将原分割栅格、坡度、历史水体、几何分类/掩膜和源元数据放入`CACHE_DIR/scenes/景号/products`，同步更新进度和JSON源路径。成果目录保留最终`water_polygon.tif/shp`、质量记录及快视图。续跑执行时临时恢复工作输入，结束后重新归档，不生成before_gui_terrain备份。

### 正式入口与成果一致性

密云案例执行 `python src/thres_segmentation/main.py configs/北京密云区/corrected_amplitude_otsu_autoheight.yaml`，使用该YAML中的参数；更换案例只更换命令行配置路径。
正式流程依次完成XML几何读取、配置选择的地形掩膜、分割与辅助修正、成面清理及最终TIFF/SHP输出。
形态学清理后重新施加地形排除，避免填孔或闭运算把排除区域重新填成水体。

`quicklook.png`显示最终`water_polygon.tif`，`terrain_quicklook.png`显示同一最终水体及地形掩膜。
`quality.json`仍统计成面清理前的分割掩膜，成面后的数量见`water_polygon.json`。
`scripts/repair_gui_terrain.py`只转调正式入口，不再维护另一套处理算法；按日期合并仍单独运行。
保持输入、辅助数据和配置一致即可复现同一处理结果；修改参数后需设置`resume: false`重新计算，
不能把旧参数的已完成缓存当作新结果。

`otsu_histogram_percentiles`只限制直方图范围：范围外样本计入首尾bin，不删除有效像元，不改写原始数值，最终阈值仍应用于完整有效影像。ORG和离线整景入口通过磁盘缓冲计算精确分位数；阈值及受限样本数写入threshold_selection。范围退化时按原规则回退百分位。

### 新校正幅度ORG与自动高程包络

当前密云输入位于`E:/InSAR/水体提取/北京密云区/曾老师校正结果`，为17景单波段Float32线性幅度GeoTIFF。同名XML保留原始SLC的雷达网格和轨道；其中Complex描述SLC，当前栅格按用户确认和TIFF的AMPLITUDE标记配置为amplitude。程序读取TIFF真实地理网格及NoData，不用XML雷达尺寸替代。

`org.filter.input_domain: amplitude`执行幅度平方→功率Lee滤波→恢复幅度。Base30在滤波后幅度上取暗侧第30百分位；Otsu在`log_amplitude`即20log10(max(A,floor))上估计，再恢复到幅度单位分类，失败回退同一幅度域Q30。Otsu可分性不是识别精度。

两组均启用20公里支持域和`height_envelope.mode: regional_dem`，根据每景潜在同斜距地形所需DEM计算高程包络。案例路径、运行命令和结果目录见`configs/北京密云区/README.md`。旧功率案例配置已清理，已有成果内的参数快照保留，不作为新幅度输入入口。
