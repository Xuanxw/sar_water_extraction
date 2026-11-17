# 北京密云区校正幅度ORG水体提取

当前输入：`E:/InSAR/水体提取/北京密云区/曾老师校正结果`。已检查17景GeoTIFF及同名XML；影像均带AMPLITUDE标记，配套XML可解析原始雷达网格与轨道。XML的SLC/Complex字段描述原始输入，当前GeoTIFF按线性幅度处理，保留其0值NoData定义。

## 两组运行配置

| 配置 | 阈值 |
|---|---|
| `corrected_amplitude_base30_autoheight.yaml` | 滤波后幅度的暗侧Q30 |
| `corrected_amplitude_otsu_autoheight.yaml` | 对数幅度Otsu，失败回退Q30 |

```powershell
conda activate sar-water
python src/thres_segmentation/main.py configs/北京密云区/corrected_amplitude_base30_autoheight.yaml
python src/thres_segmentation/main.py configs/北京密云区/corrected_amplitude_otsu_autoheight.yaml
```

建议逐组运行并观察各自processing.log。两组的阈值阶段缓存、几何缓存和成果独立，既不混用彼此统计，也不复用旧功率/旧地理网格的处理结果。

成果分别位于：

```text
E:\InSAR\水体提取\北京密云区\outputs\thres_seg_corrected_amplitude_base30_autoheight_20261116
E:\InSAR\水体提取\北京密云区\outputs\thres_seg_corrected_amplitude_otsu_autoheight_20261116
```

## 共同处理方式

裁剪至行政区外扩范围→幅度平方→功率域SNAP Lee→恢复幅度→整景阈值→土地覆盖及剖面地形修正→面清理→逐景water_polygon.tif/shp、质量JSON与快视图。不额外执行辐射校正或归一化，不自动合并日期。

Base Q30表示第30百分位，不是幅度值30或30%的精度。Otsu使用20log10(A)域，最终阈值还原到线性幅度；参数、实际方法和回退原因写入质量记录。

20公里支持域用于周边山体的阴影/叠掩计算，输出水体仍由ROI控制。每景先调查潜在同斜距竞争地形的DEM覆盖，再扫描正高H及大地水准面N，采用h=H+N，加裕度并向外取整；不固定使用密云上一景的[-500,2500]米。区域包络记录见每景几何缓存的regional_height_envelope.json。

DEM不足时按景补充共享瓦片；已有完整瓦片和极值缓存复用。缺测或基准不符时明确停止，不猜填。未知255保留。原生分辨率不同并不会改变这一输入约定，空间对齐以TIFF实际CRS/transform为准。

## 续跑与历史对照

两组均启用resume，使用同一配置重跑可复用本组完整阶段。新数据的SNAP缓存独立于旧功率输入；公共DEM、土地覆盖瓦片继续共用。

旧功率ORG、旧水体成果和分析结果保留。已完成的单景支持域对照参数归档于codex_analysis/terrain_support_0730_20261116，复核入口为该目录replay.yaml。本目录删除5份已被替代的功率/临时配置，保留三份sar_geometry_visibility*.yaml，供独立几何核验工具使用，不作为这批幅度水体的生产入口。

当前地形模型仍保留零Doppler约定尚未独立验证的标记。几何校正、包络完善和无真值质量诊断不能代替人工水体精度核验。quality.json统计面清理前掩膜；最终成果以water_polygon.tif/shp为准。

## 三景坡度风险约束对照

配置：`corrected_amplitude_otsu_slope_guard15_3scenes.yaml`。仅选择7月25日135519、7月27日025242、8月5日143332，分别检查水库恢复与山区误提。

```powershell
python src/thres_segmentation/main.py configs/北京密云区/corrected_amplitude_otsu_slope_guard15_3scenes.yaml
```

`threshold.slope_guard.enabled: true`开启独立风险约束；坡度大于`slope_deg: 15.0`时，暗侧阈值采用`min(T_Otsu, T_Q30)`。坡度等于15°或缺测时保留原Otsu，Otsu已经更严格时不反向放宽。这个规则不等于确认地形阴影或叠掩。

其他科学参数与当前Otsu保持一致，成面时重新应用风险约束，防止填孔/开闭填回。实际阈值和收紧面积写入`quality.json`的`extraction`；`slope_threshold_rejection.tif`随前置产品归档到新缓存，1=风险排除、0=未排除（含坡度未知）、255=SAR无效。缺测坡度另有面积统计及`slope.tif`。

新成果：

```text
E:\InSAR\水体提取\北京密云区\outputs\thres_seg_corrected_amplitude_otsu_slope_guard15_3scenes_20261117
```

`REUSE_RESULTS_DIR`只读引用上一轮完整的裁剪/Lee/统计/几何输入，不导入旧水体分类。新实验用自己的缓存和progress续跑。SAR通过VRT引用原缓存，因此运行和续跑时须保留原实验缓存；VRT不是独立像元备份。其余三景的原Q30和原Otsu成果保留作对照，无须再运行。

本配置没有启动真实三景处理；面积减少不能直接解释为精度提高，需重点核验岸线和新增山区候选。

## 全部ORG：细网格地形与15°坡度约束

配置：`corrected_amplitude_otsu_slope15_fine_geometry.yaml`。扫描校正结果目录中的全部ORG，不限制日期或景数。

```powershell
conda activate sar-water
python src/thres_segmentation/main.py configs/北京密云区/corrected_amplitude_otsu_slope15_fine_geometry.yaml
```

本轮保持幅度域SNAP Lee、对数幅度Otsu及失败回退Q30、坡度大于15°采用较严格阈值、土地覆盖修正和成面参数。地形计算依次调整为：

1. `scene.resolution_m: 5`：在5米网格重新求解地形点的RD坐标，再查询完整剖面类别，替代旧30米类别图放大。
2. `scene.dem_resolution_m: 30`：支持域DEM按30米存储，在求解位置双线性采样；20公里支持域与逐景自动高程包络不变，不生成覆盖整个支持域的巨型细DEM。
3. `scene.refine_sar_boundaries: true`：阴影、叠掩及未知类别的转折带在SAR原生像元中心重算RD；稳定核心最近邻对齐。完整剖面保持7.5米采样，未知255不填充，也不扩张阴影核心。
4. 继续使用本轮15°坡度风险约束，处理地形模型判为可见、但低散射可能被Otsu误提的陡坡；不将所有陡坡直接删除。

`execution.native_block_size: 256`限制原生边界复核的临时数组。边界候选图保存在各景几何缓存中的`visibility_transition.tif`，1仅表示需要重算，不是排除类别。掩膜最终与SAR的CRS、尺寸和transform一致。5米数值网格及原生像元复核不意味着源DEM获得了5米实测精度。

成果目录：

```text
E:\InSAR\水体提取\北京密云区\outputs\thres_seg_corrected_amplitude_otsu_slope15_fine_geometry_20261117
```

几何缓存和阶段缓存使用本轮独立目录。`REUSE_RESULTS_DIR`只读引用上一轮Otsu的裁剪、Lee及阈值统计；不会复用旧地形、水体栅格或矢量。必须保留原实验缓存供VRT读取。中断后用同一命令续跑，复用本轮完整阶段及已完成剖面/映射行；原生边界阶段未完整完成时重做该阶段。

每景仍输出`water_polygon.tif/shp`、`quality.json`及快视图，不自动合并日期。`processing.log`显示阶段耗时和原生复核进度，几何缓存的`geometry_quality.json`记录网格、剖面、重算点数及180秒目标是否满足。目标指几何计算与边界复核，不含辅助下载、SNAP和水体成面；尚未完成本轮全部整景性能验收。

## OSM道路与水体冲突核验

独立配置：`corrected_amplitude_otsu_slope15_fine_geometry_roads_review.yaml`。

```powershell
conda activate sar-water
python src/thres_segmentation/main.py configs/北京密云区/corrected_amplitude_otsu_slope15_fine_geometry_roads_review.yaml
```

保持前一份全部ORG配置的5米几何、原生边界复核、15°坡度规则及水体成面参数。新增步骤只读取水体，输出道路冲突供核验，不删除、不修改水体。成果和新计算使用独立缓存，运行日志中`02R`为共享路网准备，`08R`为逐景冲突核验。

`REUSE_RESULTS_DIR`指向上一轮5米基线：按景只读复用已完成的裁剪、Lee、阈值统计及相同配置的地形掩膜。参考实验尚未完成的阶段在本轮独立计算，不能读取正在生成的掩膜，也不要求固定景数全部完成。原水体分类/矢量不导入。保留参考实验及其VRT依赖缓存；本轮不修改它们。

新成果目录：

```text
E:\InSAR\水体提取\北京密云区\outputs\thres_seg_corrected_amplitude_otsu_slope15_fine_geometry_roads_review_20261117
```

所有景和日期共用`paths.OSM_CACHE_DIR`：

```text
E:\InSAR\水体提取\北京密云区\data_cache\auxiliary\osm_roads
```

路网通过Overpass按行政区及上下文范围请求固定经纬度瓦片，串行下载，按OSM道路编号去重。完整瓦片原子保存，失败或中断的请求不登记完成；再次运行只补未完成瓦片。服务端超时、返回部分数据或网络错误不能作为“无道路”继续处理。缓存无自动过期；需要更新OSM快照时使用新的OSM缓存目录。改变缓冲宽度只重建缓冲，不重新下载相同类别的瓦片。

共享`network_*.gpkg`包含`road_centerlines`与`road_buffers`两层，保留`osm_id/highway/name/width_tag/width_m/width_source/bridge/tunnel/surface`字段。桥梁和隧道仍参与冲突标记，不作删除。下载的是路网快照，并非2025年各观测日的同期真值，也不保证乡村小路完整。

参数集中在本YAML的`road_review`：

| 参数 | 含义 |
|---|---|
| `enabled` | 是否输出道路冲突核验；没有道路删除模式 |
| `highway_classes` | 请求的OSM道路类型，默认包括主要道路、乡村道路和步道 |
| `tile_degrees` | 共享下载瓦片的经纬度跨度，不是道路分辨率 |
| `context_buffer_m` | 行政区外围的路网上下文距离，米 |
| `width_by_class_m/default_width_m` | 缺少可解释OSM宽度时采用的道路总宽假设，米；不是实测宽度 |
| `min_width_m/max_width_m` | 接受OSM宽度的范围；无法解析或超范围时用道路等级宽度 |
| `extra_buffer_m` | 道路两侧各增加的配准容差，米；实际缓冲半径为总宽/2加此值 |
| `endpoints/query_timeout_seconds/http_timeout_seconds/retries` | 服务地址、查询/网络超时及重试控制 |
| `block_size/preview_size` | 分块边长与快视图最长边，像元 |

每景新增`road_review`子目录：

| 文件 | 内容 |
|---|---|
| `road_corridor.tif` | 道路缓冲：1命中、0无已记录道路命中、255共同有效域外 |
| `road_conflict.tif` | 三波段冲突：波段1原阈值候选、波段2辅助修正后、波段3最终成面；每波段均为0/1/255 |
| `road_conflicts.gpkg` | `raw_threshold_conflicts`和`final_polygon_conflicts`核验面，按块生成，不做几何并集 |
| `road_conflicts.png` | 左：SAR与橙色道路缓冲；中：原候选蓝色、冲突洋红色；右：最终水体蓝色、冲突洋红色 |
| `road_review.json` | 三阶段原生网格冲突面积、像元数、路网来源及文件说明 |

三阶段面积采用同一有效范围；主`quality.json`也记录`road_review`摘要。快视图对细线采用max聚合以免缩小后消失，不能从PNG量算冲突面积。核验面可能在分块边界处分段，不能按面数量统计独立错误。

优先检查最终仍保留的洋红色区域，以及桥梁跨河、库岸道路和疑似道路积水。冲突不等于误检，路网空白不等于已确认非道路。本轮没有真值精度或误删率结论。

本功能使用已有Fiona、Shapely和Rasterio，不需要GeoPandas、requests或QGIS。输入协议参考[Overpass QL](https://wiki.openstreetmap.org/wiki/Overpass_API/Overpass_QL)，OSM宽度默认单位为[米](https://wiki.openstreetmap.org/wiki/Key:width)。缓存、JSON和图件保留“© OpenStreetMap contributors”及[ODbL许可说明](https://www.openstreetmap.org/copyright)。

## 每景处理过程快视图

`quicklook.png`改为多面板过程对照。道路核验配置通常包含15个面板，按实际启用的规则增减：

- 裁剪后、Lee前的SAR；滤波后阈值输入；纯阈值水体候选；辅助修正及分割阶段形态学后的水体。
- 土地覆盖排除、雷达阴影、叠掩、地形联合排除、陡坡阈值约束等实际启用的掩膜；历史水体启用时另显示先验。
- OSM道路缓冲，明确标注`REVIEW ONLY`，不表示已删除道路范围水体。
- 最终清理后`water_polygon.tif`，以及读取实际`water_polygon.shp`生成的面矢量缩略图。
- 原候选→辅助修正、辅助修正→清理后栅格、原候选→最终清理后三组变化。

纯阈值候选来自Lee后的SAR阈值比较，尚未应用土地覆盖、地形、坡度风险、历史恢复及形态学。它不是未经滤波的原始SAR；两者在图中分开显示。SHP面板读取包含孔洞的实际几何，临时简化和显示栅格化不会写回SHP；缩略尺度下很小的孔洞可能无法分辨。矢量平滑边界可与清理后TIF稍有差别，变化面积按三个TIF阶段计算，不混入矢量显示差异。

颜色：蓝色为水体；掩膜面板红色为排除范围、浅黄色为未知；道路橙色仅核验。变化面板红色为删除、绿色为新增、黄色表示同一缩略单元内同时含有删除和新增。掩膜可能重叠，不能把不同掩膜覆盖面积直接相加当作删除面积。

变化先在原生网格逐块比较，再聚合到缩略图；未知不当作非水体。标题的`- / + km2`是对应阶段实际删除/新增的原生像元面积。二值层用max聚合保留细水域，不从PNG量算面积。SAR的P2–P98拉伸只用于显示，不修改像元。

新处理会保存`landcover_exclusion.tif`至阶段缓存。旧产品缺少该文件时，仅从同一WorldCover缓存和排除类别重建显示，并在面板注明`reconstructed`；其他缺失图层标为`Not available`，不伪造全0掩膜。

当前独立道路配置增加以下显示参数：

```yaml
output:
  preview_size: 1200           # 原有成果展示图尺寸
  process_preview_size: 700   # 过程图每层读取的缩略数组最长边
  refresh_quicklook: false    # true强制只重绘过程PNG，补完后可改回false
```

保持`org.save_quicklook: true`，仍用原命令运行。新景自动生成过程图；已有完整景会识别旧PNG格式并自动补图，不使分割、几何或矢量阶段失效。已是新版且尺寸相同时复用；需要手动重绘时开启`refresh_quicklook`。缓存及VRT引用的前置数据需仍可读取。这里没有启动真实整批补图，执行由使用者控制。
