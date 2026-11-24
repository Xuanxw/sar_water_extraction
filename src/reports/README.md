# 洪涝遥感专题图生成模板

用同一案例配置生成水面变化、SAR 观测覆盖、辅助判读、道路与建筑潜在暴露四张专题图。输出 A3 PNG 和 PDF，同时生成设施 GeoPackage 和统计 JSON。制图借鉴 UNOSAT 的主图、定位小图、局部放大、旁栏统计、图例及数据来源编排，地图署名和产品编号由项目自行管理。 [UNOSAT 巴基斯坦 Swat 设施暴露产品](https://unosat.org/static/unosat_filesystem/4177/UNOSAT_A3_Natural_Landscape_FL20250818PAK_SwatDistrict_20August2025.pdf) · [UNOSAT 尼泊尔两期 Sentinel-1 产品](https://unosat.org/static/unosat_filesystem/3990/UNOSAT_A3_Natural_Protrait_FL20240928NPL_01Oct2024.pdf)

## 统一入口

从仓库根目录运行。地理处理使用项目主依赖。生成 Word、Excel 或指定纸张 PDF 时，安装报告扩展依赖。

```powershell
conda activate sar-water
python -m pip install -r src/reports/requirements.txt
python src/reports/build_products.py src/reports/miyun_boundaries.json outputs/miyun_brief --products brief
python src/reports/build_products.py src/reports/miyun_boundaries.json outputs/miyun_delivery --products all
```

| 选项 | 主要输出 | 输入 |
| --- | --- | --- |
| `maps` | `maps/` 的四张 PNG/PDF、设施暴露 GeoPackage 和汇总 | 案例两期 SAR、变化及覆盖矢量、设施底图 |
| `brief` | 一页 A4 Word 灾情简报 | 水面统计 JSON、可选乡镇 CSV、`01_water_change.png` |
| `atlas` | 图幅索引、逐幅详图及清单。有四图正文时同时生成带附录的 Word 应急报告 | 设施暴露 GeoPackage、后期 SAR、DEM、乡镇界及图件配置 |
| `review` | 可筛选的 Excel 与 GeoPackage | 设施暴露 GeoPackage、设施底图及图幅清单。已有图幅时填写 `products.atlas_dir` |
| `dashboard` | `web/index.html` 与 `web_brief.zip` | 水面统计、乡镇统计、案例矢量和网页用图 |
| `all` | 上述全部成果 | 各项所需输入 |

输出现有目录时程序退出，请为每次生成指定新目录。只生成 `brief` 时直接读取已制图成果，无需读取大幅 SAR 或重新计算暴露。选 `maps` 时后续产品复用当次生成的四张图和设施统计。网页可双击离线浏览，部署时将 ZIP 的内容解压到网站目录。

图序和清单使用同一图幅分配。核查表 `map_no` 对应图号，`review_id` 对应单项道路或建筑。跨乡镇道路按实际相交线切分，建筑保留完整面轮廓。清单的报告页码由 `products.first_detail_page` 指定（密云从第 6 页开始，每页最多 3 张）。更换 Word 正文模板后，以实际排版页码为准。

## 代码位置

| 文件 | 职责 |
| --- | --- |
| `build_products.py` | 统一命令行入口、成果目录与各阶段依赖 |
| `generate.py` | 既有四张专题图及设施相交计算 |
| `atlas.py` | 图幅划分、索引和局部制图 |
| `review.py` | 道路乡镇拆分、建筑属性、Excel 和 QGIS 图层 |
| `documents.py` | 单页 Word 简报与 Word 报告附录 |
| `web.py`、`templates/web/` | 网页数据打包、HTML/CSS/JS 模板 |
| `update_report.py`、`style_qgis.py` | 独立换图和 QGIS 样式工具 |
| `export_water_transitions.py` | 两期水面变化分类 GeoTIFF 与矢量，独立入口 |

原有 `generate.py`、`update_report.py` 和 `style_qgis.py` 仍可直接运行。附录支持四图 Word 正文模板。正文、统计表和既有图片随原文档保留。

## 换案例时修改 JSON

案例传感器、日期、地名和文件由 JSON 控制。已有成果的路径集中在 `products`。相对路径以案例 JSON 所在位置为准。

- `statistics`、`statistics_key` 指定主传感器的统计来源，单位为 km²；`town_csv` 和 `town_sensor` 选择对应的乡镇统计。网页要求统计表的乡镇名称与行政区图层逐项对应。
- `maps_dir` 指向已生成的四图目录。未同时执行 `maps` 时，可单独指定 `facility_gpkg` 和 `facility_summary`。记录沿用专题图导出的字段和图层名。
- `report_docx` 指向四图 Word 正文。只生成局部图件时可省略。换案例时先核查正文统计和图号。
- `dashboard.figures` 以 `map` 指向专题图文件名，也可用 `path` 指定现成图件；`crop` 和可选 `crop_portrait` 是按图幅宽高归一化的裁切框。主题无对应图件时可不配置该视图。


| 配置 | 数据要求 |
|---|---|
| `place`、`product_id`、日期和 `sources` | 填写本案例名称、成像日期、数据版本及来源。 |
| `vectors` | `roi` 为研究区，`common` 为两期共同有效范围，`changes` 的 `class` 为 `stable_water`、`water_gain`、`water_loss`。`aux_changes` 提供辅助水面分类，可加 `aux_review` 标注复核范围。`towns`、`post_coverage` 可选，乡镇名称取 `label`。 |
| `observations`、`observation_units` | 两个成像日期，各一至四景。用 `files` 填写路径。文件符合密云命名规则时，可用 `glob` 和 `utc_offset` 自动读时刻。按源数据声明 `power`、`amplitude` 或 `db`。快视分景拉伸。 |
| `optical` | `context` 为区域定位和设施总览，`review` 为辅助复核。每组填写同日 `tiles`（`[TCI, SCL]` 文件对），也可用 `glob` 选择 TCI，配套 SCL 取同目录的 `SCL.tif`。保留 SCL 4、5、6，其他类别在光学图留空。 |
| `aux_sar`、`image_labels` | 指定辅助 SAR 的路径、波段、单位和可选固定拉伸范围，补充底图的传感器及成像日期。 |
| `facilities` | 一套道路线和至多两套建筑轮廓。为每个数据源填写唯一要素字段 `id`、图层 `layer`、显示名 `label` 和输出表名 `key`。`bridge`、`tunnel` 用于标记桥隧，可空。 |
| `crs`、`exposure_detail_bounds` | 选择本地区米制投影。局部范围按 `[xmin, ymin, xmax, ymax]` 输入，优先选同时出现道路相交段与建筑轮廓的区域。 |
| `font_file` | 换操作系统时填写已安装的中文字体文件。 |
| `basemap_mode`、`dem_folder` | `terrain` 使用浅色地形晕渲和原信息栏版式，并需提供 DEM。`imagery` 使用影像底图。 |
| `labels` | 可选的本地地名源，输入地名点、道路及水系图层，名称取 `name`，道路可取 `ref`。`overview`、`coverage`、`aux`、`detail` 可指定各图优先显示的地名。绘制时按图框避让。 |
| `boundaries` | 可选。`sources.settlements` 读居住用地面，`reference_water` 读参考水面，`river_axis` 读参考河流线。乡镇界来自 `vectors.towns`。`detail_optical` 可配置带 SCL 的 TCI 瓦片和显示日期。 |

输入路径可相对于 JSON。`facility_exposure.gpkg` 中，道路几何是共同有效区内与新增水面相交的中心线段，长度单位为 m。建筑几何保留完整轮廓，属性记录重叠面积（m²）和比例（%），不同底图分别输出。桥隧标记和任何相交结果均用于确定核查对象。建筑重叠比例用于排序，实际受淹和损坏需要其他资料核实。

影像模式的变化图中，当期 SAR 优先覆盖光学定位底图。设施图的光学和 SAR 局部使用相同范围、比例尺和矢量。底图和地名仅用于判读与定位，切换底图不会改变水面与设施统计。密云地名取本地 OSM 北京数据，图纸注明贡献者。

居住用地轮廓来自 OSM 的 `landuse=residential`。无名称的用地面在仅包含一个村庄名称时关联该名称。行政村界需另有区划矢量。参考岸线、河流线与各期 SAR 水面按不同符号显示，水面变化与暴露统计由各期 SAR 数据计算。 [OpenStreetMap 居住用地定义](https://wiki.openstreetmap.org/wiki/Tag:landuse%3Dresidential) · [OpenStreetMap 许可与署名](https://www.openstreetmap.org/copyright)

通过 `facility_print` 设置最后一张设施暴露图。密云轮廓版输出 210 × 297 mm 竖向 A4 PDF 及 300 dpi PNG。Word 末页由 `report.facility_page_mm` 同步。开启此选项需要 `pypdf`。

## 两期水面变化分类 GeoTIFF

在仓库根目录运行 `python src/reports/export_water_transitions.py`，数据目录在脚本末尾设置。有三波段 `water_transitions.tif` 时，脚本重建原网格分类图并导出显示图。如果目录里只保留了 `water_transitions_palette.tif`，脚本直接读取它生成显示图。

- **面积统计与矢量化**使用 `water_transitions_palette.tif`，保留原像元网格。
- **QGIS 制图**加载 `water_transitions_display_4326.tif`。该图已用最近邻重投影到 WGS84 经纬度，保留四类颜色及透明的 NoData（255）。同名 `.qml` 提供中文图例。

需要同时导出三类独立面和合并面时，调用同脚本的 `export_water_transitions`，输入已有变化栅格或两期二值掩膜。面矢量从原网格分类图生成。
