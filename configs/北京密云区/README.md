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
