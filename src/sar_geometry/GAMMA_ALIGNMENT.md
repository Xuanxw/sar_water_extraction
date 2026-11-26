# gc_map2 v5.7逐项对齐

对照对象固定为逆向资料中的 **gc_map2 v5.7，28-Jul-2023**，来自GAMMA 20240627安装包。目标二进制SHA256：

`6de93a4c12591753545aae9e9787f8041ba147e046ae03321737eb83c4262676`

下表的指令级对照对象为 v5.7。2026-11-26 的两景整图试验另以 WSL 实际运行的 v4.5 地图产品为参照。两个版本的证据分别记录。

## 使用顺序折返扫描

另可选 `scene.layover_scan: ordered_subpixel`。此模式在双精度下将每条剖面的最小有效斜距移到 1 m，同时平移斜距支持边界，再调用同一顺序扫描内核。地图像元中心按剖面四邻点的几何权重表决。该选项属于本项目的边界处理，在甘孜 BC3 选择并在密云 BC5 复核。

针对内部距离向因子 0.8、方位抽取因子 1、DEM 过采样 6×6 的 v4.5 参照，漏判复核使用下列场景参数。

```yaml
scene:
  layover_scan: ordered_subpixel
  map_posting_factor: 6
  layover_vote_threshold: 0.45
  dem_resampling: cubic
  dem_resolution_m: 5
  range_step_m: 3.75
  azimuth_step_m: 7.5
  resolution_m: 10
```

`map_posting_factor` 细分的是源 DEM 像元中心之间的间距。程序从源 DEM 的 CRS、中心坐标和间距建立格点，再用本法准备的椭球高 DEM 取高程。这个过程无需输入 GAMMA 分类或查找表。同一批源 DEM 瓦片须为共格的北向上栅格。原地图中心负责幅宽、阴影和可判定性；相邻格点有效时，对已知叠掩在该点重新查询，否则保留地图中心的判定。省略细分选项时直接查询地图中心。`layover_vote_threshold` 默认为 0.5，只有四邻点叠掩状态都已知时才使用该阈值。

在现有零多普勒 `mode: visibility` 配置中，将以下字段加入 `scene`。在项目根目录用 CMake 将 `src/sar_geometry/native` 编译到独立目录，再把 `paths.CORE_PATH` 指向其中的 `sar_geometry_core`。可执行文件的 `version` 输出须包含 `profiles_ordered`。

```yaml
scene:
  layover_scan: ordered
  dem_resampling: cubic
  dem_resolution_m: 5
  resolution_m: 10
  range_step_m: 7.5
  azimuth_step_m: 7.5
```

程序先用 GDAL 三次卷积将源高程重采样到 5 m 米制支持网格，再在完整横向剖面上按 7.5 m 间距采样。按地表从近到远的顺序检查斜距折返，对满足扫描条件的转折端点以及近侧同斜距分支赋叠掩。角度增加方向由剖面近远次序确定。缺测空洞、轨道边界、未知外围地形仍约束有效性。阴影输出沿用有界地平线判定。未指定新字段的配置继续使用原区间规则及双线性高程重采样。

两景都在先前公布的三法共同有效区比较。甘孜 BC3 的叠掩 IoU 从 0.8267 升至 0.9295，密云 BC5 从 0.6245 升至 0.8097。甘孜只换扫描规则时为 0.8288，只改地形重建时为 0.8618，只把地图输出从 30 m 调到 10 m 时为 0.8331。固定旧共同区的统计原本排除了旧版未知位置。另按甘孜全部 GAMMA 有效区计分，旧版在 GAMMA 叠掩内保留未知 12.4896 km²，组合方案为 0.0005 km²。密云使用与 GAMMA 同源的 1″ DEM，并沿用此前的轨道覆盖限制。两景均未用独立真值核验绝对精度。

## 对齐清单

| 项目 | v5.7证据与目标行为 | 当前状态 |
|---|---|---|
| 输入网格 | MLI/SLC真实斜距网格；使用该网格的时间、距离原点与采样间隔 | 现有BC XML解析与ORG地图网格分开；缺少新ORG对应的完整雷达参考产品，尚未验收 |
| 高程与坐标 | DEM投影、椭球及高程基准参与定位 | 当前显式使用WGS84与转换后的椭球高；尚无同输入GAMMA数值对照 |
| 轨道插值 | main调用`init_bspline_double`；具体阶次、系数、边界需要恢复 | 当前Hermite，与目标不同；不将任意SciPy样条称为GAMMA等效插值 |
| DEM曲面 | B-spline曲面参与定位，内部网格加密另有线性插值 | 可选双线性或 GDAL 三次卷积重建支持网格，再双线性读取高程；源DEM真实信息分辨率与输出采样间隔分开 |
| 直接定位 | `geometry_full.asm` 0x13450–0x13de0；二维Doppler，近似时间步 | 当前方程与近似时间步同类；初值、插值及退出条件未逐项对齐 |
| 收敛与退出 | 空间步阈值及步长下降检查；退出不一定等于收敛 | 当前Hz残差与像元步长联合判定，与目标不同；暂不丢弃自身失败状态 |
| 表面法向 | 同一曲面四邻点，偏移0.001 DEM像元；边界约0.499/0.501像元内移 | 当前一像元中心差分；不一致 |
| 斜视观测框架 | 非零Doppler分支包含Euler角修正 | 整景剖面只支持零Doppler；未对齐 |
| 内部网格 | 方位/横向重排、反查、线性加密；`r_ovr/az_dec`默认自动 | 当前按零Doppler平面与地表角构造；不能仅给两个参数相同名称就视为一致 |
| 阴影扫描 | `.omp_fn.6` 0xd130–0xd1eb；有向角度包络、float32、零值跳过、严格比较 | **已独立移植并与重汇编指令对照通过** |
| 叠掩扫描 | `.omp_fn.6` 0xd1f0–0xd3fa；等斜距参与折返，阴影影响种子，前后分支补记 | 已与重汇编指令对照，并作为 `layover_scan: ordered` 接入零多普勒剖面。默认保留区间规则 |
| 缺测与支持边界 | 本次扫描内0为哨兵；完整链另有插值、支持及输出处理 | 当前有界高程包络与0/1/255未知传播为独立设计，不能称为目标同等行为 |
| ls_map编码 | 处理位码与显示码分开；0未测试、4叠掩、16阴影、20两者；可带tested位1 | 已补兼容编码/解码及对照读取；不宣称最终编码字节已一致 |
| 输出网格 | `ls_map`为地图/DEM网格；`ls_map_rdc`为斜距网格 | 当前主要输出地图网格；未实现目标雷达网格产品与其精化规则 |
| LUT精化 | `DIFF_par/ref_flg`控制偏移多项式及参考约定 | 尚未实现目标精化链，不将名义结果与精化产品混比 |
| DEM空洞、边缘与最终重映射 | 需核对扫描前后各步骤及最终码转换 | 留存核心扫描证据不能替代这些步骤的恢复；未对齐 |

依据：`SpaceTY_SAR_vs2022/reverse_engineering/gc_map2`下的`data/usage.txt`、`data/model_evidence_manifest.json`、`report/CALC_LUT_MODEL_DERIVATION.md`、`disasm/main_named.asm`、`disasm/geometry_full.asm`及`disasm/omp_named.asm`。以修订后的推导及指令为准，不沿用早期将`geo_loc`误称为直接时间Newton迭代的说明。

## 已恢复扫描的接口

`core.scan_gc_map2_v57_shadow_layover(executable, angle, slant_range, direction)`接收两张同尺寸float32内部剖面场，输出内部处理字节。每行保持完整横向顺序，只在方位行之间并行。

- `angle`必须是目标内部有向角度场；`direction`是其±1比较方向，不能直接用左右视值代替。
- `slant_range`单位米。不能先按斜距排序，否则折返信息消失。
- 0沿用目标内部哨兵；NaN/Inf不在已恢复接口的验证范围。
- 首个采样点不在阴影相邻对循环中自动置为“已测试”。
- 角度恰好回到包络时退出阴影；不增加当前独立算法中的epsilon。
- 等斜距启动/延续折返；纯阴影会影响新折返的启动。
- 末端没有额外的折返回填。额外回填虽然可能具有独立几何理由，也不属于该目标函数。

两种剖面扫描都接受有界支持域检查。整景 `ordered` 选项使用本实现自己的零多普勒角度和斜距场，角度先平移为从星下方向起算。地形曲面、轨道插值、剖面重排和最终地图栅格化仍与原版存在差异。

## 扫描验证

在WSL中运行：

```bash
python3 /mnt/e/Code_projects/sar_water_extraction/scripts/terrain/verify_gc_map2_v57_scan.py
```

脚本从指定逆向目录提取`.omp_fn.6`的231条计算/控制指令，保留分支与运算，去掉填充nop，使用系统libgomp重汇编。随后用这些指令作为独立参考，比较移植的C++函数。

已验证20,000组剖面批次，共1,779,996个采样点，覆盖角度正反方向、零值、等值、末端折返及1/4线程，输出字节一致。该验证执行的是重建扫描函数，**没有执行原始完整ELF**。报告位于`build/gamma57_oracle/verification.json`。

Windows独立构建位于`build/sar_geometry_gamma57`，没有替换原来的`build/sar_geometry`可执行文件。新增编码与扫描边界测试位于`tests/sar_geometry/test_gamma_reference.py`。

## 参考产品对照

配置：`configs/sar_geometry_gamma_compare.yaml`。填写本实现的阴影/叠掩图，以及由目标v5.7生成的同输入`ls_map`。

```powershell
python src/sar_geometry/compare_gamma.py configs/sar_geometry_gamma_compare.yaml
```

参考可为uint8 GeoTIFF，或原始UCHAR与经GAMMA导出的同网格GeoTIFF组合。程序检查尺寸、CRS和transform完全相同，不静默重采样，不把`ls_map_rdc`当作ORG地图网格。显示编码中的255按叠掩读取，不能当作NoData丢弃。

输出为独立的阴影、叠掩差异图及`gamma_comparison.json`。差异图：

| 值 | 含义 |
|---|---|
| 0 | 双方均未知 |
| 1 | 双方已知且一致 |
| 2 | 本实现比GAMMA多排除 |
| 3 | 本实现比GAMMA少排除 |
| 4 | 本实现未知，GAMMA已知 |
| 5 | 本实现已知，GAMMA未知 |

已知区域的分类一致率与排除区域IoU只是相对GAMMA的一致性，不是相对于地物真值的精度。未知范围、原始码直方图及输入哈希同时保留，不能通过缩小共同有效范围隐藏差异。

## 完整对齐的验收条件

完整数值对齐还需恢复轨道/DEM样条和目标内部方位—横向网格，再核对输出重映射、缺测及精化规则。上游改变时需要重新验证定位和地形掩膜。

完整验收需要同一轨道、Doppler、真实MLI/SLC网格、DEM与高程基准、采样参数、精化参数的v5.7参考输出。分别比较LUT连续坐标误差、阴影/叠掩类别、未知与覆盖范围、地图及雷达网格产品。先分清误差来自定位、内部网格还是分类，不能直接用容差吸收未解释差异。

当前留存资料没有全部样条库函数及所有内部尺度的完整语义，目标v5.7可执行文件原路径也已不可用。因此仍不能宣布整链逐项一致。现有生产水体结果及阈值配置保持原样；单景不超过180秒的性能目标须在完整链接通后实测。

继续恢复所需的函数体在现有`data/symbols.txt`中有地址，但`disasm`中没有对应完整函数定义：

| 函数 | v5.7虚拟地址 | 核查内容 |
|---|---|---|
| `init_bspline_double` | 0x6d960 | 轨道样条阶次、系数与边界 |
| `interp_bspline_double` | 0x74c30 | 轨道求值及有效范围 |
| `init2D_bspline_float_mp` | 0x6b4c0 | DEM曲面系数、空洞与边缘 |
| `interp2D_bspline_float` | 0x779a0 | DEM插值与边界求值 |
| `find_az_yb_lut` | 0x16110 | 方位/横向内部网格反查 |

补充这些函数及其调用到的相关函数反汇编，或恢复相同SHA256的目标可执行文件后，才能继续核实剩余数值细节。既有调用位置、函数名与汇编常量不足以唯一确定实现。
