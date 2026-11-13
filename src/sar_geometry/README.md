# SAR地形几何模块：数学模型重实现与迁移方案

编写日期：2026-11-12。目录名称：`sar_geometry`。

本模块依据 `reverse_engineering/gc_map2` 中修订后的数学模型与静态证据，独立实现雷达地理定位和地形可见性计算。SPHERE 的现有算法仅作为对照，不复制其旧叠掩/阴影计算实现。

**当前状态（2026-11-13）：零Doppler整景观测剖面已接通。** Windows C++/OpenMP核心已编译，23项解析/合成与I/O测试通过；密云一景已输出独立阴影、叠掩及联合候选掩膜，并与旧GUI结果做了对照。**非零Doppler剖面、GAMMA数值等价、真实地形分类精度仍未验收。没有替换现有水体结果。** 以下章节同时说明当前实现和后续迁移边界，以本节能力表为准。

### 当前如何使用

在仓库根目录、`sar-water`环境执行：

```powershell
python src/sar_geometry/build.py
python -m unittest discover -s tests/sar_geometry -v
python src/sar_geometry/main.py configs/sar_geometry.yaml
```

编译需要CMake及支持C++20/OpenMP的编译器。当前验证环境为Windows、MinGW GCC 16.1、Ninja；Linux构建尚未验证。Python使用现有NumPy、Rasterio、PyProj、SciPy、PyYAML和Matplotlib。资源监测使用系统接口，不依赖psutil。

必须通过命令行传入配置，例如`python src/sar_geometry/main.py configs/sar_geometry.yaml`；也可以传入自己的配置文件。所有路径由YAML设置，相对路径以YAML所在目录为基准。

| 入口/接口 | 已实现内容 | 使用条件 |
|---|---|---|
| `mode: inspect` | 读取XML或ZIP注释，输出`metadata_inspection.json`和日志 | 默认模式；不解压或读取SLC像元 |
| `mode: nominal` | 在DEM原生网格计算LUT、状态、覆盖和局部入射角 | 明确DEM、确认雷达网格、Doppler约定及椭球高；核心已编译 |
| `core.classify_profile()` | 对已构造的同观测剖面计算阴影及叠掩 | 仅实验接口；调用者提供完整支持域、地表顺序、斜距及有向角 |
| `mode: visibility` | 整景零Doppler观测剖面、阴影、叠掩和联合掩膜 | 输出独立米制地理网格；不修改水体 |
| `validate_scene.py` | 同网格旧GUI对照、方向诊断、角点检查、快视图和性能记录 | 读取原地形成果；没有真值时不报告精度提升 |
| 非零Doppler剖面、生产水体修正 | 尚未实现 | 不自动把斜视改成零Doppler，不以几何有效性替代排除掩膜 |

名义计算前必须设置`DEM_PATH`，核实`radar_grid_confirmed`，明确`doppler_mode: zero/metadata`。使用元数据Doppler时还须明确`doppler_sign: 1/-1`。首版DEM仅接受WGS84水平基准、米制WGS84椭球高（`height_datum: WGS84_ellipsoid`），不会自动把正高转换成椭球高。配置中的空值表示尚未确认，不可为了通过检查直接猜填。

已读取的BC2样本包含51个轨道状态、11个Doppler估计，雷达网格13824×16128。Doppler时间支持起点比影像首行晚约0.020006秒；`metadata`模式不外推，超出其支持范围会标记未知/求解失败。这些字段读取成功不等于成像约定已确认。

当前名义输出为`lookup.tif`、`geometry_validity.tif`、`radar_coverage.tif`、`geometry_status.tif`、`local_incidence.tif`、`doppler_residual.tif`和`geometry_quality.json`。全部采用DEM原生网格；尚不裁剪ROI或对齐ORG。质量记录明确写入`visibility: not_computed`。单一`data_cache/progress.json`记录完成行数，块写入完成后提交进度；输入和科学参数不变时可续跑，目录迁移不影响内容身份。请勿在同一输出目录同时启动两个任务。

23项测试覆盖：独立求根参考、时变/距离相关Doppler、左右视和坐标旋转、合法(0,0)、越界/不收敛、山脊遮挡的射线参考、多分支折返、支持域空洞、XML精确字段、DEM法向/NoData、输出网格、观测平面及高程约束、1/4线程一致性、整景续跑和分块一致性。测试尚不包含真实定位精度、一般斜视二维地形遮挡或GAMMA逐像元对照。

### 整景掩膜和密云案例

```powershell
# 计算整景掩膜，使用30m剖面间隔与30m输出网格。
python src/sar_geometry/main.py configs/北京密云区/sar_geometry_visibility.yaml
# 同一案例的7.5m剖面采样检查，输出网格仍为30m，含旧GUI对照。
python src/sar_geometry/validate_scene.py configs/北京密云区/sar_geometry_visibility_7p5m.yaml
# 可选：追加已有15m配置，只统计两档采样差异，不重跑粗采样。
python src/sar_geometry/validate_scene.py configs/北京密云区/sar_geometry_visibility_7p5m.yaml configs/北京密云区/sar_geometry_visibility_15m.yaml
```

案例配置也提供`_15m.yaml`。`azimuth_step_m`、`range_step_m`控制观测剖面的数值采样，`resolution_m`控制输出地理网格；加密剖面不代表DEM真实分辨率提高。原ORG约1.46亿像元，但本模块计算的是DEM支持下的地形掩膜，不是对全部SAR原始像元重新滤波/分割。性能不能与完整SNAP或水体生产链混为一谈。

整景模式增加`ORG_PATH`、`DEM_TILES`和`GEOID_PATH`输入。输出目录仅用于本实验。EGM2008 DEM由明确格网按`h=H+N`转成椭球高；原DEM不改动。Copernicus高度定义与PROJ格网来源见：[Copernicus DEM说明](https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/DEM.html)、[PROJ EGM2008说明](https://github.com/OSGeo/PROJ-data/blob/master/us_nga/us_nga_README.txt)。首次准备完成后直接复用本地椭球高DEM，不重复联网。

实际产品为`shadow_mask.tif`、`layover_mask.tif`、`terrain_exclusion.tif`、`geometry_validity.tif`、`radar_coverage.tif`和`geometry_quality.json`。掩膜0表示确认不受该效应影响，1表示受影响，255表示未知/未覆盖；阴影与叠掩可同时成立。候选排除掩膜不含旧15°坡度规则。比较时分别评价阴影与叠掩，不能把总排除面积减少直接称为改善。

`data_cache/scene_progress.json`保存完整剖面进度和地理映射进度，失败批次重算；`inputs.json`只记录准备输入的完成状态，不设置输出目录签名锁。目录中不得同时运行两个任务。资源参数包括`threads`、`profile_batch_rows`、`map_block_rows`；横向剖面始终整体扫描，不能通过缩短横向支持域来加速。C++按独立剖面或点使用OpenMP，Python使用NumPy/PROJ批量运算，GDAL缓存限制为64MiB。RSS监测每20ms采样，包含当前Python和核心子进程，不是系统精确瞬时峰值。

如成像约定未核实，默认拒绝；只有明确设置`allow_unconfirmed_geometry: true`并填写`geometry_assumption_note`才进行探索性计算。报告始终记录`production_ready: false`，不能通过开关将实验结果变成已验证产品。

## 1. 目标与首轮边界

首轮目标是在已有ORG水体提取流程中提供更可靠的叠掩、阴影和几何未知区域，重点识别局部坡度方法无法覆盖的山体遮挡。

首轮独立计算地形几何产品，不改变ORG幅度、不重新进行辐射校正，不修改滤波、阈值、历史水体或土地覆盖规则，也不修改监督训练代码及数据。

| 范围 | 本阶段安排 |
|---|---|
| 雷达成像参数、轨道、Doppler与坐标定义 | 首轮实现并核验 |
| DEM→雷达名义LUT、求解残差和有效性 | 首轮实现 |
| 同一地表曲面的法向、局部入射角 | 首轮实现 |
| 雷达方向上的遮挡包络、距离折返及叠掩分支 | 首轮重点 |
| 地形产品对齐到ORG、水体排除接口 | 单景验证通过后接入 |
| pixel-area、强度匹配、偏移拟合、精化LUT | 后续独立阶段 |
| SLC重采样、完整地理精校正 | 后续独立阶段 |
| 辐射地形校正、相干性和InSAR处理 | 不属于本次迁移 |
| 直接恢复GAMMA完整源代码、宣称逐位等价 | 不作为交付承诺 |

`gc_map2` 名义几何、`pixel_area` 面积参考和后续匹配精化分别验收。不能用精化拟合吸收的误差，掩盖名义几何模型错误。

## 2. 证据来源与使用规则

主要依据为以下本地材料，后续修订报告优先于早期人工标注：

1. `E:\Code_projects\SpaceTY_SAR_vs2022\reverse_engineering\gc_map2\report\CALC_LUT_MODEL_DERIVATION.md`
2. 同目录下的 `data/model_evidence_manifest.json`、`data/usage.txt`。
3. `disasm/geometry_full.asm`、`disasm/omp_named.asm`、`disasm/geo_loc_named.asm`、`disasm/find_rg_az_named.asm`。
4. `data/model_synthetic_checks.json`：解析例子，仅用于建立测试，不是生产精度报告。

静态材料对应目标ELF的SHA256：

`6de93a4c12591753545aae9e9787f8041ba147e046ae03321737eb83c4262676`

此哈希用于追溯既有报告，不表示本轮重新运行或校验了GAMMA二进制。

| 数学环节 | 已有证据 | 实现边界 |
|---|---|---|
| map→radar时间求解 | 距离—Doppler约束、二维Doppler、近似时间修正 | 明确本模块的符号、时标、残差与退出条件；不称为恢复了完整GAMMA求解器 |
| 地表法向 | 相邻地表点转ECEF后叉积 | 固定本模块曲面和导数定义，不盲目复制极小采样步长 |
| 阴影 | 有向角度包络扫描 | 内部网格尺度和所有边界细节未完整恢复，需要独立定义和验证 |
| 叠掩 | 距离折返及区间两侧扩展 | 保留全部重叠分支；不先按斜距排序 |
| 掩膜编码 | 反汇编有叠掩与阴影同时成立的证据 | 不直接沿用未完整核验的GAMMA整数编码 |
| simSAR | 角度模型 | 不是面积积分，也不是实际SAR辐射校正 |

已知需要纠正的早期解释：`geo_loc` 是固定状态下的逆向交点求解，不能当作逐DEM点求时间的函数；12个OpenMP区域不等于12个线程；逆解的米制容差不能直接移作时间求解容差。

参考：[GAMMA 2017更新说明](https://www.gamma-rs.ch/uploads/media/upgrades_info_20170630.pdf)、[GAMMA 2019更新说明](https://www.gamma-rs.ch/uploads/media/upgrades_info_20191209.pdf)。这些资料支持观测方向计算与产品一致性的重要性，不替代本项目数值验证。

## 3. 目录与代码组织

优先采用少量职责明确的脚本，避免多层继承和同一公式多处实现。计算核心使用独立C++，Python负责配置、数据适配、栅格I/O和任务组织。

```text
src/sar_geometry/
├─ README.md                 # 本方案与后续使用说明
├─ main.py                   # 独立入口，读取指定YAML并组织阶段
├─ metadata.py               # BC XML/轨道/Doppler转换成明确的几何参数
├─ terrain_io.py             # DEM坐标/法向、分块I/O、名义成果及缓存
├─ core.py                   # Python到C++的唯一调用接口
├─ profiles.py               # 零Doppler观测平面、DEM采样及完整剖面
├─ scene.py                  # 整景支持域、掩膜映射、缓存与阶段日志
├─ scene_validation.py       # 对照、可选方向诊断、资源监测
├─ validate_scene.py         # 单景计算及对照的手动入口
├─ build.py                  # 独立CMake构建入口
└─ native/
   ├─ CMakeLists.txt         # 独立目标sar_geometry_core
   ├─ main.cpp              # 命令行协议、日志、退出码
   ├─ geometry.h/.cpp       # 轨道、Doppler、RD求解、局部入射角
   └─ visibility.h/.cpp     # 已构造剖面的阴影包络、叠掩分支

tests/sar_geometry/          # 解析、合成、I/O及回归测试
configs/sar_geometry.yaml   # 独立配置，不覆盖现有密云YAML
```

以上文件已创建。后续再增加`water_adapter.py`和观测支持网格实现，不生成空占位脚本。

C++数学核心不依赖SPHERE插件、Qt界面、现有项目DLL或GAMMA运行时。Python分块读写栅格，通过子进程的二进制管道传递参数和点坐标；C++不接收中文文件路径、不管理GeoTIFF或缓存。协议1采用小端uint64和Float64，每块调用一次，协议版本错误或结果长度不完整会报错。用户只编辑YAML，不维护第二套参数。大规模处理的进程启动/拷贝开销尚需基准测试，之后再评估常驻进程或共享内存。

通用投影和栅格操作使用Rasterio/PROJ及GDAL能力。OpenMP已接入并通过1/4线程结果一致性测试；Windows和Linux需要分别构建，不能把Windows DLL直接用于Linux。

## 4. 输入契约

### 4.1 当前密云数据

本轮只读检查了17景ORG元数据和17个对应SLC压缩包，按传感器、时刻及产品编号可匹配17对；实际包含BC2、BC4、BC5。SLC注释声明为复数产品，包含轨道、时间采样、斜距及Doppler字段。尚未完整读取这些SLC像元或执行几何处理。

ORG GeoTIFF是地理网格，XML中部分字段保留了原始雷达网格。例如一景BC2的XML尺寸为13824×16128，而ORG为13979×10428。两套网格不得混用。

部分BC产品XML的 `missionId` 写为 `S1B`，不能只依据这个字段选择Sentinel-1处理模型；应同时核对产品编号、目录结构和实际参数语义。

### 4.2 必需数据及含义

| 输入 | 要求 |
|---|---|
| 成像元数据 | 明确SM/Stripmap、视向、斜距/方位网格、时间原点、采样间隔、波长、Doppler及deskew语义 |
| 轨道状态 | 位置单位米、速度米/秒；明确坐标框架、时标及有效区间 |
| DEM | 有水平CRS、垂直基准、米制高程及NoData定义；不能把正高直接当椭球高 |
| 目标参考栅格 | 首轮使用ORG，锁定CRS、transform、宽高与有效范围 |
| ROI | 只约束最终输出范围，不截断判断遮挡所需的周边地形 |
| SLC/MLI像元 | 首轮名义几何不必读取全部强度；后续匹配精化必须使用与雷达参数一致的实际像元 |

轨道时间先转换为统一、连续的秒坐标，并保留原UTC记录。涉及闰秒时明确转换表和支持范围；不得直接混用UTC、TAI或跨日字符串差值。

Doppler适配必须核实距离/时间多项式的参考点、尺度、Hz单位、符号及适用范围。XML存在非零多项式不自动表示应将它直接用于处理后影像的RD约束；deskew/零Doppler语义未确认的景先报告缺项，不猜填。

高程转换使用明确的垂直基准与可用格网；缺少转换条件时停止该项真实几何验收，不通过手动改标签宣称已转为椭球高。

## 5. 数学模型与核心接口

### 5.1 轨道、Doppler与名义LUT

地面ECEF位置为P，卫星位置和速度为S(t)、V(t)，L=P−S(t)，R=|L|，波长为λ。采用与修订报告一致的约定：

\[
G(t)=V(t)\cdot L(t)-\frac{\lambda}{2}R(t)f_D(R(t),t)=0,
\qquad
f_g(t)=\frac{2V(t)\cdot L(t)}{\lambda R(t)}.
\]

候选时间更新：

\[
t_{k+1}=t_k+\frac{G(t_k)}{|V(t_k)|^2}.
\]

这是一种近似更新；不把它表述为完整加速度Newton。当前轨道使用相邻位置和速度约束的分段三次Hermite插值，速度取同一位置曲线的导数，不外推。Doppler先在各记录自己的距离原点计算Hz，再在时间方向线性插值；不混合不同原点系数，不外推。时间更新限幅但不把越界解夹入轨道范围；更换求根器需作为独立改动验收。

距离和方位坐标采用实际有效雷达网格：

\[
x_r=\frac{R-R_0}{\Delta R},\qquad y_a=\frac{t-t_0}{\Delta t}.
\]

坐标定义为零起始像元中心；多视、裁剪后的原点和间隔必须已经反映在参数中，不能重复乘多视因子或用PRF替代所有产品的行采样率。

已实现核心接口：

```text
core.solve_points(executable, model, points, normals)
    -> N×8: range_pixel, azimuth_pixel, time_s, residual_hz,
            iterations, status, local_incidence_deg, step_pixels
```

成功条件同时检查方程残差、步长、轨道支持域和几何约束。最大迭代次数到达或步长不再下降不视为成功；不能把非法解夹到图像边缘。几何解存在、落在影像范围内、地形可见，是三个不同状态。

LUT的合法(0,0)坐标必须保留，不能以零坐标替代有效性掩膜。

### 5.2 同一曲面上的位置、法向与入射角

用唯一的地表函数P(u,v)计算DEM位置与邻域位置，再由两条地表切向量的叉积计算向外法向n。定义地面指向卫星的单位向量l，则：

\[
\theta_{local}=\arccos\bigl(\operatorname{clip}(n\cdot l,-1,1)\bigr).
\]

当前实现取DEM像元中心高程，转为ECEF后以一像元邻域中心差分计算法向；不对缺邻点的边缘使用单侧差分，法向记NaN。这是本模块的工程定义，不能宣称与GAMMA内部B-spline完全一致。后续曲面插值及步长敏感性仍需独立验证，不能直接照抄0.001像素采样步长。

高程缺测不静默填成0；默认缺测影响到的法向和可见性记为未知。若后续引入填洞，作为独立选项保留原始缺测范围并单独验证。

### 5.3 雷达方向支持网格

建立P(a,s)：a表示雷达方位位置，s表示沿地面横向、近侧到远侧的单调参数。在固定a的剖面中计算R(a,s)和有向观测角。

不能直接把北向DEM行当作雷达横向剖面，也不能先按R排序；后者会消除需要检测的距离折返。

支持网格的生成方式、间隔、插值和边界是逆向证据未完整覆盖的部分，须由本项目明确实现。首版先验证零Doppler及明确支持的Stripmap几何，再扩展非零/时变Doppler，不把固定方位截面错误地等同于任意斜视条件下的垂直平面。

当前零Doppler实现：令单位速度为v，O=(S·v)v，e0=(S−O)/|S−O|，e1按左右视取±(v×e0)。地表点写成P=O+ρ(cosθ e0+sinθ e1)。逐点求ρ，使WGS84椭球高与同位置双线性DEM高程相符；因此V·(P−S)=0，θ保留地表顺序。先生成完整剖面，再由地面点的RD时间和θ采样剖面掩膜；不通过多值的斜距坐标插值掩膜。

首版约束0<θ<0.2弧度，外围地形使用显式椭球高上下界。近侧高程上包络给出未知遮挡的角度上限，近侧低包络和远侧高包络限制尚可能存在的同斜距分支。内部DEM空洞使受影响可见性保持未知，不能跨洞当平地。上述支持域边界和采样定义是本项目独立设计，不是已恢复GAMMA内部支持网格。

### 5.4 阴影包络

按剖面顺序维护前方地形的可见角度包络，检测目标到雷达的连线是否被更近地形阻挡。局部表面朝向雷达并不保证无遮挡。

当前实验接口（阴影和叠掩一次返回）：

```text
core.classify_profile(executable, points, support_complete,
                      angle_epsilon=1e-10, range_epsilon=1e-6)
    points: N×4，单调地表s、斜距米、向上有向仰角弧度、有效标记
    -> N×2，阴影/叠掩，各自0/1/255
```

以独立的二维射线—地形线段相交作为合成测试参考。若剖面起点、空洞或外围DEM不足以判断是否存在遮挡，将受影响区域记为未知，不把裁剪边缘自动判为可见。

### 5.5 距离折返与叠掩

检测沿s的距离映射折返；在统一近远方向约定下，dR/ds≤0作为折返或临界种子。将对应距离区间向两侧展开，识别所有映射到相同雷达采样区间的地面分支。

不能只标记陡坡种子，也不能仅统计某距离箱收到两个相邻DEM点就判为叠掩；正常有限采样也会产生多点落箱。需要区分正常连续采样与不同地面分支重叠。

当前与上述`classify_profile()`共用剖面输入；后续整景支持网格再调用它。概念接口：

```text
layover_profile(surface_order, slant_ranges, sampling)
    -> layover_mask, known_mask
```

首版采用可验证的剖面分支模型；显著方位耦合、强斜视或复杂二维折叠若超出支持范围，应明确限制。二维地表面元投影重叠是后续扩展，不在首版默认宣称已解决。

叠掩与阴影分别输出，允许二者同时为真，不互相清除标记。

## 6. 输出产品与水体接口

| 文件 | 数据定义 | 用途 |
|---|---|---|
| `lookup.tif` | 两波段Float64，距离/方位像元坐标；无效NaN | 名义几何验证；生产精度优化另行评估 |
| `geometry_validity.tif` | Byte，1有效/0无效 | 表示RD几何解有效，不表示无遮挡 |
| `radar_coverage.tif` | Byte，1在雷达像元中心范围内/0其他 | 当前已实现，与几何有效性分开 |
| `geometry_status.tif` | 整数状态码，字典写入质量JSON | 区分轨道越界、不收敛、DEM缺测等 |
| `local_incidence.tif` | Float32，度，无效NaN | 局部角度检查 |
| `doppler_residual.tif` | Float32，Hz，未求值NaN | 当前已实现；非收敛点可保留最终残差，必须结合状态读取 |
| `layover_mask.tif` | Byte，0否/1是/255未知 | 独立叠掩产品 |
| `shadow_mask.tif` | Byte，0否/1是/255未知 | 独立阴影产品 |
| `terrain_exclusion.tif` | Byte，0不排除/1排除/255未知 | 对齐ORG后的水体接口 |
| `geometry_quality.json` | 参数、网格、版本、覆盖、状态计数、残差和耗时 | 可追溯性；不冒充精度真值 |

上表的叠掩/阴影/联合掩膜已由`visibility`模式输出；名义LUT产品仍由`nominal`模式输出。当前尚未将候选掩膜接入水体生产。质量JSON记录参数、输入标识、采样、状态计数、阶段耗时和内存；测试报告另外记录对照耗时。

两种掩膜任意一个确定为1时，联合排除为1；二者均确定为0时为0；其余为255。未知区域不能当作已确认无水。旧15°坡度阈值不是物理阴影/叠掩定义，首轮不作为隐藏规则叠加进新模型。

名义LUT与对齐ORG后的掩膜分别记录所属网格。类别掩膜对齐默认最近邻，未知域单独传播；若需保守覆盖型重采样，作为显式独立选项测试，不用双线性制造混合类别。DEM采样更密不等于增加了真实地形分辨率。

ORG本身可能存在定位偏差，按CRS对齐只保证坐标网格一致，不能保证水体与地形实际对应正确。先检查山脊、道路和岸线等稳定位置；明显错位的景不自动用于硬排除。

正式接入位置：

1. `src/thres_segmentation/org_pipeline.py`：消费已经计算的地形产品。
2. `src/thres_segmentation/water_polygons.py`：成面清理后再次施加同一排除区。
3. 质量JSON、有效覆盖统计及核验图展示未知区域，不将其静默计入非水体真值。

当前 `gui_geometry` 保持生产默认；新方法通过独立配置显式选择，缺元数据或不支持的几何时报告失败，不悄悄退回旧算法并仍标记为新方法。

## 7. 执行、缓存与效率

拟执行顺序：读取几何参数 → 准备DEM支持域 → 求解名义LUT与法向 → 雷达剖面与可见性 → 对齐ORG → 写质量记录。首次从单景开始，用户在终端手动运行。

路径和线程等执行参数集中在一个YAML；数学容差在同一配置的算法分组中注明单位。阶段日志有编号、输入景号、进度、耗时，同时写终端和log。

续跑以一个 `data_cache/场景/进度记录` 为依据，不叠加全局输出目录签名检查。已完成阶段需有完整文件和完成标记；中断文件不视为完成。缓存记录输入内容标识、DEM/几何参数及算法版本，路径迁移本身不使缓存失效；算法或科学参数变化后重算相关下游阶段，不能复用旧模型掩膜。

效率策略：

- 轨道插值和投影准备复用；避免每像元重复解析元数据。
- DEM、轨道辅助与剖面支持数据按景缓存；大栅格采用分块或磁盘数组。
- 可见性分块须保留沿观测方向的包络状态和跨块叠掩分支。普通有限halo不能保证远距离遮挡正确，不能把各小块独立计算后直接拼接。
- 只输出用户需要的最终掩膜和质量记录；LUT等工作产品放缓存，诊断输出可关闭。
- 先冻结串行结果，再按独立方位剖面并行；记录内存峰值和线程数，避免与SNAP等任务竞争全部资源。

## 8. 分阶段验收

| 阶段 | 交付内容 | 进入下一阶段的条件 |
|---|---|---|
| A：参数与定位 | BC参数适配、独立RD求解、LUT与状态 | 解析轨道、常/二维Doppler、像元原点、左右视及轨道边界测试通过 |
| B：表面模型 | DEM坐标、高程基准、法向和局部角度 | 平面/二次曲面、平地极限、空洞、边缘及步长敏感性测试通过 |
| C：地形可见性 | 观测方向剖面、阴影与叠掩独立掩膜 | 山脊遮挡、多分支折返、组合掩膜、不同航向、支持域不足、分块一致性测试通过 |
| D：真实单景 | 密云SLC/ORG元数据、DEM上的独立产品 | 网格及几何对齐可核验，与GAMMA/人工参考的差异有明确记录 |
| E：水体对照 | 固定分割候选，仅替换地形排除 | 同有效范围比较误删水体和山区误保留；人工核验后决定是否设为默认 |
| F：完整精校正 | SLC强度/MLI、pixel-area、匹配拟合、精化LUT和重采样 | 独立验收匹配、名义/精化LUT及实际定位；不与首轮掩膜变化混做一个实验 |

测试应覆盖：

- 局部入射角小于90°但被前方山脊遮挡的低坡目标。
- 一次及多次距离折返，正常采样落同箱不产生伪叠掩。
- 不同航向、左右视、同一场景坐标旋转后的几何一致性。
- 保持外部DEM支持域时，改变最终ROI裁剪不改变内部结果。
- 分块/整条剖面、串行/并行结果一致；边界与未知范围明确。
- RD残差小但落在错误视侧/分支的反例，不能只看残差数值。

数值容差按坐标单位、源网格分辨率和独立参考误差确定，写入测试配置。离散分类在排除临界角和临界距离样本后要求一致；临界样本单独报告，不用随意放宽全局容差掩盖错误。

真实对照冻结同一ORG有效域、DEM及高程处理、阈值和水体清理参数，分别报告：名义LUT残差与有效覆盖、阴影/叠掩/未知面积、掩膜差异位置、水库误删与山区误保留。没有人工真值时只报告几何一致性与候选错误，不以面积下降宣称精度提高。

GAMMA参考运行采用明确版本和相同参数；调用本地授权环境由用户监控执行。不将参考程序输出差异直接解释为本实现错误或GAMMA绝对真值，仍需解析算例与独立空间核查。

## 9. 后续精校正的明确边界

完整精校正从SLC/MLI实际雷达网格出发，不能把已地理编码的ORG重新当作斜距影像输入。匹配参考由独立pixel-area阶段生成；强度单位、幅度/功率转换及多视参数必须与影像元数据一致。

精化LUT后的影像、覆盖与掩膜必须按一致的坐标合同输出；不能只对水体矢量施加拟合平移，而留下影像及质量掩膜不动。

逆向未完全确认的内部网格尺度、Euler约定、插值边界、编码及所有斜视模式，不作为已完成能力。必要时继续对指定二进制做有针对性的逆向核查，并将新增证据与独立设计区分记录。

独立名义几何和单景零Doppler观测方向掩膜均已接通。当前已完成的是数值/工程验证与新旧结果差异检查；下一步应确认真实成像约定并使用独立参考核查地形分类，完成前不替换密云水体成果。
