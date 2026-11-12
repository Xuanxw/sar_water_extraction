# 北京密云区 ORG 水体提取

本目录的 `threshold_segmentation.yaml` 是本次观测的独立配置。自动扫描目录内已完成的BC2–BC5线性幅度ORG，排除SLC，不限制景数和日期数；不需要水体真值或模型检查点。

## 执行

输入文件下载完成后，在仓库根目录运行（支持先处理已有影像、后续追加新景）：

```powershell
conda activate sar-water
python src/thres_segmentation/main.py
```

`main.py` 的 `CONFIG_PATH` 已指向本目录。相同命令可续跑，保留输出目录、配置和缓存即可。无需先执行 `image_process.py`。

流程为：检查ORG清单 → 行政区外扩裁剪 → SNAP Lee功率域滤波并恢复幅度 → AOI内统一分位阈值 → 土地覆盖/坡度/历史水体候选修正 → 逐景SHP、质量JSON与核验点。不进行辐射校正，也不自动合并日期。

## 配置

| 项目 | 当前值/用途 |
|---|---|
| `paths.ORG_ROOT` | `E:\InSAR\水体提取\北京密云区\ORG`，支持同名TIFF/XML或ZIP；不能同时保留同景的ZIP与解压影像 |
| `paths.AOI_PATH` | `E:\InSAR\水体提取\北京密云区\密云区行政划分.shp` |
| `paths.OUTPUT_DIR` | `E:\InSAR\水体提取\北京密云区\outputs\thres_seg\miyun_org` |
| `threshold.water_percent/mode` | 10/bottom：AOI有效观测内第10百分位的暗侧；不是Otsu、模型概率或精度保证 |
| `org.filter` | SNAP Lee，线性幅度输入，7×7窗口，ENL=1 |
| `org.normalization` | none；scene_zscore采用等价幅度阈值，不额外写标准化影像，线性标准化不改变分位分类 |
| `org.auxiliary` | WorldCover 2021与Copernicus GLO-30；本地路径优先，否则按行政区一次下载公共瓦片 |
| `org.postprocess` | 默认排除建成区50和坡度>20°；历史水体放宽候选至第15百分位；形态学默认关闭 |
| `quality_review` | 三层随机核验点各最多50个，岸线缓冲5像元，固定seed=42 |
| `processing.block_size` | 1024；限制读写内存，不改变整景统一阈值 |

没有显式NoData时保留有限0值；若核实外部黑边以0填充，应配置 `org.input_nodata_value: 0`，避免黑边参与阈值统计。不要仅凭显示黑色就把真实暗水体当无效。

WorldCover 2021水体是静态参考，不是本次观测的真值；默认只恢复满足较宽SAR阈值的候选。建成区和坡度硬筛选可能删掉真实积水，应人工核验。Copernicus GLO-30是DSM，坡度筛查不等于雷达阴影/叠掩分析。

## 输出及续跑

每景输出在 `OUTPUT_DIR/日期/影像编号/`：

- `water.shp`及配套文件：经过修正并裁到行政边界的水体面。
- `quality.json`：有效覆盖、水体面积、辅助修正、阈值敏感性、抽样与疑似错误统计；没有真值时F1/IoU为空。
- `review/review.gpkg`：`verification_points`核验点、`suspected_regions`疑似区域；人工填写`ref_class`（0非水体、1水体、-1不确定）。
- `quicklook.png`：缩略对照图；SAR面板注明仅显示使用P2–P98拉伸，不修改计算输入。
- `raw.tif/water.tif/coverage.tif`：原始阈值、修正后水体、有效覆盖；255表示未知，不是非水体。
- `metadata.json/source.xml`：原始元数据追溯，XML缺失时不伪造。

输出根目录保留运行参数和清单，仅作追溯，不参与配置/路径签名校验。续跑唯一入口是`paths.CACHE_DIR/scenes/影像编号/progress.json`：阶段完成记录最后写入，已完成且对应文件存在、非空就复用；缺失或中断的阶段重做。裁剪幅度和阈值统计放在此逐景缓存中，SNAP成果仍在`SNAP_CACHE_DIR`。

进度内的文件位置相对于缓存、成果或SNAP缓存根目录保存。迁移时复制缓存和成果目录，更新YAML的根路径即可，不要求原磁盘位置或原mtime。旧`run.json`和逐景参数文件不再阻止迁移；原有`*.done.json`不再作为续跑入口。

`resume: true`按已有进度复用，不自动识别同景源文件或算法参数的变化。改变这些内容后设`resume: false`重新处理；需保留旧成果和人工核验记录时，同时选择新的输出/缓存目录。不要在需要保留人工核验编辑的目录里强制重算。辅助公共瓦片仍可共享缓存。

每景SHP保留原SAR坐标系，目录日期按文件名UTC转换北京时间。多景覆盖不完整时，仅解释实际有效范围；逐景面积不可直接相加，以免重复计入重叠区。

如后续需要按日期求几何并集，单独配置并运行 `scripts/merge_water_shp.py`。它不会被提取主流程调用，不判断跨景冲突或时序变化。

代码已通过合成影像的裁剪、辅助修正、SHP输出、分块一致性、中断续跑及缓存/成果迁移测试；实际影像的下载、SNAP处理与提取质量需运行后核验。

## 日志与阶段耗时

日志在`paths.OUTPUT_DIR`下：`processing.log`为UTF-8纯文本运行日志，`stage_timings.csv`为可用Excel查看的阶段计时表。续跑追加保存，每次启动有独立`run_id`；时间为北京时间，秒级耗时使用单调时钟计算。

| 阶段 | 内容 |
|---|---|
| 01 | 扫描ORG及读取行政区 |
| 02 | 辅助数据下载/缓存复用及空间对齐 |
| 03 | 初始化SNAP运行环境 |
| 04 | 逐景元数据、解压和行政区裁剪 |
| 05 | SNAP Lee滤波或复用滤波缓存 |
| 06 | 整景分位阈值统计 |
| 07 | 分块提取及辅助修正 |
| 08 | 输出水体SHP |
| 09 | 质量JSON、随机核验点和疑似区域 |
| 10 | 快视图 |
| 11 | 保存逐景完成记录 |

`00`为本次运行总耗时，`SCENE`为逐景总耗时，均包含子阶段，不能与子阶段重复相加。按`run_id + scene_id + stage`区分每次执行；统计耗时时筛选`event=end`，再区分`completed/reused/skipped/failed/interrupted`。复用记录的耗时是本次检查时间，不是历史计算时间。整景完全复用时只记`SCENE=reused`，不重复展开子阶段。

每条日志及开始/结束事件立即flush。Ctrl+C和Python异常会记录中断/失败及堆栈；断电、强制结束进程可能只保留开始事件，没有结束事件即表示该次阶段未正常结束，不能当作耗时0。

`logging.console_color: true`开启termcolor：青色开始、绿色完成、黄色复用/跳过、红色错误、紫色中断。设false关闭颜色，不改变算法或续跑缓存。下层Python打印同步写入主日志；SNAP Java详细输出仍保留在缓存下的`gpt.log`，内部各步骤耗时在`timings.json`，主日志会打印它们的位置。
