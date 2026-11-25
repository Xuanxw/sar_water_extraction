# S1 VV＋DEM 配置

S1 水体分割使用 DFC Track 1 切片的第 1 波段 VV 和第 4 波段 DEM。下表各组采用同一划分（1,144 个训练块、287 个验证块），从随机权重训练 80 轮。VV 用训练集共享的 P02/P98 映射到 [0,1]，DEM 保留米制高程，再按各通道的训练集均值和标准差标准化。[输入约定与完整训练命令](../../docs/s1_shared_normalization.md)

| 网络 | 存储值 · BCE | 对数存储值 · BCE | 存储值 · BCE＋Dice |
| --- | --- | --- | --- |
| PVTv2-B2＋DeepLabv3+ | [训练 YAML](train_pvt_deeplabv3plus_trainnorm80.yaml) | [训练 YAML](train_pvt_deeplabv3plus_log_trainnorm80.yaml) | [训练 YAML](train_pvt_deeplabv3plus_trainnorm80_bce_dice.yaml) |
| MobileNetV2 U-Net | [训练 YAML](train_mobilenetv2_unet_trainnorm80.yaml) | [训练 YAML](train_mobilenetv2_unet_log_trainnorm80.yaml) | [训练 YAML](train_mobilenetv2_unet_trainnorm80_bce_dice.yaml) |
| Sentinel-1 Water U-Net | [训练 YAML](train_s1water_unet_trainnorm80.yaml) | [训练 YAML](train_s1water_unet_log_trainnorm80.yaml) | — |
| U-Net++（S1 Water 编码器） | [训练 YAML](train_s1water_unetpp_trainnorm80.yaml) | [训练 YAML](train_s1water_unetpp_log_trainnorm80.yaml) | — |

对数对照先对有效 VV 存储值计算 `10 × log10(max(x, 1e-6))`，再用训练块估计对数域 P02/P98，验证块复用同一对端点。有效零值保留，NoData 仍按原掩膜处理。DFC 整数到物理功率的换算关系尚未核实，因此这组实验评价对数压缩对分割的影响。与每行第一列 BCE 结果配对比较，输出保存到各自的 `*_vv_dem_log_trainnorm80` 目录。

当前参考模型是 PVT＋DeepLabv3+ 的 BCE＋Dice 第 63 轮。在内部验证集上，阈值 0.53 对应 Precision 80.03%、Recall 70.48%、F1 74.95%。在仓库根目录激活 `sar-water` 环境后，按用途把 YAML 传给相应脚本。

```powershell
python -u src/supervised/train.py configs/s1/train_pvt_deeplabv3plus_trainnorm80_bce_dice.yaml
```

| 复核内容 | 传给 `scripts/models/calibrate_s1_unet_bn.py` 的配置 |
| --- | --- |
| 四种网络 BCE 权重的完整验证集阈值扫描，含 PVT 末轮 | [review_trainnorm80_thresholds.yaml](review_trainnorm80_thresholds.yaml) |
| U-Net 第 35 轮和第 80 轮的 BN 校准及阈值扫描 | [review_trainnorm80_bn_unet.yaml](review_trainnorm80_bn_unet.yaml) |
| PVT 与 MobileNetV2 两种损失的完整验证集阈值扫描 | [review_trainnorm80_losses.yaml](review_trainnorm80_losses.yaml) |

两组 BCE＋Dice 训练均完成后运行损失对照。其余复核步骤及输出字段见[复核说明](../../docs/s1_trainnorm80_review.md)。

```powershell
python -u scripts/models/calibrate_s1_unet_bn.py configs/s1/review_trainnorm80_losses.yaml
```

## PVT＋DeepLabv3+ 地形输入对照

本轮按同一清单使用 1,144 个训练块和 270 个验证块（原 287 个验证块中暂时排除 17 个待复核标签）。四组均从头训练 80 轮，使用同一随机种子、BCE、对数 VV 和训练集共享 P02/P98。模型输入固定为 VV、高程、坡度三个通道，停用的通道恒为零。每组都使用第 1、4 波段的共同有效域。

| 配置后缀 | 启用的输入 | 比较目的 |
| --- | --- | --- |
| `vv_only` | VV | 取得单极化参照 |
| `elevation` | VV＋高程 | 测量绝对高程的贡献 |
| `slope` | VV＋坡度 | 测量局部坡度的贡献 |
| `elevation_slope` | VV＋高程＋坡度 | 检验同时提供两类地形信息的收益 |

DFC 第 4 波段是已对齐的 Copernicus 高程。坡度由这幅高程按 DFC 名义 10 米格距计算，两侧各取相距中心 30 米的样本，得到角度。该值是从分块估计的坡度。三通道采用训练集统计量统一标准化。历史双通道权重另作参考，四组之间按同一验证掩膜和 0.50 阈值比较 Precision、Recall、F1、IoU。[DFC 数据规格](https://www.grss-ieee.org/community/technical-committees/2024-ieee-grss-data-fusion-contest/)

在仓库根目录依次启动，结果分别写入 `outputs/s1/pvt_deeplabv3plus_log_terrain_<后缀>` 下的时间戳子目录。

```powershell
foreach ($group in 'vv_only', 'elevation', 'slope', 'elevation_slope') {
    $config = "configs/s1/train_pvt_deeplabv3plus_log_terrain_${group}.yaml"
    python -u src/supervised/train.py $config
    if ($LASTEXITCODE -ne 0) { throw "训练中断 $config" }
}
```

## 历史配置与密云推理

早期实验按每块有效 VV 的经验秩归一化，包含四个初始配置、四个扩训配置、一份 BN 对照和一份 PVT 损失对照。需要复现时，在仓库根目录恢复[历史配置包](../../docs/archive/s1_rank_configs.zip)，再按[扩训说明](../../docs/s1_extended_training.md)或[BN 与损失对照说明](../../docs/s1_bn_loss_experiment.md)运行。

```powershell
python -m zipfile -e docs/archive/s1_rank_configs.zip .
```

密云 7 景是浮点 dB。共享 P02/P98 权重依赖 DFC 的存储值，直接推理密云前须核实两者的单位转换。已有逐窗秩权重可按原流程推理密云。下面的命令需要先恢复上述配置，并准备好相应训练结果及 GLO-30 瓦片。

```powershell
$cfg = @(
    'configs/s1/train_s1water_unet.yaml'
    'configs/s1/train_pvt_deeplabv3plus.yaml'
    'configs/s1/train_mobilenetv2_unet.yaml'
)
$grd = 'E:\InSAR\水体提取\北京密云区\S1_Water\db'
$dem = 'E:\InSAR\水体提取\北京密云区\data_cache\auxiliary\tiles\copernicus'
python -u scripts/models/predict_best_s1.py @cfg $grd $dem
```

密云数据以 VV、VH 同时为 0 dB 标记无覆盖区。DEM 双线性插值到 SAR 网格，保留合法 0 米高程。密云缺少逐像元真值，现有推理可用于目视检查岸线、山影和窗口接缝。

## 天仪迁移与联合训练的后续对照

| 训练方式 | 在哪儿评估 | 回答的问题 |
| --- | --- | --- |
| S1 专用，直接预测天仪 | 天仪留出区 | 零目标标签时的迁移起点 |
| 天仪专用，从头训练 | 同一天仪留出区 | 目标域有标签时的参考精度 |
| S1 初始化，在天仪全量微调 | 同一天仪留出区，另测 S1 | 源域权重是否有益，以及微调后保留多少 S1 能力 |
| S1＋天仪联合训练，共享一套权重 | 两个传感器各自的留出区 | 一套权重能否兼顾两端 |

天仪的 BC2 和 BC3 分开报告，再汇总整体指标。先锁定天仪的跨区域测试景，并把同一事件、相邻或重叠窗口留在同一折。实验使用相同天仪标签预算，记录总更新步数与各传感器曝光次数。联合训练按传感器平衡抽样，核对训练与部署时的分辨率、窗口覆盖的实际地面尺度、有效掩膜以及水体标注范围。保留天仪原始功率域与 S1 dB 的处理记录。使用相对强度仅统一模型输入表示，两种传感器仍可能存在入射角、噪声和成像模式的差异。

以共同有效像元报告 Precision、Recall、F1、IoU，再按景或事件报告指标及山影、岸线、农田的误判。若目标是 BC3 三项指标均达到 0.80，在冻结的 BC3 测试集上分别验收。结论不稳时补做不同随机种子。若共享权重持续落后于专用模型，再测共享编码器＋传感器适配层。灾前/灾后变化、多专家路由与水深各需单独的任务定义和评估。

## 外部数据与设计依据

- **常态水体。** S1S2-Water 提供 65 组 S1、S2、人工质检水体掩膜，并配有 Copernicus DEM，适合做 VV＋DEM 的外部测试。[DLR 数据发布页](https://zenodo.org/records/11278238)
- **洪水事件。** Sen1Floods11 可按事件评估迁移。使用人工标注作精度参照，另行记录弱标签的训练用途。[数据与标签说明](https://github.com/cloudtostreet/Sen1Floods11)
- **产品路线。** 共享讨论提出了传感器专用权重、共享权重和适配层等候选。ICEYE 公开材料说明其融合 SAR、第三方数据与机器学习，未披露足以复现其洪水模型的输入张量和权重设计。[共享讨论](https://chatgpt.com/share/6afc4be0-b2ec-83e8-b76d-8a65d40640d7?ogimg=plain) · [ICEYE Flood Insights](https://www.iceye.com/insurance/sizing-losses)
