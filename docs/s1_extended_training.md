# S1 VV＋DEM：续训至 80 轮

本文记录旧版逐块秩归一化（rank）实验。复现下列命令时，先在仓库根目录恢复[历史 YAML](archive/s1_rank_configs.zip)，再核对本机的数据和检查点路径。当前 80 轮实验使用[共享归一化配置](../configs/s1/README.md)。

```powershell
python -m zipfile -e docs/archive/s1_rank_configs.zip .
```

## 实验目的

保留已有训练计算，检查增加训练预算后，验证集水体 F1、Precision、Recall 是否继续改善。输入、标签、固定划分、网络、BCE、batch 2、AdamW、增强和随机种子保持不变。

截至 2026-11-19，前三组已完成 20 轮，末轮检查点的学习率均为 0。PVT 最后五轮验证 F1 为 59.79%、60.57%、59.23%、57.63%、58.40%；U-Net 和 MobileNetV2 的验证波动更大。因此延长训练值得验证，但现有结果不能确认轮数不足是低 F1 的主要原因。BN 与阈值对照仍需单独完成。

## 本轮设置

| 配置 | 起点 | 后续训练 |
|---|---|---|
| `configs/s1/train_s1water_unet_extend80.yaml` | 原实验 `last.pth`，第 20 轮 | 第 21–80 轮 |
| `configs/s1/train_pvt_deeplabv3plus_extend80.yaml` | 原实验 `last.pth`，第 20 轮 | 第 21–80 轮 |
| `configs/s1/train_mobilenetv2_unet_extend80.yaml` | 原实验 `last.pth`，第 20 轮 | 第 21–80 轮 |
| `configs/s1/train_s1water_unetpp_extend80.yaml` | 原实验 `train_end.pth`，第 9 轮训练完成 | 先重跑第 9 轮验证，再训练第 10–80 轮 |

原配置中的 `LEARNING_RATE: 0.0001` 保留，用于核对原实验。后续训练明确使用独立的学习率计划：

```yaml
training:
  EPOCHS: 80
  EXTENSION:
    learning_rate: 0.00005
    min_lr: 0.000001
```

新阶段从 `5×10⁻⁵` 开始余弦下降，完成第 80 轮后降至 `1×10⁻⁶`。前三组的新周期为 60 轮；U-Net++ 为 71 轮。补跑 U-Net++ 第 9 轮验证时，不消耗新阶段的学习率步数，也不重复该轮训练。

保留模型权重、优化器动量、AMP 和随机状态；只重设后续学习率计划。首次扩训写入带 `_extend80` 的独立输出目录，保存来源记录，原实验的指标、最佳权重和图件保留。新目录单独选择本阶段的最佳验证 F1：前三组从第 21 轮开始；U-Net++ 包含补跑的第 9 轮验证。最终需要同时比较原实验最佳模型和扩训最佳模型。

这是“原训练＋新余弦阶段”的扩训实验，不等价于从第一轮就采用 80 轮余弦计划。U-Net++ 的切换位置早于其他三组，因此本轮不能仅凭四组最终排名，将差异全部归因于网络结构。固定跑到 80 轮，暂不增加早停、BCE＋Dice、采样或输入表示等新因素。

## 启动

在本地仓库根目录执行，顺序完成前三组。任一任务出错即停止，不自动跳过：

```powershell
cd E:\Code_projects\sar_water_extraction
conda activate sar-water
$configs = @(
    'configs/s1/train_pvt_deeplabv3plus_extend80.yaml'
    'configs/s1/train_s1water_unet_extend80.yaml'
    'configs/s1/train_mobilenetv2_unet_extend80.yaml'
)
foreach ($config in $configs) {
    python -u src/supervised/train.py $config
    if ($LASTEXITCODE -ne 0) { throw "训练中断：$config" }
}
```

归档的 U-Net++ YAML 已指向扩训目录里的 `last.pth`，用于中断后续跑。如果要从第 9 轮重新做这项历史对照，先将 `paths.RESUME_CHECKPOINT` 设为 `outputs/s1/s1water_unetpp_vv_dem/20261118_170115_824068/train_end.pth`，再运行下列命令。

```powershell
python -u src/supervised/train.py configs/s1/train_s1water_unetpp_extend80.yaml
```

每轮继续输出完整验证指标和质量图，保存 `last.pth`、验证前的 `train_end.pth` 及最佳模型。优先观察 Recall 的变化及 Precision 是否下降，再看整体 F1；保留对大水体漏检和少水区域误检的检查。

## 中断后的续跑

首次扩训配置指向旧实验。扩训过程中中断后，需将所用 YAML 的 `paths.RESUME_CHECKPOINT` 改为**新扩训结果目录**中的检查点：

- 训练阶段中断：使用新目录中上一轮的 `last.pth`。
- 验证阶段中断：使用新目录中本轮的 `train_end.pth`，只补跑验证。

保持 `EPOCHS: 80` 和 `EXTENSION` 不变，再执行该配置。续跑会恢复新阶段已保存的学习率进度，不会再次从 `5×10⁻⁵` 开始。若仍指向旧实验检查点，会创建另一条扩训分支，而不是继续刚才的分支。

不要用 `calibrated.pth` 继续训练：BN 校准模型是独立评价/推理产物。扩训完成后，用新阶段最佳模型和末轮模型做完整验证集阈值扫描与 BN 对照；归档的 20 轮 BN 配置只对应原检查点。

方案参考了[共享讨论](https://chatgpt.com/share/6afdb45d-1bb4-83e8-bddd-2d6fe70b411e)中的验证曲线检查与完整训练状态续训思路；具体学习率和续跑方式按本仓库实际检查点设置。验证集同时用于选模，最佳指标属于内部模型选择结果，不是独立测试精度。
