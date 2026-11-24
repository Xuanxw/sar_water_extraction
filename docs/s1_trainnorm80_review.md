# S1 新归一化基线：阈值、BN 与损失复核

本轮沿用训练集共享 VV P02/P98 和米制 DEM，固定 1144 个训练块、287 个验证块。已有 BCE 权重保持不变。长任务由使用者手动启动，不并发运行下列 GPU 任务。

在仓库根目录先执行：

```powershell
cd E:\Code_projects\sar_water_extraction
conda activate sar-water
```

## 1. 原始权重阈值扫描

```powershell
python -u scripts/models/calibrate_s1_unet_bn.py configs/s1/review_trainnorm80_thresholds.yaml
```

该配置为 `run.MODE: thresholds`，不会更新模型权重或 BN，也不读取训练影像像元。扫描原始最佳权重：PVT 第59轮、MobileNetV2 第14轮、U-Net++ 第37轮、U-Net 第35轮，另加 PVT 第80轮。所有模型在完整验证集上逐块推理一次，同时统计 0.01～0.99、步长0.01的阈值。

结果位于 `outputs/s1/review_trainnorm80/thresholds/`：

- `summary.csv`：固定0.5和最佳验证F1阈值两组结果。
- `precision_target.csv`：Precision≥0.80条件下的最大Recall及对应阈值。`unattainable` 表示扫描范围内不可达，指标留空；不是零分或阈值0。
- `threshold_scan.csv`：全部阈值的总体指标。
- `water_groups.csv`、`per_patch.csv`：按标签水体占比分组和逐块指标。

优先比较阈值调整能否提高召回，以及极少水体块的误检是否增加。最优阈值由内部验证集选择，后续须在独立测试数据上验证。

## 2. U-Net BN 校准

```powershell
python -u scripts/models/calibrate_s1_unet_bn.py configs/s1/review_trainnorm80_bn_unet.yaml
```

对 U-Net 第35轮最佳与第80轮末轮分别执行原始扫描、BN校准、校准后扫描。校准只用全部1144个训练影像，不用训练标签或验证数据更新BN；权重冻结，Dropout关闭，按前向顺序逐层计算统计量。

结果位于 `outputs/s1/review_trainnorm80/bn_unet/`。除步骤1的表格外，`bn_comparison.csv` 分别记录固定0.5、各自最佳阈值下的原始/校准差异。各检查点目录保存 `calibrated.pth`，仅用于推理，不作为后续训练续跑检查点。

逐层校准会多次遍历训练集，耗时可能明显长于单次验证。比较时以本脚本重新计算的原始模型为对照：本脚本用FP32并关闭TF32，历史训练验证使用AMP，二者可能有小差异。

## 3. BCE 与 BCE＋Dice

保留已完成的 BCE 实验作为控制组，只新增 PVT 和 MobileNetV2 两次 BCE＋Dice 训练。配置与对应 `trainnorm80` 基线仅有损失名称及输出目录两项不同；BCE:Dice=1:1，随机初始化，完整训练集、80轮单次余弦、学习率1e-4、batch 2、seed 42及增强均不变。使用现有有效像元BCE均值和逐图Soft Dice均值（smooth=1）。

```powershell
$configs = @(
    'configs/s1/train_pvt_deeplabv3plus_trainnorm80_bce_dice.yaml'
    'configs/s1/train_mobilenetv2_unet_trainnorm80_bce_dice.yaml'
)
foreach ($config in $configs) {
    python -u src/supervised/train.py $config
    if ($LASTEXITCODE -ne 0) { throw "训练中断：$config" }
}
```

独立输出目录分别为：

- `outputs/s1/pvt_deeplabv3plus_vv_dem_trainnorm80_bce_dice/`
- `outputs/s1/mobilenetv2_unet_vv_dem_trainnorm80_bce_dice/`

两次训练完整结束后，运行损失对照扫描：

```powershell
python -u scripts/models/calibrate_s1_unet_bn.py configs/s1/review_trainnorm80_losses.yaml
```

`paths.TRAIN_CONFIGS` 为两个新训练配置。脚本复用现有选模检查，选择配置一致、最近完整80轮实验的 `best_f1.pth`，并打印实际路径和轮次；未完成训练会报错。BCE控制权重固定为本轮PVT第59轮和MobileNetV2第14轮，不会改选其他实验。结果写入 `outputs/s1/review_trainnorm80/loss_comparison/`，比较两种损失在固定0.5和扫描阈值下的结果，不校准BN。

按水体 Precision、Recall、F1 和分组误检漏检比较两种损失；BCE 与 BCE＋Dice 的数值尺度不同。

## 中断与复用

- 扫描与BN中断后，原命令重跑：复用已完成验证块和BN层。模式、输入检查点、源文件版本或数值协议改变时，必须换输出目录。
- 三个复核配置的输出目录相互独立，不在旧结果目录中追加不同协议的结果。
- 新训练中断后，把对应训练YAML的 `RESUME_CHECKPOINT` 指向其新实验的 `last.pth`；若训练已完成而验证中断，可用 `train_end.pth`。保持总轮数80，不设置EXTENSION。
- 损失扫描运行后若又训练了更新的完整实验，自动选择会改变；应换损失扫描输出目录，或用明确的CHECKPOINTS固定已有选择。

当前阶段不改变标签、采样、网络结构或TYSAR输入协议，也不使用验证结果更新训练归一化端点。
