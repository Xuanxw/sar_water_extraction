# S1 VV＋DEM：四网络共享归一化基线

本轮同时比较 S1 Water U-Net、PVT＋DeepLabv3+、MobileNetV2 U-Net 和 U-Net++。四组都从随机权重训练 80 轮，不预先指定某个网络作为最终方案。原 rank 实验及其结果保留。

## 输入与统计范围

`data.PATCH_REPRESENTATION: vv_train_p02_p98_dem_m` 使用固定训练划分内全部共同有效 VV 像元，计算一组 P02/P98：

```text
VV_relative = clip((VV_stored - train_P02) / (train_P98 - train_P02), 0, 1)
```

训练、验证和同存储单位的评价都使用这一组端点，不按小块重新排名或重新拉伸。因此同一 VV 数值不会因窗口内水陆比例不同而改变；区间内部的强度差保持线性关系，区间外会截断。DEM 按文件 scale/offset 恢复米值，不做分位拉伸。随后两个通道分别使用训练集均值和标准差，因而模型最终输入不限定在 `[0,1]`。

- 拟合只读取训练影像，不使用验证影像、标签类别或标签水体比例；有效域为所选 VV 和 DEM 的共同影像有效域。合法的 0 值保留。
- 采用逐块读取、临时 float32 文件求精确分位数。临时文件位于系统临时目录，结束后删除；1144 个完整 512×512 训练块约需 1.2 GB 临时空间。
- 每个实验保存 `relative_normalization.json`，包括端点、有效像元数、训练样本签名、波段和输入域。签名核对相对文件名单及波段顺序，不是 TIFF 内容哈希。检查点也保存同一参数及后续 mean/std。
- 四组读取相同固定划分，因此会得到相同参数。续跑和 BN 校准从检查点恢复，不重新估计端点；缺参数或划分不匹配时明确报错。
- 若训练集 P02=P98，程序报错，不能把退化数据静默变成有效输入。

这是 **DFC 存储值的训练集统一缩放**，不是整景归一化，也不是辐射校正。它保留同一数据集的块间亮暗，但不保证不同传感器或不同存储编码的数值物理可比。固定训练参数也不会自动消除未知的新景增益。

## 四份配置

| 网络 | 配置 | 输出目录（`outputs/s1/` 下） |
|---|---|---|
| S1 Water U-Net | `configs/s1/train_s1water_unet_trainnorm80.yaml` | `s1water_unet_vv_dem_trainnorm80` |
| PVT＋DeepLabv3+ | `configs/s1/train_pvt_deeplabv3plus_trainnorm80.yaml` | `pvt_deeplabv3plus_vv_dem_trainnorm80` |
| MobileNetV2 U-Net | `configs/s1/train_mobilenetv2_unet_trainnorm80.yaml` | `mobilenetv2_unet_vv_dem_trainnorm80` |
| U-Net++ | `configs/s1/train_s1water_unetpp_trainnorm80.yaml` | `s1water_unetpp_vv_dem_trainnorm80` |

沿用原固定划分、VV＋DEM、batch 2、完整训练集、BCE、AdamW、增强和种子 42。四组均为 `PRETRAINED:false`、`RESUME_CHECKPOINT:null`、`EPOCHS:80`，初始学习率 `1e-4`，单次 80 轮余弦退火到 0，不设置 `EXTENSION`。每轮保留完整验证和质量图。

旧实验采用 20＋60 轮的分阶段日程，旧 U-Net++ 在第 9 轮切换；本轮统一从头 80 轮。因此旧新差值同时包含输入表示与学习率日程变化，不能全部解释为归一化收益。要严格隔离归一化作用，需要补同日程的 rank 对照。

## 顺序运行

等当前四组实验结束后，在仓库根目录运行。任一网络失败即停止，避免跳过错误后继续后面的实验。

```powershell
cd E:\Code_projects\sar_water_extraction
conda activate sar-water

$configs = @(
    'configs/s1/train_s1water_unet_trainnorm80.yaml'
    'configs/s1/train_pvt_deeplabv3plus_trainnorm80.yaml'
    'configs/s1/train_mobilenetv2_unet_trainnorm80.yaml'
    'configs/s1/train_s1water_unetpp_trainnorm80.yaml'
)
foreach ($config in $configs) {
    python -u src/supervised/train.py $config
    if ($LASTEXITCODE -ne 0) { throw "训练中断：$config" }
}
```

中断后只重启受影响的配置：把该 YAML 的 `RESUME_CHECKPOINT` 改为其新结果目录中的 `last.pth`；若本轮训练完成、验证中断，则用 `train_end.pth`。保持总轮数 80 和输入表示不变，不增加 `EXTENSION`。检查点会恢复固定归一化参数、优化器、随机状态和已有学习率进度。

## 评价和跨数据推理

四组分别报告整体 Precision、Recall、F1、IoU，并按水体占比检查少水体块和大水体块；使用 `best_f1.pth` 选 F1 最佳模型，`best.pth` 仍表示验证 loss 最低。独立 BN/阈值扫描脚本已支持从新检查点恢复同一变换。选择最佳轮次和阈值使用的是内部验证集，不能视为独立测试精度。

现有 DFC TIFF 缺少已核实的物理缩放与父景映射。密云输入是浮点 dB，不能直接使用 DFC 存储值端点。`predict_best_s1.py` 支持比较四份配置，但它的 dB GRD 推理入口会明确拒绝这类新权重；不会退回逐窗 rank 或猜测转换公式。需补齐 DFC 与目标数据的单位映射后，再准备密云推理对照。

TYSAR 的 `image_process.py` 已有整景 `scene_p02_p98`，统计范围和当前 DFC 方案不同。未来联合训练需要另行统一输入协议；本次没有修改 TYSAR 预处理或本地服务器配置。
