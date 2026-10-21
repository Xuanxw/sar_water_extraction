"""统一训练入口：修改顶部 METHOD 后运行 python src/train.py。"""
import csv
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

import timm

from datasets.water_dataset import IGNORE_INDEX, WaterDataset
from models.unet import UNet
from models.pvt import PVTWaterNet, PVTUANet
from models.resnet import ResNet50WaterNet

"""
| 模块 | 属于 Transformer 吗？ | 主要计算 |
|---|---|---|
| **PVT 骨干** | **是** | 多头空间缩减注意力、MLP、LayerNorm、残差 |
| **MBDC** | 否 | 多分支空洞卷积、拼接、残差 |
| **FPN** | 否 | 卷积、上采样、逐层相加 |
| **语义解码头** | 否 | 卷积、上采样、多尺度相加 |
| **PSM / CGM** | 不属于完整 Transformer Block | 空间注意力／通道调制 |
| **UAFM** | 否 | 不确定性加权、卷积与特征融合 |
| **普通 U-Net / ResNet** | 否 | 主要使用卷积结构 |
"""
ROOT = Path(__file__).resolve().parents[1]
METHOD = "uafnet"  # 同级方法：unet、pvt、resnet50、uanet、uafnet。
STAGE = "uafm"    # 仅 uanet 使用；uafnet 固定为 MBDC + FPN + 三级 UAFM，不含 PSM/CGM。
IMAGE_DIR = ROOT / "data/Track1/train/images"
LABEL_DIR = ROOT / "data/Track1/train/labels"
OUTPUT_DIR = None  # None 按方法自动选择原实验目录；试跑时可指定其他路径。
BANDS = (1, 2)    # 仍使用已检查的两通道。论文用 VH/VV/VH；确认物理顺序后再调整，不能猜测。
BASE_CHANNELS = 32       # U-Net 起始通道数。
DECODER_CHANNELS = 32    # PVT、ResNet、UANet 和 UAFNet 的解码通道数。
PRETRAINED = True        # 上述骨干加载 ImageNet 权重；解码部分随机初始化。
BATCH_SIZE = 2
EPOCHS = 20
LEARNING_RATE = 1e-4 if METHOD == "uafnet" else 1e-3
# UAFNet 论文未给出权重衰减、总轮数和全部增强角度；以下是本项目的明确选择。
WEIGHT_DECAY = 0.01
TRAIN_AUGMENTATION = True  # 仅 uafnet 使用：随机 90° 旋转和水平翻转，非论文精确的 12 倍扩增。
VAL_RATIO = 0.2
SEED = 42
MAX_SAMPLES = None
ABNORMAL_F1_THRESHOLD = 0.001


class WaterSegmentationTrainer:
    """有监督水体语义分割训练器，统一管理基线和 UAFNet 的训练、验证和保存。"""

    def __init__(
        self, model_class, model_kwargs, model_info, image_dir, label_dir, output_dir,
        bands=(1, 2), batch_size=2, epochs=20, learning_rate=1e-3,
        val_ratio=0.2, seed=42, max_samples=None, abnormal_f1_threshold=0.001,
        loss_name="ce", optimizer_name="adam", weight_decay=0.0,
        cosine_schedule=False, train_augmentation=False,
    ):
        """准备固定划分、输入统计量、模型、优化器和实验目录。"""
        self.image_dir, self.label_dir = Path(image_dir), Path(label_dir)
        self.output_dir = Path(output_dir)
        self.model_kwargs, self.model_info = dict(model_kwargs), dict(model_info)
        self.bands, self.batch_size, self.epochs = bands, batch_size, epochs
        self.learning_rate, self.val_ratio, self.seed = learning_rate, val_ratio, seed
        self.max_samples = max_samples
        self.abnormal_f1_threshold = abnormal_f1_threshold
        self.loss_name = loss_name
        self.optimizer_name, self.weight_decay = optimizer_name, weight_decay
        self.train_augmentation = train_augmentation
        torch.manual_seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = self.device.type == "cuda"

        # Subset 只保存索引；固定种子和文件列表时，所有方法使用同一划分。
        self.dataset = WaterDataset(self.image_dir, self.label_dir, bands)
        if max_samples is not None:
            self.dataset.image_paths = self.dataset.image_paths[:max_samples]
        val_size = max(1, int(len(self.dataset) * val_ratio))
        self.train_set, self.val_set = random_split(
            self.dataset, [len(self.dataset) - val_size, val_size],
            generator=torch.Generator().manual_seed(seed),
        )
        self.train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True, num_workers=0)
        self.val_loader = DataLoader(self.val_set, batch_size=batch_size, shuffle=False, num_workers=0)
        print(f"设备：{self.device}；训练/内部验证：{len(self.train_set)}/{len(self.val_set)}", flush=True)
        self.calculate_normalization()

        self.model = model_class(**self.model_kwargs).to(self.device)
        print(f"模型：{self.model_info['model_name']}；构建参数：{self.model_kwargs}", flush=True)
        print(f"损失：{loss_name}；优化器：{optimizer_name}；初始学习率：{learning_rate}；"
              f"余弦退火：{cosine_schedule}；训练增强：{train_augmentation}", flush=True)
        # UAFNet 单通道表示水体 logit；其余历史实验仍使用双类别 CE。
        self.criterion = (nn.BCEWithLogitsLoss(reduction="none") if loss_name == "bce"
                          else nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX))
        if optimizer_name == "adamw":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
                          if cosine_schedule else None)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # 每次运行单独保存，不覆盖历史实验；CSV 字段沿用原有诊断记录。
        self.run_dir = self.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.metrics_path = self.run_dir / "metrics.csv"
        self.metric_fields = [
            "epoch", "phase", "loss", "iou", "f1", "precision", "recall",
            "tn", "fp", "fn", "tp", "valid_pixels", "ignored_pixels", "skipped_batches",
            "true_water_ratio", "pred_water_ratio", "learning_rate", "amp_scale",
            "amp_skipped_steps", "nonfinite_loss_batches", "elapsed_seconds",
        ]
        with self.metrics_path.open("w", newline="", encoding="utf-8-sig") as stream:
            csv.DictWriter(stream, fieldnames=self.metric_fields).writeheader()
        print(f"本次实验目录：{self.run_dir}", flush=True)
        self.best_loss = float("inf")
        self.best_f1 = float("-inf")
        self.train_files = [self.dataset.image_paths[i].relative_to(self.image_dir).as_posix() for i in self.train_set.indices]
        self.val_files = [self.dataset.image_paths[i].relative_to(self.image_dir).as_posix() for i in self.val_set.indices]

    def calculate_normalization(self):
        """只用训练子集的有效影像像素，逐通道计算均值和标准差。"""
        channel_sum = np.zeros(len(self.bands), dtype=np.float64)
        channel_square_sum = np.zeros_like(channel_sum)
        pixel_count = 0
        for number, sample in enumerate(self.train_set, 1):
            # [C,H,W] 按空间掩膜取值后为 [C,N]；不按标签类别筛选。
            values = sample["image"].numpy()[:, sample["valid_mask"].numpy()].astype(np.float64)
            channel_sum += values.sum(axis=1)
            channel_square_sum += np.square(values).sum(axis=1)
            pixel_count += values.shape[1]
            if number % 100 == 0:
                print(f"归一化统计：{number}/{len(self.train_set)}", flush=True)
        if pixel_count == 0:
            raise ValueError("训练部分没有有效影像像素")
        mean = channel_sum / pixel_count
        std = np.sqrt(np.maximum(channel_square_sum / pixel_count - mean ** 2, 0))
        # [1,C,1,1] 可广播到任意批次；下限避免除零。
        self.mean = torch.tensor(mean, dtype=torch.float32, device=self.device).view(1, -1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32, device=self.device).clamp_min(1e-6).view(1, -1, 1, 1)
        print(f"输入通道：{self.bands}；均值：{mean}；标准差：{std}", flush=True)

    def run_epoch(self, epoch, phase, loader):
        """处理一轮训练或验证：批次计算、累计指标并追加 CSV。"""
        training = phase == "train"
        self.model.train(training)  # 训练/评价模式影响 BN，梯度开关单独控制。
        loss_sum = valid_pixels = ignored_pixels = skipped_batches = 0
        amp_skipped_steps = nonfinite_loss_batches = 0
        confusion = torch.zeros(4, dtype=torch.int64, device=self.device)
        phase_start = perf_counter()

        for step, batch in enumerate(loader, 1):
            image = batch["image"].to(self.device)
            label = batch["label"].to(self.device)
            valid = label != IGNORE_INDEX
            count = int(valid.sum().item())
            ignored_pixels += label.numel() - count
            if count == 0:
                skipped_batches += 1
                continue
            image = (image - self.mean) / self.std
            image = image.masked_fill(~batch["valid_mask"].to(self.device).unsqueeze(1), 0)
            if training:
                self.recent_train_files = (self.recent_train_files + list(batch["name"]))[-20:]
                self.optimizer.zero_grad(set_to_none=True)

            if training and self.train_augmentation:
                # 影像与标签同步变换；影像无效处已置零，标签无效标记随之变换。
                for i in range(image.shape[0]):
                    turns = int(torch.randint(4, ()).item())
                    if image.shape[-2] != image.shape[-1]:
                        turns = (turns % 2) * 2  # 非正方形只转 0°/180°，保持批次尺寸。
                    image[i] = torch.rot90(image[i], turns, dims=(-2, -1))
                    label[i] = torch.rot90(label[i], turns, dims=(-2, -1))
                    if torch.rand(()).item() < 0.5:
                        image[i] = image[i].flip(-1)
                        label[i] = label[i].flip(-1)
                # 无效影像在标准化后已置零，随影像旋转；标签掩膜重新生成。
                valid = label != IGNORE_INDEX

            with torch.set_grad_enabled(training):
                with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    outputs = self.model(image)
                    predictions = outputs if isinstance(outputs, (tuple, list)) else (outputs,)
                    logits = predictions[-1]
                    # 训练对粗预测与三级细化预测等权监督；验证仅评价最后一个输出。
                    supervised = predictions if training else (logits,)
                    if self.loss_name == "bce":
                        # -1 不能作为 BCE 目标；先填 0，再只选有效位置求均值。
                        target = label.clamp_min(0).float()
                        loss = sum(self.criterion(output[:, 0].float(), target)[valid].mean()
                                   for output in supervised)
                    else:
                        loss = sum(self.criterion(output, label) for output in supervised)
                if not torch.isfinite(loss).item():
                    raise FloatingPointError(f"Epoch {epoch} {phase} 第 {step} 批 loss 非有限；文件：{batch['name']}")
                if training:
                    previous_scale = self.scaler.get_scale()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    if self.scaler.get_scale() < previous_scale:
                        amp_skipped_steps += 1

            # 按有效像素累计损失与混淆矩阵，不直接平均每批 F1。
            loss_value = loss.item()
            if not np.isfinite(loss_value):
                nonfinite_loss_batches += 1
            loss_sum += loss_value * count
            valid_pixels += count
            # 单通道 sigmoid>=0.5 等价于 logit>=0；双类别仍使用 argmax。
            prediction = ((logits.detach()[:, 0] >= 0).long() if self.loss_name == "bce"
                          else logits.detach().argmax(dim=1))
            # 真实值*2+预测值：0=TN、1=FP、2=FN、3=TP。
            confusion += torch.bincount(label[valid] * 2 + prediction[valid], minlength=4)
            if step % 50 == 0:
                print(f"Epoch {epoch} {phase}：{step}/{len(loader)} 批", flush=True)

        if valid_pixels == 0:
            raise ValueError(f"{phase} 没有可用于损失计算的标签像素")
        tn, fp, fn, tp = confusion.tolist()
        iou = tp / (tp + fp + fn) if tp + fp + fn else float("nan")
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float("nan")
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else float("nan")
        metrics = dict(
            epoch=epoch, phase=phase, loss=loss_sum / valid_pixels, iou=iou, f1=f1,
            precision=precision, recall=recall, tn=tn, fp=fp, fn=fn, tp=tp,
            valid_pixels=valid_pixels, ignored_pixels=ignored_pixels, skipped_batches=skipped_batches,
            true_water_ratio=(tp + fn) / valid_pixels, pred_water_ratio=(tp + fp) / valid_pixels,
            learning_rate=self.optimizer.param_groups[0]["lr"], amp_scale=self.scaler.get_scale(),
            amp_skipped_steps=amp_skipped_steps, nonfinite_loss_batches=nonfinite_loss_batches,
            elapsed_seconds=perf_counter() - phase_start,
        )
        with self.metrics_path.open("a", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=self.metric_fields).writerow(metrics)
        print(f"Epoch {epoch:02d}/{self.epochs} {phase}：loss={metrics['loss']:.4f}，"
              f"水体 IoU={iou:.4f}，F1={f1:.4f}，P={precision:.4f}，R={recall:.4f}", flush=True)
        print(f"真实/预测水体比例：{metrics['true_water_ratio']:.4%}/{metrics['pred_water_ratio']:.4%}；"
              f"TN/FP/FN/TP：{tn}/{fp}/{fn}/{tp}；AMP 跳过更新：{amp_skipped_steps} 次", flush=True)
        return metrics

    def save_checkpoint(self, epoch):
        """保持最低验证 loss 选 best，另外保存 last 和满足条件的异常快照。"""
        metrics = self.phase_metrics["val"]
        epoch_loss = metrics["loss"]
        improved = epoch_loss < self.best_loss
        if improved:
            self.best_loss = epoch_loss
        abnormal = not np.isfinite(epoch_loss) or (
            metrics["tp"] + metrics["fn"] > 0 and metrics["f1"] < self.abnormal_f1_threshold
        )
        # 保留已有字段；state_dict 包含 BN 缓冲区，UAFNet 另存优化器等训练状态。
        checkpoint = {
            "model_state_dict": self.model.state_dict(), **self.model_info,
            "model_kwargs": self.model_kwargs, "in_channels": len(self.bands),
            "bands": self.bands, "mean": self.mean.cpu().flatten(), "std": self.std.cpu().flatten(),
            "ignore_index": IGNORE_INDEX, "epoch": epoch, "val_loss": epoch_loss,
            "best_val_loss": self.best_loss, "phase_metrics": self.phase_metrics,
            "seed": self.seed, "learning_rate": self.learning_rate, "batch_size": self.batch_size,
            "epochs": self.epochs, "val_ratio": self.val_ratio, "max_samples": self.max_samples,
            "image_dir": str(self.image_dir), "label_dir": str(self.label_dir),
            "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
            "device": str(self.device), "amp_enabled": self.use_amp,
            "gpu_name": torch.cuda.get_device_name(self.device) if self.use_amp else None,
            "abnormal_f1_threshold": self.abnormal_f1_threshold,
            "recent_train_files": self.recent_train_files,
            "train_files": self.train_files, "val_files": self.val_files,
        }
        if self.loss_name == "bce":
            # best 仍按 loss 保存；另存 F1 最佳，避免与日志最高 F1 混淆。
            checkpoint.update(loss_name=self.loss_name, optimizer_name=self.optimizer_name,
                              weight_decay=self.weight_decay, train_augmentation=self.train_augmentation,
                              optimizer_state_dict=self.optimizer.state_dict(),
                              scaler_state_dict=self.scaler.state_dict(),
                              scheduler_state_dict=self.scheduler.state_dict() if self.scheduler else None)
        filenames = ["last.pth"]
        if self.loss_name == "bce" and metrics["f1"] > self.best_f1:
            self.best_f1 = metrics["f1"]
            filenames.append("best_f1.pth")
        if self.loss_name == "bce":
            checkpoint["best_val_f1"] = self.best_f1
        if improved:
            filenames.append("best.pth")
        if abnormal:
            filenames.append("abnormal.pth")
        for filename in filenames:
            torch.save(checkpoint, self.run_dir / filename)
            print(f"已保存：{self.run_dir / filename}", flush=True)

    def train(self):
        """逐轮执行训练 → 验证 → 保存。"""
        for epoch in range(1, self.epochs + 1):
            self.phase_metrics = {}
            self.recent_train_files = []
            for phase, loader in (("train", self.train_loader), ("val", self.val_loader)):
                self.phase_metrics[phase] = self.run_epoch(epoch, phase, loader)
            # CSV 记录本轮实际使用的 lr；调度后保存，为下一轮保留调度器状态。
            if self.scheduler is not None:
                self.scheduler.step()
            self.save_checkpoint(epoch)


def create_trainer():
    """平级选择网络与保存目录，然后创建同一个公共训练器。"""
    if METHOD == "unet":
        model_class = UNet
        model_kwargs = dict(in_channels=len(BANDS), base_channels=BASE_CHANNELS)
        model_info = dict(model_name="unet", normalization="BatchNorm", base_channels=BASE_CHANNELS)
        experiment = "unet_baseline"

    elif METHOD == "pvt":
        model_class = PVTWaterNet
        model_kwargs = dict(in_channels=len(BANDS), num_classes=2,
                            decoder_channels=DECODER_CHANNELS, pretrained=PRETRAINED)
        model_info = dict(model_name="pvt", backbone="pvt_v2_b2.in1k",
                          normalization="LayerNorm + decoder BatchNorm", stage="backbone_projection_decoder")
        experiment = "pvt_baseline"

    elif METHOD == "resnet50":
        model_class = ResNet50WaterNet
        model_kwargs = dict(in_channels=len(BANDS), num_classes=2,
                            decoder_channels=DECODER_CHANNELS, pretrained=PRETRAINED)
        model_info = dict(model_name="resnet50", backbone="resnet50.tv_in1k",
                          normalization="BatchNorm", stage="backbone_projection_decoder")
        experiment = "resnet50_baseline"

    elif METHOD == "uanet":
        model_class = PVTUANet
        model_kwargs = dict(in_channels=len(BANDS), num_classes=2,
                            decoder_channels=DECODER_CHANNELS, pretrained=PRETRAINED, stage=STAGE)
        # 保留旧检查点名称 pvt_uanet，兼容既有加载器和实验记录。
        model_info = dict(model_name="pvt_uanet", backbone="pvt_v2_b2.in1k", stage=STAGE,
                          normalization="LayerNorm + BatchNorm", output_order="coarse_to_fine",
                          supervision="sum_CE_all_train_heads; final_head_validation")
        experiment = f"pvt_uanet_{STAGE}"

    elif METHOD == "uafnet":
        # 水体论文的单模型训练：单通道 BCE、四输出监督，不添加建筑网络的 PSM/CGM。
        model_class = PVTUANet
        model_kwargs = dict(in_channels=len(BANDS), num_classes=1,
                            decoder_channels=DECODER_CHANNELS, pretrained=PRETRAINED, stage="uafm")
        model_info = dict(model_name="pvt_uafnet", backbone="pvt_v2_b2.in1k", stage="uafm",
                          normalization="LayerNorm + BatchNorm", output_order="coarse_to_fine",
                          supervision="sum_BCE_all_train_heads; final_head_validation",
                          implementation="UAFNet paper-guided SAR adaptation; single PVT-B2",
                          paper_doi="10.1109/IGARSS53475.2024.10641026")
        experiment = "pvt_uafnet"

    else:
        raise ValueError("METHOD 应为 unet、pvt、resnet50、uanet 或 uafnet")

    if METHOD != "unet":
        model_info.update(decoder="SemanticFPNDecoder", decoder_channels=DECODER_CHANNELS,
                          timm_version=timm.__version__, pretrained=PRETRAINED)

    output_dir = ROOT / "outputs" / experiment if OUTPUT_DIR is None else Path(OUTPUT_DIR)
    print(f"训练方法：{METHOD}；实验目录：{output_dir}", flush=True)
    return WaterSegmentationTrainer(
        model_class=model_class, model_kwargs=model_kwargs, model_info=model_info,
        image_dir=IMAGE_DIR, label_dir=LABEL_DIR, output_dir=output_dir,
        bands=BANDS, batch_size=BATCH_SIZE, epochs=EPOCHS, learning_rate=LEARNING_RATE,
        val_ratio=VAL_RATIO, seed=SEED, max_samples=MAX_SAMPLES,
        abnormal_f1_threshold=ABNORMAL_F1_THRESHOLD,
        loss_name="bce" if METHOD == "uafnet" else "ce",
        optimizer_name="adamw" if METHOD == "uafnet" else "adam",
        weight_decay=WEIGHT_DECAY if METHOD == "uafnet" else 0.0,
        cosine_schedule=METHOD == "uafnet",
        train_augmentation=METHOD == "uafnet" and TRAIN_AUGMENTATION,
    )


if __name__ == "__main__":
    trainer = create_trainer()
    trainer.train()
