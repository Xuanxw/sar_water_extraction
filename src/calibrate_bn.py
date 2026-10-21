"""只用训练影像逐层校准 BN，比较前后验证指标，不重新训练权重。"""
import csv
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from datasets.water_dataset import WaterDataset
from models.checkpoint import load_model

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = ROOT / "outputs/unet_baseline"  # 其他模型/阶段改为其对应的实验目录。
RUN_DIR = None  # None 从 EXPERIMENT_DIR 选择最新实验；也可直接指定某次实验目录。
CHECKPOINT_NAME = "last.pth"  # 默认恢复最后一轮；改为 best.pth 可单独校准最佳轮。
CALIBRATION_SAMPLES = 128  # 固定取训练划分列表的前 128 幅，与此前成功恢复的方案一致。


class ChannelStatistics:
    """单层 BN 的输入统计器；作为前向钩子累计逐通道一阶、二阶矩。"""

    def __init__(self, channels, device):
        self.channel_sum = torch.zeros(channels, dtype=torch.float64, device=device)
        self.channel_square_sum = torch.zeros_like(self.channel_sum)
        self.pixel_count = 0

    def __call__(self, module, inputs):
        """PyTorch 在 BN 前向之前调用，inputs[0] 是 [B,C,H,W] 特征。"""
        feature = inputs[0].double()  # 只在统计时用双精度，不改变网络前向精度。
        self.channel_sum.add_(feature.sum(dim=(0, 2, 3)))
        self.channel_square_sum.add_(feature.square().sum(dim=(0, 2, 3)))
        self.pixel_count += feature.shape[0] * feature.shape[2] * feature.shape[3]

    def apply_to(self, layer):
        """将累计的 E[x]、E[x²] 换算为均值和方差，写入当前 BN 缓冲区。"""
        mean = self.channel_sum / self.pixel_count
        var = (self.channel_square_sum / self.pixel_count - mean.square()).clamp_min(0)
        layer.running_mean.copy_(mean)
        layer.running_var.copy_(var)


class BNCalibrator:
    """BN 校准器：加载模型与原划分，只用训练影像逐层恢复运行统计量。"""

    def __init__(self):
        """加载源模型、固定训练校准子集和完整验证集。"""
        self.run_dir = RUN_DIR
        if self.run_dir is None:
            self.run_dir = sorted(
                path for path in EXPERIMENT_DIR.iterdir()
                if path.is_dir() and (path / CHECKPOINT_NAME).is_file()
            )[-1]
        self.run_dir = Path(self.run_dir)
        self.checkpoint_path = self.run_dir / CHECKPOINT_NAME
        self.checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        # 校准只能恢复统计量不匹配，不能修复已经变成 NaN/Inf 的模型参数。
        if not all(torch.isfinite(value).all().item() for value in self.checkpoint["model_state_dict"].values()):
            raise ValueError("模型包含 NaN/Inf，不能通过 BN 校准恢复，请选择有限权重的 BN 模型。")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(self.checkpoint).to(self.device)  # 根据模型名称重建并严格加载。
        self.model.eval()

        # 完全使用检查点保存的划分；校准只需要训练影像，不读取它们的标签。
        train_data = WaterDataset(self.checkpoint["image_dir"], bands=self.checkpoint["bands"])
        val_data = WaterDataset(self.checkpoint["image_dir"], self.checkpoint["label_dir"], self.checkpoint["bands"])
        file_indices = {
            path.relative_to(train_data.image_dir).as_posix(): index
            for index, path in enumerate(train_data.image_paths)
        }
        self.calibration_files = self.checkpoint["train_files"][:CALIBRATION_SAMPLES]
        train_ids = [file_indices[name] for name in self.calibration_files]
        val_ids = [file_indices[name] for name in self.checkpoint["val_files"]]
        batch_size = self.checkpoint["batch_size"]
        self.train_loader = DataLoader(Subset(train_data, train_ids), batch_size=batch_size, shuffle=False)
        self.val_loader = DataLoader(Subset(val_data, val_ids), batch_size=batch_size, shuffle=False)
        self.mean = self.checkpoint["mean"].to(self.device).view(1, -1, 1, 1)
        self.std = self.checkpoint["std"].to(self.device).view(1, -1, 1, 1)
        self.ignore_index = self.checkpoint["ignore_index"]
        print(f"模型：{self.checkpoint_path}；设备：{self.device}；校准/验证：{len(train_ids)}/{len(val_ids)}", flush=True)

    def evaluate(self):
        """用标准 eval、float32 计算完整验证集指标。"""
        self.model.eval()
        confusion = torch.zeros(4, dtype=torch.int64, device=self.device)
        loss_sum, pixel_count = 0.0, 0
        with torch.no_grad():
            for batch in self.val_loader:
                image = (batch["image"].to(self.device) - self.mean) / self.std
                image = image.masked_fill(~batch["valid_mask"].to(self.device).unsqueeze(1), 0)
                label = batch["label"].to(self.device)
                valid = label != self.ignore_index
                count = int(valid.sum().item())
                if count == 0:
                    continue
                outputs = self.model(image)
                logits = outputs[-1] if isinstance(outputs, (tuple, list)) else outputs
                loss = nn.functional.cross_entropy(logits, label, ignore_index=self.ignore_index)
                if not torch.isfinite(logits).all().item() or not torch.isfinite(loss).item():
                    raise FloatingPointError(f"验证出现非有限输出或 loss；文件：{batch['name']}")
                confusion += torch.bincount(label[valid] * 2 + logits.argmax(1)[valid], minlength=4)
                loss_sum += loss.item() * count
                pixel_count += count
        tn, fp, fn, tp = confusion.tolist()
        return dict(
            loss=loss_sum / pixel_count,
            iou=tp / (tp + fp + fn) if tp + fp + fn else float("nan"),
            f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float("nan"),
            precision=tp / (tp + fp) if tp + fp else 0.0,
            recall=tp / (tp + fn) if tp + fn else float("nan"),
            tn=tn, fp=fp, fn=fn, tp=tp, valid_pixels=pixel_count,
            true_water_ratio=(tp + fn) / pixel_count,
            pred_water_ratio=(tp + fp) / pixel_count,
        )

    def calibrate_layers(self):
        """依次重估各层 BN 输入统计量，不把所有 BN 同时切到 train。"""
        # 当前模型的 BN 注册顺序满足上游先于下游；PVT 的 LayerNorm 不参与校准。
        # 上游先校准，所有层始终保持 eval，不更新任何可学习参数。
        for name, layer in self.model.named_modules():
            if not isinstance(layer, nn.BatchNorm2d):
                continue
            statistics = ChannelStatistics(layer.num_features, self.device)
            handle = layer.register_forward_pre_hook(statistics)
            try:
                with torch.no_grad():
                    for batch in self.train_loader:
                        image = (batch["image"].to(self.device) - self.mean) / self.std
                        image = image.masked_fill(~batch["valid_mask"].to(self.device).unsqueeze(1), 0)
                        self.model(image)
                    statistics.apply_to(layer)
            finally:
                handle.remove()
            print(f"已校准：{name}", flush=True)

    def save_result(self, before, after):
        """另存校准模型和前后指标，保留原始检查点。"""
        # 校准产物单独保存，不覆盖原始模型，不按验证成绩反复挑选校准样本。
        output_dir = self.run_dir / "bn_calibration" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_dir.mkdir(parents=True, exist_ok=False)
        with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=["phase", *before.keys()])
            writer.writeheader()
            writer.writerow(dict(phase="before", **before))
            writer.writerow(dict(phase="after", **after))
        self.checkpoint["model_state_dict"] = {name: value.cpu() for name, value in self.model.state_dict().items()}
        # 保留原归一化类型，PVT 同时包含 LayerNorm 和解码头的 BatchNorm。
        self.checkpoint.setdefault("normalization", "BatchNorm")
        self.checkpoint["source_val_loss"] = self.checkpoint["val_loss"]
        self.checkpoint["source_phase_metrics"] = self.checkpoint.get("phase_metrics", {})
        self.checkpoint["source_best_val_loss"] = self.checkpoint.pop("best_val_loss", None)
        self.checkpoint["val_loss"] = after["loss"]
        self.checkpoint["phase_metrics"] = {"val": after}
        self.checkpoint["bn_calibration"] = {
            "method": "sequential_train_feature_moments", "source_checkpoint": str(self.checkpoint_path),
            "train_files": self.calibration_files, "batch_size": self.checkpoint["batch_size"], "precision": "float32",
            "before": before, "after": after,
        }
        torch.save(self.checkpoint, output_dir / "calibrated.pth")
        print(f"校准模型和前后指标：{output_dir}", flush=True)

    def run(self):
        """校准前验证 → 逐层校准 → 校准后验证 → 保存。"""
        before = self.evaluate()
        print(f"校准前：{before}", flush=True)
        self.calibrate_layers()
        after = self.evaluate()
        print(f"校准后：{after}", flush=True)
        self.save_result(before, after)


if __name__ == "__main__":
    calibrator = BNCalibrator()
    calibrator.run()
