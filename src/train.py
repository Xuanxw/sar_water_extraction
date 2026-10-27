"""统一训练入口：修改顶部 METHOD 后运行 python src/train.py。"""
import csv
import gc
import hashlib
import shutil
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, random_split

import timm
from termcolor import cprint

# 终端配色：青=配置，蓝=准备进度，黄=训练，绿=验证，紫=保存，红=异常提示。
# 保留 flush=True；重定向到日志文件时由 termcolor 自动关闭颜色转义序列。

from datasets.water_dataset import IGNORE_INDEX, WaterDataset, sample_name
from models.losses import WaterSegmentationLoss
from evaluation.metrics import metrics_from_counts
from models.unet import UNet
from models.pvt import PVTWaterNet, PVTUANet
from models.resnet import ResNet50WaterNet


ROOT = Path(__file__).resolve().parents[1]
RUN_MODE = "train_postprocess"  # train_postprocess：训练后自动后处理；train：仅训练；postprocess：仅后处理。
POSTPROCESS_CHECKPOINTS = []  # 仅后处理时填写已有 .pth 路径；训练后自动使用本次最佳模型。
POSTPROCESS_OPTIONS = dict(
    use_saved_val=True, with_labels=True,  # 默认只处理检查点中保存的内部验证集。
    threshold=0.5, use_landcover=True, landcover_band=5, water_class=80,
    save_masks=True, save_visuals=True, max_visualizations=None,
)  # 仅后处理无标签数据时，可添加 image_dir，并将前两个开关设为 False。
METHOD = "uafnet"  # 同级方法：unet、pvt、resnet50、uanet、uafnet。
STAGE = "uafm"    # 仅 uanet 使用；uafnet 固定为 MBDC + FPN + 三级 UAFM，不含 PSM/CGM。
UAFNET_BACKBONE = "pvt_v2_b2.in1k"  # 仅 uafnet 使用：可改 b3、b4、b5；每个成员单独训练。
UAFNET_FINAL_RESIDUAL = True  # 沿用已验证的残差基线；False 恢复原最终头。
# 当前固定主基线：四级 MBDC 都沿用原分支空洞率 (4,8,16)/(2,4,8)。
# None 使用模型默认值；按尺度调整的历史检查点仍由各自 model_kwargs 正确重建。
UAFNET_MBDC_DILATIONS = None
UAFNET_HIGH_RESOLUTION_REFINEMENT = False  # 第一轮重跑新划分控制基线；第二轮设 True，只增加半分辨率细化。
# 两轮均从相同 ImageNet 初始化开始；不把旧划分训练过的检查点拿来做新验证集的公平对照。
IMAGE_DIR = ROOT / "data/Track1/train/images"
LABEL_DIR = ROOT / "data/Track1/train/labels"
OUTPUT_DIR = None  # None 按方法自动选择原实验目录；试跑时可指定其他路径。
POLARIZATIONS = ("VV", "VH")  # 单极化用 ("VV",)；双极化用 ("VV", "VH")，顺序就是网络输入顺序。
POLARIZATION_BANDS = {"VV": 1, "VH": 2}  # 当前 DFC 文件的通道对应；本地单波段 VV 文件改为 {"VV": 1}。
# 其他极化组合需按实际文件修改映射；这里只选择已有波段，不生成缺失极化或做辐射转换。
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
LOSS_NAME = "bce" if METHOD == "uafnet" else "ce"  # 固定主基线使用 BCE，不叠加其他损失。
BCE_WEIGHT = 1.0   # 仅 bce_dice 使用；保留 BCE 项原有系数。
DICE_WEIGHT = 1.0  # 仅 bce_dice 使用，本轮不启用 Dice。
# 仅 uafnet 使用，顺序：粗预测、第一/第二级细化、最终预测。全设 1 即恢复旧等权监督。
HEAD_LOSS_WEIGHTS = (1.0, 1.0, 1.0, 1.0)  # 恢复等权控制组，不叠加上一轮权重改动。
SAMPLING = "hard" if METHOD == "uafnet" else "random"  # 回到全局困难采样基线，保持其规则不变。
HARD_WARMUP_EPOCHS = 2  # 前两轮完整随机遍历，为所有训练样本建立难度记录。
HARD_SAMPLE_RATIO = 0.5  # 后续每轮约一半抽样位置来自困难池，总抽样数不增加。
HARD_POOL_RATIO = 0.25   # 困难池占有分数训练样本的25%；分层模式组内按BCE排序，不读取验证损失。
LOW_WATER_POOL_RATIO = 0.20  # 混合模式：困难池20%的名额用于补充少水体切片，其余保留全局高BCE样本。
WATER_GROUP_NAMES = ("无水体", "水体≤1%", "水体1%–10%", "水体10%–50%", "水体>50%")
SAVE_FEATURE_MAPS = False  # 默认不输出训练中间层特征图；最终质量图仍由 quality 流程输出。
FEATURE_MAP_INTERVAL = 5  # 第1轮、每5轮和最后一轮保存；不会额外执行模型前向。
FEATURE_MAP_SAMPLES = 2  # 最多保存首个验证批次的前2幅；不保存全通道大张量。


def build_hard_sample_plan(losses, hard_ratio=0.5, pool_ratio=0.25, seed=42, groups=None, low_water_pool_ratio=None):
    """返回抽样计划和困难池；groups 开启分层，再传补充比例则改用混合池。"""
    if low_water_pool_ratio is not None and (groups is None or not 0 <= low_water_pool_ratio <= 1):
        raise ValueError("混合池需要训练样本组号，少水体补充比例应为0～1")
    generator = torch.Generator().manual_seed(seed)  # 不消耗模型/增强使用的全局随机序列。
    count = len(losses)
    ordinary = torch.randperm(count, generator=generator)
    known = torch.where(torch.isfinite(losses))[0]  # 全忽略或未观测样本不按虚构分数加入困难池。
    if groups is not None:
        known = known[groups[known] >= 0]  # 没有有效标签的样本不分配虚构水体类别。
    hard_count = int(count * hard_ratio)
    if not len(known) or hard_count == 0:
        return ordinary.tolist(), []
    ranked = known[torch.argsort(losses[known], descending=True, stable=True)]
    pool_size = max(1, int(len(known) * pool_ratio))
    if groups is None:
        pool = ranked[:pool_size]  # 原全局困难池，保持旧调用和随机序列不变。
    elif low_water_pool_ratio is not None:
        # 先保留全局最高分的主要名额，再从尚未入池的少水体切片中补充。
        # 组1是 0<水体占比≤1%，无水体切片仍可通过全局排名或普通分支被抽到。
        extra_count = int(pool_size * low_water_pool_ratio)
        global_pool = ranked[:pool_size - extra_count]
        remaining = ranked[pool_size - extra_count:]
        extra_pool = remaining[groups[remaining] == 1][:extra_count]
        pool = torch.cat((global_pool, extra_pool))
        # 少水体候选不足时，按全局BCE补齐，保持池大小且不重复入池。
        missing = pool_size - len(pool)
        if missing:
            pool = torch.cat((pool, remaining[~torch.isin(remaining, extra_pool)][:missing]))
    else:
        group_ids, counts = torch.unique(groups[known], sorted=True, return_counts=True)
        # 最大余数法：先向下取整，剩余名额按小数部分从大到小补齐；并列按组号。
        shares = counts.double() * pool_size / len(known)
        quotas = shares.floor().long()
        remainder_order = torch.argsort(shares - quotas, descending=True, stable=True)
        quotas[remainder_order[:pool_size - int(quotas.sum())]] += 1
        pool = torch.cat([ranked[groups[ranked] == group][:int(quota)]
                          for group, quota in zip(group_ids, quotas)])
    repeated = pool[torch.randint(len(pool), (hard_count,), generator=generator)]
    plan = torch.cat((ordinary[:count - hard_count], repeated))
    plan = plan[torch.randperm(count, generator=generator)]
    return plan.tolist(), pool.tolist()


class WaterSegmentationTrainer:
    """有监督水体语义分割训练器，统一管理基线和 UAFNet 的训练、验证和保存。"""

    def __init__(
        self, model_class, model_kwargs, model_info, image_dir, label_dir, output_dir,
        bands=(1, 2), batch_size=2, epochs=20, learning_rate=1e-3,
        val_ratio=0.2, seed=42, max_samples=None, abnormal_f1_threshold=0.001,
        loss_name="ce", optimizer_name="adam", weight_decay=0.0,
        cosine_schedule=False, train_augmentation=False, loss_kwargs=None,
        sampling="random", hard_warmup_epochs=2, hard_sample_ratio=0.5, hard_pool_ratio=0.25,
        low_water_pool_ratio=0.20, head_loss_weights=None,
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
        self.loss_kwargs = dict(loss_kwargs or {})
        # None 兼容旧调用：每个输出头权重均为 1；权重只用于训练，不改变模型结构。
        self.head_loss_weights = tuple(head_loss_weights) if head_loss_weights is not None else None
        self.optimizer_name, self.weight_decay = optimizer_name, weight_decay
        self.train_augmentation = train_augmentation
        if sampling not in ("random", "hard", "stratified_hard", "hybrid_hard") or not 0 <= hard_sample_ratio <= 1 or not 0 < hard_pool_ratio <= 1:
            raise ValueError("采样选 random/hard/stratified_hard/hybrid_hard；困难抽样比例为0～1，困难池比例为0～1且大于零")
        if sampling != "random" and (loss_name != "bce" or hard_warmup_epochs < 1):
            raise ValueError("本轮困难采样仅搭配 BCE，且至少普通训练一轮")
        if sampling == "hybrid_hard" and not 0 <= low_water_pool_ratio <= 1:
            raise ValueError("少水体补充比例应为0～1")
        self.sampling = sampling
        self.sampling_config = dict(mode=sampling, warmup_epochs=hard_warmup_epochs,
                                    hard_sample_ratio=hard_sample_ratio, hard_pool_ratio=hard_pool_ratio)
        if sampling == "stratified_hard":
            self.sampling_config.update(water_ratio_boundaries=[0., .01, .1, .5],
                                        pool_quota_rule="largest_remainder_of_finite_training_group_counts")
        if sampling == "hybrid_hard":
            self.sampling_config.update(water_ratio_boundaries=[0., .01, .1, .5],
                                        low_water_pool_ratio=low_water_pool_ratio,
                                        low_water_group=1,
                                        pool_quota_rule="global_top_then_unselected_low_water; global_fill_if_short")
        torch.manual_seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = self.device.type == "cuda"

        self.dataset = WaterDataset(self.image_dir, self.label_dir, bands)
        split_path = self.image_dir.parent / "split_manifest.csv"
        self.split_info = {"split_strategy": "legacy_random", "split_manifest_sha256": ""}
        if split_path.is_file():
            # 物理文件全部保留；只有清单 active=1 的代表样本参加训练/验证。
            with split_path.open(encoding="utf-8-sig", newline="") as stream:
                split_rows = list(csv.DictReader(stream))
            indices = {sample_name(path, self.image_dir): i for i, path in enumerate(self.dataset.image_paths)}
            active = [row for row in split_rows if int(row["active"]) == 1]
            train_rows = [row for row in active if row["split"] == "train"]
            val_rows = [row for row in active if row["split"] == "val"]
            if {r["group_id"] for r in train_rows} & {r["group_id"] for r in val_rows}:
                raise ValueError("固定清单的同组样本不能跨 train/val")
            train_ids = [indices[row["name"]] for row in train_rows]
            val_ids = [indices[row["name"]] for row in val_rows]
            if max_samples is not None:
                # 短程试跑也保留两侧的固定归属，不从全部文件重新随机抽样。
                limit = max(1, int(max_samples * (1 - val_ratio)))
                train_ids, val_ids = train_ids[:limit], val_ids[:max(1, max_samples - limit)]
            if not train_ids or not val_ids:
                raise ValueError("固定清单需要非空的 train 和 val 有效样本")
            self.train_set = Subset(self.dataset, train_ids)
            self.val_set = Subset(self.dataset, val_ids)
            self.split_info = dict(split_strategy="scene_stratified_grouped",
                                   split_manifest=str(split_path),
                                   split_manifest_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest())
            # 新划分的分数不可直接与历史随机划分混排，使用独立实验目录。
            self.output_dir = self.output_dir.with_name(self.output_dir.name + "_grouped")
            cprint(f"固定分组清单：{split_path}；停用副本/标签冲突：{len(split_rows) - len(active)}", "cyan", flush=True)
        else:
            if (self.image_dir / "train").is_dir() or (self.image_dir / "val").is_dir():
                raise FileNotFoundError("已分区的数据需要 split_manifest.csv，不能静默重新随机划分")
            # 保留旧目录与合成测试兼容路径；新正式数据必须使用上面的固定清单。
            if max_samples is not None:
                self.dataset.image_paths = self.dataset.image_paths[:max_samples]
            val_size = max(1, int(len(self.dataset) * val_ratio))
            self.train_set, self.val_set = random_split(
                self.dataset, [len(self.dataset) - val_size, val_size],
                generator=torch.Generator().manual_seed(seed),
            )
        self.train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True, num_workers=0)
        self.val_loader = DataLoader(self.val_set, batch_size=batch_size, shuffle=False, num_workers=0)
        cprint(f"设备：{self.device}；训练/内部验证：{len(self.train_set)}/{len(self.val_set)}", color="cyan", flush=True)
        self.calculate_normalization()

        self.model = model_class(**self.model_kwargs).to(self.device)
        # 新分支增加了初始化随机数消耗；重置后两轮的抽样、增强和 dropout 从相同随机状态开始。
        if split_path.is_file():
            torch.manual_seed(seed)
        cprint(f"模型：{self.model_info['model_name']}；构建参数：{self.model_kwargs}", color="cyan", flush=True)
        cprint(f"损失：{loss_name}；优化器：{optimizer_name}；初始学习率：{learning_rate}；"
              f"余弦退火：{cosine_schedule}；训练增强：{train_augmentation}", color="cyan", flush=True)
        # UAFNet 单通道表示水体 logit；其余历史实验仍使用双类别 CE。
        self.criterion = WaterSegmentationLoss(loss_name, ignore_index=IGNORE_INDEX, **self.loss_kwargs)
        cprint(f"损失配置：{self.loss_kwargs}；训练头权重：{self.head_loss_weights or '全部为1'}；"
              "验证仅最终头，不乘训练头权重", color="cyan", flush=True)
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
        if split_path.is_file():
            shutil.copy2(split_path, self.run_dir / "split_manifest.csv")
        self.metrics_path = self.run_dir / "metrics.csv"
        self.metric_fields = [
            "epoch", "phase", "loss", "iou", "f1", "precision", "recall",
            "tn", "fp", "fn", "tp", "valid_pixels", "ignored_pixels", "skipped_batches",
            "true_water_ratio", "pred_water_ratio", "learning_rate", "amp_scale",
            "amp_skipped_steps", "nonfinite_loss_batches", "elapsed_seconds",
            "accuracy", "specificity", "false_positive_rate", "false_negative_rate",
            "false_discovery_rate", "background_iou", "mean_iou", "balanced_accuracy", "kappa", "mcc",
        ]
        with self.metrics_path.open("w", newline="", encoding="utf-8-sig") as stream:
            csv.DictWriter(stream, fieldnames=self.metric_fields).writeheader()
        cprint(f"本次实验目录：{self.run_dir}", color="cyan", attrs=["bold"], flush=True)
        self.best_loss = float("inf")
        self.best_f1 = float("-inf")
        self.train_files = [sample_name(self.dataset.image_paths[i], self.image_dir) for i in self.train_set.indices]
        self.val_files = [sample_name(self.dataset.image_paths[i], self.image_dir) for i in self.val_set.indices]
        if sampling != "random":
            self.train_position = {name: i for i, name in enumerate(self.train_files)}
            self.sample_losses = torch.full((len(self.train_set),), float("nan"))
            self.sample_last_seen = torch.zeros(len(self.train_set), dtype=torch.long)
            self.sampling_fields = ["epoch", "name", "selected_count", "in_hard_pool",
                                    "observations", "last_mean_bce", "last_seen_epoch"]
            if sampling in ("stratified_hard", "hybrid_hard"):
                self.sampling_fields += ["water_ratio", "water_group"]
                self.group_fields = ["epoch", "water_group", "group_name", "train_samples", "hard_pool_samples", "selected_count"]
                with (self.run_dir / "sampling_groups.csv").open("w", newline="", encoding="utf-8-sig") as stream:
                    csv.DictWriter(stream, fieldnames=self.group_fields).writeheader()
            with (self.run_dir / "sampling.csv").open("w", newline="", encoding="utf-8-sig") as stream:
                csv.DictWriter(stream, fieldnames=self.sampling_fields).writeheader()
        cprint(f"采样配置：{self.sampling_config}", color="cyan", flush=True)

    def prepare_train_sampling(self, epoch):
        """按上一轮结束时的难度记录生成本轮计划；前几轮仍使用原随机 DataLoader。"""
        if self.sampling == "random":
            return
        count = len(self.train_set)
        self.hard_pool = []
        self.sampling_plan = None
        if epoch > self.sampling_config["warmup_epochs"]:
            self.sampling_plan, self.hard_pool = build_hard_sample_plan(
                self.sample_losses, self.sampling_config["hard_sample_ratio"],
                self.sampling_config["hard_pool_ratio"], self.seed + epoch,
                groups=self.sample_groups if self.sampling in ("stratified_hard", "hybrid_hard") else None,
                low_water_pool_ratio=self.sampling_config.get("low_water_pool_ratio"))
            # 索引始终属于 train_set，不是原始 dataset，更不包含 val_set。
            self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size,
                                           sampler=self.sampling_plan, num_workers=0)
        self.sample_loss_sum = torch.zeros(count, dtype=torch.float64)
        self.sample_observations = torch.zeros(count, dtype=torch.long)
        self.sample_draws = (torch.ones(count, dtype=torch.long) if self.sampling_plan is None else
                             torch.bincount(torch.tensor(self.sampling_plan), minlength=count))
        cprint(f"Epoch {epoch} 采样：{'普通预热' if self.sampling_plan is None else '随机 + 困难'}；"
              f"总抽样 {int(self.sample_draws.sum())}；不同样本 {int((self.sample_draws > 0).sum())}；"
              f"困难池 {len(self.hard_pool)}", color="blue", flush=True)

    def calculate_normalization(self):
        """只用训练集统计均值/标准差；分层和混合模式同时记录标签水体比例。"""
        channel_sum = np.zeros(len(self.bands), dtype=np.float64)
        channel_square_sum = np.zeros_like(channel_sum)
        pixel_count = 0
        if self.sampling in ("stratified_hard", "hybrid_hard"):
            self.sample_water_ratios = torch.full((len(self.train_set),), float("nan"))
            self.sample_groups = torch.full((len(self.train_set),), -1, dtype=torch.long)
        for number, sample in enumerate(self.train_set, 1):
            # [C,H,W] 按空间掩膜取值后为 [C,N]；不按标签类别筛选。
            values = sample["image"].numpy()[:, sample["valid_mask"].numpy()].astype(np.float64)
            channel_sum += values.sum(axis=1)
            channel_square_sum += np.square(values).sum(axis=1)
            pixel_count += values.shape[1]
            if self.sampling in ("stratified_hard", "hybrid_hard"):
                # 借用现有读取循环，不额外读图，也不读取验证标签；全忽略样本保留组号 -1。
                label = sample["label"]
                valid_label = label != IGNORE_INDEX
                if valid_label.any():
                    ratio = (label == 1).sum().double() / valid_label.sum()
                    self.sample_water_ratios[number - 1] = ratio
                    self.sample_groups[number - 1] = torch.bucketize(
                        ratio, torch.tensor([0., .01, .1, .5], dtype=torch.float64))
            if number % 100 == 0:
                cprint(f"归一化统计：{number}/{len(self.train_set)}", color="blue", flush=True)
        if pixel_count == 0:
            raise ValueError("训练部分没有有效影像像素")
        mean = channel_sum / pixel_count
        std = np.sqrt(np.maximum(channel_square_sum / pixel_count - mean ** 2, 0))
        # [1,C,1,1] 可广播到任意批次；下限避免除零。
        self.mean = torch.tensor(mean, dtype=torch.float32, device=self.device).view(1, -1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32, device=self.device).clamp_min(1e-6).view(1, -1, 1, 1)
        cprint(f"输入通道：{self.bands}；均值：{mean}；标准差：{std}", color="cyan", flush=True)

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
                    visualizer = None
                    if (not training and step == 1 and SAVE_FEATURE_MAPS
                            and isinstance(self.model, PVTUANet)
                            and (epoch == 1 or epoch == self.epochs or epoch % FEATURE_MAP_INTERVAL == 0)):
                        from evaluation.visual import FeatureVisualizer
                        visualizer = FeatureVisualizer(self.model, FEATURE_MAP_SAMPLES)
                    # 钩子只读取本次验证特征，离开 with 后自动移除，不影响训练或其他批次。
                    with visualizer if visualizer is not None else nullcontext():
                        outputs = self.model(image)
                    predictions = outputs if isinstance(outputs, (tuple, list)) else (outputs,)
                    logits = predictions[-1]
                    if training:
                        # 权重对应模型返回顺序；保持加权求和，不改变原 BCE 的有效像素均值。
                        weights = (self.head_loss_weights if self.head_loss_weights is not None
                                   else (1.0,) * len(predictions))
                        if len(weights) != len(predictions):
                            raise ValueError("训练头权重数量必须与模型输出数量一致")
                        loss = sum(weight * self.criterion(output, label)
                                   for weight, output in zip(weights, predictions))
                    else:
                        # 验证仍是最终预测的原始损失，可与等权基线按相同规则选模。
                        loss = self.criterion(logits, label)
                if training and self.sampling != "random":
                    # 难度仍取最终头更新前的原始 BCE，不乘该头的训练权重。
                    # 同一图本轮重复抽到时取均值；这是增强/训练模式下的在线难度估计。
                    with torch.no_grad():
                        pixel_loss = F.binary_cross_entropy_with_logits(
                            logits[:, 0].detach().float(), label.clamp_min(0).float(), reduction="none")
                        counts = valid.sum(dim=(1, 2))
                        scores = pixel_loss.masked_fill(~valid, 0).sum(dim=(1, 2)) / counts.clamp_min(1)
                        for name, score, pixels in zip(batch["name"], scores.cpu().tolist(), counts.cpu().tolist()):
                            if pixels and np.isfinite(score):
                                position = self.train_position[name]
                                self.sample_loss_sum[position] += score
                                self.sample_observations[position] += 1
                if not torch.isfinite(loss).item():
                    raise FloatingPointError(f"Epoch {epoch} {phase} 第 {step} 批 loss 非有限；文件：{batch['name']}")
                if training:
                    previous_scale = self.scaler.get_scale()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    if self.scaler.get_scale() < previous_scale:
                        amp_skipped_steps += 1

            if visualizer is not None:
                folder = self.run_dir / "features" / f"epoch_{epoch:03d}"
                try:
                    paths = visualizer.save(batch, predictions, folder, epoch, self.bands)
                    cprint(f"中间特征图：{folder}（{len(paths)}幅）", color="magenta", flush=True)
                except (OSError, ValueError) as error:
                    cprint(f"特征图保存失败，训练指标继续正常统计：{error}", color="red", attrs=["bold"], flush=True)
                visualizer = None  # 释放本批二维响应，避免跨批累积内存。

            # 按有效像素累计损失与混淆矩阵，不直接平均每批 F1。
            loss_value = loss.item()
            if not np.isfinite(loss_value):
                nonfinite_loss_batches += 1
            loss_sum += loss_value * count
            valid_pixels += count
            # 单通道 sigmoid>=0.5 等价于 logit>=0；双类别仍使用 argmax。
            prediction = ((logits.detach()[:, 0] >= 0).long() if self.criterion.binary
                          else logits.detach().argmax(dim=1))
            # 真实值*2+预测值：0=TN、1=FP、2=FN、3=TP。
            confusion += torch.bincount(label[valid] * 2 + prediction[valid], minlength=4)
            if step % 50 == 0:
                cprint(f"Epoch {epoch} {phase}：{step}/{len(loader)} 批", color="yellow" if training else "green", flush=True)

        if valid_pixels == 0:
            raise ValueError(f"{phase} 没有可用于损失计算的标签像素")
        if training and self.sampling != "random":
            seen = self.sample_observations > 0
            self.sample_losses[seen] = (self.sample_loss_sum[seen] / self.sample_observations[seen]).float()
            self.sample_last_seen[seen] = epoch  # 本轮未抽到的样本保留最后一次观测值及时间。
            pool = set(self.hard_pool)
            with (self.run_dir / "sampling.csv").open("a", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=self.sampling_fields)
                for i, name in enumerate(self.train_files):
                    row = dict(epoch=epoch, name=name, selected_count=int(self.sample_draws[i]),
                               in_hard_pool=int(i in pool), observations=int(self.sample_observations[i]),
                               last_mean_bce=float(self.sample_losses[i]) if torch.isfinite(self.sample_losses[i]) else "",
                               last_seen_epoch=int(self.sample_last_seen[i]))
                    if self.sampling in ("stratified_hard", "hybrid_hard"):
                        row.update(water_ratio=float(self.sample_water_ratios[i]), water_group=int(self.sample_groups[i]))
                    writer.writerow(row)
            if self.sampling in ("stratified_hard", "hybrid_hard"):
                pool_indices = torch.tensor(self.hard_pool, dtype=torch.long)
                with (self.run_dir / "sampling_groups.csv").open("a", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=self.group_fields)
                    for group in range(-1, len(WATER_GROUP_NAMES)):
                        members = self.sample_groups == group
                        if not members.any():
                            continue
                        writer.writerow(dict(epoch=epoch, water_group=group,
                            group_name=WATER_GROUP_NAMES[group] if group >= 0 else "无有效标签",
                            train_samples=int(members.sum()),
                            hard_pool_samples=int((self.sample_groups[pool_indices] == group).sum()),
                            selected_count=int(self.sample_draws[members].sum())))
        tn, fp, fn, tp = confusion.tolist()
        pixel_metrics = metrics_from_counts((tn, fp, fn, tp))
        iou, f1 = pixel_metrics["iou"], pixel_metrics["f1"]
        precision, recall = pixel_metrics["precision"], pixel_metrics["recall"]
        metrics = dict(
            epoch=epoch, phase=phase, loss=loss_sum / valid_pixels, iou=iou, f1=f1,
            precision=precision, recall=recall, tn=tn, fp=fp, fn=fn, tp=tp,
            valid_pixels=valid_pixels, ignored_pixels=ignored_pixels, skipped_batches=skipped_batches,
            true_water_ratio=(tp + fn) / valid_pixels, pred_water_ratio=(tp + fp) / valid_pixels,
            learning_rate=self.optimizer.param_groups[0]["lr"], amp_scale=self.scaler.get_scale(),
            amp_skipped_steps=amp_skipped_steps, nonfinite_loss_batches=nonfinite_loss_batches,
            elapsed_seconds=perf_counter() - phase_start,
        )
        metrics.update(pixel_metrics)  # 原有四指标不变，补充完整像元质量指标。
        with self.metrics_path.open("a", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=self.metric_fields).writerow(metrics)
        cprint(f"Epoch {epoch:02d}/{self.epochs} {phase}：loss={metrics['loss']:.4f}，"
              f"水体 IoU={iou:.4f}，F1={f1:.4f}，P={precision:.4f}，R={recall:.4f}", color="yellow" if training else "green", attrs=["bold"], flush=True)
        cprint(f"真实/预测水体比例：{metrics['true_water_ratio']:.4%}/{metrics['pred_water_ratio']:.4%}；"
              f"TN/FP/FN/TP：{tn}/{fp}/{fn}/{tp}；AMP 跳过更新：{amp_skipped_steps} 次", color="red" if amp_skipped_steps else ("yellow" if training else "green"), flush=True)
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
            "model_state_dict": self.model.state_dict(), **self.model_info, **getattr(self, "split_info", {}),
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
            "loss_name": self.loss_name, "loss_kwargs": self.loss_kwargs,
            "head_loss_weights": self.head_loss_weights,  # None 表示旧版默认等权。
            "sampling_config": self.sampling_config,
            "loss_reduction": ("CE valid-pixel mean" if not self.criterion.binary else
                               "BCE valid-pixel mean; optional Dice per-image mean, smooth=1"),
        }
        if self.sampling != "random":
            checkpoint.update(sampling_losses=self.sample_losses, sampling_last_seen=self.sample_last_seen,
                              sampling_plan=self.sampling_plan, hard_pool=self.hard_pool)
            if self.sampling in ("stratified_hard", "hybrid_hard"):
                checkpoint.update(sampling_water_ratios=self.sample_water_ratios, sampling_groups=self.sample_groups)
        if self.criterion.binary:
            # best 仍按 loss 保存；另存 F1 最佳，避免与日志最高 F1 混淆。
            checkpoint.update(loss_name=self.loss_name, optimizer_name=self.optimizer_name,
                              weight_decay=self.weight_decay, train_augmentation=self.train_augmentation,
                              optimizer_state_dict=self.optimizer.state_dict(),
                              scaler_state_dict=self.scaler.state_dict(),
                              scheduler_state_dict=self.scheduler.state_dict() if self.scheduler else None)
        filenames = ["last.pth"]
        if self.criterion.binary and metrics["f1"] > self.best_f1:
            self.best_f1 = metrics["f1"]
            filenames.append("best_f1.pth")
        if self.criterion.binary:
            checkpoint["best_val_f1"] = self.best_f1
        if improved:
            filenames.append("best.pth")
        if abnormal:
            filenames.append("abnormal.pth")
        for filename in filenames:
            torch.save(checkpoint, self.run_dir / filename)
            cprint(f"已保存：{self.run_dir / filename}", color="magenta", flush=True)

    def train(self):
        """逐轮执行训练 → 验证 → 保存。"""
        for epoch in range(1, self.epochs + 1):
            self.phase_metrics = {}
            self.recent_train_files = []
            self.prepare_train_sampling(epoch)
            for phase, loader in (("train", self.train_loader), ("val", self.val_loader)):
                self.phase_metrics[phase] = self.run_epoch(epoch, phase, loader)
            # CSV 记录本轮实际使用的 lr；调度后保存，为下一轮保留调度器状态。
            if self.scheduler is not None:
                self.scheduler.step()
            self.save_checkpoint(epoch)


def create_trainer():
    """平级选择网络与保存目录，然后创建同一个公共训练器。"""
    # 在创建训练器时解析，单极化自动构建单通道网络，并重新统计该通道的均值和标准差。
    bands = tuple(POLARIZATION_BANDS[name] for name in POLARIZATIONS)
    if METHOD == "unet":
        model_class = UNet
        model_kwargs = dict(in_channels=len(bands), base_channels=BASE_CHANNELS)
        model_info = dict(model_name="unet", normalization="BatchNorm", base_channels=BASE_CHANNELS)
        experiment = "unet_baseline"

    elif METHOD == "pvt":
        model_class = PVTWaterNet
        model_kwargs = dict(in_channels=len(bands), num_classes=2,
                            decoder_channels=DECODER_CHANNELS, pretrained=PRETRAINED)
        model_info = dict(model_name="pvt", backbone="pvt_v2_b2.in1k",
                          normalization="LayerNorm + decoder BatchNorm", stage="backbone_projection_decoder")
        experiment = "pvt_baseline"

    elif METHOD == "resnet50":
        model_class = ResNet50WaterNet
        model_kwargs = dict(in_channels=len(bands), num_classes=2,
                            decoder_channels=DECODER_CHANNELS, pretrained=PRETRAINED)
        model_info = dict(model_name="resnet50", backbone="resnet50.tv_in1k",
                          normalization="BatchNorm", stage="backbone_projection_decoder")
        experiment = "resnet50_baseline"

    elif METHOD == "uanet":
        model_class = PVTUANet
        model_kwargs = dict(in_channels=len(bands), num_classes=2,
                            decoder_channels=DECODER_CHANNELS, pretrained=PRETRAINED, stage=STAGE)
        # 保留旧检查点名称 pvt_uanet，兼容既有加载器和实验记录。
        model_info = dict(model_name="pvt_uanet", backbone="pvt_v2_b2.in1k", stage=STAGE,
                          normalization="LayerNorm + BatchNorm", output_order="coarse_to_fine",
                          supervision="sum_CE_all_train_heads; final_head_validation")
        experiment = f"pvt_uanet_{STAGE}"

    elif METHOD == "uafnet":
        # 水体论文的单模型结构；损失由 LOSS_NAME 选择，BCE+Dice 属于本项目对照实验。
        model_class = PVTUANet
        model_kwargs = dict(in_channels=len(bands), num_classes=1,
                            decoder_channels=DECODER_CHANNELS, pretrained=PRETRAINED, stage="uafm",
                            backbone_name=UAFNET_BACKBONE, final_residual=UAFNET_FINAL_RESIDUAL)
        if UAFNET_HIGH_RESOLUTION_REFINEMENT:
            model_kwargs["high_resolution_refinement"] = True
        if UAFNET_MBDC_DILATIONS is not None:
            # dilation 不属于 state_dict，必须保存到 model_kwargs 才能正确重建新结构。
            model_kwargs["mbdc_dilations"] = UAFNET_MBDC_DILATIONS
        model_info = dict(model_name="pvt_uafnet", backbone=UAFNET_BACKBONE, stage="uafm",
                          normalization="LayerNorm + BatchNorm", output_order="coarse_to_fine",
                          supervision=f"sum_{LOSS_NAME}_all_train_heads; final_head_validation",
                          implementation=f"UAFNet paper-guided SAR adaptation; single {UAFNET_BACKBONE}",
                          final_prediction="coarse_plus_zero_init_correction" if UAFNET_FINAL_RESIDUAL else "direct_refinement",
                          paper_doi="10.1109/IGARSS53475.2024.10641026")
        variant = UAFNET_BACKBONE.split(".")[0].removeprefix("pvt_v2_")
        experiment = "pvt_uafnet" if variant == "b2" else f"pvt_uafnet_{variant}"

    else:
        raise ValueError("METHOD 应为 unet、pvt、resnet50、uanet 或 uafnet")

    if METHOD != "unet":
        model_info.update(decoder="SemanticFPNDecoder", decoder_channels=DECODER_CHANNELS,
                          timm_version=timm.__version__, pretrained=PRETRAINED)

    baseline_loss = "bce" if METHOD == "uafnet" else "ce"
    if (METHOD == "uafnet" and LOSS_NAME not in ("bce", "bce_dice")) or (METHOD != "uafnet" and LOSS_NAME != "ce"):
        raise ValueError("当前 uafnet 可选 bce/bce_dice；其余双类别模型使用 ce")
    # BCE 保留历史目录；新损失进入独立目录，时间戳继续隔离每次运行。
    if LOSS_NAME != baseline_loss:
        experiment += f"_{LOSS_NAME}"
    if SAMPLING in ("hard", "stratified_hard", "hybrid_hard"):
        if METHOD != "uafnet" or LOSS_NAME != "bce":
            raise ValueError("本轮困难采样只比较 UAFNet + BCE；其他方法或损失请设 SAMPLING='random'")
        experiment += "_hard_sampling"
    if METHOD == "uafnet" and UAFNET_FINAL_RESIDUAL:
        experiment += "_residual"
    if SAMPLING == "stratified_hard":
        experiment += "_stratified"
    elif SAMPLING == "hybrid_hard":
        experiment += "_hybrid"
    if METHOD == "uafnet" and UAFNET_HIGH_RESOLUTION_REFINEMENT:
        experiment += "_highres"
    if METHOD == "uafnet" and UAFNET_MBDC_DILATIONS is not None:
        experiment += "_scale_mbdc"
    # 非水体论文分支保持旧等权规则；修改 UAFNet 监督权重时单独保存实验。
    head_loss_weights = tuple(HEAD_LOSS_WEIGHTS) if METHOD == "uafnet" else None
    if head_loss_weights is not None and head_loss_weights != (1.0, 1.0, 1.0, 1.0):
        experiment += "_head_weights"
        model_info["supervision"] = f"weighted_sum_{LOSS_NAME}_all_train_heads; final_head_validation"
    # 保留原 VV/VH 目录；单极化或其他组合独立保存，防止比较时混淆。
    if POLARIZATIONS != ("VV", "VH"):
        experiment += "_" + "_".join(name.lower() for name in POLARIZATIONS)
    model_info.update(polarizations=POLARIZATIONS, polarization_bands=dict(zip(POLARIZATIONS, bands)))
    output_dir = ROOT / "outputs" / experiment if OUTPUT_DIR is None else Path(OUTPUT_DIR)
    cprint(f"训练方法：{METHOD}；极化：{POLARIZATIONS}；文件波段：{bands}；实验目录：{output_dir}", color="cyan", flush=True)
    return WaterSegmentationTrainer(
        model_class=model_class, model_kwargs=model_kwargs, model_info=model_info,
        image_dir=IMAGE_DIR, label_dir=LABEL_DIR, output_dir=output_dir,
        bands=bands, batch_size=BATCH_SIZE, epochs=EPOCHS, learning_rate=LEARNING_RATE,
        val_ratio=VAL_RATIO, seed=SEED, max_samples=MAX_SAMPLES,
        abnormal_f1_threshold=ABNORMAL_F1_THRESHOLD,
        loss_name=LOSS_NAME,
        loss_kwargs=dict(bce_weight=BCE_WEIGHT, dice_weight=DICE_WEIGHT) if LOSS_NAME == "bce_dice" else {},
        optimizer_name="adamw" if METHOD == "uafnet" else "adam",
        weight_decay=WEIGHT_DECAY if METHOD == "uafnet" else 0.0,
        cosine_schedule=METHOD == "uafnet",
        train_augmentation=METHOD == "uafnet" and TRAIN_AUGMENTATION,
        sampling=SAMPLING, hard_warmup_epochs=HARD_WARMUP_EPOCHS,
        hard_sample_ratio=HARD_SAMPLE_RATIO, hard_pool_ratio=HARD_POOL_RATIO,
        low_water_pool_ratio=LOW_WATER_POOL_RATIO,
        head_loss_weights=head_loss_weights,
    )


def main():
    """统一执行训练与后处理；后处理重载本次最佳权重，不沿用内存中的最后一轮。"""
    if RUN_MODE not in ("train_postprocess", "train", "postprocess"):
        raise ValueError("RUN_MODE 应为 train_postprocess、train 或 postprocess")
    checkpoints = list(POSTPROCESS_CHECKPOINTS)
    options = dict(POSTPROCESS_OPTIONS)
    if RUN_MODE != "postprocess":
        trainer = create_trainer()
        trainer.train()
        if RUN_MODE == "train":
            return trainer.run_dir
        # 单通道水体按验证F1选模；其余方法仍按验证loss选择 best.pth。
        filename = "best_f1.pth" if trainer.criterion.binary else "best.pth"
        checkpoints = [trainer.run_dir / filename]
        options.update(image_dir=trainer.image_dir, label_dir=trainer.label_dir,
                       use_saved_val=True, with_labels=True, output_dir=None,
                       batch_size=trainer.batch_size, use_amp=trainer.use_amp)
        # 先释放训练模型、梯度和优化器状态，避免重载最佳模型时占用两份显存。
        del trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if not checkpoints:
        raise ValueError("仅后处理时，请在 POSTPROCESS_CHECKPOINTS 填写检查点路径")
    from evaluation.postprocess import run_postprocess
    cprint(f"开始后处理，使用检查点：{checkpoints}", color="cyan", attrs=["bold"], flush=True)
    return run_postprocess(checkpoints, **options)


if __name__ == "__main__":
    main()
