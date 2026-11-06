"""统一训练入口：修改顶部 METHOD 后运行 python src/supervised/train.py。"""
import csv
import gc
import hashlib
import shutil
import random
import traceback
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, random_split

import timm
from termcolor import cprint

# 终端配色：青=配置，蓝=准备进度，黄=训练，绿=验证，紫=保存，红=异常提示。
# 保留 flush=True；重定向到日志文件时由 termcolor 自动关闭颜色转义序列。

# 支持按文件路径直接运行，也支持包导入。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from supervised.load.sar_scene_dataset import IGNORE_INDEX, SARSceneDataset
from datasets.speckle import SpeckleFilter
from supervised.models.losses import WaterSegmentationLoss
from supervised.evaluation.metrics import metrics_from_counts
from supervised.models.unet import UNet
from supervised.models.pvt import PVTWaterNet, PVTUANet, PVTDeepLabV3Plus
from supervised.models.resnet import ResNet50WaterNet


# ---------- 执行入口 ----------
ROOT = Path(r"E:\Code_projects\sar_water_extraction")
METHOD = "deeplabv3"  # deeplabv3、uafnet、uanet、unet、pvt、resnet50。
RUN_MODE = "train"  # 先手动运行src/datasets/image_process.py；本入口只读取预处理结果并训练。
# None 从头训练；填写 last.pth 从已完成轮次继续，train_end.pth 可重跑本轮验证。
RESUME_CHECKPOINT = None  # 新权重方案从同一ImageNet初始化开始，不续跑旧损失实验。

# ---------- 数据与输出：修改成实际绝对路径 ----------
IMAGE_DIR = Path(r"I:\水体数据\processed_scenes\normalized\images")
LABEL_DIR = Path(r"I:\水体数据\processed_scenes\normalized\labels")
OUTPUT_ROOT = Path(r"E:\Code_projects\sar_water_extraction\outputs")
OUTPUT_DIR = Path(r"E:\Code_projects\sar_water_extraction\outputs\pvt_deeplabv3plus_vv_sqrt_weight_bce_offline_normalized_bc_scenes")
# OUTPUT_DIR是最终实验根目录，只在其下增加时间戳；不再隐藏追加滤波/数据集后缀。
WINDOW_CACHE_DIR = Path(r"E:\Code_projects\sar_water_extraction\data_cache\bc_scenes")
SNAP_GPT_PATH = Path(r"D:\SNAP\esa-snap\bin\gpt.exe")
SNAP_CACHE_DIR = Path(r"I:\水体数据\snap_lee_cache")
QUALITY_CHECKPOINT = Path(r"E:\Code_projects\sar_water_extraction\outputs\pvt_deeplabv3plus_vv_balanced_bce_labels_corrected_bc_scenes\20261103_203233_244527\best_f1.pth")
QUALITY_OUTPUT_DIR = Path(r"E:\Code_projects\sar_water_extraction\outputs\pvt_deeplabv3plus_vv_balanced_bce_labels_corrected_bc_scenes\20261103_203233_244527\quality\baseline_scan")
AUXILIARY_CACHE_DIR = Path(r"E:\Code_projects\sar_water_extraction\data_cache\auxiliary")
RUN_DIRECTORY_FORMAT = "%Y%m%d_%H%M%S_%f"
# 以下名称相对于本次时间戳目录；续训在原目录内继续，不另开目录。
RESULT_FILES = dict(
    metrics="metrics.csv", sampling="sampling.csv", sampling_groups="sampling_groups.csv",
    split="split_manifest.csv", labels="label_revision.json", error="training_error.log",
    last="last.pth", train_end="train_end.pth", best="best.pth", best_f1="best_f1.pth",
    best_balanced="best_balanced_accuracy.pth", abnormal="abnormal.pth",
    features="features", val_scenes="val_scenes_epoch_{epoch:03d}.csv",
)

# 有 split_manifest.csv 时使用固定 active train/val，VAL_RATIO 不重新拆分数据。
POLARIZATIONS = ("VV",)  # 单极化改为 ("VV",)，不会自动生成缺失极化。
POLARIZATION_BANDS = {"VV": 1}
VAL_RATIO = 0.2  # 仅无固定清单的旧数据/合成测试使用。
MAX_SAMPLES = None  # None 使用全部有效样本；小整数用于试跑。

# ---------- 正式训练不执行图像预处理；滤波参数在image_process.py设置 ----------
SPECKLE_FILTER = "none"  # 正式训练读取离线归一化结果，不再滤波。
SPECKLE_INPUT_DOMAIN = "amplitude"  # ORG已确认是线性幅度；仍未做辐射校正。
SPECKLE_WINDOW_SIZE = 3  # 普通Lee的奇数窗口；先用3×3，避免同时改变多个实验因素。
SPECKLE_ENL = 1.0  # 功率域等效视数参数，不代表已测得本批ORG的真实ENL。
# 开启/修改滤波时设 RESUME_CHECKPOINT=None；旧模型不能作为同一次实验直接续跑。
# 不改写源影像；平方→GPT功率域Lee→开方，最后仍以幅度进入标准化。
# GPT与缓存路径集中在顶部；image_process.py和本入口需使用同一配置。

# ---------- 整景读取：只在内存中取窗口，不生成切片文件 ----------
DATA_LAYOUT = "full_scene"  # 旧DFC实验可明确设为patch，才使用旧数据集。
SCENE_OPTIONS = dict(
    cache_dir=WINDOW_CACHE_DIR,
    window_size=1024, halo=128,  # 中心768×768计分，四周128像素仅提供上下文。
    normalization_windows=2,    # 每个训练景固定抽2个窗口估计输入统计量；不使用val。
    max_open_scenes=4,
    windows_per_epoch=4096,     # 每轮抽取的训练窗口上限；验证仍覆盖全部已标注窗口。
    num_workers=2,              # 每worker独立句柄；有界预取和GDAL缓存。
)

# ---------- 通用训练参数 ----------
PRETRAINED = True  # 从本地缓存的ImageNet权重初始化编码器；解码器重新初始化。
# image_process重建后：SHP面内水体1、面外非水体0。仅加载编码器预训练，不续训旧标签下的水体模型。
CPU_THREADS = 4  # 限制主进程CPU线程，避免模型初始化/小算子过度并行。
BATCH_SIZE = 2
EPOCHS = 20
LEARNING_RATE = 1e-4
SEED = 42
LOSS_NAME = "bce"  # deeplabv3/uafnet：bce 或 bce_dice；其余双类别方法用 ce。
BCE_WEIGHT = 1.0   # 以下两项仅 bce_dice 使用。
DICE_WEIGHT = 1.0  # LOSS_NAME="bce"时不参与损失。
CLASS_WEIGHT_MODE = "train_pixels_sqrt"  # none、train_pixels（原基线）、train_pixels_sqrt（本次对照）。
# 逆频率或其平方根，再使训练分布下平均权重为1；未知像素和val均不参与。
# 仅作用于训练损失，验证loss仍使用未加权BCE。
# 单通道水体方法的优化设置；旧双类别方法仍用 Adam、不启用余弦和增强。
OPTIMIZER_NAME = "adamw"
WEIGHT_DECAY = 0.01
COSINE_SCHEDULE = True
TRAIN_AUGMENTATION = True  # 随机90度旋转和水平翻转。
ABNORMAL_F1_THRESHOLD = 0.001  # 仅用于日志提示，不参与损失或选模。

# ---------- 训练样本采样 ----------
SAMPLING = "random"  # 本轮用固定种子的随机窗口抽样，与类别权重配合；先关闭困难重采样。
HARD_WARMUP_EPOCHS = 2
HARD_SAMPLE_RATIO = 0.5
HARD_POOL_RATIO = 0.25
LOW_WATER_POOL_RATIO = 0.20  # 仅 hybrid_hard：困难池中少水体样本的补充比例。
WATER_GROUP_NAMES = ("无水体", "水体≤1%", "水体1%–10%", "水体10%–50%", "水体>50%")

# ---------- 当前 DeepLabv3+ 风格模型 ----------
DEEPLAB_BACKBONE = "pvt_v2_b2.in1k"
DEEPLAB_DECODER_CHANNELS = 256
DEEPLAB_LOW_LEVEL_CHANNELS = 48
DEEPLAB_ASPP_RATES = (3, 6, 9)  # PVT 最深特征步长32；不是原论文步长16的配置。
DEEPLAB_DROPOUT = 0.1
# DeepLab 单头权重固定为1；下面的四头权重仅用于 UAFNet。

# ---------- 其他模型选项：只有选中对应 METHOD 时生效 ----------
BASE_CHANNELS = 32       # U-Net。
DECODER_CHANNELS = 32    # PVT简单头、ResNet、UANet/UAFNet。
STAGE = "uafm"           # 仅 UANet 分阶段使用。
UAFNET_BACKBONE = "pvt_v2_b2.in1k"
UAFNET_FINAL_RESIDUAL = True
UAFNET_MBDC_DILATIONS = None  # None：固定原空洞率。
UAFNET_HIGH_RESOLUTION_REFINEMENT = False
UAFNET_REFINEMENT_DECODER = "uafm"  # uafm 或 unetpp。
HEAD_LOSS_WEIGHTS = (1.0, 1.0, 1.0, 1.0)  # UAFNet：粗预测、两级辅助、最终头。

# ---------- 质量输出 ----------
POSTPROCESS_CHECKPOINTS = [QUALITY_CHECKPOINT]
# 整景仅单模型，默认扫描全部val；复用相同output_dir可继续未完成景。
SCENE_QUALITY_OPTIONS = dict(
    output_dir=QUALITY_OUTPUT_DIR,
    thresholds=[i/100 for i in range(1,100)], threshold=0.5,
    boundary_radii=(1,3,5), save_probabilities=True, save_visuals=True,
    opening_radius=0, closing_radius=0, num_workers=0, scene_limit=None,
)
POSTPROCESS_OPTIONS = dict(
    # 数据准备：只对所选验证/外部影像下载；与阈值分割共用缓存，本地路径优先。
    image_dir=IMAGE_DIR, label_dir=LABEL_DIR,
    use_saved_val=True, with_labels=True,
    auxiliary_cache_dir=AUXILIARY_CACHE_DIR,
    auxiliary_options=dict(download_history=False, download_elevation=False,
                           worldcover_year=2021, buffer_m=1000.,
                           max_tiles_per_scene=64, max_mosaic_pixels=100000000,
                           timeout=60, retries=3),
    # 模型推理参数：复现检查点的滤波/归一化，生成后处理输入raw。
    threshold=0.5,
    # 后处理1：内置/外部历史水体显式补入；与下载开关独立，默认只对比。
    use_landcover=False, landcover_band=5, water_class=80,
    use_external_water=False,
    # 后处理2/3：面积与历史变化统计，以及DEM/DSM几何水深估计。
    # 监督入口尚未接入阈值GIS的坡度/坡向、地形、OSM建筑道路或农田排除。
    analysis_options=dict(
        reference_path=None, reference_water_values=(1,), reference_dry_values=(0,),
        reference_coverage_path=None, vector_assume_complete=False,
        elevation_path=None, elevation_kind="DEM", elevation_scale=1.,
        depth_scope="new_water", depth_neighbors=100, depth_min_boundary=3,
        depth_max_distance_m=5000., depth_max_pixels=25000000,
    ),  # 下载高程是DSM（米）；new_water水深需要历史参考，永久水体没有水底高程。
    # 后处理4：评价与输出；分别保留raw/post分析成果及相同有效域内的质量指标。
    save_masks=True, save_visuals=True, max_visualizations=None,
)  # 无标签数据需填写影像绝对路径，label_dir=None，前两个开关设为False。
SAVE_FEATURE_MAPS = False
FEATURE_MAP_INTERVAL = 5
FEATURE_MAP_SAMPLES = 2


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
        low_water_pool_ratio=0.20, head_loss_weights=None, scene_options=None, class_weight_mode="none",
        resume_checkpoint=None, speckle=None, require_preprocessed=False, output_dir_is_final=False, require_normalized=False,
    ):
        """准备固定划分、输入统计量、模型、优化器和实验目录。"""
        self.image_dir, self.label_dir = Path(image_dir), Path(label_dir)
        self.output_dir = Path(output_dir)
        self.require_preprocessed = require_preprocessed
        self.inputs_normalized = False
        self.input_preprocessing = {}
        self.model_kwargs, self.model_info = dict(model_kwargs), dict(model_info)
        self.resume_path = Path(resume_checkpoint) if resume_checkpoint is not None else None
        # 仅加载用户指定的本地训练检查点；先放CPU，避免临时占用两份显存。
        resume = torch.load(self.resume_path, map_location="cpu", weights_only=False) if self.resume_path else None
        self.speckle = SpeckleFilter(speckle).config
        # 尽早拦截，避免先花时间统计整套影像后才发现续跑预处理不同。
        if resume is not None and SpeckleFilter(resume.get('speckle')).config != self.speckle:
            raise ValueError('续跑滤波配置与检查点不同；滤波对照请设RESUME_CHECKPOINT=None')
        cprint(f"标准化前斑点滤波：{self.speckle or '关闭'}", 'cyan', flush=True)
        if self.speckle is not None and self.speckle['method'] == 'snap_lee':
            self.model_info['input_representation'] = (
                f"uncalibrated_{self.speckle['input_domain']}; SNAP_Lee_power_domain; "
                "restore_input_domain; train_mean_std")
            cprint('SNAP整景缓存只读复用；缺少成果请先运行src/datasets/image_process.py。辐射校正未执行。', 'cyan', flush=True)
        if self.speckle is not None and not output_dir_is_final:
            self.output_dir = self.output_dir.with_name(self.output_dir.name + '_' + self.speckle['method'])
        self.start_epoch, self.resume_validation = 1, False
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

        self.scene_options = dict(scene_options) if scene_options is not None else None
        self.loader_options = dict(num_workers=0)
        if self.scene_options is not None:
            if self.label_dir.resolve() != (self.image_dir.parent/'labels').resolve():
                raise ValueError('整景影像/标签需位于同一prepared_scenes根目录的images和labels下')
            options = dict(self.scene_options)
            workers = options.pop('num_workers', 2)
            self.windows_per_epoch = options.pop('windows_per_epoch', None)
            self.loader_options = dict(num_workers=workers, pin_memory=self.use_amp)
            if workers:
                self.loader_options.update(persistent_workers=True, prefetch_factor=2)
            self.dataset = SARSceneDataset(self.image_dir.parent, bands=bands, seed=seed, speckle=self.speckle,
                                           require_prepared=require_preprocessed, **options)
            self.inputs_normalized = self.dataset.inputs_normalized
            self.input_preprocessing = self.dataset.input_preprocessing
            if self.inputs_normalized:
                self.model_info['input_representation'] = 'offline_normalized_GeoTIFF; no_train_preprocessing'
            if require_normalized and not self.inputs_normalized:
                raise ValueError('正式训练需使用离线归一化影像；先运行image_process.py的preprocess或normalize模式')
            split_path = self.dataset.manifest_path
            train_ids, val_ids = self.dataset.train_ids, self.dataset.val_ids
            if max_samples is not None:
                # 仅试跑限制窗口数，不改变地理归属；正式配置为None。
                limit = max(1, int(max_samples*(1-val_ratio)))
                train_ids = [train_ids[i] for i in np.linspace(0, len(train_ids)-1, min(limit,len(train_ids)), dtype=int)]
                val_ids = [val_ids[i] for i in np.linspace(0, len(val_ids)-1, min(max(1,max_samples-limit),len(val_ids)), dtype=int)]
            self.train_set, self.val_set = Subset(self.dataset, train_ids), Subset(self.dataset, val_ids)
            self.split_info = dict(split_strategy='scene_grouped_core_windows', data_layout='full_scene',
                                   split_manifest=str(split_path),
                                   split_manifest_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest(),
                                   window_size=self.dataset.window_size, window_halo=self.dataset.halo,
                                   validation_protocol='nonoverlap_core_all_labeled_pixels; halo_context; no resize',
                                   window_cache_key=self.dataset.cache_key,
                                   loader_workers=workers, loader_prefetch=2 if workers else 0,
                                   normalization_protocol='fixed_training_windows_estimate',
                                   normalization_windows_per_scene=self.dataset.normalization_windows,
                                   train_windows_per_epoch=self.windows_per_epoch,
                                   evaluation_scope='smoke_window_subset' if max_samples is not None else 'all_active_labeled_scene_cores')
            revision = self.image_dir.parent / 'revision_settings.json'
            if revision.is_file():
                self.split_info['label_revision_sha256'] = hashlib.sha256(revision.read_bytes()).hexdigest()
                self.split_info['label_semantics'] = '0 non_water; 1 water; 255 invalid; SHP union is water'
            if not output_dir_is_final:
                self.output_dir = self.output_dir.with_name(self.output_dir.name + '_bc_scenes')
        else:
            from datasets.dfc_sample.s1_dataset import WaterDataset, sample_name
            self.dataset = WaterDataset(self.image_dir, self.label_dir, bands, speckle=self.speckle)
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
                if not output_dir_is_final:
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
        self.train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True, **self.loader_options)
        self.val_loader = DataLoader(self.val_set, batch_size=batch_size, shuffle=False, **self.loader_options)
        cprint(f"设备：{self.device}；训练/内部验证：{len(self.train_set)}/{len(self.val_set)}", color="cyan", flush=True)
        self.calculate_normalization()

        # 整景索引已记录每个不重叠核心的0/1数量，直接求和，不重复扫描影像。
        if class_weight_mode not in ("none", "train_pixels", "train_pixels_sqrt"):
            raise ValueError("class_weight_mode应为none、train_pixels或train_pixels_sqrt")
        if class_weight_mode in ("train_pixels", "train_pixels_sqrt"):
            if self.scene_options is None or loss_name != "bce" or sampling != "random":
                raise ValueError("本轮类别平衡采用整景+BCE+random，其他实验请设CLASS_WEIGHT_MODE='none'")
            water, background = self.dataset.windows[self.train_set.indices, 5:7].sum(axis=0).tolist()
            if min(water, background) == 0:
                raise ValueError("类别平衡需要训练标签同时含水体和非水体")
            total = water + background
            weights = np.array((total / (2 * background), total / (2 * water)))
            if class_weight_mode == 'train_pixels_sqrt':
                weights = np.sqrt(weights)
                weights /= np.dot(weights, (background / total, water / total))
            self.loss_kwargs['class_weights'] = tuple(weights.tolist())
            self.split_info['train_class_pixels'] = dict(background=background, water=water)
            cprint(f"训练类别像素：非水体={background}，水体={water}；BCE类别权重={self.loss_kwargs['class_weights']}", 'cyan', flush=True)
        self.model_info['class_weight_mode'] = class_weight_mode
        self.validation_loss_kwargs = dict(self.loss_kwargs)
        self.validation_loss_kwargs.pop('class_weights', None)  # val保持原始损失口径。

        if resume is not None:
            self.check_resume_compatibility(resume)
        construction_kwargs = dict(self.model_kwargs)
        if resume is not None and "pretrained" in construction_kwargs:
            construction_kwargs["pretrained"] = False  # 随后加载完整模型，不再访问HF下载编码器。
        self.model = model_class(**construction_kwargs).to(self.device)
        # 新分支增加了初始化随机数消耗；重置后两轮的抽样、增强和 dropout 从相同随机状态开始。
        if split_path.is_file():
            torch.manual_seed(seed)
        cprint(f"模型：{self.model_info['model_name']}；构建参数：{self.model_kwargs}", color="cyan", flush=True)
        cprint(f"损失：{loss_name}；优化器：{optimizer_name}；初始学习率：{learning_rate}；"
              f"余弦退火：{cosine_schedule}；训练增强：{train_augmentation}", color="cyan", flush=True)
        # 单通道水体方法输出 logit；历史双类别方法使用 CE。
        self.criterion = WaterSegmentationLoss(loss_name, ignore_index=IGNORE_INDEX, **self.loss_kwargs)
        self.validation_criterion = WaterSegmentationLoss(loss_name, ignore_index=IGNORE_INDEX, **self.validation_loss_kwargs)
        cprint(f"损失配置：{self.loss_kwargs}；训练头权重：{self.head_loss_weights or '全部为1'}；"
              "验证仅最终头，不乘训练头/类别权重", color="cyan", flush=True)
        if optimizer_name == "adamw":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
                          if cosine_schedule else None)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # 每次运行单独保存，不覆盖历史实验；CSV 字段沿用原有诊断记录。
        self.run_dir = (self.resume_path.parent if resume is not None else
                        self.output_dir / datetime.now().strftime(RUN_DIRECTORY_FORMAT))
        if resume is None:
            self.run_dir.mkdir(parents=True, exist_ok=False)
        if resume is None and split_path.is_file():
            shutil.copy2(split_path, self.run_dir / RESULT_FILES["split"])
            revision = self.image_dir.parent / 'revision_settings.json'
            if revision.is_file():
                shutil.copy2(revision, self.run_dir / RESULT_FILES["labels"])
        self.metrics_path = self.run_dir / RESULT_FILES["metrics"]
        self.metric_fields = [
            "epoch", "phase", "loss", "iou", "f1", "precision", "recall",
            "tn", "fp", "fn", "tp", "valid_pixels", "ignored_pixels", "skipped_batches",
            "true_water_ratio", "pred_water_ratio", "learning_rate", "amp_scale",
            "amp_skipped_steps", "nonfinite_loss_batches", "elapsed_seconds",
            "accuracy", "specificity", "false_positive_rate", "false_negative_rate",
            "false_discovery_rate", "background_iou", "mean_iou", "balanced_accuracy", "kappa", "mcc",
        ]
        if resume is None:
            with self.metrics_path.open("w", newline="", encoding="utf-8-sig") as stream:
                csv.DictWriter(stream, fieldnames=self.metric_fields).writeheader()
        cprint(f"本次实验目录：{self.run_dir}", color="cyan", attrs=["bold"], flush=True)
        self.best_loss = float("inf")
        self.best_f1 = float("-inf")
        self.best_balanced_accuracy = float("-inf")
        if self.scene_options is not None:
            self.train_files = [self.dataset.names[i] for i in self.train_set.indices]
            self.val_files = [self.dataset.names[i] for i in self.val_set.indices]
            self.split_info['train_scene_files'] = sorted({self.dataset.scenes[int(self.dataset.windows[i,0])]['name'] for i in self.train_set.indices})
            self.split_info['val_scene_files'] = sorted({self.dataset.scenes[int(self.dataset.windows[i,0])]['name'] for i in self.val_set.indices})
        else:
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
                if resume is None:
                    with (self.run_dir / RESULT_FILES["sampling_groups"]).open("w", newline="", encoding="utf-8-sig") as stream:
                        csv.DictWriter(stream, fieldnames=self.group_fields).writeheader()
            if resume is None:
                with (self.run_dir / RESULT_FILES["sampling"]).open("w", newline="", encoding="utf-8-sig") as stream:
                    csv.DictWriter(stream, fieldnames=self.sampling_fields).writeheader()
        cprint(f"采样配置：{self.sampling_config}", color="cyan", flush=True)
        if resume is not None:
            self.restore_training(resume)

    def check_resume_compatibility(self, checkpoint):
        """续跑沿用原实验，拒绝把不同标签、波段或网络混入同一记录。"""
        same_model = lambda values: {k: v for k, v in values.items() if k != "pretrained"}
        if checkpoint.get('inputs_normalized', False) != self.inputs_normalized or checkpoint.get('input_preprocessing', {}) != self.input_preprocessing:
            raise ValueError('续跑检查点与当前离线预处理协议不一致')
        if SpeckleFilter(checkpoint.get('speckle')).config != self.speckle:
            raise ValueError('续跑滤波配置与检查点不同')
        expected = dict(model_name=self.model_info['model_name'], bands=tuple(self.bands),
                        loss_name=self.loss_name, loss_kwargs=self.loss_kwargs,
                        optimizer_name=self.optimizer_name, weight_decay=self.weight_decay,
                        train_augmentation=self.train_augmentation, sampling_config=self.sampling_config,
                        head_loss_weights=self.head_loss_weights, batch_size=self.batch_size,
                        seed=self.seed, epochs=self.epochs, max_samples=self.max_samples, learning_rate=self.learning_rate)
        if same_model(checkpoint['model_kwargs']) != same_model(self.model_kwargs):
            raise ValueError('续跑模型结构与检查点不同；新实验请设RESUME_CHECKPOINT=None')
        for key, value in expected.items():
            if checkpoint.get(key) != value:
                raise ValueError(f'续跑配置{key}与检查点不同；EPOCHS表示原实验总轮数，不能当作追加轮数')
        for key in ('split_manifest_sha256', 'label_revision_sha256', 'window_cache_key',
                    'window_size', 'window_halo', 'train_windows_per_epoch'):
            if key in self.split_info and checkpoint.get(key) != self.split_info[key]:
                raise ValueError(f'续跑数据{key}与检查点不同；请确认仍使用同一标签和划分')
        for key in ('mean', 'std'):
            current = getattr(self, key).detach().cpu().flatten()
            if not torch.allclose(current, checkpoint[key].float().flatten(), rtol=1e-6, atol=1e-6):
                raise ValueError(f'续跑归一化{key}与检查点不同')
        if 'optimizer_state_dict' not in checkpoint:
            raise ValueError('旧检查点缺少优化器状态，不能作为完整续跑；仅加载模型属于另一次微调')

    def restore_training(self, checkpoint):
        """恢复轮次边界状态；本轮验证尚未完成时，只重跑验证，不重复训练。"""
        epoch = int(checkpoint['epoch'])
        phase = checkpoint.get('resume_phase', 'epoch_end')  # 兼容已有last.pth。
        if phase not in ('epoch_end', 'train_end'):
            raise ValueError(f'不支持的续跑阶段：{phase}')
        if tuple(checkpoint['train_files']) != tuple(self.train_files) or tuple(checkpoint['val_files']) != tuple(self.val_files):
            raise ValueError('续跑训练/验证样本顺序与检查点不同')
        if (self.scheduler is None) != (checkpoint.get('scheduler_state_dict') is None):
            raise ValueError('续跑学习率调度器设置不一致')
        self.mean = checkpoint['mean'].to(self.device).view(1, -1, 1, 1)
        self.std = checkpoint['std'].to(self.device).view(1, -1, 1, 1)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if phase == 'train_end':
            # 本轮optimizer更新已完成；验证重放后推进LR，不触发首轮未step的误报。
            self.optimizer._opt_called = True
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        elif self.use_amp:
            cprint('旧检查点未保存有效AMP状态：采用新的GradScaler，不保证逐批一致。', 'yellow', flush=True)
        self.best_loss = checkpoint['best_val_loss']
        self.best_f1 = checkpoint.get('best_val_f1', float('-inf'))
        self.best_balanced_accuracy = checkpoint.get('best_val_balanced_accuracy', float('-inf'))
        self.phase_metrics = checkpoint['phase_metrics']
        self.recent_train_files = checkpoint.get('recent_train_files', [])
        self.resume_validation = phase == 'train_end'
        self.start_epoch = epoch if self.resume_validation else epoch + 1
        if self.sampling != 'random':
            self.sample_losses = checkpoint['sampling_losses']
            self.sample_last_seen = checkpoint['sampling_last_seen']
            self.sampling_plan, self.hard_pool = checkpoint.get('sampling_plan'), checkpoint.get('hard_pool', [])

        # 中断可能留下已完成train但尚未保存checkpoint的CSV行。先备份，再保留对应状态以前的记录。
        for filename in (RESULT_FILES["metrics"], RESULT_FILES["sampling"], RESULT_FILES["sampling_groups"]):
            path = self.run_dir / filename
            if not path.exists():
                if filename == RESULT_FILES["metrics"]:
                    raise ValueError('原实验缺少metrics.csv，不能在原目录续写')
                continue
            with path.open(encoding='utf-8-sig', newline='') as stream:
                reader = csv.DictReader(stream)
                fields, rows = reader.fieldnames, list(reader)
            if filename == RESULT_FILES["metrics"] and fields != self.metric_fields:
                raise ValueError('原metrics.csv字段与当前训练器不一致')
            kept = [row for row in rows if int(row['epoch']) <= epoch and
                    not (self.resume_validation and int(row['epoch']) == epoch and row.get('phase') == 'val')]
            if kept != rows:
                backup = path.with_name(path.stem + '.before_resume_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.csv')
                shutil.copy2(path, backup)
                pending = path.with_suffix('.csv.partial')
                with pending.open('w', encoding='utf-8-sig', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(kept)
                pending.replace(path)
                cprint(f'未对应检查点的尾部记录已备份：{backup}', 'yellow', flush=True)
        rng = checkpoint.get('rng_state')
        if rng:
            random.setstate(rng['python'])
            np.random.set_state(rng['numpy'])
            torch.set_rng_state(rng['torch'])
            if self.use_amp and rng.get('cuda'):
                torch.cuda.set_rng_state_all(rng['cuda'])
        else:
            cprint('旧检查点没有随机数状态：已恢复模型/优化器/调度器/AMP，但不能逐批复现中断前的随机增强。', 'yellow', flush=True)
        cprint(f'续跑：{self.resume_path}；第{self.start_epoch}轮至{self.epochs}轮；'
               f'{"先重跑本轮验证" if self.resume_validation else "开始下一轮训练"}；'
               f'LR={self.optimizer.param_groups[0]["lr"]:.8g}；AMP scale={self.scaler.get_scale():g}',
               'cyan', attrs=['bold'], flush=True)

    def prepare_train_sampling(self, epoch):
        """按上一轮结束时的难度记录生成本轮计划；前几轮仍使用原随机 DataLoader。"""
        if self.sampling == "random":
            if self.scene_options is not None and self.windows_per_epoch is not None:
                plan = torch.randperm(len(self.train_set), generator=torch.Generator().manual_seed(self.seed+epoch))[:self.windows_per_epoch].tolist()
                self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, sampler=plan, **self.loader_options)
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
                                           sampler=self.sampling_plan, **self.loader_options)
        if self.scene_options is not None and self.windows_per_epoch is not None:
            if self.sampling_plan is None:
                self.sampling_plan = torch.randperm(count, generator=torch.Generator().manual_seed(self.seed+epoch)).tolist()
            self.sampling_plan = self.sampling_plan[:self.windows_per_epoch]
            self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size,
                                           sampler=self.sampling_plan, **self.loader_options)
        self.sample_loss_sum = torch.zeros(count, dtype=torch.float64)
        self.sample_observations = torch.zeros(count, dtype=torch.long)
        self.sample_draws = (torch.ones(count, dtype=torch.long) if self.sampling_plan is None else
                             torch.bincount(torch.tensor(self.sampling_plan), minlength=count))
        cprint(f"Epoch {epoch} 采样：{'随机 + 困难' if self.hard_pool else '普通采样/预热'}；"
              f"总抽样 {int(self.sample_draws.sum())}；不同样本 {int((self.sample_draws > 0).sum())}；"
              f"困难池 {len(self.hard_pool)}", color="blue", flush=True)

    def calculate_normalization(self):
        """只用训练集统计均值/标准差；分层和混合模式同时记录标签水体比例。"""
        if self.scene_options is not None:
            mean, std, indices = self.dataset.normalization(self.train_set.indices)
            self.mean = torch.tensor(mean, dtype=torch.float32, device=self.device).view(1,-1,1,1)
            self.std = torch.tensor(std, dtype=torch.float32, device=self.device).view(1,-1,1,1)
            self.split_info['normalization_window_indices'] = indices
            if self.sampling in ('stratified_hard', 'hybrid_hard'):
                counts = self.dataset.windows[self.train_set.indices, 5:7]
                self.sample_water_ratios = torch.from_numpy(counts[:,0]/counts.sum(axis=1))
                self.sample_groups = torch.bucketize(self.sample_water_ratios, torch.tensor([0.,.01,.1,.5],dtype=torch.float64))
            if self.inputs_normalized:
                cprint(f"读取离线归一化影像；原训练统计mean={self.input_preprocessing['mean']}，std={self.input_preprocessing['std']}；训练不再标准化", 'cyan', flush=True)
            else:
                cprint(f"训练窗口统计：mean={mean}；std={std}；缓存可复用", 'cyan', flush=True)
            return
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
        scene_counts = {}

        for step, batch in enumerate(loader, 1):
            image = batch["image"].to(self.device, non_blocking=True)
            label = batch["label"].to(self.device, non_blocking=True)
            valid = label != IGNORE_INDEX
            count = int(valid.sum().item())
            ignored_pixels += label.numel() - count
            if count == 0:
                skipped_batches += 1
                continue
            # 离线归一化输入直接进入模型；原始输入实验才应用标准化。
            if not self.inputs_normalized:
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
                        from supervised.evaluation.visual import FeatureVisualizer
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
                        loss = self.validation_criterion(logits, label)
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
                folder = self.run_dir / RESULT_FILES["features"] / f"epoch_{epoch:03d}"
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
            if not training and self.scene_options is not None:
                for i, scene in enumerate(batch['scene_id']):
                    counts = torch.bincount(label[i][valid[i]]*2 + prediction[i][valid[i]], minlength=4).cpu().numpy()
                    scene_counts[scene] = scene_counts.get(scene, np.zeros(4, dtype=np.int64)) + counts
            if step % 50 == 0:
                cprint(f"Epoch {epoch} {phase}：{step}/{len(loader)} 批", color="yellow" if training else "green", flush=True)

        if valid_pixels == 0:
            raise ValueError(f"{phase} 没有可用于损失计算的标签像素")
        if training and self.sampling != "random":
            seen = self.sample_observations > 0
            self.sample_losses[seen] = (self.sample_loss_sum[seen] / self.sample_observations[seen]).float()
            self.sample_last_seen[seen] = epoch  # 本轮未抽到的样本保留最后一次观测值及时间。
            pool = set(self.hard_pool)
            with (self.run_dir / RESULT_FILES["sampling"]).open("a", newline="", encoding="utf-8") as stream:
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
                with (self.run_dir / RESULT_FILES["sampling_groups"]).open("a", newline="", encoding="utf-8") as stream:
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
        if scene_counts:
            # 每景统计已标注核心像素；面积较大的景不掩盖小景的表现。
            scene_rows = [dict(scene_id=scene, **metrics_from_counts(counts)) for scene, counts in scene_counts.items()]
            with (self.run_dir/RESULT_FILES["val_scenes"].format(epoch=epoch)).open('w',encoding='utf-8-sig',newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=scene_rows[0])
                writer.writeheader(); writer.writerows(scene_rows)
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
        cprint(f"非水体召回率={metrics['specificity']:.4f}；两类平均召回率={metrics['balanced_accuracy']:.4f}；"
              f"两类mIoU={metrics['mean_iou']:.4f}", color="yellow" if training else "green", flush=True)
        if not training and (tp == 0 or tn == 0) and tp + fn > 0 and tn + fp > 0:
            cprint("验证中至少一个类别没有正确预测，不能把水体F1高视为训练成功。", "red", flush=True)
        return metrics

    def save_checkpoint(self, epoch, phase="epoch_end"):
        """保持最低验证 loss 选 best，另外保存 last 和满足条件的异常快照。"""
        metrics = self.phase_metrics.get("val", {})
        epoch_loss = metrics.get("loss", float("inf"))
        improved = phase == "epoch_end" and epoch_loss < self.best_loss
        if improved:
            self.best_loss = epoch_loss
        abnormal = phase == "epoch_end" and (not np.isfinite(epoch_loss) or (
            metrics["tp"] + metrics["fn"] > 0 and metrics["f1"] < self.abnormal_f1_threshold
        ))
        # NumPy随机状态用普通列表保存，继续兼容weights_only=True的检查点读取器。
        numpy_state = np.random.get_state()
        numpy_state = (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:])
        # 保留已有字段；state_dict包含BN缓冲区。
        checkpoint = {
            "model_state_dict": self.model.state_dict(), **self.model_info, **getattr(self, "split_info", {}),
            "model_kwargs": self.model_kwargs, "in_channels": len(self.bands),
            "bands": self.bands, "mean": self.mean.cpu().flatten(), "std": self.std.cpu().flatten(),
            "inputs_normalized": self.inputs_normalized, "input_preprocessing": self.input_preprocessing,
            "speckle": self.speckle,  # 评价/预测恢复相同预处理；旧检查点缺此字段表示关闭。
            "ignore_index": IGNORE_INDEX, "epoch": epoch, "val_loss": epoch_loss,
            "resume_phase": phase,
            "rng_state": dict(python=random.getstate(), numpy=numpy_state,
                              torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if self.use_amp else None),
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
            "validation_loss_kwargs": self.validation_loss_kwargs,
            "head_loss_weights": self.head_loss_weights,  # None 表示旧版默认等权。
            "sampling_config": self.sampling_config,
            "loss_reduction": ("CE valid-pixel mean" if not self.criterion.binary else
                               "train: class-weighted BCE valid-pixel mean; val: unweighted BCE; optional Dice per-image mean, smooth=1"),
        }
        if self.sampling != "random":
            checkpoint.update(sampling_losses=self.sample_losses, sampling_last_seen=self.sample_last_seen,
                              sampling_plan=self.sampling_plan, hard_pool=self.hard_pool)
            if self.sampling in ("stratified_hard", "hybrid_hard"):
                checkpoint.update(sampling_water_ratios=self.sample_water_ratios, sampling_groups=self.sample_groups)
        # 所有网络都保存完整训练状态；验证前的train_end尚未推进调度器。
        checkpoint.update(loss_name=self.loss_name, optimizer_name=self.optimizer_name,
                          weight_decay=self.weight_decay, train_augmentation=self.train_augmentation,
                          optimizer_state_dict=self.optimizer.state_dict(),
                          scaler_state_dict=self.scaler.state_dict(),
                          scheduler_state_dict=self.scheduler.state_dict() if self.scheduler else None)
        filenames = [RESULT_FILES["last"]] if phase == "epoch_end" else [RESULT_FILES["train_end"]]
        if phase == "epoch_end" and self.criterion.binary and metrics["f1"] > self.best_f1:
            self.best_f1 = metrics["f1"]
            filenames.append(RESULT_FILES["best_f1"])
        if self.criterion.binary:
            checkpoint["best_val_f1"] = self.best_f1
        # 两类平均召回率作为补充选模依据；全水/全非水的退化预测仅为0.5。
        if phase == "epoch_end" and metrics["balanced_accuracy"] > self.best_balanced_accuracy:
            self.best_balanced_accuracy = metrics["balanced_accuracy"]
            filenames.append(RESULT_FILES["best_balanced"])
        checkpoint["best_val_balanced_accuracy"] = self.best_balanced_accuracy
        if improved:
            filenames.append(RESULT_FILES["best"])
        if abnormal:
            filenames.append(RESULT_FILES["abnormal"])
        for filename in filenames:
            target = self.run_dir / filename
            pending = target.with_suffix('.pth.partial')
            torch.save(checkpoint, pending)
            pending.replace(target)  # 完整写好后替换，避免中断损坏上一份last.pth。
            cprint(f"已保存：{target}", color="magenta", flush=True)

    def train(self):
        """按总目标轮数续跑；验证失败保留train_end，训练失败回到上一轮last。"""
        try:
            for epoch in range(self.start_epoch, self.epochs + 1):
                if not (self.resume_validation and epoch == self.start_epoch):
                    self.phase_metrics = {}
                    self.recent_train_files = []
                    self.prepare_train_sampling(epoch)
                    self.phase_metrics["train"] = self.run_epoch(epoch, "train", self.train_loader)
                    self.save_checkpoint(epoch, phase="train_end")
                self.phase_metrics["val"] = self.run_epoch(epoch, "val", self.val_loader)
                self.resume_validation = False
                # 本轮验证成功后才推进LR；保存的last对应下一轮的学习率。
                if self.scheduler is not None:
                    self.scheduler.step()
                self.save_checkpoint(epoch)
        except (Exception, KeyboardInterrupt):
            path = self.run_dir / RESULT_FILES["error"]
            with path.open('a', encoding='utf-8') as stream:
                stream.write('\n' + datetime.now().isoformat() + '\n' + traceback.format_exc())
            cprint(f'训练中断，完整报错已保存：{path}', 'red', flush=True)
            raise  # 不隐藏数据错误，也不静默跳过样本。
        finally:
            if self.scene_options is not None:
                self.dataset.close()


def create_trainer(method=None):
    """平级选择网络与保存目录，然后创建同一个公共训练器。"""
    from datasets.snap_preprocess import configure_paths
    configure_paths(SNAP_GPT_PATH, SNAP_CACHE_DIR)
    if DATA_LAYOUT not in ('full_scene', 'patch'):
        raise ValueError('DATA_LAYOUT应为full_scene或patch')
    method = METHOD if method is None else method
    binary_method = method in ("uafnet", "deeplabv3")
    # 在创建训练器时解析，单极化自动构建单通道网络，并重新统计该通道的均值和标准差。
    bands = tuple(POLARIZATION_BANDS[name] for name in POLARIZATIONS)
    if method == "unet":
        model_class = UNet
        model_kwargs = dict(in_channels=len(bands), base_channels=BASE_CHANNELS)
        model_info = dict(model_name="unet", normalization="BatchNorm", base_channels=BASE_CHANNELS)
        experiment = "unet_baseline"

    elif method == "pvt":
        model_class = PVTWaterNet
        model_kwargs = dict(in_channels=len(bands), num_classes=2,
                            decoder_channels=DECODER_CHANNELS, pretrained=PRETRAINED)
        model_info = dict(model_name="pvt", backbone="pvt_v2_b2.in1k",
                          normalization="LayerNorm + decoder BatchNorm", stage="backbone_projection_decoder")
        experiment = "pvt_baseline"

    elif method == "resnet50":
        model_class = ResNet50WaterNet
        model_kwargs = dict(in_channels=len(bands), num_classes=2,
                            decoder_channels=DECODER_CHANNELS, pretrained=PRETRAINED)
        model_info = dict(model_name="resnet50", backbone="resnet50.tv_in1k",
                          normalization="BatchNorm", stage="backbone_projection_decoder")
        experiment = "resnet50_baseline"

    elif method == "uanet":
        model_class = PVTUANet
        model_kwargs = dict(in_channels=len(bands), num_classes=2,
                            decoder_channels=DECODER_CHANNELS, pretrained=PRETRAINED, stage=STAGE)
        # 保留旧检查点名称 pvt_uanet，兼容既有加载器和实验记录。
        model_info = dict(model_name="pvt_uanet", backbone="pvt_v2_b2.in1k", stage=STAGE,
                          normalization="LayerNorm + BatchNorm", output_order="coarse_to_fine",
                          supervision="sum_CE_all_train_heads; final_head_validation")
        experiment = f"pvt_uanet_{STAGE}"

    elif method == "deeplabv3":
        model_class = PVTDeepLabV3Plus
        model_kwargs = dict(in_channels=len(bands), num_classes=1, pretrained=PRETRAINED,
                            backbone_name=DEEPLAB_BACKBONE, decoder_channels=DEEPLAB_DECODER_CHANNELS,
                            low_level_channels=DEEPLAB_LOW_LEVEL_CHANNELS,
                            atrous_rates=DEEPLAB_ASPP_RATES, dropout=DEEPLAB_DROPOUT)
        model_info = dict(
            model_name="pvt_deeplabv3plus", backbone=DEEPLAB_BACKBONE,
            decoder="DeepLabv3+ style ASPP + low-level fusion", output_stride=32,
            decoder_channels=DEEPLAB_DECODER_CHANNELS, low_level_channels=DEEPLAB_LOW_LEVEL_CHANNELS,
            aspp_rates=DEEPLAB_ASPP_RATES, aspp_dropout=DEEPLAB_DROPOUT,
            normalization="PVT LayerNorm + decoder BatchNorm; global pool branch without BN",
            decoder_convolution="standard Conv2d, not separable", align_corners=False,
            timm_version=timm.__version__, pretrained=PRETRAINED, supervision="single_head_" + LOSS_NAME,
            input_representation="single_date_DFC_stored_values; train_mean_std",
            implementation="PVT OS32 + DeepLabv3+ style decoder",
            decoder_reference="https://arxiv.org/abs/1802.02611",
        )
        experiment = "pvt_deeplabv3plus"

    elif method == "uafnet":
        # 水体论文的单模型结构；损失由 LOSS_NAME 选择，BCE+Dice 属于本项目对照实验。
        model_class = PVTUANet
        model_kwargs = dict(in_channels=len(bands), num_classes=1,
                            decoder_channels=DECODER_CHANNELS, pretrained=PRETRAINED, stage="uafm",
                            backbone_name=UAFNET_BACKBONE, final_residual=UAFNET_FINAL_RESIDUAL)
        if UAFNET_REFINEMENT_DECODER not in ("uafm", "unetpp"):
            raise ValueError("UAFNET_REFINEMENT_DECODER 应为 uafm 或 unetpp")
        if UAFNET_REFINEMENT_DECODER == "unetpp":
            model_kwargs["refinement_decoder"] = "unetpp"
        if UAFNET_HIGH_RESOLUTION_REFINEMENT:
            model_kwargs["high_resolution_refinement"] = True
        if UAFNET_MBDC_DILATIONS is not None:
            # dilation 不属于 state_dict，必须保存到 model_kwargs 才能正确重建新结构。
            model_kwargs["mbdc_dilations"] = UAFNET_MBDC_DILATIONS
        resnet_backbone = UAFNET_BACKBONE == "resnet50.tv_in1k"
        model_info = dict(model_name="resnet50_uafnet" if resnet_backbone else "pvt_uafnet",
                          backbone=UAFNET_BACKBONE, stage="uafm",
                          normalization="BatchNorm" if resnet_backbone else "LayerNorm + BatchNorm",
                          output_order="coarse_to_fine",
                          supervision=f"sum_{LOSS_NAME}_all_train_heads; final_head_validation",
                          implementation=f"UAFNet paper-guided SAR adaptation; single {UAFNET_BACKBONE}",
                          final_prediction="coarse_plus_zero_init_correction" if UAFNET_FINAL_RESIDUAL else "direct_refinement",
                          paper_doi="10.1109/IGARSS53475.2024.10641026")
        if resnet_backbone:
            experiment = "resnet50_uafnet"
            model_info["implementation"] = "UAFNet decoder with ResNet50 encoder; project backbone ablation"
        else:
            variant = UAFNET_BACKBONE.split(".")[0].removeprefix("pvt_v2_")
            experiment = "pvt_uafnet" if variant == "b2" else f"pvt_uafnet_{variant}"
        if UAFNET_REFINEMENT_DECODER == "unetpp":
            experiment = "pvt_unetpp"
            model_info.update(model_name="pvt_unetpp_refinement",
                              refinement_decoder="NestedUNetDecoder",
                              implementation="PVT/MBDC/FPN coarse + UNet++-style nested refinement; project ablation",
                              decoder_reference="https://arxiv.org/abs/1807.10165")
        else:
            model_info["refinement_decoder"] = "UAFM"

    else:
        raise ValueError("METHOD 应为 deeplabv3、uafnet、uanet、unet、pvt 或 resnet50")

    if method not in ("unet", "deeplabv3"):
        model_info.update(decoder=("SemanticFPNDecoder + NestedUNetDecoder" if model_info.get("model_name") == "pvt_unetpp_refinement"
                                   else "SemanticFPNDecoder"), decoder_channels=DECODER_CHANNELS,
                          timm_version=timm.__version__, pretrained=PRETRAINED)

    baseline_loss = "bce" if binary_method else "ce"
    if (binary_method and LOSS_NAME not in ("bce", "bce_dice")) or (not binary_method and LOSS_NAME != "ce"):
        raise ValueError("deeplabv3/uafnet 可选 bce/bce_dice；其余双类别模型使用 ce")
    # BCE 保留历史目录；新损失进入独立目录，时间戳继续隔离每次运行。
    if LOSS_NAME != baseline_loss:
        experiment += f"_{LOSS_NAME}"
    if SAMPLING in ("hard", "stratified_hard", "hybrid_hard"):
        if not binary_method or LOSS_NAME != "bce":
            raise ValueError("困难采样支持 DeepLab/UAFNet + BCE；其他方法或损失请设 SAMPLING='random'")
        experiment += "_hard_sampling"
    if method == "uafnet" and UAFNET_FINAL_RESIDUAL:
        experiment += "_residual"
    if SAMPLING == "stratified_hard":
        experiment += "_stratified"
    elif SAMPLING == "hybrid_hard":
        experiment += "_hybrid"
    if method == "uafnet" and UAFNET_HIGH_RESOLUTION_REFINEMENT:
        experiment += "_highres"
    if method == "uafnet" and UAFNET_MBDC_DILATIONS is not None:
        experiment += "_scale_mbdc"
    # 非水体论文分支保持旧等权规则；修改 UAFNet 监督权重时单独保存实验。
    head_loss_weights = (1.0,) if method == "deeplabv3" else tuple(HEAD_LOSS_WEIGHTS) if method == "uafnet" else None
    if method == "uafnet" and head_loss_weights != (1.0, 1.0, 1.0, 1.0):
        experiment += "_head_weights"
        model_info["supervision"] = f"weighted_sum_{LOSS_NAME}_all_train_heads; final_head_validation"
    # 保留原 VV/VH 目录；单极化或其他组合独立保存，防止比较时混淆。
    if POLARIZATIONS != ("VV", "VH"):
        experiment += "_" + "_".join(name.lower() for name in POLARIZATIONS)
    model_info.update(polarizations=POLARIZATIONS, polarization_bands=dict(zip(POLARIZATIONS, bands)))
    if CLASS_WEIGHT_MODE == "train_pixels":
        experiment += "_balanced_bce"
    if DATA_LAYOUT == 'full_scene':
        model_info['input_representation'] = 'BC_single_date_stored_VV; sampled_train_window_mean_std'
    output_dir = OUTPUT_ROOT / experiment if OUTPUT_DIR is None else Path(OUTPUT_DIR)
    cprint(f"训练方法：{method}；极化：{POLARIZATIONS}；文件波段：{bands}；实验目录：{output_dir}", color="cyan", flush=True)
    return WaterSegmentationTrainer(
        model_class=model_class, model_kwargs=model_kwargs, model_info=model_info,
        image_dir=IMAGE_DIR, label_dir=LABEL_DIR, output_dir=output_dir,
        bands=bands, batch_size=BATCH_SIZE, epochs=EPOCHS, learning_rate=LEARNING_RATE,
        val_ratio=VAL_RATIO, seed=SEED, max_samples=MAX_SAMPLES,
        abnormal_f1_threshold=ABNORMAL_F1_THRESHOLD,
        loss_name=LOSS_NAME,
        loss_kwargs=dict(bce_weight=BCE_WEIGHT, dice_weight=DICE_WEIGHT) if LOSS_NAME == "bce_dice" else {},
        optimizer_name=OPTIMIZER_NAME if binary_method else "adam",
        weight_decay=WEIGHT_DECAY if binary_method else 0.0,
        cosine_schedule=binary_method and COSINE_SCHEDULE,
        train_augmentation=binary_method and TRAIN_AUGMENTATION,
        sampling=SAMPLING, hard_warmup_epochs=HARD_WARMUP_EPOCHS,
        hard_sample_ratio=HARD_SAMPLE_RATIO, hard_pool_ratio=HARD_POOL_RATIO,
        low_water_pool_ratio=LOW_WATER_POOL_RATIO,
        head_loss_weights=head_loss_weights,
        scene_options=SCENE_OPTIONS if DATA_LAYOUT == "full_scene" else None,
        class_weight_mode=CLASS_WEIGHT_MODE,
        resume_checkpoint=RESUME_CHECKPOINT,
        require_preprocessed=DATA_LAYOUT == "full_scene",
        require_normalized=DATA_LAYOUT == "full_scene",
        output_dir_is_final=OUTPUT_DIR is not None,
        speckle=(dict(method=SPECKLE_FILTER, input_domain=SPECKLE_INPUT_DOMAIN,
                      window_size=SPECKLE_WINDOW_SIZE, enl=SPECKLE_ENL)
                 if SPECKLE_FILTER != 'none' else None),
    )


def main(method=None):
    """统一执行训练与后处理；后处理重载本次最佳权重，不沿用内存中的最后一轮。"""
    from datasets.snap_preprocess import configure_paths
    configure_paths(SNAP_GPT_PATH, SNAP_CACHE_DIR)
    torch.set_num_threads(CPU_THREADS)
    if RUN_MODE not in ("train_postprocess", "train", "postprocess"):
        raise ValueError("RUN_MODE 应为 train_postprocess、train 或 postprocess")
    checkpoints = list(POSTPROCESS_CHECKPOINTS)
    options = dict(POSTPROCESS_OPTIONS)
    if RUN_MODE != "postprocess":
        trainer = create_trainer() if method is None else create_trainer(method)
        trainer.train()
        if RUN_MODE == "train":
            return trainer.run_dir
        # 单通道水体按验证F1选模；其余方法仍按验证loss选择 best.pth。
        filename = RESULT_FILES["best_f1"] if trainer.criterion.binary else RESULT_FILES["best"]
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
    if DATA_LAYOUT == 'full_scene':
        if len(checkpoints) != 1:
            raise ValueError('整景质量入口目前一次处理一个模型，模型间用同一阈值网格对照')
        from supervised.evaluation.scene_quality import run_scene_quality
        scene_options = dict(SCENE_QUALITY_OPTIONS)
        if RUN_MODE == 'train_postprocess':
            # 新实验不能写进旧模型的quality目录。
            scene_options['output_dir'] = None
        return run_scene_quality(checkpoints[0], image_dir=IMAGE_DIR, label_dir=LABEL_DIR, **scene_options)
    from supervised.evaluation.postprocess import run_postprocess
    cprint(f"开始后处理，使用检查点：{checkpoints}", color="cyan", attrs=["bold"], flush=True)
    return run_postprocess(checkpoints, **options)


if __name__ == "__main__":
    main()
