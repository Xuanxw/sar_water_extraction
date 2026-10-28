"""冻结模型诊断解码细化与输入敏感性；不训练、不更新 BN、不执行后处理。"""
from contextlib import contextmanager
from datetime import datetime
from hashlib import file_digest
from pathlib import Path
import csv
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from datasets.water_dataset import WaterDataset
from models.checkpoint import load_model
from evaluation.metrics import metrics_from_counts
from evaluation.model_eval import threshold_counts, precision_matches

# 修改顶部变量即可独立运行：python src/evaluation/diagnose_decoder.py。
CHECKPOINTS = {
    "pvt_uafm": ROOT / "outputs/pvt_uafnet_hard_sampling_residual_grouped/20261027_212638_633640/best_f1.pth",
    "pvt_nested": ROOT / "outputs/pvt_unetpp_hard_sampling_residual_grouped/20261028_105832_612811/best_f1.pth",
    "pvt_highres": ROOT / "outputs/pvt_uafnet_hard_sampling_residual_highres_grouped/20261027_231028_620780/best_f1.pth",
}
CONDITIONS = ("clean", "storage_gain_0.5", "storage_gain_2.0", "vv_gain_1.5", "resolution_x2")
RANK_CONTROL_MODEL = "pvt_uafm"
RANK_CONDITIONS = ("rank_flat", "rank_spatial_roll", "rank_route_only")
BATCH_SIZE = 2
THRESHOLDS = [i / 100 for i in range(1, 100)]
PRECISION_TARGETS = (.80, .85, .90)
OUTPUT_DIR = None  # None 在 codex_analysis/decoder_diagnosis 下创建时间戳目录。
HEADS = ("coarse", "aux_deep", "aux_middle", "final")
CONFIDENCE_BINS = ("p<=0.1", "0.1<p<0.4", "0.4<=p<0.6", "0.6<=p<0.9", "p>=0.9")
TRANSITION_FIELDS = ("pixels", "water_pixels", "background_pixels", "coarse_fn", "coarse_fp",
                     "final_fn", "final_fp", "tp_recovered", "tp_lost", "fp_removed", "fp_added")


def transform_input(image, valid, condition, bands=(1, 2)):
    """在归一化之前扰动 SAR 原始存储值，标签和有效区域保持原样。"""
    if condition == "clean" or condition in RANK_CONDITIONS:
        result = image
    elif condition.startswith("storage_gain_"):
        result = image * float(condition.rsplit("_", 1)[1])
    elif condition == "vv_gain_1.5":
        result = image.clone()
        result[:, tuple(bands).index(1)] *= 1.5  # 当前 DFC 第 1 波段为 VV，VH 保持不变。
    elif condition == "resolution_x2":
        # 有效像素加权池化，避免无效位置的占位 0 拉低临近 SAR 值。
        mask = valid.unsqueeze(1).to(image.dtype)
        weight = F.avg_pool2d(mask, 2, 2, ceil_mode=True, count_include_pad=False)
        pooled = F.avg_pool2d(image * mask, 2, 2, ceil_mode=True, count_include_pad=False)
        reduced = pooled / weight.clamp_min(1e-6)
        result = F.interpolate(reduced, size=image.shape[-2:], mode="bilinear", align_corners=False)
    else:
        raise ValueError(f"未知条件：{condition}")
    return result.masked_fill(~valid.unsqueeze(1), 0)


def transition_counts(coarse_logits, final_logits, label, ignore_index=-1):
    """按粗预测置信度分层，统计细化修复和新增错误；每个有效像素只属于一层。"""
    p = coarse_logits.float().sigmoid()
    coarse, final = coarse_logits >= 0, final_logits >= 0
    valid, water, background = label != ignore_index, label == 1, label == 0
    strata = (p <= .1, (p > .1) & (p < .4), (p >= .4) & (p < .6),
              (p >= .6) & (p < .9), p >= .9)
    events = (valid, water, background, water & ~coarse, background & coarse,
              water & ~final, background & final, water & ~coarse & final,
              water & coarse & ~final, background & coarse & ~final,
              background & ~coarse & final)
    return torch.stack([torch.stack([(event & valid & region).sum() for event in events])
                        for region in strata])


@contextmanager
def rank_counterfactual(model, mode, statistics=None):
    """临时替换 rank 输出；finally 移除 hook，原参数、BN 和模型结构均不改变。"""
    handles = []
    if mode not in RANK_CONDITIONS:
        yield
        return

    def replace_rank(layer_name):
        def hook(module, inputs, outputs):
            replaced = []
            for branch_name, rank in zip(("foreground", "background"), outputs):
                # flat 保留每图每分支的均值幅值；roll 保留直方图但改变空间对应。
                if mode == "rank_flat":
                    new_rank = rank.float().mean(dim=(-2, -1), keepdim=True).to(rank.dtype).expand_as(rank)
                elif mode == "rank_spatial_roll":
                    new_rank = rank.roll(rank.shape[-1] // 2, dims=-1)
                else:
                    # 保留非零位置的前景/背景路由，只抹去该侧五级权重的区别。
                    support = rank > 0
                    mean_positive = (rank.float().sum(dim=(-2, -1), keepdim=True)
                                     / support.sum(dim=(-2, -1), keepdim=True).clamp_min(1))
                    new_rank = torch.where(support, mean_positive.to(rank.dtype), 0)
                replaced.append(new_rank)
                if statistics is not None:
                    key = (mode, layer_name, branch_name)
                    # 只保留六个标量，不缓存中间等级图。
                    values = torch.stack((rank.new_tensor(rank.numel(), dtype=torch.float64),
                                          rank.double().sum(), new_rank.double().sum(),
                                          rank.double().square().sum(), new_rank.double().square().sum(),
                                          (rank - new_rank).double().abs().sum()))
                    statistics[key] = statistics.get(key, 0) + values
            return tuple(replaced)
        return hook

    try:
        for layer_name in ("ufm_layer4", "ufm_layer3", "ufm_layer2"):
            handles.append(getattr(model, layer_name).rank.register_forward_hook(replace_rank(layer_name)))
        yield
    finally:
        for handle in handles:
            handle.remove()


def prepare_dataset(checkpoint):
    """严格按检查点保存的验证文件与顺序读取，不重新抽样或重新估计均值方差。"""
    dataset = WaterDataset(checkpoint["image_dir"], checkpoint["label_dir"], checkpoint["bands"])
    dataset.select_files(checkpoint["val_files"])
    return dataset


def read_scenes(manifest_path, val_files):
    """场景字段只用于分组描述，不能解释为相互独立的地理区域。"""
    with Path(manifest_path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    active_val = {r["name"] for r in rows if r["split"] == "val" and r["active"] == "1"}
    if active_val != set(val_files):
        raise ValueError("当前清单中的有效 val 与检查点验证集不一致")
    return {r["name"]: r.get("scene") or "未分类" for r in rows if r["name"] in active_val}


def file_sha256(path):
    """分块计算文件摘要，不把大检查点一次性读入内存。"""
    with Path(path).open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def write_csv(path, rows):
    """写入可直接人工核对的统计表；每完成一个模型刷新一次已有结果。"""
    if rows:
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def run_diagnosis(checkpoints=CHECKPOINTS, output_dir=OUTPUT_DIR):
    """同验证集、同阈值、冻结权重推理；返回报告目录，不作跨传感器泛化断言。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir) if output_dir else ROOT / "codex_analysis/decoder_diagnosis" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir.mkdir(parents=True, exist_ok=False)
    tables = {name: [] for name in ("head_metrics", "threshold_metrics", "precision_comparison",
                                  "coarse_to_final", "scene_metrics", "per_image_metrics", "rank_statistics")}
    notes = ["已训练模型冻结权重诊断，不训练、不后处理。",
             "增益作用于原始存储值，是输入压力测试，不等价实际 C/X 波段、物理 dB 变化或真实域外数据。",
             "resolution_x2 是有效值平均池化后双线性恢复原尺寸；标签不变，不等价完整传感器成像仿真。",
             "scene 来自辅助场景分类，不是独立区域；结果来自反复开发使用的同一 val。",
             "rank_flat 将每图每侧等级空间均值铺满；rank_spatial_roll 沿宽平移半幅；rank_route_only 保留路由但统一非零等级。",
             "rank 反事实只测试冻结权重的敏感性；不能证明去除权重后重新训练更优或 UAFM 是唯一瓶颈。",
             f"设备={device}；batch_size={BATCH_SIZE}；num_workers=0；固定阈值=0.5；扫描={THRESHOLDS}",
             f"Precision 要求={PRECISION_TARGETS}；各模型恢复自身检查点训练均值/标准差并核验一致。"]
    reference = None
    for model_name, checkpoint_path in checkpoints.items():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        identity = (tuple(checkpoint["val_files"]), checkpoint["image_dir"], checkpoint["label_dir"],
                    tuple(checkpoint["bands"]), checkpoint.get("ignore_index", -1),
                    checkpoint["mean"].tolist(), checkpoint["std"].tolist(),
                    checkpoint.get("amp_enabled", False))
        if reference is not None and identity != reference:
            raise ValueError("模型的验证名单、数据目录、波段、忽略值、归一化或 AMP 协议不一致")
        reference = identity
        if checkpoint["model_kwargs"].get("num_classes") != 1:
            raise ValueError("本诊断对应当前四头单通道水体模型")
        manifest = Path(checkpoint["split_manifest"])
        manifest_hash = file_sha256(manifest)
        if manifest_hash != checkpoint["split_manifest_sha256"]:
            raise ValueError("清单文件已改变，先核验划分再诊断")
        scenes = read_scenes(manifest, checkpoint["val_files"])
        dataset = prepare_dataset(checkpoint)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        model = load_model(checkpoint).to(device).eval()
        mean = checkpoint["mean"].to(device).view(1, -1, 1, 1)
        std = checkpoint["std"].to(device).clamp_min(1e-6).view(1, -1, 1, 1)
        use_amp = bool(checkpoint.get("amp_enabled", False) and device.type == "cuda")
        ignore_index = checkpoint.get("ignore_index", -1)
        conditions = CONDITIONS + (RANK_CONDITIONS if model_name == RANK_CONTROL_MODEL else ())
        head_counts = {c: torch.zeros(4, 4, dtype=torch.int64, device=device) for c in conditions}
        scans = {c: torch.zeros(len(THRESHOLDS), 4, dtype=torch.int64, device=device) for c in conditions}
        transitions = torch.zeros(5, len(TRANSITION_FIELDS), dtype=torch.int64, device=device)
        scene_counts, scene_sizes, rank_stats = {}, {}, {}
        notes.extend([f"\n模型={model_name}；checkpoint={checkpoint_path}；epoch={checkpoint['epoch']}",
                      f"checkpoint_sha256={file_sha256(checkpoint_path)}",
                      f"manifest_sha256={manifest_hash}；验证样本={len(dataset)}；AMP={use_amp}",
                      f"mean={checkpoint['mean'].tolist()}；std={checkpoint['std'].tolist()}",
                      f"model_kwargs={checkpoint['model_kwargs']}"])
        (output_dir / "run_info.txt").write_text("\n".join(notes), encoding="utf-8")
        print(f"开始 {model_name}：{len(dataset)} 张影像，{len(conditions)} 个条件；{device}；AMP={use_amp}", flush=True)
        with torch.inference_mode():
            for step, batch in enumerate(loader, 1):
                raw = batch["image"].to(device)
                label = batch["label"].to(device)
                valid = batch["valid_mask"].to(device)
                label_valid = label != ignore_index
                for condition in conditions:
                    image = transform_input(raw, valid, condition, checkpoint["bands"])
                    image = ((image - mean) / std).masked_fill(~valid.unsqueeze(1), 0)
                    with rank_counterfactual(model, condition, rank_stats):
                        with torch.autocast(device_type=device.type, enabled=use_amp):
                            outputs = model(image)
                    if not isinstance(outputs, (tuple, list)) or len(outputs) != 4:
                        raise ValueError("本诊断需要 coarse、两个辅助头和 final 共四头输出")
                    for index, logits in enumerate(outputs):
                        prediction = logits[:, 0] >= 0
                        head_counts[condition][index] += torch.bincount(
                            label[label_valid] * 2 + prediction[label_valid].long(), minlength=4)
                    final_score = outputs[-1][:, 0].float()
                    scans[condition] += threshold_counts(final_score, label, THRESHOLDS, ignore_index)
                    if condition == "clean":
                        transitions += transition_counts(outputs[0][:, 0], final_score, label, ignore_index)
                    # 逐图的四个计数很小，可用于配对复核；不保存概率或特征图。
                    for index, name in enumerate(batch["name"]):
                        keep = label_valid[index]
                        counts = torch.bincount(label[index][keep] * 2 + (final_score[index][keep] >= 0).long(), minlength=4)
                        scene = scenes[name]
                        key = condition, scene
                        scene_counts[key] = scene_counts.get(key, 0) + counts
                        scene_sizes[key] = scene_sizes.get(key, 0) + 1
                        tables["per_image_metrics"].append(dict(model=model_name, condition=condition,
                            name=name, scene=scene, **metrics_from_counts(counts.cpu().tolist())))
                    del outputs, image
                if step % 20 == 0 or step == len(loader):
                    print(f"{model_name}：{step}/{len(loader)} 批，已完成同批全部条件", flush=True)
        for condition in conditions:
            for head, counts in zip(HEADS, head_counts[condition].cpu().tolist()):
                row = dict(model=model_name, condition=condition, head=head, threshold=.5,
                           **metrics_from_counts(counts))
                tables["head_metrics"].append(row)
                if head == "final":
                    print(f"{model_name}/{condition}：F1={row['f1']:.5f}，P={row['precision']:.5f}，R={row['recall']:.5f}", flush=True)
            rows = [dict(threshold=t, **metrics_from_counts(counts))
                    for t, counts in zip(THRESHOLDS, scans[condition].cpu().tolist())]
            tables["threshold_metrics"].extend(dict(model=model_name, condition=condition, **r) for r in rows)
            tables["precision_comparison"].extend(dict(model=model_name, condition=condition, **r)
                                                 for r in precision_matches(rows, PRECISION_TARGETS))
        for band, counts in zip(CONFIDENCE_BINS + ("all",),
                               torch.cat((transitions, transitions.sum(0, keepdim=True))).cpu().tolist()):
            tables["coarse_to_final"].append(dict(model=model_name, condition="clean", confidence=band,
                                                 **dict(zip(TRANSITION_FIELDS, counts))))
        for (condition, scene), counts in scene_counts.items():
            tables["scene_metrics"].append(dict(model=model_name, condition=condition, scene=scene,
                samples=scene_sizes[condition, scene], **metrics_from_counts(counts.cpu().tolist())))
        for (condition, layer, branch), values in rank_stats.items():
            count, old_sum, new_sum, old_square, new_square, difference = values.cpu().tolist()
            tables["rank_statistics"].append(dict(model=model_name, condition=condition, layer=layer,
                branch=branch, values=int(count), original_mean=old_sum/count, replaced_mean=new_sum/count,
                original_std=max(old_square/count-(old_sum/count)**2, 0)**.5,
                replaced_std=max(new_square/count-(new_sum/count)**2, 0)**.5,
                mean_absolute_change=difference/count))
        for name, rows in tables.items():
            write_csv(output_dir / f"{name}.csv", rows)
        del model, checkpoint, loader, dataset
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(f"诊断完成：{output_dir}", flush=True)
    return output_dir


if __name__ == "__main__":
    run_diagnosis()
