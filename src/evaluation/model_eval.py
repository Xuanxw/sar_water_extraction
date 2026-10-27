"""独立评价已训练模型；修改顶部变量后运行 python src/evaluation/model_eval.py。"""
from math import log
from pathlib import Path
from datetime import datetime
import csv
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))  # 从 evaluation 直接运行时找到项目数据集与模型模块。

import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets.water_dataset import WaterDataset
from models.checkpoint import load_model
from evaluation.metrics import metrics_from_counts

CHECKPOINT_PATH = ROOT / "outputs/pvt_uafnet_hard_sampling_residual/20261022_160255_467390/best_f1.pth"
IMAGE_DIR = None  # None 使用检查点中的影像目录；数据迁移后可直接填写新路径。
LABEL_DIR = None  # None 使用检查点中的标签目录；计算 F1 必须有真实标签。
USE_SAVED_VAL = True  # True 只评价原内部验证集；独立有标签数据集设为 False。
BATCH_SIZE = None     # None 沿用训练 batch size，便于复核日志。
USE_AMP = None        # None 沿用检查点设置；CPU 始终使用 float32。
THRESHOLD = 0.5       # 单通道水体概率阈值；双类别模型仍用 argmax。
# ---------- 可选评估模式：批量进行阈值扫描，或单次评价指定阈值。 ----------
MODE = "scan"        # scan 比较保留的三组主线；single 评价上面的单个检查点。
SCAN_CHECKPOINTS = {
    "Residual": ROOT / "outputs/pvt_uafnet_hard_sampling_residual/20261022_160255_467390/best_f1.pth",
    "HeadWeights": ROOT / "outputs/pvt_uafnet_hard_sampling_residual_head_weights/20261026_214355_751734/best_f1.pth",
    "ScaleMBDC": ROOT / "outputs/pvt_uafnet_hard_sampling_residual_scale_mbdc/20261027_153406_423363/best_f1.pth",
}
SCAN_THRESHOLDS = [i / 1000 for i in range(10, 991)]  # 0.01～0.99，步长 0.001；各模型相同。
PRECISION_TARGETS = (0.80, 0.85, 0.90)


def threshold_counts(score, label, thresholds, ignore_index=-1):
    """按水体 logit 分箱，一次统计所有阈值的 [TN,FP,FN,TP]，不创建阈值×像素大张量。"""
    valid = label != ignore_index
    edges = torch.logit(torch.as_tensor(thresholds, dtype=torch.float32, device=score.device))
    # right=True：恰等于阈值的像素也算水体，与 probability >= threshold 一致。
    bins = torch.bucketize(score[valid].float().contiguous(), edges, right=True)
    size = len(thresholds) + 1
    histogram = torch.bincount(label[valid] * size + bins, minlength=2 * size).reshape(2, size)
    tail = histogram.flip(1).cumsum(1).flip(1)[:, 1:]
    fp, tp = tail[0], tail[1]
    tn, fn = histogram[0].sum() - fp, histogram[1].sum() - tp
    return torch.stack((tn, fp, fn, tp), dim=1)


def precision_matches(rows, targets):
    """每档要求选 P>=目标且 Recall 最高的实测点；并列时选 Precision 更高者。"""
    matches = []
    for target in targets:
        eligible = [r for r in rows if r["precision"] >= target and r["tp"] + r["fp"] > 0]
        best = max(eligible, key=lambda r: (r["recall"], r["precision"])) if eligible else None
        matches.append(dict(target_precision=target, status="reachable" if best else "unreachable",
                            **({k: best[k] for k in ("threshold", "precision", "recall", "f1", "fp", "fn")}
                               if best else {k: "" for k in ("threshold", "precision", "recall", "f1", "fp", "fn")})))
    return matches


def evaluate(checkpoint_path=CHECKPOINT_PATH, image_dir=IMAGE_DIR, label_dir=LABEL_DIR,
             use_saved_val=USE_SAVED_VAL, batch_size=BATCH_SIZE, use_amp=USE_AMP,
             threshold=THRESHOLD, thresholds=None):
    """恢复模型和输入处理，累计整套数据的混淆矩阵，在终端打印并返回指标。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = load_model(checkpoint).to(device).eval()  # 不更新权重或 BN 统计量。
    bands = checkpoint["bands"]
    ignore_index = checkpoint.get("ignore_index", -1)
    image_dir = Path(checkpoint["image_dir"] if image_dir is None else image_dir)
    label_dir = Path(checkpoint["label_dir"] if label_dir is None else label_dir)
    dataset = WaterDataset(image_dir, label_dir, bands)
    if use_saved_val:
        # 按训练时保存的文件名和顺序评价，不重新随机划分。
        dataset.select_files(checkpoint["val_files"])
    batch_size = checkpoint["batch_size"] if batch_size is None else batch_size
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    use_amp = checkpoint.get("amp_enabled", False) if use_amp is None else use_amp
    use_amp = bool(use_amp and device.type == "cuda")
    mean = checkpoint["mean"].to(device).view(1, -1, 1, 1)
    std = checkpoint["std"].to(device).clamp_min(1e-6).view(1, -1, 1, 1)
    binary = checkpoint.get("model_kwargs", {}).get("num_classes", 2) == 1
    criterion = (nn.BCEWithLogitsLoss(reduction="none") if binary
                 else nn.CrossEntropyLoss(ignore_index=ignore_index))
    # 直接比较 logit，避免半精度 sigmoid 将接近 0.5 的概率舍入到阈值。
    logit_threshold = log(threshold / (1 - threshold)) if binary else None
    confusion = torch.zeros(4, dtype=torch.int64, device=device)
    scan = (torch.zeros(len(thresholds), 4, dtype=torch.int64, device=device)
            if thresholds is not None else None)
    loss_sum = valid_pixels = ignored_pixels = 0

    print(f"检查点：{checkpoint_path}；训练轮次：{checkpoint['epoch']}", flush=True)
    print(f"数据：{'原内部验证集' if use_saved_val else '指定目录全部有标签影像'}；"
          f"样本数：{len(dataset)}；波段：{bands}；设备：{device}；AMP：{use_amp}", flush=True)
    print(f"判别方式：{'单通道概率阈值 ' + str(threshold) if binary else '双类别 argmax'}", flush=True)
    with torch.inference_mode():
        for step, batch in enumerate(loader, 1):
            image = batch["image"].to(device)
            label = batch["label"].to(device)
            valid = label != ignore_index
            count = int(valid.sum().item())
            ignored_pixels += label.numel() - count
            if count == 0:
                continue
            image = (image - mean) / std
            image = image.masked_fill(~batch["valid_mask"].to(device).unsqueeze(1), 0)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(image)
                logits = outputs[-1] if isinstance(outputs, (tuple, list)) else outputs
                # 与训练验证一致：仅使用最后一个输出头，排除标签 -1。
                if binary:
                    loss = criterion(logits[:, 0].float(), label.clamp_min(0).float())[valid].mean()
                    prediction = (logits[:, 0] >= logit_threshold).long()
                else:
                    loss = criterion(logits, label)
                    prediction = logits.argmax(dim=1)
            loss_sum += loss.item() * count
            valid_pixels += count
            # 编码：真实类别*2+预测类别，依次对应 TN、FP、FN、TP。
            confusion += torch.bincount(label[valid] * 2 + prediction[valid], minlength=4)
            if scan is not None:
                # 双类别的 logit 差等价于水体 softmax 的 logit；只使用最终输出头。
                score = logits[:, 0].float() if binary else logits[:, 1].float() - logits[:, 0].float()
                scan += threshold_counts(score, label, thresholds, ignore_index)
            if step % 50 == 0:
                print(f"评价：{step}/{len(loader)} 批", flush=True)

    if valid_pixels == 0:
        raise ValueError("没有可用于评价的有效标签像素")
    tn, fp, fn, tp = confusion.tolist()
    metrics = dict(loss=loss_sum / valid_pixels, ignored_pixels=ignored_pixels,
                   **metrics_from_counts((tn, fp, fn, tp)))
    print(f"Loss={metrics['loss']:.6f}；水体 F1={metrics['f1']:.6f}；IoU={metrics['iou']:.6f}；"
          f"Precision={metrics['precision']:.6f}；Recall={metrics['recall']:.6f}", flush=True)
    print(f"TN/FP/FN/TP：{tn}/{fp}/{fn}/{tp}；有效/忽略像素：{valid_pixels}/{ignored_pixels}", flush=True)
    print(f"真实/预测水体占比：{metrics['true_water_ratio']:.4%}/{metrics['pred_water_ratio']:.4%}", flush=True)
    if scan is None:
        return metrics
    rows = []
    for value, (tn, fp, fn, tp) in zip(thresholds, scan.cpu().tolist()):
        rows.append(dict(threshold=value, **metrics_from_counts((tn, fp, fn, tp))))
    return rows


def run_threshold_scan(checkpoints=SCAN_CHECKPOINTS, thresholds=SCAN_THRESHOLDS,
                       targets=PRECISION_TARGETS, output_dir=None):
    """同一验证集各推理一次，保存阈值表、共同精确率要求对照表和 PR 曲线。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not thresholds or not all(0 < t < 1 for t in thresholds) or sorted(set(thresholds)) != list(thresholds):
        raise ValueError("扫描阈值须在 0～1 之间，严格递增且不重复")
    # 先确认各模型使用相同验证名单及标签来源，再开始推理。
    reference = None
    epochs = {}
    for name, path in checkpoints.items():
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        identity = (checkpoint["val_files"], checkpoint["image_dir"], checkpoint["label_dir"],
                    checkpoint["bands"], checkpoint.get("ignore_index", -1))
        if reference is not None and identity != reference:
            raise ValueError("扫描要求相同验证名单、数据目录、输入波段和忽略值")
        reference = identity
        epochs[name] = checkpoint["epoch"]
        del checkpoint
    output_dir = (ROOT / "codex_analysis/threshold_scan" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                  if output_dir is None else Path(output_dir))
    output_dir.mkdir(parents=True, exist_ok=False)
    all_rows, comparisons = [], []
    fig, axis = plt.subplots(figsize=(8, 6))
    for name, path in checkpoints.items():
        print(f"开始扫描：{name}；{len(thresholds)} 个阈值；不使用土地覆盖后处理", flush=True)
        rows = evaluate(path, use_saved_val=True, threshold=.5, thresholds=thresholds)
        all_rows.extend(dict(model=name, checkpoint=str(path), epoch=epochs[name], **r) for r in rows)
        matched = precision_matches(rows, targets)
        comparisons.extend(dict(model=name, **r) for r in matched)
        for r in matched:
            print(f"{name}：P要求={r['target_precision']:.0%}；{r}", flush=True)
        best = max(rows, key=lambda r: r["f1"] if r["tp"] + r["fp"] + r["fn"] else -1)
        print(f"{name} 扫描最佳 F1：{best}", flush=True)
        # 无预测点的 P 不定义，不把它画成精确率 100% 的曲线端点；不做插值或包络美化。
        plotted = sorted((r for r in rows if r["tp"] + r["fp"] > 0), key=lambda r:r["recall"])
        axis.plot([r["recall"] for r in plotted], [r["precision"] for r in plotted], label=name)
        for r in matched:
            if r["status"] == "reachable":
                axis.scatter(r["recall"], r["precision"], s=25)
    for filename, rows in (("threshold_metrics.csv", all_rows), ("precision_comparison.csv", comparisons)):
        with (output_dir / filename).open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    axis.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1),
             title="Validation PR sweep (raw predictions, no postprocessing)")
    axis.grid(alpha=.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "pr_curve.png", dpi=180)
    plt.close(fig)
    print(f"扫描完成：{output_dir}；结果属于验证集调参，不代表独立测试成绩。", flush=True)
    return output_dir


def main():
    """独立评价入口；既不训练，也不重复执行质量图和 ESA 修正。"""
    if MODE == "scan":
        return run_threshold_scan()
    if MODE == "single":
        return evaluate()
    raise ValueError("MODE 应为 scan 或 single")


if __name__ == "__main__":
    main()
