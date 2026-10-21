"""独立评价已训练模型；修改顶部变量后运行 python src/evaluate.py。"""
from math import log
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets.water_dataset import WaterDataset
from models.checkpoint import load_model

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = ROOT / "outputs/pvt_uafnet/20261021_102433_387373/best_f1.pth"
IMAGE_DIR = None  # None 使用检查点中的影像目录；数据迁移后可直接填写新路径。
LABEL_DIR = None  # None 使用检查点中的标签目录；计算 F1 必须有真实标签。
USE_SAVED_VAL = True  # True 只评价原内部验证集；独立有标签数据集设为 False。
BATCH_SIZE = None     # None 沿用训练 batch size，便于复核日志。
USE_AMP = None        # None 沿用检查点设置；CPU 始终使用 float32。
THRESHOLD = 0.5       # 单通道水体概率阈值；双类别模型仍用 argmax。


def evaluate(checkpoint_path=CHECKPOINT_PATH, image_dir=IMAGE_DIR, label_dir=LABEL_DIR,
             use_saved_val=USE_SAVED_VAL, batch_size=BATCH_SIZE, use_amp=USE_AMP,
             threshold=THRESHOLD):
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
        dataset.image_paths = [image_dir / name for name in checkpoint["val_files"]]
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
            if step % 50 == 0:
                print(f"评价：{step}/{len(loader)} 批", flush=True)

    if valid_pixels == 0:
        raise ValueError("没有可用于评价的有效标签像素")
    tn, fp, fn, tp = confusion.tolist()
    # 分母为 0 时给出 NaN：例如整套数据没有水体，F1 无法据此评价水体提取能力。
    metrics = dict(
        loss=loss_sum / valid_pixels,
        f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float("nan"),
        iou=tp / (tp + fp + fn) if tp + fp + fn else float("nan"),
        precision=tp / (tp + fp) if tp + fp else 0.0,
        recall=tp / (tp + fn) if tp + fn else float("nan"),
        tn=tn, fp=fp, fn=fn, tp=tp,
        valid_pixels=valid_pixels, ignored_pixels=ignored_pixels,
        true_water_ratio=(tp + fn) / valid_pixels,
        pred_water_ratio=(tp + fp) / valid_pixels,
    )
    print(f"Loss={metrics['loss']:.6f}；水体 F1={metrics['f1']:.6f}；IoU={metrics['iou']:.6f}；"
          f"Precision={metrics['precision']:.6f}；Recall={metrics['recall']:.6f}", flush=True)
    print(f"TN/FP/FN/TP：{tn}/{fp}/{fn}/{tp}；有效/忽略像素：{valid_pixels}/{ignored_pixels}", flush=True)
    print(f"真实/预测水体占比：{metrics['true_water_ratio']:.4%}/{metrics['pred_water_ratio']:.4%}", flush=True)
    return metrics


if __name__ == "__main__":
    evaluate()
