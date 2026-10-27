"""固定模型与传统阈值的同范围对比：python src/evaluation/compare_threshold.py。"""
from datetime import datetime
from pathlib import Path
import gc
import hashlib
import sys

import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from datasets.water_dataset import WaterDataset, resolve_sample_path
from evaluation.boundary_eval import BoundaryEvaluator, confusion_counts, save_boundary_figure
from evaluation.metrics import metrics_from_counts
from evaluation.postprocess import WaterEnsemble
from thres_seg.threshold import ThresholdSegmenter
from validation import build_manifest, save_reports, summarize, write_csv, read_csv

# 模型和阈值都只评价同一组有标签验证图；不使用整个 train 目录，也不重选最佳模型。
CHECKPOINT = None  # 新 grouped 控制组训练完成后，填写该次 best_f1.pth；不默认使用旧划分模型。
SPLIT_MANIFEST = ROOT / "data/Track1/train/split_manifest.csv"  # 直接执行时强制使用当前有效 val。
IMAGE_DIR = None
LABEL_DIR = None
USE_SAVED_VAL = True  # False 用于明确指定的外部有标签集合；不得用内部验证结果代称独立测试。
MODEL_THRESHOLD = 0.5
MODEL_MASK_DIR = None  # 可填同一检查点、同一概率阈值已生成的 raw 掩膜目录，完全跳过模型推理。
THRESHOLD_BANDS = {"VV": 1, "VH": 2}
THRESHOLD_OPTIONS = dict(water_percent=10, mode="bottom", lee_window=5,
                         noise_ratio=.12, stretch=(1, 99), power_to_db=False)
# 上面是沿用的 Lee + 百分位阈值，不是 Otsu；q=10 固定，不从验证标签估计水体比例。
BATCH_SIZE = 2
USE_AMP = True
SAVE_MASKS = True
MAX_VISUALIZATIONS = 10  # 0 关闭；None 保存全部最终对照图，不输出中间特征。
OUTPUT_DIR = None  # outputs/threshold_comparison/quality/时间戳
SCENE_CSV = ROOT / "docs/s1_samples/scene_catalog_20261026_163548/sample_scenes.csv"
REGION_CSV = None
REGIONS_ARE_INDEPENDENT = False
BOUNDARY_RADII = (1, 3, 5)  # 单位像素；当前切片不推断米制缓冲区。
BOUNDARY_POINTS = 256
SEED = 42


def save_comparison_figure(image, bands, image_valid, label, valid, predictions, rows, output_path):
    """同时显示 SAR、真值、各方法预测及误差；图中精度直接使用逐图计数结果。"""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.patches import Patch

    figure = Figure(figsize=(5 * max(len(predictions), len(bands) + 1), 12))
    FigureCanvasAgg(figure)
    axes = figure.subplots(3, max(len(predictions), len(bands) + 1), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for axis, (name, values) in zip(axes[0], zip(bands, image)):
        low, high = np.percentile(values[image_valid], (2, 98)) if image_valid.any() else (0, 1)
        display = np.clip((values - low) / max(float(high - low), 1e-6), 0, 1)
        axis.imshow(np.where(image_valid, display, np.nan), cmap="gray", vmin=0, vmax=1)
        axis.set_title(f"{name}: P2-P98 display")
    axes[0, len(bands)].imshow(np.where(valid, label, np.nan), cmap="gray", vmin=0, vmax=1)
    axes[0, len(bands)].set_title("Ground truth (common valid area)")
    for i, (method, prediction) in enumerate(predictions.items()):
        row = rows[method]
        axes[1, i].imshow(np.where(valid, prediction, np.nan), cmap="gray", vmin=0, vmax=1)
        axes[1, i].set_title(f"{method}\nF1={row['f1']:.3f} IoU={row['iou']:.3f}\n"
                              f"P={row['precision']:.3f} R={row['recall']:.3f}")
        errors = np.full((*label.shape, 3), 45, dtype=np.uint8)
        errors[valid & (label == 1) & prediction] = (45, 120, 255)
        errors[valid & (label == 0) & prediction] = (255, 65, 65)
        errors[valid & (label == 1) & ~prediction] = (255, 220, 40)
        errors[~valid] = 0
        axes[2, i].imshow(errors)
        axes[2, i].set_title(f"TP={row['tp']} FP={row['fp']} FN={row['fn']} TN={row['tn']}")
    figure.legend(handles=[Patch(color=color, label=label) for color, label in
                            (("#2d78ff", "TP"), ("#ff4141", "FP"), ("#ffdc28", "FN"),
                             ("#2d2d2d", "TN"), ("black", "Ignored"))],
                  loc="lower center", ncol=5)
    figure.suptitle(f"{next(iter(rows.values()))['name']} | same pixels, no land-cover correction")
    figure.tight_layout(rect=(0, .04, 1, .96))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=120)


def run_comparison(checkpoint_path=CHECKPOINT, image_dir=IMAGE_DIR, label_dir=LABEL_DIR,
                   use_saved_val=USE_SAVED_VAL, output_dir=OUTPUT_DIR,
                   threshold_bands=THRESHOLD_BANDS, threshold_options=THRESHOLD_OPTIONS,
                   model_threshold=MODEL_THRESHOLD, batch_size=BATCH_SIZE, use_amp=USE_AMP,
                   model_mask_dir=MODEL_MASK_DIR,
                   scene_csv=SCENE_CSV, region_csv=REGION_CSV,
                   independent_regions=REGIONS_ARE_INDEPENDENT,
                   save_masks=SAVE_MASKS, max_visualizations=MAX_VISUALIZATIONS,
                   boundary_radii=BOUNDARY_RADII, boundary_points=BOUNDARY_POINTS, split_manifest=None):
    """一次模型推理生成全部对照结果；各方法共享名单、有效域、缓冲带与固定采样点。"""
    if not threshold_bands or not 0 < model_threshold < 1:
        raise ValueError("需要阈值波段，且模型概率阈值须在 0 与 1 之间")
    if not use_saved_val and (image_dir is None or label_dir is None):
        raise ValueError("外部评价须明确提供影像与真实标签目录")
    if independent_regions and (use_saved_val or region_csv is None):
        raise ValueError("独立区域测试须使用外部有标签集合与真实区域映射")
    if checkpoint_path is None:
        raise ValueError("请将 CHECKPOINT 设为新 grouped 实验的 best_f1.pth，再比较当前 val；旧模型可能训练过这些样本")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    images = Path(checkpoint["image_dir"] if image_dir is None else image_dir)
    labels = Path(checkpoint["label_dir"] if label_dir is None else label_dir)
    split_sha256 = ""
    if split_manifest is not None:
        # 新 val 曾属于旧模型的训练集，必须核对划分来源，不能只改评价目录。
        split_manifest = Path(split_manifest)
        split_sha256 = hashlib.sha256(split_manifest.read_bytes()).hexdigest()
        split_rows = read_csv(split_manifest)
        active = [r for r in split_rows if int(r["active"]) == 1]
        expected_train = {r["name"] for r in active if r["split"] == "train"}
        expected_val = {r["name"] for r in active if r["split"] == "val"}
        if (not use_saved_val or checkpoint.get("split_manifest_sha256") != split_sha256
                or set(checkpoint["train_files"]) != expected_train
                or set(checkpoint["val_files"]) != expected_val
                or len(checkpoint["val_files"]) != len(expected_val)
                or expected_train & expected_val):
            raise ValueError("检查点与当前分组清单不一致；必须使用本划分训练的模型及完整 active val 名单")
        images = split_manifest.parent / "images/val"
        labels = split_manifest.parent / "labels/val"
    manifest = build_manifest(checkpoint, images, labels, use_saved_val,
                              scene_csv if use_saved_val else None)
    epoch = checkpoint["epoch"]
    model_bands = tuple(checkpoint["bands"])
    del checkpoint
    gc.collect()
    destination = (ROOT / "outputs/threshold_comparison/quality" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                   if output_dir is None else Path(output_dir))
    destination.mkdir(parents=True, exist_ok=False)
    write_csv(destination / "samples.csv", manifest)

    network = WaterEnsemble([checkpoint_path], use_amp) if model_mask_dir is None else None
    # 联合波段的有效域用于所有方法，避免 VV/VH 有效面积不同造成不公平比较。
    bands = tuple(sorted(set(model_bands) | set(threshold_bands.values())))
    dataset = WaterDataset(images, labels, bands)
    dataset.select_files([row["name"] for row in manifest])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    # predict 接收排序后的成员波段并集，再按 checkpoint 顺序重排一次，不能预先重复重排。
    model_channels = [bands.index(band) for band in network.bands] if network is not None else None
    segmenter = ThresholdSegmenter(**threshold_options)
    methods = ["model", *[f"threshold_{name}" for name in threshold_bands]]
    rows_by_method = {method: [] for method in methods}
    boundary = BoundaryEvaluator(destination / "boundary", boundary_radii, boundary_points, SEED)
    thresholds, per_image = [], []
    try:
        for batch in loader:
            # 标准化仍由 WaterEnsemble 使用 checkpoint 的训练均值/标准差完成。
            probabilities = None
            if network is not None:
                _, probability = network.predict(batch["image"][:, model_channels], batch["valid_mask"])
                probabilities = probability.cpu().numpy()
            for index, name in enumerate(batch["name"]):
                image = batch["image"][index].numpy()
                image_valid = batch["valid_mask"][index].numpy()
                label = batch["label"][index].numpy()
                valid = image_valid & np.isin(label, (0, 1))
                if probabilities is None:
                    mask_path = Path(model_mask_dir) / Path(name).with_suffix(".tif")
                    with rasterio.open(mask_path) as source, rasterio.open(resolve_sample_path(images, name)) as original:
                        if (source.count != 1 or source.shape != original.shape or
                                source.crs != original.crs or source.transform != original.transform):
                            raise ValueError(f"缓存模型掩膜与输入影像网格不一致：{mask_path}")
                        mask = source.read(1, masked=True)
                    available = ~np.ma.getmaskarray(mask) & np.isin(mask.data, (0, 1))
                    # 缓存若在影像有效域缺像素，立即报错；不能仅对某方法缩小计量范围。
                    if np.any(image_valid & ~available):
                        raise ValueError(f"缓存模型掩膜未覆盖共同影像有效域：{mask_path}")
                    predictions = {"model": (mask.data == 1) & image_valid}
                else:
                    predictions = {"model": (probabilities[index] >= model_threshold) & image_valid}
                used_thresholds = {"model": model_threshold}
                for caption, band in threshold_bands.items():
                    method = f"threshold_{caption}"
                    # 分割只接收影像有效域；标签仅在后续计量中使用。
                    prediction, _, level = segmenter.predict(image[bands.index(band)], image_valid)
                    predictions[method] = prediction
                    used_thresholds[method] = level
                    thresholds.append(dict(name=name, method=method, water_percent=segmenter.water_percent,
                                           intensity_threshold=level))
                current = {}
                for method, prediction in predictions.items():
                    row = dict(name=name, **metrics_from_counts(confusion_counts(label, prediction, valid)))
                    rows_by_method[method].append(row)
                    current[method] = row
                    per_image.append(dict(method=method, **row))
                    if save_masks:
                        path = destination / "masks" / method / Path(name).with_suffix(".tif")
                        path.parent.mkdir(parents=True, exist_ok=True)
                        with rasterio.open(resolve_sample_path(images, name)) as source:
                            profile = source.profile.copy()
                        profile.update(driver="GTiff", count=1, dtype="uint8", nodata=255)
                        # 输出掩膜依据影像有效域，不用标签孔洞删除预测，便于独立使用。
                        with rasterio.open(path, "w", **profile) as target:
                            target.write(np.where(image_valid, prediction, 255).astype("uint8"), 1)
                regions, points = boundary.update(name, label, valid, predictions,
                                                  probabilities=None if probabilities is None else {"model": probabilities[index]},
                                                  thresholds=used_thresholds)
                count = len(rows_by_method["model"])
                if max_visualizations is None or count <= max_visualizations:
                    display_bands = {caption: image[bands.index(band)] for caption, band in threshold_bands.items()}
                    save_comparison_figure(np.asarray(list(display_bands.values())), display_bands,
                                           image_valid, label, valid, predictions, current,
                                           destination / "figures" / Path(name).with_suffix(".png"))
                    save_boundary_figure(label, valid, regions, points,
                                         destination / "boundary/figures" / Path(name).with_suffix(".png"))
            if len(rows_by_method["model"]) % 50 < batch_size:
                print(f"同范围对比：{len(rows_by_method['model'])}/{len(dataset)}", flush=True)
        boundary.save()
    finally:
        boundary.close()

    # 同一套逐图结果分别按重复关系、自动场景与可选真实区域汇总，不自动挑选阈值方法。
    summary = []
    for method, rows in rows_by_method.items():
        method_dir = destination / method
        method_dir.mkdir()
        reports = save_reports(method_dir, rows, manifest, region_csv, independent_regions)
        summary.extend(dict(method=method, **row) for row in reports)
    write_csv(destination / "metrics.csv", summary)
    write_csv(destination / "per_image.csv", per_image)
    write_csv(destination / "thresholds.csv", thresholds)
    # 局部指标也提供同样的去重子集；保留 all_pixels/sample_points，不能混为面积指标。
    subset_names = {
        "all": {row["name"] for row in manifest},
        "duplicate_in_train": {row["name"] for row in manifest if row["duplicate_in_train"]},
        "no_exact_train_duplicate": {row["name"] for row in manifest if not row["duplicate_in_train"]},
        "unique_nontrain": {row["name"] for row in manifest if row["unique_nontrain"]},
    }
    boundary_groups = []
    for subset, names in subset_names.items():
        for method, radius, region, scope in boundary.totals:
            selected = [row for row in boundary.rows if row["name"] in names and
                        (row["method"], row["radius_px"], row["region"], row["scope"]) ==
                        (method, radius, region, scope)]
            boundary_groups.append(dict(subset=subset, method=method, radius_px=radius,
                                        region=region, scope=scope, **summarize(selected)))
    write_csv(destination / "boundary_groups.csv", boundary_groups)
    write_csv(destination / "protocol.csv", [dict(
        checkpoint=str(Path(checkpoint_path).resolve()), epoch=epoch, image_dir=str(images),
        label_dir=str(labels), use_saved_val=use_saved_val, samples=len(manifest),
        split_manifest=str(split_manifest or ""), split_manifest_sha256=split_sha256,
        dataset_role="internal_model_selection" if use_saved_val else "external_labeled",
        model_threshold=model_threshold, normalization="checkpoint training mean/std",
        batch_size=batch_size, use_amp=network.use_amp if network is not None else False, common_bands=str(bands),
        prediction_source="cached_masks" if model_mask_dir is not None else "checkpoint_inference",
        model_mask_dir=str(Path(model_mask_dir).resolve()) if model_mask_dir is not None else "",
        model_probability_available=network is not None,
        cached_mask_provenance="caller must verify same checkpoint and declared threshold" if model_mask_dir else "",
        valid_rule="all selected SAR bands valid and finite; evaluation additionally requires label 0/1",
        threshold_method="Lee + fixed per-image percentile (not Otsu)", threshold_options=str(threshold_options),
        threshold_bands=str(threshold_bands), threshold_tuning=False, landcover_correction=False,
        boundary_radii_px=str(tuple(boundary_radii)), boundary_points_per_region=boundary_points,
        boundary_seed=SEED, boundary_metric="strict band pixel counts; not tolerance Boundary F1",
        region_csv=str(region_csv or ""), independent_regions=independent_regions)])
    for row in summary:
        if row["subset"] in ("all", "unique_nontrain"):
            print(f"{row['method']}/{row['subset']}: n={row['samples']} "
                  f"IoU={row['iou']:.6f} F1={row['f1']:.6f} "
                  f"P={row['precision']:.6f} R={row['recall']:.6f}", flush=True)
    print(f"对比结果：{destination}", flush=True)
    return summary


if __name__ == "__main__":
    run_comparison(split_manifest=SPLIT_MANIFEST)
