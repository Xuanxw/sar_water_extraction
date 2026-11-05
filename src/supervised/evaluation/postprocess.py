"""模型推理后执行以下后处理步骤：

1. 内置/外部历史水体显式补入（按开关执行，保留raw）。
2. 统计水体面积，并在共同有效覆盖内计算历史新增/消退范围与面积。
3. 使用DEM/DSM高程估计水深，输出水位、距离及质量标记。
4. 对照raw/post的像元和岸线精度，输出掩膜、统计表及质量图。

阈值GIS的gis_worker.py另有DEM坡度/坡向与地形排除、OSM建筑/道路排除、
WorldCover农田排除、填孔及连通域清理；这些步骤尚未接入本入口。
本文件terrain_gradient仅供质量图显示，不参与水体掩膜修正。
"""
from supervised.models.checkpoint import load_model
from supervised.evaluation.metrics import metrics_from_counts
from supervised.evaluation.landcover import read_landcover, merge_landcover
from supervised.evaluation.boundary_eval import BoundaryEvaluator, save_boundary_figure
from datasets.s1_dataset import WaterDataset, resolve_sample_path, sample_name
from thres_segmentation.auxiliary_data import prepare_auxiliary_inputs
from thres_segmentation.water_analysis import analyze_water, read_reference_water
from datasets.speckle import SpeckleFilter
from torch.utils.data import DataLoader
import torch
import rasterio
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import csv
import json
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
matplotlib.use("Agg")  # 只保存对照图，不弹出窗口。


DEM_NODATA = -9999     # 本地 DEM 的缺测编码，仅用于辅助图；0 高程仍保留。
DEM_PIXEL_SIZE_METERS = None  # 确认共同切片网格后填写 (行间距, 列间距)，单位米。
DEM_HEIGHT_SCALE = None       # 确认后填写“每个存储值对应多少米”，例如 1.0。


class WaterEnsemble:
    """逐模型恢复各自的输入处理，平均水体概率；一个成员时就是普通推理。"""

    def __init__(self, checkpoint_paths, use_amp=True):
        if not checkpoint_paths:
            raise ValueError("checkpoint_paths 至少需要一个已训练检查点")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = use_amp and self.device.type == "cuda"
        self.members = []
        for path in checkpoint_paths:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            model = load_model(checkpoint).to(self.device).eval()
            # 不保留优化器和整份 state_dict，避免多个模型占用不必要的 CPU 内存。
            self.members.append(dict(
                path=Path(path), model=model, bands=tuple(checkpoint["bands"]),
                mean=checkpoint["mean"].to(self.device).view(1, -1, 1, 1),
                std=checkpoint["std"].to(self.device).clamp_min(1e-6).view(1, -1, 1, 1),
                image_dir=checkpoint["image_dir"], label_dir=checkpoint["label_dir"],
                val_files=checkpoint["val_files"], ignore_index=checkpoint.get("ignore_index", -1),
                speckle_filter=SpeckleFilter(checkpoint.get('speckle')),
            ))
            print(f"集成成员：{path}；模型：{checkpoint['model_name']}；轮次：{checkpoint['epoch']}", flush=True)
        # 支持成员各自选择不同波段、重复波段；读取它们的并集，再分别重排和标准化。
        self.bands = tuple(sorted({b for member in self.members for b in member["bands"]}))

    @torch.inference_mode()
    def predict(self, image, valid_mask):
        """返回各成员的水体概率和等权平均概率；不平均 logits 或二值掩膜。"""
        image, valid_mask = image.to(self.device), valid_mask.to(self.device)
        probabilities = []
        for member in self.members:
            indices = [self.bands.index(b) for b in member["bands"]]
            inputs = image[:, indices]
            # 输入是文件存储值，不能假定已做辐射校正；复现各成员训练时的滤波和标准化。
            filter_op = member['speckle_filter']
            if filter_op.config is not None:
                raw, masks = inputs.cpu().numpy(), valid_mask.cpu().numpy()
                filtered = np.stack([filter_op(sample, mask) for sample, mask in zip(raw, masks)])
                inputs = torch.from_numpy(filtered).to(self.device)
            inputs = (inputs - member["mean"]) / member["std"]
            inputs = inputs.masked_fill(~valid_mask.unsqueeze(1), 0)
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = member["model"](inputs)
                logits = outputs[-1] if isinstance(outputs, (tuple, list)) else outputs
            # 单通道用float64算概率并保留精度：微小负logit的float32 sigmoid
            # 可能舍入成0.5，导致quality比训练的logit>=0多判水体。网络本身仍沿用AMP。
            probability = (logits[:, 0].double().sigmoid() if logits.shape[1] == 1
                           else logits.float().softmax(dim=1)[:, 1])
            probabilities.append(probability)
        return probabilities, torch.stack(probabilities).mean(dim=0)


def terrain_gradient(dem, pixel_size=None, height_scale=None):
    """尺度已知时返回角度坡度；否则返回原始高程值/像素的梯度幅值，不能当作坡度角。"""
    # NaN 邻域自然传播，不跨越缺测孔洞求导；切片外没有数据，边缘使用单边差分。
    if pixel_size is not None and height_scale is not None:
        dy, dx = np.gradient(dem * height_scale, *pixel_size)
        slope = np.degrees(np.arctan(np.hypot(dx, dy)))
        slope[~np.isfinite(dem)] = np.nan
        return slope, "Slope (degrees)"
    dy, dx = np.gradient(dem)
    gradient = np.hypot(dx, dy)
    gradient[~np.isfinite(dem)] = np.nan
    return gradient, "Elevation gradient (raw units/pixel)"


def save_result_figure(image_path, label, valid, probability, corrected, threshold, output_path, row):
    """输出逐图质量对照；精度直接取同一条 CSV 记录，无标签时不报告精度。"""
    with rasterio.open(image_path) as source:
        layers = source.read(out_dtype="float32", masked=True).filled(np.nan)
    sar = layers[:2]
    prediction = probability >= threshold
    errors = np.full((*probability.shape, 3), 45, dtype=np.uint8)
    # 无标签时不生成 TP/FP/FN，只显示新增水体的位置。
    if label is not None:
        errors[valid & (label == 1) & prediction] = (45, 120, 255)  # TP 蓝色。
        errors[valid & (label == 0) & prediction] = (255, 65, 65)   # FP 红色。
        errors[valid & (label == 1) & ~prediction] = (255, 220, 40)  # FN 黄色。
    errors[~valid] = 0
    changes = np.full_like(errors, 45)
    added = corrected & ~prediction & valid
    if label is not None:
        changes[added & (label == 1)] = (0, 210, 120)
        changes[added & (label == 0)] = (230, 50, 220)
    else:
        changes[added] = (0, 210, 120)
    changes[~valid] = 0
    fig, axes = plt.subplots(4, 4, figsize=(20, 22))
    for i in range(len(sar)):
        display_valid = valid & np.isfinite(sar[i])
        values = sar[i][display_valid]
        low, high = np.percentile(values, (2, 98)) if values.size else (0., 1.)
        # 只作显示拉伸，不改变模型输入，也不擅自转 dB。
        stretched = np.clip((sar[i] - low) / max(float(high - low), 1e-6), 0, 1)
        axes[0, i].imshow(np.where(display_valid, stretched, np.nan), cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(f"SAR band {i + 1} (P2-P98 display)")
    if len(sar) < 2:
        axes[0, 1].text(.5, .5, "Second SAR band unavailable", ha="center", va="center")
    gray = plt.get_cmap("gray").copy()
    gray.set_bad("black")
    for axis, mask, title in ((axes[0, 2], label, "Ground truth"),
                              (axes[2, 2], prediction, "Raw prediction"),
                              (axes[3, 0], corrected, "Postprocessed prediction")):
        if mask is None:
            axis.text(.5, .5, "No ground truth", ha="center", va="center", transform=axis.transAxes)
        else:
            axis.imshow(np.where(valid, mask, np.nan), cmap=gray, vmin=0, vmax=1)
        # 预测面板同时给出混淆计数，便于判断改善来自减少漏检还是增加误检。
        if label is not None and title in ("Raw prediction", "Postprocessed prediction"):
            prefix = "raw" if title == "Raw prediction" else "post"
            title += (f"\nTN={row[prefix + '_tn']:,}  FP={row[prefix + '_fp']:,}"
                      f"\nFN={row[prefix + '_fn']:,}  TP={row[prefix + '_tp']:,}")
        axis.set_title(title, fontsize=10)
    image = axes[0, 3].imshow(np.where(valid, probability, np.nan), cmap="viridis", vmin=0, vmax=1)
    axes[0, 3].set_title("Water probability\n(not calibrated uncertainty)")
    fig.colorbar(image, ax=axes[0, 3], fraction=0.045)
    if label is not None:
        axes[2, 3].imshow(errors)
        axes[2, 3].set_title("Raw errors\nblue TP / red FP / yellow FN")
    else:
        axes[2, 3].text(.5, .5, "Errors unavailable without labels", ha="center", va="center",
                        transform=axes[2, 3].transAxes)
    axes[3, 1].imshow(changes)
    axes[3, 1].set_title("Added water\ngreen correct / magenta wrong" if label is not None
                         else "Added water (unverified)")
    # 辅助层只作并排显示，不输入网络，也不据此删除或增加预测水体。
    for i, name in enumerate(("MERIT DEM / band 3", "Copernicus DEM / band 4")):
        dem = layers[i + 2].copy() if len(layers) > i + 2 else np.full(valid.shape, np.nan)
        dem[(dem == DEM_NODATA) | ~np.isfinite(dem)] = np.nan
        gradient, gradient_title = terrain_gradient(dem, DEM_PIXEL_SIZE_METERS, DEM_HEIGHT_SCALE)
        for axis, values, title, cmap in (
                (axes[1, i], dem, name + "\nraw elevation values", "terrain"),
                (axes[1, i + 2], gradient, name.split(" / ")[0] + "\n" + gradient_title, "magma")):
            axis.set_title(title)
            finite = values[np.isfinite(values)]
            if finite.size:
                # DEM 不做逐图百分位拉伸；色标给出实际存储值，梯度不混称阴影/叠掩。
                plot = axis.imshow(np.ma.masked_invalid(values), cmap=cmap)
                fig.colorbar(plot, ax=axis, fraction=.045)
            else:
                axis.text(.5, .5, "DEM / neighborhood unavailable", ha="center", va="center",
                          transform=axis.transAxes)
    codes = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100])
    cover_colors = ["#006400", "#ffbb22", "#ffff4c", "#f096ff", "#fa0000", "#b4b4b4",
                    "#f0f0f0", "#0064c8", "#0096a0", "#00cf75", "#fae6a0"]
    cover = layers[4] if len(layers) > 4 else np.full(valid.shape, np.nan)
    cover = np.ma.masked_where(~np.isin(cover, codes), cover)
    image = axes[2, 0].imshow(cover, cmap=ListedColormap(cover_colors),
                              norm=BoundaryNorm([5, 15, 25, 35, 45, 55, 65, 75, 85, 92.5, 97.5, 105], 11))
    fig.colorbar(image, ax=axes[2, 0], ticks=codes, fraction=.045)
    axes[2, 0].set_title("WorldCover / band 5\ncategory codes; water = 80")
    occurrence = layers[5] if len(layers) > 5 else np.full(valid.shape, np.nan)
    axes[2, 1].set_title("Historical water occurrence / band 6\nraw codes; scale unverified")
    if np.isfinite(occurrence).any():
        image = axes[2, 1].imshow(np.ma.masked_invalid(occurrence), cmap="Blues")
        fig.colorbar(image, ax=axes[2, 1], fraction=.045)
    else:
        axes[2, 1].text(.5, .5, "Band unavailable", ha="center", va="center", transform=axes[2, 1].transAxes)
    axes[3, 2].text(0, .95,
                    "Terrain interpretation\n\nCompare red FP with DEM gradients.\n"
                    "Gradients alone do not identify radar shadow.\n"
                    "Slope angles require confirmed XY/Z units.\n"
                    "No layover / shadow mask is computed.\n"
                    "DEM NoData is blank; zero height is retained.",
                    va="top", fontsize=10, transform=axes[3, 2].transAxes)
    # 水体四指标在图顶显示；这里补充背景表现、误检/漏检率及整体一致性。
    # 全部由同一条混淆计数计算，不把岸线质量与像元精度混为一谈。
    quality_rows = []
    for key, caption in (("accuracy", "OA"), ("background_iou", "Background IoU"),
                         ("mean_iou", "Mean IoU (2 classes)"), ("specificity", "Specificity"),
                         ("false_positive_rate", "FPR = FP / (FP+TN)"),
                         ("false_negative_rate", "FNR = FN / (FN+TP)"),
                         ("false_discovery_rate", "FDR = FP / (FP+TP)"),
                         ("balanced_accuracy", "Balanced accuracy"),
                         ("kappa", "Kappa"), ("mcc", "MCC")):
        values = []
        for prefix in ("raw", "post"):
            value = row.get(f"{prefix}_{key}", np.nan) if label is not None else np.nan
            values.append((f"{value:.3f}" if key in ("kappa", "mcc") else f"{value:.2%}")
                          if np.isfinite(value) else "N/A")
        quality_rows.append([caption, *values])
    axes[3, 3].set_title("Additional pixel metrics (same valid area)", fontsize=10)
    table = axes[3, 3].table(cellText=quality_rows, colLabels=["Metric", "Raw", "Post"],
                            colWidths=[.58, .21, .21], bbox=[0, .10, 1, .84], cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    axes[3, 3].text(0, .02, "N/A: undefined or no labels. OA may hide water errors.",
                    fontsize=8, transform=axes[3, 3].transAxes)
    for axis in axes.flat:
        axis.axis("off")
    title = f"{row['name']} | threshold={threshold:g} | per-image water metrics"
    if label is not None:
        # 图与 per_image.csv 共用已经统计的数值；不能把单幅精度当成整套验证精度。
        title += f" | valid pixels={row['raw_valid_pixels']:,}"
        for prefix, name in (("raw", "Raw"), ("post", "Postprocessed")):
            scores = []
            for key, caption in (("iou", "IoU"), ("precision", "Precision"),
                                 ("recall", "Recall"), ("f1", "F1")):
                value = row[f"{prefix}_{key}"]
                # 无有效标签或分母无定义时显示 N/A；其他值以百分比显示。
                display = (f"{value:.2%}" if row[f"{prefix}_valid_pixels"] and np.isfinite(value)
                           else "N/A")
                scores.append(f"{caption}={display}")
            title += f"\n{name}: " + " | ".join(scores)
    else:
        title += "\nUnlabeled inference | IoU / Precision / Recall / F1: N/A (no ground truth)"
    fig.suptitle(title, fontsize=13, y=.99)
    colors = ([((45, 120, 255), "TP"), ((255, 65, 65), "FP"), ((255, 220, 40), "FN"),
               ((45, 45, 45), "TN / unchanged"), ((0, 0, 0), "Ignored"),
               ((0, 210, 120), "Added TP"), ((230, 50, 220), "Added FP")] if label is not None
              else [((0, 210, 120), "Added water (unverified)"), ((0, 0, 0), "Ignored")])
    fig.legend(handles=[Patch(color=np.array(color) / 255, label=text) for color, text in colors],
               loc="lower center", ncol=len(colors), fontsize=9)
    fig.subplots_adjust(left=.025, right=.98, bottom=.045, top=.925, wspace=.26, hspace=.36)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def run_postprocess(checkpoint_paths, image_dir=None, label_dir=None,
                    use_saved_val=True, with_labels=True,
                    batch_size=2, use_amp=True, threshold=0.5,
                    use_landcover=True, landcover_band=5,
                    water_class=80, save_masks=True, output_dir=None,
                    save_visuals=True, max_visualizations=None,
                    evaluate_boundaries=True, auxiliary_options=None,
                    auxiliary_cache_dir=None, analysis_options=None,
                    scene_analysis_options=None, use_external_water=False):
    """准备数据并推理后执行编号后处理步骤；标签仅用于步骤4的独立评价。

    默认不下载、不补入外部历史水体。面积自动统计；配置参考/高程后分析变化/水深。
    历史补水会压制退水，故始终分别分析 raw 与 postprocessed。
    """
    # 数据准备：恢复检查点输入契约、锁定样本清单，再下载/缓存外部辅助数据。
    ensemble = WaterEnsemble(checkpoint_paths, use_amp)
    first = ensemble.members[0]
    if use_saved_val and any(m["val_files"] != first["val_files"] for m in ensemble.members):
        raise ValueError("成员的验证划分不同，请用所有成员都未参与训练的独立有标签集合评价")
    image_dir = Path(first["image_dir"] if image_dir is None else image_dir)
    label_dir = Path(first["label_dir"] if label_dir is None else label_dir) if with_labels else None
    dataset = WaterDataset(image_dir, label_dir, ensemble.bands)
    if use_saved_val:
        dataset.select_files(first["val_files"])
    selected = {sample_name(path, image_dir): path for path in dataset.image_paths}
    if not selected:
        raise ValueError("后处理没有选中的影像")
    download_options = dict(auxiliary_options or {})
    if scene_analysis_options is not None:
        if download_options:
            raise ValueError("scene_analysis_options 已提供逐景辅助数据，请勿同时配置 auxiliary_options")
        scene_options = {name: {**(analysis_options or {}), **scene_analysis_options[name]} for name in selected}
    elif download_options or analysis_options:
        cache = (Path(auxiliary_cache_dir) if auxiliary_cache_dir else
                 Path(__file__).resolve().parents[3] / "data_cache/auxiliary")
        scene_options = prepare_auxiliary_inputs(selected, cache, analysis_options, **download_options)
    else:
        scene_options = {name: {} for name in selected}
    if use_external_water and any(not options.get("reference_path") for options in scene_options.values()):
        raise ValueError("use_external_water 需要 reference_path 或 download_history=True")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    output_dir = (first["path"].parent / "quality" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                  if output_dir is None else Path(output_dir))
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "postprocess.json").write_text(json.dumps(dict(
        stages=["1 prepare", "2 predict", "3 auxiliary correction and analysis", "4 evaluate and export"],
        checkpoints=[str(path) for path in checkpoint_paths], threshold=threshold,
        use_landcover=use_landcover, landcover_band=landcover_band, water_class=water_class,
        use_external_water=use_external_water, scene_analysis_options=scene_options,
        note="Historical water addition can suppress retreat; compare raw and postprocessed analyses."
    ), indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    boundary = BoundaryEvaluator(output_dir / "boundary") if with_labels and evaluate_boundaries else None
    rows, visualized = [], 0
    # 每个成员单独统计，再比较集成与后处理，不能预设集成一定更好。
    names = [f"member_{i + 1}" for i in range(len(ensemble.members))] + ["ensemble", "postprocessed"]
    confusions = {name: torch.zeros(4, dtype=torch.int64, device=ensemble.device) for name in names}
    changed_pixels = total_valid = 0
    observed_codes = set()
    print(f"样本数：{len(dataset)}；成员数：{len(ensemble.members)}；阈值：{threshold}；土地覆盖修正：{use_landcover}", flush=True)

    for step, batch in enumerate(loader, 1):
        # 模型推理：各成员独立标准化/推理，平均概率后阈值化，得到后处理输入raw。
        image_valid = batch["valid_mask"].to(ensemble.device)
        probabilities, average = ensemble.predict(batch["image"], image_valid)
        raw = (average >= threshold) & image_valid
        corrected = raw.clone()
        if with_labels:
            label = batch["label"].to(ensemble.device)
            valid = label != first["ignore_index"]
        for i, name in enumerate(batch["name"]):
            image_path = resolve_sample_path(image_dir, name)
            options = scene_options[name]
            # ------ 后处理1：内置/外部历史水体显式补入。------
            # 内置WorldCover按water_class判水，外部参考按声明编码/覆盖面判水；
            # 只在SAR有效区补入已知历史水体，未知区不参与；raw保留用于检查真实退水。
            if use_landcover:
                cover, land_valid = read_landcover(image_path, landcover_band)
                observed_codes.update(np.unique(cover[land_valid]).tolist())
                corrected[i] = merge_landcover(
                    raw[i], torch.as_tensor(cover, device=ensemble.device),
                    torch.as_tensor(land_valid, device=ensemble.device), image_valid[i], water_class)
            if use_external_water:
                reference_keys = ("reference_path", "reference_water_values", "reference_dry_values",
                                  "reference_coverage_path", "vector_assume_complete")
                history, known = read_reference_water(image_path, **{k: options[k] for k in reference_keys if k in options})
                corrected[i] |= torch.as_tensor(history & known, device=ensemble.device) & image_valid[i]
            row = dict(name=name, auxiliary_added_pixels=int((corrected[i] & ~raw[i]).sum().item()))
            # ------ 后处理2：水体面积与历史新增/消退范围、面积统计。------
            # raw/post分别交给analyze_water；比较仅使用共同有效覆盖，参考缺失时只统计面积。
            # 不保存掩膜时使用临时栅格；不覆盖模型原始结果。
            with TemporaryDirectory(prefix="water_analysis_", dir=output_dir) as scratch:
                with rasterio.open(image_path) as source:
                    profile = source.profile.copy()
                # 继承源栅格尺寸/定位；源文件没有地理参考时，输出也不能用于地理矢量定位。
                profile.update(count=1, dtype="uint8", nodata=255, compress="lzw")
                for kind, mask in (("raw", raw[i]), ("postprocessed", corrected[i])):
                    path = (output_dir / kind / Path(name).with_suffix(".tif") if save_masks
                            else Path(scratch) / (kind + ".tif"))
                    path.parent.mkdir(parents=True, exist_ok=True)
                    array = mask.cpu().numpy().astype(np.uint8)
                    array[~batch["valid_mask"][i].numpy()] = 255
                    with rasterio.open(path, "w", **profile) as target:
                        target.write(array, 1)
                    # ------ 后处理3：DEM/DSM用于几何水深估计（由analyze_water按配置执行）。------
                    # 步骤2/3共用该调用；高程缺失则跳过水深。坡度/坡向及地形排除目前仅在
                    # thres_segmentation/gis_worker.py中实现，本入口尚未执行这些掩膜步骤。
                    analysis = analyze_water(path, output_dir / "analysis" / kind / Path(name).with_suffix(""), **options)
                    prefix = "raw" if kind == "raw" else "post"
                    row[prefix + "_water_area_m2"] = analysis["current_water"]["m2"] if analysis["current_water"] else None
                    row[prefix + "_area_status"] = analysis["status"]
            # ------ 后处理4：评价与输出。------
            # 相同有效标签域比较raw/post，按混淆计数汇总；无标签不报告精度。
            for prefix, prediction in (("raw", raw[i]), ("post", corrected[i])):
                if with_labels:
                    counts = torch.bincount(label[i][valid[i]] * 2 + prediction[valid[i]].long(), minlength=4)
                    row.update({f"{prefix}_{k}": v for k, v in metrics_from_counts(counts.tolist()).items()})
                else:
                    # 无标签只记录预测占比，不输出虚构的 F1/FP/FN。
                    count = int(image_valid[i].sum().item())
                    row[f"{prefix}_pred_water_ratio"] = float(prediction.sum().item()) / count if count else float("nan")
            if with_labels:
                row["delta_f1"] = row["post_f1"] - row["raw_f1"]
            rows.append(row)
            if boundary is not None:
                boundary_predictions = {f"member_{j + 1}": (p[i] >= threshold).cpu().numpy()
                                        for j, p in enumerate(probabilities)}
                boundary_predictions.update(raw=raw[i].cpu().numpy(), post=corrected[i].cpu().numpy())
                boundary_probabilities = {f"member_{j + 1}": p[i].cpu().numpy()
                                          for j, p in enumerate(probabilities)}
                boundary_probabilities.update(raw=average[i].cpu().numpy())
                # post 是确定性土地覆盖修正，无独立概率，避免误把原网络概率当作修正后概率。
                regions, point_sets = boundary.update(
                    name, batch["label"][i].numpy(), valid[i].cpu().numpy(),
                    boundary_predictions, boundary_probabilities,
                    {method: threshold for method in boundary_predictions})
            if save_visuals and (max_visualizations is None or visualized < max_visualizations):
                valid_array = valid[i].cpu().numpy() if with_labels else image_valid[i].cpu().numpy()
                save_result_figure(
                    resolve_sample_path(image_dir, name), batch["label"][i].numpy() if with_labels else None,
                    valid_array, average[i].cpu().numpy(), corrected[i].cpu().numpy(), threshold,
                    output_dir / "figures" / Path(name).with_suffix(".png"), row)
                if boundary is not None:
                    save_boundary_figure(batch["label"][i].numpy(), valid_array, regions, point_sets,
                                         output_dir / "boundary/figures" / Path(name).with_suffix(".png"))
                visualized += 1
        changed_pixels += int((raw != corrected).sum().item())
        total_valid += int(image_valid.sum().item())
        if with_labels:
            predictions = [p >= threshold for p in probabilities] + [raw, corrected]
            for name, prediction in zip(names, predictions):
                confusions[name] += torch.bincount(label[valid] * 2 + prediction[valid].long(), minlength=4)
        if step % 50 == 0:
            print(f"处理：{step}/{len(loader)} 批", flush=True)

    print(f"土地覆盖观测编码：{sorted(observed_codes)}；修正像素：{changed_pixels}/{total_valid}", flush=True)
    metrics = {}
    if with_labels:
        for name, counts in confusions.items():
            tn, fp, fn, tp = counts.tolist()
            metrics[name] = metrics_from_counts((tn, fp, fn, tp))
            m = metrics[name]
            print(f"{name}：F1={m['f1']:.6f}，IoU={m['iou']:.6f}，P={m['precision']:.6f}，R={m['recall']:.6f}；"
                  f"TN/FP/FN/TP={tn}/{fp}/{fn}/{tp}", flush=True)
    # 精度与可视化随本次后处理一起保存，不再另建 bad case 分析流程或自动排名。
    if metrics:
        with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=["result", *next(iter(metrics.values())).keys()])
            writer.writeheader()
            writer.writerows(dict(result=name, **values) for name, values in metrics.items())
    if rows:
        with (output_dir / "per_image.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    if boundary is not None:
        boundary.save()
    print(f"结果保存到：{output_dir}；对照图：{visualized} 幅", flush=True)
    if save_masks:
        print("掩膜：raw/postprocessed/；0=非水体，1=水体，255=无效", flush=True)
    return metrics

