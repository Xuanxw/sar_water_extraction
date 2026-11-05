"""1 数据准备 → 2 阈值预测 → 3 辅助修正/水体分析 → 4 独立评价与输出。"""
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio

from supervised.evaluation.boundary_eval import BoundaryEvaluator, confusion_counts, save_boundary_figure
from supervised.evaluation.metrics import metrics_from_counts
from supervised.evaluation.landcover import read_landcover, merge_landcover
from thres_segmentation.threshold import ThresholdSegmenter

EXTENSIONS = {".tif", ".tiff", ".img", ".vrt", ".png"}


def write_csv(path, rows):
    """UTF-8 BOM 方便 Windows Excel 查看中文。"""
    if rows:
        with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def save_figure(image, band_names, label, valid, predictions, rows, output_path, image_valid=None):
    """输出输入波段、分割与可选误差；无标签时不绘制虚构的 TP/FP/FN。"""
    image_valid = valid if image_valid is None else image_valid
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    figure = Figure(figsize=(14, 3 * (len(predictions) + 1)))
    FigureCanvasAgg(figure)
    axes = figure.subplots(len(predictions) + 1, 3, squeeze=False)
    for index, name in enumerate(list(band_names)[:2]):
        values = image[index][image_valid]
        low, high = np.percentile(values, (2, 98)) if values.size else (0, 1)
        display = np.clip((image[index] - low) / max(high - low, 1e-6), 0, 1)
        axes[0, index].imshow(np.where(image_valid, display, np.nan), cmap="gray", vmin=0, vmax=1)
        axes[0, index].set_title(name + " (display P2-P98)")
    if label is not None:
        axes[0, 2].imshow(np.where(valid, label, np.nan), vmin=0, vmax=1, cmap="gray")
    else:
        axes[0, 2].text(.1, .5, "No labels: quality metrics unavailable")
    axes[0, 2].set_title("Ground truth")
    for index, (method, prediction) in enumerate(predictions.items(), 1):
        axes[index, 0].imshow(np.where(image_valid, prediction, np.nan), vmin=0, vmax=1, cmap="gray")
        axes[index, 0].set_title(method)
        row = rows[method]
        if label is not None:
            error = np.full((*valid.shape, 3), 45, dtype=np.uint8)
            error[valid & (label == 1) & prediction] = (45, 120, 255)
            error[valid & (label == 0) & prediction] = (255, 65, 65)
            error[valid & (label == 1) & ~prediction] = (255, 220, 40)
            error[~valid] = 0
            axes[index, 1].imshow(error)
            axes[index, 1].set_title("Blue TP / red FP / yellow FN")
            text = "\n".join(f"{key}: {row[key]:.4f}" for key in ("f1", "iou", "precision", "recall"))
            text += "\n" + "\n".join(f"{key.upper()}: {row[key]:,}" for key in ("tn", "fp", "fn", "tp"))
        else:
            text = f"Predicted water ratio: {row['pred_water_ratio']:.2%}\nNo accuracy evaluation without labels"
        axes[index, 2].text(.05, .95, text, va="top", transform=axes[index, 2].transAxes)
    for axis in axes.flat:
        axis.axis("off")
    figure.suptitle(Path(output_path).stem)
    figure.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=110, bbox_inches="tight")


def run_segmentation(image_dir, label_dir=None, bands=None, output_dir=None,
                     threshold_options=None, use_landcover=True, landcover_band=5,
                     save_masks=True, max_visualizations=10, split_manifest=None, analysis_options=None,
                     scene_analysis_options=None):
    """按固定 val 清单或外部目录独立分割；标签仅评价，不自动选参。"""
    # 1. 数据准备：锁定影像/标签清单；外部辅助数据由main下载并通过逐景路径传入。
    image_dir = Path(image_dir)
    bands = {"VV": 1, "VH": 2} if bands is None else bands
    image_paths = sorted(p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if split_manifest is not None:
        # 不加载 checkpoint、不导入 torch；与训练共享的是数据划分清单。
        split_manifest = Path(split_manifest)
        if image_dir.resolve() != (split_manifest.parent / "images/val").resolve():
            raise ValueError("固定划分的阈值评价必须使用 images/val 目录")
        with split_manifest.open(encoding="utf-8-sig", newline="") as stream:
            val_rows = [r for r in csv.DictReader(stream) if r["split"] == "val" and int(r["active"]) == 1]
        image_paths = [image_dir / r["name"] for r in val_rows]
        if any(not p.is_file() for p in image_paths):
            raise FileNotFoundError("固定 val 清单中的影像缺失")
    if not image_paths:
        raise FileNotFoundError(f"影像目录中没有可读取的栅格：{image_dir}")
    label_paths = None
    if label_dir is not None:
        label_dir = Path(label_dir)
        label_paths = {p.relative_to(label_dir).with_suffix("").as_posix().casefold(): p
                       for p in label_dir.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS}
    output_dir = (Path(output_dir) if output_dir is not None else
                  Path(__file__).resolve().parents[2] / "outputs/thres_seg/quality" / datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output_dir.mkdir(parents=True, exist_ok=False)
    segmenter = ThresholdSegmenter(**(threshold_options or {}))
    methods = [dict(method=f"threshold_{name}", band=band, parameters=str(vars(segmenter)),
                    use_landcover=use_landcover, landcover_band=landcover_band)
               for name, band in bands.items()]
    write_csv(output_dir / "methods.csv", methods)
    write_csv(output_dir / "files.csv", [dict(name=p.relative_to(image_dir).as_posix()) for p in image_paths])
    boundary = BoundaryEvaluator(output_dir / "boundary") if label_paths is not None else None
    counts, rows = {}, []
    print(f"独立阈值分割：{len(image_paths)} 幅；波段：{bands}；标签评价：{label_paths is not None}", flush=True)
    try:
        for index, image_path in enumerate(image_paths):
            name = image_path.relative_to(image_dir).as_posix()
            scene_options = scene_analysis_options[name] if scene_analysis_options is not None else (analysis_options or {})
            with rasterio.open(image_path) as source:
                data = source.read(list(bands.values()), out_dtype="float32", masked=True)
                profile = source.profile.copy()
            # 与训练数据读取保持同样的文件有效性语义，不把合法零值当 NoData。
            image_valid = np.isfinite(data.data).all(axis=0) & ~np.ma.getmaskarray(data).any(axis=0)
            image = data.filled(0)
            image[:, ~image_valid] = 0
            label, valid = None, image_valid
            if label_paths is not None:
                key = Path(name).with_suffix("").as_posix().casefold()
                with rasterio.open(label_paths[key]) as source:
                    truth = source.read(1, masked=True)
                if truth.shape != image_valid.shape:
                    raise ValueError(f"{name}：标签与影像尺寸不一致")
                valid = image_valid & ~np.ma.getmaskarray(truth) & np.isin(truth.data, (0, 1))
                label = np.where(valid, truth.data, -1)
            predictions, thresholds = {}, {}
            # 2. 基础预测：Lee滤波、可选功率转dB、百分位归一化与阈值；不是辐射校正。
            for band_index, band_name in enumerate(bands):
                method = f"threshold_{band_name}"
                predictions[method], _, thresholds[method] = segmenter.predict(image[band_index], image_valid)
            # 3.1 辅助修正：保留原始预测，内置土地覆盖补水作为独立_esa方法。
            if use_landcover:
                cover, cover_valid = read_landcover(image_path, landcover_band)
                for method, prediction in list(predictions.items()):
                    predictions[method + "_esa"] = merge_landcover(prediction, cover, cover_valid, image_valid)
                    thresholds[method + "_esa"] = thresholds[method]
            # 3.2 水体分析：每种方法独立统计面积/变化/水深；外部参考默认只对比。
            # 不保存掩膜时使用临时栅格，避免分析功能受SAVE_MASKS影响。
            from tempfile import TemporaryDirectory
            from thres_segmentation.water_analysis import analyze_water
            analyses = {}
            with TemporaryDirectory(prefix='water_analysis_', dir=output_dir) as scratch:
                profile.update(count=1, dtype="uint8", nodata=255, compress="lzw")
                for method, prediction in predictions.items():
                    path = (output_dir / "masks" / method / Path(name).with_suffix(".tif")
                            if save_masks else Path(scratch) / (method + '.tif'))
                    path.parent.mkdir(parents=True, exist_ok=True)
                    mask = np.where(image_valid, prediction, 255).astype(np.uint8)
                    with rasterio.open(path, "w", **profile) as target:
                        target.write(mask, 1)
                    analyses[method] = analyze_water(path, output_dir/'analysis'/method/Path(name).with_suffix(''),
                                                     **scene_options)
            # 4. 评价与输出：标签只参与计分；各方法共用有效域，汇总混淆计数。
            image_rows = {}
            for method, prediction in predictions.items():
                row = dict(name=name, method=method, decision_threshold=thresholds[method])
                if label is not None:
                    confusion = confusion_counts(label, prediction, valid)
                    counts[method] = counts.get(method, np.zeros(4, dtype=np.int64)) + confusion
                    row.update(metrics_from_counts(confusion))
                else:
                    count = int(image_valid.sum())
                    row.update(image_valid_pixels=count,
                               pred_water_ratio=float(prediction.sum() / count) if count else float("nan"))
                analysis = analyses[method]
                row['water_area_m2'] = analysis['current_water']['m2'] if analysis['current_water'] else None
                row['area_status'] = analysis['status']
                rows.append(row)
                image_rows[method] = row
            if boundary is not None:
                regions, points = boundary.update(name, label, valid, predictions, thresholds=thresholds)
            if max_visualizations is None or index < max_visualizations:
                save_figure(image, bands, label, valid, predictions, image_rows,
                            output_dir / "figures" / Path(name).with_suffix(".png"))
                if boundary is not None:
                    save_boundary_figure(label, valid, regions, points,
                                         output_dir / "boundary/figures" / Path(name).with_suffix(".png"))
            if (index + 1) % 25 == 0:
                print(f"处理：{index + 1}/{len(image_paths)}", flush=True)
        if boundary is not None:
            boundary.save()
    finally:
        if boundary is not None:
            boundary.close()
    metrics = {method: metrics_from_counts(confusion) for method, confusion in counts.items()}
    write_csv(output_dir / "metrics.csv", [dict(method=method, **values) for method, values in metrics.items()])
    write_csv(output_dir / "per_image.csv", rows)
    for method, values in metrics.items():
        print(f"{method}：F1={values['f1']:.4f}，P={values['precision']:.4f}，R={values['recall']:.4f}", flush=True)
    print(f"结果：{output_dir}", flush=True)
    return metrics
