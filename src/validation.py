"""固定模型后的质量验证入口：python src/validation.py；不训练、不扫描阈值。"""
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import csv
import gc
import hashlib

import numpy as np
import rasterio
import torch
from termcolor import cprint

from datasets.water_dataset import WaterDataset, resolve_sample_path, sample_name
from evaluation.metrics import metrics_from_counts
from evaluation.postprocess import run_postprocess

ROOT = Path(__file__).resolve().parents[1]
# 当前收束基线：固定 MBDC + 全局困难采样 + 最终头残差 + 四头等权 BCE。
CHECKPOINTS = {
    "Residual": ROOT / "outputs/pvt_uafnet_hard_sampling_residual/20261022_160255_467390/best_f1.pth",
}
IMAGE_DIR = None  # None 沿用检查点目录；新数据必须保持相同极化、数值单位和预处理。
LABEL_DIR = None  # 精度验证必须有独立参考标签，不能使用 WorldCover 或预测代替。
USE_SAVED_VAL = True  # 当前为历史内部验证；外部有标签集合设 False，并填写以上两个目录。
THRESHOLD = 0.5       # 固定工作点；不根据本次测试结果自动选取最佳阈值。
BATCH_SIZE = 2
USE_AMP = True
SAVE_MASKS = True
MAX_VISUALIZATIONS = 10  # None 输出全部质量图；不输出中间特征。
OUTPUT_DIR = None  # 默认 outputs/validation/quality/时间戳，各模型分目录。
SCENE_CSV = ROOT / "docs/s1_samples/scene_catalog_20261026_163548/sample_scenes.csv"
REGION_CSV = None  # 可选 CSV：name,region_id；name 为相对影像目录的文件名，须覆盖全部待评估图。
REGIONS_ARE_INDEPENDENT = False  # 只有核实区域互相独立、且未参与训练/调参后才设 True。
BOOTSTRAP_REPEATS = 2000
SEED = 42
COUNT_KEYS = ("tn", "fp", "fn", "tp")
METRIC_KEYS = ("iou", "f1", "precision", "recall")


def read_csv(path):
    """读取带中文列内容的 CSV，兼容 Excel 的 UTF-8 BOM。"""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    """写出非空记录表；各行使用相同字段，不为缺失数据伪造指标。"""
    if rows:
        with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def image_fingerprint(path, bands):
    """对实际输入波段及有效掩膜取哈希，检查整幅重复；不代表地理区域独立。"""
    with rasterio.open(path) as source:
        array = source.read(indexes=bands)
        masks = source.read_masks(indexes=bands)
    digest = hashlib.sha256(str((array.shape, array.dtype.str)).encode())
    digest.update(array.tobytes())
    digest.update(masks.tobytes())
    return digest.hexdigest()


def build_manifest(checkpoint, image_dir, label_dir, use_saved_val, scene_csv=None):
    """重新核查训练/验证整幅重复；保留全部原验证图，并标出可用于敏感性分析的唯一图。"""
    image_dir = Path(image_dir)
    dataset = WaterDataset(image_dir, label_dir, checkpoint["bands"])
    if use_saved_val:
        dataset.select_files(checkpoint["val_files"])
    if not len(dataset):
        raise ValueError("验证目录为空")
    # 核查基于实际训练文件，而不是旧审查名单；外部测试也检查与源训练影像的精确重复。
    train_dir = image_dir if use_saved_val else Path(checkpoint["image_dir"])
    # 允许显式指定 images/val；训练指纹仍从同级 images 根目录查找。
    if (use_saved_val and image_dir.name in ("train", "val")
            and (image_dir.parent.parent / "split_manifest.csv").is_file()):
        train_dir = image_dir.parent
    train_hashes = set()
    for index, name in enumerate(checkpoint["train_files"], 1):
        train_hashes.add(image_fingerprint(resolve_sample_path(train_dir, name), checkpoint["bands"]))
        if index % 300 == 0:
            cprint(f"重复核查：训练 {index}/{len(checkpoint['train_files'])}", "cyan", flush=True)
    scenes = {row["name"]: row for row in read_csv(scene_csv)} if scene_csv else {}
    records, groups = [], defaultdict(list)
    for path in dataset.image_paths:
        name = sample_name(path, image_dir)
        digest = image_fingerprint(path, checkpoint["bands"])
        label_path = dataset.label_paths[Path(name).with_suffix("").as_posix().casefold()]
        label_hash = hashlib.sha256(label_path.read_bytes()).hexdigest()
        scene = scenes.get(name, {})
        # 只有标签指纹吻合才沿用已有自动场景分类，防止同名外部切片误套旧分类。
        scene_matches = scene.get("label_sha256") == label_hash
        record = dict(name=name, input_sha256=digest, label_sha256=label_hash,
                      label_content_sha256=image_fingerprint(label_path, (1,)),
                      duplicate_in_train=int(digest in train_hashes),
                      validation_copies=0, conflicting_duplicate_labels=0, unique_nontrain=0,
                      scene=scene.get("water_context_scene", "unknown") if scene_matches else "unknown")
        records.append(record)
        groups[digest].append(record)
    for group in groups.values():
        conflict = len({r["label_content_sha256"] for r in group}) > 1
        for row in group:
            row["validation_copies"] = len(group)
            row["conflicting_duplicate_labels"] = int(conflict)
        # 同影像同标签只计一次；比较解码后的标签，避免文件压缩方式造成假冲突。
        # 异标签组全部保留在总表，但不任意挑一份当正确标签。
        if not group[0]["duplicate_in_train"] and not conflict:
            group[0]["unique_nontrain"] = 1
    return records


def summarize(rows):
    """先累加混淆计数求整体指标，再计算逐图/逐区域等权均值；二者含义不同。"""
    counts = np.sum([[int(r[k]) for k in COUNT_KEYS] for r in rows], axis=0) if rows else [0] * 4
    result = dict(samples=len(rows), **metrics_from_counts(counts))
    for key in METRIC_KEYS:
        defined = [r for r in rows if np.isfinite(float(r[key]))]
        values = [float(r[key]) for r in defined]
        result.update({f"macro_{key}": float(np.mean(values)) if values else float("nan"),
                       f"std_{key}": float(np.std(values)) if values else float("nan"),
                       f"defined_{key}": len(values),
                       f"worst_{key}": min(values) if values else float("nan"),
                       f"worst_{key}_name": min(defined, key=lambda r: float(r[key]))["name"] if values else ""})
    return result


def save_reports(output_dir, rows, manifest, region_csv=None, independent_regions=False,
                 bootstrap_repeats=BOOTSTRAP_REPEATS):
    """报告重复分组、自动场景与可选真实区域；没有区域映射就不对切片做置信区间。"""
    output_dir = Path(output_dir)
    metadata = {r["name"]: r for r in manifest}
    if {r["name"] for r in rows} != set(metadata) or len(rows) != len(metadata):
        raise ValueError("逐图结果与本次验证名单不一致")
    rows = [dict(**metadata[r["name"]], **{k: v for k, v in r.items() if k != "name"}) for r in rows]
    subsets = {
        "all": rows,
        "duplicate_in_train": [r for r in rows if r["duplicate_in_train"]],
        "no_exact_train_duplicate": [r for r in rows if not r["duplicate_in_train"]],
        "unique_nontrain": [r for r in rows if r["unique_nontrain"]],
    }
    summary = [dict(subset=name, **summarize(group)) for name, group in subsets.items()]
    write_csv(output_dir / "validation_metrics.csv", summary)
    write_csv(output_dir / "validation_per_image.csv", rows)
    # 最差样本用于人工复核，不据此自动删除数据或改变阈值。
    ranked = sorted((r for r in rows if np.isfinite(float(r["f1"]))), key=lambda r: float(r["f1"]))
    write_csv(output_dir / "worst_images.csv", ranked[:20])
    scene_rows = []
    for subset in ("all", "unique_nontrain"):
        for scene in sorted({r["scene"] for r in subsets[subset]}):
            selected = [r for r in subsets[subset] if r["scene"] == scene]
            scene_rows.append(dict(subset=subset, scene=scene, **summarize(selected)))
    write_csv(output_dir / "scene_metrics.csv", scene_rows)
    # 边界也拆分重复组；这里使用全部带内像素，固定点表仍保留在 boundary/。
    boundary_path = output_dir / "boundary/per_image.csv"
    if boundary_path.exists():
        boundaries = [r for r in read_csv(boundary_path) if r["method"] == "raw" and r["scope"] == "all_pixels"]
        grouped_boundary = defaultdict(list)
        for subset, selected in subsets.items():
            names = {r["name"] for r in selected}
            for r in boundaries:
                if r["name"] in names:
                    grouped_boundary[(subset, r["radius_px"], r["region"])].append(r)
        write_csv(output_dir / "boundary_groups.csv", [
            dict(subset=key[0], radius_px=key[1], region=key[2], **summarize(group))
            for key, group in grouped_boundary.items()])
    if region_csv:
        mapping = {r["name"]: r["region_id"] for r in read_csv(region_csv)}
        if any(not mapping.get(r["name"]) for r in rows):
            raise ValueError("区域清单须为全部待评估影像提供非空 region_id")
        grouped = defaultdict(list)
        for r in rows:
            grouped[mapping[r["name"]]].append(r)
        regions = [dict(name=name, **summarize(group)) for name, group in sorted(grouped.items())]
        write_csv(output_dir / "per_region.csv", regions)
        write_csv(output_dir / "region_summary.csv", [summarize(regions)])
        if independent_regions:
            if any(r["duplicate_in_train"] or r["validation_copies"] > 1 for r in rows):
                raise ValueError("存在整幅重复，不能标记为独立区域测试")
            if len(regions) < 2:
                raise ValueError("只有一个区域，无法进行区域重抽样")
            rng = np.random.default_rng(SEED)
            scores = []
            for _ in range(bootstrap_repeats):
                sample = [regions[i] for i in rng.integers(len(regions), size=len(regions))]
                scores.append(summarize(sample)["iou"])
            scores = np.asarray(scores)
            finite = scores[np.isfinite(scores)]
            low, high = np.quantile(finite, [.025, .975]) if len(finite) else (np.nan, np.nan)
            write_csv(output_dir / "region_bootstrap.csv", [dict(
                metric="pooled_water_iou", regions=len(regions), repeats=bootstrap_repeats,
                defined_repeats=len(finite), seed=SEED, ci95_low=low, ci95_high=high)])
    elif independent_regions:
        raise ValueError("独立区域统计需要 REGION_CSV；不能把切片编号当区域编号")
    return summary


def run_validation(checkpoints=CHECKPOINTS, image_dir=IMAGE_DIR, label_dir=LABEL_DIR,
                   use_saved_val=USE_SAVED_VAL, output_dir=OUTPUT_DIR, scene_csv=SCENE_CSV,
                   region_csv=REGION_CSV, independent_regions=REGIONS_ARE_INDEPENDENT,
                   threshold=THRESHOLD, batch_size=BATCH_SIZE, use_amp=USE_AMP,
                   save_masks=SAVE_MASKS, max_visualizations=MAX_VISUALIZATIONS):
    """按模型逐个验证；复用质量流程，独立输出结果，不集成模型、不更新归一化参数。"""
    if not checkpoints or not 0 < threshold < 1:
        raise ValueError("需要检查点，且概率阈值须在 0 与 1 之间")
    if not use_saved_val and (image_dir is None or label_dir is None):
        raise ValueError("外部验证请明确填写影像和真实标签目录")
    if independent_regions and use_saved_val:
        raise ValueError("反复用于选模的内部验证集不能标记为独立测试")
    if independent_regions and region_csv is None:
        raise ValueError("独立区域测试需要 REGION_CSV，不能用切片编号替代")
    destination = (ROOT / "outputs/validation/quality" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                   if output_dir is None else Path(output_dir))
    destination.mkdir(parents=True, exist_ok=False)
    comparison, reference, manifest = [], None, None
    for method, path in checkpoints.items():
        checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        images = Path(checkpoint["image_dir"] if image_dir is None else image_dir)
        labels = Path(checkpoint["label_dir"] if label_dir is None else label_dir)
        identity = (str(images), str(labels), tuple(checkpoint["bands"]), checkpoint["train_files"],
                    checkpoint["val_files"] if use_saved_val else None, str(checkpoint["image_dir"]))
        if reference is not None and reference != identity:
            raise ValueError("本次并列比较要求相同训练/验证名单、数据目录和输入波段")
        if manifest is None:
            manifest = build_manifest(checkpoint, images, labels, use_saved_val,
                                      scene_csv if use_saved_val else None)
            write_csv(destination / "samples.csv", manifest)
            cprint(f"验证 {len(manifest)} 张；与训练整幅重复 {sum(r['duplicate_in_train'] for r in manifest)} 张；"
                   f"未发现训练重复且去重后 {sum(r['unique_nontrain'] for r in manifest)} 张。", "yellow", flush=True)
        reference = identity
        epoch = checkpoint["epoch"]
        del checkpoint
        gc.collect()
        model_dir = destination / method
        cprint(f"开始固定模型验证：{method}，epoch={epoch}，threshold={threshold}", "cyan", flush=True)
        # 每次只传一个检查点，关闭土地覆盖修正；raw 即网络本身的预测。
        run_postprocess([path], image_dir=images, label_dir=labels, use_saved_val=use_saved_val,
                        with_labels=True, threshold=threshold, batch_size=batch_size, use_amp=use_amp,
                        use_landcover=False, save_masks=save_masks, output_dir=model_dir,
                        save_visuals=max_visualizations != 0, max_visualizations=max_visualizations)
        rows = [dict(name=r["name"], **metrics_from_counts([int(r["raw_" + k]) for k in COUNT_KEYS]))
                for r in read_csv(model_dir / "per_image.csv")]
        summary = save_reports(model_dir, rows, manifest, region_csv, independent_regions)
        comparison.extend(dict(model=method, **r) for r in summary)
        write_csv(model_dir / "protocol.csv", [dict(
            checkpoint=str(Path(path).resolve()), epoch=epoch, threshold=threshold, batch_size=batch_size,
            use_amp=bool(use_amp and torch.cuda.is_available()), image_dir=str(images), label_dir=str(labels),
            use_saved_val=use_saved_val, normalization="saved training mean/std", landcover=False,
            region_csv=str(region_csv or ""), independent_regions=independent_regions)])
        for r in summary:
            cprint(f"{method}/{r['subset']}: n={r['samples']} F1={r['f1']:.6f} "
                   f"IoU={r['iou']:.6f} P={r['precision']:.6f} R={r['recall']:.6f}", "green", flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_csv(destination / "comparison.csv", comparison)
    cprint(f"验证结果：{destination}。未发现整幅重复不等于区域/事件独立。", "magenta", flush=True)
    return destination


if __name__ == "__main__":
    run_validation()
