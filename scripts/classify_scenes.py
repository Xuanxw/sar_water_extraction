"""按辅助土地覆盖给样本增加场景标签，不移动影像、不改变训练划分。"""
import csv
import hashlib
import sys
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import distance_transform_edt

ROOT = Path("E:/Code_projects/sar_water_extraction")
sys.path.insert(0, str(ROOT / "src"))
from datasets.water_dataset import sample_name, resolve_sample_path

IMAGE_DIR = Path("E:/Code_projects/sar_water_extraction/data/Track1/train/images")
LABEL_DIR = Path("E:/Code_projects/sar_water_extraction/data/Track1/train/labels")
# 此脚本汇总历史实验，split 列仍是 REFERENCE_RUN 的名单，不代表新的物理分区。
# 不重新 random_split，也不需要加载模型权重。
REFERENCE_RUN = Path("E:/Code_projects/sar_water_extraction/outputs/pvt_uafnet_hard_sampling_residual/20261022_160255_467390")
TRAIN_RECORD = REFERENCE_RUN / "sampling.csv"
VAL_RECORD = REFERENCE_RUN / "postprocess/20261022_170519_875509/per_image.csv"
OUTPUT_ROOT = Path("E:/Code_projects/sar_water_extraction/docs/s1_samples")
LANDCOVER_BAND = 5
BUFFER_PIXELS = 5  # 到标签水体的欧氏距离；文件空间参考不足，不解释为米。
DOMINANT_RATIO = 0.5  # 占区域已知覆盖像素至少50%，才称为主场景。
TAG_RATIO = 0.1       # 多标签保留至少10%的场景，不强制每张图只有一种背景。
MIN_LAND_PIXELS = 100
MIN_COVERAGE = 0.8    # 已知土地覆盖不足80%时保留“辅助数据不足”。
CLASSES = {10: "林地", 20: "灌丛", 30: "草地", 40: "农田", 50: "建成区",
           60: "裸地或稀疏植被", 70: "冰雪", 80: "永久水体", 90: "草本湿地",
           95: "红树林", 100: "苔藓地衣"}


def describe_region(cover, cover_valid, region, land_only=False):
    """统计类别占比；水体周边主场景排除80类，但仍保留完整原始计数。"""
    total = int(region.sum())
    known = region & cover_valid
    counts = {code: int(((cover == code) & known).sum()) for code in CLASSES}
    known_count = sum(counts.values())
    selected = {k: n for k, n in counts.items() if not land_only or k != 80}
    denominator = sum(selected.values())
    ordered = sorted(selected, key=lambda k: (-selected[k], k))
    shares = {k: n / denominator if denominator else 0. for k, n in selected.items()}
    if not total:
        primary = "无可用区域"
    elif known_count / total < MIN_COVERAGE:
        primary = "辅助数据不足"
    elif land_only and denominator < MIN_LAND_PIXELS:
        primary = "陆地背景不足"
    else:
        primary = CLASSES[ordered[0]] if shares[ordered[0]] >= DOMINANT_RATIO else "混合背景"
    usable = primary not in ("无可用区域", "辅助数据不足", "陆地背景不足")
    tags = [CLASSES[k] for k in ordered if shares[k] >= TAG_RATIO] if usable else []
    return dict(total=total, known=known_count, land=known_count - counts[80],
                primary=primary, tags=tags, counts=counts, shares=shares)


def classify_arrays(cover, cover_valid, label, valid):
    """分别描述整图、真值水体内，以及真值外侧5像素环带的陆地背景。"""
    water = valid & (label == 1)
    # 无水体时不计算距离，防止距离变换对空目标的特殊结果生成伪环带。
    ring = (valid & (label == 0) & (distance_transform_edt(~water) <= BUFFER_PIXELS)
            if water.any() else np.zeros_like(valid))
    tile = describe_region(cover, cover_valid, valid)
    inside = describe_region(cover, cover_valid, water)
    context = describe_region(cover, cover_valid, ring, land_only=True)
    if not water.any():
        context["primary"] = "无标签水体"
    return tile, inside, context


def write_csv(path, rows):
    """UTF-8 BOM方便Excel打开；数值保持原精度，空分母留空。"""
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    """逐图统计场景，导出清单，并按场景汇总既有验证预测的混淆矩阵。"""
    with TRAIN_RECORD.open(encoding="utf-8-sig") as stream:
        train_names = {r["name"] for r in csv.DictReader(stream) if int(r["epoch"]) == 1}
    with VAL_RECORD.open(encoding="utf-8-sig") as stream:
        val_rows = list(csv.DictReader(stream))
    val_metrics = {r["name"]: r for r in val_rows}
    paths = sorted(IMAGE_DIR.rglob("*.tif"))
    all_names = {sample_name(p, IMAGE_DIR) for p in paths}
    if train_names & val_metrics.keys() or train_names | val_metrics.keys() != all_names:
        raise ValueError("实验训练/验证名单与当前影像不完全对应，请检查顶部来源路径")
    if len(val_metrics) != len(val_rows):
        raise ValueError("验证记录存在重复文件名")
    output = OUTPUT_ROOT / ("scene_catalog_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    output.mkdir(parents=True, exist_ok=False)
    samples, histograms, memberships = [], [], []
    observed_codes = Counter()
    warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
    for number, path in enumerate(paths, 1):
        name = sample_name(path, IMAGE_DIR)
        label_path = resolve_sample_path(LABEL_DIR, Path(name).with_suffix(".png"))
        with rasterio.open(path) as source:
            sar = source.read((1, 2), masked=True)
            cover_data = source.read(LANDCOVER_BAND, masked=True)
        with rasterio.open(label_path) as source:
            label_data = source.read(1, masked=True)
        valid = (np.isfinite(sar.data).all(axis=0) & ~np.ma.getmaskarray(sar).any(axis=0)
                 & np.isfinite(label_data.data) & ~np.ma.getmaskarray(label_data))
        if not np.isin(label_data.data[valid], (0, 1)).all():
            raise ValueError(f"{name} 存在非0/1标签，需明确忽略规则")
        cover = cover_data.data
        cover_valid = (~np.ma.getmaskarray(cover_data) & np.isin(cover, list(CLASSES)))
        codes, counts = np.unique(cover[valid], return_counts=True)
        observed_codes.update(dict(zip(map(int, codes), map(int, counts))))
        tile, inside, context = classify_arrays(cover, cover_valid, label_data.data, valid)
        split = "train" if name in train_names else "val"
        row = dict(name=name, split=split, valid_pixels=int(valid.sum()), water_pixels=inside["total"],
                   water_ratio=inside["total"] / int(valid.sum()) if valid.any() else "",
                   tile_scene=tile["primary"], tile_tags=";".join(tile["tags"]),
                   water_context_scene=context["primary"], water_context_tags=";".join(context["tags"]),
                   context_pixels=context["total"], context_known_pixels=context["known"],
                   context_land_pixels=context["land"],
                   context_cover_fraction=context["known"] / context["total"] if context["total"] else "",
                   gt_on_worldcover_water=inside["counts"][80] / inside["known"] if inside["known"] else "",
                   label_sha256=hashlib.sha256(label_path.read_bytes()).hexdigest(),
                   landcover_sha256=hashlib.sha256(cover.tobytes() + cover_valid.tobytes()).hexdigest(),
                   review_status="自动分类_未经逐图人工确认")
        if split == "val":
            recorded = val_metrics[name]
            if (row["water_pixels"] != int(recorded["raw_tp"]) + int(recorded["raw_fn"])
                    or row["valid_pixels"] != int(recorded["raw_valid_pixels"])):
                raise ValueError(f"{name} 当前标签/有效范围与已有验证记录不同，不能直接分组比较")
        for code, title in CLASSES.items():
            if code != 80:
                row[f"context_{code}_fraction"] = context["shares"][code] if context["land"] else ""
        samples.append(row)
        for region_name, region in (("tile", tile), ("gt_water", inside), ("outer_ring", context)):
            for code, title in CLASSES.items():
                histograms.append(dict(name=name, split=split, region=region_name, code=code, scene=title,
                    class_pixels=region["counts"][code], known_pixels=region["known"],
                    region_pixels=region["total"],
                    fraction_of_known=region["counts"][code] / region["known"] if region["known"] else ""))
        # 主场景互斥且覆盖全部样本；多标签可重叠，不能把组计数相加当总数。
        for basis, labels in (("整图主场景", [tile["primary"]]),
                              ("水体周边主场景", [context["primary"]]),
                              ("水体周边多标签", context["tags"])):
            memberships.extend(dict(name=name, split=split, basis=basis, scene=s) for s in labels)
        if number % 100 == 0:
            print(f"场景统计：{number}/{len(paths)}", flush=True)

    totals = Counter((m["basis"], m["scene"], m["split"]) for m in memberships)
    keys = sorted({(m["basis"], m["scene"]) for m in memberships})
    summary = [dict(basis=b, scene=s, train_count=totals[b, s, "train"],
                    val_count=totals[b, s, "val"], total=totals[b, s, "train"] + totals[b, s, "val"])
               for b, s in keys]
    quality = []
    for basis, scene in keys:
        names = [m["name"] for m in memberships if m["split"] == "val" and m["basis"] == basis and m["scene"] == scene]
        if not names:
            continue
        for mode in ("raw", "post"):
            tn, fp, fn, tp = (sum(int(val_metrics[n][f"{mode}_{k}"]) for n in names) for k in ("tn", "fp", "fn", "tp"))
            quality.append(dict(basis=basis, scene=scene, prediction=mode, images=len(names),
                tn=tn, fp=fp, fn=fn, tp=tp,
                precision=tp / (tp + fp) if tp + fp else "",
                recall=tp / (tp + fn) if tp + fn else "",
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else "",
                iou=tp / (tp + fp + fn) if tp + fp + fn else ""))
    # 主场景计数和混淆矩阵必须能还原全体验证，防止分类中漏掉困难样本。
    for basis in ("整图主场景", "水体周边主场景"):
        assert sum(r["train_count"] for r in summary if r["basis"] == basis) == len(train_names)
        assert sum(r["val_count"] for r in summary if r["basis"] == basis) == len(val_metrics)
        for mode in ("raw", "post"):
            for k in ("tn", "fp", "fn", "tp"):
                assert sum(r[k] for r in quality if r["basis"] == basis and r["prediction"] == mode) == sum(int(r[f"{mode}_{k}"]) for r in val_rows)
    write_csv(output / "sample_scenes.csv", samples)
    write_csv(output / "scene_memberships.csv", memberships)
    write_csv(output / "landcover_histograms.csv", histograms)
    write_csv(output / "scene_counts.csv", summary)
    write_csv(output / "baseline_scene_metrics.csv", quality)
    manifest_hash = hashlib.sha256("\n".join(f"{r['name']}:{r['split']}:{r['label_sha256']}:{r['landcover_sha256']}" for r in samples).encode()).hexdigest()
    report = ["# 场景分类结果", "", f"样本：{len(samples)}；训练：{len(train_names)}；内部验证：{len(val_metrics)}。",
        "", "## 统计口径", "",
        f"第5波段按WorldCover类别解释。整图主场景占已知类别至少{DOMINANT_RATIO:.0%}；否则为混合背景。",
        f"水体周边为标签外侧{BUFFER_PIXELS}像素欧氏距离环带，主场景及多标签的分母排除80类永久水体。多标签阈值{TAG_RATIO:.0%}，陆地至少{MIN_LAND_PIXELS}像素，已知覆盖比例至少{MIN_COVERAGE:.0%}。",
        "这是自动辅助分类，不是人工场景真值。建成区含城市和乡村；农田不细分作物。年度土地覆盖不能证明当前地类或洪水机制。",
        "环带只描述图内可见背景，边缘截断和缺失数据不补造；不作配准偏移或真实岸线判断。未使用DEM坡度推断山体阴影，也不推断海岸、湖泊、河流或养殖塘类型。",
        "不移动样本、不改标签、不改划分、不加入训练输入。无水体和辅助数据不足样本仍保留。",
        "", "## 来源", "", f"训练名单：`{TRAIN_RECORD}`（epoch=1）。", f"验证名单及已有指标：`{VAL_RECORD}`。",
        "类别定义：[ESA WorldCover](https://esa-worldcover.org/en/data-access)。本地文件未记录辅助层年份，类别对应不等于已核实版本。",
        f"实际观测编码：{sorted(observed_codes)}。", f"样本/划分/标签清单SHA256：`{manifest_hash}`。",
        "", "## 如何评估", "",
        "scene_counts.csv 分别统计训练和验证。scene_memberships.csv 通过name与未来per_image.csv关联，固定分类版本用于模型比较。",
        "baseline_scene_metrics.csv 使用原残差全局困难采样最佳模型的既有预测；raw阈值0.5，post为已有土地覆盖补水结果。按组累计TN/FP/FN/TP后计算指标，非逐图F1平均。",
        "每个场景组的指标评价这些切片的全部有效像素，不是仅该地类像素，也不是边界指标。多标签组重叠，不能相加；图数少的类别只作描述。",
        "分类与post修正共用辅助土地覆盖，不能将post场景提升当作独立地类证据；模型能力优先看raw。",
        "", "## 水体周边主场景", "", "| 场景 | 训练 | 验证 |", "|---|---:|---:|"]
    report += [f"| {r['scene']} | {r['train_count']} | {r['val_count']} |" for r in summary if r["basis"] == "水体周边主场景"]
    (output / "README.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"完成：{output}", flush=True)
    for r in summary:
        if r["basis"] == "水体周边主场景":
            print(r, flush=True)


if __name__ == "__main__":
    main()
