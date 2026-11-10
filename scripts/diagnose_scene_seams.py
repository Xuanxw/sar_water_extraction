"""固定检查点、真实整景上下文的局部接缝诊断，不启动训练或修改数据。

A: 原网格中心区硬拼；B: 原点平移半个中心区；C/D: 半步长高斯/Hann logits融合。
仅比较预先指定的 ROI，不可将这些局部结果称为全验证集精度。
"""
from pathlib import Path
import csv
import gc
import hashlib
import json
import math
import sys

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.special import expit
import torch

PROJECT = Path(r"E:\Code_projects\sar_water_extraction")
sys.path.insert(0, str(PROJECT / "src"))
from supervised.models.checkpoint import load_model
from supervised.evaluation.metrics import metrics_from_counts
from supervised.evaluation.boundary_eval import boundary_regions
from datasets.speckle import read_filtered_window, SpeckleFilter

CHECKPOINT = PROJECT / "outputs/pvt_deeplabv3plus_vv_sqrt_weight_bce_offline_normalized_bc_scenes/20261109_203148_859948/train_end.pth"
DATA_ROOT = Path(r"I:\水体数据\processed_scenes\normalized")
OUTPUT = PROJECT / "codex_analysis/seam_diagnosis_20261110"
ROIS = {
    "BC3-SM-ORG-VV-20230727T023738-000554": (6144, 4608, 2304, 2304),
    "BC3-SM-ORG-VV-20230813T153719-000767": (3840, 1536, 2304, 2304),
    "BC2-SM-ORG-VV-20240404T024601-012612": (7680, 3072, 2304, 2304),
}
SIZE, HALO, CORE, STRIDE = 1024, 128, 768, 384
SEAM_RADIUS, BOUNDARY_RADIUS = 16, 5
CACHE_ONLY = False  # True仅重融合已有logits，不加载模型到GPU；缺少缓存即停止。


def write_rows(path, rows):
    """每景完成及时更新小型统计表。"""
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def counts(label, prediction, region):
    """直接按布尔集合计数，避免整幅 int64 中间数组。"""
    water = label == 1
    return np.array([np.count_nonzero(region & ~water & ~prediction),
                     np.count_nonzero(region & ~water & prediction),
                     np.count_nonzero(region & water & ~prediction),
                     np.count_nonzero(region & water & prediction)], np.int64)


def origins(start, length, step, offset=0):
    """选取所有与 ROI 相交的全景中心区；ROI 边界不是网络输入边界。"""
    first = (math.floor((start - offset - CORE) / step) + 1) * step + offset
    return range(first, start + length, step)


def draw_result(path, scene, image, label, valid, probabilities, metrics):
    """显示压缩到768边长；所有计量仍使用原分辨率。"""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    stride = max(1, math.ceil(max(label.shape) / 768))
    cut = (slice(None, None, stride), slice(None, None, stride))
    figure = Figure(figsize=(15, 3.4 * len(probabilities)), dpi=100)
    FigureCanvasAgg(figure)
    axes = figure.subplots(len(probabilities), 4)
    lo, hi = np.percentile(image[valid], (2, 98))
    for row, (method, probability) in enumerate(probabilities.items()):
        pred = probability >= .5
        error = np.full((*label[cut].shape, 3), 50, np.uint8)
        v, gt, p = valid[cut], label[cut] == 1, pred[cut]
        error[~v] = 0
        error[v & gt & p] = (40, 140, 255)
        error[v & ~gt & p] = (255, 60, 60)
        error[v & gt & ~p] = (255, 220, 40)
        axes[row, 0].imshow(image[cut], cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
        axes[row, 1].imshow(np.ma.masked_where(~v, label[cut]), cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
        axes[row, 2].imshow(np.ma.masked_where(~v, probability[cut]), cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
        axes[row, 3].imshow(error, interpolation="nearest")
        m = metrics[method]
        axes[row, 2].set_title(f"{method}\nF1={m['f1']:.3f} P={m['precision']:.3f} R={m['recall']:.3f}", fontsize=10)
        for ax, title in zip(axes[row], ("SAR display p2-p98", "GT water=1 black", "", "TP blue / FP red / FN yellow")):
            if title:
                ax.set_title(title, fontsize=10)
            ax.axis("off")
    figure.suptitle(f"{scene}\nFixed epoch 6 | ROI only, native-grid metrics", fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, .96))
    try:
        figure.savefig(path)
    finally:
        figure.clear()
        del figure, axes
        gc.collect()


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    # 缓存必须与本次源文件相同；不对数百MB原景反复计算内容哈希。
    source_records = []
    for scene in ROIS:
        for kind in ("images", "labels"):
            path = DATA_ROOT / kind / "val" / (scene + ".tif")
            st = path.stat()
            source_records.append(dict(path=str(path), size=st.st_size, mtime_ns=st.st_mtime_ns))
    source_path = OUTPUT / "source_records.json"
    if source_path.exists() and json.loads(source_path.read_text(encoding="utf-8")) != source_records:
        raise ValueError("源影像/标签改变，不能复用原logits缓存")
    source_path.write_text(json.dumps(source_records, ensure_ascii=False, indent=2), encoding="utf-8")
    torch.set_num_threads(4)
    c = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if not c.get("inputs_normalized") or c.get("window_size") != SIZE or c.get("window_halo") != HALO:
        raise ValueError("本诊断要求与当前离线归一化1024/128训练协议一致")
    with CHECKPOINT.open("rb") as stream:
        checkpoint_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    config = dict(checkpoint=str(CHECKPOINT), checkpoint_sha256=checkpoint_hash,
                  epoch=c["epoch"], checkpoint_phase=c.get("checkpoint_phase"), rois=ROIS,
                  size=SIZE, halo=HALO, core=CORE, stride=STRIDE,
                  gaussian_sigma=CORE*.25, threshold=.5, seam_radius=SEAM_RADIUS,
                  boundary_radius=BOUNDARY_RADIUS, inputs_normalized=True,
                  amp=bool(c.get("amp_enabled", False)), batch_size=1,
                  scope="three_prespecified_rois_not_full_validation")
    destination = OUTPUT / "configuration.json"
    if destination.exists() and json.loads(destination.read_text(encoding="utf-8")) != json.loads(json.dumps(config)):
        raise ValueError("诊断配置变化；请使用新的输出目录")
    destination.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None if CACHE_ONLY else load_model(c).to(device).eval()
    amp = config["amp"] and device.type == "cuda"
    del c
    gc.collect()
    axis = (np.arange(CORE, dtype=np.float32) - (CORE - 1)/2) / (CORE*.25)
    gaussian = np.exp(-.5 * (axis[:, None]**2 + axis[None, :]**2)).astype("float32")
    hann_axis = np.hanning(CORE).astype("float32")
    hann = np.maximum(hann_axis[:, None] * hann_axis[None, :], 1e-3)
    (OUTPUT / "cache_refinement.json").write_text(json.dumps(dict(
        method="D_hann", weight="max(outer(np.hanning(core),np.hanning(core)),1e-3)",
        stride=STRIDE, cache_only=CACHE_ONLY), indent=2), encoding="utf-8")
    rows, comparisons, jump_rows = [], [], []
    for scene, (top, left, height, width) in ROIS.items():
        folder = OUTPUT / scene
        (folder / "logit_cache").mkdir(parents=True, exist_ok=True)
        image_path = DATA_ROOT / "images/val" / (scene + ".tif")
        label_path = DATA_ROOT / "labels/val" / (scene + ".tif")
        with rasterio.Env(GDAL_CACHEMAX=128*1024**2), rasterio.open(image_path) as src, rasterio.open(label_path) as target:
            image, valid = read_filtered_window(src, (1,), Window(left, top, width, height), SpeckleFilter())
            image = image[0]
            gt = target.read(1, window=Window(left, top, width, height), masked=True)
            valid &= ~np.ma.getmaskarray(gt) & np.isin(gt.data, (0, 1))
            label = np.where(valid, gt.data, 255).astype("uint8")
            scores = {}
            for method, step, offset, blend in (("A_hard", CORE, 0, None),
                                                 ("B_shift384", CORE, STRIDE, None),
                                                 ("C_gaussian", STRIDE, 0, gaussian),
                                                 ("D_hann", STRIDE, 0, hann)):
                accum = np.zeros((height, width), np.float32)
                weights = np.zeros_like(accum)
                tiles = [(r, col) for r in origins(top, height, step, offset)
                         for col in origins(left, width, step, offset)]
                for i, (r, col) in enumerate(tiles, 1):
                    cache = folder / "logit_cache" / f"r{r}_c{col}.npy"
                    if cache.exists():
                        core = np.load(cache)
                    else:
                        if CACHE_ONLY:
                            raise FileNotFoundError(f"CPU重融合缺少logits缓存：{cache}")
                        # 这些预选 ROI 均在整景内部，不允许以 ROI 为界补零。
                        if r-HALO < 0 or col-HALO < 0 or r+CORE+HALO > src.height or col+CORE+HALO > src.width:
                            raise ValueError("本实验ROI过于靠近全景边缘")
                        tile, good = read_filtered_window(src, (1,), Window(col-HALO, r-HALO, SIZE, SIZE), SpeckleFilter())
                        tile[:, ~good] = 0
                        x = torch.from_numpy(tile).unsqueeze(0).to(device)
                        with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=amp):
                            result = model(x)
                            out = result[-1] if isinstance(result, (list, tuple)) else result
                        score = out[0, 0] if out.shape[1] == 1 else out[0, 1] - out[0, 0]
                        core = score[HALO:HALO+CORE, HALO:HALO+CORE].float().cpu().numpy().copy()
                        if not np.isfinite(core).all():
                            raise FloatingPointError("非有限logits")
                        np.save(cache, core)
                        del x, result, out, score, tile, good
                    y0, x0 = max(top, r), max(left, col)
                    y1, x1 = min(top+height, r+CORE), min(left+width, col+CORE)
                    dest = (slice(y0-top, y1-top), slice(x0-left, x1-left))
                    source = (slice(y0-r, y1-r), slice(x0-col, x1-col))
                    weight = blend[source] if blend is not None else 1.
                    accum[dest] += core[source] * weight
                    weights[dest] += weight
                    if i % 10 == 0 or i == len(tiles):
                        print(f"{scene} {method}: {i}/{len(tiles)}", flush=True)
                if (weights == 0).any():
                    raise ValueError("ROI有未覆盖像素")
                accum /= weights
                scores[method] = expit(accum)
                np.save(folder / f"{method}_probability.npy", scores[method])
                del accum, weights
            ys, xs = np.arange(top, top+height), np.arange(left, left+width)
            seam = (np.minimum(ys % CORE, CORE-ys % CORE)[:, None] <= SEAM_RADIUS) | (np.minimum(xs % CORE, CORE-xs % CORE)[None, :] <= SEAM_RADIUS)
            shore = boundary_regions(label, valid, (BOUNDARY_RADIUS,))[BOUNDARY_RADIUS]
            regions = {"all": valid, "grid_seam16": valid & seam, "nonseam": valid & ~seam,
                       "shore5": shore["inner"] | shore["outer"]}
            metrics = {}
            for method, probability in scores.items():
                pred = probability >= .5
                # 只量化相同原网格线上的一像素概率跳变；不把它当作岸线精度。
                horizontal = [r-top for r in range((top//CORE+1)*CORE, top+height, CORE)]
                vertical = [x-left for x in range((left//CORE+1)*CORE, left+width, CORE)]
                jumps = [np.abs(probability[k]-probability[k-1])[valid[k] & valid[k-1]] for k in horizontal]
                jumps += [np.abs(probability[:,k]-probability[:,k-1])[valid[:,k] & valid[:,k-1]] for k in vertical]
                jump_rows.append(dict(scene=scene, method=method,
                                      grid_adjacent_probability_jump=float(np.concatenate(jumps).mean())))
                for region_name, region in regions.items():
                    row = dict(scene=scene, method=method, region=region_name,
                               **metrics_from_counts(counts(label, pred, region)))
                    rows.append(row)
                    if region_name == "all":
                        metrics[method] = row
                for alternative in ("B_shift384", "C_gaussian", "D_hann") if method == "A_hard" else ():
                    for region_name, region in regions.items():
                        p = scores[alternative]
                        comparisons.append(dict(scene=scene, comparison=f"A_vs_{alternative}", region=region_name,
                            probability_mae=float(np.abs(probability[region] - p[region]).mean()) if region.any() else float("nan"),
                            classification_flip_ratio=float(np.mean(pred[region] != (p[region] >= .5))) if region.any() else float("nan"),
                            valid_pixels=int(region.sum())))
            draw_result(folder / "comparison.png", scene, image, label, valid, scores, metrics)
            write_rows(OUTPUT / "metrics.csv", rows)
            write_rows(OUTPUT / "grid_sensitivity.csv", comparisons)
            write_rows(OUTPUT / "seam_jumps.csv", jump_rows)
            print("已完成ROI：", scene, {k: round(v["f1"], 5) for k, v in metrics.items()}, flush=True)
            del scores, image, valid, label, shore, regions
            gc.collect()
    print("诊断完成：", OUTPUT, flush=True)


if __name__ == "__main__":
    main()
