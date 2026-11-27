"""以冻结的 DFC VV 秩＋DEM 权重评估 S1S2-Water v1.0.1 官方测试集。"""
import argparse
from contextlib import closing
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from datasets.dfc_sample.s1_dataset import vv_rank_dem_m
from supervised.evaluation.boundary_eval import confusion_counts
from supervised.evaluation.metrics import metrics_from_counts
from supervised.evaluation.sliding_inference import iter_scene_predictions
from supervised.models.checkpoint import load_model
from utils.raster import windows

DEFAULT_CHECKPOINT = (ROOT / "outputs/s1/pvt_deeplabv3plus_vv_dem_extend80"
                      / "20261119_094759_387905/best_f1.pth")


def test_scenes(root):
    """从发行目录枚举官方测试景。31 为 UInt16 且缺源产品号，暂不纳入 dB 评价。"""
    catalog = json.loads((root / "catalog.json").read_text(encoding="utf-8"))
    scenes, excluded = [], []
    for link in catalog["links"]:
        if link["rel"] != "item":
            continue
        meta = root / link["href"]
        props = json.loads(meta.read_text(encoding="utf-8"))["properties"]
        if props["split"] != "test":
            continue
        scene = meta.parent.name
        if scene == "31":
            excluded.append(dict(scene=scene, reason="v1.0.1 补发产品；UInt16 辐射编码未确认"))
            continue
        if not props["s1_srcids"] or any("_1SDV_" not in s for s in props["s1_srcids"]):
            raise ValueError(f"第 {scene} 景的源产品未确认双极化 VV/VH")
        scenes.append(meta.parent)
    if not scenes or len(set(scenes)) != len(scenes):
        raise ValueError("官方测试集为空或存在重复景号")
    return sorted(scenes, key=lambda x: int(x.name)), excluded


def prepare_inputs(image, dem, target, label, valid):
    """分块解码第 1 波段 VV 至 dB，将米制 DEM 对齐到 SAR 原网格。标签仅核对网格。"""
    with rasterio.open(image) as sar, rasterio.open(dem) as elevation:
        if sar.dtypes != ("int16", "int16") or not sar.crs:
            raise ValueError(f"不符合已核实的 Int16 双极化 dB×100 编码：{image}")
        if not elevation.crs or elevation.count != 1 or elevation.scales != (1.0,) or elevation.offsets != (0.0,):
            raise ValueError(f"DEM 需为带坐标系的单波段米值：{dem}")
        for mask in (label, valid):
            if mask.count != 1 or (mask.crs, mask.transform, mask.shape) != (sar.crs, sar.transform, sar.shape):
                raise ValueError(f"掩膜与 SAR 网格不一致：{mask.name}")
        grid = dict(crs=sar.crs, transform=sar.transform, width=sar.width, height=sar.height)
        with WarpedVRT(elevation, **grid, dtype="float32", nodata=float("nan"),
                       resampling=Resampling.bilinear) as aligned, rasterio.open(
                target, "w", driver="GTiff", **grid, count=2, dtype="float32",
                tiled=True, nodata=float("nan"), BIGTIFF="IF_SAFER") as dst:
            for window in windows(sar, 512):
                vv = sar.read(1, window=window, masked=True, out_dtype="float32").filled(np.nan)
                dst.write(vv / 100.0, 1, window=window)
                dst.write(aligned.read(1, window=window, masked=True).filled(np.nan), 2, window=window)


def evaluate_scene(folder, model, checkpoint, device, out):
    """先融合重叠窗 logits，再在官方有效区、已知标签和有效预测的交集上计分。"""
    scene = folder.name
    stem = f"sentinel12_s1_{scene}"
    counts = np.zeros(4, dtype=np.int64)
    ref_pixels = 0
    with TemporaryDirectory(dir=out) as tmp, rasterio.open(folder / (stem + "_msk.tif")) as labels, \
            rasterio.open(folder / (stem + "_valid.tif")) as valid:
        target = Path(tmp) / "vv_dem.tif"
        prepare_inputs(folder / (stem + "_img.tif"), folder / f"sentinel12_copdem30_{scene}_elevation.tif",
                       target, labels, valid)
        # 与现有 S1 整景推理一致：每窗先对有效 VV 排秩，DEM 保持米值，再用训练统计量标准化。
        blocks = iter_scene_predictions(model, target, bands=(1, 2), window_size=512, halo=64, stride=192,
            blend="gaussian", mean=checkpoint["mean"], std=checkpoint["std"], device=device,
            window_transform=lambda a, m: (vv_rank_dem_m(a, m), m), boundary_context=False, return_labels=False)
        next_progress = 25
        with closing(blocks):
            for block in blocks:
                top, left, height, width = block["window"]
                win = Window(left, top, width, height)
                truth = labels.read(1, window=win, masked=True)
                known = (valid.read(1, window=win, masked=True).filled(0) == 1) & ~np.ma.getmaskarray(truth)
                known &= np.isin(truth.data, (0, 1))
                ref_pixels += int(known.sum())
                counts += confusion_counts(truth.data, block["scores"] >= 0, known & block["valid_mask"])
                pct = 100 * (top + height) // labels.height
                if left + width == labels.width and pct >= next_progress:
                    print(f"  {scene}：{pct}%", flush=True)
                    next_progress = (pct // 25 + 1) * 25
    if not counts.sum():
        raise ValueError(f"第 {scene} 景没有共同有效像元")
    return dict(scene=scene, reference_pixels=ref_pixels, **metrics_from_counts(counts))


def evaluate(root, checkpoint_path, output_root):
    folders, excluded = test_scenes(root)
    missing = [s.name for s in folders if not (s / f"sentinel12_copdem30_{s.name}_elevation.tif").is_file()]
    if missing:
        raise FileNotFoundError("测试景缺 DEM，请先运行下载命令 --dem-only。缺少景号：" + ", ".join(missing))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    if (checkpoint.get("patch_representation") != "vv_rank_dem_m"
            or tuple(checkpoint.get("input_channels", ())) != ("VV", "DEM")
            or tuple(checkpoint.get("bands", ())) != (1, 4)
            or checkpoint.get("dem_unit") != "m" or checkpoint.get("inputs_normalized") or checkpoint.get("speckle")):
        raise ValueError("该检查点的训练输入协议与本次 VV 分块秩＋米制 DEM 验证不兼容")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint).to(device).eval()
    out = output_root / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)
    print(f"测试 {len(folders)} 景；排除 {excluded}；设备 {device}；输出 {out}", flush=True)
    results = []
    with (out / "per_scene.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        for idx, folder in enumerate(folders, 1):
            print(f"[{idx}/{len(folders)}] 第 {folder.name} 景", flush=True)
            row = evaluate_scene(folder, model, checkpoint, device, out)
            if not results:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            stream.flush()
            results.append(row)
            print(f"  F1={row['f1']:.4f}，IoU={row['iou']:.4f}", flush=True)
    micro = metrics_from_counts([sum(r[k] for r in results) for k in ("tn", "fp", "fn", "tp")])
    summary = dict(split="test", scenes=[r["scene"] for r in results], excluded=excluded,
                   checkpoint=str(checkpoint_path.resolve()), epoch=checkpoint["epoch"], threshold=0.5,
                   patch_representation=checkpoint["patch_representation"], reference_pixels=sum(r["reference_pixels"] for r in results),
                   pixel_metrics={k: v if math.isfinite(v) else None for k, v in micro.items()})
    for k in ("f1", "iou", "precision", "recall"):
        values = [r[k] for r in results if math.isfinite(r[k])]
        summary["mean_scene_" + k] = sum(values) / len(values) if values else None
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"完成，像素汇总 F1={micro['f1']:.4f}，IoU={micro['iou']:.4f}。结果在 {out}", flush=True)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="含 catalog.json 和各景子目录的数据集根目录")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/s1s2_water_external")
    args = parser.parse_args()
    evaluate(args.dataset, args.checkpoint, args.output_root)
