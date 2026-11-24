"""核查归一化 BC VV 整景分布。用法：python scripts/data/audit_normalized_scenes.py 归一化目录 结果目录

每景使用全部有效标注像素。TMPDIR（Windows 为 TEMP）可指向本机 SSD，
临时空间约为最大单景有效标注像素数 × 4 字节，逐景释放。
"""

import csv
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from utils.files import sha256, signature, write_json
from utils.raster import file_percentiles


PERCENTILES = (2, 5, 25, 50, 75, 95, 98)
CLASSES = ("nonwater", "water")  # 索引即原始标签值 0、1


def exact_percentiles(path, count):
    """计算归一化像素分位数，缺类时保留空值。"""
    if count == 0:
        return {f"p{p:02}": None for p in PERCENTILES}
    result = file_percentiles(path, count, PERCENTILES)
    return {f"p{p:02}": float(v) for p, v in zip(PERCENTILES, result)}


def measure_scene(image_path, label_path):
    """在原始分辨率上匹配 VV 与同网格标签，统计有限且两侧掩膜有效的 0/1 像素。"""
    with TemporaryDirectory(prefix="bc_quantiles_") as tmp:
        paths = [Path(tmp) / name for name in CLASSES]
        counts = [0, 0]
        with rasterio.open(image_path) as image, rasterio.open(label_path) as label:
            if (image.shape, image.transform, image.crs) != (label.shape, label.transform, label.crs):
                raise ValueError(f"影像与标签网格不一致：{image_path} / {label_path}")
            if image.count != 1 or label.count != 1 or image.dtypes[0] != "float32":
                raise ValueError(f"预期单波段 float32 归一化 VV 和单波段标签：{image_path}")
            total = image.width * image.height
            with paths[0].open("wb") as land, paths[1].open("wb") as water:
                for _, window in image.block_windows(1):
                    z = image.read(1, window=window, masked=True)
                    y = label.read(1, window=window, masked=True)
                    valid = ~np.ma.getmaskarray(z) & np.isfinite(z.data) & ~np.ma.getmaskarray(y)
                    for code, stream in enumerate((land, water)):
                        pixels = z.data[valid & (y.data == code)]
                        pixels.tofile(stream)
                        counts[code] += pixels.size
        result = {"excluded_pixels": total - sum(counts)}
        for name, path, count in zip(CLASSES, paths, counts):
            result[name + "_pixels"] = count
            for key, value in exact_percentiles(path, count).items():
                result[f"{name}_{key}"] = value
        result["water_fraction"] = counts[1] / sum(counts) if sum(counts) else None
        result["nonwater_minus_water_p50"] = (
            result["nonwater_p50"] - result["water_p50"] if all(counts) else None
        )
        return result


def summarize(rows):
    """每景等权，比较同类地物的归一化值。缺类景只计入总景数。"""
    for split in ("train", "val"):
        for sensor in ("BC2", "BC3"):
            group = [r for r in rows if (r["split"], r["sensor"]) == (split, sensor)]
            for name in CLASSES:
                usable = [r for r in group if r[name + "_pixels"]]
                record = dict(split=split, sensor=sensor, label_class=name, scenes=len(group),
                              scenes_with_class=len(usable), pixels=sum(r[name + "_pixels"] for r in usable))
                for p in PERCENTILES:
                    key = f"{name}_p{p:02}"
                    record[f"median_of_scene_p{p:02}"] = np.median([r[key] for r in usable]) if usable else None
                medians = [r[name + "_p50"] for r in usable]
                for p in (10, 90):
                    record[f"p{p}_of_scene_p50"] = np.percentile(medians, p) if usable else None
                yield record


def write_csv(path, rows):
    pending = path.with_suffix(".csv.partial")
    with pending.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    pending.replace(path)


def main(root, output):
    receipt_path = root / "preprocessing.json"
    meta = json.loads(receipt_path.read_text(encoding="utf-8"))
    if ((meta.get("stage"), meta.get("bc_org_domain"), meta.get("input_domain"),
         meta.get("complete")) != ("normalized", "intensity", "intensity", True)
            or meta.get("output_domain") not in ("standardized", "scene_p02_p98", "scene_relative_db_p02_p98")
            or (root / ".preprocessing_incomplete").exists()):
        raise ValueError(f"请在功率域归一化完成后运行：{receipt_path}")
    with (root / "split_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        scenes = [r for r in csv.DictReader(stream) if int(r["active"]) == 1]
    if not scenes or len({r["scene_id"] for r in scenes}) != len(scenes):
        raise ValueError("清单为空，或同一景出现多次")
    print(f"共 {len(scenes)} 景；输出尺度 {meta['output_domain']}", flush=True)
    if meta["output_domain"] == "standardized":
        print(f"训练统计 mean={meta['mean']}，std={meta['std']}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    # 缓存每个完整景；成对的 TIFF 或归一化协议发生变化时重新计算该景。
    protocol = sha256(receipt_path)
    rows = []
    start = monotonic()
    for index, scene in enumerate(scenes, 1):
        sid, split = scene["scene_id"], scene["split"]
        sensor = sid.split("-", 1)[0].upper()
        if split not in ("train", "val") or sensor not in ("BC2", "BC3"):
            raise ValueError(f"未识别的划分或传感器：{split} / {sid}")
        paths = [root / "images" / split / scene["name"], root / "labels" / split / scene["label_name"]]
        stamps = []
        for path in paths:
            st = path.stat()
            stamps.append((path.name, st.st_size, st.st_mtime_ns))
            mask = Path(str(path) + ".msk")
            if mask.exists():
                st = mask.stat()
                stamps.append((mask.name, st.st_size, st.st_mtime_ns))
        key = signature([1, PERCENTILES, protocol, split, stamps])
        cache = output / ".scene_cache" / (sid + ".json")
        cached = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
        reuse = cached.get("signature") == key
        print(f"[{index}/{len(scenes)}] {split} {sid} {'复用' if reuse else '读取全景'}", flush=True)
        if reuse:
            values = cached["stats"]
        else:
            values = measure_scene(*paths)
            write_json(cache, {"signature": key, "stats": values})
        rows.append(dict(scene_id=sid, group_id=scene["group_id"], split=split, sensor=sensor, **values))
    write_csv(output / "scene_quantiles.csv", rows)
    write_csv(output / "group_quantiles.csv", list(summarize(rows)))
    print(f"完成 {len(rows)} 景，用时 {(monotonic() - start) / 60:.1f} 分钟。结果目录 {output}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
