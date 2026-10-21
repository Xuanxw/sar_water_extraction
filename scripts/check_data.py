"""检查栅格能否完整读取及影像/标签配对情况，所有结果直接打印。"""
from collections import Counter
from pathlib import Path
import warnings

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning

# 输入只定义目录，不预设文件数量、波段数、数值单位或标签含义。
DATA_ROOT = Path("E:/Code_projects/sar_water_extraction/data")
DATASETS = [
    ("train", DATA_ROOT / "Track1/train/images", DATA_ROOT / "Track1/train/labels"),
    ("val", DATA_ROOT / "Track1/val/images", None),
    ("test", DATA_ROOT / "images", None),
]
EXTENSIONS = {".tif", ".tiff", ".png", ".img", ".vrt"}

# 无地理参考的文件也可读取；CRS 和变换会在下方显示，避免反复打印同类警告。
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


def check_data(image_dir, label_dir=None):
    """逐文件读取、统计实际数据，并按相对路径和同名文件检查标签配对。"""
    folders = [("影像", image_dir)]
    if label_dir is not None:
        folders.append(("标签", label_dir))

    file_keys = {}
    shapes = {}
    band_stats = {}
    label_counts = {}
    failures = 0

    for role, folder in folders:
        print(f"\n{role}目录：{folder}")
        if not folder.is_dir():
            print("  目录不存在")
        files = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
        print(f"  文件数量：{len(files)}")
        # 保留相对目录，去掉扩展名后配对；重复名称单独显示，避免随意选取。
        keys = Counter(p.relative_to(folder).with_suffix("").as_posix().casefold() for p in files)
        file_keys[role] = keys
        for key, count in keys.items():
            if count > 1:
                print(f"  同名文件重复：{key}，共 {count} 个")

        metadata = Counter()
        readable = 0
        for number, path in enumerate(files, 1):
            try:
                with rasterio.open(path) as src:
                    # 读取整个文件及掩膜，检查是否能够解码；每次只保留一个文件。
                    data = src.read()
                    masks = src.read_masks()
                    key = path.relative_to(folder).with_suffix("").as_posix().casefold()
                    shapes[(role, key)] = (src.height, src.width)
                    info = (f"尺寸={src.height}×{src.width}，波段数={src.count}，类型={src.dtypes}\n"
                            f"    描述={src.descriptions}，单位={src.units}，NoData={src.nodatavals}\n"
                            f"    scale={src.scales}，offset={src.offsets}\n"
                            f"    CRS={src.crs}，仿射变换={tuple(src.transform)[:6]}")
                    metadata[info] += 1
                    for index, (band, mask) in enumerate(zip(data, masks), 1):
                        finite = np.isfinite(band)
                        if role == "影像":
                            stats = band_stats.setdefault(index, dict(min=None, max=None, mask_invalid=0, nonfinite=0))
                            stats["mask_invalid"] += int(np.count_nonzero(mask == 0))
                            stats["nonfinite"] += int(np.count_nonzero(~finite))
                            # 统计原始存储值，不应用 scale/offset，不猜测未知填充值。
                            values = band[finite & (mask != 0)]
                            if values.size:
                                low, high = float(values.min()), float(values.max())
                                stats["min"] = low if stats["min"] is None else min(stats["min"], low)
                                stats["max"] = high if stats["max"] is None else max(stats["max"], high)
                        else:
                            # 包含有限的 NoData 标签值，便于观察实际编码；不推断类别含义。
                            values, counts = np.unique(band[finite], return_counts=True)
                            label_counts.setdefault(index, Counter()).update(
                                {value.item(): int(count) for value, count in zip(values, counts)})
                    readable += 1
            except Exception as exc:
                failures += 1
                print(f"  读取失败：{path}\n    {exc}")
            if number % 100 == 0:
                print(f"  已检查 {number}/{len(files)} 个文件", flush=True)

        print(f"  完整读取成功：{readable}/{len(files)}")
        for info, count in metadata.items():
            print(f"  {count} 个文件：{info}")

    print("\n影像波段统计（编号从 1 开始）：")
    for index, stats in sorted(band_stats.items()):
        print(f"  波段 {index}：min={stats['min']}，max={stats['max']}，"
              f"掩膜无效={stats['mask_invalid']}，NaN/Inf={stats['nonfinite']}")
    for index, counts in sorted(label_counts.items()):
        print(f"标签波段 {index} 的原始值计数：{dict(sorted(counts.items()))}")

    if label_dir is None:
        print("\n未指定标签目录，仅检查影像读取情况。")
    else:
        image_keys, label_keys = file_keys["影像"], file_keys["标签"]
        missing = image_keys.keys() - label_keys.keys()
        orphan = label_keys.keys() - image_keys.keys()
        print(f"\n缺少标签：{len(missing)}；没有对应影像的标签：{len(orphan)}")
        for key in sorted(missing):
            print(f"  缺少标签：{key}")
        for key in sorted(orphan):
            print(f"  没有对应影像：{key}")
        mismatches = 0
        for key in sorted(image_keys.keys() & label_keys.keys()):
            if image_keys[key] == label_keys[key] == 1 and ("影像", key) in shapes and ("标签", key) in shapes:
                if shapes[("影像", key)] != shapes[("标签", key)]:
                    mismatches += 1
                    print(f"  尺寸不一致：{key}，影像={shapes[('影像', key)]}，标签={shapes[('标签', key)]}")
        print(f"尺寸不一致：{mismatches} 对")
    print(f"读取失败文件数：{failures}")


if __name__ == "__main__":
    for name, image_dir, label_dir in DATASETS:
        print(f"\n{'=' * 20} {name} {'=' * 20}")
        check_data(image_dir, label_dir)
