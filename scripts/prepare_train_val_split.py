"""按场景与内容重复组固定划分数据；先生成计划，再显式执行搬迁。

这里只能减少已知的整幅重复及同辅助层泄漏，不能证明真实地区/事件独立。
所有原文件均保留；active=0 的副本和标签冲突样本不进入训练或验证。
"""
import csv
import hashlib
import random
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/Track1/train"
SCENE_CSV = ROOT / "docs/s1_samples/scene_catalog_20261026_163548/sample_scenes.csv"
PLAN_DIR = ROOT / "docs/s1_samples/grouped_split_20261027"
MANIFEST = PLAN_DIR / "split_manifest.csv"
MODE = "prepare"  # prepare 仅生成计划；move 搬迁；restore 按原清单恢复平铺目录。
VAL_RATIO = 0.2
SEED = 42
GROUP_MATCHING_AUXILIARY = True
DEM_MIN_UNIQUE = 100  # 两个 DEM 至少一个满足，排除常量/过于平坦辅助层的偶然一致。


def read_csv(path):
    """CSV 使用 BOM 编码，兼容 Excel 和 Python。"""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    """保存可复核的逐文件记录。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def file_hash(path):
    """文件字节哈希用于搬迁完整性；它与解码后的内容哈希用途不同。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(data, mask):
    """同时比较数组尺寸、类型、数值和有效掩膜，不依赖 TIFF 压缩方式。"""
    digest = hashlib.sha256()
    digest.update(str((data.shape, data.dtype.str)).encode("ascii"))
    digest.update(np.ascontiguousarray(data).tobytes())
    digest.update(np.ascontiguousarray(mask, dtype=np.uint8).tobytes())
    return digest.hexdigest()


def scan_samples(data_dir, scene_csv):
    """重新读取原始文件；仅在标签文件哈希一致时复用既有自动场景分类。"""
    data_dir = Path(data_dir)
    images, labels = data_dir / "images", data_dir / "labels"
    if any((base / split).exists() for base in (images, labels) for split in ("train", "val")):
        raise ValueError("已存在划分子目录，请复用原计划，不能重新扫描后再随机划分。")
    paths = sorted(images.glob("*.tif"))
    if not paths:
        raise ValueError("平铺 images 目录没有 TIFF。")
    expected_labels = {p.with_suffix(".png").name for p in paths}
    if expected_labels != {p.name for p in labels.glob("*.png")}:
        raise ValueError("TIFF 与同名 PNG 清单不对应。")
    scene_rows = {r["name"]: r for r in read_csv(scene_csv)} if Path(scene_csv).exists() else {}
    rows = []
    warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
    for number, path in enumerate(paths, 1):
        label_path = labels / path.with_suffix(".png").name
        label_file_sha = file_hash(label_path)
        with rasterio.open(path) as source:
            data = source.read(masked=True)
        with rasterio.open(label_path) as source:
            label = source.read(1, masked=True)
        if data.shape[0] < 2 or data.shape[1:] != label.shape:
            raise ValueError(f"{path.name} 波段或标签尺寸不匹配。")
        # 与 WaterDataset 一致：VV/VH 转 float32，无效位置填 0；有效零值仍保留。
        sar = data[:2].astype(np.float32)
        valid = np.isfinite(sar.data).all(axis=0) & ~np.ma.getmaskarray(sar).any(axis=0)
        sar_values = sar.filled(0)
        sar_values[:, ~valid] = 0
        label_valid = ~np.ma.getmaskarray(label) & np.isfinite(label.data)
        dem_unique = [0, 0]
        aux_hash = ""
        if data.shape[0] >= 6:
            auxiliary = data[2:6]
            aux_hash = array_hash(auxiliary.data, np.ma.getmaskarray(auxiliary))
            for index in range(2):
                dem = auxiliary[index]
                dem_valid = ~np.ma.getmaskarray(dem) & np.isfinite(dem.data)
                dem_unique[index] = int(np.unique(dem.data[dem_valid]).size)
        scene = scene_rows.get(path.name)
        scene_verified = bool(scene and scene.get("label_sha256") == label_file_sha)
        usable = valid & label_valid & np.isin(label.data, (0, 1))
        rows.append(dict(name=path.name, label_name=label_path.name,
                         scene=scene["water_context_scene"] if scene_verified else "unknown",
                         scene_verified=int(scene_verified),
                         scene_status="label_hash_verified" if scene_verified else "missing_or_changed_label",
                         water_pixels=int(((label.data == 1) & usable).sum()),
                         valid_pixels=int(usable.sum()),
                         input_sha256=array_hash(sar_values, valid),
                         label_sha256=array_hash(label.data, np.ma.getmaskarray(label)),
                         all_bands_sha256=array_hash(data.data, np.ma.getmaskarray(data)),
                         auxiliary_sha256=aux_hash, dem3_unique=dem_unique[0], dem4_unique=dem_unique[1],
                         image_file_sha256=file_hash(path), label_file_sha256=label_file_sha))
        if number % 100 == 0:
            print(f"读取与哈希：{number}/{len(paths)}", flush=True)
    return rows


def assign_split(rows, val_ratio=VAL_RATIO, seed=SEED, group_auxiliary=GROUP_MATCHING_AUXILIARY,
                 dem_min_unique=DEM_MIN_UNIQUE):
    """先合并不能跨侧的组，再按组主场景近似分层；从不让重复输入跨侧。"""
    rows = [dict(row) for row in rows]
    by_input, by_aux = defaultdict(list), defaultdict(list)
    parent = list(range(len(rows)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    # 不写额外类：并查集仅用于把精确重复和非平凡辅助层一致关系合为完整组。
    for index, row in enumerate(rows):
        by_input[row["input_sha256"]].append(index)
        if (group_auxiliary and row["auxiliary_sha256"] and
                max(int(row["dem3_unique"]), int(row["dem4_unique"])) >= dem_min_unique):
            by_aux[row["auxiliary_sha256"]].append(index)
    for members in list(by_input.values()) + list(by_aux.values()):
        for index in members[1:]:
            parent[find(index)] = find(members[0])

    # 同 SAR 异标签整组暂不参与；其余完全相同输入只保留文件名排序第一张。
    for members in by_input.values():
        members.sort(key=lambda i: rows[i]["name"])
        conflict = len({rows[i]["label_sha256"] for i in members}) > 1
        for position, index in enumerate(members):
            rows[index]["active"] = int(not conflict and position == 0)
            rows[index]["reason"] = "label_conflict" if conflict else ("retained" if position == 0 else "duplicate")
            rows[index]["representative"] = "" if conflict else rows[members[0]]["name"]
            rows[index]["exact_group_size"] = len(members)

    groups = defaultdict(list)
    for index in range(len(rows)):
        groups[find(index)].append(index)
    strata = defaultdict(list)
    group_records = []
    for members in groups.values():
        active = [i for i in members if rows[i]["active"]]
        scenes = Counter(rows[i]["scene"] for i in (active or members))
        scene = sorted(scenes, key=lambda item: (-scenes[item], item))[0]
        names = sorted(rows[i]["name"] for i in members)
        group_id = "g_" + hashlib.sha256("\n".join(names).encode()).hexdigest()[:16]
        # 区分“同影像重复组”与仅有辅助层重合线索的组，不能作为真实 region_id。
        aux_link = len({rows[i]["input_sha256"] for i in members}) > 1
        basis = "exact_sar+provisional_auxiliary_footprint" if aux_link else "exact_sar"
        record = dict(group_id=group_id, scene=scene, active_count=len(active), file_count=len(members),
                      group_basis=basis, scene_count=len(scenes), scene_members=";".join(sorted(scenes)), members=members)
        group_records.append(record)
        strata[scene].append(record)

    rng = random.Random(seed)
    for scene in sorted(strata):
        records = strata[scene]
        usable_groups = sorted((r for r in records if r["active_count"]), key=lambda r: r["group_id"])
        rng.shuffle(usable_groups)
        target = sum(r["active_count"] for r in usable_groups) * val_ratio
        chosen, val_count = set(), 0
        # 单组类别全部留在训练；其余仅在更接近目标时选入验证，保留至少一个训练组。
        for record in usable_groups:
            proposed = val_count + record["active_count"]
            if (len(chosen) < len(usable_groups) - 1 and
                    abs(proposed - target) < abs(val_count - target)):
                chosen.add(record["group_id"])
                val_count = proposed
        for record in records:
            split = "val" if record["group_id"] in chosen else "train"
            for index in record["members"]:
                row = rows[index]
                row.update(split=split, group_id=record["group_id"], split_scene=scene,
                           group_basis=record["group_basis"], group_file_count=record["file_count"],
                           group_scene_count=record["scene_count"], group_scenes=record["scene_members"],
                           image_relative=f"{split}/{row['name']}", label_relative=f"{split}/{row['label_name']}",
                           split_seed=seed, target_val_ratio=val_ratio)
    return sorted(rows, key=lambda row: row["name"])


def prepare_split(data_dir=DATA_DIR, scene_csv=SCENE_CSV, plan_dir=PLAN_DIR):
    """冻结计划和场景统计；此步骤绝不搬动数据，已有计划不被覆盖。"""
    plan_dir = Path(plan_dir)
    manifest = plan_dir / "split_manifest.csv"
    if manifest.exists():
        print(f"已有固定计划，直接复用：{manifest}")
        return manifest
    rows = assign_split(scan_samples(data_dir, scene_csv))
    write_csv(manifest, rows)
    scenes = sorted({r["scene"] for r in rows})
    counts = []
    for scene in scenes:
        records = [r for r in rows if r["scene"] == scene]
        counts.append(dict(scene=scene,
                           train_active=sum(r["active"] for r in records if r["split"] == "train"),
                           val_active=sum(r["active"] for r in records if r["split"] == "val"),
                           files=len(records), excluded_duplicate=sum(r["reason"] == "duplicate" for r in records),
                           excluded_label_conflict=sum(r["reason"] == "label_conflict" for r in records),
                           groups=len({r["group_id"] for r in records})))
    write_csv(plan_dir / "scene_counts.csv", counts)
    write_csv(plan_dir / "excluded_samples.csv", [r for r in rows if not r["active"]] or [dict(name="", reason="none")])
    print(f"计划已写入：{manifest}")
    for split in ("train", "val"):
        selected = [r for r in rows if r["split"] == split]
        print(f"{split}：物理文件对 {len(selected)}；active {sum(r['active'] for r in selected)}")
    print("未移动数据；场景为辅助层自动分类，group_id 不是已确认地理区域编号。")
    return manifest


def move_split(manifest=MANIFEST, data_dir=DATA_DIR, restore=False):
    """先校验所有路径和字节，再逐文件重命名并校验；重跑可接续，restore 可恢复。

    每一步写入 move_log.csv 或 restore_log.csv。发生中断时不清理文件，按同一
    manifest 重跑继续，或设置 restore=True 按逆序恢复；不会覆盖任何目标文件。
    """
    manifest, data_dir = Path(manifest).resolve(), Path(data_dir).resolve()
    rows = read_csv(manifest)
    installed_manifest = data_dir / "split_manifest.csv"
    if installed_manifest.exists() and file_hash(installed_manifest) != file_hash(manifest):
        raise ValueError("data 内已有不同划分清单，拒绝覆盖。")
    operations, unique_paths = [], set()
    group_sides, input_sides = defaultdict(set), defaultdict(set)
    for row in rows:
        if row["split"] not in ("train", "val") or row["active"] not in ("0", "1"):
            raise ValueError("manifest 的 split 或 active 字段无效。")
        group_sides[row["group_id"]].add(row["split"])
        input_sides[row["input_sha256"]].add(row["split"])
        for kind, basename_key, relative_key, hash_key in (
                ("images", "name", "image_relative", "image_file_sha256"),
                ("labels", "label_name", "label_relative", "label_file_sha256")):
            base = (data_dir / kind).resolve()
            basename = row[basename_key]
            if Path(basename).name != basename or "/" in basename or "\\" in basename:
                raise ValueError(f"原文件必须是平铺文件名：{basename}")
            expected_relative = f"{row['split']}/{basename}"
            if row[relative_key] != expected_relative:
                raise ValueError(f"目标相对路径与 split 不符：{row[relative_key]}")
            original, target = (base / basename).resolve(), (base / expected_relative).resolve()
            if not original.is_relative_to(base) or not target.is_relative_to(base):
                raise ValueError(f"路径越出指定目录：{original} -> {target}")
            if original in unique_paths or target in unique_paths:
                raise ValueError(f"重复搬迁路径：{original}")
            unique_paths.update((original, target))
            source, destination = (target, original) if restore else (original, target)
            if source.exists() and destination.exists():
                raise FileExistsError(f"源和目标同时存在，拒绝覆盖：{source} -> {destination}")
            existing = source if source.exists() else destination
            if not existing.is_file() or file_hash(existing) != row[hash_key]:
                raise ValueError(f"文件缺失或哈希已变化：{existing}")
            operations.append((source, destination, row[hash_key]))
    if any(len(sides) > 1 for sides in list(group_sides.values()) + list(input_sides.values())):
        raise ValueError("manifest 存在同组跨 train/val，拒绝执行。")
    # 所有源和目标已验证在 data_dir/images 或 labels 内；不使用其他 shell 搬文件。
    log_path = manifest.parent / ("restore_log.csv" if restore else "move_log.csv")
    new_log = not log_path.exists()
    with log_path.open("a", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("source", "destination", "sha256", "status"))
        if new_log:
            writer.writeheader()
        for number, (source, destination, expected_sha) in enumerate(reversed(operations) if restore else operations, 1):
            if source.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                # Windows 的 rename 在目标已存在时失败；这里明确检查避免覆盖。
                if destination.exists():
                    raise FileExistsError(destination)
                source.rename(destination)
                if file_hash(destination) != expected_sha:
                    raise IOError(f"搬迁后哈希不符，请根据日志恢复：{destination}")
                writer.writerow(dict(source=str(source), destination=str(destination), sha256=expected_sha, status="verified"))
                stream.flush()
            if number % 200 == 0:
                print(f"{'恢复' if restore else '搬迁'}：{number}/{len(operations)}", flush=True)
    if restore:
        if installed_manifest.exists() and installed_manifest != manifest:
            installed_manifest.unlink()  # 已确认同哈希的安装副本；原始计划始终保留。
        for kind in ("images", "labels"):
            for split in ("train", "val"):
                directory = data_dir / kind / split
                if directory.exists() and not any(directory.iterdir()):
                    directory.rmdir()  # 只删除本步骤产生的空目录。
    elif installed_manifest != manifest:
        installed_manifest.write_bytes(manifest.read_bytes())
    print(f"{'恢复' if restore else '搬迁'}完成，全部文件哈希一致；日志：{log_path}")
    return installed_manifest


if __name__ == "__main__":
    if MODE == "prepare":
        prepare_split()
    elif MODE == "move":
        move_split()
    elif MODE == "restore":
        move_split(restore=True)
    else:
        raise ValueError("MODE 应为 prepare、move 或 restore。")
