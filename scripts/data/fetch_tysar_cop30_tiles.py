"""导出 TYSAR train/val 的 Copernicus GLO-30 瓦片计划，在本地按计划下载。"""

import hashlib
import json
from pathlib import Path
import re
import sys

from filelock import FileLock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from datasets.dem_tile_download import _tile_plan, _usable
from thres_segmentation.auxiliary_data import download_tile
from utils.files import write_json


USAGE = """用法：
  python -u scripts/data/fetch_tysar_cop30_tiles.py plan <已归一化的VV目录> <瓦片计划.json>
  python -u scripts/data/fetch_tysar_cop30_tiles.py download <瓦片计划.json> <本地瓦片目录>
"""
TILE_NAME = re.compile(r"Copernicus_DSM_COG_10_[NS]\d{2}_00_[EW]\d{3}_00_DEM\.tif", re.ASCII)
BUCKET = "https://copernicus-dem-30m.s3.amazonaws.com"


def export_plan(vv_root, output):
    """读取已完成的 VV 清单和 GeoTIFF 头，导出去重后的原生瓦片名与对应景号。"""
    receipt = vv_root / "preprocessing.json"
    if not receipt.is_file():
        raise ValueError(f"缺少归一化完成记录：{receipt}")
    domain = json.loads(receipt.read_text(encoding="utf-8"))["output_domain"]
    method = "zscore" if domain == "standardized" else domain
    manifest = vv_root / "split_manifest.csv"
    before = hashlib.sha256(manifest.read_bytes()).hexdigest()
    specs, scenes, counts = _tile_plan(vv_root, method)
    if before != hashlib.sha256(manifest.read_bytes()).hexdigest():
        raise ValueError(f"扫描时划分清单发生变化，请重新导出：{manifest}")
    write_json(output, {
        "schema": 1,
        "product": "Copernicus GLO-30",
        "vv_output_domain": domain,
        "split_manifest_sha256": before,
        "scene_counts": dict(counts),
        "tiles": {name: scenes[name] for name in sorted(specs)},
    })
    print(f"已导出 {output}。训练 {counts['train']} 景，验证 {counts['val']} 景，"
          f"去重后 {len(specs)} 张 GLO-30 瓦片。", flush=True)


def download_plan(plan, directory):
    """校验计划格式，从公开 GLO-30 桶下载缺片，按已完成的 GeoTIFF 续跑。"""
    record = json.loads(plan.read_text(encoding="utf-8-sig"))
    if record.get("schema") != 1 or record.get("product") != "Copernicus GLO-30":
        raise ValueError(f"不支持的瓦片计划：{plan}")
    tiles = record.get("tiles")
    if not isinstance(tiles, dict) or not tiles or any(not TILE_NAME.fullmatch(name) for name in tiles):
        raise ValueError(f"瓦片计划为空或含非法文件名：{plan}")
    directory.mkdir(parents=True, exist_ok=True)
    missing = [name for name in sorted(tiles) if not _usable(directory / name)]
    print(f"共 {len(tiles)} 张，已有 {len(tiles) - len(missing)} 张有效，待下载 {len(missing)} 张。", flush=True)
    failed = []
    for index, name in enumerate(missing, 1):
        target = directory / name
        print(f"[{index}/{len(missing)}] {name}", flush=True)
        with FileLock(str(target) + ".lock"):
            if _usable(target):
                continue
            try:
                download_tile(f"{BUCKET}/{target.stem}/{name}", target)
            except (OSError, ValueError, RuntimeError) as error:
                failed.append(name)
                print(f"下载未完成：{name}，{error}", flush=True)
    if failed:
        raise RuntimeError(f"{len(failed)} 张待补，重跑同一命令续跑：{', '.join(failed)}")
    print(f"{len(tiles)} 张 GLO-30 原生瓦片已就绪：{directory}", flush=True)


def main(args):
    if args in (["-h"], ["--help"]):
        print(USAGE)
        return
    modes = {"plan": export_plan, "download": download_plan}
    if len(args) != 3 or args[0] not in modes:
        raise SystemExit(USAGE)
    modes[args[0]](*(Path(value).expanduser() for value in args[1:]))


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except KeyboardInterrupt:
        sys.exit("已中断。已完成的瓦片保留，重跑原命令即可续跑。")
    except (KeyError, OSError, ValueError, RuntimeError) as error:
        sys.exit(str(error))
