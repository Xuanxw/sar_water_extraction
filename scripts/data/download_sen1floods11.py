"""下载 Sen1Floods11 v1.1 的 446 组人工标注数据，并逐组生成水体判读图。

python -u scripts/data/download_sen1floods11.py 输出目录 [样本编号]
中断后重复相同命令。单个样本例如 Bolivia_103757。
"""

import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote, urlencode

from download_s1s2_water import Download, download_file, download_lock


BUCKET = "sen1floods11"
PREFIX = "v1.1/data/flood_events/HandLabeled/"
LAYERS = ("S1Hand", "S2Hand", "LabelHand")
PAIR_COUNT = 446
# 发布清单按本地相对路径排序，对逐文件名称、字节数和官方 MD5 签名。
CONTENTS_SHA256 = "82a406f86cb50e661603d29f94b83633d3de6a9518bd7a3724a698b7697ad6e5"


def fetch_catalog(curl):
    """读取三个官方目录的全部分页。对象 generation 用于钉住后续 Range 请求。"""
    items = []
    api = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"
    for layer in LAYERS:
        query = {"prefix": PREFIX + layer + "/", "maxResults": "1000",
                 "fields": "nextPageToken,items(name,size,md5Hash,generation)"}
        while True:
            url = api + "?" + urlencode(query)
            reply = subprocess.run(
                [curl, "--disable", "--fail", "--silent", "--show-error", "--location",
                 "--connect-timeout", "20", "--max-time", "60", "--retry", "3", "--retry-all-errors", url],
                stdout=subprocess.PIPE, check=True,
            )
            page = json.loads(reply.stdout)
            items.extend(v for v in page.get("items", []) if v["name"].endswith(".tif"))
            if "nextPageToken" not in page:
                break
            query["pageToken"] = page["nextPageToken"]
    return items


def paired_files(items):
    """依样本编号精确配齐 SAR、光学与人工标签，同时核对本版完整官方发布清单。"""
    groups = {}
    files = []
    for entry in items:
        remote = entry["name"]
        if not remote.startswith(PREFIX):
            raise ValueError(f"清单对象目录有误：{remote}")
        name = remote[len(PREFIX):]
        layer, filename = name.split("/", 1)
        suffix = f"_{layer}.tif"
        sample = filename.removesuffix(suffix)
        if (layer not in LAYERS or not filename.endswith(suffix)
                or not re.fullmatch(r"[A-Za-z][A-Za-z0-9-]*_[0-9]+", sample)):
            raise ValueError(f"未知发布对象：{remote}")
        checksum = base64.b64decode(entry["md5Hash"], validate=True)
        size, generation = int(entry["size"]), str(entry["generation"])
        if len(checksum) != 16 or size <= 0 or not re.fullmatch(r"[1-9][0-9]*", generation):
            raise ValueError(f"对象校验字段不完整：{remote}")
        url = f"https://storage.googleapis.com/{BUCKET}/{quote(remote)}?generation={generation}"
        item = Download(name, size, "md5", checksum.hex(), url)
        triplet = groups.setdefault(sample, {})
        if layer in triplet:
            raise ValueError(f"同一编号存在多份 {layer}：{sample}")
        triplet[layer] = item
        files.append(item)
    if len(groups) != PAIR_COUNT or any(set(v) != set(LAYERS) for v in groups.values()):
        raise ValueError("官方目录未配齐 446 组 S1、S2、人工 GT")
    signed = "".join(f"{f.name}\t{f.size}\t{f.checksum}\n" for f in sorted(files, key=lambda f: f.name))
    if hashlib.sha256(signed.encode()).hexdigest() != CONTENTS_SHA256:
        raise ValueError("v1.1 文件名称、大小或 MD5 与核实的发布清单不同")
    return groups


def write_json(path, value):
    """进程持锁时原子发布状态，避免中断留下半个 JSON。"""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(directory, selected=None):
    """读取固定清单；逐景续传原 TIFF，核验后输出事件目录下的 PNG。"""
    import sen1floods11_quicklook as quicklook

    curl = shutil.which("curl.exe" if os.name == "nt" else "curl")
    if not curl:
        raise RuntimeError("请安装 curl 并加入 PATH 后重试")
    directory = Path(directory).expanduser().resolve()
    state = directory / ".sen1floods11"
    state.mkdir(exist_ok=True, parents=True)
    with download_lock(state):
        catalog = state / "objects-v1.1.json"
        items = json.loads(catalog.read_bytes()) if catalog.is_file() else fetch_catalog(curl)
        groups = paired_files(items)
        if not catalog.is_file():
            write_json(catalog, items)
        if selected and selected not in groups:
            raise ValueError(f"在 446 组手工标签中未找到样本 {selected}")
        names = [selected] if selected else sorted(groups)
        size = sum(item.size for name in names for item in groups[name].values())
        print(f"共 {len(names)} 组，{len(names) * 3} 幅 TIFF，{size/1e9:.3f} GB；目标目录 {directory}", flush=True)

        receipt = state / "quicklooks.json"
        completed = json.loads(receipt.read_bytes()) if receipt.is_file() else {}
        version = file_sha256(Path(quicklook.__file__))

        def process(sample):
            triplet = groups[sample]
            for layer in LAYERS:
                download_file(directory, triplet[layer], curl)
            event = sample.rsplit("_", 1)[0]
            png = directory / "quicklooks" / event / f"{sample}.png"
            previous = completed.get(sample, {})
            # 每次校验原图 MD5；只有同一绘图实现生成且输出字节完好的 PNG 可直接复用。
            if previous.get("renderer") == version and png.is_file() and file_sha256(png) == previous.get("png"):
                return previous, "复用"
            png.parent.mkdir(parents=True, exist_ok=True)
            quicklook.render_quicklook(*(directory / triplet[layer].name for layer in LAYERS), png)
            return {"renderer": version, "png": file_sha256(png)}, "生成"

        pool = ThreadPoolExecutor(max_workers=min(3, len(names)))
        futures = {}
        try:
            futures = {pool.submit(process, name): name for name in names}
            for number, future in enumerate(as_completed(futures), 1):
                sample = futures[future]
                try:
                    result, action = future.result()
                except Exception as exc:
                    raise RuntimeError(f"{sample} 未完成：{exc}") from exc
                if completed.get(sample) != result:
                    completed[sample] = result
                    write_json(receipt, completed)
                print(f"配对与预览 {number}/{len(names)}  {sample}  {action}", flush=True)
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
    print(f"已完成 {len(names)} 组：{directory / 'quicklooks'}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3) or sys.argv[1] in {"-h", "--help"}:
        print(__doc__)
        sys.exit(0 if sys.argv[1:] in (["-h"], ["--help"]) else 2)
    try:
        run(*sys.argv[1:])
    except KeyboardInterrupt:
        sys.exit("已中断；原命令可续传。")
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        sys.exit(str(exc))
