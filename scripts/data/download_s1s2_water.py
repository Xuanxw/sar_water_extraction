"""下载 S1S2-Water v1.0.1。用法见 --help，依赖 Python 3 和系统 curl。"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from typing import NamedTuple
from uuid import uuid4


# 固定到作者的 v1.0.1 发布记录。字节数和 MD5 来自该记录的 files 接口。
FILES_URL = "https://zenodo.org/api/records/11278238/files"
FILES = {
    "catalog.json": (11717, "c41494472bcf68e3e6d46891af5be37e"),
    "part1.zip": (29478583513, "8b40e4281944faf55d8438996796d373"),
    "part2.zip": (32717311258, "8803693121ab03aa30e1e6cb4bf7a51d"),
    "part3.zip": (30763336558, "cdf974388e66c44652dc1d91f5bfa203"),
    "part4.zip": (29709506508, "e004d93785157f1585b8b114495d3344"),
    "part5.zip": (19792863346, "a969bb1d8446fa96d10c42b4a63706b4"),
    "part6.zip": (23079787372, "b462543bfcac14f5fc788fa4a6d4cce1"),
}
HF_REPO = "idomogalla/s1s2_water"
HF_REVISION = "e0551de726bf2fdd12ba4b5c24b470f6bc42d824"
MAX_STALLED = 5


class Download(NamedTuple):
    name: str  # 相对于输出目录的文件名，按景下载时保留景号目录。
    size: int
    algorithm: str
    checksum: str
    url: str


def zenodo_files(parts):
    names = ["catalog.json"]
    for part in parts:
        if part in ("catalog", "catalog.json"):
            continue
        name = part if part.endswith(".zip") else part + ".zip"
        if name not in FILES:
            raise ValueError(f"未知分卷 {part}，可用 part1 至 part6 或 catalog")
        if name not in names:
            names.append(name)
    return [Download(n, FILES[n][0], "md5", FILES[n][1], f"{FILES_URL}/{n}/content")
            for n in (names if parts else FILES)]


def mirror_index(curl):
    """从固定提交读取文件清单和散列，避免两次续传取到不同版本。"""
    url = f"https://huggingface.co/api/datasets/{HF_REPO}/revision/{HF_REVISION}?blobs=true"
    result = subprocess.run([curl, "--disable", "--fail", "--location", "--silent", "--show-error",
                             "--connect-timeout", "15", "--max-time", "45", "--retry", "4",
                             "--retry-all-errors", url], stdout=subprocess.PIPE)
    if result.returncode:
        raise RuntimeError(f"无法读取 Hugging Face 文件清单（curl {result.returncode}）。可重跑原命令。")
    index = json.loads(result.stdout)
    if index["sha"] != HF_REVISION:
        raise ValueError("镜像清单的提交号与指定版本不符")
    return {entry["rfilename"]: entry for entry in index["siblings"]}


def s1_files(index, selected, dem=False):
    """按真实景号配齐 SAR、标签、有效区和元数据，可附加高程。"""
    scenes = {name.split("/")[0] for name in index if "/" in name and name.split("/")[0].isdigit()}
    if not scenes:
        raise ValueError("镜像中找不到景号目录")
    unknown = set(selected) - scenes
    if unknown:
        raise ValueError("镜像没有这些景号：" + ", ".join(sorted(unknown)))
    files = []
    for scene in sorted(set(selected) or scenes, key=int):
        names = [f"sentinel12_s1_{scene}_{layer}.tif" for layer in ("img", "msk", "valid")]
        names.append(f"sentinel12_{scene}_meta.json")
        if dem:
            names.append(f"sentinel12_copdem30_{scene}_elevation.tif")
        for filename in names:
            name = f"{scene}/{filename}"
            if name not in index:
                raise ValueError(f"该景缺少文件：{name}")
            entry = index[name]
            if "lfs" in entry:
                algorithm, checksum = "sha256", entry["lfs"]["sha256"]
            else:
                algorithm, checksum = "git-sha1", entry["blobId"]
            url = f"https://huggingface.co/datasets/{HF_REPO}/resolve/{HF_REVISION}/{name}?download=true"
            files.append(Download(name, entry["size"], algorithm, checksum, url))
    return files


def verify(path, item):
    if path.stat().st_size != item.size:
        return False
    print(f"校验 {item.algorithm}  {item.name}  ({item.size:,} 字节)", flush=True)
    if item.algorithm == "git-sha1":
        # Git 直接保存的小文件使用 blob 对象 ID，散列输入包括 Git 对象头。
        digest = hashlib.sha1(f"blob {item.size}\0".encode("ascii"))
    else:
        digest = hashlib.new(item.algorithm)
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest() == item.checksum


@contextmanager
def download_lock(directory):
    # 操作系统在进程退出时释放锁，中断后可直接重启。
    with (directory / ".s1s2-water.lock").open("a+b") as handle:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError(f"无法锁定下载目录，请检查是否已有下载任务：{directory}") from exc
        else:
            import fcntl

            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"此目录已有下载任务：{directory}") from exc
        yield


def download_file(directory, item, curl):
    name, size = item.name, item.size
    final = directory / name
    final.parent.mkdir(parents=True, exist_ok=True)
    partial = directory / (name + ".part")
    if final.exists():
        if not verify(final, item):
            raise RuntimeError(f"现有文件与发布清单不符，请移到其他位置后重试：{final}")
        print(f"已完整，跳过  {name}", flush=True)
        return

    stalled = 0
    while True:
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == size:
            break
        if offset > size:
            raise RuntimeError(f"临时文件超过发布大小，请移走后重试：{partial}")
        needed = size - offset
        if shutil.disk_usage(directory).free < needed:
            raise RuntimeError(f"{name} 还需 {needed / 1e9:.2f} GB，目标盘剩余空间不足。")

        print(f"下载 {name}  已有 {offset:,} / {size:,} 字节", flush=True)
        # 每次重试都让 curl 根据 .part 的实际大小发 Range 请求。
        result = subprocess.run(
            [curl, "--disable", "--fail", "--location", "--continue-at", "-",
             "--connect-timeout", "30", "--speed-limit", "1024", "--speed-time", "120",
             "--output", str(partial), "--write-out", "%{http_code}",
             item.url],
            stdout=subprocess.PIPE, text=True,
        )
        status = result.stdout.strip()
        received = partial.stat().st_size if partial.exists() else 0
        if received == size:
            break
        transient = result.returncode in {0, 5, 6, 7, 16, 18, 28, 35, 52, 55, 56, 92}
        transient = transient or status in {"408", "429", "500", "502", "503", "504"}
        # 只计算连续无进展的失败次数；大文件即使断线多次也保留已写入的字节。
        stalled = stalled + 1 if received <= offset else 0
        if not transient or stalled >= MAX_STALLED:
            raise RuntimeError(f"{name} 下载中断（curl {result.returncode}, HTTP {status}）。"
                               f"进度保存在 {partial}，可用原命令续传。")
        delay = 30 if status == "429" else 2 ** max(1, stalled)
        print(f"curl {result.returncode} / HTTP {status}，已保存 {received:,} 字节，{delay} 秒后续传", flush=True)
        time.sleep(delay)

    if not verify(partial, item):
        damaged = final.with_name(f"{final.name}.bad-{uuid4().hex[:8]}")
        partial.rename(damaged)
        raise RuntimeError(f"{item.algorithm} 不匹配，已保留待排查文件 {damaged}。重跑将重新下载此文件。")
    partial.replace(final)
    print(f"校验通过  {final}", flush=True)


def main(args):
    parser = argparse.ArgumentParser(description="下载 S1S2-Water v1.0.1，支持中断后原命令续传。")
    parser.add_argument("directory", nargs="?", type=Path, help="输出目录")
    parser.add_argument("parts", nargs="*", help="Zenodo 分卷 part1 至 part6，或 catalog；默认全部")
    parser.add_argument("--s1", nargs="*", metavar="景号", help="从镜像按景下载 Sentinel-1；省略景号选取全部")
    parser.add_argument("--dem", action="store_true", help="按景下载时附加 Copernicus 高程")
    parser.add_argument("--list", action="store_true", help="仅列出所选文件、大小和校验值")
    options = parser.parse_args(args)
    if options.directory is None and not options.list:
        parser.error("请先指定输出目录")
    if options.s1 is not None and options.parts:
        parser.error("Zenodo 分卷与 --s1 需分开下载")
    if options.dem and options.s1 is None:
        parser.error("--dem 需要 --s1")
    curl = shutil.which("curl.exe" if os.name == "nt" else "curl")
    if not curl and (options.s1 is not None or not options.list):
        raise RuntimeError("未找到 curl，请先安装 curl 并加入 PATH。")
    files = (zenodo_files(options.parts) if options.s1 is None
             else s1_files(mirror_index(curl), options.s1, options.dem))
    total = sum(item.size for item in files)
    if options.list:
        for item in files:
            print(f"{item.name:12}  {item.size:>15,} bytes  {item.algorithm} {item.checksum}")
        print(f"共 {len(files)} 个文件，{total / 1e9:.2f} GB")
        return
    directory = options.directory.expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    source = "Zenodo" if options.s1 is None else f"Hugging Face {HF_REVISION[:8]}"
    print(f"S1S2-Water v1.0.1  {source}  {directory}\n共 {len(files)} 个文件，{total/1e9:.2f} GB", flush=True)
    with download_lock(directory):
        for i, item in enumerate(files, 1):
            print(f"[{i}/{len(files)}] {item.name}", flush=True)
            download_file(directory, item, curl)


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except KeyboardInterrupt:
        sys.exit("下载已中断；原命令可继续。")
    except (ValueError, RuntimeError, OSError) as error:
        sys.exit(str(error))
