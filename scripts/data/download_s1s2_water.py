"""下载 S1S2-Water v1.0.1。用法见 --help，依赖 Python 3 和系统 curl。"""

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
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
USAGE = """python scripts/data/download_s1s2_water.py 输出目录 [part1 ... part6]
不指定分包时下载全部；每次都包含 catalog.json。仅下载目录文件可指定 catalog。
使用 --list 查看大小和 MD5。相同命令可继续中断的下载。"""


def verify(path, size, checksum):
    if path.stat().st_size != size:
        return False
    print(f"校验 MD5  {path.name}  ({size:,} 字节)", flush=True)
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest() == checksum


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


def download_file(directory, name, curl):
    size, checksum = FILES[name]
    final = directory / name
    partial = directory / (name + ".part")
    if final.exists():
        if not verify(final, size, checksum):
            raise RuntimeError(f"现有文件与官方记录不符，请移到其他位置后重试：{final}")
        print(f"已完整，跳过  {name}", flush=True)
        return

    for attempt in range(1, 6):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == size:
            break
        if offset > size:
            raise RuntimeError(f"临时文件超过官方大小，请移走后重试：{partial}")
        needed = size - offset
        if shutil.disk_usage(directory).free < needed:
            raise RuntimeError(f"{name} 还需 {needed / 1e9:.2f} GB，目标盘剩余空间不足。")

        print(f"下载 {name}  已有 {offset:,} / {size:,} 字节  尝试 {attempt}/5", flush=True)
        # 每次重试都让 curl 根据 .part 的实际大小发 Range 请求。
        result = subprocess.run(
            [curl, "--disable", "--fail", "--location", "--continue-at", "-",
             "--connect-timeout", "30", "--speed-limit", "1024", "--speed-time", "120",
             "--output", str(partial), "--write-out", "%{http_code}",
             f"{FILES_URL}/{name}/content"],
            stdout=subprocess.PIPE, text=True,
        )
        status = result.stdout.strip()
        if partial.exists() and partial.stat().st_size == size:
            break
        transient = result.returncode in {0, 5, 6, 7, 16, 18, 28, 52, 55, 56, 92}
        transient = transient or status in {"408", "429", "500", "502", "503", "504"}
        if not transient or attempt == 5:
            raise RuntimeError(f"{name} 下载中断（curl {result.returncode}, HTTP {status}）。"
                               f"进度保存在 {partial}，可用原命令续传。")
        delay = 30 if status == "429" else 2 ** attempt
        print(f"{delay} 秒后续传", flush=True)
        time.sleep(delay)

    if not verify(partial, size, checksum):
        damaged = directory / f"{name}.bad-{uuid4().hex[:8]}"
        partial.rename(damaged)
        raise RuntimeError(f"MD5 不匹配，已保留待排查文件 {damaged}。重跑将重新下载此包。")
    partial.replace(final)
    print(f"校验通过  {final}", flush=True)


def main(args):
    if args in (["-h"], ["--help"]):
        print(USAGE)
        return
    if args == ["--list"]:
        for name, (size, checksum) in FILES.items():
            print(f"{name:12}  {size:>15,} bytes  md5 {checksum}")
        return
    if not args:
        raise ValueError(USAGE)
    selected = ["catalog.json"]
    for arg in args[1:]:
        name = arg if arg.endswith(".zip") else arg + ".zip"
        if arg in {"catalog", "catalog.json"}:
            continue
        if name not in FILES:
            raise ValueError(f"未知分包 {arg}\n{USAGE}")
        if name not in selected:
            selected.append(name)
    if len(args) == 1:
        selected = list(FILES)
    curl = shutil.which("curl.exe" if os.name == "nt" else "curl")
    if not curl:
        raise RuntimeError("未找到 curl，请先安装 curl 并加入 PATH。")
    directory = Path(args[0]).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    total = sum(FILES[name][0] for name in selected)
    print(f"S1S2-Water v1.0.1  {directory}\n已选 {', '.join(selected)}  共 {total/1e9:.2f} GB", flush=True)
    with download_lock(directory):
        for name in selected:
            download_file(directory, name, curl)


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except KeyboardInterrupt:
        sys.exit("下载已中断；原命令可继续。")
    except (ValueError, RuntimeError, OSError) as error:
        sys.exit(str(error))
