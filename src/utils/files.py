"""通用文件工具：内容哈希、文件记录与原子JSON写入。"""

import hashlib
import json
from pathlib import Path


def sha256(path):
    """计算文件SHA256十六进制摘要，分块读取避免大影像/权重占用整块内存。"""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data, *, default=None, allow_nan=False):
    """原子写入UTF8 JSON，失败时保留已有完整文件。

    Parameters: path      - 输出文件路径。
                data      - 可序列化对象。
                default   - 可选类型转换器；需要保存Path时显式传str。
                allow_nan - 是否允许NaN；科学缓存默认禁止非标准数值。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, ensure_ascii=False, indent=2, default=default, allow_nan=allow_nan)
    pending = path.with_suffix(path.suffix + '.partial')
    pending.write_text(text, encoding='utf-8')
    pending.replace(path)


def stamp(path):
    """记录绝对路径、字节数及修改时间；用于快速缓存检查，不等同于内容哈希。"""
    path = Path(path)
    stat = path.stat()
    return {'path': str(path.resolve()), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def signature(value):
    """生成已有阈值缓存使用的20位JSON签名，保持序列化规则和缓存键不变。"""
    encoded = json.dumps(value, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()[:20]
