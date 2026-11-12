"""整景清单的源路径相对 CSV 存放；路径搬迁不改变影像和训练划分。"""
from pathlib import Path
import csv
import hashlib
import io
import json
import os


def source_path(value, manifest):
    """解析 source_image/source_label；绝对路径兼容旧清单。"""
    path = Path(value)
    return path if path.is_absolute() else (Path(manifest).resolve().parent / path).resolve()


def relative_source(path, manifest):
    """写出相对 CSV 的路径；外部盘数据不能随目录搬迁，保留绝对路径。"""
    path = Path(path).resolve()
    try:
        return Path(os.path.relpath(path, Path(manifest).resolve().parent)).as_posix()
    except ValueError:  # Windows 不同盘符之间不存在相对路径。
        return str(path)


def manifest_fingerprint(path):
    """只在转换后的文件完全匹配时复用原指纹；其他清单改动正常使缓存失效。"""
    path = Path(path)
    current = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = path.with_suffix('.paths.json')
    if receipt.exists():
        record = json.loads(receipt.read_text(encoding='utf-8'))
        if record.get('converted_sha256') == current:
            return record['source_sha256']
    return current


def write_relative_manifest(source, target=None):
    """原位转换或跨目录复制清单，同时重新计算相对路径的基准。

    .paths.json 仅记录路径转换前后的哈希，让已有窗口索引和预处理记录
    无需因为路径写法变化而重算；不屏蔽像素、分组、标签计数等改动。
    原位批量转换前由调用者备份 CSV 和已有 .paths.json。
    """
    source = Path(source)
    target = Path(target) if target is not None else source
    fingerprint = manifest_fingerprint(source)
    with source.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    changed = 0
    for row in rows:
        for key in ('source_image', 'source_label'):
            if row.get(key):
                value = relative_source(source_path(row[key], source), target)
                changed += value != row[key]
                row[key] = value
    if source.resolve() == target.resolve() and not changed:
        return 0
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    data = stream.getvalue().encode('utf-8-sig')
    record = dict(source_sha256=fingerprint, converted_sha256=hashlib.sha256(data).hexdigest(),
                  path_base='csv_directory', fields=['source_image', 'source_label'])
    # 先发布兼容记录，再发布 CSV；中断时哈希不匹配会拒绝复用，不会误认完成。
    receipt = target.with_suffix('.paths.json')
    temporary = receipt.with_suffix('.json.partial')
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(receipt)
    temporary = target.with_suffix(target.suffix + '.paths.partial')
    temporary.write_bytes(data)
    temporary.replace(target)
    return changed
