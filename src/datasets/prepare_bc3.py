"""从原始SHP重新生成整景二值标签，直接替换prepared_scenes/labels。

运行前停止使用此数据的训练/评价；运行完成后再训练。
不复制SAR、不生成VRT、不修改原始SHP，也不改变train/val划分。
"""
from pathlib import Path
import csv
import json
import sys

import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.revise_bc_labels import LABEL_RULE, rebuild_scene

# 输入与输出使用同一个目录；按清单找到原始SHP和对应SAR。
DATA_DIR = Path(r"I:\水体数据\prepared_scenes")
READ_WINDOW_SIZE = 1024  # 只限制内存用量，输出仍是一景一幅完整标签。


def prepare_labels(data_dir=DATA_DIR, read_window_size=READ_WINDOW_SIZE):
    """逐景原位替换标签，最后同步清单计数；中断后重新运行即可。"""
    data_dir = Path(data_dir)
    manifest = data_dir / 'split_manifest.csv'
    with manifest.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    if not rows:
        raise ValueError('清单为空，没有可重建的标签')

    # 防止生成中断后，训练误读“部分新标签+部分旧标签”的混合数据。
    marker = data_dir / '.labels_rebuilding'
    marker.write_text('标签正在重建。请重新运行prepare_bc3.py完成后再训练。', encoding='utf-8')
    print(f'原位重建标签：{data_dir / "labels"}；共{len(rows)}景', flush=True)
    print('实际像素：水体=1，非水体=0，无效=255；显示：0白、1黑。', flush=True)
    totals = {split: dict(scenes=0, water=0, background=0, ignored=0)
              for split in ('train', 'val')}
    pending = manifest.with_name('split_manifest.labels.partial')
    with rasterio.Env(GDAL_CACHEMAX=64 * 1024 * 1024), \
            pending.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for number, row in enumerate(rows, 1):
            # 从源SHP重栅格化，不反转旧栅格，也不依赖查看软件的黑白显示。
            # 每幅临时标签写完并关闭后再替换原文件；原SAR文件及路径不动。
            result = rebuild_scene(row, data_dir, data_dir, read_window_size,
                                   create_vrt=False)
            writer.writerow(result)
            stream.flush()
            total = totals[result['split']]
            total['scenes'] += 1
            for key, field in (('water', 'water_pixels'), ('background', 'background_pixels'),
                               ('ignored', 'ignored_pixels')):
                total[key] += int(result[field])
            print(f"{number}/{len(rows)} {row['scene_id']} [{row['split']}]："
                  f"1(水体)={result['water_pixels']:,}，0(非水体)={result['background_pixels']:,}，"
                  f"255(无效)={result['ignored_pixels']:,}", flush=True)

    # 清单计数和规则一并更新，训练会按标签文件变化自动重建窗口缓存与类别权重。
    pending.replace(manifest)
    (data_dir / 'label_statistics.json').write_text(
        json.dumps(totals, ensure_ascii=False, indent=2), encoding='utf-8')
    (data_dir / 'revision_settings.json').write_text(json.dumps(dict(
        label_rule=LABEL_RULE, labels='0 non_water / 1 water / 255 invalid',
        source='split_manifest.csv: source_label', storage='original SAR and in-place GeoTIFF labels',
        palette='0 white / 1 black', read_window_size=read_window_size),
        ensure_ascii=False, indent=2), encoding='utf-8')
    (data_dir / '标签说明.txt').write_text(
        LABEL_RULE + '\n有效区域：水体1、非水体0；255仅为无效像素。\n'
        '色带0白、1黑，颜色不等于像素数值。原SAR、原SHP和地理分组不变。\n', encoding='utf-8')
    marker.unlink()
    print(f'全部标签替换完成：{totals}\n现在可以运行src/supervised/train.py。', flush=True)
    return data_dir


if __name__ == '__main__':
    try:
        prepare_labels()
    except Exception as error:
        print(f'标签重建未完成：{error}。修复后重新运行本脚本，完成前不要训练。', flush=True)
        raise SystemExit(1)
