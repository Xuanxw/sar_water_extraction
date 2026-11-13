"""整景SAR按需窗口读取：保持源文件完整，只缓存小型索引与训练统计量。"""
from collections import OrderedDict
from pathlib import Path
import csv
import hashlib
import json
import xml.etree.ElementTree as ET

import numpy as np
import rasterio
from rasterio.windows import Window
import torch
from torch.utils.data import Dataset
from datasets.speckle import SpeckleFilter, read_filtered_window
from datasets.manifest_paths import manifest_fingerprint, source_path

IGNORE_INDEX = -1


def check_preprocessed_input(image_dir, checkpoint):
    """校验输入阶段及归一化统计协议，防止漏做或重复做标准化。"""
    folder = Path(image_dir).parent
    path = folder / 'preprocessing.json'
    record = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    normalized = record.get('stage') == 'normalized'
    if normalized != checkpoint.get('inputs_normalized', False):
        raise ValueError('输入影像的归一化状态与检查点不一致')
    if normalized and (not record.get('complete') or (folder/'.preprocessing_incomplete').exists()):
        raise ValueError('离线归一化尚未完成')
    if normalized and record != checkpoint.get('input_preprocessing'):
        raise ValueError('输入影像的离线处理协议与检查点不一致')


class SARSceneDataset(Dataset):
    """以不重叠核心划分整景；窗口四周提供上下文，标签只保留核心区。"""

    def __init__(self, root, cache_dir, bands=(1,), window_size=1024, halo=128,
                 normalization_windows=2, seed=42, max_open_scenes=4, scene_limit_per_split=None,
                 speckle=None, require_prepared=False, _prepared_windows=None):
        self.root, self.cache_dir = Path(root), Path(cache_dir)
        self.require_prepared = require_prepared
        receipt = self.root / 'preprocessing.json'
        self.input_preprocessing = json.loads(receipt.read_text(encoding='utf-8')) if receipt.exists() else {}
        self.inputs_normalized = self.input_preprocessing.get('stage') == 'normalized'
        if require_prepared and (self.root / '.preprocessing_incomplete').exists():
            raise RuntimeError('离线影像处理尚未完成，请重新运行image_process.py')
        if self.inputs_normalized and require_prepared and not self.input_preprocessing.get('complete'):
            raise RuntimeError('归一化成果未完整发布')
        if self.inputs_normalized and speckle is not None:
            raise ValueError('输入已离线归一化，不能再次滤波')
        self.speckle_filter = SpeckleFilter(speckle)
        self.bands, self.window_size, self.halo = tuple(bands), window_size, halo
        self.core_size = window_size - 2 * halo
        if self.core_size <= 0 or window_size % 32 or halo < 0:
            raise ValueError('窗口需为32的倍数，且边缘上下文不能超过窗口一半')
        self.normalization_windows, self.seed = normalization_windows, seed
        self.max_open_scenes = max_open_scenes
        self._handles = OrderedDict()
        self._env = None
        if (self.root / '.labels_rebuilding').exists():
            raise RuntimeError('标签重建尚未完成，请先运行image_process.py（MODE=labels）完成全部标签替换')
        self.manifest_path = self.root/'split_manifest.csv'
        with self.manifest_path.open(encoding='utf-8-sig', newline='') as stream:
            self.scenes = [r for r in csv.DictReader(stream) if int(r['active']) == 1]
        if scene_limit_per_split is not None:
            # 仅供真实数据试跑；正式训练不设置此限制。
            self.scenes = [r for split in ('train', 'val')
                           for r in [x for x in self.scenes if x['split'] == split][:scene_limit_per_split]]
        if self.speckle_filter.snap is not None:
            for row in self.scenes:
                self.speckle_filter.snap.prepare_file(
                    self.root/'images'/row['split']/row['name'], self.bands, create=False)
        train_groups = {r['group_id'] for r in self.scenes if r['split'] == 'train'}
        val_groups = {r['group_id'] for r in self.scenes if r['split'] == 'val'}
        if not train_groups or not val_groups or train_groups & val_groups:
            raise ValueError('整景清单需有互不交叉的train/val地理组')
        # 文件尺寸/修改时间变化时重建缓存；不为缓存扫描整幅SAR内容。
        source_stamp = []
        for r in self.scenes:
            for kind, key in (('images', 'name'), ('labels', 'label_name')):
                path = self.root/kind/r['split']/r[key]
                st = path.stat()
                source_stamp.append((str(path), st.st_size, st.st_mtime_ns))
                # VRT本身很小；其引用的SAR变化时也必须使缓存失效。
                if path.suffix.lower() == '.vrt':
                    for node in ET.parse(path).findall('.//SourceFilename'):
                        linked = Path(node.text)
                        if node.get('relativeToVRT') == '1':
                            linked = path.parent / linked
                        linked_stat = linked.stat()
                        source_stamp.append((str(linked.resolve()), linked_stat.st_size, linked_stat.st_mtime_ns))
        signature = dict(version=1, manifest=manifest_fingerprint(self.manifest_path),
                         sources=source_stamp, window=window_size, halo=halo, bands=bands)
        self.cache_key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:20]
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache = self.cache_dir/f'windows_{self.cache_key}.npz'
        if _prepared_windows is not None and not cache.exists():
            # 合并已准备好的窗口索引，只调整景编号，不重新扫描整幅标签。
            pending = cache.with_suffix('.partial')
            with pending.open('wb') as stream:
                np.savez_compressed(stream, windows=_prepared_windows)
            pending.replace(cache)
        if cache.exists():
            with np.load(cache) as data:
                self.windows = data['windows']
            print(f'复用窗口索引：{len(self.windows)}个窗口', flush=True)
        else:
            if require_prepared:
                raise FileNotFoundError('缺少窗口索引，请先运行 python src/datasets/image_process.py <配置.yaml>')
            records = []
            for sid, r in enumerate(self.scenes):
                with rasterio.open(self.root/'labels'/r['split']/r['label_name']) as src:
                    count = 0
                    # 顺序条带读取，减少逐小窗GDAL调用和同一压缩块的重复解码。
                    for top in range(0, src.height, self.core_size):
                        h = min(self.core_size, src.height-top)
                        stripe = src.read(1, window=Window(0, top, src.width, h))
                        for left in range(0, src.width, self.core_size):
                            w = min(self.core_size, src.width-left)
                            y = stripe[:, left:left+w]
                            water, background = int((y == 1).sum()), int((y == 0).sum())
                            if water + background:
                                records.append((sid, top, left, h, w, water, background))
                                count += water + background
                    if count != int(r['water_pixels']) + int(r['background_pixels']):
                        raise ValueError(f"标签计数与准备清单不一致：{r['name']}")
                if (sid+1) % 25 == 0 or sid+1 == len(self.scenes):
                    print(f'标签窗口索引：{sid+1}/{len(self.scenes)}景；{len(records)}个窗口', flush=True)
            self.windows = np.asarray(records, dtype=np.int64).reshape(-1, 7)
            pending = cache.with_suffix('.partial')
            with pending.open('wb') as stream:
                np.savez_compressed(stream, windows=self.windows)
            pending.replace(cache)
        self.names = [f"{self.scenes[int(sid)]['scene_id']}_r{top}_c{left}" for sid, top, left, *_ in self.windows]
        self.train_ids = [i for i, r in enumerate(self.windows) if self.scenes[int(r[0])]['split'] == 'train']
        self.val_ids = [i for i, r in enumerate(self.windows) if self.scenes[int(r[0])]['split'] == 'val']
        if not self.train_ids or not self.val_ids:
            raise ValueError('train/val均需有已标注窗口')

    def __len__(self):
        return len(self.windows)

    def close(self):
        """关闭进程内文件缓存，避免遍历后持有大量文件。"""
        for pair in self._handles.values():
            for src in pair:
                src.close()
        self._handles.clear()
        if self._env is not None:
            self._env.__exit__(None, None, None)
            self._env = None

    def __getstate__(self):
        """Windows worker通过pickle启动，不传递GDAL句柄。"""
        state = self.__dict__.copy()
        state['_handles'], state['_env'] = OrderedDict(), None
        return state

    def __setstate__(self, state):
        """子进程恢复旧版本数据集时补齐关闭的滤波器，不改变旧训练的输入。"""
        self.__dict__.update(state)
        # 主进程可能在加入滤波前已创建数据集，Windows新worker会导入更新后的类。
        # 只兼容缺少字段的旧对象；新实验已有的滤波配置必须原样保留。
        if 'require_prepared' not in state:
            self.require_prepared = False
        if 'speckle_filter' not in state:
            self.speckle_filter = SpeckleFilter()

    def __getitem__(self, index):
        sid, top, left, h, w, _, _ = map(int, self.windows[index])
        r = self.scenes[sid]
        if self._env is None:
            self._env = rasterio.Env(GDAL_CACHEMAX=64 * 1024 * 1024, GDAL_NUM_THREADS='1')
            self._env.__enter__()
        if sid not in self._handles:
            if len(self._handles) >= self.max_open_scenes:
                for src in self._handles.popitem(last=False)[1]:
                    src.close()
            self._handles[sid] = (rasterio.open(self.root/'images'/r['split']/r['name']),
                                  rasterio.open(self.root/'labels'/r['split']/r['label_name']))
        self._handles.move_to_end(sid)
        image_src, label_src = self._handles[sid]
        # 裁到真实影像范围后读入固定大小数组，边界补无效值，不缩放。
        x0, y0 = left-self.halo, top-self.halo
        x1, y1 = max(0, x0), max(0, y0)
        x2, y2 = min(image_src.width, x0+self.window_size), min(image_src.height, y0+self.window_size)
        win = Window(x1, y1, x2-x1, y2-y1)
        # 离线归一化成果直接读取；原始输入模式可读取已准备的SNAP缓存。
        data, good = read_filtered_window(image_src, self.bands, win, self.speckle_filter)
        image = np.zeros((len(self.bands), self.window_size, self.window_size), dtype=np.float32)
        valid = np.zeros((self.window_size, self.window_size), dtype=bool)
        ys, xs = slice(y1-y0, y2-y0), slice(x1-x0, x2-x0)
        image[:, ys, xs] = data
        valid[ys, xs] = good
        label = np.full(valid.shape, IGNORE_INDEX, dtype=np.int64)
        target = label_src.read(1, window=Window(left, top, w, h), masked=True)
        core = label[self.halo:self.halo+h, self.halo:self.halo+w]
        usable = (~np.ma.getmaskarray(target) & np.isin(target.data, (0, 1)) &
                  valid[self.halo:self.halo+h, self.halo:self.halo+w])
        core[usable] = target.data[usable]
        return dict(image=torch.from_numpy(image), label=torch.from_numpy(label),
                    valid_mask=torch.from_numpy(valid), name=self.names[index], scene_id=r['scene_id'])

    def normalization(self, train_ids):
        """训练景内固定抽取窗口估计均值/标准差；不读取val像素，结果缓存并随模型保存。"""
        if self.inputs_normalized:
            # 浮点影像已标准化；返回恒等参数，供检查点记录，训练不再变换。
            return np.zeros(len(self.bands)), np.ones(len(self.bands)), self.input_preprocessing.get('statistics_window_indices', [])
        selected_ids = np.asarray(train_ids, dtype=np.int64)
        key = hashlib.sha256(selected_ids.tobytes()).hexdigest()[:12]
        # 标签窗口索引可以复用；滤波后的数值统计必须另存，不能沿用原始影像均值。
        if self.speckle_filter.config is not None:
            config = dict(version=1, **self.speckle_filter.config)
            key += '_' + hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
        path = self.cache_dir/f'normalization_{self.cache_key}_{key}_{self.seed}_{self.normalization_windows}.npz'
        if path.exists():
            with np.load(path) as saved:
                return saved['mean'], saved['std'], saved['indices'].tolist()
        if self.require_prepared:
            raise FileNotFoundError('缺少匹配的训练集归一化统计，请先运行 python src/datasets/image_process.py <配置.yaml>')
        rng = np.random.default_rng(self.seed)
        selected = []
        for sid in np.unique(self.windows[selected_ids, 0]):
            candidates = selected_ids[self.windows[selected_ids, 0] == sid]
            selected.extend(rng.choice(candidates, min(len(candidates), self.normalization_windows), replace=False).tolist())
        total = np.zeros(len(self.bands), dtype=np.float64)
        squares, count = total.copy(), 0
        for n, idx in enumerate(selected, 1):
            sample = self[idx]
            values = sample['image'].numpy()[:, sample['valid_mask'].numpy()].astype(np.float64)
            total += values.sum(axis=1)
            squares += np.square(values).sum(axis=1)
            count += values.shape[1]
            if n % 100 == 0:
                print(f'训练窗口归一化：{n}/{len(selected)}', flush=True)
        self.close()
        if count == 0:
            raise ValueError('训练窗口无有效SAR像素')
        mean = total/count
        std = np.maximum(np.sqrt(np.maximum(squares/count-mean**2, 0)), 1e-6)
        if not np.isfinite(mean).all() or not np.isfinite(std).all():
            raise ValueError('归一化参数不是有限值')
        with path.with_suffix('.partial').open('wb') as stream:
            np.savez(stream, mean=mean, std=std, indices=np.asarray(selected), pixels=count)
        path.with_suffix('.partial').replace(path)
        return mean, std, selected


def merge_scene_manifests(manifest_paths, **dataset_options):
    """合并多个整景目录的清单和窗口索引；影像不搬迁，预处理必须一致。"""
    paths = [Path(p).resolve() for p in manifest_paths]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError('manifest列表不能为空，也不能重复指定同一个文件')
    if any(p.name != 'split_manifest.csv' for p in paths):
        raise ValueError('每个目录需提供自己的split_manifest.csv')
    if len(paths) == 1:
        return paths[0].parent

    # 忽略来源路径和采样索引，比较影响数值含义的处理步骤及归一化参数。
    def protocol(record):
        keys = ('stage', 'bands', 'input_domain', 'output_domain', 'quantity', 'filter',
                'calibration_reference', 'mean', 'std')
        result = {k: record.get(k) for k in keys}
        if record.get('upstream'):
            result['upstream'] = protocol(record['upstream'])
        return result

    merged, windows, sources, seen = [], [], [], set()
    reference = None
    for path in paths:
        receipt = path.parent/'preprocessing.json'
        record = json.loads(receipt.read_text(encoding='utf-8')) if receipt.exists() else {}
        if record.get('stage') != 'normalized' or not record.get('complete'):
            raise ValueError(f'多目录训练要求已完成离线归一化：{path.parent}')
        if reference is None:
            reference = record
        elif protocol(record) != protocol(reference):
            raise ValueError('多个目录的归一化mean/std或预处理方案不同；请先统一训练统计和处理方案，不能直接混合')
        child = SARSceneDataset(path.parent, **dataset_options)
        try:
            offset = len(merged)
            for row in child.scenes:
                identifier = row['scene_id']
                if identifier.casefold() in seen:
                    raise ValueError(f'多个manifest存在重复景号：{identifier}；请移除重复来源')
                seen.add(identifier.casefold())
                # 合并清单放在仓库缓存，先按各自CSV解析源路径，防止基准目录改变。
                row = dict(row)
                for field in ('source_image', 'source_label'):
                    if row.get(field):
                        row[field] = str(source_path(row[field], path))
                # name使用绝对路径，现有训练/整景验证的Path拼接均会保留该绝对路径。
                merged.append(dict(row, original_name=row['name'],
                                   name=str((path.parent/'images'/row['split']/row['name']).resolve()),
                                   label_name=str((path.parent/'labels'/row['split']/row['label_name']).resolve()),
                                   manifest_source=str(path), dataset_root=str(path.parent)))
            part = child.windows.copy()
            part[:, 0] += offset
            windows.append(part)
        finally:
            child.close()
        sources.append(dict(manifest=str(path), sha256=manifest_fingerprint(path),
                            preprocessing=record))
    train_groups = {r['group_id'] for r in merged if r['split'] == 'train'}
    val_groups = {r['group_id'] for r in merged if r['split'] == 'val'}
    if train_groups & val_groups:
        raise ValueError(f'多个manifest的地理组跨train/val：{sorted(train_groups & val_groups)}')

    # 小型合并目录供训练和后续验证共同读取，不创建SAR副本。
    key = hashlib.sha256(json.dumps(dict(sources=sources, scenes=merged), sort_keys=True).encode()).hexdigest()[:20]
    root = Path(dataset_options['cache_dir'])/'merged_manifests'/key
    root.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in merged for k in row))
    pending = root/'split_manifest.csv.partial'
    with pending.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(merged)
    pending.replace(root/'split_manifest.csv')
    metadata = dict(reference, manifest_sources=sources)
    (root/'preprocessing.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    # 复用同一缓存签名逻辑，后续评估可以直接加载这个合并根目录。
    combined = SARSceneDataset(root, _prepared_windows=np.concatenate(windows), **dataset_options)
    combined.close()
    print(f'合并{len(paths)}个manifest，共{len(merged)}景；来源影像不复制：{root}', flush=True)
    return root
