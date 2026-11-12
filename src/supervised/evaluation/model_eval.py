"""独立评价已训练模型；修改顶部变量后运行 python src/supervised/evaluation/model_eval.py。"""

from math import log
from pathlib import Path
import csv
import hashlib
import sys
import json
from time import perf_counter

import numpy as np

ROOT = Path("E:/Code_projects/sar_water_extraction")
sys.path.insert(0, str(ROOT / "src"))  # 从 evaluation 直接运行时找到项目数据集与模型模块。

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from supervised.load.sar_scene_dataset import SARSceneDataset, check_preprocessed_input
from datasets.manifest_paths import manifest_fingerprint
from supervised.models.checkpoint import load_model
from supervised.evaluation.metrics import metrics_from_counts
from supervised.evaluation.sliding_inference import iter_scene_predictions

RUN_DIR = ROOT / "outputs/pvt_deeplabv3plus_vv_sqrt_weight_bce_offline_normalized_bc_scenes_overlap/20261110_221430_589086"
CHECKPOINT_PATH = RUN_DIR / "best_f1.pth"
IMAGE_DIR = None  # 默认使用检查点保存的输入路径和处理协议。
LABEL_DIR = None
USE_SAVED_VAL = True
BATCH_SIZE = None
USE_AMP = None
THRESHOLD = 0.5
MODE = "scan"  # scan：完整val融合扫描；single：旧窗口固定阈值评价，不用于本轮融合对照。
SCAN_CHECKPOINTS = {
    "epoch_006": CHECKPOINT_PATH,
    "epoch_007": RUN_DIR / "train_end.pth",
    "epoch_007_bn": RUN_DIR / "bn_calibration/20261111_164753_515753/calibrated.pth",
}
EXPECTED_EPOCHS = {"epoch_006": 6, "epoch_007": 7, "epoch_007_bn": 7}
# 固定目录自动续跑：每个模型/每景完成后保存小型计数文件；中断的景重算。
# 改模型、阈值或数据时需换新目录，避免混用不同协议的结果。
SCAN_OUTPUT_DIR = RUN_DIR / "threshold_scan_bn"
CPU_THREADS = 2

SCAN_THRESHOLDS = [i / 100 for i in range(1, 100)]  # 0.01～0.99，步长0.01，含0.5；各模型相同。
PRECISION_TARGETS = (0.80, 0.85, 0.90)
RECALL_TARGETS = (0.80, 0.85, 0.90)


def threshold_counts(score, label, thresholds, ignore_index=-1):
    """按水体 logit 分箱，一次统计所有阈值的 [TN,FP,FN,TP]，不创建阈值×像素大张量。"""
    valid = label != ignore_index
    edges = torch.logit(torch.as_tensor(thresholds, dtype=torch.float32, device=score.device))
    # right=True：恰等于阈值的像素也算水体，与 probability >= threshold 一致。
    bins = torch.bucketize(score[valid].float().contiguous(), edges, right=True)
    size = len(thresholds) + 1
    histogram = torch.bincount(label[valid] * size + bins, minlength=2 * size).reshape(2, size)
    tail = histogram.flip(1).cumsum(1).flip(1)[:, 1:]
    fp, tp = tail[0], tail[1]
    tn, fn = histogram[0].sum() - fp, histogram[1].sum() - tp
    return torch.stack((tn, fp, fn, tp), dim=1)


def precision_matches(rows, targets):
    """每档要求选 P>=目标且 Recall 最高的实测点；并列时选 Precision 更高者。"""
    matches = []
    for target in targets:
        eligible = [r for r in rows if r["precision"] >= target and r["tp"] + r["fp"] > 0]
        best = max(eligible, key=lambda r: (r["recall"], r["precision"])) if eligible else None
        matches.append(dict(target_precision=target, status="reachable" if best else "unreachable",
                            **({k: best[k] for k in ("threshold", "precision", "recall", "f1", "fp", "fn")}
                               if best else {k: "" for k in ("threshold", "precision", "recall", "f1", "fp", "fn")})))
    return matches


def evaluate(checkpoint_path=CHECKPOINT_PATH, image_dir=None, label_dir=None,
             use_saved_val=USE_SAVED_VAL, batch_size=BATCH_SIZE, use_amp=USE_AMP,
             threshold=THRESHOLD, thresholds=None):
    """恢复模型和输入处理，累计整套数据的混淆矩阵，在终端打印并返回指标。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = load_model(checkpoint).to(device).eval()  # 不更新权重或 BN 统计量。
    bands = checkpoint["bands"]
    ignore_index = checkpoint.get("ignore_index", -1)
    image_dir = Path(checkpoint["image_dir"] if image_dir is None else image_dir)
    label_dir = Path(checkpoint["label_dir"] if label_dir is None else label_dir)
    check_preprocessed_input(image_dir, checkpoint)
    scene_dataset = None
    if checkpoint.get('data_layout') == 'full_scene':
        if not use_saved_val:
            raise ValueError('整景复评需使用保存的验证窗口名单；外部数据需单独定义验证协议')
        split_path = image_dir.parent/'split_manifest.csv'
        if manifest_fingerprint(split_path) != checkpoint['split_manifest_sha256']:
            raise ValueError('整景采用清单与检查点不一致')
        scene_dataset = SARSceneDataset(image_dir.parent,
            Path(r'E:\Code_projects\sar_water_extraction\data_cache\bc_scenes'), bands=bands,
            window_size=checkpoint['window_size'], halo=checkpoint['window_halo'],
            speckle=checkpoint.get('speckle'))
        if scene_dataset.inputs_normalized != checkpoint.get('inputs_normalized', False):
            raise ValueError('评价输入归一化状态与检查点不一致')
        positions = {name:i for i,name in enumerate(scene_dataset.names)}
        ids = [positions[name] for name in checkpoint['val_files']]
        if not set(ids) <= set(scene_dataset.val_ids):
            raise ValueError('验证名单包含训练窗口')
        dataset = Subset(scene_dataset, ids)
    else:
        from datasets.dfc_sample.s1_dataset import WaterDataset
        dataset = WaterDataset(image_dir, label_dir, bands, speckle=checkpoint.get('speckle'))
        if use_saved_val:
            dataset.select_files(checkpoint['val_files'])
    batch_size = checkpoint["batch_size"] if batch_size is None else batch_size
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    use_amp = checkpoint.get("amp_enabled", False) if use_amp is None else use_amp
    use_amp = bool(use_amp and device.type == "cuda")
    mean = checkpoint["mean"].to(device).view(1, -1, 1, 1)
    std = checkpoint["std"].to(device).clamp_min(1e-6).view(1, -1, 1, 1)
    binary = checkpoint.get("model_kwargs", {}).get("num_classes", 2) == 1
    criterion = (nn.BCEWithLogitsLoss(reduction="none") if binary
                 else nn.CrossEntropyLoss(ignore_index=ignore_index))
    # 直接比较 logit，避免半精度 sigmoid 将接近 0.5 的概率舍入到阈值。
    logit_threshold = log(threshold / (1 - threshold)) if binary else None
    confusion = torch.zeros(4, dtype=torch.int64, device=device)
    scan = (torch.zeros(len(thresholds), 4, dtype=torch.int64, device=device)
            if thresholds is not None else None)
    loss_sum = valid_pixels = ignored_pixels = 0

    print(f"检查点：{checkpoint_path}；训练轮次：{checkpoint['epoch']}", flush=True)
    print(f"数据：{'原内部验证集' if use_saved_val else '指定目录全部有标签影像'}；"
          f"样本数：{len(dataset)}；波段：{bands}；设备：{device}；AMP：{use_amp}", flush=True)
    print(f"判别方式：{'单通道概率阈值 ' + str(threshold) if binary else '双类别 argmax'}", flush=True)
    with torch.inference_mode():
        for step, batch in enumerate(loader, 1):
            image = batch["image"].to(device)
            label = batch["label"].to(device)
            valid = label != ignore_index
            count = int(valid.sum().item())
            ignored_pixels += label.numel() - count
            if count == 0:
                continue
            if not checkpoint.get('inputs_normalized', False):
                image = (image - mean) / std
            image = image.masked_fill(~batch["valid_mask"].to(device).unsqueeze(1), 0)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(image)
                logits = outputs[-1] if isinstance(outputs, (tuple, list)) else outputs
                # 与训练验证一致：仅使用最后一个输出头，排除标签 -1。
                if binary:
                    loss = criterion(logits[:, 0].float(), label.clamp_min(0).float())[valid].mean()
                    prediction = (logits[:, 0] >= logit_threshold).long()
                else:
                    loss = criterion(logits, label)
                    prediction = logits.argmax(dim=1)
            loss_sum += loss.item() * count
            valid_pixels += count
            # 编码：真实类别*2+预测类别，依次对应 TN、FP、FN、TP。
            confusion += torch.bincount(label[valid] * 2 + prediction[valid], minlength=4)
            if scan is not None:
                # 双类别的 logit 差等价于水体 softmax 的 logit；只使用最终输出头。
                score = logits[:, 0].float() if binary else logits[:, 1].float() - logits[:, 0].float()
                scan += threshold_counts(score, label, thresholds, ignore_index)
            if step % 50 == 0:
                print(f"评价：{step}/{len(loader)} 批", flush=True)

    if scene_dataset is not None:
        scene_dataset.close()
    if valid_pixels == 0:
        raise ValueError("没有可用于评价的有效标签像素")
    tn, fp, fn, tp = confusion.tolist()
    metrics = dict(loss=loss_sum / valid_pixels, ignored_pixels=ignored_pixels,
                   **metrics_from_counts((tn, fp, fn, tp)))
    print(f"Loss={metrics['loss']:.6f}；水体 F1={metrics['f1']:.6f}；IoU={metrics['iou']:.6f}；"
          f"Precision={metrics['precision']:.6f}；Recall={metrics['recall']:.6f}", flush=True)
    print(f"TN/FP/FN/TP：{tn}/{fp}/{fn}/{tp}；有效/忽略像素：{valid_pixels}/{ignored_pixels}", flush=True)
    print(f"真实/预测水体占比：{metrics['true_water_ratio']:.4%}/{metrics['pred_water_ratio']:.4%}", flush=True)
    if scan is None:
        return metrics
    rows = []
    for value, (tn, fp, fn, tp) in zip(thresholds, scan.cpu().tolist()):
        rows.append(dict(threshold=value, **metrics_from_counts((tn, fp, fn, tp))))
    return rows


def file_hash(path):
    """流式指纹：防止 train_end 被覆盖后，续跑混入另一轮权重。"""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, record):
    """先写临时文件再替换，中断时保留上一次完整记录。"""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def prepare_scan(checkpoints, expected_epochs=None, image_dir=None, label_dir=None):
    """只读核查三个模型的原val、归一化和融合协议；不加载GPU、不扫描影像像素。"""
    if not checkpoints:
        raise ValueError("未指定检查点")
    reference, models = None, {}
    for name, path in checkpoints.items():
        if Path(name).name != name:
            raise ValueError("模型名称不能包含目录")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if expected_epochs and name in expected_epochs and checkpoint['epoch'] != expected_epochs[name]:
            raise ValueError(f"{name}实际为第{checkpoint['epoch']}轮，检查点可能被覆盖")
        config = checkpoint.get('validation_inference', {})
        if checkpoint.get('data_layout') != 'full_scene' or not config.get('blend'):
            raise ValueError("本次扫描要求保存了重叠融合协议的整景检查点")
        images = Path(checkpoint['image_dir'] if image_dir is None else image_dir)
        labels = Path(checkpoint['label_dir'] if label_dir is None else label_dir)
        check_preprocessed_input(images, checkpoint)
        root = images.parent
        if (root / '.labels_rebuilding').exists():
            raise ValueError('标签正在重建，不能扫描')
        if manifest_fingerprint(root / 'split_manifest.csv') != checkpoint['split_manifest_sha256']:
            raise ValueError('验证清单与检查点不同')
        revision = checkpoint.get('label_revision_sha256')
        if revision and file_hash(root / 'revision_settings.json') != revision:
            raise ValueError('标签版本与检查点不同')
        with (root / 'split_manifest.csv').open(encoding='utf-8-sig', newline='') as stream:
            active = [r for r in csv.DictReader(stream) if int(r['active']) == 1]
        wanted = set(checkpoint['val_scene_files'])
        scenes = sorted([r for r in active if r['split'] == 'val' and r['name'] in wanted],
                        key=lambda r: r['scene_id'])
        if not scenes or len(scenes) != len(wanted) or {r['name'] for r in scenes} != wanted:
            raise ValueError('检查点的完整验证景未在清单中找到')
        if len({r['scene_id'] for r in scenes}) != len(scenes):
            raise ValueError('验证景编号重复')
        train_groups = {r['group_id'] for r in active if r['split'] == 'train'}
        if train_groups & {r['group_id'] for r in scenes}:
            raise ValueError('train/val地理组交叉')
        protocol = dict(image_dir=str(images.resolve()), label_dir=str(labels.resolve()),
            split_manifest_sha256=checkpoint['split_manifest_sha256'], label_revision_sha256=revision,
            val_files=sorted(checkpoint['val_files']), val_scene_files=sorted(wanted),
            bands=list(checkpoint['bands']), window_size=checkpoint['window_size'],
            halo=checkpoint['window_halo'], validation_inference=config,
            mean=torch.as_tensor(checkpoint['mean']).flatten().tolist(),
            std=torch.as_tensor(checkpoint['std']).flatten().tolist(),
            inputs_normalized=checkpoint.get('inputs_normalized', False),
            input_preprocessing=checkpoint.get('input_preprocessing'), speckle=checkpoint.get('speckle'),
            batch_size=checkpoint['batch_size'], amp_enabled=checkpoint.get('amp_enabled', False))
        # 文件大小/时间核查无需读取整景；若文件变化，拒绝复用已完成景。
        stamps = []
        for row in scenes:
            for folder, key in ((images, 'name'), (labels, 'label_name')):
                source = folder / 'val' / row[key]
                stat = source.stat()
                stamps.append((str(source.resolve()), stat.st_size, stat.st_mtime_ns))
        protocol['source_stamps'] = stamps
        protocol = json.loads(json.dumps(protocol, default=str))
        if reference is not None and protocol != reference:
            raise ValueError(f'{name}的验证数据或推理协议与其他模型不同')
        reference = protocol
        models[name] = dict(path=str(Path(path).resolve()), sha256=file_hash(path), epoch=checkpoint['epoch'])
        print(f"核查通过：{name}，epoch={checkpoint['epoch']}，val={len(scenes)}景", flush=True)
        del checkpoint
    return dict(protocol=reference, models=models, scenes=scenes)


def fused_threshold_counts(scores, labels, valid_mask, edges):
    """融合logit分箱后反向累计，一次获得所有阈值的计数；每个有效像素仅计一次。"""
    valid = valid_mask & np.isin(labels, (0, 1))
    values = scores[valid]
    if not np.isfinite(values).all():
        raise FloatingPointError('融合输出包含NaN/Inf')
    bins = np.searchsorted(edges, values, side='right')
    size = len(edges) + 1
    bins += labels[valid].astype(np.int64) * size
    histogram = np.bincount(bins, minlength=2 * size).reshape(2, size)
    tail = histogram[:, ::-1].cumsum(axis=1)[:, ::-1][:, 1:]
    fp, tp = tail
    return np.stack((histogram[0].sum() - fp, fp, histogram[1].sum() - tp, tp), axis=1)


def save_scan_tables(output_dir, all_rows, scene_rows, targets, recall_targets):
    """更新CSV与曲线；匹配条件采用实际扫描点，不插值制造可达精度。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    comparisons, summary = [], []
    groups = sorted({(r['model'], r['subset']) for r in all_rows})
    for model, subset in groups:
        rows = [r for r in all_rows if (r['model'], r['subset']) == (model, subset)]
        for metric, levels, objective in (('precision', targets, 'recall'),
                                           ('recall', recall_targets, 'precision')):
            for level in levels:
                eligible = [r for r in rows if r[metric] >= level and r['tp'] + r['fp'] > 0]
                best = max(eligible, key=lambda r: (r[objective], r[metric])) if eligible else None
                comparisons.append(dict(model=model, subset=subset, constraint=metric, target=level,
                    status='reachable' if best else 'unreachable',
                    **{k: best[k] if best else '' for k in ('threshold', 'precision', 'recall', 'f1', 'iou')}))
        best = max(rows, key=lambda r: r['f1'] if np.isfinite(r['f1']) else -1)
        summary.append(dict(model=model, subset=subset, rule='best_f1', **{
            k: best[k] for k in ('threshold', 'precision', 'recall', 'f1', 'iou')}))
        for row in rows:
            if row['precision'] > .8 and row['recall'] > .8:
                summary.append(dict(model=model, subset=subset, rule='P_R_above_80', **{
                    k: row[k] for k in ('threshold', 'precision', 'recall', 'f1', 'iou')}))
    for filename, rows in (('threshold_metrics.csv', all_rows), ('per_scene_thresholds.csv', scene_rows),
                           ('matched_precision_recall.csv', comparisons), ('operating_points.csv', summary)):
        path = output_dir / filename
        temporary = path.with_suffix('.csv.tmp')
        with temporary.open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    try:
        for ax, subset in zip(axes, ('all', 'BC3')):
            for model in sorted({r['model'] for r in all_rows}):
                rows = [r for r in all_rows if r['model'] == model and r['subset'] == subset
                        and r['tp'] + r['fp'] > 0 and np.isfinite(r['recall'])]
                if rows:
                    ax.plot([r['recall'] for r in rows], [r['precision'] for r in rows], label=model)
            ax.axvline(.8, color='gray', linestyle='--')
            ax.axhline(.8, color='gray', linestyle='--')
            ax.set(title=subset, xlabel='Recall', ylabel='Precision', xlim=(0, 1), ylim=(0, 1))
            ax.grid(alpha=.3)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / 'pr_curve.png', dpi=140)
    finally:
        plt.close(fig)


# 仅兼容本次已做逐像元等价测试的实现升级；模型、数据、阈值及融合协议仍须完全一致。
LEGACY_SCAN_HASHES = (
    "987dcf2a0eb36cdaeb704f394856ac63964591b7b5f0c6a32c0d36e7aefcc76a",
    "1a4be4a54913aa9844f0a87f5f245ffcc43adaf6cc30ff5d3f2e9f1a3d4bb518",
)
OPTIMIZED_INFERENCE_HASH = "9cd25d3459f2bb4da6168f887655ddad2f37089a092365c11800ac07d84e00b7"
PATH_UPGRADE_PREVIOUS_HASH = "afe72a4e0dd84832c44bb4fbadc167f8b32b299f1bcc00882955edc8d17d1924"


def same_scan_protocol(saved, current):
    """允许已核验的等价提速版本续跑；其他源码变化仍需新输出目录。"""
    if saved == current:
        return True
    previous = (saved.get('code_hash'), saved.get('inference_code_hash'))
    # 本次仅兼容清单路径转换指纹；不改变推理、阈值或统计口径。
    compatible = (LEGACY_SCAN_HASHES, (PATH_UPGRADE_PREVIOUS_HASH, OPTIMIZED_INFERENCE_HASH))
    if previous not in compatible or current.get('inference_code_hash') != OPTIMIZED_INFERENCE_HASH:
        return False
    return ({k: v for k, v in saved.items() if k not in ('code_hash', 'inference_code_hash')} ==
            {k: v for k, v in current.items() if k not in ('code_hash', 'inference_code_hash')})


def run_threshold_scan(checkpoints=SCAN_CHECKPOINTS, thresholds=SCAN_THRESHOLDS,
                       targets=PRECISION_TARGETS, output_dir=SCAN_OUTPUT_DIR, image_dir=None, label_dir=None,
                       recall_targets=RECALL_TARGETS, expected_epochs=EXPECTED_EPOCHS):
    """完整val采用原Hann等融合协议；一景推理一次，原子保存计数，重启自动续跑。"""
    thresholds = list(thresholds)
    if not thresholds or not all(0 < t < 1 for t in thresholds) or sorted(set(thresholds)) != thresholds:
        raise ValueError('阈值须在0～1之间，严格递增且不重复')
    torch.set_num_threads(CPU_THREADS)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    metadata = prepare_scan(checkpoints, expected_epochs, image_dir, label_dir)
    protocol, scenes = metadata['protocol'], metadata['scenes']
    metadata.update(thresholds=thresholds, precision_targets=list(targets), recall_targets=list(recall_targets),
                    device=str(device), algorithm='fused_logit_histogram_v1',
                    code_hash=file_hash(__file__), inference_code_hash=file_hash(
                        Path(__file__).with_name('sliding_inference.py')))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / 'scan_protocol.json'
    if manifest.exists():
        if not same_scan_protocol(json.loads(manifest.read_text(encoding='utf-8')), metadata):
            raise ValueError('续跑目录的模型、数据或扫描协议已改变，请设置新的SCAN_OUTPUT_DIR')
    else:
        if any(output_dir.iterdir()):
            raise ValueError('输出目录非空且缺少扫描协议，请换新目录')
        write_json(manifest, metadata)
    # sigmoid单调，比较logit与log(t/(1-t))等价，避免半精度概率舍入。
    edges = np.log(np.asarray(thresholds, np.float64) / (1 - np.asarray(thresholds, np.float64)))
    all_rows, scene_rows, denominators = [], [], {}
    start = perf_counter()
    for model_number, (name, path) in enumerate(checkpoints.items(), 1):
        folder = output_dir / 'scene_counts' / name
        folder.mkdir(parents=True, exist_ok=True)
        completed = sum((folder / f"{r['scene_id']}.npz").exists() for r in scenes)
        print(f"模型{model_number}/{len(checkpoints)}：{name}；复用{completed}/{len(scenes)}景；设备{device}", flush=True)
        model = checkpoint = None
        totals = {'all': np.zeros((len(thresholds), 4), np.int64),
                  'BC3': np.zeros((len(thresholds), 4), np.int64)}
        numbers = {'all': 0, 'BC3': 0}
        for number, row in enumerate(scenes, 1):
            destination = folder / f"{row['scene_id']}.npz"
            reused = destination.exists()
            scene_start = perf_counter()
            if reused:
                with np.load(destination, allow_pickle=False) as saved:
                    counts = saved['counts']
                    if not np.array_equal(saved['thresholds'], thresholds):
                        raise ValueError(f'景缓存阈值不一致：{destination}')
            else:
                if model is None:
                    if file_hash(path) != metadata['models'][name]['sha256']:
                        raise ValueError('检查点在扫描期间被修改')
                    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
                    model = load_model(checkpoint).to(device).eval()
                    del checkpoint
                counts = np.zeros((len(thresholds), 4), np.int64)
                config = protocol['validation_inference']
                print(f"  开始{name} {number}/{len(scenes)}景：{row['scene_id']}", flush=True)
                for index, block in enumerate(iter_scene_predictions(model,
                        Path(protocol['image_dir']) / 'val' / row['name'],
                        Path(protocol['label_dir']) / 'val' / row['label_name'],
                        bands=protocol['bands'], window_size=protocol['window_size'], halo=protocol['halo'],
                        stride=config.get('stride'), blend=config['blend'], mean=protocol['mean'], std=protocol['std'],
                        inputs_normalized=protocol['inputs_normalized'], device=device,
                        use_amp=device.type == 'cuda' and protocol['amp_enabled'],
                        batch_size=protocol['batch_size'], num_workers=config.get('num_workers', 0),
                        speckle=protocol['speckle']), 1):
                    counts += fused_threshold_counts(block['scores'], block['label'], block['valid_mask'], edges)
                    if index % 100 == 0:
                        print(f"    {name} 景{number}/{len(scenes)}：已融合{index}块；耗时{perf_counter()-scene_start:.0f}s", flush=True)
                temporary = destination.with_suffix('.npz.tmp')
                with temporary.open('wb') as stream:
                    np.savez_compressed(stream, counts=counts, thresholds=thresholds)
                temporary.replace(destination)
            if counts.shape != (len(thresholds), 4) or np.any(counts < 0):
                raise ValueError(f'景缓存计数异常：{destination}')
            # 各阈值/各模型的真值正负样本必须完全相同，不能重复计算重叠区域。
            truth = counts[:, [0, 3]] + counts[:, [1, 2]]
            if not np.all(truth == truth[0]) or truth[0].sum() == 0:
                raise ValueError(f"无有效像素或计量范围不一致：{row['scene_id']}")
            previous = denominators.setdefault(row['scene_id'], truth[0].tolist())
            if previous != truth[0].tolist():
                raise ValueError(f"不同模型的有效像素/标签不一致：{row['scene_id']}")
            subsets = ['all'] + (['BC3'] if row['scene_id'].upper().startswith('BC3-') else [])
            for subset in subsets:
                totals[subset] += counts
                numbers[subset] += 1
            for threshold, count in zip(thresholds, counts):
                scene_rows.append(dict(model=name, scene_id=row['scene_id'], group_id=row['group_id'],
                    threshold=threshold, **metrics_from_counts(count)))
            print(f"  {'复用' if reused else '完成'} {name} {number}/{len(scenes)}景：{row['scene_id']}", flush=True)
            write_json(output_dir / 'progress.json', dict(complete=False, model=name,
                model_number=model_number, model_count=len(checkpoints), completed_scenes=number,
                total_scenes=len(scenes), scene_id=row['scene_id'], invocation_seconds=perf_counter()-start))
        for subset, counts in totals.items():
            if numbers[subset]:
                for threshold, count in zip(thresholds, counts):
                    all_rows.append(dict(model=name, epoch=metadata['models'][name]['epoch'], subset=subset,
                        scenes=numbers[subset], threshold=threshold, **metrics_from_counts(count)))
        for subset in ('all', 'BC3'):
            rows = [r for r in all_rows if r['model'] == name and r['subset'] == subset]
            if not rows:
                continue
            best = max(rows, key=lambda r: r['f1'] if np.isfinite(r['f1']) else -1)
            print(f"{name} / {subset}：最佳F1={best['f1']:.4f}，阈值={best['threshold']:.3f}，"
                  f"P={best['precision']:.4f}，R={best['recall']:.4f}", flush=True)
            fixed = next((r for r in rows if r['threshold'] == .5), None)
            if fixed:
                print(f"  0.5复核：F1={fixed['f1']:.4f}，P={fixed['precision']:.4f}，"
                      f"R={fixed['recall']:.4f}，有效像素={fixed['valid_pixels']}", flush=True)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        save_scan_tables(output_dir, all_rows, scene_rows, targets, recall_targets)
    write_json(output_dir / 'progress.json', dict(complete=True, models=list(checkpoints),
        total_scenes=len(scenes), invocation_seconds=perf_counter()-start))
    print(f'扫描完成：{output_dir}；阈值由val选取，泛化仍需独立测试集。', flush=True)
    return output_dir


def main():
    """独立评价入口；既不训练，也不重复执行质量图和 ESA 修正。"""
    if MODE == "scan":
        return run_threshold_scan(image_dir=IMAGE_DIR, label_dir=LABEL_DIR)
    if MODE == "single":
        return evaluate(image_dir=IMAGE_DIR, label_dir=LABEL_DIR)
    raise ValueError("MODE 应为 scan 或 single")


if __name__ == "__main__":
    main()
