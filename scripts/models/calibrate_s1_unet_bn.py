"""S1 固定切片基线的训练集 BN 校准与完整验证集阈值扫描。

python scripts/models/calibrate_s1_unet_bn.py configs/s1/calibrate_baselines_bn.yaml

run.MODE=thresholds 只扫描原权重；bn（默认）另执行 BN 校准及复扫。
从检查点恢复 VV/DEM 表示、训练统计量和固定划分。BN 按网络顺序校准，
每层使用已校准的上游层，Dropout 始终关闭。阈值在完整内部验证集上选择。
设备由 YAML 指定，使用 FP32；原始模型也重新评价，与历史 CUDA AMP 可有差异。
同一配置再次运行会续用完整的 BN 层和验证块。结果目录包含 calibrated.pth、
threshold_scan.csv、summary.csv 和 per_patch.csv；校准模型供独立推理使用。
"""

import csv
from importlib.metadata import version
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO)]

from scripts.models.calibrate_bn import ChannelStatistics
from configuration import config_argument, load_yaml
from datasets.dfc_sample.s1_dataset import WaterDataset, sample_name
from datasets.dfc_terrain import DFC_TERRAIN_CHANNELS, dfc_terrain_from_checkpoint
from datasets.relative_normalization import RELATIVE_REPRESENTATIONS, training_files_signature, validate_relative_normalization
from supervised.evaluation.metrics import metrics_from_counts
from supervised.evaluation.model_eval import threshold_counts
from supervised.models.checkpoint import load_model
from utils.files import sha256, write_json

MODEL_SOURCES = {
    "sentinel1_water_unet": "src/supervised/models/sentinel1_water_unet.py",
    "sentinel1_water_unetpp": "src/supervised/models/unetpp.py",
    "pvt_deeplabv3plus": "src/supervised/models/pvt.py",
    "ai4g_mobilenetv2_unet_dfc": "src/supervised/experimental/experimental_model_unet_mobilenet.py",
}


def checkpoint_protocol(checkpoint):
    """提取可比较的输入和划分协议；均值/标准差顺序与输入通道一致。"""
    keys = ("bands", "input_channels", "patch_representation", "relative_normalization",
            "inputs_normalized", "speckle", "ignore_index", "train_files", "val_files",
            "split_manifest_sha256", "dem_source", "dem_unit")
    protocol = {key: checkpoint.get(key) for key in keys}
    terrain_mode = dfc_terrain_from_checkpoint(checkpoint)
    if terrain_mode is not None:
        protocol["dfc_terrain"] = checkpoint["dfc_terrain"]
    if protocol["patch_representation"] in RELATIVE_REPRESENTATIONS:
        parameters = validate_relative_normalization(
            protocol["relative_normalization"], protocol["patch_representation"])
        signature = training_files_signature(checkpoint["train_files"], checkpoint["bands"])
        if any(parameters[key] != value for key, value in signature.items()):
            raise ValueError("固定 VV 分位数的训练文件签名与检查点不一致")
        expected_channels = DFC_TERRAIN_CHANNELS if terrain_mode is not None else ("VV", "DEM")
        if tuple(checkpoint.get("input_channels", ())) != expected_channels:
            raise ValueError(f"固定 VV 分位数检查点需要输入通道 {expected_channels}")
        protocol["relative_normalization"] = parameters
    elif protocol["relative_normalization"] is not None:
        raise ValueError("检查点的表示与固定 VV 分位数参数不匹配")
    for key in ("mean", "std"):
        protocol[key] = torch.as_tensor(checkpoint[key]).flatten().tolist()
    return json.loads(json.dumps(protocol))


def prepare_experiment(config):
    """核对指定轮次、相同划分和文件版本，建立无训练增强的数据集。"""
    paths, run = config["paths"], config["run"]
    checkpoints = {}
    reference = None
    if set(paths["CHECKPOINTS"]) != set(run["EXPECTED_EPOCHS"]):
        raise ValueError("CHECKPOINTS 与 EXPECTED_EPOCHS 的别名必须一一对应")
    for name, path in paths["CHECKPOINTS"].items():
        if Path(name).name != name or name in (".", ".."):
            raise ValueError(f"检查点别名须为单个目录名：{name}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if checkpoint["epoch"] != run["EXPECTED_EPOCHS"][name]:
            raise ValueError(f"{path} 实际为第 {checkpoint['epoch']} 轮，请核对配置")
        if (checkpoint["model_name"] not in MODEL_SOURCES
                or checkpoint.get("data_layout") == "full_scene"
                or checkpoint.get("validation_inference")):
            raise ValueError("此入口仅支持 S1 U-Net/U-Net++、PVT DeepLabv3+、MobileNetV2 U-Net 固定切片验证")
        protocol = checkpoint_protocol(checkpoint)
        if reference is not None and protocol != reference:
            raise ValueError("检查点的划分或输入预处理协议不同，不能合并对照")
        reference = protocol
        checkpoints[name] = checkpoint
    if not checkpoints:
        raise ValueError("CHECKPOINTS 不能为空")
    first = next(iter(checkpoints.values()))
    train_names, val_names = first["train_files"], first["val_files"]
    if (not train_names or not val_names or set(train_names) & set(val_names)
            or len(set(train_names)) != len(train_names) or len(set(val_names)) != len(val_names)):
        raise ValueError("检查点的 train/val 名单为空、重复或存在交集")
    samples = run["CALIBRATION_SAMPLES"]
    if samples is not None:
        if not isinstance(samples, int) or not 0 < samples <= len(train_names):
            raise ValueError("CALIBRATION_SAMPLES 应为 null 或训练块数量以内的正整数")
        indices = np.random.default_rng(run["SEED"]).choice(len(train_names), samples, replace=False)
        train_names = [train_names[index] for index in sorted(indices)]

    options = dict(bands=first["bands"], speckle=first.get("speckle"),
                   patch_representation=first.get("patch_representation", "stored"),
                   relative_normalization=first.get("relative_normalization"),
                   dfc_terrain_input=dfc_terrain_from_checkpoint(first))
    image_dir = paths.get("IMAGE_DIR") or Path(first["image_dir"])
    label_dir = paths.get("LABEL_DIR") or Path(first["label_dir"])
    train_data = WaterDataset(image_dir, **options)
    if options["patch_representation"] in RELATIVE_REPRESENTATIONS:
        # 拟合签名始终对应完整训练划分，不能改成 BN 抽样子集，也不能用验证集重新拟合。
        train_data.select_files(first["train_files"])
        train_data.set_relative_normalization(options["relative_normalization"], range(len(train_data)))
    train_data.select_files(train_names)
    val_data = WaterDataset(image_dir, label_dir, **options)
    val_data.select_files(val_names)
    for key in ("IMAGE_DIR", "LABEL_DIR"):
        if paths.get(key) is None:
            directories = {Path(checkpoint[key.lower()]).resolve() for checkpoint in checkpoints.values()}
            if len(directories) != 1:
                raise ValueError(f"检查点数据目录不同，请显式指定 {key}")

    # 记录参与校准的影像和全部验证影像/标签版本；训练标签不参与 BN。
    files = []
    for split, dataset in (("train", train_data), ("val", val_data)):
        for path in dataset.image_paths:
            name = sample_name(path, dataset.image_dir)
            sources = [("image", path)]
            if split == "val":
                key = Path(name).with_suffix("").as_posix().casefold()
                sources.append(("label", dataset.label_paths[key]))
            for kind, source in sources:
                stat = source.stat()
                files.append(dict(split=split, name=name, kind=kind,
                                  size=stat.st_size, mtime_ns=stat.st_mtime_ns))
    code_paths = {"scripts/models/calibrate_s1_unet_bn.py", "scripts/models/calibrate_bn.py",
                  "src/datasets/dfc_sample/s1_dataset.py", "src/supervised/models/checkpoint.py",
                  "src/supervised/evaluation/model_eval.py", "src/supervised/evaluation/metrics.py"}
    code_paths.update(MODEL_SOURCES[item["model_name"]] for item in checkpoints.values())
    if options["patch_representation"] in RELATIVE_REPRESENTATIONS:
        code_paths.add("src/datasets/relative_normalization.py")
    if options["dfc_terrain_input"] is not None:
        code_paths.add("src/datasets/dfc_terrain.py")
    packages = ["torch", "numpy", "rasterio"]
    if any(item["model_name"] == "pvt_deeplabv3plus" for item in checkpoints.values()):
        packages.append("timm")
    if any(item["model_name"] == "ai4g_mobilenetv2_unet_dfc" for item in checkpoints.values()):
        packages.extend(("timm", "segmentation-models-pytorch"))
    metadata = dict(protocol=reference, calibration_files=train_names, files=files,
                    checkpoints={name: sha256(path) for name, path in paths["CHECKPOINTS"].items()},
                    models={name: dict(model_name=item["model_name"], model_kwargs=item["model_kwargs"],
                                       epoch=item["epoch"]) for name, item in checkpoints.items()},
                    algorithm="forward_order_sequential_eval_bn_input_moments_v2", precision="float32",
                    device=run["DEVICE"], batch_size=run["BATCH_SIZE"],
                    cpu_threads=run["CPU_THREADS"], tf32=False,
                    versions={package: version(package) for package in sorted(set(packages))},
                    code={path: sha256(REPO / path) for path in sorted(code_paths)})
    metadata = json.loads(json.dumps(metadata))
    return checkpoints, train_data, val_data, metadata


def model_input(batch, checkpoint, device):
    """复用训练输入约定，返回 FP32 [B,C,H,W]；标准化后将影像无效区置零。"""
    image = batch["image"].to(device=device, dtype=torch.float32)
    if not checkpoint.get("inputs_normalized", False):
        mean = torch.as_tensor(checkpoint["mean"], device=device).view(1, -1, 1, 1)
        std = torch.as_tensor(checkpoint["std"], device=device).view(1, -1, 1, 1)
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or torch.any(std <= 0):
            raise ValueError("检查点的训练均值或标准差无效")
        image = (image - mean) / std
    valid = batch["valid_mask"].to(device)
    image = image.masked_fill(~valid.unsqueeze(1), 0)
    if not torch.isfinite(image).all():
        raise ValueError(f"模型输入含非有限数值：{batch['name']}")
    return image


class ReachedBatchNorm(Exception):
    """当前层统计已取得，提前结束本次前向以省去下游计算。"""

def batchnorm_forward_order(model, example):
    """用一次 eval 前向确定 BN 执行顺序；拒绝共享或未执行的 BN，避免静默误校准。"""
    model.eval()
    layers = {name: layer for name, layer in model.named_modules() if isinstance(layer, nn.BatchNorm2d)}
    if not layers or any(not layer.track_running_stats for layer in layers.values()):
        raise ValueError("模型须含有使用 running statistics 的 BatchNorm2d")
    names, handles = [], []
    try:
        for name, layer in layers.items():
            handles.append(layer.register_forward_pre_hook(lambda module, inputs, key=name: names.append(key)))
        with torch.inference_mode():
            model(example)
    finally:
        for handle in handles:
            handle.remove()
    if len(names) != len(set(names)) or set(names) != set(layers):
        raise ValueError("前向含重复调用或未执行的 BN，当前逐层校准不适用")
    return [(name, layers[name]) for name in names]


def calibrate_layers(model, loader, checkpoint, device, folder, log_every):
    """逐层累计训练特征 E[x]、E[x²]，写入总体均值/方差，只更新 BN 缓冲区。

    先确认实际前向顺序。无效输入采用训练时的置零规则，
    中间特征的统计域包含整个特征图，保持 BN 定义；合法零高程仍参与计算。
    """
    model.eval()
    layers = batchnorm_forward_order(model, model_input(next(iter(loader)), checkpoint, device))
    names = [name for name, _ in layers]
    progress_path = folder / "calibration_progress.pth"
    completed = []
    if progress_path.exists():
        saved = torch.load(progress_path, map_location="cpu", weights_only=True)
        completed = saved["completed_layers"]
        if completed != names[:len(completed)]:
            raise ValueError("BN 续跑层顺序与当前网络不同")
        expected = {f"{name}.{key}" for name in completed for key in ("running_mean", "running_var")}
        if set(saved["buffers"]) != expected:
            raise ValueError("BN 续跑统计量不完整")
        with torch.no_grad():
            for name, value in saved["buffers"].items():
                model.get_buffer(name).copy_(value)
    print(f"BN 校准复用 {len(completed)}/{len(layers)} 层", flush=True)
    for number, (name, layer) in enumerate(layers[len(completed):], len(completed) + 1):
        statistics = ChannelStatistics(layer.num_features, device)

        def collect_and_stop(module, inputs):
            statistics(module, inputs)
            raise ReachedBatchNorm

        handle = layer.register_forward_pre_hook(collect_and_stop)
        started = perf_counter()
        print(f"BN {number}/{len(layers)} {name} 开始，训练批次 {len(loader)}", flush=True)
        try:
            with torch.inference_mode():
                for step, batch in enumerate(loader, 1):
                    try:
                        model(model_input(batch, checkpoint, device))
                    except ReachedBatchNorm:
                        pass
                    else:
                        raise RuntimeError(f"前向未经过目标 BN 层：{name}")
                    if step % log_every == 0 or step == len(loader):
                        print(f"  BN {number}/{len(layers)}：{step}/{len(loader)} 批，"
                              f"耗时 {perf_counter() - started:.1f}s", flush=True)
                if not statistics.pixel_count:
                    raise ValueError("训练校准集没有特征像元")
                statistics.apply_to(layer)
        finally:
            handle.remove()
        if not torch.isfinite(layer.running_mean).all() or not torch.isfinite(layer.running_var).all():
            raise FloatingPointError(f"BN 统计出现 NaN/Inf：{name}")
        completed.append(name)
        buffers = {f"{item}.{key}": model.get_buffer(f"{item}.{key}").detach().cpu().clone()
                   for item in completed for key in ("running_mean", "running_var")}
        pending = progress_path.with_suffix(".pth.partial")
        torch.save(dict(completed_layers=completed, buffers=buffers), pending)
        pending.replace(progress_path)


def save_calibrated(model, source, path, source_path, train_names):
    """另存推理模型及校准来源；检查每个可学习参数与源模型逐元素一致。"""
    for name, parameter in model.named_parameters():
        if not torch.equal(parameter.detach().cpu(), source["model_state_dict"][name]):
            raise RuntimeError(f"校准意外改变了可学习参数：{name}")
    result = dict(source)
    result["model_state_dict"] = {name: value.detach().cpu().clone()
                                  for name, value in model.state_dict().items()}
    result["source_phase_metrics"] = result.get("phase_metrics", {})
    result["source_val_loss"] = result.get("val_loss")
    result["phase_metrics"] = {}
    result["val_loss"] = None
    result["resume_phase"] = "bn_calibrated_inference_only"
    result["bn_calibration"] = dict(method="sequential_train_feature_moments",
                                    source_checkpoint=str(source_path), train_files=train_names,
                                    precision="float32", dropout_enabled=False)
    for key in ("optimizer_state_dict", "scheduler_state_dict", "scaler_state_dict", "rng_state",
                "best_val_loss", "best_val_f1", "best_val_balanced_accuracy"):
        result.pop(key, None)
    pending = path.with_suffix(".pth.partial")
    torch.save(result, pending)
    pending.replace(path)


def scan_validation(model, dataset, checkpoint, thresholds, device, folder, batch_size, log_every, num_workers=0):
    """完整验证集每块只推理一次，分箱同时计算各阈值的 TN/FP/FN/TP，逐块原子落盘。"""
    model.eval()
    folder.mkdir(parents=True, exist_ok=True)
    names = [sample_name(path, dataset.image_dir) for path in dataset.image_paths]
    all_counts = [None] * len(dataset)
    missing = []
    for index, name in enumerate(names):
        path = folder / f"{index:05d}.json"
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved["name"] != name:
                raise ValueError(f"验证缓存样本名不同：{path}")
            counts = np.asarray(saved["counts"], dtype=np.int64)
            if counts.shape != (len(thresholds), 4) or np.any(counts < 0):
                raise ValueError(f"验证缓存计数无效：{path}")
            all_counts[index] = counts
        else:
            missing.append(index)
    print(f"验证复用 {len(dataset) - len(missing)}/{len(dataset)} 块", flush=True)
    loader = DataLoader(Subset(dataset, missing), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    processed = 0
    started = perf_counter()
    with torch.inference_mode():
        for step, batch in enumerate(loader, 1):
            scores = model(model_input(batch, checkpoint, device))
            if scores.ndim != 4 or scores.shape[1] != 1 or not torch.isfinite(scores).all():
                raise ValueError(f"模型输出形状或数值异常：{batch['name']}")
            for item, name in enumerate(batch["name"]):
                index = missing[processed]
                if name != names[index]:
                    raise ValueError("验证 DataLoader 样本顺序改变")
                label = batch["label"][item].to(device)
                valid = batch["valid_mask"][item].to(device) & (label != checkpoint["ignore_index"])
                if torch.any(valid & (label != 0) & (label != 1)):
                    raise ValueError(f"标签有效区包含 0/1 之外的值：{name}")
                label = label.masked_fill(~valid, checkpoint["ignore_index"])
                counts = threshold_counts(scores[item, 0], label, thresholds,
                                          checkpoint["ignore_index"]).cpu().numpy()
                write_json(folder / f"{index:05d}.json", dict(name=name, counts=counts.tolist()))
                all_counts[index] = counts
                processed += 1
            if step % log_every == 0 or step == len(loader):
                print(f"  验证 {len(dataset) - len(missing) + processed}/{len(dataset)} 块，"
                      f"本次耗时 {perf_counter() - started:.1f}s", flush=True)
    result = np.stack(all_counts)
    truth = np.stack((result[..., 0] + result[..., 1], result[..., 2] + result[..., 3]), axis=-1)
    if not np.all(truth == truth[:, :1]) or result[:, 0].sum() == 0:
        raise ValueError("阈值间评价像元不一致或完整验证集无有效像元")
    return names, result


def write_tables(output, results, thresholds, min_precision=0.8):
    """汇总完整验证集；最优阈值按全体像元 F1 选择，并列时优先接近 0.5。"""
    scan_rows, summary_rows, patch_rows, group_rows = [], [], [], []
    precision_rows = []
    reference = None
    for model_name, (names, counts) in results.items():
        truth = np.stack((counts[:, 0, 0] + counts[:, 0, 1],
                          counts[:, 0, 2] + counts[:, 0, 3]), axis=1)
        if reference is not None and (reference[0] != names or not np.array_equal(reference[1], truth)):
            raise ValueError("评价结果的逐块有效像元或真值计数不同")
        reference = (names, truth)
        # 按标签有效区的水体比例分组，所有模型共享分组；不按预测结果分组。
        pixels = truth.sum(axis=1)
        ratio = np.divide(truth[:, 1], pixels, out=np.zeros(len(names), dtype=float), where=pixels > 0)
        groups = {
            "water_0": (pixels > 0) & (ratio == 0),
            "water_0_to_1pct": (ratio > 0) & (ratio <= 0.01),
            "water_1_to_10pct": (ratio > 0.01) & (ratio <= 0.1),
            "water_10_to_50pct": (ratio > 0.1) & (ratio <= 0.5),
            "water_over_50pct": ratio > 0.5,
        }
        rows = [dict(model=model_name, threshold=value, **metrics_from_counts(total))
                for value, total in zip(thresholds, counts.sum(axis=0))]
        scan_rows.extend(rows)
        # 约束 Precision 后最大化 Recall；不可达时留空，不能伪造为零阈值。
        eligible = [row for row in rows if np.isfinite(row['precision'])
                    and np.isfinite(row['recall']) and row['precision'] >= min_precision]
        selected = max(eligible, key=lambda row: (row['recall'], row['f1'],
                                                 -abs(row['threshold'] - 0.5))) if eligible else None
        precision_rows.append(dict(model=model_name, min_precision=min_precision,
                                   status='attainable' if selected else 'unattainable',
                                   **{key: selected[key] if selected else ''
                                      for key in ('threshold', 'precision', 'recall', 'f1', 'iou')}))
        best = max(range(len(rows)), key=lambda i: (rows[i]["f1"], -abs(thresholds[i] - 0.5)))
        for selection, index in (("threshold_0.50", thresholds.index(0.5)), ("best_validation_f1", best)):
            summary_rows.append(dict(selection=selection, **rows[index]))
            for group, selected in groups.items():
                if selected.any():
                    group_rows.append(dict(model=model_name, selection=selection, threshold=thresholds[index],
                                           group=group, patches=int(selected.sum()),
                                           **metrics_from_counts(counts[selected, index].sum(axis=0))))
            for name, patch_counts in zip(names, counts):
                patch_rows.append(dict(model=model_name, name=name, selection=selection,
                                       threshold=thresholds[index], **metrics_from_counts(patch_counts[index])))
        print(f"{model_name}：F1@0.5={rows[thresholds.index(0.5)]['f1']:.4f}；"
              f"最佳阈值={thresholds[best]:.2f}，F1={rows[best]['f1']:.4f}，"
              f"P={rows[best]['precision']:.4f}，R={rows[best]['recall']:.4f}", flush=True)
    # 固定阈值的差异隔离 BN 效应；各自最优阈值的差异另外报告。
    lookup = {(row["model"], row["selection"]): row for row in summary_rows}
    comparison_rows = []
    for name in results:
        if not name.endswith("_original"):
            continue
        alias = name[:-len("_original")]
        if alias + "_calibrated" not in results:
            continue
        for selection in ("threshold_0.50", "best_validation_f1"):
            before, after = (lookup[(alias + suffix, selection)] for suffix in ("_original", "_calibrated"))
            row = dict(checkpoint=alias, selection=selection,
                       original_threshold=before["threshold"], calibrated_threshold=after["threshold"])
            for metric in ("precision", "recall", "f1", "iou"):
                row.update({f"original_{metric}": before[metric], f"calibrated_{metric}": after[metric],
                            f"delta_{metric}_pp": 100 * (after[metric] - before[metric])})
            comparison_rows.append(row)
    for name, rows in (("threshold_scan.csv", scan_rows), ("summary.csv", summary_rows),
                       ("per_patch.csv", patch_rows), ("water_groups.csv", group_rows),
                       ("precision_target.csv", precision_rows),
                       ("bn_comparison.csv", comparison_rows)):
        if not rows:
            continue
        path = output / name
        pending = path.with_suffix(".csv.partial")
        with pending.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        pending.replace(path)


def resolve_training_checkpoints(config):
    """可选：按训练 YAML 选择最近完整实验的 best_f1，复用现有选模核验。

    仅供尚未产生时间戳目录的新损失实验；已有基线用显式检查点固定。
    返回配置文件签名与解析结果，续跑时检查点变化会触发协议检查。
    """
    paths, run = config['paths'], config['run']
    sources = paths.get('TRAIN_CONFIGS', {})
    if not sources:
        return {}
    from scripts.models.predict_best_s1 import candidate
    checkpoints = dict(paths.get('CHECKPOINTS', {}))
    epochs = dict(run.get('EXPECTED_EPOCHS', {}))
    selected = {}
    for alias, path in sources.items():
        if alias in checkpoints or alias in epochs:
            raise ValueError(f'训练配置别名与显式检查点重复：{alias}')
        row, _ = candidate(path)
        checkpoints[alias] = Path(row['checkpoint'])
        epochs[alias] = row['epoch']
        selected[alias] = dict(config=str(path), config_sha256=sha256(path),
                               checkpoint=row['checkpoint'], epoch=row['epoch'])
        print(f"{alias}：完整训练实验最佳 F1，第{row['epoch']}轮，{row['checkpoint']}", flush=True)
    paths['CHECKPOINTS'], run['EXPECTED_EPOCHS'] = checkpoints, epochs
    return selected


def matches_calibration_protocol(saved, current):
    """检查能否续跑 S1 校准，只认可目录整理时已核对的脚本哈希对。"""
    if saved == current:
        return True
    code = saved.get('code', {}).copy()
    if 'scripts/calibrate_s1_unet_bn.py' not in code:
        return False
    moves = json.loads(Path(__file__).with_name('protocol_script_moves.json').read_text(encoding='utf-8'))
    for old_path, move in moves.items():
        if code.get(old_path) == move['before'] and move['path'] not in code:
            del code[old_path]
            code[move['path']] = move['after']
    # 检查点、其他源码、数据、运行参数及新脚本哈希仍须全部相同。
    return dict(saved, code=code) == current


def main(config_path):
    """逐检查点评价原始模型、校准 BN、评价校准模型；各阶段均支持续跑。"""
    config = load_yaml(config_path)
    paths, run = config["paths"], config["run"]
    mode = run.get('MODE', 'bn')
    if mode not in ('thresholds', 'bn'):
        raise ValueError('MODE 应为 thresholds 或 bn')
    min_precision = float(run.get('MIN_PRECISION', 0.8))
    if not np.isfinite(min_precision) or not 0 < min_precision <= 1:
        raise ValueError('MIN_PRECISION 应位于 (0,1]')
    selections = resolve_training_checkpoints(config)
    for key in ("CPU_THREADS", "BATCH_SIZE", "LOG_EVERY"):
        if not isinstance(run[key], int) or run[key] <= 0:
            raise ValueError(f"{key} 必须为正整数")
    device = torch.device(run["DEVICE"])
    if device.type not in ("cpu", "cuda"):
        raise ValueError("DEVICE 支持 cpu 或明确指定的 cuda 设备")
    workers = run.get("NUM_WORKERS", 0)
    if not isinstance(workers, int) or workers < 0:
        raise ValueError("NUM_WORKERS 必须为非负整数")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("配置要求 CUDA，但当前环境不可用；请检查环境或显式使用 CPU")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    torch.set_num_threads(run["CPU_THREADS"])
    interval = run["THRESHOLDS"]
    low, high, step = (float(interval[key]) for key in ("start", "stop", "step"))
    if not 0 < low <= high < 1 or step <= 0:
        raise ValueError("阈值区间须位于 (0,1)，步长须为正")
    thresholds = np.round(np.arange(low, high + step * 0.01, step), 10).tolist()
    if 0.5 not in thresholds or sorted(set(thresholds)) != thresholds:
        raise ValueError("阈值应严格递增且包含 0.5")
    output = paths["OUTPUT_DIR"]
    if any(path.resolve().is_relative_to(output.resolve()) for path in paths["CHECKPOINTS"].values()):
        raise ValueError("源检查点不能位于本次对照输出目录内")
    checkpoints, train_data, val_data, metadata = prepare_experiment(config)
    metadata["thresholds"] = thresholds
    metadata.update(mode=mode, min_precision=min_precision, selected_training_configs=selections)
    if mode == 'thresholds':
        metadata['algorithm'] = 'frozen_weights_full_validation_thresholds_v1'
    if selections:
        metadata['code']['scripts/models/predict_best_s1.py'] = sha256(REPO/'scripts/models/predict_best_s1.py')
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "protocol.json"
    if manifest.exists():
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        if not matches_calibration_protocol(saved, metadata):
            raise ValueError("对照目录的检查点、数据或计算协议已改变，请指定新的 OUTPUT_DIR")
    else:
        if any(output.iterdir()):
            raise ValueError("输出目录已有文件且缺少 protocol.json，请指定新目录")
        write_json(manifest, metadata)
    print(f"设备 {device}，CPU 线程 {run['CPU_THREADS']}，FP32，batch={run['BATCH_SIZE']}\n"
          f"模式 {mode}；BN 校准训练块 {len(train_data) if mode == 'bn' else 0}；"
          f"完整验证 {len(val_data)} 块；输出 {output}", flush=True)
    loader = (DataLoader(train_data, batch_size=run["BATCH_SIZE"], shuffle=False,
                         num_workers=workers, persistent_workers=workers > 0) if mode == 'bn' else None)
    started = perf_counter()
    results = {}
    for name, checkpoint in checkpoints.items():
        folder = output / name
        folder.mkdir(exist_ok=True)
        print(f"{name}/original 完整验证集阈值扫描", flush=True)
        model = load_model(checkpoint).to(device).eval()
        if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
            raise ValueError(f"{name} 源模型含非有限参数或 BN 统计")
        results[f"{name}_original"] = scan_validation(
            model, val_data, checkpoint, thresholds, device, folder / "original",
            run["BATCH_SIZE"], run["LOG_EVERY"], workers)
        write_tables(output, results, thresholds, min_precision)
        if mode == 'thresholds':
            del model
            continue
        destination = folder / "calibrated.pth"
        receipt = folder / "calibration_complete.json"
        if destination.exists() and receipt.exists():
            saved = json.loads(receipt.read_text(encoding="utf-8"))
            if (saved["source_sha256"] != metadata["checkpoints"][name]
                    or saved["checkpoint_sha256"] != sha256(destination)):
                raise ValueError(f"{name} 校准模型已被修改，请使用新的输出目录")
            print(f"{name} 复用已完成的校准模型", flush=True)
        else:
            print(f"{name} 开始训练集 BN 校准", flush=True)
            calibrate_layers(model, loader, checkpoint, device, folder, run["LOG_EVERY"])
            save_calibrated(model, checkpoint, destination, paths["CHECKPOINTS"][name], metadata["calibration_files"])
            write_json(receipt, dict(source_sha256=metadata["checkpoints"][name],
                                    checkpoint_sha256=sha256(destination)))
        del model
        calibrated = torch.load(destination, map_location="cpu", weights_only=True, mmap=True)
        print(f"{name}/calibrated 完整验证集阈值扫描", flush=True)
        model = load_model(calibrated).to(device).eval()
        results[f"{name}_calibrated"] = scan_validation(
            model, val_data, calibrated, thresholds, device, folder / "calibrated",
            run["BATCH_SIZE"], run["LOG_EVERY"], workers)
        del model, calibrated
        write_tables(output, results, thresholds, min_precision)
    write_json(output / "completed.json", dict(models=list(results), validation_patches=len(val_data),
                                               elapsed_seconds=perf_counter() - started))
    print(f"对照完成：{output / 'summary.csv'}", flush=True)
    return output


if __name__ == "__main__":
    main(config_argument())
