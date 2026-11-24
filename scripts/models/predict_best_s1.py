"""对比多个 DFC S1 训练配置，按内部验证水体 F1 选模并预测密云 S1 GRD dB 整景。

python scripts/models/predict_best_s1.py MODEL1.yaml MODEL2.yaml [MODEL3.yaml MODEL4.yaml ...] GRD目录或单景文件 [DEM米制栅格或GLO-30瓦片目录]

该入口只接收已支持的秩表示权重。训练集统一分位缩放的 DFC 存储值模型，
在取得 DFC 与 Sigma0 dB 的明确数值映射前，不能直接用于此 dB 整景入口。
"""
import json
import math
from datetime import datetime
from pathlib import Path
import sys

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from configuration import load_yaml
from datasets.relative_normalization import RELATIVE_REPRESENTATIONS, training_files_signature, validate_relative_normalization
from supervised.models.checkpoint import load_model
from supervised.evaluation.s1_grd import predict_grd_scene


def training_settings(path):
    # 训练配置快照原样复制到结果目录，其中相对路径仍须按原配置位置解释。
    with path.open(encoding="utf-8-sig") as stream:
        saved = yaml.safe_load(stream)
    return {section: saved[section] for section in ("model", "training", "data")}


def candidate(config_path):
    cfg = load_yaml(config_path)
    names = cfg["run"]["RESULT_FILES"]
    expected = training_settings(config_path)
    root = cfg["paths"]["OUTPUT_DIR"]
    if root is None or cfg["data"]["DATA_LAYOUT"] != "patch":
        raise ValueError(f"{config_path} 需要独立 OUTPUT_DIR 的 DFC 切片训练配置")
    for path in sorted(root.glob("*/" + names["last"]), reverse=True):
        snapshots = sorted(path.parent.glob("train_config_*.yaml"))
        if not snapshots or training_settings(snapshots[-1]) != expected:
            continue
        # 使用 CPU 文件映射，仅查看元数据，不将每个候选的优化器状态复制进内存。
        final = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if (final["epoch"] != cfg["training"]["EPOCHS"] or final["resume_phase"] != "epoch_end"
                or Path(final["image_dir"]).resolve() != cfg["paths"]["IMAGE_DIR"].resolve()
                or Path(final["label_dir"]).resolve() != cfg["paths"]["LABEL_DIR"].resolve()):
            continue
        expected_f1 = final["best_val_f1"]
        del final
        best_path = path.with_name(names["best_f1"])
        best = torch.load(best_path, map_location="cpu", weights_only=True, mmap=True)
        metrics = best["phase_metrics"]["val"]
        if not math.isfinite(metrics["f1"]) or expected_f1 != metrics["f1"]:
            raise ValueError(f"{best_path} 与最后一轮记录的最优 F1 不一致")
        result = dict(config=str(config_path), checkpoint=str(best_path), model=best["model_name"],
                      epoch=best["epoch"], **{k: metrics[k] for k in ("f1", "iou", "precision", "recall")})
        protocol = {key: best[key] for key in (
            "split_manifest_sha256", "train_files", "val_files", "bands", "polarizations", "seed",
            "epochs", "batch_size", "validation_inference", "inputs_normalized", "speckle",
        )}
        protocol["patch_representation"] = best.get("patch_representation")
        protocol["relative_normalization"] = best.get("relative_normalization")
        if protocol["patch_representation"] in RELATIVE_REPRESENTATIONS:
            parameters = validate_relative_normalization(
                protocol["relative_normalization"], protocol["patch_representation"])
            signature = training_files_signature(best["train_files"], best["bands"])
            if any(parameters[key] != value for key, value in signature.items()):
                raise ValueError(f"{best_path} 的固定 VV 分位数与训练文件签名不一致")
            protocol["relative_normalization"] = parameters
        for key in ("mean", "std"):
            protocol[key] = torch.as_tensor(best[key]).flatten().tolist()
        protocol["input_channels"] = tuple(best.get("input_channels", best["polarizations"]))
        protocol["dem_source"] = best.get("dem_source")
        protocol["dem_unit"] = best.get("dem_unit")
        protocol["input_dir"] = cfg["paths"]["IMAGE_DIR"].resolve()
        protocol["label_dir"] = cfg["paths"]["LABEL_DIR"].resolve()
        protocol["training"] = cfg["training"]
        return result, protocol
    raise FileNotFoundError(f"{config_path.name} 尚无匹配的完整 {cfg['training']['EPOCHS']} 轮实验：{root}")


def main(configs, image_dir, dem_path=None):
    if len(configs) < 2:
        raise ValueError("请提供至少两份 S1 训练 YAML")
    candidates = [candidate(Path(path).resolve()) for path in configs]
    rows = [row for row, _ in candidates]
    if len({row["model"] for row in rows}) != len(configs):
        raise ValueError("每份配置必须对应不同的参评网络")
    protocol = candidates[0][1]
    if any(other != protocol for _, other in candidates[1:]):
        raise ValueError("参评实验的训练设置、输入或验证划分不同，不能直接按 F1 选模")
    if protocol["patch_representation"] in RELATIVE_REPRESENTATIONS:
        raise ValueError("训练集 P02/P98 参数属于 DFC 存储值，不能直接应用于 Sigma0 dB 整景；"
                         "请先核实 DFC 到 dB 的数值映射，再准备同输入单位的推理数据。"
                         "当前可在 DFC 固定验证集进行评价和 BN/阈值对照。")
    source = Path(image_dir).resolve()
    if source.is_dir() and (source / "db").is_dir():
        source = source / "db"
    if source.is_file() and source.suffix.lower() in (".tif", ".tiff"):
        images = [source]
    else:
        images = sorted(p for p in source.glob("*") if p.name.lower().endswith(("_db.tif", "_db.tiff")))
    if not images:
        raise FileNotFoundError(f"未找到 *_db.tif 浮点 dB 整景：{source}")
    channels = protocol["input_channels"]
    representation = protocol["patch_representation"]
    if protocol["inputs_normalized"] or protocol["speckle"]:
        raise ValueError("此整景入口需要未预先归一化和未预滤波的 DFC 训练配置")
    if channels == ("VV", "DEM") and representation == "vv_rank_dem_m":
        if dem_path is None or not Path(dem_path).exists():
            raise FileNotFoundError("VV+DEM 权重需要指定米制 DEM 或 Copernicus GLO-30 瓦片目录")
        dem_path = Path(dem_path).resolve()
    elif set(channels) != {"VV", "VH"} or representation != "relative_backscatter_ranks" or dem_path is not None:
        raise ValueError(f"输入波段和预处理不匹配：{channels}，{representation}")
    # F1 相同时取命令行中靠前的配置；目标影像不参与选模或阈值调整。
    winner = max(rows, key=lambda r: r["f1"])
    for row in rows:
        print(f"{row['model']} 内部验证 F1={row['f1']:.5f}，第 {row['epoch']} 轮", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(winner["checkpoint"], map_location="cpu", weights_only=True, mmap=True)
    model = load_model(checkpoint).eval().to(device)
    folder = Path(winner["checkpoint"]).parent / "s1_inference" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder.mkdir(parents=True)
    report = dict(candidates=rows, winner=winner["model"], checkpoint=winner["checkpoint"],
                  input_channels=list(channels), dem=str(dem_path) if dem_path else None,
                  dem_source=protocol["dem_source"], dem_unit=protocol["dem_unit"],
                  images=[str(p) for p in images], threshold=0.5, split_manifest_sha256=protocol["split_manifest_sha256"],
                  patch_representation=protocol["patch_representation"], polarizations=list(protocol["polarizations"]))
    (folder / "selection.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"使用 {winner['model']} 在 {device} 推理，输出 {folder}", flush=True)
    for index, image in enumerate(images, 1):
        mask = folder / (image.stem + "_water.tif")
        print(f"{index}/{len(images)} {image.name} → {mask.name}", flush=True)
        predict_grd_scene(model, image, mask, mean=checkpoint["mean"], std=checkpoint["std"],
                          patch_representation=protocol["patch_representation"], bands=protocol["bands"],
                          threshold=0.5, device=device, dem_path=dem_path)
    return folder


def parse_arguments(arguments):
    """读取连续的训练 YAML，随后为影像路径和可选 DEM；不按固定模型数量切片。"""
    count = 0
    while count < len(arguments) and Path(arguments[count]).suffix.lower() in (".yaml", ".yml"):
        count += 1
    if count < 2 or len(arguments) - count not in (1, 2):
        raise ValueError(__doc__)
    return arguments[:count], arguments[count], arguments[count + 1] if count + 1 < len(arguments) else None


if __name__ == "__main__":
    try:
        arguments = parse_arguments(sys.argv[1:])
    except ValueError as error:
        raise SystemExit(str(error)) from error
    main(*arguments)
