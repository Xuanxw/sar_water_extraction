# 辅助脚本

从仓库根目录运行下表中的脚本。训练、整景推理和正式阈值分割分别从 `src/supervised/train.py`、`src/supervised/predict.py` 和 `src/thres_segmentation/main.py` 启动。

| 目录 | 脚本 | 用途 |
|---|---|---|
| [data](data/) | [download_s1s2_water](data/download_s1s2_water.py) | 下载 S1S2-Water 官方分卷，或按景下载 Sentinel-1，支持断点续传和文件校验 |
| | [classify_scenes](data/classify_scenes.py)、[prepare_train_val_split](data/prepare_train_val_split.py) | 按土地覆盖归类 DFC 样本，按场景与重复组划分训练和验证集 |
| | [audit_normalized_scenes](data/audit_normalized_scenes.py) | 核查归一化后 BC 整景的像元分布 |
| [models](models/) | [calibrate_bn](models/calibrate_bn.py)、[calibrate_s1_unet_bn](models/calibrate_s1_unet_bn.py) | 重估 BN 统计，或用 S1 固定验证集比较校准与阈值 |
| | [predict_best_s1](models/predict_best_s1.py) | 按验证 F1 选择模型，对 S1 GRD 整景预测 |
| [threshold](threshold/) | [base_threshold_scan](threshold/base_threshold_scan.py)、[history_threshold_comparison](threshold/history_threshold_comparison.py)、[slope_threshold_comparison](threshold/slope_threshold_comparison.py) | 分别比较基础分位数、历史水体阈值和坡度阈值 |
| | [inspect_otsu_outliers](threshold/inspect_otsu_outliers.py)、[evaluate_threshold_review](threshold/evaluate_threshold_review.py) | 检查影响 Otsu 的亮值，或用人工判读点估计精度 |
| | [repair_gui_terrain](threshold/repair_gui_terrain.py) | 用正式阈值流程补处理已有 ORG 成果 |
| [terrain](terrain/) | [compare_gui_terrain_direction](terrain/compare_gui_terrain_direction.py)、[compare_terrain_support](terrain/compare_terrain_support.py) | 核对视向与坡向约定，比较 DEM 支持范围和高程包络 |
| | [audit_terrain_disagreements](terrain/audit_terrain_disagreements.py)、[verify_gc_map2_v57_scan](terrain/verify_gc_map2_v57_scan.py) | 定位地形掩膜差异，在 Linux 或 WSL 核验 gc_map2 v5.7 扫描步骤 |
| [products](products/) | [preview_org_coverage](products/preview_org_coverage.py)、[export_water_polygons](products/export_water_polygons.py)、[merge_water_shp](products/merge_water_shp.py) | 查看裁剪前覆盖、补做逐景水体面、按日期拼接水体面 |

### 下载 S1S2-Water v1.0.1

只需要 Sentinel-1 时可以先下载第 1 景。每景包含 SAR 影像（`img`）、水体标签（`msk`）、有效区掩膜（`valid`）和元数据。需要高程时加 `--dem`。

```powershell
python -u scripts/data/download_s1s2_water.py "G:/水体数据/S1/S1S2-Water-v1.0.1" --s1 1
```

省略 `--s1` 后面的景号会下载全部 S1 场景。`--s1 1 59 --list` 可先查看两景的文件和总大小。结果保存在输出目录的景号子目录中。镜像固定到同一提交，大文件使用镜像清单的 SHA-256 校验，Git 直接保存的小文件按对象 ID 校验。[按景浏览和下载](https://huggingface.co/datasets/idomogalla/s1s2_water/tree/main)

完整数据使用作者的六个 ZIP 分卷。已有 `.zip.part` 时，在原目录重跑原命令即可续传。下面的命令只下载目录文件和第一个分卷。[Zenodo 官方分卷下载](https://zenodo.org/records/11278238)

```powershell
python -u scripts/data/download_s1s2_water.py "G:/水体数据/S1/S1S2-Water-v1.0.1" part1
```

两种模式都可用 Ctrl+C 中断，重新运行会校验已完成的文件，从 `.part` 已写入的位置继续。每次断线仍有新字节写入时会自动续传。连续五次无进展的临时故障会退出并保留进度。网页手动下载完成的同名文件也可放到对应位置，由脚本校验。按景文件与 Zenodo 的 ZIP 分卷分别保存进度。

旧下载入口 `python scripts/download_s1s2_water.py` 支持相同参数。旧模型评估入口 [evaluate.py](evaluate.py) 调用 [model_eval.py](../src/supervised/evaluation/model_eval.py)。
