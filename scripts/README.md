# 辅助脚本

从仓库根目录运行下表中的脚本。训练、整景推理和正式阈值分割分别从 `src/supervised/train.py`、`src/supervised/predict.py` 和 `src/thres_segmentation/main.py` 启动。

| 目录 | 脚本 | 用途 |
|---|---|---|
| [data](data/) | [download_s1s2_water](data/download_s1s2_water.py) | 下载 S1S2-Water 官方分卷，支持续传和 MD5 校验 |
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

查看下载分卷的大小和校验值：

```powershell
python scripts/data/download_s1s2_water.py --list
```

旧下载命令 `python scripts/download_s1s2_water.py` 仍接受相同参数。继续下载时指定原输出目录。旧模型评估入口 [evaluate.py](evaluate.py) 调用 [model_eval.py](../src/supervised/evaluation/model_eval.py)。
