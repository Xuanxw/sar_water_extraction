# 固定基线与验证使用说明

当前结构收束到 **PVT-UAFNet + 固定空洞率 MBDC + 全局困难采样 + 最终头残差 + 四头等权 BCE**。验证入口仍默认指向历史检查点 `outputs/pvt_uafnet_hard_sampling_residual/20261022_160255_467390/best_f1.pth`（第20轮），用于复核旧实验。固定分组控制基线已完成：`outputs/pvt_uafnet_hard_sampling_residual_grouped/20261027_212638_633640/best_f1.pth`（第17轮，F1=79.8582%）。后续对照使用该模型，不能把旧模型直接视为新 val 上的公平基线。头监督加权、尺度 MBDC 的已有分数保留为历史参考。

## 新划分下的控制与改良顺序

固定清单是 `data/Track1/train/split_manifest.csv`；文件保存在 `images/train`、`images/val` 及对应 `labels/train`、`labels/val`。物理 train/val 分别为 **1314/317 对**，真正参加训练/验证的 `active=1` 样本为 **1144/287 对**。停用的重复副本和标签冲突记录保留在目录中，因此不能把扫描目录得到的全部数量当成有效训练/验证数量。

1. PVT 控制组使用 `METHOD="uafnet"`、`UAFNET_BACKBONE="pvt_v2_b2.in1k"`、`UAFNET_HIGH_RESOLUTION_REFINEMENT=False`、`UAFNET_MBDC_DILATIONS=None`，写入 `outputs/pvt_uafnet_hard_sampling_residual_grouped/时间戳/`。
2. 当前下一轮回到 `UAFNET_BACKBONE="pvt_v2_b2.in1k"`，只启用 `UAFNET_HIGH_RESOLUTION_REFINEMENT=True`，执行 `python src/train.py`，写入 `outputs/pvt_uafnet_hard_sampling_residual_highres_grouped/时间戳/`。两组均采用同一 ImageNet 初始化流程重新训练，固定 MBDC、四头等权 BCE、采样和其他训练参数不变；不从控制组 best_f1.pth 续训。ResNet50 骨干对照已完成，最佳 F1=75.6857%，暂不作为下一轮主基线。
3. 将本页验证入口的 `CHECKPOINTS` 改为上述两次实验的 `best_f1.pth`，保留 `USE_SAVED_VAL=True`。脚本从各自检查点读取相同287张名单与训练归一化参数；先比较原始 F1/IoU/P/R，再看内侧边界 Recall、外侧 FPR 和双侧带 F1。传统阈值对比脚本也改为对应新检查点；若复用预测掩膜，必须使用该模型、该阈值的新 raw 输出。

验证新模型时保持 `USE_SAVED_VAL=True`、`IMAGE_DIR=None`、`LABEL_DIR=None`，从检查点恢复287个 active 验证名单。不要改成 `USE_SAVED_VAL=False` 直接扫描物理 `images/val`：该目录有317张，其中30张是停用副本/冲突记录，不能一起计入新基线指标。

`train.py` 的 IMAGE_DIR/LABEL_DIR 仍指向 images/labels 父目录；训练器按清单选择 active 样本，仅用训练样本计算均值/标准差，不重新随机划分。每次实验保存清单快照及 SHA256，便于核对两轮数据一致。高分辨率分支只是待验证的细节恢复假设，当前不声称它提升精度。

旧检查点的裸文件名可继续解析移动后的数据，并复核原326张历史验证结果。**不能给旧模型换成新287张名单后，就称为新的独立验证**：新 val 可能包含旧模型曾训练过的样本。重新训练消除这一跨划分问题，但当前场景分类和重复组隔离仍不能证明事件/地理独立；没有可靠区域编号时，仍不启用区域95% CI。

下文的326/262/258数量及默认旧检查点，均描述历史验证协议；新实验应以新检查点保存的名单和清单快照为准。

## 运行入口

在项目根目录、sar-water 环境中运行：

```powershell
python src/validation.py
```

先结束正在运行的训练，再执行完整 GPU 验证。配置集中在文件顶部，无 argparse。`train.py` 负责训练及训练后的 quality；`validation.py` 负责固定检查点的质量核验；`evaluation/model_eval.py` 保留开发阶段的阈值扫描。`thres_seg/main.py` 仍是独立传统阈值流程。模型与传统法的共同评价放在 `evaluation/compare_threshold.py`，不向 thres_seg 注入检查点依赖；当前传统法是原有 Lee 去斑加固定百分位，并未新增 Otsu。

当前 `train.py` 的默认配置已恢复固定 MBDC（`UAFNET_MBDC_DILATIONS=None`），其余训练参数保持不变；已启动的训练进程不会因配置修改而中途切换骨干。`validation.py` 明确加载上面的历史残差基线，二者互不覆盖。已有检查点可直接复核历史结果；上面的新固定划分对照使用已完成的 grouped PVT 控制组。

## 默认协议

- 使用检查点保存的内部验证名单，保持顺序；历史默认模型为326张，新分组实验为287张，不会重新划分。
- 使用保存的输入波段及训练集 mean/std，采用 eval + inference_mode，只读取最终头。
- 概率阈值固定 0.5，batch size=2，CUDA AMP；不会扫描阈值、校准 BN 或更新参数。
- 不使用 ESA/WorldCover 补水，报告网络本身的精度。辅助波段仍可在质量图中并排显示。
- 每次单独运行一个模型；在 CHECKPOINTS 中追加候选即逐个评估，不做概率集成。
- 指标和图采用同一概率判断。与旧的直接 logit 比较在阈值附近可能有极少量浮点差异。

## 数据重复分组

脚本读取实际输入波段及有效掩膜计算 SHA256，而不是简单沿用旧的重复名单。当前数据分为：

| subset | 当前数量 | 用途 |
|---|---:|---|
| all | 326 | 复核历史训练日志与完整验证表现 |
| duplicate_in_train | 64 | 明确暴露与训练整幅重复的乐观表现 |
| no_exact_train_duplicate | 262 | 未发现与训练整幅重复的验证子集 |
| unique_nontrain | 258 | 在上一组内对同输入同标签去重，并排除同输入异标签组 |

258 的来源：262 张中有两对相同输入且相同标签，各保留一张；1277/1040 这一对输入相同但标签不一致，两张都不参与这个敏感性分组。完整 326 张的结果始终保留，原标签和训练划分不改变。标签内容比较使用解码像元与掩膜，文件字节指纹另外用于核对场景清单。

**去掉整幅重复不代表空间/事件独立。** 当前切片缺少 CRS、日期和可靠区域编号，不能排除邻近或跨时相重叠；且这些验证影像已反复参与选模。262/258 张上的成绩仍是开发验证，而非新建立的独立测试集。未来严格实验需要重复组与真实事件/区域共同划分，再从头训练，或者获得未参与训练和调参的新区域标签。不能把训练过的图重新划到测试集后沿用原模型，称作独立测试。

## 结果目录

`outputs/validation/quality/时间戳/` 保存：

- `samples.csv`：验证名单、输入/标签指纹、重复标记及自动场景。
- `comparison.csv`：各检查点四个分组的总体与逐图统计。
- `Residual/validation_metrics.csv`：单模型相同汇总。
- `Residual/validation_per_image.csv`：逐图完整混淆计数与指标。
- `Residual/scene_metrics.csv`：整组及 unique_nontrain 的场景表现。场景来自已有 WorldCover 辅助自动分类，不是地理区域，也不是河流/湖泊类型真值。
- `Residual/worst_images.csv`：按有定义的 F1 排列的最差 20 幅，供人工复核；不据此自动删图。
- `Residual/boundary_groups.csv`：按重复分组统计 1/3/5 像素内侧带、外侧带、核心区及双侧带。
- `Residual/boundary/`：完整带内像素指标、固定采样点及边界示意图。
- `Residual/raw/`：二值预测 TIFF，0=非水体，1=水体，255=无效。无地理参考的源切片不会凭空得到真实坐标。
- `Residual/figures/`：默认前 10 幅最终质量图，包含 SAR、辅助层、GT、概率和误差。设 `MAX_VISUALIZATIONS=None` 可输出全部；这不是中间特征图。
- `Residual/protocol.csv`：检查点、轮数、阈值、数据路径和运行口径。

复用现有 quality 流程还会保留原始 `metrics.csv`、`per_image.csv` 和 `postprocessed/`。本入口关闭了后处理，单模型的 member_1/ensemble/postprocessed 与 raw 完全相同；不能把这些行/面板理解为不同算法或精度改善。看标准化后的 `validation_metrics.csv` 即可。

## 指标如何阅读

整体指标先相加 TN/FP/FN/TP，再计算 IoU、F1、Precision、Recall；`macro_*` 是每幅有定义指标的等权均值，`std_*` 是总体标准差（ddof=0），`defined_*` 给出有效样本数，`worst_*_name` 标识最低值样本。无水且无预测时水体 F1/IoU 为 NaN，不人为记为 1；无预测时 Precision 沿用项目 0 的约定。各指标有定义的样本集合可能不同。

内侧/核心主要看 Recall，外侧主要看 FPR，双侧带可看 F1。这是缓冲带内严格像素精度，**不是带容差的 Boundary F1**。当前距离单位是像素。固定采样点用于复核；主统计采用全部有效带内像素，避免两侧定额抽样改变类别比例。

## 真正按区域验证

取得可靠地理区域或事件对应关系后，设置 REGION_CSV，其格式为：

```csv
name,region_id
region_A/tile_01.tif,event_A
region_A/tile_02.tif,event_A
region_B/tile_01.tif,event_B
```

name 必须与输入目录相对文件名完全一致，并覆盖全部待评估影像。同一个事件的相邻切片应归到同组，不能直接用每张切片编号当作独立区域。脚本先汇总区域内混淆计数，输出 per_region.csv，再等权报告各区域均值、标准差与最差区域。

只有确认各区域独立、没有参与训练/调参时才设 `REGIONS_ARE_INDEPENDENT=True`。脚本此时按区域有放回抽样 2000 次，每次带上该区域全部像素混淆计数，给出整体水体 IoU 的 percentile 95% 区间。少量区域的区间很不稳定；这个开关是经过数据审查后的声明，脚本不能从无坐标切片证明区域独立。目前默认关闭，不生成伪造的 patch 独立性区间。

外部有标签集合：设置 `USE_SAVED_VAL=False` 并填写 IMAGE_DIR/LABEL_DIR。仍用训练归一化参数，源训练影像必须可读以检查整幅重复。需先确认极化、辐射校正、存储尺度、分辨率和网格一致；这里不包含 GAMMA/SLC 预处理或跨传感器适配。现有 data/images 没有参考标签，只能预测和人工查看，不能计算 F1。


## 模型与传统阈值：统一评价协议

运行 `python src/evaluation/compare_threshold.py`。默认对比固定 Residual 模型与 VV、VH 各自的 Lee + 第10百分位暗水体分割，不应用土地覆盖补水，不用标签挑水体百分比，也不从两个波段结果中自动挑最好者。

对比使用检查点内的 `val_files`，其影像物理位置仍为 `data/Track1/train/images`，标签位于同名 labels 目录。这些图是从有标签池留出的内部验证子集，不是梯度训练使用的 `train_files`。官方 `data/Track1/val/images` 缺少参考标签，不能因文件夹名为 val 就在本地计算 F1/IoU。比赛官网明确 validation/test 不随影像提供参考标签，并以 F1 进行服务器评估：[DFC2024 官方说明](https://www.grss-ieee.org/community/technical-committees/2024-ieee-grss-data-fusion-contest/)。

所有方法必须使用相同文件、共同有效标签像素、相同半径的边界区域、同一份固定采样点。整体主要看水体 F1、IoU、Precision、Recall；同时保留 OA 等完整指标。局部主要看内侧带 Recall、外侧带 FPR、双侧带 F1，以及核心水体 Recall。传统法读取当前影像的直方图求百分位阈值，不是用标签训练；但是人工选择百分位、滤波窗口等超参数仍属于开发调参，不能反复看最终测试标签来选取。

`MODEL_MASK_DIR=None` 时从检查点重新推理；填写原始 raw 掩膜目录时，复用已有预测，不占用模型推理 GPU。缓存模式严格核对网格与有效位置上的 0/1 编码，缺少预测就报错，不能为某种方法偷偷减少评价像元。缓存必须来自指定检查点与固定概率阈值，不能把 ESA 修正后的 postprocessed 掩膜填入；脚本不能从 TIFF 自证模型来源，仍须核对原实验记录。缓存模式没有原模型概率，只比较二值掩膜与真实标签。

结果进入 `outputs/threshold_comparison/quality/时间戳/`；整体和逐图指标、重复分组、边界区域与固定点的结果分表保存。没有可靠区域编号时不伪造 IoU 置信区间。

## F1、IoU 和 95% 置信区间如何使用

对同一个水体正类、同一有效像素集合的汇总混淆矩阵：

$$
IoU=\frac{TP}{TP+FP+FN},\qquad
F1=\frac{2TP}{2TP+FP+FN}=\frac{2IoU}{1+IoU}.
$$

两者是严格单调对应关系，选模排序相同。保留 F1 作为主指标有利于与比赛口径衔接，同时报告更严格的 IoU。这里的等价关系不保证逐图分别求均值的 macro F1 与 macro IoU 仍给出相同排序。F1 的定义见 [scikit-learn 官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)。

目前关注漏检恢复，所以不能只看 F1 上升：还要检查 Precision 约束下的 Recall、少水体/不同场景的 Recall，以及边界内侧是否恢复真实水体。整体面积精度提高，也不自动代表岸线形状改善。训练集指标主要检查收敛与拟合程度，泛化结论来自留出的验证/测试；反复参与选模的验证集最终还需要独立测试区域核验。[交叉验证与测试集说明](https://scikit-learn.org/stable/modules/cross_validation.html)。

95% CI 描述估计的不确定性，不是合格阈值，也不是要求 IoU 达到95%。例如 `IoU=0.67，95% CI=[0.62,0.71]` 是“点估计+区间”，无需再判断0.67是否落在由它构造的区间内。按同样抽样程序反复构造区间，95%的覆盖率描述这些区间覆盖目标总体参数的频率；不是对某次已算出的固定区间赋予真值落入的95%概率。[NIST 置信区间说明](https://itl.nist.gov/div898/handbook/prc/section1/prc14.htm)。

今后取得可靠的近似独立区域/事件映射后，比较网络与阈值法应按**同一批区域索引成对重抽样**，每次汇总各自TP/FP/FN，计算 `ΔIoU=IoU_网络−IoU_阈值`，再给出差值的95% CI。差值区间全部大于0才支持该抽样假设下的正向优势；跨0表示证据不足，不代表方法等效。不要用两个独立CI是否重叠替代差值推断。[SciPy paired bootstrap说明](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)。

当前脚本的区域 Bootstrap 是固定模型的区域抽样不确定性，不包含重新训练的随机性。CI不能修复训练重复、标签误差或反复调参偏差；近邻patch与像素也不能直接当独立样本以获得看似很窄的区间。场景类别“农田/城市”是分组分析标签，不是独立地理区域编号。


阈值对照入口现已固定当前分组清单：运行 `src/evaluation/compare_threshold.py` 前填写新 grouped 检查点，程序核对清单与名单后只评价物理 val 中的287张 active 图。`src/thres_seg/main.py` 也已按相同清单过滤，可先独立运行传统分割。
