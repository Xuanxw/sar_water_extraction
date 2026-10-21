# 水体、建筑与后续研究论文索引

整理日期：2026-10-19。本目录现有以下 9 份 PDF，包含预印本和正式发表版；新增 DAD 正式版已完成文件与页面核验。

统一规则：`年份_模型或专题_任务_来源_版本.pdf`。`Preprint` 与 `Published` 分别表示预印本和正式发表版。

| 文献与本地全文 | 页数 | 正式来源或预印本入口 |
|---|---:|---|
| [DAD 早期预印本](2022_DAD_BinarySegmentation_arXiv_Preprint.pdf) | 14 | [来源](https://arxiv.org/abs/2210.15156) |
| [UANet 建筑提取预印本](2023_UANet_BuildingExtraction_arXiv_Preprint.pdf) | 13 | [来源](https://arxiv.org/abs/2307.12309) |
| [UANet 建筑提取正式版](2024_UANet_BuildingExtraction_TGRS_Published.pdf) | 13 | [来源](https://doi.org/10.1109/TGRS.2024.3361211) |
| [UAFNet：SAR 水体/洪水提取](2024_UAFNet_SARFloodMapping_IGARSS_Published.pdf) | 5 | [来源](https://doi.org/10.1109/IGARSS53475.2024.10641026) |
| [UADPNet：多源光学水体提取](2024_UADPNet_OpticalFloodMapping_IGARSS_Published.pdf) | 5 | [来源](https://doi.org/10.1109/IGARSS53475.2024.10641074) |
| [UA-BCD：建筑变化检测](2025_UABCD_BuildingChangeDetection_ISPRS_Published.pdf) | 17 | [来源](https://doi.org/10.1016/j.isprsjprs.2024.11.017) |
| [DAD：差异感知二分类解码器正式版](2026_DAD_BinarySegmentation_TCSVT_Published.pdf) | 15 | [来源](https://doi.org/10.1109/TCSVT.2025.3612574) |
| [PUGNet：遥感二分类分割](2026_PUGNet_BinarySegmentation_ISPRS_Published.pdf) | 17 | [来源](https://doi.org/10.1016/j.isprsjprs.2026.01.010) |
| [DFC2024 两赛道前两名比赛总结](2026_DFC2024_RapidFloodMapping_JSTARS_Published.pdf) | 15 | [来源](https://doi.org/10.1109/JSTARS.2026.3652462) |

## DAD 正式版核验

[DAD，TCSVT 正式论文](2026_DAD_BinarySegmentation_TCSVT_Published.pdf) 已下载，与[作者官方引用记录](https://github.com/Henryjiepanli/DAD)一致：

- 题名：Toward Complex Backgrounds: A Unified Difference-Aware Decoder for Binary Segmentation。
- 作者：Jiepan Li、Wei He、Fangxiao Lu、Hongyan Zhang。
- 卷期：IEEE Transactions on Circuits and Systems for Video Technology，2026 年 2 月，36(2)，2372–2386。
- DOI：`10.1109/TCSVT.2025.3612574`；[IEEE 入口](https://ieeexplore.ieee.org/document/11175179)。2025-09-22 在线发表，2026 年正式卷期，因此文件名使用 2026 是正确的。
- 文件大小：4,420,485 字节。15 页全部可解析、可渲染，印刷页码从 2372 到 2386 连续，正文、图表、参考文献和作者简介齐全；未发现缺页、空白页或明显截断。

保留 2022 年早期预印本用于版本比较；后续研究正式 DAD 时优先使用本次核验的 TCSVT 正式版。

## 版本与阅读说明

- UANet 预印本和正式版均保留，作者顺序/题名存在差异；以正式 TGRS 版作为当前主要依据。
- DAD 早期预印本早于 UANet；不能把期刊发表较晚理解为 UANet 的直接升级。
- UAFNet 是 SAR Track 1；UADPNet 是多源光学 Track 2；两者和建筑 UANet 不应混称。
- 当前项目优先阅读 UAFNet、2026 比赛总结、UANet 正式版，再看 PUGNet 与 UA-BCD。
- GeoFloodNet 为相关最新工作，[作者项目](https://github.com/Henryjiepanli/GeoFlood-275)标注仍在评审，不在本目录已有全文中。

原有 8 份 PDF 重命名时逐文件核对了 SHA256 一致。新增 DAD 正式版仅作只读核验，没有修改 PDF 内容。

## 作者公开代码

- [UANet](https://github.com/Henryjiepanli/Uncertainty-aware-Network)
- [UA-BCD](https://github.com/Henryjiepanli/UA-BCD)
- [PUGNet](https://github.com/Henryjiepanli/PU_RS)
- [DAD](https://github.com/Henryjiepanli/DAD)

作者身份及论文列表：[作者主页](https://henryjiepanli.github.io/Jiepanli_Henry.github.io/)。
