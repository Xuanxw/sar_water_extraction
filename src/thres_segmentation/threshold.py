"""共用百分位阈值：BC复用离线预处理数值；legacy仅供旧DFC/GUI实验。"""
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import uniform_filter


def percentile_mask(values, valid, water_percent=25, mode="bottom"):
    """原脚本规则：暗水体 < 第 q 百分位，亮水体 >= 第 100-q 百分位。"""
    if mode not in ("bottom", "top") or not 0 <= water_percent <= 100:
        raise ValueError("mode 使用 bottom/top，water_percent 范围 0～100")
    valid = np.asarray(valid, dtype=bool) & np.isfinite(values)
    if not valid.any():
        return np.zeros(values.shape, dtype=bool), float("nan")
    threshold = float(np.percentile(values[valid], water_percent if mode == "bottom" else 100 - water_percent))
    prediction = values < threshold if mode == "bottom" else values >= threshold
    return prediction & valid, threshold


class ThresholdSegmenter:
    """DFC 适配：文件有效掩膜优先；保留合法零值，不从标签估计当前图水体面积。"""

    def __init__(self, water_percent=25, mode="bottom", lee_window=None,
                 noise_ratio=None, stretch=None, power_to_db=False, preprocessing="legacy"):
        self.water_percent, self.mode = water_percent, mode
        if preprocessing not in ("legacy", "offline_normalized"):
            raise ValueError("preprocessing使用offline_normalized或legacy")
        self.preprocessing = preprocessing
        if preprocessing == "offline_normalized":
            if any(value is not None for value in (lee_window, noise_ratio, stretch)) or power_to_db:
                raise ValueError("离线归一化成果不能再次滤波、拉伸或转dB，请移除旧预处理参数")
            self.lee_window, self.noise_ratio = 1, 0.0
            self.stretch, self.power_to_db = None, False
        else:
            # 旧实验明确保留原算法，避免历史阈值比较被悄悄更换。
            self.lee_window = 5 if lee_window is None else lee_window
            self.noise_ratio = .12 if noise_ratio is None else noise_ratio
            self.stretch = (1, 99) if stretch is None else stretch
            self.power_to_db = power_to_db

    def prepare(self, image, valid):
        """离线模式直接返回输入；legacy才执行旧Lee近似及逐景拉伸。"""
        valid = np.asarray(valid, dtype=bool) & np.isfinite(image)
        if not valid.any():
            return np.full(image.shape, np.nan)
        working = np.asarray(image, dtype=np.float64).copy()
        if self.preprocessing == "offline_normalized":
            # 保留z-score负值和合法0；这里只屏蔽无效像素，不重新估计均值/标准差。
            working[~valid] = np.nan
            return working
        working[~valid] = np.median(working[valid])
        if self.lee_window > 1:
            # mirror 对应 OpenCV 默认边缘反射；双精度减少相减计算局部方差的舍入误差。
            mean = uniform_filter(working, self.lee_window, mode="mirror")
            variance = np.maximum(uniform_filter(working ** 2, self.lee_window, mode="mirror") - mean ** 2, 0)
            noise = np.var(working[valid]) * self.noise_ratio
            weight = variance / (variance + noise + 1e-10)
            working = mean + weight * (working - mean)
        if self.power_to_db:
            if np.any(working[valid] < 0):
                raise ValueError("10log10 仅供已确认的非负线性功率；DFC 默认保留存储域")
            working = 10 * np.log10(working + 1e-6)
        low, high = np.percentile(working[valid], self.stretch)
        score = np.clip((working - low) / (high - low + 1e-6), 0, 1)
        score[~valid] = np.nan
        return score

    def predict(self, image, valid):
        """返回二值掩膜、归一化强度和本图阈值；强度不是水体概率。"""
        score = self.prepare(image, valid)
        prediction, threshold = percentile_mask(score, valid, self.water_percent, self.mode)
        return prediction, score, threshold


def read_input_preprocessing(data_root, segmenter, bands):
    """核对离线完成记录；不读取标签、不估计val统计量，也不启动SNAP。"""
    root = Path(data_root)
    receipt = root / "preprocessing.json"
    record = json.loads(receipt.read_text(encoding="utf-8")) if receipt.is_file() else {}
    normalized = record.get("stage") == "normalized"
    if segmenter.preprocessing == "legacy":
        if normalized:
            raise ValueError("输入已经归一化，不能使用legacy二次处理；请用offline_normalized")
        return {}
    if (not normalized or not record.get("complete") or
            (root / ".preprocessing_incomplete").exists() or (root / ".labels_rebuilding").exists()):
        raise ValueError(f"离线归一化成果尚未完成：{root}；请先运行image_process.py的preprocess模式")
    mean = np.asarray(record.get("mean", []), dtype=float)
    std = np.asarray(record.get("std", []), dtype=float)
    if (record.get("statistics_protocol") != "fixed_train_windows_only" or
            mean.ndim != 1 or mean.size == 0 or std.shape != mean.shape or
            not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any()):
        raise ValueError("缺少有效的训练集归一化统计，不能改用val逐景统计")
    if not bands or min(bands) < 1 or max(bands) > len(mean):
        raise ValueError("选择的波段超出离线归一化成果的通道数")
    return record
