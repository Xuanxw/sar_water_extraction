"""共用百分位阈值与 Lee 去斑；不依赖 GUI/GDAL，不使用标签生成预测。"""
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

    def __init__(self, water_percent=25, mode="bottom", lee_window=5,
                 noise_ratio=.12, stretch=(1, 99), power_to_db=False):
        self.water_percent, self.mode = water_percent, mode
        self.lee_window, self.noise_ratio = lee_window, noise_ratio
        self.stretch, self.power_to_db = stretch, power_to_db

    def prepare(self, image, valid):
        """局部均值/方差 Lee 近似，沿用原全局方差×noise_ratio；非自适应 ENL 估计。"""
        valid = np.asarray(valid, dtype=bool) & np.isfinite(image)
        if not valid.any():
            return np.full(image.shape, np.nan)
        working = np.asarray(image, dtype=np.float64).copy()
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
