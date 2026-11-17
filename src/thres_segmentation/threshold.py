"""Otsu/百分位水体阈值；BC复用离线预处理数值，legacy供旧实验。"""
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import uniform_filter


def slope_otsu_rejection(values, valid, slope_deg, otsu_threshold, percentile_threshold,
                         slope_limit_deg=15.0):
    """返回坡度风险排除布尔图，供暗侧Otsu分割和成面共同使用。

    values为滤波后的线性幅度/功率，两个阈值须与它同单位；坡度为同网格的度。
    仅在Otsu比百分位更宽松时收紧坡度>limit区域，等于limit不收紧。
    坡度NaN/Inf保留原判别；这不是几何阴影或叠掩掩膜。已确认SAR无效不标排除。
    将不满足严格阈值的陡坡像元全部保护，避免后续填孔重新生成这些水体。
    """
    if otsu_threshold <= percentile_threshold:
        return np.zeros(values.shape, dtype=bool)
    steep = np.isfinite(slope_deg) & (slope_deg > slope_limit_deg)
    return valid & steep & (values >= percentile_threshold)


def select_otsu_or_percentile(values, percentile_threshold, method="percentile", otsu_bins=1024,
                             otsu_min_separability=0.5, otsu_min_class_fraction=0.001,
                             otsu_histogram_range=None, otsu_domain='linear', otsu_log_floor=1e-6):
    """优先Otsu，退化或不满足统计判据时回退给定的百分位阈值。

    values为一维有限有效值，可传磁盘映射数组；阈值保持输入数值域和单位。
    分块累计整景等宽直方图，最大化两类类间方差/总方差；不逐影像块求阈值。
    可分性和类别占比是运行判据，不代表真值精度或直方图确有两个峰。
    otsu_histogram_range为同单位的[下限,上限]；范围外样本计入边缘bin，
    不删像元、不改原值，阈值仍用于全部有效数据。None保留原全范围算法。
    log_amplitude使用20log10(A)，log_intensity使用10log10(I)，
    估计后均还原为原输入单位；要求非负线性数据，不能输入z-score。
    合法0通过floor参与最低bin，floor不是NoData，不改原值。
    I/O、内存及配置错误直接报错，不伪装成Otsu失败。
    """
    if method not in ('otsu', 'percentile'):
        raise ValueError('threshold.method使用otsu或percentile')
    if otsu_domain not in ('linear', 'log_amplitude', 'log_intensity'):
        raise ValueError('otsu_domain使用linear、log_amplitude或log_intensity')
    log_coefficient = {'log_amplitude': 20., 'log_intensity': 10.}.get(otsu_domain)
    if not np.isfinite(otsu_log_floor) or otsu_log_floor <= 0:
        raise ValueError('otsu_log_floor需为原输入单位下的正有限值')
    report = dict(method_requested=method, method_used='percentile', fallback_reason=None,
                  percentile_threshold=float(percentile_threshold), otsu_threshold=None,
                  otsu_separability=None, otsu_lower_fraction=None, otsu_upper_fraction=None,
                  otsu_bin_width=None, histogram_min=None, histogram_max=None,
                  histogram_clipped_lower_pixels=0, histogram_clipped_upper_pixels=0,
                  input_min=None, input_max=None, histogram_percentiles=None,
                  otsu_domain=otsu_domain, otsu_log_floor=otsu_log_floor if log_coefficient is not None else None,
                  log_coefficient=log_coefficient,
                  log_floor_pixels=0, histogram_axis_min=None, histogram_axis_max=None,
                  otsu_threshold_axis=None, otsu_lower_neighbor=None, otsu_upper_neighbor=None)
    if method == 'percentile':
        return float(percentile_threshold), report
    if (not isinstance(otsu_bins, int) or otsu_bins < 2
            or not 0 <= otsu_min_separability <= 1 or not 0 <= otsu_min_class_fraction < 0.5):
        raise ValueError('Otsu bins需为>=2的整数，可分性范围0～1，最小类别占比范围[0,0.5)')
    values = np.asanyarray(values).reshape(-1)
    if values.size == 0:
        raise ValueError('没有有效SAR像元，不能执行Otsu或百分位回退')
    chunk_size = 1024 * 1024
    low, high = np.inf, -np.inf
    for start in range(0, values.size, chunk_size):
        part = values[start:start + chunk_size]
        if not np.isfinite(part).all():
            raise ValueError('Otsu输入必须仅包含有限有效像元')
        low = min(low, float(part.min()))
        high = max(high, float(part.max()))
    report.update(input_min=low, input_max=high)
    if log_coefficient is not None and low < 0:
        raise ValueError('对数域Otsu要求非负线性幅度/功率，不能输入负值或标准化z-score')
    if otsu_histogram_range is not None:
        if len(otsu_histogram_range) != 2 or not np.isfinite(otsu_histogram_range).all():
            raise ValueError('otsu_histogram_range需为两个有限幅度值')
        lower, upper = map(float, otsu_histogram_range)
        if lower > upper or lower < low or upper > high:
            raise ValueError('Otsu直方图上下限需递增且位于有效数据范围内')
        low, high = lower, upper
    report.update(histogram_min=low, histogram_max=high)
    if high <= low:
        report['fallback_reason'] = 'constant_image' if report['input_min'] == report['input_max'] else 'collapsed_histogram_range'
        return float(percentile_threshold), report
    if not np.isfinite(high - low):
        report['fallback_reason'] = 'nonfinite_intensity_range'
        return float(percentile_threshold), report
    if log_coefficient is not None:
        axis_low, axis_high = log_coefficient * np.log10(np.maximum([low, high], otsu_log_floor))
    else:
        axis_low, axis_high = low, high
    report.update(histogram_axis_min=float(axis_low), histogram_axis_max=float(axis_high))
    if axis_high <= axis_low:
        report['fallback_reason'] = 'collapsed_log_range'
        return float(percentile_threshold), report
    counts = np.zeros(otsu_bins, dtype=np.int64)
    report['otsu_bin_width'] = float((axis_high - axis_low) / otsu_bins)
    for start in range(0, values.size, chunk_size):
        part = values[start:start + chunk_size]
        if log_coefficient is not None:
            report['log_floor_pixels'] += int(np.count_nonzero(part < otsu_log_floor))
        if otsu_histogram_range is not None:
            report['histogram_clipped_lower_pixels'] += int(np.count_nonzero(part < low))
            report['histogram_clipped_upper_pixels'] += int(np.count_nonzero(part > high))
            part = np.clip(part, low, high)
        if log_coefficient is not None:
            part = log_coefficient * np.log10(np.maximum(part, otsu_log_floor))
        counts += np.histogram(part, bins=otsu_bins, range=(axis_low, axis_high))[0]
    # 将bin中心线性映射到0～1，避免幅度平方溢出；线性变换不改变最优划分。
    centers = (np.arange(otsu_bins, dtype=np.float64) + 0.5) / otsu_bins
    probability = counts / counts.sum()
    weight = np.cumsum(probability)[:-1]
    moment = np.cumsum(probability * centers)[:-1]
    mean = float(np.dot(probability, centers))
    variance = float(np.dot(probability, (centers - mean) ** 2))
    eligible = (weight > 0) & (weight < 1)
    if variance <= 0 or not eligible.any():
        report['fallback_reason'] = 'no_two_nonempty_classes'
        return float(percentile_threshold), report
    between = np.full(otsu_bins - 1, -np.inf)
    between[eligible] = (mean * weight[eligible] - moment[eligible]) ** 2 / (weight[eligible] * (1 - weight[eligible]))
    best = int(np.argmax(between))
    # 采用分界bin的右边缘，保证预测的< / >=两类与直方图划分一致。
    threshold_axis = axis_low + (axis_high - axis_low) * ((best + 1) / otsu_bins)
    neighbors = axis_low + (axis_high - axis_low) * (np.array([best, best + 2]) / otsu_bins)
    if log_coefficient is not None:
        threshold = float(10 ** (threshold_axis / log_coefficient))
        neighbors = 10 ** (neighbors / log_coefficient)
    else:
        threshold = threshold_axis
    separability = float(np.clip(between[best] / variance, 0, 1))
    report.update(otsu_threshold=float(threshold) if np.isfinite(threshold) else None, otsu_separability=separability,
                  otsu_lower_fraction=float(weight[best]), otsu_upper_fraction=float(1 - weight[best]),
                  otsu_threshold_axis=float(threshold_axis), otsu_lower_neighbor=float(neighbors[0]),
                  otsu_upper_neighbor=float(neighbors[1]))
    if not np.isfinite(threshold):
        report['fallback_reason'] = 'nonfinite_otsu_threshold'
    elif min(weight[best], 1 - weight[best]) < otsu_min_class_fraction:
        report['fallback_reason'] = 'class_fraction_below_minimum'
    elif separability < otsu_min_separability:
        report['fallback_reason'] = 'separability_below_minimum'
    else:
        report['method_used'] = 'otsu'
        return float(threshold), report
    return float(percentile_threshold), report


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
                 noise_ratio=None, stretch=None, power_to_db=False, preprocessing="legacy",
                 method="percentile", otsu_bins=1024, otsu_min_separability=0.5,
                 otsu_min_class_fraction=0.001, otsu_histogram_percentiles=None):
        self.water_percent, self.mode = water_percent, mode
        self.method = method
        self.otsu_bins = otsu_bins
        self.otsu_min_separability = otsu_min_separability
        self.otsu_min_class_fraction = otsu_min_class_fraction
        self.otsu_histogram_percentiles = otsu_histogram_percentiles
        if (otsu_histogram_percentiles is not None and
                (len(otsu_histogram_percentiles) != 2 or not 0 <= otsu_histogram_percentiles[0] < otsu_histogram_percentiles[1] <= 100)):
            raise ValueError('otsu_histogram_percentiles需为0～100内递增的两个百分位')
        self.threshold_report = {}
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
        finite = np.asarray(valid, dtype=bool) & np.isfinite(score)
        if finite.any():
            threshold = self.select_threshold(score[finite], threshold)
            prediction = (score < threshold if self.mode == 'bottom' else score >= threshold) & finite
        else:
            _, self.threshold_report = select_otsu_or_percentile(np.empty(0), 0.)
            self.threshold_report.update(method_requested=self.method, method_used='not_available',
                                         fallback_reason='no_valid_pixels', percentile_threshold=None)
        return prediction, score, threshold

    def select_threshold(self, values, percentile_threshold, histogram_range=None):
        """数组/整景执行层共用同一Otsu判据；保留本景方法与回退原因供输出。"""
        if self.method == 'otsu' and self.otsu_histogram_percentiles is not None and histogram_range is None:
            # 数组入口可直接计算；大影像执行层传入磁盘精确分位，避免复制整景。
            histogram_range = np.percentile(values, self.otsu_histogram_percentiles)
        threshold, self.threshold_report = select_otsu_or_percentile(
            values, percentile_threshold, self.method, self.otsu_bins,
            self.otsu_min_separability, self.otsu_min_class_fraction, histogram_range)
        self.threshold_report['histogram_percentiles'] = self.otsu_histogram_percentiles
        print(f"阈值方法：{self.threshold_report['method_used']}；阈值={threshold:.6g}；"
              f"回退原因={self.threshold_report['fallback_reason'] or '无'}", flush=True)
        return threshold


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
