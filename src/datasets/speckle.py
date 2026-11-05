"""在线 Lee 斑点滤波：只处理内存数组，不改写 SAR 文件，也不做辐射校正。

对照 SNAP S1TBX 的 Lee.getLeeValue / SpeckleFilter.getVarianceValue：
https://github.com/senbox-org/s1tbx/blob/master/s1tbx-op-sar-processing/src/main/java/org/esa/s1tbx/sar/gpf/filtering/SpeckleFilters/Lee.java
这里按局部统计公式独立实现 NumPy/SciPy 版本，不复制 Java 源码。
固定 ENL；样本方差使用 n-1；边界/无效邻域不参与统计。不是 Refined Lee。
本项目额外提供幅度→功率→滤波→幅度转换，不声称与 SNAP 整个算子逐像素等价。
"""
import numpy as np


class SpeckleFilter:
    """config=None 关闭；开启时显式指定线性幅度 amplitude 或线性功率 intensity。"""

    def __init__(self, config=None):
        self.config, self.radius = None, 0
        if config is None or config.get('method', 'none') == 'none':
            return
        method = config['method']
        domain = config['input_domain']
        size, enl = config.get('window_size', 3), float(config.get('enl', 1.0))
        if method != 'lee' or domain not in ('amplitude', 'intensity'):
            raise ValueError('滤波只支持 lee；input_domain 必须明确为 amplitude 或 intensity，不能填 dB')
        if not isinstance(size, int) or size < 3 or size % 2 != 1 or not np.isfinite(enl) or enl <= 0:
            raise ValueError('Lee 窗口应为不小于3的奇数，ENL应为正数')
        self.config = dict(method=method, input_domain=domain, window_size=size, enl=enl)
        self.radius = size // 2

    def __call__(self, image, valid_mask):
        """输入[C,H,W]与[H,W]有效掩膜；各极化独立滤波，输出保持输入物理表达。"""
        if self.config is None:
            return image
        from scipy.ndimage import uniform_filter

        valid = np.asarray(valid_mask, dtype=bool)
        if not np.isfinite(image[:, valid]).all() or (image[:, valid] < 0).any():
            raise ValueError('Lee 输入必须是有限、非负的线性 SAR 值，请在标准化/dB转换前调用')
        size, enl = self.config['window_size'], self.config['enl']
        # 零是合法观测；只用显式掩膜剔除无效值。图像外视为缺测，不复制边缘。
        counts = np.rint(uniform_filter(valid.astype(np.float64), size, mode='constant') * size**2)
        denominator = np.maximum(counts, 1)
        result = np.zeros_like(image, dtype=np.float32)
        for channel, values in enumerate(image):
            power = np.where(valid, values, 0).astype(np.float64)
            if self.config['input_domain'] == 'amplitude':
                power = np.square(power)
            mean = uniform_filter(power, size, mode='constant') * size**2 / denominator
            second_sum = uniform_filter(power**2, size, mode='constant') * size**2
            variance = np.maximum(second_sum - counts * mean**2, 0) / np.maximum(counts - 1, 1)
            # SNAP 普通 Lee 的权重：w=max(0,1-mean²/(ENL*variance))。
            # 匀质区域趋向局部均值；高变异区域保留更多原像素。ENL越大平滑越弱。
            noise_ratio = np.ones_like(mean)
            np.divide(mean**2 / enl, variance, out=noise_ratio, where=variance > 0)
            weight = np.clip(1 - noise_ratio, 0, 1)
            filtered = np.maximum(mean + weight * (power - mean), 0)
            if self.config['input_domain'] == 'amplitude':
                filtered = np.sqrt(filtered)
            result[channel] = np.where(valid, filtered, 0)
        return result


def read_filtered_window(src, bands, window, speckle_filter):
    """多读滤波半径，再裁回请求窗口；整景分块结果与整幅滤波保持相同邻域。"""
    from rasterio.windows import Window

    x, y, w, h = map(int, (window.col_off, window.row_off, window.width, window.height))
    radius = speckle_filter.radius
    left, top = max(0, x-radius), max(0, y-radius)
    right, bottom = min(src.width, x+w+radius), min(src.height, y+h+radius)
    data = src.read(bands, window=Window(left, top, right-left, bottom-top), masked=True, out_dtype='float32')
    valid = ~np.ma.getmaskarray(data).any(axis=0) & np.isfinite(data.data).all(axis=0)
    image = speckle_filter(np.where(valid[None], data.data, 0), valid)
    ys, xs = slice(y-top, y-top+h), slice(x-left, x-left+w)
    return image[:, ys, xs], valid[ys, xs]
