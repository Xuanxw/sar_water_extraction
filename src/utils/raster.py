"""通用栅格I/O工具，不包含阈值、地形或监督学习算法。"""

import numpy as np
from rasterio.windows import Window


def file_percentiles(path, count, percentiles, *, transform=None):
    """对临时文件中的有效 float32 值就地求分位数，返回 float64。

    transform 可选，仅适用于单调非降的逐像元变换。先变换相邻秩值再线性插值，
    等价于在变换后的全部像元上求分位数；调用方须先检查全部像元的定义域。
    """
    ranks, remainder = np.divmod((count - 1) * np.asarray(percentiles, dtype="int64"), 100)
    upper = ranks + (remainder > 0)
    values = np.memmap(path, dtype="float32", mode="r+", shape=(count,))
    try:
        values.partition(np.unique(np.r_[ranks, upper]))
        low = values[ranks].astype("float64")
        high = values[upper].astype("float64")
        if transform is not None:
            low = transform(low)
            high = transform(high)
        return low + (high - low) * remainder / 100
    finally:
        del values


def windows(source, size):
    """按行优先遍历不重叠窗口，边缘窗口截短，像元网格保持不变。

    Parameters: source - 含width、height属性的栅格。
                size   - 核心窗口边长，单位像元。
    Yields:     window - Rasterio Window，不含算法邻域halo。
    """
    for row in range(0, source.height, size):
        for col in range(0, source.width, size):
            yield Window(col, row, min(size, source.width - col), min(size, source.height - row))
