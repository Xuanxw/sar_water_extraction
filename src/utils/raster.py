"""通用栅格I/O工具，不包含阈值、地形或监督学习算法。"""

import numpy as np
from rasterio.windows import Window


def file_percentiles(path, count, percentiles):
    """对临时文件内 count 个有效 float32 值求精确线性分位数；就地重排，仅用于临时文件。"""
    ranks, remainder = np.divmod((count - 1) * np.asarray(percentiles, dtype="int64"), 100)
    upper = ranks + (remainder > 0)
    values = np.memmap(path, dtype="float32", mode="r+", shape=(count,))
    try:
        values.partition(np.unique(np.r_[ranks, upper]))
        low = values[ranks].astype("float64")
        return low + (values[upper] - low) * remainder / 100
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
