"""通用栅格I/O工具，不包含阈值、地形或监督学习算法。"""

from rasterio.windows import Window


def windows(source, size):
    """按行优先遍历不重叠窗口，边缘窗口截短，像元网格保持不变。

    Parameters: source - 含width、height属性的栅格。
                size   - 核心窗口边长，单位像元。
    Yields:     window - Rasterio Window，不含算法邻域halo。
    """
    for row in range(0, source.height, size):
        for col in range(0, source.width, size):
            yield Window(col, row, min(size, source.width - col), min(size, source.height - row))
