"""SNAP Lee滤波配置与缓存读取；config=None关闭滤波。"""
import numpy as np


class SpeckleFilter:
    """config=None 关闭；开启时显式指定线性幅度 amplitude 或线性功率 intensity。"""

    def __init__(self, config=None):
        self.config, self.radius = None, 0
        self.snap = None
        if config is None or config.get('method', 'none') == 'none':
            return
        method = config['method']
        domain = config['input_domain']
        size, enl = config.get('window_size', 3), float(config.get('enl', 1.0))
        if method != 'snap_lee' or domain not in ('amplitude', 'intensity'):
            raise ValueError('滤波支持 snap_lee；input_domain 必须为 amplitude 或 intensity，不能填dB')
        if not isinstance(size, int) or size < 3 or size % 2 != 1 or not np.isfinite(enl) or enl <= 0:
            raise ValueError('Lee 窗口应为不小于3的奇数，ENL应为正数')
        self.config = dict(method=method, input_domain=domain, window_size=size, enl=enl)
        from datasets.snap_preprocess import GPT_PATH, SnapSpeckleFilter, snap_runtime
        runtime = snap_runtime(str(GPT_PATH))
        if 'snap_runtime' in config and config['snap_runtime'] != runtime:
            raise ValueError('检查点的SNAP版本与本机不同，请使用原版本复评/续训')
        if config.get('pipeline_version', 1) != 1:
            raise ValueError('不支持该SNAP预处理版本')
        self.config.update(snap_runtime=runtime, pipeline_version=1, calibration='not_applied')
        self.snap, self.radius = SnapSpeckleFilter(self.config), 0

    def __call__(self, image, valid_mask):
        """输入[C,H,W]与[H,W]有效掩膜；各极化独立滤波，输出保持输入物理表达。"""
        if self.config is None:
            return image
        return self.snap(image, valid_mask)


def read_filtered_window(src, bands, window, speckle_filter):
    """读取已准备的整景SNAP缓存；关闭滤波时直接读取原SAR窗口。"""
    if getattr(speckle_filter, 'snap', None) is not None:
        import rasterio
        selected = tuple(bands) if bands is not None else tuple(src.indexes)
        cached = speckle_filter.snap.prepare_file(src.name, selected, create=False)
        # 缓存波段按selected顺序排列，不再使用原文件波段编号。
        with rasterio.open(cached) as filtered:
            data = filtered.read(window=window, masked=True)
        valid = ~np.ma.getmaskarray(data).any(axis=0) & np.isfinite(data.data).all(axis=0)
        return np.where(valid[None], data.data, 0).astype('float32'), valid

    data = src.read(bands, window=window, masked=True, out_dtype='float32')
    valid = ~np.ma.getmaskarray(data).any(axis=0) & np.isfinite(data.data).all(axis=0)
    return np.where(valid[None], data.data, 0).astype('float32'), valid
