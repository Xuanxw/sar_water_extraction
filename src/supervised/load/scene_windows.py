"""整景推理窗口：沿用训练的上下文和补零规则，不依赖标签决定是否预测。"""
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window
import torch
from torch.utils.data import Dataset
from datasets.speckle import SpeckleFilter, read_filtered_window


class SceneWindows(Dataset):
    """遍历单景全部核心区，每个像素只输出一次；标签可为空。"""
    def __init__(self, image_path, label_path=None, bands=(1,), window_size=1024, halo=128, speckle=None):
        self.image_path = Path(image_path)
        self.label_path = Path(label_path) if label_path else None
        self.bands, self.size, self.halo = tuple(bands), window_size, halo
        self.core = window_size - 2 * halo
        if self.core <= 0 or halo < 0 or window_size % 32:
            raise ValueError('窗口应为32的倍数且核心区大于0')
        self.filter = SpeckleFilter(speckle)
        self.image = self.label = self.env = None
        with rasterio.open(self.image_path) as src:
            self.height, self.width, self.profile = src.height, src.width, src.profile.copy()
            if self.label_path:
                with rasterio.open(self.label_path) as label:
                    if (src.shape, src.crs, src.transform) != (label.shape, label.crs, label.transform):
                        raise ValueError(f'影像与标签网格不一致：{self.image_path}')
        self.windows = [(top, left, min(self.core, self.height-top), min(self.core, self.width-left))
                        for top in range(0, self.height, self.core)
                        for left in range(0, self.width, self.core)]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        if self.image is None:
            self.image = rasterio.open(self.image_path)
            self.label = rasterio.open(self.label_path) if self.label_path else None
        top, left, h, w = self.windows[index]
        x0, y0 = left-self.halo, top-self.halo
        x1, y1 = max(0, x0), max(0, y0)
        x2, y2 = min(self.width, x0+self.size), min(self.height, y0+self.size)
        window = Window(x1, y1, x2-x1, y2-y1)
        data, good = read_filtered_window(self.image, self.bands, window, self.filter)
        image = np.zeros((len(self.bands), self.size, self.size), np.float32)
        valid = np.zeros((self.size, self.size), bool)
        label = np.full((self.size, self.size), -1, np.int64)
        ys, xs = slice(y1-y0, y2-y0), slice(x1-x0, x2-x0)
        image[:, ys, xs], valid[ys, xs] = data, good
        if self.label is not None:
            target = self.label.read(1, window=window, masked=True)
            usable = good & ~np.ma.getmaskarray(target) & np.isin(target.data, (0, 1))
            label[ys, xs] = np.where(usable, target.data.astype(np.int64), -1)
        # 完整窗口标签用于边界距离；计分及写盘只用核心区。
        return dict(image=torch.from_numpy(image), valid_mask=torch.from_numpy(valid),
                    label=torch.from_numpy(label), window=torch.tensor((top, left, h, w)))

    def close(self):
        for src in (self.image, self.label):
            if src is not None:
                src.close()
        self.image = self.label = None
        if self.env is not None:
            self.env.__exit__(None, None, None)
            self.env = None
