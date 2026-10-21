"""将 SAR 栅格和同名标签读取为 PyTorch 张量。"""
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader, Dataset

# 直接修改这里的变量，然后运行 python src/datasets/water_dataset.py。
DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
IMAGE_DIR = DATA_ROOT / "Track1/train/images"
LABEL_DIR = DATA_ROOT / "Track1/train/labels"  # 无标签数据填写 None。
BANDS = None       # None 读取全部波段；例如 (1, 2) 读取第 1、2 波段，编号从 1 开始。
BATCH_SIZE = 2
IGNORE_INDEX = -1  # 内部忽略标记；后续损失函数使用同一个 ignore_index。
EXTENSIONS = {".tif", ".tiff", ".png", ".img", ".vrt"}


class WaterDataset(Dataset):
    """按需读取一个影像及标签，返回 image、valid_mask、name 和可选的 label。

    image 为 float32 [C,H,W]，label 为 int64 [H,W]。
    valid_mask 为 bool [H,W]，表示所选影像波段均有限且文件掩膜有效。
    保留原始存储值和标签编码，不应用 scale/offset，不推断水体类别。
    """

    def __init__(self, image_dir, label_dir=None, bands=None):
        """建立文件列表；标签按相对路径和不含扩展名的文件名配对。"""
        self.image_dir = Path(image_dir)
        self.bands = bands
        self.image_paths = sorted(path for path in self.image_dir.rglob("*")
                                  if path.is_file() and path.suffix.lower() in EXTENSIONS)
        self.label_paths = None
        if label_dir is not None:
            label_dir = Path(label_dir)
            self.label_paths = {
                path.relative_to(label_dir).with_suffix("").as_posix().casefold(): path
                for path in label_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in EXTENSIONS
            }

    def __len__(self):
        """返回影像数量，供 DataLoader 确定迭代长度。"""
        return len(self.image_paths)

    def __getitem__(self, index):
        """读取一个样本；坏文件或缺失标签直接保留库的异常，方便定位。"""
        image_path = self.image_paths[index]
        with rasterio.open(image_path) as src:
            data = src.read(indexes=self.bands, out_dtype="float32", masked=True)

        # 只根据文件掩膜和 NaN/Inf 判断有效性，不把 0 或 -9999 自动当作无效值。
        valid = np.isfinite(data.data).all(axis=0) & ~np.ma.getmaskarray(data).any(axis=0)
        image = data.filled(0)
        image[:, ~valid] = 0  # 无效位置占位为 0，有效性仍由 valid_mask 单独记录。
        sample = {
            "image": torch.from_numpy(image),
            "valid_mask": torch.from_numpy(valid),
            "name": image_path.relative_to(self.image_dir).as_posix(),
        }

        if self.label_paths is not None:
            key = image_path.relative_to(self.image_dir).with_suffix("").as_posix().casefold()
            with rasterio.open(self.label_paths[key]) as src:
                label_data = src.read(1, masked=True)
            # 标签自身无效或影像无效的位置不参与后续损失计算。
            label_valid = valid & ~np.ma.getmaskarray(label_data) & np.isfinite(label_data.data)
            label = np.full(label_data.shape, IGNORE_INDEX, dtype=np.int64)
            label[label_valid] = label_data.data[label_valid]
            sample["label"] = torch.from_numpy(label)
        return sample



if __name__ == "__main__":
    dataset = WaterDataset(IMAGE_DIR, LABEL_DIR, BANDS)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    print(f"样本数量：{len(dataset)}")
    print(f"文件：{batch['name']}")
    print(f"影像：{batch['image'].shape}，{batch['image'].dtype}")
    print(f"影像有效像素：{batch['valid_mask'].sum().item()}/{batch['valid_mask'].numel()}")
    for index, band in enumerate(batch["image"].unbind(dim=1), 1):
        values = band[batch["valid_mask"]]
        if values.numel():
            print(f"输入通道 {index}：min={values.min().item()}，max={values.max().item()}")
    if "label" in batch:
        print(f"标签：{batch['label'].shape}，{batch['label'].dtype}")
        print(f"标签值：{torch.unique(batch['label']).tolist()}")
