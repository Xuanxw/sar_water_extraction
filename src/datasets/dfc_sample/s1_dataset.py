"""将 SAR 栅格和同名标签读取为 PyTorch 张量。"""
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader, Dataset
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from datasets.speckle import SpeckleFilter, read_filtered_window

# 直接修改这里的变量，然后运行 python src/supervised/datasets/water_dataset.py。
DATA_ROOT = Path("E:/Code_projects/sar_water_extraction/data")
IMAGE_DIR = Path("E:/Code_projects/sar_water_extraction/data/Track1/train/images")
LABEL_DIR = Path("E:/Code_projects/sar_water_extraction/data/Track1/train/labels")  # 无标签数据填写 None。
BANDS = None       # None 读取全部波段；例如 (1, 2) 读取第 1、2 波段，编号从 1 开始。
BATCH_SIZE = 2
IGNORE_INDEX = -1  # 内部忽略标记；后续损失函数使用同一个 ignore_index。
EXTENSIONS = {".tif", ".tiff", ".png", ".img", ".vrt"}


def sample_name(path, root):
    """保留稳定样本名；固定划分的 train/val 是存储分区，不是样本编号的一部分。"""
    root = Path(root)
    relative = Path(path).relative_to(root)
    if ((root.parent / "split_manifest.csv").is_file() and len(relative.parts) > 1
            and relative.parts[0] in ("train", "val")):
        relative = Path(*relative.parts[1:])
    return relative.as_posix()


def resolve_sample_path(root, name):
    """兼容旧检查点的裸文件名；只查原路径和两个明确分区，不做模糊匹配。"""
    root, name = Path(root), Path(name)
    direct = root / name
    if direct.is_file():
        return direct
    candidates = [root / split / name for split in ("train", "val")
                  if (root / split / name).is_file()]
    if len(candidates) != 1:
        raise FileNotFoundError(f"样本不存在或存在多个同名候选：{root / name}")
    return candidates[0]


class WaterDataset(Dataset):
    """按需读取一个影像及标签，返回 image、valid_mask、name 和可选的 label。

    image 为 float32 [C,H,W]，label 为 int64 [H,W]。
    valid_mask 为 bool [H,W]，表示所选影像波段均有限且文件掩膜有效。
    保留原始存储值和标签编码，不应用 scale/offset，不推断水体类别。
    """

    def __init__(self, image_dir, label_dir=None, bands=None, speckle=None):
        """建立文件列表；标签按相对路径和不含扩展名的文件名配对。"""
        self.image_dir = Path(image_dir)
        self.bands = bands
        self.speckle_filter = SpeckleFilter(speckle)
        self.image_paths = sorted(path for path in self.image_dir.rglob("*")
                                  if path.is_file() and path.suffix.lower() in EXTENSIONS)
        self.label_paths = None
        if label_dir is not None:
            label_dir = Path(label_dir)
            self.label_paths = {}
            for path in label_dir.rglob("*"):
                if path.is_file() and path.suffix.lower() in EXTENSIONS:
                    key = Path(sample_name(path, label_dir)).with_suffix("").as_posix().casefold()
                    if key in self.label_paths:
                        raise ValueError(f"存在同名标签，不能确定配对：{key}")
                    self.label_paths[key] = path

    def select_files(self, names):
        """按检查点或固定清单选择文件；名单顺序、历史样本名保持不变。"""
        self.image_paths = [resolve_sample_path(self.image_dir, name) for name in names]


    def __len__(self):
        """返回影像数量，供 DataLoader 确定迭代长度。"""
        return len(self.image_paths)

    def __getitem__(self, index):
        """读取一个样本；坏文件或缺失标签直接保留库的异常，方便定位。"""
        image_path = self.image_paths[index]
        with rasterio.open(image_path) as src:
            from rasterio.windows import Window
            image, valid = read_filtered_window(src, self.bands, Window(0, 0, src.width, src.height), self.speckle_filter)
        sample = {
            "image": torch.from_numpy(image),
            "valid_mask": torch.from_numpy(valid),
            "name": sample_name(image_path, self.image_dir),
        }

        if self.label_paths is not None:
            key = Path(sample["name"]).with_suffix("").as_posix().casefold()
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
