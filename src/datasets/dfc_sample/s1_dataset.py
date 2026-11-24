"""将 SAR 栅格和同名标签读取为 PyTorch 张量。"""
from pathlib import Path
from numbers import Integral
from tempfile import TemporaryDirectory

import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader, Dataset
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from datasets.speckle import SpeckleFilter, read_filtered_window
from datasets.relative_normalization import (
    LOG_REPRESENTATION, RELATIVE_REPRESENTATIONS, apply_vv_train_p02_p98_dem_m,
    relative_log10_power, relative_normalization_protocol, training_files_signature, validate_relative_normalization,
)
from utils.raster import file_percentiles

# 直接修改这里的变量，然后运行 python src/supervised/datasets/water_dataset.py。
DATA_ROOT = Path("E:/Code_projects/sar_water_extraction/data")
IMAGE_DIR = Path("E:/Code_projects/sar_water_extraction/data/Track1/train/images")
LABEL_DIR = Path("E:/Code_projects/sar_water_extraction/data/Track1/train/labels")  # 无标签数据填写 None。
BANDS = None       # None 读取全部波段；例如 (1, 2) 读取第 1、2 波段，编号从 1 开始。
BATCH_SIZE = 2
IGNORE_INDEX = -1  # 内部忽略标记；后续损失函数使用同一个 ignore_index。
EXTENSIONS = {".tif", ".tiff", ".png", ".img", ".vrt"}


def relative_backscatter_ranks(image, valid):
    """每个极化独立计算有效像元的经验中秩；同值同秩，掩膜外填零。"""
    image = np.asarray(image)
    valid = np.asarray(valid, dtype=bool)
    if image.ndim != 3 or valid.shape != image.shape[1:]:
        raise ValueError("需要 [C,H,W] 影像和同网格 [H,W] 掩膜")
    result = np.zeros(image.shape, dtype=np.float32)
    if not valid.any():
        return result
    for channel, band in enumerate(image):
        values = band[valid]
        if not np.isfinite(values).all():
            raise ValueError("有效掩膜内的后向散射值必须为有限数")
        _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
        mids = (counts.cumsum() - 0.5 * counts) / values.size
        result[channel, valid] = mids[inverse]
    return result


def vv_rank_dem_m(image, valid):
    """共同有效域上转换 VV 中秩，第二通道保留地表高程（米）。"""
    image = np.asarray(image)
    valid = np.asarray(valid, dtype=bool)
    if image.ndim != 3 or image.shape[0] != 2 or valid.shape != image.shape[1:]:
        raise ValueError("VV + DEM 需要同网格的 [2,H,W] 影像和 [H,W] 掩膜")
    if not np.isfinite(image[1, valid]).all():
        raise ValueError("有效掩膜内的 DEM 高程必须为有限数")
    result = np.empty(image.shape, dtype=np.float32)
    result[:1] = relative_backscatter_ranks(image[:1], valid)
    result[1] = np.where(valid, image[1], 0)
    return result


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
    默认读取原始存储值；relative_backscatter_ranks 按有效像元计算逐波段中秩。标签编码原样读取。
    """

    def __init__(self, image_dir, label_dir=None, bands=None, speckle=None, patch_representation="stored",
                 relative_normalization=None):
        """建立文件列表；标签按相对路径和不含扩展名的文件名配对。"""
        if patch_representation not in ("stored", "relative_backscatter_ranks", "vv_rank_dem_m", *RELATIVE_REPRESENTATIONS):
            raise ValueError(f"未知切片输入表示：{patch_representation}")
        self.patch_representation = patch_representation
        self.bands = bands
        self.relative_normalization = None
        if relative_normalization is not None:
            self.set_relative_normalization(relative_normalization)
        self.image_dir = Path(image_dir)
        self.speckle_filter = SpeckleFilter(speckle)
        if patch_representation in ("vv_rank_dem_m", *RELATIVE_REPRESENTATIONS):
            if bands is None or len(bands) != 2 or bands[0] == bands[1]:
                raise ValueError("VV + DEM 需要两个不同的文件波段，依次为 VV 和 DEM")
            if self.speckle_filter.config is not None:
                raise ValueError("含 DEM 的切片不能对所有通道执行斑点滤波")
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

    def read_image(self, index):
        """只读原始所选通道和共同有效域；DEM 转成米值，不读标签或执行 VV 变换。"""
        image_path = self.image_paths[index]
        with rasterio.open(image_path) as src:
            from rasterio.windows import Window
            image, valid = read_filtered_window(src, self.bands, Window(0, 0, src.width, src.height), self.speckle_filter)
            if self.patch_representation in ("vv_rank_dem_m", *RELATIVE_REPRESENTATIONS):
                dem_index = self.bands[1] - 1
                image[1] = image[1] * src.scales[dem_index] + src.offsets[dem_index]
        return image, valid

    def normalization_training_signature(self, indices):
        """按相对文件名计算训练子集签名；与读取顺序和数据根目录无关。"""
        indices = list(indices)
        if not indices or any(isinstance(i, bool) or not isinstance(i, Integral) or not 0 <= i < len(self) for i in indices):
            raise ValueError("拟合归一化需要非空、有效的训练索引")
        if len(set(indices)) != len(indices):
            raise ValueError("拟合归一化的训练索引不能重复")
        names = [sample_name(self.image_paths[i], self.image_dir) for i in indices]
        return training_files_signature(names, self.bands)

    def set_relative_normalization(self, parameters, indices=None):
        """恢复固定参数；提供训练索引时核验来源，续跑不得静默重新估计。"""
        if self.patch_representation not in RELATIVE_REPRESENTATIONS:
            raise ValueError("只有训练集共享分位数表示使用此归一化参数")
        parameters = validate_relative_normalization(parameters, self.patch_representation)
        current_bands = tuple(self.bands) if self.bands is not None else ()
        if tuple(parameters["bands"]) != current_bands:
            raise ValueError("归一化参数的 VV/DEM 波段顺序与当前输入不同")
        if indices is not None:
            signature = self.normalization_training_signature(indices)
            if any(parameters[key] != value for key, value in signature.items()):
                raise ValueError("归一化参数的训练文件划分与当前实验不同")
        self.relative_normalization = parameters
        return parameters

    def fit_relative_normalization(self, indices):
        """用训练索引内全部共同有效 VV 像元拟合 P02/P98；不读取或使用标签。

        各块原始 VV 的 float32 值写入临时文件；对数方案在秩相邻值变换后插值，
        内存只保留当前块；验证块和 DEM 数值不进入分位数统计，DEM 掩膜参与有效域。
        """
        if self.patch_representation not in RELATIVE_REPRESENTATIONS:
            raise ValueError("只有训练集共享分位数表示需要拟合训练分位数")
        log_input = self.patch_representation == LOG_REPRESENTATION
        indices = list(indices)
        signature = self.normalization_training_signature(indices)
        count = 0
        with TemporaryDirectory(prefix="sar_vv_percentiles_") as directory:
            path = Path(directory) / "vv.float32"
            with path.open("wb") as stream:
                for position, index in enumerate(indices, 1):
                    image, valid = self.read_image(index)
                    if not np.isfinite(image[:, valid]).all():
                        raise ValueError(f"有效 VV/DEM 含非有限值：{self.image_paths[index]}")
                    values = np.asarray(image[0, valid], dtype=np.float32)
                    if log_input and np.any(values < 0):
                        raise ValueError(f"对数输入要求有效 VV 存储值非负：{self.image_paths[index]}")
                    values.tofile(stream)
                    count += values.size
                    if position == 1 or position % 100 == 0 or position == len(indices):
                        print(f"训练集 VV 分位数：{position}/{len(indices)} 块，累计 {count:,} 个有效像元", flush=True)
            if count == 0:
                raise ValueError("训练集没有共同有效的 VV/DEM 像元，无法拟合分位数")
            transform = relative_log10_power if log_input else None
            lower, upper = file_percentiles(path, count, [2, 98], transform=transform)
        parameters = relative_normalization_protocol(self.patch_representation)
        parameters.update(p02=float(lower), p98=float(upper), fit_pixels=count, **signature)
        return self.set_relative_normalization(parameters, indices)

    def __getitem__(self, index):
        """读取一个样本；坏文件或缺失标签直接保留库的异常，方便定位。"""
        image, valid = self.read_image(index)
        if self.patch_representation == "vv_rank_dem_m":
            image = vv_rank_dem_m(image, valid)
        elif self.patch_representation in RELATIVE_REPRESENTATIONS:
            image = apply_vv_train_p02_p98_dem_m(image, valid, self.relative_normalization)
        elif self.patch_representation == "relative_backscatter_ranks":
            image = relative_backscatter_ranks(image, valid)
        sample = {
            "image": torch.from_numpy(image),
            "valid_mask": torch.from_numpy(valid),
            "name": sample_name(self.image_paths[index], self.image_dir),
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
